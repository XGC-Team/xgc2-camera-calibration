#!/usr/bin/env python3
"""ROS1 camera/pose adapter and HTTP entrypoint for extrinsic calibration."""

import sys
import threading
import time
import math
import uuid
from dataclasses import replace
from pathlib import Path

import cv2
import numpy as np
import rospkg
import rospy
from xgc2_xrpc import Runtime
from geometry_msgs.msg import PoseStamped
from rosgraph_msgs.msg import Clock
from sensor_msgs.msg import CameraInfo, CompressedImage

from xgc_camera_calibration.pose_freshness import pose_is_fresh
from xgc_camera_calibration.ros_lifecycle import NodeParameters, RosMasterBinding
from xgc_camera_calibration.extrinsic_application import resolve_application, application_arguments
from xgc_camera_calibration.extrinsic_samples import image_dimensions

from xgc_camera_calibration.web_service import (
    ApiError,
    CalibrationHttpServer,
    CalibrationService,
    FrameSnapshot,
    MarkerObservation,
)
from xgc_camera_calibration.intrinsic_solver import load_intrinsic
from xgc_camera_calibration.intrinsic_validation import (
    intrinsic_parameters,
)
from xgc_camera_calibration.media_snapshot import MediaSnapshotClient, MediaSnapshotError
from xgc_camera_calibration.extrinsic_coordinates import coordinate_provenance


def normalize_topic(value):
    normalized = str(value).strip()
    return normalized if normalized.startswith("/") else "/" + normalized


class RosCalibrationSource:
    """Thread-safe camera frame and latest static pose-marker snapshot."""

    def __init__(
        self,
        intrinsic_file,
        calibration_mode,
        camera_name,
        snapshot_client=None,
        frozen_extrinsic=None,
        parameters=None,
    ):
        self.parameters = parameters
        self.pose_coordinate_source = str(parameters.get("~pose_coordinate_source", "unspecified"))
        if self.pose_coordinate_source not in ("unspecified", "raw-vrpn", "experiment-world"):
            raise ValueError("invalid pose coordinate source")
        if frozen_extrinsic is None:
            raise ValueError("calibration source requires its frozen camera context")
        self.pose_world_offset = tuple(frozen_extrinsic["targetCoordinates"]["worldOffset"])
        self.lock = threading.RLock()
        self.camera_info = None
        self.optical_frame = str(parameters.get("~child_frame"))
        self._readiness_stopping = False
        self.camera_info_topic = normalize_topic(parameters.get("~camera_info_topic", "/usb_cam/camera_info"))
        self.info_subscriber = None
        self.snapshot_client = snapshot_client
        self.source_size = None
        selected = Path(str(intrinsic_file).strip()) if str(intrinsic_file).strip() else None
        if selected is not None and (not selected.is_absolute() or not selected.is_file()):
            raise ValueError("intrinsic_file must be an explicit absolute input file")
        self.use_camera_info = selected is None
        if selected is None:
            self.intrinsic_file = ""
            self.intrinsic_matrix = None
            self.intrinsic_distortion = None
            self.intrinsic_size = None
            self.intrinsic_provenance = {"intrinsic_source": "camera-info", "distortion_model": "plumb_bob"}
        else:
            self.intrinsic_file = selected
            intrinsic_document = load_intrinsic(self.intrinsic_file)
            if intrinsic_document.get("calibration_mode", calibration_mode) != calibration_mode:
                raise ValueError("selected intrinsic mode does not match ~calibration_mode")
            self.intrinsic_provenance = {"intrinsic_source": "selected-file",
                "intrinsic_file": str(self.intrinsic_file),
                                         "intrinsic_sha256": intrinsic_document["source_sha256"],
                                         "distortion_model": intrinsic_document.get("distortion_model", "plumb_bob")}
            if str(intrinsic_document.get("camera_name", "")).strip() != camera_name:
                raise ValueError(
                    "selected intrinsic camera_name does not match ~camera_name"
                )
            (
                self.intrinsic_matrix,
                self.intrinsic_distortion,
                self.intrinsic_size,
            ) = intrinsic_parameters(intrinsic_document)
        self.image_topic = normalize_topic(parameters.get("~image_topic", "/usb_cam/image_raw/compressed"))
        self.preview_image_topic = normalize_topic(
            parameters.get(
                "~preview_image_topic", "/usb_cam/image_raw/compressed"
            )
        )
        self.freeze_image_timeout = float(
            parameters.get("~freeze_image_timeout", 2.0)
        )
        if self.freeze_image_timeout <= 0.0:
            raise ValueError("~freeze_image_timeout must be positive")
        self.pose_prefix = normalize_topic(
            parameters.get("~pose_prefix", "/vrpn_client_node")
        ).rstrip("/")
        tracker_value = parameters.get("~trackers", [])
        if isinstance(tracker_value, list):
            self.tracker_filter = {
                str(item).strip() for item in tracker_value if str(item).strip()
            }
        else:
            self.tracker_filter = {
                item.strip() for item in str(tracker_value).split(",") if item.strip()
            }
        self.preview_jpeg = None
        self.preview_stamp_sec = None
        self.marker_latest = {}
        self.marker_receipts = {}
        self.pose_max_age = float(parameters.get("~pose_max_age", 2.0))
        if not math.isfinite(self.pose_max_age) or self.pose_max_age <= 0:
            raise ValueError("pose_max_age must be finite and positive")
        self.marker_subscribers = {}
        self.marker_topics = {}
        self.preview_subscriber = None
        self.image_subscriber = None
        self._latest_image = None
        self._preview_received = None
        self._state_listeners = set()
        self._state_condition = threading.Condition(self.lock)
        self._state_thread = None
        self._state_stopping = False
        self._state_freshness = None
        self._state_deadline = None
        self._clock_bounds = (float("-inf"), float("inf"))
        self._state_clock_subscriber = None
        self._state_clock_retiring = False
        self._ros_started = False
        self._ros_users = 0
        self._ros_retiring = False
        self._ros_generation = 0
        self._ros_callbacks = 0

    def _guard_ros_callback(self, generation, callback):
        def guarded(message):
            with self._state_condition:
                if generation != self._ros_generation or self._ros_retiring:
                    return
                self._ros_callbacks += 1
            try:
                callback(message)
            finally:
                with self._state_condition:
                    self._ros_callbacks -= 1
                    self._state_condition.notify_all()
        return guarded

    def _start_ros_locked(self):
        self._ros_generation += 1
        generation = self._ros_generation
        guard = lambda callback: self._guard_ros_callback(generation, callback)
        try:
            self.info_subscriber = rospy.Subscriber(self.camera_info_topic, CameraInfo,
                guard(self._camera_info_callback), queue_size=1)
            self.preview_subscriber = rospy.Subscriber(self.preview_image_topic, CompressedImage,
                guard(self._preview_callback), queue_size=1, buff_size=2**20)
            if self.snapshot_client is None and self.image_topic != self.preview_image_topic:
                self.image_subscriber = rospy.Subscriber(self.image_topic, CompressedImage,
                    guard(self._image_callback), queue_size=1, buff_size=2**20)
            for name in sorted(self.tracker_filter):
                topic = self.pose_prefix + "/" + name + "/pose"
                self.marker_subscribers[topic] = rospy.Subscriber(topic, PoseStamped,
                    guard(self._marker_callback(name)), queue_size=1)
        except BaseException:
            self._ros_generation += 1
            subscribers = [self.info_subscriber, self.preview_subscriber, self.image_subscriber]
            subscribers += list(self.marker_subscribers.values())
            self.info_subscriber = self.preview_subscriber = self.image_subscriber = None
            self.marker_subscribers = {}
            for subscriber in subscribers:
                if subscriber is not None:
                    subscriber.unregister()
            raise

    def retain_source(self):
        with self._state_condition:
            while self._ros_retiring:
                self._state_condition.wait()
            if self._readiness_stopping:
                raise RuntimeError("calibration source is stopping")
            self._ros_users += 1
            try:
                if self._ros_started and self._ros_users == 1:
                    self._start_ros_locked()
            except BaseException:
                self._ros_users -= 1
                raise
        released = False
        def release():
            nonlocal released
            with self._state_condition:
                if released:
                    return
                released = True
                self._ros_users -= 1
                if self._ros_users:
                    return
                self._ros_retiring = True
                self._ros_generation += 1
                subscribers = [self.info_subscriber, self.preview_subscriber, self.image_subscriber]
                subscribers += list(self.marker_subscribers.values())
                self.info_subscriber = self.preview_subscriber = self.image_subscriber = None
                self.marker_subscribers = {}
            try:
                for subscriber in subscribers:
                    if subscriber is not None:
                        subscriber.unregister()
                with self._state_condition:
                    self._state_condition.wait_for(lambda: self._ros_callbacks == 0)
                    self.camera_info = None
                    self.preview_jpeg = self.preview_stamp_sec = self._preview_received = None
                    self._latest_image = None
                    self.source_size = None
                    self.marker_latest.clear()
                    self.marker_receipts.clear()
            finally:
                with self._state_condition:
                    self._ros_retiring = False
                    self._state_condition.notify_all()
        return release

    def subscribe_state(self, callback):
        release_source = self.retain_source()
        try:
            with self._state_condition:
                while self._state_stopping and self._state_thread is not None:
                    self._state_condition.wait()
                self._state_listeners.add(callback)
                if self._state_thread is None:
                    try:
                        self._state_stopping = False
                        self._state_freshness = self._freshness_locked()
                        self._state_deadline = None
                        self._state_thread = threading.Thread(target=self._state_expiry_loop,
                                                              name="calibration-state-expiry")
                        self._state_thread.start()
                    except BaseException:
                        self._state_thread = None
                        self._state_listeners.discard(callback)
                        raise
        except BaseException:
            release_source()
            raise
        release_media = None
        released = False
        def release():
            nonlocal released
            with self._state_condition:
                if released:
                    return
                released = True
                self._state_listeners.discard(callback)
                if self._state_listeners:
                    thread = clock = None
                else:
                    self._state_stopping = True
                    self._state_condition.notify_all()
                    thread, clock = self._state_thread, self._state_clock_subscriber
                    self._state_clock_subscriber = None
            try:
                if release_media is not None:
                    release_media()
            finally:
                try:
                    if clock is not None:
                        clock.unregister()
                finally:
                    if thread is not None and thread is not threading.current_thread():
                        thread.join()
                    with self._state_condition:
                        if thread is not None and self._state_thread is thread:
                            self._state_thread = None
                            self._state_condition.notify_all()
                    release_source()
        try:
            release_media = self.snapshot_client.subscribe_state(callback) if self.snapshot_client else None
            self._retain_state_clock()
        except BaseException:
            release()
            raise
        return release

    def _retain_state_clock(self):
        with self._state_condition:
            while self._state_clock_retiring and self._state_listeners:
                self._state_condition.wait()
            now = time.monotonic()
            if (not self._state_listeners or self._state_clock_subscriber is not None
                    or not any(now - received <= self.pose_max_age and math.isfinite(stamp) and stamp > 0
                               for received, stamp in self.marker_receipts.values())
                    or not rospy.rostime.is_rostime_initialized() or rospy.rostime.is_wallclock()):
                return
            self._state_clock_subscriber = rospy.Subscriber("/clock", Clock,
                self._guard_ros_callback(self._ros_generation, self._state_clock), queue_size=1)

    def _state_changed_locked(self):
        for callback in tuple(self._state_listeners):
            callback()

    def _wake_for_deadline_locked(self, deadline):
        # Extending a deadline never needs to wake the expiry owner. It will
        # recheck its old deadline and re-arm without projecting stale state.
        if (self._state_listeners and
                (self._state_deadline is None or deadline < self._state_deadline - 0.000001)):
            self._state_condition.notify_all()

    def _freshness_locked(self, ros_now=None):
        wall_now = time.monotonic()
        if ros_now is None:
            ros_now = rospy.Time.now().to_sec() if rospy.rostime.is_rostime_initialized() else 0.0
        preview = self._preview_received is not None and wall_now - self._preview_received <= self.freeze_image_timeout
        names = tuple(sorted(name for name in self.marker_latest
            if pose_is_fresh(self.marker_receipts.get(name), wall_now, ros_now, self.pose_max_age)))
        boundaries = [boundary for received, stamped in self.marker_receipts.values()
                      if wall_now - received <= self.pose_max_age and stamped > 0
                      for boundary in (stamped, float(np.nextafter(stamped + self.pose_max_age, np.inf)))]
        self._clock_bounds = (max((value for value in boundaries if value <= ros_now), default=float("-inf")),
                              min((value for value in boundaries if value > ros_now), default=float("inf")))
        return preview, names

    def _refresh_freshness_locked(self, ros_now=None):
        if not self._state_listeners:
            return
        current = self._freshness_locked(ros_now)
        if current != self._state_freshness:
            self._state_freshness = current
            self._state_changed_locked()

    def _state_clock(self, message):
        value = message.clock.to_sec()
        with self.lock:
            lower, upper = self._clock_bounds
            if value < lower or value >= upper:
                self._refresh_freshness_locked(value)
                self._state_condition.notify_all()

    def _state_expiry_loop(self):
        while True:
            retired_clock = None
            with self._state_condition:
                if self._state_stopping:
                    return
                now = time.monotonic()
                if (self._state_clock_subscriber is not None
                        and not any(now - received <= self.pose_max_age and math.isfinite(stamp) and stamp > 0
                                    for received, stamp in self.marker_receipts.values())):
                    retired_clock = self._state_clock_subscriber
                    self._state_clock_subscriber = None
                    self._state_clock_retiring = True
                preview, names = self._state_freshness
                names = set(names)
                deadlines = []
                if preview and self._preview_received is not None:
                    deadlines.append(self._preview_received + self.freeze_image_timeout + 0.001)
                wall_clock = rospy.rostime.is_rostime_initialized() and rospy.rostime.is_wallclock()
                ros_now = rospy.Time.now().to_sec() if wall_clock else 0.0
                for name, (received, stamped) in self.marker_receipts.items():
                    wall_end = received + self.pose_max_age + 0.001
                    if name in names or (self._state_clock_subscriber is not None and wall_end > now):
                        deadlines.append(wall_end)
                    if wall_clock and math.isfinite(stamped) and stamped > 0 and now - received <= self.pose_max_age:
                        if name in names:
                            deadlines.append(now + stamped + self.pose_max_age - ros_now + 0.001)
                        elif ros_now < stamped:
                            deadlines.append(now + stamped - ros_now + 0.001)
                        elif ros_now <= stamped + self.pose_max_age:
                            deadlines.append(now)
                if retired_clock is None:
                    deadline = min(deadlines) if deadlines else None
                    if deadline is not None and deadline <= now:
                        # This is a current fact crossing its own deadline,
                        # not a periodic state snapshot or an obsolete timer.
                        self._refresh_freshness_locked()
                        continue
                    self._state_deadline = deadline
                    self._state_condition.wait(None if deadline is None else deadline - now)
                else:
                    self._state_deadline = None
            if retired_clock is not None:
                try:
                    retired_clock.unregister()
                finally:
                    with self._state_condition:
                        self._state_clock_retiring = False
                        self._state_condition.notify_all()

    def start_ros(self):
        with self._state_condition:
            self._ros_started = True
            if self._ros_users:
                self._start_ros_locked()
        self._retain_state_clock()

    def close(self):
        self.stop_readiness()

    def _image_callback(self, message):
        with self.lock:
            self._latest_image = (message, time.monotonic())

    def _preview_callback(self, message):
        image_format = str(message.format).strip().lower()
        payload = bytes(message.data)
        if (
            not payload.startswith(b"\xff\xd8")
            or ("jpeg" not in image_format and "jpg" not in image_format)
        ):
            rospy.logwarn_throttle(
                5.0,
                "Ignoring non-JPEG compressed preview on %s (format=%r)",
                self.preview_image_topic,
                message.format,
            )
            return
        stamp = message.header.stamp
        stamp_sec = float(
            (stamp if not stamp.is_zero() else rospy.Time.now()).to_sec()
        )
        with self.lock:
            previous_stamp = self.preview_stamp_sec
            previous_received = self._preview_received
            self.preview_jpeg = payload
            self._preview_received = time.monotonic()
            if self.image_topic == self.preview_image_topic:
                self._latest_image = (message, self._preview_received)
            try:
                self.source_size = image_dimensions(payload, "image/jpeg")
            except ApiError:
                self.source_size = None
            self.preview_stamp_sec = stamp_sec
            if previous_stamp != stamp_sec:
                self._state_changed_locked()
            if (previous_received is None
                    or self._preview_received - previous_received > self.freeze_image_timeout):
                self._refresh_freshness_locked()
            self._wake_for_deadline_locked(self._preview_received + self.freeze_image_timeout + 0.001)

    def _marker_callback(self, name):
        def callback(message):
            position = message.pose.position
            observation = MarkerObservation(
                name=name,
                position=(float(position.x), float(position.y), float(position.z)),
                frame_id=message.header.frame_id,
                observation_id=uuid.uuid4().hex,
                source_stamp_sec=float(message.header.stamp.to_sec()),
                received_at_sec=time.time(),
                received_monotonic_sec=time.monotonic(),
            )
            with self.lock:
                previous = self.marker_receipts.get(name)
                self.marker_latest[name] = observation
                self.marker_receipts[name] = (observation.received_monotonic_sec, observation.source_stamp_sec)
                if (previous is None or previous[1] != observation.source_stamp_sec
                        or observation.received_monotonic_sec - previous[0] > self.pose_max_age):
                    self._refresh_freshness_locked()
                deadline = observation.received_monotonic_sec + self.pose_max_age + 0.001
                if (rospy.rostime.is_rostime_initialized() and rospy.rostime.is_wallclock()
                        and math.isfinite(observation.source_stamp_sec) and observation.source_stamp_sec > 0):
                    ros_now = rospy.Time.now().to_sec()
                    if observation.source_stamp_sec > ros_now:
                        deadline = min(deadline, observation.received_monotonic_sec
                                       + observation.source_stamp_sec - ros_now + 0.001)
                    elif ros_now <= observation.source_stamp_sec + self.pose_max_age:
                        deadline = min(deadline, observation.received_monotonic_sec
                                       + observation.source_stamp_sec + self.pose_max_age - ros_now + 0.001)
                self._wake_for_deadline_locked(deadline)
            self._retain_state_clock()

        return callback

    @property
    def source_id(self):
        return self.snapshot_client.source_id if self.snapshot_client is not None else "ros:" + self.image_topic

    def preview_jpeg_bytes(self):
        with self.lock:
            return (self.preview_jpeg if self._preview_received is not None
                    and time.monotonic() - self._preview_received <= self.freeze_image_timeout else None)

    def status(self):
        with self.lock:
            media = self.snapshot_client.status() if self.snapshot_client is not None else {"ready": False, "reason": "not_configured"}
            snapshot_ready = media["ready"]
            if media.get("width") and media.get("height"):
                self.source_size = (media["width"], media["height"])
            preview_ready = self.preview_jpeg_bytes() is not None
            wall_now = time.monotonic()
            # init_node marks the node initialized before it initializes time.
            ros_now = rospy.Time.now().to_sec() if rospy.rostime.is_rostime_initialized() else 0.0
            marker_names = sorted(name for name in self.marker_latest
                if pose_is_fresh(self.marker_receipts.get(name), wall_now, ros_now, self.pose_max_age))
            return {
                "image_topic": (
                    "media:{}".format(self.snapshot_client.source_id)
                    if snapshot_ready
                    else self.image_topic
                ),
                "preview_image_topic": self.preview_image_topic,
                "intrinsic_file": str(self.intrinsic_file),
                "intrinsic_source": (
                    "camera-info" if self.use_camera_info else "selected-file"
                ),
                "pose_prefix": self.pose_prefix,
                "image_ready": preview_ready,
                "capture_ready": snapshot_ready or preview_ready,
                "preview_ready": preview_ready,
                "intrinsic_ready": not self.use_camera_info or self.camera_info is not None,
                "camera_info_topic": self.camera_info_topic,
                "camera_info_received": self.camera_info is not None,
                "world_pose_available": bool(marker_names),
                "media": media,
                "marker_count": len(marker_names),
                "marker_names": marker_names,
                "latest_image_stamp_sec": self.preview_stamp_sec,
            }

    def freeze(self, parent_frame):
        release_media = self.snapshot_client.peer.retain() if self.snapshot_client is not None else None
        try:
            release = self.retain_source()
            try:
                return self._freeze(parent_frame)
            finally:
                release()
        finally:
            if release_media is not None:
                release_media()

    def _freeze(self, parent_frame):
        if self.snapshot_client is not None:
            try:
                snapshot = self.snapshot_client.capture()
            except MediaSnapshotError as error:
                raise ApiError(
                    503, "Could not capture a Media Edge camera snapshot: {}".format(error)
                ) from error
            with self.lock:
                self.source_size = (snapshot.width, snapshot.height)
            stamp_sec = float(snapshot.timestamp_nanoseconds) / 1.0e9
            return self._frame_snapshot(
                snapshot.bgr,
                stamp_sec,
                snapshot.frame_id,
                parent_frame,
            )
        with self.lock:
            latest = self._latest_image
        if latest is None or time.monotonic() - latest[1] > self.freeze_image_timeout:
            raise ApiError(503, "No recent compressed camera image is available")
        image_message = latest[0]
        image = cv2.imdecode(np.frombuffer(bytes(image_message.data), dtype=np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            raise ApiError(409, "Could not decode compressed camera image")
        return self._frame_snapshot(image, float(image_message.header.stamp.to_sec()),
                                    image_message.header.frame_id, parent_frame)

    def _camera_info_callback(self, message):
        matrix = np.asarray(message.K, dtype=np.float64).reshape(3, 3)
        distortion = np.asarray(message.D, dtype=np.float64)
        if (message.width <= 0 or message.height <= 0 or not message.header.frame_id
                or message.header.frame_id != self.optical_frame
                or message.distortion_model != "plumb_bob" or distortion.size < 4
                or not np.all(np.isfinite(matrix)) or not np.all(np.isfinite(distortion))
                or matrix[0, 0] <= 0 or matrix[1, 1] <= 0):
            return
        with self.lock:
            first = self.camera_info is None
            self.camera_info = (matrix, distortion, (message.width, message.height), message.header.frame_id)
            if first:
                self._state_changed_locked()

    def wait_ready(self, timeout):
        with self.lock:
            return not self._readiness_stopping

    def stop_readiness(self):
        with self.lock:
            self._readiness_stopping = True

    def _frame_snapshot(
        self,
        image,
        stamp_sec,
        frame_id,
        parent_frame,
    ):
        context = self._observation_context(int(image.shape[1]), int(image.shape[0]), parent_frame)
        if frame_id != self.optical_frame:
            raise ApiError(409, "Image and CameraInfo optical frames differ")
        return FrameSnapshot(image=image, stamp_sec=stamp_sec, frame_id=frame_id,
                             camera_matrix=context["camera_matrix"], distortion=context["distortion"],
                             markers=context["markers"], camera_model=context["camera_model"],
                             pose_coordinates=context["pose_coordinates"])

    def observe_marker(self, marker, width, height, parent_frame):
        release = self.retain_source()
        try:
            return self._observe_marker(marker, width, height, parent_frame)
        finally:
            release()

    def _observe_marker(self, marker, width, height, parent_frame):
        # Metadata and latest pose only: the browser already owns the clicked
        # image. Never capture another frame to reinterpret its pixels.
        # The selected intrinsic model owns dimensions. A browser-owned frame
        # must not depend on a separate ROS preview or native capture having run.
        # _observation_context checks the model here; commit validates the image.
        context = self._observation_context(width, height, parent_frame, marker)
        observation = context.pop("markers").get(marker)
        if observation is None:
            raise ApiError(409, "Selected marker has no fresh pose observation")
        context["marker"] = observation
        return context

    def _observation_context(self, width, height, parent_frame, selected_marker=None):
        image_size = (width, height)
        if self.use_camera_info:
            with self.lock:
                info = self.camera_info
            if info is None:
                raise ApiError(409, "No calibrated CameraInfo has arrived")
            intrinsic_matrix, intrinsic_distortion, intrinsic_size, optical_frame = info
            intrinsic_matrix, intrinsic_distortion = intrinsic_matrix.copy(), intrinsic_distortion.copy()
            provenance_model = {"intrinsic_source": "camera-info", "distortion_model": "plumb_bob",
                                "camera_info_topic": self.camera_info_topic, "frame_id": optical_frame}
        else:
            intrinsic_matrix = self.intrinsic_matrix.copy()
            intrinsic_distortion = self.intrinsic_distortion.copy()
            intrinsic_size = self.intrinsic_size
            provenance_model = dict(self.intrinsic_provenance)
        if image_size != intrinsic_size:
            raise ApiError(409, "Camera intrinsics are {}x{}, but the captured image is {}x{}".format(
                intrinsic_size[0], intrinsic_size[1], width, height))
        with self.lock:
            wall_now, ros_now = time.monotonic(), rospy.Time.now().to_sec()
            observations = {name: observation for name, observation in self.marker_latest.items()
                if (selected_marker is None or name == selected_marker)
                and pose_is_fresh(self.marker_receipts.get(name), wall_now, ros_now, self.pose_max_age)}
        provenance = {}
        if self.pose_coordinate_source != "unspecified":
            provenance = coordinate_provenance("experiment-world", parent_frame, self.pose_world_offset)
            provenance["input_kind"] = self.pose_coordinate_source
            provenance["input_frame"] = parent_frame
        markers = {}
        wrong_frames = []
        for name, observation in observations.items():
            if observation.frame_id and observation.frame_id != parent_frame:
                wrong_frames.append(name)
                continue
            markers[name] = replace(
                observation,
                position=(tuple(float(observation.position[i]) + self.pose_world_offset[i] for i in range(3))
                          if self.pose_coordinate_source == "raw-vrpn" else observation.position),
                frame_id=observation.frame_id,
                source_position=observation.position,
            )
        if wrong_frames:
            raise ApiError(
                409,
                "Pose markers are not expressed in parent frame '{}': {}".format(
                    parent_frame, ", ".join(sorted(wrong_frames))
                ),
            )
        return {"camera_matrix": intrinsic_matrix, "distortion": intrinsic_distortion,
                "markers": markers, "camera_model": provenance_model,
                "pose_coordinates": provenance}


def split_list_parameter(value):
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    return [item.strip() for item in str(value).split(",") if item.strip()]


def main():
    args = application_arguments(rospy.myargv()[1:])
    parameters = NodeParameters("xgc_camera_extrinsic_calibrator_web")
    binding = RosMasterBinding("xgc_camera_extrinsic_calibrator_web")
    runtime = None
    snapshot_client = None
    source = None
    server = None
    try:
        target_id = str(parameters.get("~target_id"))
        socket_path = str(parameters.get("~socket_path"))
        runtime = Runtime(blocking_workers=4, max_calls=16, max_connections=24)
        media_edge_rpc_socket = str(parameters.get("~media_edge_rpc_socket", "")).strip()
        snapshot_client = None
        if media_edge_rpc_socket:
            snapshot_client = MediaSnapshotClient(
                media_edge_rpc_socket,
                parameters.get("~media_source_id", "usb_cam"),
                float(parameters.get("~snapshot_timeout", 5.0)),
                runtime=runtime,
                local_target=target_id,
            )
        calibration_mode = str(parameters.get("~calibration_mode")).strip()
        camera_name = str(parameters.get("~camera_name")).strip()
        source = RosCalibrationSource(
            parameters.get("~intrinsic_file", ""),
            calibration_mode,
            camera_name,
            snapshot_client=snapshot_client,
            parameters=parameters,
            frozen_extrinsic=resolve_application(args, camera_name),
        )
        package_root = Path(rospkg.RosPack().get_path("xgc_camera_calibration"))
        web_root = Path(parameters.get("~web_root", str(package_root / "web" / "extrinsic")))
        service = CalibrationService(
            source,
            calibration_mode=calibration_mode,
            camera_name=camera_name,
            parent_frame=parameters.get("~parent_frame"),
            child_frame=parameters.get("~child_frame"),
            ransac_threshold_px=float(parameters.get("~ransac_threshold_px", 3.0)),
            maximum_inlier_error_px=float(
                parameters.get("~maximum_inlier_error_px", 10.0)
            ),
            jpeg_quality=int(parameters.get("~jpeg_quality", 80)),
            resolved_extrinsic_json=args.resolved_extrinsic_json,
            frame_roles_json=args.frame_roles_json,
        )
        frozen_offset = service.frozen_extrinsic["targetCoordinates"]["worldOffset"]
        if source.pose_coordinate_source == "unspecified" or list(source.pose_world_offset) != frozen_offset:
            raise ValueError("calibration sample coordinates must match the frozen camera context")
        bind_address = str(parameters.get("~bind_address", "127.0.0.1"))
        http_port = int(parameters.get("~http_port", 8765))
        if not 1 <= http_port <= 65535:
            raise ValueError("~http_port must be between 1 and 65535")
        def peer_status_from_source(status):
            return {"ros_master": binding.status(), "media": status["media"],
                    "camera_info": {"received": status["camera_info_received"]},
                    "world": {"pose_available": status["world_pose_available"], "marker_count": status["marker_count"]}}
        def peer_status():
            return peer_status_from_source(source.status())
        server = CalibrationHttpServer(
            (bind_address, http_port),
            service,
            web_root,
            frame_ancestors=str(
                parameters.get(
                    "~frame_ancestors",
                    "'self' http://127.0.0.1:* http://localhost:*",
                )
            ),
            allowed_origins=split_list_parameter(
                parameters.get("~allowed_origins", [])
            ),
            logger=lambda message: rospy.logdebug("Web calibrator: %s", message),
            runtime=runtime,
            target_id=target_id,
            rpc_socket=socket_path,
            lifecycle=binding, peer_status=peer_status,
            peer_status_for_state=lambda state: peer_status_from_source(state["source"]),
        )
        server.start()
        if not binding.initialize():
            return 0
        source.start_ros()
        binding.mark_ready()
        rospy.loginfo("Camera extrinsic WebUI listening on http://%s:%d", bind_address, http_port)
        rospy.spin()
        return 0
    except Exception as error:
        rospy.logfatal("Could not start camera extrinsic WebUI: %s", error)
        return 1
    finally:
        binding.stop_readiness()
        if server is not None:
            server.close()
        if source is not None:
            source.close()
        if snapshot_client is not None:
            snapshot_client.close()
        binding.close()
        if runtime is not None:
            runtime.close()


if __name__ == "__main__":
    sys.exit(main())
