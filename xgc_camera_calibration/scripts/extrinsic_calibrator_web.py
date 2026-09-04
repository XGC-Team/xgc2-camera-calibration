#!/usr/bin/env python3
"""ROS1 camera/pose adapter and HTTP entrypoint for extrinsic calibration."""

import re
import sys
import threading
from pathlib import Path

import cv2
import rospkg
import rospy
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import CompressedImage, Image

from xgc_camera_calibration.web_service import (
    ApiError,
    CalibrationHttpServer,
    CalibrationService,
    FrameSnapshot,
    MarkerObservation,
    image_message_to_bgr,
)
from xgc_camera_calibration.intrinsic_solver import load_intrinsic
from xgc_camera_calibration.intrinsic_validation import (
    ideal_intrinsic_parameters,
    intrinsic_parameters,
)
from xgc_camera_calibration.media_snapshot import MediaSnapshotClient, MediaSnapshotError
from xgc_camera_calibration.solver import optional_selected_intrinsic_path


def normalize_topic(value):
    normalized = str(value).strip()
    return normalized if normalized.startswith("/") else "/" + normalized


class RosCalibrationSource:
    """Thread-safe camera frame and latest static pose-marker snapshot."""

    def __init__(
        self,
        intrinsic_file,
        calibration_root,
        calibration_mode,
        camera_name,
        snapshot_client=None,
        snapshot_available=False,
    ):
        self.lock = threading.RLock()
        self.snapshot_client = snapshot_client
        self.snapshot_available = bool(snapshot_available)
        selected = optional_selected_intrinsic_path(
            calibration_root, calibration_mode, camera_name, intrinsic_file
        )
        self.use_ideal_intrinsics = selected is None
        self.ideal_horizontal_fov_degrees = float(
            rospy.get_param("~ideal_horizontal_fov_degrees", 110.0)
        )
        ideal_intrinsic_parameters(1, 1, self.ideal_horizontal_fov_degrees)
        if selected is None:
            self.intrinsic_file = ""
            self.intrinsic_matrix = None
            self.intrinsic_distortion = None
            self.intrinsic_size = None
        else:
            self.intrinsic_file = selected
            intrinsic_document = load_intrinsic(self.intrinsic_file)
            if str(intrinsic_document.get("camera_name", "")).strip() != camera_name:
                raise ValueError(
                    "selected intrinsic camera_name does not match ~camera_name"
                )
            (
                self.intrinsic_matrix,
                self.intrinsic_distortion,
                self.intrinsic_size,
            ) = intrinsic_parameters(intrinsic_document)
        self.image_topic = normalize_topic(rospy.get_param("~image_topic", "/usb_cam/image_raw"))
        self.preview_image_topic = normalize_topic(
            rospy.get_param(
                "~preview_image_topic", "/usb_cam/image_raw/compressed"
            )
        )
        self.freeze_image_timeout = float(
            rospy.get_param("~freeze_image_timeout", 2.0)
        )
        if self.freeze_image_timeout <= 0.0:
            raise ValueError("~freeze_image_timeout must be positive")
        self.pose_prefix = normalize_topic(
            rospy.get_param("~pose_prefix", "/vrpn_client_node")
        ).rstrip("/")
        tracker_value = rospy.get_param("~trackers", [])
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
        self.marker_subscribers = {}
        self.marker_topics = {}
        self.preview_subscriber = None
        if self.snapshot_client is None:
            self.preview_subscriber = rospy.Subscriber(
                self.preview_image_topic,
                CompressedImage,
                self._preview_callback,
                queue_size=1,
                buff_size=2**20,
            )
        self.discovery_timer = rospy.Timer(rospy.Duration(1.0), self._refresh_markers)
        self.snapshot_health_timer = None
        if self.snapshot_client is not None:
            self.snapshot_health_timer = rospy.Timer(
                rospy.Duration(1.0), self._refresh_snapshot_health
            )
        self._refresh_markers(None)

    def _refresh_snapshot_health(self, _event):
        try:
            self.snapshot_client.health()
            available = True
        except MediaSnapshotError as error:
            available = False
            rospy.logwarn_throttle(
                5.0, "Calibration Media Edge source is unavailable: %s", error
            )
        with self.lock:
            self.snapshot_available = available

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
            self.preview_jpeg = payload
            self.preview_stamp_sec = stamp_sec

    def _refresh_markers(self, _event):
        pattern = re.compile(r"^" + re.escape(self.pose_prefix) + r"/([^/]+)/pose$")
        desired = {}
        try:
            topics = rospy.get_published_topics()
        except rospy.ROSException as error:
            rospy.logwarn_throttle(5.0, "Could not discover pose markers: %s", error)
            return
        for topic, message_type in topics:
            match = pattern.match(topic)
            if not match or message_type != "geometry_msgs/PoseStamped":
                continue
            name = match.group(1)
            if self.tracker_filter and name not in self.tracker_filter:
                continue
            desired[topic] = name
        with self.lock:
            for topic in set(self.marker_subscribers) - set(desired):
                self.marker_subscribers.pop(topic).unregister()
                removed_name = self.marker_topics.pop(topic, "")
                if removed_name and removed_name not in desired.values():
                    self.marker_latest.pop(removed_name, None)
            for topic, name in desired.items():
                if topic in self.marker_subscribers:
                    continue
                self.marker_topics[topic] = name
                self.marker_subscribers[topic] = rospy.Subscriber(
                    topic, PoseStamped, self._marker_callback(name), queue_size=20
                )

    def _marker_callback(self, name):
        def callback(message):
            position = message.pose.position
            observation = MarkerObservation(
                name=name,
                position=(float(position.x), float(position.y), float(position.z)),
                frame_id=message.header.frame_id,
            )
            with self.lock:
                self.marker_latest[name] = observation

        return callback

    def _convert_image(self, message):
        try:
            return image_message_to_bgr(message)
        except (TypeError, ValueError, cv2.error) as error:
            raise ApiError(409, "Could not convert camera image: {}".format(error)) from error

    def preview_jpeg_bytes(self):
        with self.lock:
            return self.preview_jpeg

    def status(self):
        with self.lock:
            snapshot_ready = self.snapshot_available
            preview_ready = snapshot_ready or self.preview_jpeg is not None
            marker_names = sorted(self.marker_latest)
            return {
                "image_topic": (
                    "media:{}".format(self.snapshot_client.source_id)
                    if snapshot_ready
                    else self.image_topic
                ),
                "preview_image_topic": self.preview_image_topic,
                "intrinsic_file": str(self.intrinsic_file),
                "intrinsic_source": (
                    "ideal-pinhole" if self.use_ideal_intrinsics else "selected-file"
                ),
                "pose_prefix": self.pose_prefix,
                "image_ready": preview_ready,
                "preview_ready": preview_ready,
                "intrinsic_ready": True,
                "marker_count": len(marker_names),
                "marker_names": marker_names,
                "latest_image_stamp_sec": self.preview_stamp_sec,
            }

    def freeze(self, parent_frame):
        if self.snapshot_client is not None:
            try:
                snapshot = self.snapshot_client.capture()
            except MediaSnapshotError as error:
                with self.lock:
                    self.snapshot_available = False
                raise ApiError(
                    503, "Could not capture a Media Edge camera snapshot: {}".format(error)
                ) from error
            with self.lock:
                self.snapshot_available = True
            stamp_sec = float(snapshot.timestamp_nanoseconds) / 1.0e9
            return self._frame_snapshot(
                snapshot.bgr,
                stamp_sec,
                snapshot.frame_id,
                parent_frame,
            )
        try:
            image_message = rospy.wait_for_message(
                self.image_topic,
                Image,
                timeout=self.freeze_image_timeout,
            )
        except rospy.ROSException as error:
            raise ApiError(
                503,
                "No raw camera image arrived within {:.3f}s".format(
                    self.freeze_image_timeout
                ),
            ) from error
        image_stamp = image_message.header.stamp
        if image_stamp.is_zero():
            image_stamp = rospy.Time.now()
        stamp_sec = float(image_stamp.to_sec())
        image = self._convert_image(image_message)
        return self._frame_snapshot(
            image,
            stamp_sec,
            image_message.header.frame_id,
            parent_frame,
        )

    def _frame_snapshot(
        self,
        image,
        stamp_sec,
        frame_id,
        parent_frame,
    ):
        image_size = (int(image.shape[1]), int(image.shape[0]))
        if self.use_ideal_intrinsics:
            (
                intrinsic_matrix,
                intrinsic_distortion,
                intrinsic_size,
            ) = ideal_intrinsic_parameters(
                image_size[0],
                image_size[1],
                self.ideal_horizontal_fov_degrees,
            )
        else:
            intrinsic_matrix = self.intrinsic_matrix.copy()
            intrinsic_distortion = self.intrinsic_distortion.copy()
            intrinsic_size = self.intrinsic_size
        if not self.use_ideal_intrinsics and image_size != intrinsic_size:
            raise ApiError(
                409,
                "Camera intrinsics are {}x{}, but the captured image is {}x{}".format(
                    intrinsic_size[0],
                    intrinsic_size[1],
                    image_size[0],
                    image_size[1],
                ),
            )
        with self.lock:
            observations = dict(self.marker_latest)
        markers = {}
        wrong_frames = []
        for name, observation in observations.items():
            if observation.frame_id and observation.frame_id != parent_frame:
                wrong_frames.append(name)
                continue
            markers[name] = MarkerObservation(
                name=observation.name,
                position=observation.position,
                frame_id=observation.frame_id,
            )
        if wrong_frames:
            raise ApiError(
                409,
                "Pose markers are not expressed in parent frame '{}': {}".format(
                    parent_frame, ", ".join(sorted(wrong_frames))
                ),
            )
        return FrameSnapshot(
            image=image,
            stamp_sec=stamp_sec,
            frame_id=frame_id,
            camera_matrix=intrinsic_matrix,
            distortion=intrinsic_distortion,
            markers=markers,
        )


def split_list_parameter(value):
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    return [item.strip() for item in str(value).split(",") if item.strip()]


def main():
    rospy.init_node("xgc_camera_extrinsic_calibrator_web")
    try:
        media_edge_address = str(rospy.get_param("~media_edge_address", "")).strip()
        snapshot_client = None
        if media_edge_address:
            snapshot_client = MediaSnapshotClient(
                media_edge_address,
                rospy.get_param("~media_source_id", "usb_cam"),
                float(rospy.get_param("~snapshot_timeout", 5.0)),
            )
            try:
                snapshot_client.health()
                snapshot_available = True
            except MediaSnapshotError as error:
                snapshot_available = False
                rospy.logwarn(
                    "Camera extrinsic WebUI started with its Media Edge source unavailable; "
                    "the source will be rechecked without failing the Experiment: %s",
                    error,
                )
        else:
            snapshot_available = False
        calibration_root = str(rospy.get_param("~calibration_root")).strip()
        calibration_mode = str(rospy.get_param("~calibration_mode")).strip()
        camera_name = str(rospy.get_param("~camera_name")).strip()
        source = RosCalibrationSource(
            rospy.get_param("~intrinsic_file", ""),
            calibration_root,
            calibration_mode,
            camera_name,
            snapshot_client=snapshot_client,
            snapshot_available=snapshot_available,
        )
        package_root = Path(rospkg.RosPack().get_path("xgc_camera_calibration"))
        web_root = Path(rospy.get_param("~web_root", str(package_root / "web" / "extrinsic")))
        service = CalibrationService(
            source,
            calibration_root=calibration_root,
            calibration_mode=calibration_mode,
            camera_name=camera_name,
            parent_frame=rospy.get_param("~parent_frame", "map"),
            child_frame=rospy.get_param("~child_frame", "usb_cam_optical_frame"),
            ransac_threshold_px=float(rospy.get_param("~ransac_threshold_px", 3.0)),
            maximum_inlier_error_px=float(
                rospy.get_param("~maximum_inlier_error_px", 5.0)
            ),
            jpeg_quality=int(rospy.get_param("~jpeg_quality", 80)),
        )
        bind_address = str(rospy.get_param("~bind_address", "127.0.0.1"))
        http_port = int(rospy.get_param("~http_port", 8765))
        if not 1 <= http_port <= 65535:
            raise ValueError("~http_port must be between 1 and 65535")
        server = CalibrationHttpServer(
            (bind_address, http_port),
            service,
            web_root,
            frame_ancestors=str(
                rospy.get_param(
                    "~frame_ancestors",
                    "'self' http://127.0.0.1:* http://localhost:*",
                )
            ),
            allowed_origins=split_list_parameter(
                rospy.get_param("~allowed_origins", [])
            ),
            logger=lambda message: rospy.logdebug("Web calibrator: %s", message),
        )
    except Exception as error:
        rospy.logfatal("Could not start camera extrinsic WebUI: %s", error)
        return 1

    server_thread = threading.Thread(
        target=server.serve_forever,
        name="camera-calibration-http",
        daemon=True,
    )
    server_thread.start()
    rospy.loginfo(
        "Camera extrinsic WebUI listening on http://%s:%d "
        "(capture=%s, preview=%s, poses=%s)",
        bind_address,
        http_port,
        "media:{}".format(snapshot_client.source_id)
        if snapshot_client is not None
        else source.image_topic,
        "WebRTC" if snapshot_client is not None else source.preview_image_topic,
        source.pose_prefix,
    )
    if source.intrinsic_file:
        rospy.loginfo(
            "Camera extrinsic calibration uses intrinsics from %s",
            source.intrinsic_file,
        )
    else:
        rospy.loginfo(
            "Camera extrinsic calibration uses a %.3f-degree ideal pinhole at "
            "the captured image geometry until a timestamped intrinsic YAML is selected",
            source.ideal_horizontal_fov_degrees,
        )
    rospy.loginfo(
        "Camera extrinsic results will be saved under %s (%s/%s)",
        calibration_root,
        calibration_mode,
        camera_name,
    )
    try:
        rospy.spin()
    finally:
        server.shutdown()
        server.server_close()
        server_thread.join(timeout=5.0)
    return 0


if __name__ == "__main__":
    sys.exit(main())
