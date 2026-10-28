"""HTTP-independent camera extrinsic calibration service and web transport."""

from __future__ import annotations

import asyncio
import copy
import hashlib
import json
import math
import mimetypes
import os
import threading
import uuid
from dataclasses import asdict, dataclass, field, replace
from http import HTTPStatus
from aiohttp import web
from xgc2_xrpc import AppRouter, Endpoint, Fault, Host, Limits, RawStreamResponse, Response, Runtime, ServiceRef, iter_body
from xgc_camera_calibration.peer_connection import PeerUnavailable
from pathlib import Path
from typing import Any, Callable, Dict, Mapping, Optional, Sequence, Tuple
from urllib.parse import parse_qs

import cv2
import numpy as np

from xgc_camera_calibration.solver import (
    CalibrationError,
    ExtrinsicResult,
    result_document,
    solve_extrinsic,
)


from xgc_camera_calibration.extrinsic_samples import ApiError, SampleCollection, IMAGE_BYTES, IMAGE_PATH
from xgc_camera_calibration.extrinsic_application import parse_frame_roles
from xgc_camera_calibration.extrinsic_resolver import decode_frozen


@dataclass(frozen=True)
class MarkerObservation:
    name: str
    position: Tuple[float, float, float]
    frame_id: str
    source_position: Optional[Tuple[float, float, float]] = None
    observation_id: str = ""
    source_stamp_sec: Optional[float] = None
    received_at_sec: Optional[float] = None
    received_monotonic_sec: Optional[float] = None


@dataclass(frozen=True)
class FrameSnapshot:
    image: np.ndarray
    stamp_sec: float
    frame_id: str
    camera_matrix: np.ndarray
    distortion: np.ndarray
    markers: Mapping[str, MarkerObservation]
    camera_model: Mapping[str, Any] = field(default_factory=dict)
    pose_coordinates: Mapping[str, Any] = field(default_factory=dict)

    @property
    def width(self) -> int:
        return int(self.image.shape[1])

    @property
    def height(self) -> int:
        return int(self.image.shape[0])


def image_message_to_bgr(message: Any) -> np.ndarray:
    """Convert common 8-bit sensor_msgs/Image encodings without cv_bridge."""
    height = int(message.height)
    width = int(message.width)
    if height <= 0 or width <= 0:
        raise ValueError("Image dimensions must be positive")

    encoding = str(message.encoding).strip().lower()
    formats = {
        "bgr8": (3, None),
        "8uc3": (3, None),
        "rgb8": (3, cv2.COLOR_RGB2BGR),
        "bgra8": (4, cv2.COLOR_BGRA2BGR),
        "8uc4": (4, cv2.COLOR_BGRA2BGR),
        "rgba8": (4, cv2.COLOR_RGBA2BGR),
        "mono8": (1, cv2.COLOR_GRAY2BGR),
        "8uc1": (1, cv2.COLOR_GRAY2BGR),
    }
    if encoding not in formats:
        raise ValueError(
            "Unsupported image encoding '{}'; expected an 8-bit color or mono image".format(
                message.encoding
            )
        )
    channels, conversion = formats[encoding]
    row_bytes = width * channels
    step = int(message.step)
    if step < row_bytes:
        raise ValueError("Image step is smaller than the encoded row width")

    try:
        raw = np.frombuffer(message.data, dtype=np.uint8)
    except TypeError:
        raw = np.asarray(message.data, dtype=np.uint8)
    required = step * height
    if raw.size < required:
        raise ValueError("Image data is shorter than height * step")
    rows = raw[:required].reshape(height, step)
    image = rows[:, :row_bytes].reshape(height, width, channels).copy()
    if channels == 1:
        image = image.reshape(height, width)
    if conversion is not None:
        image = cv2.cvtColor(image, conversion)
    return image


def _finite_pixel(value: Any, name: str) -> Tuple[float, float]:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ApiError(HTTPStatus.BAD_REQUEST, "{} must be a two-element array".format(name))
    try:
        pixel = (float(value[0]), float(value[1]))
    except (TypeError, ValueError) as error:
        raise ApiError(
            HTTPStatus.BAD_REQUEST, "{} must contain numeric coordinates".format(name)
        ) from error
    if not all(math.isfinite(item) for item in pixel):
        raise ApiError(HTTPStatus.BAD_REQUEST, "{} must contain finite coordinates".format(name))
    return pixel


def _result_payload(
    result: ExtrinsicResult,
    marker_names: Sequence[str],
    world_points: np.ndarray,
    camera_matrix: np.ndarray,
    distortion: np.ndarray,
) -> Dict[str, Any]:
    rotation_vector, _ = cv2.Rodrigues(result.rotation_world_to_camera)
    projected, _ = cv2.projectPoints(
        world_points.reshape(-1, 1, 3),
        rotation_vector,
        result.translation_world_to_camera,
        camera_matrix,
        distortion,
    )
    camera_points = (
        result.rotation_world_to_camera.dot(world_points.T).T
        + result.translation_world_to_camera
    )
    projections = []
    for name, pixel, camera_point in zip(
        marker_names, projected.reshape(-1, 2), camera_points
    ):
        if float(camera_point[2]) > 0.0:
            projections.append(
                {"marker": name, "pixel": [float(pixel[0]), float(pixel[1])]}
            )
    return {
        "translation": [float(item) for item in result.translation],
        "quaternion_xyzw": [float(item) for item in result.quaternion_xyzw],
        "mean_reprojection_error_px": result.mean_reprojection_error_px,
        "max_reprojection_error_px": result.max_reprojection_error_px,
        "inlier_indices": [int(item) for item in result.inlier_indices],
        "warnings": list(result.warnings),
        "projections": projections,
    }


class CalibrationService:
    """Own one operator calibration session over a ROS-backed frame source."""

    def __init__(
        self,
        source: Any,
        *,
        calibration_mode: str,
        camera_name: str,
        parent_frame: str,
        child_frame: str,
        ransac_threshold_px: float = 3.0,
        maximum_inlier_error_px: float = 10.0,
        jpeg_quality: int = 80,
        resolved_extrinsic_json: Optional[str] = None,
        frame_roles_json: Optional[str] = None,
    ):
        if not parent_frame or not child_frame:
            raise ValueError("parent_frame and child_frame must not be empty")
        if not 1 <= int(jpeg_quality) <= 100:
            raise ValueError("jpeg_quality must be between 1 and 100")
        self.source = source
        self.frozen_extrinsic = None
        self.frame_roles = None
        if resolved_extrinsic_json is not None:
            self.frame_roles = parse_frame_roles(frame_roles_json)
            self.frozen_extrinsic = decode_frozen(resolved_extrinsic_json, camera_name, self.frame_roles)
            if (parent_frame != self.frame_roles["parentFrame"]
                    or child_frame != self.frame_roles["opticalFrames"].get(calibration_mode)):
                raise ValueError("calibrator frames do not match the controlled camera roles")
        self.calibration_mode = str(calibration_mode).strip()
        self.camera_name = str(camera_name).strip()
        self.parent_frame = parent_frame
        self.child_frame = child_frame
        self.ransac_threshold_px = float(ransac_threshold_px)
        self.maximum_inlier_error_px = float(maximum_inlier_error_px)
        self.jpeg_quality = int(jpeg_quality)
        self.lock = threading.RLock()
        self.generation = 0
        self.frozen: Optional[FrameSnapshot] = None
        self.frozen_jpeg: Optional[bytes] = None
        self.result: Optional[ExtrinsicResult] = None
        self.result_payload: Optional[Dict[str, Any]] = None
        self.candidate_id: Optional[str] = None
        self.candidate_points: Optional[Sequence[Dict[str, Any]]] = None
        self.candidate_metadata = None
        self._state_listeners = set()
        self.samples = SampleCollection(source, parent_frame, self.lock, self._samples_changed)

    def subscribe_state(self, callback):
        with self.lock:
            self._state_listeners.add(callback)
        subscribe = getattr(self.source, "subscribe_state", None)
        try:
            release_source = subscribe(callback) if callable(subscribe) else None
        except BaseException:
            with self.lock:
                self._state_listeners.discard(callback)
            raise
        def release():
            with self.lock:
                self._state_listeners.discard(callback)
            if release_source is not None:
                release_source()
        return release

    def _state_changed_locked(self):
        for callback in tuple(self._state_listeners):
            callback()

    def _samples_changed(self) -> None:
        self.result = None
        self.result_payload = None
        self.candidate_id = None
        self.candidate_points = None
        self.candidate_metadata = None
        self._state_changed_locked()

    def begin_sample(self, request):
        return self.samples.begin(request)

    def commit_sample(self, identity, payload, mime):
        self.samples.commit(identity, payload, mime)
        return self.state()

    def mutate_sample(self, action, request):
        self.samples.mutate(action, request)
        return self.state()

    def wait_ready(self, timeout):
        return self.source.wait_ready(timeout)

    def stop_readiness(self):
        self.source.stop_readiness()

    def read_result(self):
        with self.lock:
            if self.result is None:
                return None
            document = result_document(self.result, calibration_mode=self.calibration_mode,
                camera_name=self.camera_name, parent_frame=self.parent_frame, child_frame=self.child_frame,
                points=self.candidate_points, metadata={**self.candidate_metadata, "candidate_id": self.candidate_id})
            document.pop("created_at")
            document.update(sample_count=len(self.candidate_points),
                            inlier_count=len(self.result.inlier_indices),
                            reprojection_errors_px=self.result.reprojection_errors_px.tolist())
            return json.loads(json.dumps({"candidate_id": self.candidate_id, "document": document}, allow_nan=False))

    def _encode_jpeg(self, image: np.ndarray) -> bytes:
        ok, encoded = cv2.imencode(
            ".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), self.jpeg_quality]
        )
        if not ok:
            raise ApiError(HTTPStatus.INTERNAL_SERVER_ERROR, "Could not encode camera frame")
        return encoded.tobytes()

    def state(self) -> Dict[str, Any]:
        source_state = self.source.status()
        with self.lock:
            frozen = self.frozen
            result = self.result_payload
            payload: Dict[str, Any] = {
                **self.samples.state(),
                "mode": "frozen" if frozen is not None else "live",
                "generation": self.generation,
                "calibration_mode": self.calibration_mode,
                "camera_name": self.camera_name,
                "parent_frame": self.parent_frame,
                "child_frame": self.child_frame,
                "source": {**source_state, "source_id": self.source.source_id},
                "result": result,
            }
            if frozen is None:
                payload["frame"] = None
                payload["markers"] = []
            else:
                payload["frame"] = {
                    "stamp_sec": frozen.stamp_sec,
                    "frame_id": frozen.frame_id,
                    "width": frozen.width,
                    "height": frozen.height,
                    "camera_model": frozen.camera_model,
                    "pose_coordinates": frozen.pose_coordinates,
                }
                payload["markers"] = [
                    {
                        "name": marker.name,
                        "position": list(marker.position),
                    }
                    for marker in sorted(frozen.markers.values(), key=lambda item: item.name)
                ]
            return copy.deepcopy(payload)

    def freeze(self) -> Dict[str, Any]:
        snapshot = self.source.freeze(self.parent_frame)
        if snapshot.image.ndim != 3 or snapshot.image.shape[2] != 3:
            raise ApiError(HTTPStatus.CONFLICT, "Camera frame is not a BGR color image")
        intrinsic = np.asarray(snapshot.camera_matrix, dtype=np.float64)
        if (
            intrinsic.shape != (3, 3)
            or not np.all(np.isfinite(intrinsic))
            or intrinsic[0, 0] <= 0.0
            or intrinsic[1, 1] <= 0.0
        ):
            raise ApiError(
                HTTPStatus.CONFLICT,
                "Selected camera intrinsics are invalid",
            )
        if not snapshot.markers:
            raise ApiError(
                HTTPStatus.CONFLICT,
                "No pose marker is available",
            )
        # Copy mutable source buffers and capture model identity once. Later
        # status changes cannot replace evidence of the frozen solve.
        model = json.loads(json.dumps(dict(snapshot.camera_model), allow_nan=False))
        model.update({"camera_matrix": intrinsic.reshape(-1).tolist(),
                      "distortion": np.asarray(snapshot.distortion).reshape(-1).tolist(),
                      "image_width": snapshot.width, "image_height": snapshot.height,
                      "stamp_sec": snapshot.stamp_sec, "frame_id": snapshot.frame_id})
        model.setdefault("intrinsic_source", "unspecified")
        model.setdefault("distortion_model", "plumb_bob")
        snapshot = replace(snapshot, image=snapshot.image.copy(), camera_matrix=intrinsic.copy(),
                           distortion=np.asarray(snapshot.distortion).copy(),
                           markers=dict(snapshot.markers), camera_model=model,
                           pose_coordinates=json.loads(json.dumps(dict(snapshot.pose_coordinates), allow_nan=False)))
        encoded = self._encode_jpeg(snapshot.image)
        with self.lock:
            self.generation += 1
            self.frozen = snapshot
            self.frozen_jpeg = encoded
            self._state_changed_locked()
        return self.state()

    def live(self) -> Dict[str, Any]:
        with self.lock:
            changed = self.frozen is not None or self.frozen_jpeg is not None
            self.frozen = None
            self.frozen_jpeg = None
            if changed:
                self._state_changed_locked()
        return self.state()

    def image_jpeg(self) -> bytes:
        with self.lock:
            if self.frozen_jpeg is not None:
                return self.frozen_jpeg
        preview = self.source.preview_jpeg_bytes()
        if preview is None:
            raise ApiError(
                HTTPStatus.SERVICE_UNAVAILABLE,
                "No compressed camera preview has arrived",
            )
        return preview

    def solve(self, request: Any) -> Dict[str, Any]:
        if not isinstance(request, dict):
            raise ApiError(HTTPStatus.BAD_REQUEST, "Request body must be a JSON object")
        with self.lock:
            points, model, coordinates = self.samples.solve_inputs(request)
            world = [point["world"] for point in points]
            pixels = [point["pixel"] for point in points]
            matrix = np.asarray(model["camera_matrix"], dtype=np.float64).reshape(3, 3)
            distortion = np.asarray(model["distortion"], dtype=np.float64)

            try:
                result = solve_extrinsic(
                    world,
                    pixels,
                    matrix,
                    distortion,
                    ransac_reprojection_error_px=self.ransac_threshold_px,
                    maximum_accepted_error_px=self.maximum_inlier_error_px,
                )
            except (CalibrationError, cv2.error) as error:
                raise ApiError(HTTPStatus.UNPROCESSABLE_ENTITY, str(error)) from error

            inliers = set(map(int, result.inlier_indices))
            observed_points = [dict(point, inlier=index in inliers,
                reprojection_error_px=float(result.reprojection_errors_px[index]))
                for index, point in enumerate(points)]
            payload = _result_payload(result, [point["sample_id"] for point in points],
                                      np.asarray(world, dtype=np.float64), matrix, distortion)
            by_id = {point["sample_id"]: point for point in points}
            for projection in payload["projections"]:
                identity = projection.pop("marker")
                projection.update(sample_id=identity, marker=by_id[identity]["marker"])
            payload.update(points=observed_points, camera_model=model, pose_coordinates=coordinates,
                           dataset_revision=self.samples.revision)
            candidate_document = {
                "sampling_session_id": self.samples.session_id,
                "dataset_revision": self.samples.revision,
                "camera_model": model,
                "pose_coordinates": coordinates,
                "points": observed_points,
                "translation": payload["translation"],
                "quaternion_xyzw": payload["quaternion_xyzw"],
                "parent_frame": self.parent_frame,
                "child_frame": self.child_frame,
            }
            encoded = json.dumps(
                candidate_document, sort_keys=True, separators=(",", ":"), allow_nan=False
            ).encode("utf-8")
            candidate_id = "extrinsic-candidate-{}".format(hashlib.sha256(encoded).hexdigest())
            payload["candidate_id"] = candidate_id
            payload["commit_owner"] = "core"
            self.result = result
            self.result_payload = payload
            self.candidate_id = candidate_id
            self.candidate_points = observed_points
            self.candidate_metadata = json.loads(json.dumps({
                "camera_model": model, "pose_coordinates": coordinates,
                "sampling_session_id": self.samples.session_id, "dataset_revision": self.samples.revision,
                "image_topic": self.source.image_topic, "pose_prefix": self.source.pose_prefix,
            }, allow_nan=False))
            self._state_changed_locked()
            return payload


_RESPONSE_STARTED = "calibration.response_started"


class CalibrationHttpServer:
    """Public aiohttp adapter owned by one explicit XRPC runtime.

    ROS, locks, codecs, solver and evidence I/O run in the runtime's fixed
    blocking pool. Disconnecting a caller never releases a still-running job.
    """

    max_request_bytes = 128 * 1024
    static_files = {
        "/": "index.html", "/index.html": "index.html",
        "/app.js": "app.js", "/styles.css": "styles.css",
    }

    def __init__(self, address, service, web_root, *, frame_ancestors,
                 allowed_origins=(), logger=None, intrinsic_service=None,
                 runtime=None, limits=None, upload_timeout=30.0, rpc_socket=None, target_id=None,
                 validation_workers=2, lifecycle=None, peer_status=None, activate_peers=None,
                 peer_status_for_state=None):
        if service is None and intrinsic_service is None:
            raise ValueError("at least one of service / intrinsic_service is required")
        root = Path(web_root).resolve()
        for required in ("index.html", "app.js", "styles.css"):
            if not (root / required).is_file():
                raise FileNotFoundError("Web asset is missing: {}".format(root / required))
        if "\r" in frame_ancestors or "\n" in frame_ancestors:
            raise ValueError("frame_ancestors must not contain newlines")
        self.service, self.intrinsic_service = service, intrinsic_service
        self.lifecycle = lifecycle
        self.peer_status = peer_status or (lambda: {})
        self.peer_status_for_state = peer_status_for_state
        self.activate_peers = activate_peers or (lambda: None)
        self.web_root = root
        self.frame_ancestors = frame_ancestors.strip() or "'self'"
        self.allowed_origins = frozenset(allowed_origins)
        self.logger = logger or (lambda _message: None)
        self.upload_timeout = float(upload_timeout)
        if not math.isfinite(self.upload_timeout) or self.upload_timeout <= 0:
            raise ValueError("positive finite upload timeout required")
        self.limits = limits or Limits(connections=16, in_flight=8,
            body_bytes=IMAGE_BYTES, response_bytes=1 << 30, call_timeout=3600.0)
        private_reference = None
        if rpc_socket is not None:
            if str(rpc_socket) == "/" or str(rpc_socket).startswith("//") or "\x00" in str(rpc_socket):
                raise ValueError("A private canonical Unix socket path is required")
            private_reference = ServiceRef(target_id, "xgc2.calibration.v1.Calibration", "1",
                uuid.uuid4().hex, "http.v1", Endpoint("unix", str(rpc_socket))).validate()
        self.runtime = runtime or Runtime(blocking_workers=4, max_calls=8,
                                          max_connections=16)
        self._owns_runtime = runtime is None
        self._closed = False
        self._stopping = False
        # The one intrinsic detector belongs to this existing server, not to
        # each event response. A replacement waits for the prior physical join.
        self._live_capture_viewers = 0
        self._live_capture_owner = None
        self._live_capture_retirement = None
        self.router = AppRouter(client_max_size=self.limits.body_bytes)
        self.router.add_route("*", "/{path:.*}", self._handle)
        self.host = Host.from_app(self.router.app, address=address,
                                  runtime=self.runtime, limits=self.limits)
        self.private_host = None
        self.service_ref = None
        if rpc_socket is not None:
            self.service_ref = private_reference
            routes = self._private_routes()
            self.private_host = Host(str(rpc_socket), routes, runtime=self.runtime,
                instance_id=self.service_ref.instance_id, discovery_routes=("/v1/describe",),
                limits=Limits(connections=8, in_flight=4, body_bytes=128 * 1024,
                              response_bytes=8 << 20, call_timeout=35.0))
        if intrinsic_service is not None and hasattr(intrinsic_service, "attach_work_runtime"):
            intrinsic_service.attach_work_runtime(self.runtime, self.host, validation_workers=validation_workers)

    def _private_routes(self):
        """One thin native facade for actual Core consumers of this domain.

        The public browser edge is independent. The private host enforces
        official request identity, deadlines and instance fencing; domain
        objects keep their existing plain data and never receive a ServiceRef.
        """
        routes = {}
        def adapt(function, decoder=lambda value: (), status=200):
            def call(context, value):
                if self._stopping:
                    raise Fault("unavailable", "Calibration application is draining", 503)
                try:
                    return Response.json(function(*decoder(value)), status=status, max_bytes=8 << 20)
                except ApiError as error:
                    code = {400: "invalid_argument", 404: "not_found", 409: "conflict",
                            429: "resource_exhausted", 501: "unsupported", 503: "unavailable"}.get(error.status, "internal")
                    raise Fault(code, error.message, error.status) from error
            return call
        def empty(value):
            if value not in ({}, None):
                raise ApiError(400, "State/action request must be an empty object")
            return ()
        if self.service is not None:
            routes[("GET", "/api/v1/state")] = adapt(self.service.state, empty)
            routes[("POST", "/api/v1/solve")] = adapt(self.service.solve, lambda value: (value,))
        if self.intrinsic_service is not None:
            routes[("GET", "/api/v1/intrinsic/state")] = adapt(self.intrinsic_service.state, empty)
            routes[("POST", "/api/v1/intrinsic/candidate")] = adapt(self.intrinsic_service.start_candidate, empty, 202)
        domain = self.intrinsic_service or self.service
        kind = "intrinsic" if self.intrinsic_service is not None else "extrinsic"
        def read_result(context, value):
            if self._stopping:
                raise Fault("unavailable", "Calibration application is draining", 503)
            if value not in ({}, None):
                raise Fault("invalid_argument", "Result reads require an empty body", 400)
            if context.query:
                raise Fault("invalid_argument", "Result reads do not accept query arguments", 400)
            result = domain.read_result()
            revision = hashlib.sha256(json.dumps(result, sort_keys=True, separators=(",", ":"),
                                                 allow_nan=False).encode()).hexdigest()
            return {"kind": kind, "camera_id": domain.camera_name,
                    "calibration_mode": domain.calibration_mode, "revision": revision, "result": result}
        routes[("GET", "/v1/result")] = read_result
        async def describe(context, value):
            if self._stopping:
                raise Fault("unavailable", "Calibration application is draining", 503)
            empty(value)
            wait_ms = 0
            if context.query:
                query = parse_qs(context.query, keep_blank_values=True)
                wait = query.get("wait_ready_ms", [])
                if (set(query) != {"wait_ready_ms"} or len(wait) != 1
                        or len(wait[0]) > 5 or not wait[0].isascii() or not wait[0].isdigit()
                        or int(wait[0]) > 30000):
                    raise Fault("invalid_argument", "wait_ready_ms must be in [0, 30000]", 400)
                wait_ms = int(wait[0])
            # The application owns these HTTP endpoints. ROS, media and World
            # availability are peer facts, checked by the actual control calls.
            ready = not self._stopping
            capabilities = {"intrinsic": self.intrinsic_service is not None, "extrinsic": self.service is not None}
            return {"service": self.service_ref.service, "api_version": self.service_ref.api_version,
                "instance_id": self.service_ref.instance_id, "ready": ready,
                "facts": {"capabilities": capabilities, "camera_id": domain.camera_name,
                          "calibration_mode": domain.calibration_mode, "result_path": "/v1/result",
                          "result_kind": kind, "commit_owner": "core",
                          "ros_master_uri": os.environ.get("ROS_MASTER_URI", ""),
                          "peers": self.peer_status()},
                "service_ref": asdict(self.service_ref), "capabilities": capabilities,
                "routes": [{"method": method, "path": path} for method, path in sorted(routes)]}
        routes[("GET", "/v1/describe")] = describe
        return routes

    @property
    def server_address(self):
        return self.host.bound_address

    def start(self):
        try:
            self.host.start()
            if self.private_host is not None:
                self.private_host.start()
        except BaseException:
            self.host.close()
            if self.private_host is not None:
                self.private_host.close()
            if self._owns_runtime:
                self.runtime.close()
            raise
        return self

    def close(self):
        if self._closed:
            return
        self._stopping = True
        if self.private_host is not None:
            if self.lifecycle is not None:
                self.lifecycle.stop_readiness()
            (self.intrinsic_service or self.service).stop_readiness()
        # Host retains its resources if domain work has not really quiesced.
        self.host.close()
        if self.private_host is not None:
            self.private_host.close()
        if self._owns_runtime:
            self.runtime.close()
        self._closed = True

    async def _domain(self, function, *args):
        return await self.runtime.blocking(self.host, function, *args)

    def _headers(self, request):
        result = {
            "Cache-Control": "no-store", "X-Content-Type-Options": "nosniff",
            "Referrer-Policy": "no-referrer",
            "Content-Security-Policy":
                "default-src 'self'; base-uri 'none'; object-src 'none'; "
                "script-src 'self'; style-src 'self'; img-src 'self' blob: data:; "
                "connect-src 'self'; frame-ancestors {}".format(self.frame_ancestors),
        }
        origin = request.headers.get("Origin", "")
        if origin and ("*" in self.allowed_origins or origin in self.allowed_origins):
            result.update({"Access-Control-Allow-Origin": origin, "Vary": "Origin"})
        return result

    def _bytes(self, request, payload, mime, status=200):
        return web.Response(body=payload, status=int(status),
                            headers={**self._headers(request), "Content-Type": mime})

    def _json(self, request, payload, status=200):
        return self._bytes(request,
            json.dumps(payload, separators=(",", ":"), allow_nan=False).encode("utf-8"),
            "application/json; charset=utf-8", status)

    async def _read_body(self, request, maximum, timeout):
        length = request.content_length
        if length is None or length < 0 or length > maximum:
            raise ApiError(413, "Request body exceeds the byte limit or has no length")
        body = bytearray()
        async def read():
            async for chunk in iter_body(request, max_bytes=maximum):
                body.extend(chunk)
        try:
            await asyncio.wait_for(read(), timeout)
        except web.HTTPRequestEntityTooLarge as error:
            raise ApiError(413, "Request body exceeds the byte limit") from error
        except asyncio.TimeoutError as error:
            raise ApiError(408, "Request body upload timed out") from error
        if len(body) != length:
            raise ApiError(400, "Request body is incomplete")
        return bytes(body)

    async def _request_json(self, request):
        if request.content_type != "application/json":
            raise ApiError(415, "Content-Type must be application/json")
        raw = await self._read_body(request, self.max_request_bytes, self.upload_timeout)
        try:
            def fields(items):
                value = {}
                for key, item in items:
                    if key in value:
                        raise ValueError("duplicate field")
                    value[key] = item
                return value
            def invalid_constant(_):
                raise ValueError("nonfinite value")
            return json.loads(raw, object_pairs_hook=fields, parse_constant=invalid_constant)
        except (UnicodeDecodeError, ValueError, RecursionError) as error:
            raise ApiError(400, "Request body is not valid JSON") from error

    def _intrinsic(self):
        if self.intrinsic_service is None:
            raise ApiError(404, "Intrinsic calibration is not enabled")
        return self.intrinsic_service

    def _extrinsic(self):
        if self.service is None:
            raise ApiError(404, "Extrinsic calibration is not enabled")
        return self.service

    def _retain_state_observation(self, domain):
        if self._live_capture_owner is None:
            previous = self._live_capture_retirement
            cohort = {"active": True, "state": None, "payload": None, "revision": 0,
                      "tick": asyncio.get_running_loop().create_future(), "error": None, "pump": None,
                      "changed": asyncio.Event(), "subscriptions": []}
            loop = asyncio.get_running_loop()

            def changed():
                if cohort["active"]:
                    try:
                        loop.call_soon_threadsafe(cohort["changed"].set)
                    except RuntimeError:
                        pass

            async def acquire():
                if previous is not None:
                    await asyncio.shield(previous)
                if not cohort["active"]:
                    return False
                def subscribe_sources():
                    for source in (domain, self.lifecycle):
                        subscribe = getattr(source, "subscribe_state", None)
                        if callable(subscribe):
                            cohort["subscriptions"].append(subscribe(changed))
                await self._domain(subscribe_sources)
                if not cohort["active"]:
                    return False
                self.activate_peers()
                interval = getattr(domain, "live_capture_interval", None) if domain is self.intrinsic_service else None
                retained = interval is not None
                if retained:
                    await self._domain(domain.retain_live_capture, interval)
                if cohort["active"]:
                    cohort["changed"].set()
                    cohort["pump"] = asyncio.create_task(self._pump_state(domain, cohort))
                return retained

            self._live_capture_owner = (asyncio.create_task(acquire()), cohort)
        self._live_capture_viewers += 1
        return self._live_capture_owner

    @staticmethod
    def _wake_intrinsic_observers(cohort):
        tick = cohort["tick"]
        cohort["tick"] = asyncio.get_running_loop().create_future()
        if not tick.done():
            tick.set_result(None)

    async def _pump_state(self, domain, cohort):
        """Project only source changes, once for all current event readers."""
        try:
            while cohort["active"]:
                await cohort["changed"].wait()
                cohort["changed"].clear()
                if not cohort["active"]:
                    break
                state = await self._domain(self._state, domain)
                if not cohort["active"]:
                    break
                if state != cohort["state"]:
                    payload = json.dumps(state, separators=(",", ":"), allow_nan=False).encode()
                    # Domain state snapshots are detached under their owner lock.
                    cohort["state"] = state
                    if payload != cohort["payload"]:
                        cohort["payload"] = payload
                        cohort["revision"] += 1
                        self._wake_intrinsic_observers(cohort)
        except Exception as error:
            cohort["error"] = error
        finally:
            self._wake_intrinsic_observers(cohort)

    def _release_state_observation(self, domain, owner):
        if self._live_capture_owner is not owner or self._live_capture_viewers <= 0:
            raise RuntimeError("intrinsic observation is not retained")
        self._live_capture_viewers -= 1
        if self._live_capture_viewers:
            return None
        self._live_capture_owner = None
        acquired, cohort = owner
        cohort["active"] = False
        cohort["changed"].set()

        async def retire():
            try:
                retained = await asyncio.shield(acquired)
            except Exception:
                retained = False
            try:
                pump = cohort["pump"]
                if pump is not None:
                    await asyncio.shield(pump)
            finally:
                try:
                    if cohort["subscriptions"]:
                        def release_sources():
                            for release in reversed(cohort["subscriptions"]):
                                release()
                        await self._domain(release_sources)
                finally:
                    if retained:
                        await self._domain(domain.release_live_capture)

        retirement = asyncio.create_task(retire())
        # The task remains owned even if the request is cancelled a second
        # time while its physical detector join is still in progress.
        retirement.add_done_callback(lambda task: task.exception() if not task.cancelled() else None)
        self._live_capture_retirement = retirement
        return retirement

    async def _events(self, request):
        domain = self.intrinsic_service or self.service
        response = RawStreamResponse(max_bytes=self.limits.response_bytes, headers={**self._headers(request),
                                               "Content-Type": "text/event-stream"})
        if request.method == "HEAD":
            await response.prepare(request)
            request[_RESPONSE_STARTED] = True
            await response.write_eof()
            return response
        owner = None
        try:
            owner = self._retain_state_observation(domain)
            await asyncio.shield(owner[0])
            await response.prepare(request)
            request[_RESPONSE_STARTED] = True
            event_id = 0
            cohort = owner[1]
            revision = 0
            while request.transport is not None and not request.transport.is_closing():
                if cohort["error"] is not None:
                    raise cohort["error"]
                if cohort["revision"] != revision:
                    revision = cohort["revision"]
                    event_id += 1
                    await response.write("id: {}\nevent: state\ndata: ".format(event_id).encode()
                                         + cohort["payload"] + b"\n\n")
                else:
                    await asyncio.shield(cohort["tick"])
        finally:
            if owner is not None:
                retirement = self._release_state_observation(domain, owner)
                if retirement is not None:
                    await asyncio.shield(retirement)
        return response

    def _state(self, domain):
        state = domain.state()
        state["peers"] = (self.peer_status_for_state(state) if self.peer_status_for_state is not None
                          else self.peer_status())
        if self.lifecycle is not None:
            state["ready"] = self.lifecycle.wait_ready(0)
        return state

    async def _file(self, request, path, filename):
        if Path(filename).name != filename or not filename:
            raise ApiError(500, "Download filename is invalid")
        # One open file, one bounded chunk and native async backpressure.
        io_task = asyncio.create_task(self._domain(path.open, "rb"))
        stream = None
        try:
            stream = await asyncio.shield(io_task)
            io_task = asyncio.create_task(self._domain(lambda: path.stat().st_size))
            size = await asyncio.shield(io_task)
            if size > self.limits.response_bytes:
                raise ApiError(413, "Evidence download exceeds the response byte limit")
            response = RawStreamResponse(max_bytes=self.limits.response_bytes, headers={**self._headers(request),
                "Content-Type": "application/zip", "Content-Length": str(size),
                "Content-Disposition": 'attachment; filename="{}"'.format(filename)})
            await response.prepare(request)
            request[_RESPONSE_STARTED] = True
            if request.method != "HEAD":
                while True:
                    io_task = asyncio.create_task(self._domain(stream.read, 1024 * 1024))
                    chunk = await asyncio.shield(io_task)
                    if not chunk:
                        break
                    await response.write(chunk)
            await response.write_eof()
            return response
        finally:
            # Await a real read/open before closing its descriptor. A cancelled
            # HTTP task must not close a file still used by the blocking worker.
            try:
                result = await asyncio.shield(io_task)
                if stream is None:
                    stream = result
            finally:
                if stream is not None:
                    stream.close()  # read-only descriptor; no flush or disk work

    async def _dispatch(self, request):
        path = request.path
        if request.method == "OPTIONS":
            headers = self._headers(request)
            if "Access-Control-Allow-Origin" in headers:
                headers.update({"Access-Control-Allow-Methods": "GET, HEAD, POST, PUT, OPTIONS",
                                "Access-Control-Allow-Headers": "Content-Type"})
            return web.Response(status=204, headers=headers)
        if request.method in ("GET", "HEAD"):
            image_match = IMAGE_PATH.fullmatch(path)
            if image_match:
                payload, mime = await self._domain(self._extrinsic().samples.image, image_match[1])
                return self._bytes(request, payload, mime)
            if path == "/healthz":
                payload = {"status": "ok"}
                if self.service is not None:
                    state = await self._domain(self.service.state)
                    payload.update(image_ready=bool(state["source"].get("image_ready")),
                        intrinsic_ready=bool(state["source"].get("intrinsic_ready")),
                        marker_count=int(state["source"].get("marker_count", 0)))
                if self.intrinsic_service is not None:
                    state = await self._domain(self.intrinsic_service.state)
                    payload.setdefault("image_ready", bool(state.get("image_ready")))
                    payload["camera_control"] = bool(state.get("camera_control"))
                return self._json(request, payload)
            if path == "/api/v1/state":
                return self._json(request, await self._domain(self._state, self._extrinsic()))
            if path == "/api/v1/image.jpg":
                return self._bytes(request, await self._domain(self._extrinsic().image_jpeg), "image/jpeg")
            if path in ("/api/v1/intrinsic/events", "/api/v1/events"):
                return await self._events(request)
            reads = {
                "/api/v1/intrinsic/state": ("state", "json"),
                "/api/v1/intrinsic/image.jpg": ("image_jpeg", "image/jpeg"),
                "/api/v1/intrinsic/snapshot.jpg": ("snapshot_jpeg", "image/jpeg"),
                "/api/v1/intrinsic/targets": ("targets_document", "json"),
                "/api/v1/intrinsic/calibrations": ("calibration_history", "json"),
            }
            if path in reads:
                method, mime = reads[path]
                payload = (await self._domain(self._state, self._intrinsic()) if method == "state"
                           else await self._domain(getattr(self._intrinsic(), method)))
                return self._json(request, payload) if mime == "json" else self._bytes(request, payload, mime)
            if path == "/api/v1/intrinsic/evidence.zip":
                filename, evidence_path = await self._domain(self._intrinsic().evidence_bundle)
                return await self._file(request, evidence_path, filename)
            if path.startswith("/api/v1/intrinsic/validation/image/"):
                image_id = path[len("/api/v1/intrinsic/validation/image/"):]
                if not image_id.endswith(".jpg"):
                    raise ApiError(404, "Intrinsic validation image must be JPEG")
                values = request.query.getall("generation", [])
                generation = None
                if values:
                    try:
                        if len(values) != 1:
                            raise ValueError()
                        generation = int(values[0])
                        if generation <= 0:
                            raise ValueError()
                    except ValueError as error:
                        raise ApiError(400, "Intrinsic validation generation must be a positive integer") from error
                payload = await self._domain(self._intrinsic().validation_image, image_id[:-4], generation)
                return self._bytes(request, payload, "image/jpeg")
            if path.startswith("/api/v1/intrinsic/ref/"):
                try:
                    index = int(path[len("/api/v1/intrinsic/ref/"):].split(".", 1)[0])
                except ValueError as error:
                    raise ApiError(400, "Reference index must be an integer") from error
                payload = await self._domain(self._intrinsic().ref, index)
                if payload is None:
                    raise ApiError(404, "No reference image for that target")
                return self._bytes(request, payload, "image/jpeg")
            asset = self.static_files.get(path)
            if asset:
                payload = await self._domain((self.web_root / asset).read_bytes)
                mime = mimetypes.guess_type(asset)[0] or "application/octet-stream"
                if mime.startswith("text/") or mime in ("application/javascript", "application/json"):
                    mime += "; charset=utf-8"
                return self._bytes(request, payload, mime)
            raise ApiError(404, "Route not found")
        if request.method != "POST":
            raise ApiError(405, "Method not allowed")
        image_match = IMAGE_PATH.fullmatch(path)
        if image_match:
            mime = request.headers.get("Content-Type", "").strip().lower()
            if mime not in ("image/png", "image/jpeg"):
                raise ApiError(415, "Sample image must be PNG or JPEG")
            payload = await self._read_body(request, IMAGE_BYTES, self.upload_timeout)
            return self._json(request, await self._domain(self._extrinsic().commit_sample,
                                                        image_match[1], payload, mime))
        value = await self._request_json(request)
        if path == "/api/v1/samples/begin":
            return self._json(request, await self._domain(self._extrinsic().begin_sample, value))
        if path in {"/api/v1/samples/" + action for action in ("cancel", "remove", "pixel", "clear")}:
            return self._json(request, await self._domain(self._extrinsic().mutate_sample,
                                                        path.rsplit("/", 1)[1], value))
        if path == "/api/v1/solve":
            return self._json(request, await self._domain(self._extrinsic().solve, value))
        if path == "/api/v1/intrinsic/apply-metadata":
            if not isinstance(value, dict) or set(value) != {"candidate_id"} or not isinstance(value["candidate_id"], str):
                raise ApiError(400, "Apply requires only candidate_id")
            return self._json(request, await self._domain(self._intrinsic().apply_metadata, value["candidate_id"]))
        if path == "/api/v1/intrinsic/goto":
            index = value.get("index") if isinstance(value, dict) else None
            if not isinstance(index, int) or isinstance(index, bool):
                raise ApiError(400, "goto requires an integer 'index'")
            return self._json(request, await self._domain(self._intrinsic().goto, index))
        if path == "/api/v1/intrinsic/validation":
            if not isinstance(value, dict) or set(value) != {"reference", "comparison"}:
                raise ApiError(400, "Intrinsic validation requires reference and comparison objects")
            return self._json(request, await self._domain(self._intrinsic().validate_intrinsic,
                                                        value["reference"], value["comparison"]))
        empty_actions = {
            "/api/v1/freeze": ("extrinsic", "freeze", 200),
            "/api/v1/live": ("extrinsic", "live", 200),
            "/api/v1/intrinsic/candidate": ("intrinsic", "start_candidate", 202),
            "/api/v1/intrinsic/continue": ("intrinsic", "continue_collection", 200),
            "/api/v1/intrinsic/reset": ("intrinsic", "reset", 200),
            "/api/v1/intrinsic/capture": ("intrinsic", "capture", 200),
            "/api/v1/intrinsic/auto_capture/stop": ("intrinsic", "stop_auto_capture", 200),
            "/api/v1/intrinsic/reset_pose": ("intrinsic", "reset_pose", 200),
            "/api/v1/intrinsic/auto_run": ("intrinsic", "auto_run", 202),
        }
        if path in empty_actions or path == "/api/v1/intrinsic/auto_capture/start":
            if value not in ({}, None):
                raise ApiError(400, "Action request must be an empty object")
            if path == "/api/v1/intrinsic/auto_capture/start":
                service = self._intrinsic()
                def start_capture():
                    with service.lock:
                        interval = service._resume_auto_capture_interval_locked()
                    return service.start_auto_capture(interval=interval)
                return self._json(request, await self._domain(start_capture))
            kind, method, status = empty_actions[path]
            service = self._extrinsic() if kind == "extrinsic" else self._intrinsic()
            payload = await self._domain(getattr(service, method))
            if method == "start_candidate" and not payload.get("accepted"):
                status = 200
            return self._json(request, payload, status)
        raise ApiError(404, "Route not found")

    async def _handle(self, request):
        try:
            return await self._dispatch(request)
        except PeerUnavailable as error:
            return self._json(request, {"error": str(error)}, 503)
        except (ApiError, Fault) as error:
            if request.get(_RESPONSE_STARTED):
                if request.transport:
                    request.transport.close()
                raise
            payload = {"error": error.message if isinstance(error, ApiError) else str(error)}
            if isinstance(error, ApiError) and error.details:
                payload["details"] = error.details
            response = self._json(request, payload, error.status)
            if request.can_read_body:
                response.force_close()
            return response
        except TimeoutError:
            if request.transport:
                request.transport.close()
            raise
        except (BrokenPipeError, ConnectionResetError):
            raise
        except Exception:
            if request.get(_RESPONSE_STARTED):
                if request.transport:
                    request.transport.close()
                raise
            self.logger("Unhandled calibration HTTP request failure")
            response = self._json(request, {"error": "Internal server error"}, 500)
            response.force_close()
            return response
