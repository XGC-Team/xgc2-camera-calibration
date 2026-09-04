"""HTTP-independent camera extrinsic calibration service and web transport."""

from __future__ import annotations

import hashlib
import json
import math
import mimetypes
import threading
import time
from dataclasses import dataclass
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Callable, Dict, Mapping, Optional, Sequence, Tuple
from urllib.parse import parse_qs, urlsplit

import cv2
import numpy as np

from xgc_camera_calibration.solver import (
    CalibrationError,
    ExtrinsicResult,
    extrinsic_calibration_directory,
    load_extrinsic_selection,
    save_extrinsic,
    solve_extrinsic,
    versioned_extrinsic_path,
    write_extrinsic_selection,
)


class ApiError(RuntimeError):
    """An expected request or calibration-input failure."""

    def __init__(self, status: int, message: str, *, details: Optional[Dict[str, Any]] = None):
        super().__init__(message)
        self.status = int(status)
        self.message = str(message)
        self.details = dict(details or {})


@dataclass(frozen=True)
class MarkerObservation:
    name: str
    position: Tuple[float, float, float]
    frame_id: str


@dataclass(frozen=True)
class FrameSnapshot:
    image: np.ndarray
    stamp_sec: float
    frame_id: str
    camera_matrix: np.ndarray
    distortion: np.ndarray
    markers: Mapping[str, MarkerObservation]

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
        calibration_root: str,
        calibration_mode: str,
        camera_name: str,
        parent_frame: str,
        child_frame: str,
        ransac_threshold_px: float = 3.0,
        maximum_inlier_error_px: float = 5.0,
        jpeg_quality: int = 80,
    ):
        if not parent_frame or not child_frame:
            raise ValueError("parent_frame and child_frame must not be empty")
        if not 1 <= int(jpeg_quality) <= 100:
            raise ValueError("jpeg_quality must be between 1 and 100")
        self.source = source
        self.output_directory = extrinsic_calibration_directory(
            calibration_root, calibration_mode, camera_name
        )
        self.calibration_mode = str(calibration_mode).strip()
        self.camera_name = str(camera_name).strip()
        self.output_file: Optional[str] = None
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
        self.saved_candidate_id: Optional[str] = None
        self.candidate_points: Optional[Sequence[Dict[str, Any]]] = None
        self.result_restored = False
        self._pending_output_file: Optional[Tuple[str, Path]] = None
        self.recovery_error: Optional[str] = None
        try:
            self._restore_selected_result()
        except Exception as error:
            self.recovery_error = str(error)

    def _restore_selected_result(self) -> None:
        restored = load_extrinsic_selection(
            str(self.output_directory.parents[1]), self.calibration_mode, self.camera_name
        )
        if restored is None:
            return
        output_file, document, selection = restored
        if (
            document.get("parent_frame") != self.parent_frame
            or document.get("child_frame") != self.child_frame
        ):
            raise CalibrationError("selected extrinsic frame identity does not match")
        metadata = document.get("metadata", {})
        if not isinstance(metadata, Mapping):
            raise CalibrationError("selected extrinsic metadata must be an object")
        points = document.get("points", [])
        if not isinstance(points, list):
            raise CalibrationError("selected extrinsic points must be an array")
        inliers = document.get("inlier_indices", [])
        warnings = document.get("warnings", [])
        if not isinstance(inliers, list) or not isinstance(warnings, list):
            raise CalibrationError("selected extrinsic diagnostics are invalid")
        candidate_id = str(selection["candidate_id"])
        self.output_file = str(output_file)
        self.candidate_id = candidate_id
        self.saved_candidate_id = candidate_id
        self.candidate_points = list(points)
        self.result_payload = {
            "candidate_id": candidate_id,
            "saved": True,
            "translation": [float(value) for value in document["translation_array"]],
            "quaternion_xyzw": [
                float(value) for value in document["quaternion_xyzw_array"]
            ],
            "mean_reprojection_error_px": float(
                document.get("mean_reprojection_error_px", 0.0)
            ),
            "max_reprojection_error_px": float(
                document.get("max_reprojection_error_px", 0.0)
            ),
            "inlier_indices": [int(value) for value in inliers],
            "warnings": [str(value) for value in warnings],
            "projections": [],
            "points": list(points),
            "output_file": str(output_file),
            "save_blocked": None,
            "selection_file": str(
                self.output_directory.parents[1]
                / "selections" / self.camera_name
                / "{}-extrinsic.json".format(self.calibration_mode)
            ),
        }
        if metadata.get("candidate_id") != candidate_id:
            raise CalibrationError("selected extrinsic candidate identity is invalid")
        self.result_restored = True

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
                "mode": "frozen" if frozen is not None else "live",
                "generation": self.generation,
                "output_file": self.output_file,
                "saved_candidate_id": self.saved_candidate_id,
                "calibration_mode": self.calibration_mode,
                "camera_name": self.camera_name,
                "parent_frame": self.parent_frame,
                "child_frame": self.child_frame,
                "result_restored": self.result_restored,
                "recovery_error": self.recovery_error,
                "source": source_state,
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
                }
                payload["markers"] = [
                    {
                        "name": marker.name,
                        "position": list(marker.position),
                    }
                    for marker in sorted(frozen.markers.values(), key=lambda item: item.name)
                ]
            return payload

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
        encoded = self._encode_jpeg(snapshot.image)
        with self.lock:
            self.generation += 1
            self.frozen = snapshot
            self.frozen_jpeg = encoded
            self.result = None
            self.result_payload = None
            self.candidate_id = None
            self.saved_candidate_id = None
            self.candidate_points = None
            self.result_restored = False
            self._pending_output_file = None
            self.recovery_error = None
            self.output_file = None
        return self.state()

    def live(self) -> Dict[str, Any]:
        with self.lock:
            self.frozen = None
            self.frozen_jpeg = None
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
            snapshot = self.frozen
            if snapshot is None:
                raise ApiError(HTTPStatus.CONFLICT, "Freeze a camera frame first")
            try:
                generation = int(request.get("generation"))
            except (TypeError, ValueError) as error:
                raise ApiError(HTTPStatus.BAD_REQUEST, "generation must be an integer") from error
            if generation != self.generation:
                raise ApiError(
                    HTTPStatus.CONFLICT,
                    "Frozen frame changed; clear the browser selection and try again",
                )
            points = request.get("points")
            if not isinstance(points, list) or len(points) < 4:
                raise ApiError(
                    HTTPStatus.BAD_REQUEST,
                    "At least four marker-to-pixel correspondences are required",
                )
            if len(points) > len(snapshot.markers):
                raise ApiError(HTTPStatus.BAD_REQUEST, "More points than available markers")

            seen = set()
            marker_names = []
            world = []
            pixels = []
            for index, item in enumerate(points):
                if not isinstance(item, dict):
                    raise ApiError(
                        HTTPStatus.BAD_REQUEST,
                        "points[{}] must be an object".format(index),
                    )
                marker_name = item.get("marker")
                if not isinstance(marker_name, str) or marker_name not in snapshot.markers:
                    raise ApiError(
                        HTTPStatus.BAD_REQUEST,
                        "points[{}] references an unavailable marker".format(index),
                    )
                if marker_name in seen:
                    raise ApiError(
                        HTTPStatus.BAD_REQUEST,
                        "Marker '{}' is selected more than once".format(marker_name),
                    )
                pixel = _finite_pixel(item.get("pixel"), "points[{}].pixel".format(index))
                if not (0.0 <= pixel[0] < snapshot.width and 0.0 <= pixel[1] < snapshot.height):
                    raise ApiError(
                        HTTPStatus.BAD_REQUEST,
                        "points[{}].pixel is outside the frozen image".format(index),
                    )
                seen.add(marker_name)
                marker_names.append(marker_name)
                world.append(snapshot.markers[marker_name].position)
                pixels.append(pixel)

            try:
                result = solve_extrinsic(
                    world,
                    pixels,
                    snapshot.camera_matrix,
                    snapshot.distortion,
                    ransac_reprojection_error_px=self.ransac_threshold_px,
                    maximum_accepted_error_px=self.maximum_inlier_error_px,
                )
            except (CalibrationError, cv2.error) as error:
                raise ApiError(HTTPStatus.UNPROCESSABLE_ENTITY, str(error)) from error

            inliers = set(map(int, result.inlier_indices))
            persisted_points = []
            for index, (name, pixel, position) in enumerate(zip(marker_names, pixels, world)):
                persisted_points.append(
                    {
                        "marker": name,
                        "pixel": list(map(float, pixel)),
                        "world": list(map(float, position)),
                        "inlier": index in inliers,
                        "reprojection_error_px": float(result.reprojection_errors_px[index]),
                    }
                )
            all_names = sorted(snapshot.markers)
            all_world = np.asarray(
                [snapshot.markers[name].position for name in all_names], dtype=np.float64
            )
            payload = _result_payload(
                result,
                all_names,
                all_world,
                np.asarray(snapshot.camera_matrix, dtype=np.float64),
                np.asarray(snapshot.distortion, dtype=np.float64),
            )
            payload["points"] = persisted_points
            candidate_document = {
                "generation": self.generation,
                "points": persisted_points,
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
            payload["saved"] = False
            payload["output_file"] = None
            payload["save_blocked"] = "explicit_save_required"
            self.output_file = None
            self.result = result
            self.result_payload = payload
            self.candidate_id = candidate_id
            self.saved_candidate_id = None
            self.candidate_points = persisted_points
            if (
                self._pending_output_file is not None
                and self._pending_output_file[0] != candidate_id
            ):
                self._pending_output_file = None
            return payload

    def save(self, candidate_id: str) -> Dict[str, Any]:
        identity = str(candidate_id).strip()
        if not identity:
            raise ApiError(HTTPStatus.BAD_REQUEST, "candidate_id must not be empty")
        with self.lock:
            if self.saved_candidate_id is not None:
                if identity == self.saved_candidate_id and self.result_payload is not None:
                    try:
                        selected = load_extrinsic_selection(
                            str(self.output_directory.parents[1]),
                            self.calibration_mode,
                            self.camera_name,
                        )
                    except Exception as error:
                        raise ApiError(
                            HTTPStatus.CONFLICT,
                            "Shared extrinsic selection is no longer valid: {}".format(error),
                        ) from error
                    if selected is None:
                        raise ApiError(
                            HTTPStatus.CONFLICT,
                            "Shared extrinsic selection is no longer available",
                        )
                    if (
                        str(selected[2]["candidate_id"]) != identity
                        or self.output_file is None
                        or selected[0] != Path(self.output_file).resolve()
                    ):
                        raise ApiError(
                            HTTPStatus.CONFLICT,
                            "A newer shared extrinsic selection superseded this candidate",
                        )
                    return dict(self.result_payload)
                raise ApiError(HTTPStatus.CONFLICT, "A different extrinsic candidate is already saved")
            if self.result is None or self.result_payload is None or self.candidate_points is None:
                raise ApiError(HTTPStatus.CONFLICT, "No extrinsic candidate is ready")
            if identity != self.candidate_id:
                raise ApiError(
                    HTTPStatus.CONFLICT,
                    "Extrinsic candidate changed; solve the frozen correspondences again",
                    details={"expected_candidate_id": self.candidate_id},
                )
            snapshot = self.frozen
            if snapshot is None:
                raise ApiError(HTTPStatus.CONFLICT, "Frozen frame is unavailable")
            try:
                pending = self._pending_output_file
                output_file = pending[1] if pending is not None and pending[0] == identity else None
                if output_file is None:
                    output_file = versioned_extrinsic_path(self.output_directory)
                    save_extrinsic(
                        output_file,
                        self.result,
                        calibration_mode=self.calibration_mode,
                        camera_name=self.camera_name,
                        parent_frame=self.parent_frame,
                        child_frame=self.child_frame,
                        points=self.candidate_points,
                        metadata={
                            "candidate_id": identity,
                            "image_topic": self.source.image_topic,
                            "intrinsic_file": str(self.source.intrinsic_file),
                            "pose_prefix": self.source.pose_prefix,
                            "image_width": snapshot.width,
                            "image_height": snapshot.height,
                            "web_calibrator": True,
                        },
                    )
                    self._pending_output_file = (identity, output_file)
                selection_file = write_extrinsic_selection(
                    str(self.output_directory.parents[1]),
                    self.calibration_mode,
                    self.camera_name,
                    output_file,
                    identity,
                )
            except (OSError, ValueError, CalibrationError) as error:
                raise ApiError(
                    HTTPStatus.INTERNAL_SERVER_ERROR,
                    "Could not save or select calibration result: {}".format(error),
                ) from error
            self.output_file = str(output_file)
            self.saved_candidate_id = identity
            self.result_payload = {
                **self.result_payload,
                "saved": True,
                "output_file": self.output_file,
                "selection_file": str(selection_file),
                "save_blocked": None,
            }
            self.result_restored = False
            self._pending_output_file = None
            self.recovery_error = None
            return dict(self.result_payload)


class CalibrationHttpServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True

    def __init__(
        self,
        address: Tuple[str, int],
        service: Optional[CalibrationService],
        web_root: Path,
        *,
        frame_ancestors: str,
        allowed_origins: Sequence[str] = (),
        logger: Optional[Callable[[str], None]] = None,
        intrinsic_service: Optional[Any] = None,
    ):
        # The extrinsic and intrinsic calibrators are separate apps that share
        # this transport; each runs with only its own service present.
        if service is None and intrinsic_service is None:
            raise ValueError("at least one of service / intrinsic_service is required")
        root = Path(web_root).resolve()
        for required in ("index.html", "app.js", "styles.css"):
            if not (root / required).is_file():
                raise FileNotFoundError("Web asset is missing: {}".format(root / required))
        if "\r" in frame_ancestors or "\n" in frame_ancestors:
            raise ValueError("frame_ancestors must not contain newlines")
        self.service = service
        self.intrinsic_service = intrinsic_service
        self.web_root = root
        self.frame_ancestors = frame_ancestors.strip() or "'self'"
        self.allowed_origins = set(allowed_origins)
        self.logger = logger or (lambda _message: None)
        super().__init__(address, CalibrationRequestHandler)


class CalibrationRequestHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    max_request_bytes = 128 * 1024
    static_files = {
        "/": "index.html",
        "/index.html": "index.html",
        "/app.js": "app.js",
        "/styles.css": "styles.css",
    }

    @property
    def calibration_server(self) -> CalibrationHttpServer:
        return self.server  # type: ignore[return-value]

    def log_message(self, format_string: str, *args: Any) -> None:
        self.calibration_server.logger(format_string % args)

    def _origin(self) -> Optional[str]:
        origin = self.headers.get("Origin", "")
        allowed = self.calibration_server.allowed_origins
        if not origin or not allowed:
            return None
        if "*" in allowed or origin in allowed:
            return origin
        return None

    def _common_headers(self, content_type: str, length: int) -> None:
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(length))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Referrer-Policy", "no-referrer")
        self.send_header(
            "Content-Security-Policy",
            "default-src 'self'; base-uri 'none'; object-src 'none'; "
            "script-src 'self'; style-src 'self'; img-src 'self' blob: data:; "
            "connect-src 'self'; frame-ancestors {}".format(
                self.calibration_server.frame_ancestors
            ),
        )
        origin = self._origin()
        if origin:
            self.send_header("Access-Control-Allow-Origin", origin)
            self.send_header("Vary", "Origin")

    def _send_bytes(self, status: int, content_type: str, payload: bytes) -> None:
        self.send_response(int(status))
        self._common_headers(content_type, len(payload))
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(payload)

    def _send_file(
        self,
        status: int,
        content_type: str,
        path: Path,
        download_name: str,
    ) -> None:
        if Path(download_name).name != download_name or not download_name:
            raise ApiError(HTTPStatus.INTERNAL_SERVER_ERROR, "Download filename is invalid")
        size = path.stat().st_size
        self.send_response(int(status))
        self._common_headers(content_type, size)
        self.send_header(
            "Content-Disposition", 'attachment; filename="{}"'.format(download_name)
        )
        self.end_headers()
        if self.command == "HEAD":
            return
        with path.open("rb") as stream:
            while True:
                chunk = stream.read(1024 * 1024)
                if not chunk:
                    break
                self.wfile.write(chunk)

    def _send_json(self, status: int, payload: Dict[str, Any]) -> None:
        encoded = json.dumps(payload, separators=(",", ":"), allow_nan=False).encode("utf-8")
        self._send_bytes(status, "application/json; charset=utf-8", encoded)

    def _send_intrinsic_state_events(self) -> None:
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        origin = self._origin()
        if origin:
            self.send_header("Access-Control-Allow-Origin", origin)
            self.send_header("Vary", "Origin")
        self.end_headers()
        previous = b""
        event_id = 0
        while True:
            payload = json.dumps(
                self._intrinsic().state(), separators=(",", ":"), allow_nan=False
            ).encode("utf-8")
            if payload != previous:
                event_id += 1
                self.wfile.write(
                    "id: {}\nevent: state\ndata: ".format(event_id).encode("ascii")
                    + payload
                    + b"\n\n"
                )
                self.wfile.flush()
                previous = payload
            time.sleep(0.1)

    def _send_error(self, error: ApiError) -> None:
        payload: Dict[str, Any] = {"error": error.message}
        if error.details:
            payload["details"] = error.details
        self._send_json(error.status, payload)

    def _request_json(self) -> Any:
        content_type = self.headers.get("Content-Type", "").split(";", 1)[0].strip().lower()
        if content_type != "application/json":
            raise ApiError(HTTPStatus.UNSUPPORTED_MEDIA_TYPE, "Content-Type must be application/json")
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError as error:
            raise ApiError(HTTPStatus.BAD_REQUEST, "Invalid Content-Length") from error
        if length < 0 or length > self.max_request_bytes:
            raise ApiError(HTTPStatus.REQUEST_ENTITY_TOO_LARGE, "Request body is too large")
        try:
            return json.loads(self.rfile.read(length).decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ApiError(HTTPStatus.BAD_REQUEST, "Request body is not valid JSON") from error

    def _intrinsic(self) -> Any:
        service = self.calibration_server.intrinsic_service
        if service is None:
            raise ApiError(HTTPStatus.NOT_FOUND, "Intrinsic calibration is not enabled")
        return service

    def _extrinsic(self) -> Any:
        service = self.calibration_server.service
        if service is None:
            raise ApiError(HTTPStatus.NOT_FOUND, "Extrinsic calibration is not enabled")
        return service

    def _intrinsic_ref(self, path: str) -> bytes:
        token = path[len("/api/v1/intrinsic/ref/"):].split(".", 1)[0]
        try:
            index = int(token)
        except ValueError as error:
            raise ApiError(HTTPStatus.BAD_REQUEST, "Reference index must be an integer") from error
        jpeg = self._intrinsic().ref(index)
        if jpeg is None:
            raise ApiError(HTTPStatus.NOT_FOUND, "No reference image for that target")
        return jpeg

    def _intrinsic_validation_image(self, path: str, query: str) -> bytes:
        prefix = "/api/v1/intrinsic/validation/image/"
        token = path[len(prefix):]
        if not token.endswith(".jpg"):
            raise ApiError(HTTPStatus.NOT_FOUND, "Intrinsic validation image must be JPEG")
        parameters = parse_qs(query, keep_blank_values=True)
        generation_values = parameters.get("generation")
        generation = None
        if generation_values is not None:
            if len(generation_values) != 1:
                raise ApiError(
                    HTTPStatus.BAD_REQUEST,
                    "Intrinsic validation generation must appear once",
                )
            try:
                generation = int(generation_values[0])
            except ValueError as error:
                raise ApiError(
                    HTTPStatus.BAD_REQUEST,
                    "Intrinsic validation generation must be a positive integer",
                ) from error
            if generation <= 0:
                raise ApiError(
                    HTTPStatus.BAD_REQUEST,
                    "Intrinsic validation generation must be a positive integer",
                )
        return self._intrinsic().validation_image(token[:-4], generation)

    def _dispatch(self) -> None:
        request_url = urlsplit(self.path)
        path = request_url.path
        if self.command in ("GET", "HEAD"):
            if path == "/healthz":
                payload: Dict[str, Any] = {"status": "ok"}
                if self.calibration_server.service is not None:
                    state = self.calibration_server.service.state()
                    payload["image_ready"] = bool(state["source"].get("image_ready"))
                    payload["intrinsic_ready"] = bool(state["source"].get("intrinsic_ready"))
                    payload["marker_count"] = int(state["source"].get("marker_count", 0))
                if self.calibration_server.intrinsic_service is not None:
                    intrinsic_state = self.calibration_server.intrinsic_service.state()
                    payload.setdefault("image_ready", bool(intrinsic_state.get("image_ready")))
                    payload["camera_control"] = bool(intrinsic_state.get("camera_control"))
                self._send_json(HTTPStatus.OK, payload)
                return
            if path == "/api/v1/state":
                self._send_json(HTTPStatus.OK, self._extrinsic().state())
                return
            if path == "/api/v1/image.jpg":
                self._send_bytes(HTTPStatus.OK, "image/jpeg", self._extrinsic().image_jpeg())
                return
            if path == "/api/v1/intrinsic/state":
                self._send_json(HTTPStatus.OK, self._intrinsic().state())
                return
            if path == "/api/v1/intrinsic/events":
                self._send_intrinsic_state_events()
                return
            if path == "/api/v1/intrinsic/image.jpg":
                self._send_bytes(HTTPStatus.OK, "image/jpeg", self._intrinsic().image_jpeg())
                return
            if path == "/api/v1/intrinsic/targets":
                self._send_json(HTTPStatus.OK, self._intrinsic().targets_document())
                return
            if path == "/api/v1/intrinsic/calibrations":
                self._send_json(HTTPStatus.OK, self._intrinsic().calibration_history())
                return
            if path == "/api/v1/intrinsic/evidence.zip":
                filename, evidence_path = self._intrinsic().evidence_bundle()
                self._send_file(
                    HTTPStatus.OK,
                    "application/zip",
                    evidence_path,
                    filename,
                )
                return
            if path.startswith("/api/v1/intrinsic/validation/image/"):
                self._send_bytes(
                    HTTPStatus.OK,
                    "image/jpeg",
                    self._intrinsic_validation_image(path, request_url.query),
                )
                return
            if path.startswith("/api/v1/intrinsic/ref/"):
                self._send_bytes(HTTPStatus.OK, "image/jpeg", self._intrinsic_ref(path))
                return
            asset = self.static_files.get(path)
            if asset:
                payload = (self.calibration_server.web_root / asset).read_bytes()
                content_type = mimetypes.guess_type(asset)[0] or "application/octet-stream"
                if content_type.startswith("text/") or content_type in (
                    "application/javascript",
                    "application/json",
                ):
                    content_type += "; charset=utf-8"
                self._send_bytes(HTTPStatus.OK, content_type, payload)
                return
            raise ApiError(HTTPStatus.NOT_FOUND, "Route not found")
        if self.command == "POST":
            request = self._request_json()
            if path == "/api/v1/freeze":
                if request not in ({}, None):
                    raise ApiError(HTTPStatus.BAD_REQUEST, "Freeze request must be an empty object")
                self._send_json(HTTPStatus.OK, self._extrinsic().freeze())
                return
            if path == "/api/v1/live":
                if request not in ({}, None):
                    raise ApiError(HTTPStatus.BAD_REQUEST, "Live request must be an empty object")
                self._send_json(HTTPStatus.OK, self._extrinsic().live())
                return
            if path == "/api/v1/solve":
                self._send_json(HTTPStatus.OK, self._extrinsic().solve(request))
                return
            if path == "/api/v1/save":
                if not isinstance(request, dict) or set(request) != {"candidate_id"}:
                    raise ApiError(
                        HTTPStatus.BAD_REQUEST,
                        "Extrinsic save requires only candidate_id",
                    )
                candidate_id = request.get("candidate_id")
                if not isinstance(candidate_id, str) or not candidate_id.strip():
                    raise ApiError(
                        HTTPStatus.BAD_REQUEST,
                        "Extrinsic save candidate_id must be a non-empty string",
                    )
                self._send_json(
                    HTTPStatus.OK,
                    self._extrinsic().save(candidate_id.strip()),
                )
                return
            if path == "/api/v1/intrinsic/candidate":
                if request not in ({}, None):
                    raise ApiError(
                        HTTPStatus.BAD_REQUEST,
                        "Intrinsic candidate request must be an empty object",
                    )
                self._send_json(HTTPStatus.OK, self._intrinsic().calibrate())
                return
            if path == "/api/v1/intrinsic/save":
                if not isinstance(request, dict) or set(request) != {"candidate_id"}:
                    raise ApiError(
                        HTTPStatus.BAD_REQUEST,
                        "Intrinsic save requires only candidate_id",
                    )
                candidate_id = request.get("candidate_id")
                if not isinstance(candidate_id, str) or not candidate_id.strip():
                    raise ApiError(
                        HTTPStatus.BAD_REQUEST,
                        "Intrinsic save candidate_id must be a non-empty string",
                    )
                self._send_json(
                    HTTPStatus.OK,
                    self._intrinsic().save(candidate_id.strip()),
                )
                return
            if path == "/api/v1/intrinsic/continue":
                if request not in ({}, None):
                    raise ApiError(
                        HTTPStatus.BAD_REQUEST,
                        "Intrinsic continue request must be an empty object",
                    )
                self._send_json(HTTPStatus.OK, self._intrinsic().continue_collection())
                return
            if path == "/api/v1/intrinsic/reset":
                if request not in ({}, None):
                    raise ApiError(HTTPStatus.BAD_REQUEST, "Reset request must be an empty object")
                self._send_json(HTTPStatus.OK, self._intrinsic().reset())
                return
            if path == "/api/v1/intrinsic/capture":
                if request not in ({}, None):
                    raise ApiError(HTTPStatus.BAD_REQUEST, "Capture request must be an empty object")
                self._send_json(HTTPStatus.OK, self._intrinsic().capture())
                return
            if path == "/api/v1/intrinsic/validation":
                if not isinstance(request, dict) or set(request) != {"reference", "comparison"}:
                    raise ApiError(
                        HTTPStatus.BAD_REQUEST,
                        "Intrinsic validation requires reference and comparison objects",
                    )
                result = self._intrinsic().validate_intrinsic(
                    request["reference"], request["comparison"]
                )
                self._send_json(HTTPStatus.OK, result)
                return
            if path == "/api/v1/intrinsic/auto_capture/start":
                if request not in ({}, None):
                    raise ApiError(HTTPStatus.BAD_REQUEST, "auto_capture start request must be an empty object")
                intrinsic = self._intrinsic()
                with intrinsic.lock:
                    interval = intrinsic._resume_auto_capture_interval_locked()
                self._send_json(HTTPStatus.OK, intrinsic.start_auto_capture(interval=interval))
                return
            if path == "/api/v1/intrinsic/auto_capture/stop":
                if request not in ({}, None):
                    raise ApiError(HTTPStatus.BAD_REQUEST, "auto_capture stop request must be an empty object")
                self._send_json(HTTPStatus.OK, self._intrinsic().stop_auto_capture())
                return
            if path == "/api/v1/intrinsic/goto":
                index = request.get("index") if isinstance(request, dict) else None
                if not isinstance(index, int) or isinstance(index, bool):
                    raise ApiError(HTTPStatus.BAD_REQUEST, "goto requires an integer 'index'")
                self._send_json(HTTPStatus.OK, self._intrinsic().goto(index))
                return
            if path == "/api/v1/intrinsic/reset_pose":
                if request not in ({}, None):
                    raise ApiError(HTTPStatus.BAD_REQUEST, "reset_pose request must be an empty object")
                self._send_json(HTTPStatus.OK, self._intrinsic().reset_pose())
                return
            if path == "/api/v1/intrinsic/auto_run":
                if request not in ({}, None):
                    raise ApiError(HTTPStatus.BAD_REQUEST, "auto_run request must be an empty object")
                self._send_json(HTTPStatus.ACCEPTED, self._intrinsic().auto_run())
                return
            raise ApiError(HTTPStatus.NOT_FOUND, "Route not found")
        raise ApiError(HTTPStatus.METHOD_NOT_ALLOWED, "Method not allowed")

    def do_GET(self) -> None:
        self._handle()

    def do_HEAD(self) -> None:
        self._handle()

    def do_POST(self) -> None:
        self._handle()

    def do_OPTIONS(self) -> None:
        origin = self._origin()
        self.send_response(HTTPStatus.NO_CONTENT)
        self.send_header("Content-Length", "0")
        if origin:
            self.send_header("Access-Control-Allow-Origin", origin)
            self.send_header("Vary", "Origin")
            self.send_header("Access-Control-Allow-Methods", "GET, HEAD, POST, OPTIONS")
            self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def _handle(self) -> None:
        try:
            self._dispatch()
        except ApiError as error:
            self._send_error(error)
        except (BrokenPipeError, ConnectionResetError):
            return
        except Exception as error:  # pragma: no cover - defensive transport boundary
            self.calibration_server.logger("Unhandled HTTP request failure: {}".format(error))
            self._send_error(
                ApiError(HTTPStatus.INTERNAL_SERVER_ERROR, "Internal server error")
            )
