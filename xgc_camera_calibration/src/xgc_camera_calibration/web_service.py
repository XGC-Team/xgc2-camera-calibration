"""HTTP-independent camera extrinsic calibration service and web transport."""

from __future__ import annotations

import asyncio
import hashlib
import json
import math
import mimetypes
import threading
import uuid
from dataclasses import asdict, dataclass, field, replace
from http import HTTPStatus
from aiohttp import web
from xgc2_xrpc import AppRouter, Endpoint, Fault, Host, Limits, RawStreamResponse, Response, Runtime, ServiceRef, iter_body
from pathlib import Path
from typing import Any, Callable, Dict, Mapping, Optional, Sequence, Tuple

import cv2
import numpy as np

from xgc_camera_calibration.solver import (
    CalibrationError,
    ExtrinsicResult,
    extrinsic_calibration_directory,
    save_extrinsic,
    solve_extrinsic,
    versioned_extrinsic_path,
)


from xgc_camera_calibration.extrinsic_samples import ApiError, SampleCollection, IMAGE_BYTES, IMAGE_PATH
from xgc_camera_calibration.extrinsic_application import SavedExtrinsicApplication, parse_frame_roles
from xgc_camera_calibration.extrinsic_resolver import decode_frozen
from xgc_camera_calibration.extrinsic_selection import SelectionConflict, read_version


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
        calibration_root: str,
        calibration_mode: str,
        camera_name: str,
        parent_frame: str,
        child_frame: str,
        ransac_threshold_px: float = 3.0,
        maximum_inlier_error_px: float = 10.0,
        jpeg_quality: int = 80,
        resolved_extrinsic_json: Optional[str] = None,
        frame_roles_json: Optional[str] = None,
        application_state_reader: Optional[Callable] = None,
    ):
        if not parent_frame or not child_frame:
            raise ValueError("parent_frame and child_frame must not be empty")
        if not 1 <= int(jpeg_quality) <= 100:
            raise ValueError("jpeg_quality must be between 1 and 100")
        self.source = source
        self.frozen_extrinsic = None
        self.frame_roles = None
        self.application = None
        self._application_factory = None
        if resolved_extrinsic_json is not None:
            self.frame_roles = parse_frame_roles(frame_roles_json)
            self.frozen_extrinsic = decode_frozen(resolved_extrinsic_json, camera_name, self.frame_roles)
            if (parent_frame != self.frame_roles["parentFrame"]
                    or child_frame != self.frame_roles["opticalFrames"].get(calibration_mode)):
                raise ValueError("calibrator frames do not match the controlled camera roles")
            if application_state_reader is None:
                raise ValueError("the exact camera publisher state reader is required")
            self._application_factory = lambda: SavedExtrinsicApplication(
                calibration_root, camera_name, resolved_extrinsic_json, self.frame_roles,
                application_state_reader)
            self.application = self._application_factory()
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
        self.candidate_metadata = None
        self.samples = SampleCollection(source, parent_frame, self.lock, self._samples_changed)
        try:
            self._restore_selected_result()
        except Exception as error:
            self.recovery_error = str(error)

    def _samples_changed(self) -> None:
        self.result = None
        self.result_payload = None
        self.candidate_id = None
        self.saved_candidate_id = None
        self.candidate_points = None
        self.candidate_metadata = None
        self.result_restored = False
        self._pending_output_file = None
        self.recovery_error = None
        self.output_file = None
        self.application = self._application_factory() if self._application_factory else None

    def begin_sample(self, request):
        return self.samples.begin(request)

    def commit_sample(self, identity, payload, mime):
        self.samples.commit(identity, payload, mime)
        return self.state()

    def mutate_sample(self, action, request):
        self.samples.mutate(action, request)
        return self.state()

    def _restore_selected_result(self) -> None:
        # Restore the exact frozen version only. A new global application or an
        # old mode-specific pointer cannot retarget this running calibration UI.
        if self.frozen_extrinsic is None or "result" not in self.frozen_extrinsic:
            return
        ref = self.frozen_extrinsic["result"]
        document = read_version(str(self.output_directory.parents[1]), self.camera_name,
                                ref, self.frame_roles)
        output_file = self.output_directory.parents[1] / ref["sourceMode"] / self.camera_name / ref["fileName"]
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
        candidate_id = str(metadata.get("candidate_id", ""))
        if not candidate_id:
            raise CalibrationError("selected result has no calibration candidate identity")
        self.output_file = str(output_file)
        self.candidate_id = candidate_id
        self.saved_candidate_id = candidate_id
        self.candidate_points = list(points)
        self.result_payload = {
            "candidate_id": candidate_id,
            "saved": True,
            "translation": list(self.frozen_extrinsic["resolvedOpticalPose"]["translation"]),
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
            "camera_model": metadata.get("camera_model"),
            "output_file": str(output_file),
            "save_blocked": None,
            "selection_file": str(
                self.output_directory.parents[1]
                / "selections" / self.camera_name
                / "extrinsic.json"
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
            if result is not None and self.application is not None and self.application.request is not None:
                try:
                    result = {**result, "application": self.application.status()}
                except (OSError, ValueError, CalibrationError) as error:
                    result = {**result, "application": {"status": "unavailable", "error": str(error)}}
            payload: Dict[str, Any] = {
                **self.samples.state(),
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
        # Copy mutable source buffers and capture model identity once. Save and
        # subsequent status changes cannot replace evidence of the frozen solve.
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
            persisted_points = [dict(point, inlier=index in inliers,
                reprojection_error_px=float(result.reprojection_errors_px[index]))
                for index, point in enumerate(points)]
            payload = _result_payload(result, [point["sample_id"] for point in points],
                                      np.asarray(world, dtype=np.float64), matrix, distortion)
            by_id = {point["sample_id"]: point for point in points}
            for projection in payload["projections"]:
                identity = projection.pop("marker")
                projection.update(sample_id=identity, marker=by_id[identity]["marker"])
            payload.update(points=persisted_points, camera_model=model, pose_coordinates=coordinates,
                           dataset_revision=self.samples.revision)
            candidate_document = {
                "sampling_session_id": self.samples.session_id,
                "dataset_revision": self.samples.revision,
                "camera_model": model,
                "pose_coordinates": coordinates,
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
            self.candidate_metadata = json.loads(json.dumps({
                "camera_model": model, "pose_coordinates": coordinates,
                "sampling_session_id": self.samples.session_id, "dataset_revision": self.samples.revision,
                "image_topic": self.source.image_topic, "pose_prefix": self.source.pose_prefix,
            }, allow_nan=False))
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
                    if not self.result_restored:
                        self._stage_saved_result(identity)
                    return dict(self.result_payload)
                raise ApiError(HTTPStatus.CONFLICT, "A different extrinsic candidate is already saved")
            if self.result is None or self.result_payload is None or self.candidate_points is None:
                raise ApiError(HTTPStatus.CONFLICT, "No extrinsic candidate is ready")
            if identity != self.candidate_id:
                raise ApiError(
                    HTTPStatus.CONFLICT,
                    "Extrinsic candidate changed; solve the current samples again",
                    details={"expected_candidate_id": self.candidate_id},
                )
            metadata = self.candidate_metadata
            if metadata is None or metadata["dataset_revision"] != self.samples.revision:
                raise ApiError(HTTPStatus.CONFLICT, "Candidate samples are unavailable")
            if self.frozen_extrinsic is not None:
                coordinates = metadata.get("pose_coordinates", {})
                target = self.frozen_extrinsic["targetCoordinates"]
                if (coordinates.get("kind") != "experiment-world"
                        or coordinates.get("frame") != target["frame"]
                        or list(coordinates.get("world_offset", [])) != target["worldOffset"]):
                    raise ApiError(HTTPStatus.CONFLICT, "Sample coordinates do not match the frozen camera context")
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
                            **metadata,
                            "candidate_id": identity,
                            "intrinsic_file": metadata["camera_model"].get("intrinsic_file", ""),
                            "image_width": metadata["camera_model"]["image_width"],
                            "image_height": metadata["camera_model"]["image_height"],
                            "web_calibrator": True,
                        },
                    )
                    self._pending_output_file = (identity, output_file)
            except (OSError, ValueError, CalibrationError) as error:
                raise ApiError(
                    HTTPStatus.INTERNAL_SERVER_ERROR,
                    "Could not save calibration result: {}".format(error),
                ) from error
            self.output_file = str(output_file)
            self.saved_candidate_id = identity
            self.result_payload = {
                **self.result_payload,
                "saved": True,
                "output_file": self.output_file,
                "selection_file": str(self.output_directory.parents[1] / "selections" / self.camera_name / "extrinsic.json") if self.application else None,
                "save_blocked": None,
            }
            self.result_restored = False
            self._pending_output_file = None
            self.recovery_error = None
            self._stage_saved_result(identity)
            return dict(self.result_payload)

    def _stage_saved_result(self, identity):
        if self.application is None:
            # Library-only capture sessions may save immutable results, but
            # cannot claim that any producer has applied them.
            self.result_payload["application"] = {"status": "unavailable"}
            return
        try:
            self.result_payload["application"] = self.application.stage(
                Path(self.output_file), self.calibration_mode, identity)
        except SelectionConflict as error:
            self.result_payload["application"] = {"status": "conflict"}
            raise ApiError(HTTPStatus.CONFLICT, "Camera application was superseded") from error
        except (OSError, ValueError, CalibrationError) as error:
            self.result_payload["application"] = {"status": "unavailable"}
            raise ApiError(HTTPStatus.SERVICE_UNAVAILABLE,
                "Calibration is saved; application is unavailable: {}".format(error)) from error


_RESPONSE_STARTED = web.RequestKey("calibration.response_started", bool)


class CalibrationHttpServer:
    """Public aiohttp adapter owned by one explicit XRPC runtime.

    ROS, locks, codecs, solver and persistence run in the runtime's fixed
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
                 validation_workers=2, preferences=None):
        if service is None and intrinsic_service is None:
            raise ValueError("at least one of service / intrinsic_service is required")
        root = Path(web_root).resolve()
        for required in ("index.html", "app.js", "styles.css"):
            if not (root / required).is_file():
                raise FileNotFoundError("Web asset is missing: {}".format(root / required))
        if "\r" in frame_ancestors or "\n" in frame_ancestors:
            raise ValueError("frame_ancestors must not contain newlines")
        self.service, self.intrinsic_service = service, intrinsic_service
        self.preferences = preferences
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
            private_reference = ServiceRef(target_id, "xgc2.calibration.v1.Calibration", "1",
                uuid.uuid4().hex, "http.v1", Endpoint("unix", str(rpc_socket))).validate()
        self.runtime = runtime or Runtime(blocking_workers=4, max_calls=8,
                                          max_connections=16)
        self._owns_runtime = runtime is None
        self._closed = False
        self._stopping = False
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
                              response_bytes=8 << 20, call_timeout=30.0))
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
        def candidate(value):
            if not isinstance(value, dict) or set(value) != {"candidate_id"} or not isinstance(value["candidate_id"], str) or not value["candidate_id"].strip():
                raise ApiError(400, "Save requires only a non-empty candidate_id")
            return (value["candidate_id"].strip(),)
        if self.service is not None:
            routes[("GET", "/api/v1/state")] = adapt(self.service.state, empty)
            routes[("POST", "/api/v1/solve")] = adapt(self.service.solve, lambda value: (value,))
            routes[("POST", "/api/v1/save")] = adapt(self.service.save, candidate)
        if self.intrinsic_service is not None:
            routes[("GET", "/api/v1/intrinsic/state")] = adapt(self.intrinsic_service.state, empty)
            for path, method, status in (("candidate", "start_candidate", 202), ("save", "save", 200)):
                if hasattr(self.intrinsic_service, method):
                    routes[("POST", "/api/v1/intrinsic/" + path)] = adapt(
                        getattr(self.intrinsic_service, method), candidate if path == "save" else empty, status)
        def describe(context, value):
            if self._stopping:
                raise Fault("unavailable", "Calibration application is draining", 503)
            empty(value)
            return {"service_ref": asdict(self.service_ref),
                "capabilities": {"intrinsic": self.intrinsic_service is not None, "extrinsic": self.service is not None},
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
        # Host retains its resources if domain work has not really quiesced.
        self.host.close()
        if self.private_host is not None:
            self.private_host.close()
        if self._owns_runtime:
            self.runtime.close()
        self._closed = True

    async def _domain(self, function, *args):
        return await self.runtime.blocking(self.host, function, *args)

    async def _preference_domain(self, function, *args):
        from xgc_camera_calibration.preferences import PreferenceError
        try:
            return await self._domain(function, *args)
        except PreferenceError as error:
            raise ApiError(error.status, str(error), details={"code": error.code,
                "outcome": error.outcome, "request_id": error.request_id}) from error

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
        try:
            async with asyncio.timeout(timeout):
                async for chunk in iter_body(request, max_bytes=maximum):
                    body.extend(chunk)
        except web.HTTPRequestEntityTooLarge as error:
            raise ApiError(413, "Request body exceeds the byte limit") from error
        except TimeoutError as error:
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

    async def _events(self, request):
        intrinsic = self.intrinsic_service
        response = RawStreamResponse(max_bytes=self.limits.response_bytes, headers={**self._headers(request),
                                               "Content-Type": "text/event-stream"})
        await response.prepare(request)
        request[_RESPONSE_STARTED] = True
        if request.method == "HEAD":
            await response.write_eof()
            return response
        previous, event_id = b"", 0
        while True:
            state = await self._domain(intrinsic.state) if intrinsic is not None else {}
            if self.preferences is not None:
                state["preferences"] = await self._domain(self.preferences.event_state)
            else:
                state["preferences"] = {"available": False, "snapshot": None,
                    "error": {"code": "unavailable", "message": "Preference storage was not granted"}}
            payload = json.dumps(state, separators=(",", ":"), allow_nan=False).encode()
            if payload != previous:
                event_id += 1
                await response.write("id: {}\nevent: state\ndata: ".format(event_id).encode()
                                     + payload + b"\n\n")
                previous = payload
            await asyncio.sleep(0.1)

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
        if path == "/api/v1/preferences":
            if self.preferences is None:
                raise ApiError(503, "Preference storage was not granted")
            if request.method in ("GET", "HEAD"):
                if set(request.query) - {"request_id"} or len(request.query.getall("request_id", [])) > 1:
                    raise ApiError(400, "Unknown preference query")
                identity = request.query.get("request_id")
                value = (await self._preference_domain(self.preferences.resolve, identity) if identity is not None
                         else await self._preference_domain(self.preferences.get))
                return self._json(request, value)
            if request.method == "PUT":
                if request.query:
                    raise ApiError(400, "Preference writes do not accept query arguments")
                return self._json(request, await self._preference_domain(self.preferences.put, await self._request_json(request)))
            raise ApiError(405, "Preference method not allowed")
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
                return self._json(request, await self._domain(self._extrinsic().state))
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
                payload = await self._domain(getattr(self._intrinsic(), method))
                return self._json(request, payload) if mime == "json" else self._bytes(request, payload, mime)
            if path == "/api/v1/intrinsic/evidence.zip":
                filename, evidence_path = await self._domain(self._intrinsic().evidence_bundle)
                return await self._file(request, evidence_path, filename)
            if path.startswith("/api/v1/intrinsic/validation/image/"):
                token = path[len("/api/v1/intrinsic/validation/image/"):]
                if not token.endswith(".jpg"):
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
                payload = await self._domain(self._intrinsic().validation_image, token[:-4], generation)
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
        if path in ("/api/v1/save", "/api/v1/intrinsic/save", "/api/v1/intrinsic/apply-metadata"):
            if not isinstance(value, dict) or set(value) != {"candidate_id"}:
                raise ApiError(400, "Save/apply requires only candidate_id")
            identity = value["candidate_id"]
            if not isinstance(identity, str) or not identity.strip():
                raise ApiError(400, "Save/apply candidate_id must be a non-empty string")
            service = self._extrinsic() if path == "/api/v1/save" else self._intrinsic()
            action = service.apply_metadata if path.endswith("apply-metadata") else service.save
            return self._json(request, await self._domain(action, identity.strip()))
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
