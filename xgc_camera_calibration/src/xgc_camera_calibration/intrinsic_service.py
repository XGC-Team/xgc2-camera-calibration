"""HTTP-independent intrinsic calibration session, driven frame by frame.

Camera-agnostic: it consumes decoded BGR frames (``process_frame``) from any
source -- a real camera or a simulated one -- and never moves the camera.  It
mirrors the ROS ``camera_calibration`` operator loop (auto-collect geometrically
diverse chessboard views, report X/Y/Size/Skew coverage, then calibrate) using
the cv2-direct, cv_bridge-free ``intrinsic_solver``.

It also owns a **sample guide**: a catalogue of recommended sample viewpoints
(the 3D guide's spheres) plus their pre-recorded reference images.  The guide is
pure visual guidance for any camera.  When an optional camera-control adapter is
attached (only meaningful in simulation), the guide additionally greens each
viewpoint as the camera aligns to it, exposes the live pose, and can fly the
camera through the catalogue (goto / auto-run / reset).
"""

from __future__ import annotations

import json
import math
import os
import re
import tempfile
import threading
import time
from datetime import datetime, timezone
from http import HTTPStatus
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from xgc_camera_calibration import intrinsic_solver, intrinsic_validation
from xgc_camera_calibration.solver import CalibrationError
from xgc_camera_calibration.web_service import ApiError

APRILGRID_ADAPTIVE_DETECTION_WIDTH = 2200
CAMERA_NAME_PATTERN = re.compile(r"^[A-Za-z][A-Za-z0-9._-]{0,63}$")


def intrinsic_calibration_directory(root: str, mode: str, camera_name: str) -> Path:
    calibration_root = Path(str(root)).expanduser()
    calibration_mode = str(mode).strip()
    identity = str(camera_name).strip()
    if not calibration_root.is_absolute():
        raise ValueError("calibration root must be absolute")
    if calibration_mode not in ("sim", "phy"):
        raise ValueError("calibration mode must be sim or phy")
    if not CAMERA_NAME_PATTERN.fullmatch(identity):
        raise ValueError("camera name must be a stable identifier")
    return calibration_root / calibration_mode / identity


def recommended_views(
    board_center: Sequence[float], board_extent: float = 1.6
) -> List[Dict[str, Any]]:
    """Spatially-distinct sample poses that together fill X / Y / Size / Skew.

    Filling X and Y needs the board off-centre in the image, so those poses carry
    a yaw/pitch aim offset (the camera adapter applies it through
    look_at_orientation) -- a camera that simply aims at the board keeps it
    centred and never moves the X/Y bars.  Size needs near and far views; Skew
    needs the oblique corners.  Each pose sits at its own point so it is a
    distinct, clickable marker in the 3D guide.
    """
    tx, ty, tz = float(board_center[0]), float(board_center[1]), float(board_center[2])
    # Scale translations with the actual target extent so every board keeps the
    # same projected tag size. X/Y coverage comes from bounded aim offsets while
    # the camera stays near the plate, rather than from distant viewpoints.
    extent = float(board_extent)
    if extent <= 0.0:
        raise ValueError("board extent must be positive")
    view_scale = extent / 1.6

    def position(dx: float, dy: float, dz: float) -> Tuple[float, float, float]:
        return (
            tx + dx * view_scale,
            ty + dy * view_scale,
            tz + dz * view_scale,
        )
    # Tuned for the shared 3840x2160, 110-degree camera profile. Keep the whole
    # plate visible while moving it far enough toward each edge to satisfy the
    # ROS camera_calibration X/Y coverage ranges.
    specs = [
        ("left edge", position(-2.91, 0.05, 0.00), -0.58, 0.00, 0.08),
        ("right edge", position(-2.91, -0.05, 0.00), 0.58, 0.00, -0.08),
        ("lower edge", position(-2.42, 0.00, 0.00), 0.00, -0.30, 0.00),
        ("upper edge", position(-2.42, 0.03, 0.00), 0.00, 0.30, 0.00),
        ("left edge tilted", position(-2.20, 0.15, -0.10), -0.24, 0.00, 0.18),
        ("right edge tilted", position(-2.20, -0.15, -0.08), 0.24, 0.00, -0.18),
        ("lower edge tilted", position(-2.42, 0.05, 0.02), 0.00, -0.06, 0.10),
        ("upper edge tilted", position(-2.42, -0.05, -0.02), 0.00, 0.06, -0.10),
        ("center face", position(-2.30, 0.00, 0.00), 0.00, 0.00, 0.00),
        ("near large", position(-2.10, 0.00, 0.00), 0.00, 0.00, 0.00),
        ("near maximum", position(-1.30, 0.00, 0.00), 0.00, 0.00, 0.00),
        ("clockwise skew", position(-2.20, 0.04, 0.04), 0.00, 0.00, 0.46),
        ("counter-clockwise skew", position(-2.20, -0.04, -0.04), 0.00, 0.00, -0.46),
        ("oblique high", position(-2.20, 0.35, 0.35), 0.00, 0.00, 0.28),
        ("oblique low", position(-2.20, -0.35, -0.35), 0.00, 0.00, -0.28),
    ]
    return [{
        "name": name,
        "position": [round(value, 2) for value in position],
        "yaw_offset": yaw_offset,
        "pitch_offset": pitch_offset,
        "roll": roll,
    } for (name, position, yaw_offset, pitch_offset, roll) in specs]


class IntrinsicCalibrationService:
    """Own one operator intrinsic-calibration session over a stream of frames."""

    def __init__(
        self,
        *,
        board_size: Sequence[int],
        square: float,
        output_file: str,
        camera_name: str,
        jpeg_quality: int = 80,
        sample_distance: float = intrinsic_solver.SAMPLE_DISTANCE,
        maximum_detect_width: int = 960,
        display_width: int = 960,
        board_center: Sequence[float] = (2.0, 0.0, 2.2),
        references_dir: str = "",
        align_threshold: float = 1.8,
        media_source: str = "",
        board_type: str = "checkerboard",
        tag_spacing: float = 0.0,
        tag_family: str = "tag36h11",
        tag_start_id: int = 0,
        min_tags: int = 6,
    ):
        if not output_file:
            raise ValueError("output_file must not be empty")
        if not CAMERA_NAME_PATTERN.fullmatch(str(camera_name).strip()):
            raise ValueError("camera_name must be a stable identifier")
        if int(board_size[0]) < 2 or int(board_size[1]) < 2:
            raise ValueError("board_size must be at least 2x2")
        if float(square) <= 0.0:
            raise ValueError("square size must be positive")
        if not 1 <= int(jpeg_quality) <= 100:
            raise ValueError("jpeg_quality must be between 1 and 100")
        kind = str(board_type or "checkerboard").strip().lower()
        if kind not in ("checkerboard", "chessboard", "aprilgrid"):
            raise ValueError("unsupported calibration board type: {}".format(board_type))
        if kind == "aprilgrid" and float(tag_spacing) < 0.0:
            raise ValueError("AprilGrid tag spacing must be non-negative")
        self.board_type = "aprilgrid" if kind == "aprilgrid" else "checkerboard"
        self.tag_spacing = float(tag_spacing)
        self.tag_family = str(tag_family or "tag36h11").strip().lower()
        self.tag_start_id = int(tag_start_id)
        self.min_tags = int(min_tags)
        self.board_size = (int(board_size[0]), int(board_size[1]))
        self.square = float(square)
        self.output_file_base = str(Path(output_file).expanduser())
        self.camera_name = str(camera_name).strip()
        self.output_file = self.output_file_base
        self.checkpoint_file = self.output_file_base + ".session.npz"
        self.media_source = str(media_source).strip()
        self.jpeg_quality = int(jpeg_quality)
        self.sample_distance = float(sample_distance)
        self.maximum_detect_width = int(maximum_detect_width)
        self.display_width = int(display_width)
        self.lock = threading.RLock()
        self._detection_condition = threading.Condition(self.lock)
        self._capture_lock = threading.Lock()
        self.samples: List[Tuple[float, float, float, float]] = []
        self.image_points: List[np.ndarray] = []
        self.object_points: List[np.ndarray] = []
        self.image_size: Optional[Tuple[int, int]] = None
        self._display: Optional[np.ndarray] = None
        self.result: Optional[intrinsic_solver.IntrinsicResult] = None
        self.result_payload: Optional[Dict[str, Any]] = None
        self.result_restored = False
        self.restored_coverage: List[Dict[str, Any]] = []
        self._recovery_error: Optional[str] = None
        self._frame_sequence = 0
        expected_corners = self.board_size[0] * self.board_size[1]
        if self.board_type == "aprilgrid":
            expected_corners *= 4
        self.latest_detection: Dict[str, Any] = {
            "status": "waiting",
            "corner_count": 0,
            "expected_corner_count": expected_corners,
            "frame_width": 0,
            "frame_height": 0,
            "sequence": 0,
            "metrics": [],
            "accepted": False,
            "duplicate": False,
        }

        # Sample guide: chessboard uses interior corners + 1 squares; AprilGrid
        # uses the printed tag grid including the gaps between tags.
        self.board_center = tuple(float(value) for value in board_center)
        if self.board_type == "aprilgrid":
            board_width = self.board_size[0] * self.square + (self.board_size[0] - 1) * self.tag_spacing
            board_height = self.board_size[1] * self.square + (self.board_size[1] - 1) * self.tag_spacing
        else:
            board_width = (self.board_size[0] + 1) * self.square
            board_height = (self.board_size[1] + 1) * self.square
        self.board_geometry = {
            "center": list(self.board_center),
            "width": board_width,
            "height": board_height,
        }
        self.views: List[Dict[str, Any]] = recommended_views(
            self.board_center, max(board_width, board_height)
        )
        self.target_done: List[bool] = [False] * len(self.views)
        self.references_dir = str(Path(references_dir).expanduser()) if references_dir else ""
        self.refs: Dict[int, bytes] = {}
        self._validation_generation = 0
        self._validation_report: Optional[Dict[str, Any]] = None
        self._validation_images: Dict[str, bytes] = {}
        self.align_threshold = float(align_threshold)
        self.camera: Optional[Any] = None
        self.frame_capture: Optional[Callable[[], np.ndarray]] = None
        self._recording = False
        self.action: Optional[Dict[str, Any]] = None
        self._selected_target_index: Optional[int] = None
        self._auto_run_thread: Optional[threading.Thread] = None
        self._auto_capture_thread: Optional[threading.Thread] = None
        self._auto_capture_stop = threading.Event()
        self._auto_capture_interval = 0.0
        self._auto_capture_error: Optional[str] = None
        self._auto_capture_completed = False
        self._load_refs()
        self._load_recovery()

    # -- guide wiring ---------------------------------------------------------
    def attach_camera_control(self, camera: Any) -> None:
        """Attach an optional sim camera adapter (goto/reset/current pose)."""
        with self.lock:
            self.camera = camera

    def attach_frame_capture(self, capture: Callable[[], np.ndarray]) -> None:
        """Attach an immutable Media Edge snapshot transaction."""
        with self.lock:
            self.frame_capture = capture

    def _auto_capture_document_locked(self) -> Dict[str, Any]:
        thread = self._auto_capture_thread
        enabled = bool(
            thread is not None
            and thread.is_alive()
            and not self._auto_capture_stop.is_set()
        )
        return {
            "enabled": enabled,
            "interval_seconds": self._auto_capture_interval,
            "last_error": self._auto_capture_error,
            "coverage_complete": self._auto_capture_completed,
        }

    def start_auto_capture(self, interval: float = 0.0) -> Dict[str, Any]:
        """Continuously inspect snapshots from either camera origin.

        Live playback remains WebRTC. This lower-rate loop is the one shared
        simulation/physical detection path: ``process_frame`` replaces the one
        in-memory annotated preview and stores only corner coordinates for
        accepted samples. Reaching full coverage stops sample growth, not live
        detection; the result surface must continue to follow the moving camera.
        """
        interval = float(interval)
        if not 0.0 <= interval <= 10.0:
            raise ApiError(
                HTTPStatus.UNPROCESSABLE_ENTITY,
                "detection interval must be between 0 and 10 seconds",
            )
        with self.lock:
            if self.frame_capture is None:
                raise ApiError(
                    HTTPStatus.SERVICE_UNAVAILABLE,
                    "No calibration frame source is available",
                )
            current = self._auto_capture_thread
            if current is not None and current.is_alive():
                return {"ok": True, "auto_capture": self._auto_capture_document_locked()}
            self._auto_capture_interval = interval
            self._auto_capture_error = None
            self._auto_capture_completed = False
            self._auto_capture_stop.clear()
            thread = threading.Thread(
                target=self._run_auto_capture,
                name="intrinsic-live-detection",
                daemon=True,
            )
            self._auto_capture_thread = thread
        try:
            thread.start()
        except RuntimeError as error:
            with self.lock:
                self._auto_capture_thread = None
                self._auto_capture_error = str(error) or "Could not start auto capture"
            raise ApiError(
                HTTPStatus.INTERNAL_SERVER_ERROR,
                "Could not start continuous intrinsic detection",
            ) from error
        with self.lock:
            return {"ok": True, "auto_capture": self._auto_capture_document_locked()}

    def _run_auto_capture(self) -> None:
        next_capture_at = time.monotonic()
        try:
            while not self._auto_capture_stop.is_set():
                failed = False
                try:
                    self._capture_frame()
                except Exception as error:
                    failed = True
                    with self.lock:
                        self._auto_capture_error = str(error) or error.__class__.__name__
                else:
                    with self.lock:
                        self._auto_capture_error = None
                        guidance = intrinsic_solver.next_view_guidance(self.samples)
                        self._auto_capture_completed = bool(guidance["complete"])
                if self._auto_capture_interval <= 0.0:
                    # The source transaction (fresh camera frame) and detector
                    # provide natural backpressure. Only failures get a small
                    # retry backoff so a disconnected source cannot spin.
                    if failed and self._auto_capture_stop.wait(0.1):
                        break
                    continue
                next_capture_at += self._auto_capture_interval
                wait = next_capture_at - time.monotonic()
                if wait <= 0.0:
                    # Never build a frame backlog. A slow detector immediately
                    # consumes the newest snapshot and starts a fresh cadence.
                    next_capture_at = time.monotonic()
                    continue
                if self._auto_capture_stop.wait(wait):
                    break
        finally:
            with self.lock:
                if self._auto_capture_thread is threading.current_thread():
                    self._auto_capture_thread = None

    def stop_auto_capture(self) -> Dict[str, Any]:
        with self.lock:
            thread = self._auto_capture_thread
            self._auto_capture_stop.set()
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=6.0)
        with self.lock:
            if self._auto_capture_thread is not None and not self._auto_capture_thread.is_alive():
                self._auto_capture_thread = None
            return {"ok": True, "auto_capture": self._auto_capture_document_locked()}

    def _load_refs(self) -> None:
        if not self.references_dir:
            return
        for index in range(len(self.views)):
            path = os.path.join(self.references_dir, "{}.jpg".format(index))
            if os.path.isfile(path):
                try:
                    with open(path, "rb") as handle:
                        self.refs[index] = handle.read()
                except OSError:
                    pass

    def _recovery_fingerprint(self) -> Dict[str, Any]:
        return {
            "schema": 1,
            "board_type": self.board_type,
            "board_size": list(self.board_size),
            "square": self.square,
            "tag_spacing": self.tag_spacing,
            "tag_family": self.tag_family,
            "tag_start_id": self.tag_start_id,
            "media_source": self.media_source,
            "camera_name": self.camera_name,
        }

    def _saved_board_matches(self, document: Dict[str, Any]) -> bool:
        board = document.get("board")
        if not isinstance(board, dict):
            return False
        try:
            size = tuple(int(value) for value in board.get("size", ()))
            square = float(board.get("square_size_m"))
        except (TypeError, ValueError):
            return False
        if size != self.board_size or abs(square - self.square) > 1e-9:
            return False
        if document.get("camera_name") != self.camera_name:
            return False
        saved_type = str(board.get("type", "checkerboard")).strip().lower()
        if saved_type != self.board_type:
            return False
        if self.board_type != "aprilgrid":
            return True
        try:
            spacing = float(board.get("tag_spacing_m"))
            start_id = int(board.get("start_id"))
        except (TypeError, ValueError):
            return False
        return (
            abs(spacing - self.tag_spacing) <= 1e-9
            and str(board.get("tag_family", "")).strip().lower() == self.tag_family
            and start_id == self.tag_start_id
        )

    def _result_document(self, result: intrinsic_solver.IntrinsicResult) -> Dict[str, Any]:
        matrix = result.camera_matrix
        return {
            "camera_matrix": [float(value) for value in matrix.reshape(-1)],
            "distortion": [float(value) for value in result.distortion],
            "fx": float(matrix[0, 0]),
            "fy": float(matrix[1, 1]),
            "cx": float(matrix[0, 2]),
            "cy": float(matrix[1, 2]),
            "image_width": result.image_size[0],
            "image_height": result.image_size[1],
            "rms_reprojection_error_px": result.rms_reprojection_error_px,
            "sample_count": result.sample_count,
            "output_file": self.output_file,
        }

    def _versioned_output_path(self) -> Path:
        base = Path(self.output_file_base)
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
        candidate = base.with_name("{}-{}{}".format(base.stem, timestamp, base.suffix))
        sequence = 1
        while candidate.exists():
            candidate = base.with_name(
                "{}-{}-{:02d}{}".format(base.stem, timestamp, sequence, base.suffix)
            )
            sequence += 1
        return candidate

    def _saved_result_candidates(self) -> List[Path]:
        base = Path(self.output_file_base)
        if not base.parent.is_dir():
            return []
        return [
            path for path in base.parent.glob("{}-*{}".format(base.stem, base.suffix))
            if path.is_file()
        ]

    def calibration_history(self) -> Dict[str, Any]:
        items = []
        for path in self._saved_result_candidates():
            try:
                document = intrinsic_solver.load_intrinsic(path)
                created_time = self._saved_result_time(document, path)
                intrinsic_validation.intrinsic_parameters(document)
            except (OSError, ValueError, CalibrationError):
                continue
            if document.get("camera_name") != self.camera_name:
                continue
            items.append({
                "id": path.name,
                "created_at": str(document.get("created_at", "")),
                "created_time": created_time,
                "image_width": int(document.get("image_width", 0)),
                "image_height": int(document.get("image_height", 0)),
                "rms_reprojection_error_px": float(
                    document.get("rms_reprojection_error_px", 0.0)
                ),
                "sample_count": int(document.get("sample_count", 0)),
            })
        items.sort(key=lambda item: (item["created_time"], item["id"]), reverse=True)
        for index, item in enumerate(items):
            item["latest"] = index == 0
            del item["created_time"]
        return {
            "items": items,
            "selected": items[0]["id"] if items else None,
        }

    def _calibration_document(self, calibration_id: str) -> Tuple[Path, Dict[str, Any]]:
        if not isinstance(calibration_id, str) or not calibration_id or Path(calibration_id).name != calibration_id:
            raise ApiError(HTTPStatus.BAD_REQUEST, "calibration_id must be a result filename")
        candidates = {path.name: path for path in self._saved_result_candidates()}
        path = candidates.get(calibration_id)
        if path is None:
            raise ApiError(HTTPStatus.NOT_FOUND, "Selected intrinsic calibration is unavailable")
        try:
            document = intrinsic_solver.load_intrinsic(path)
            intrinsic_validation.intrinsic_parameters(document)
        except (OSError, ValueError, CalibrationError) as error:
            raise ApiError(
                HTTPStatus.UNPROCESSABLE_ENTITY,
                "Selected intrinsic calibration is invalid: {}".format(error),
            ) from error
        if document.get("camera_name") != self.camera_name:
            raise ApiError(
                HTTPStatus.UNPROCESSABLE_ENTITY,
                "Selected intrinsic calibration belongs to another camera",
            )
        return path, document

    def _capture_validation_frame(self) -> np.ndarray:
        with self._capture_lock:
            with self.lock:
                capture = self.frame_capture
            if capture is None:
                raise ApiError(
                    HTTPStatus.SERVICE_UNAVAILABLE,
                    "No calibration frame source is available",
                )
            try:
                frame = capture()
            except ApiError:
                raise
            except Exception as error:
                raise ApiError(
                    HTTPStatus.SERVICE_UNAVAILABLE,
                    "Could not capture an intrinsic validation frame: {}".format(error),
                ) from error
            jpeg = getattr(frame, "jpeg", None)
            if isinstance(jpeg, bytes):
                image = cv2.imdecode(np.frombuffer(jpeg, dtype=np.uint8), cv2.IMREAD_COLOR)
            elif isinstance(frame, np.ndarray):
                image = frame.copy()
            else:
                source = getattr(frame, "bgr", None)
                image = source.copy() if isinstance(source, np.ndarray) else None
            if not isinstance(image, np.ndarray) or image.ndim != 3 or image.shape[2] != 3:
                raise ApiError(
                    HTTPStatus.SERVICE_UNAVAILABLE,
                    "Calibration frame source returned no validation image",
                )
            return image

    def validate_intrinsic(self, calibration_id: str) -> Dict[str, Any]:
        path, document = self._calibration_document(calibration_id)
        image = self._capture_validation_frame()
        try:
            validation = intrinsic_validation.generate_intrinsic_validation(
                image,
                document,
                calibration_id=path.name,
                jpeg_quality=self.jpeg_quality,
            )
        except (ValueError, CalibrationError, cv2.error) as error:
            raise ApiError(HTTPStatus.UNPROCESSABLE_ENTITY, str(error)) from error
        with self.lock:
            self._validation_generation += 1
            self._validation_images = dict(validation.images)
            self._validation_report = {
                **validation.report,
                "generation": self._validation_generation,
            }
            return dict(self._validation_report)

    def validation_image(self, view_id: str) -> bytes:
        with self.lock:
            image = self._validation_images.get(view_id)
        if image is None:
            raise ApiError(HTTPStatus.NOT_FOUND, "Intrinsic validation image is unavailable")
        return image

    @staticmethod
    def _saved_result_time(document: Dict[str, Any], path: Path) -> float:
        raw = document.get("created_at")
        if isinstance(raw, str) and raw.strip():
            try:
                parsed = datetime.fromisoformat(raw.strip().replace("Z", "+00:00"))
                if parsed.tzinfo is None:
                    parsed = parsed.replace(tzinfo=timezone.utc)
                return parsed.timestamp()
            except ValueError:
                pass
        return path.stat().st_mtime

    def _load_saved_result(self) -> bool:
        saved = []
        load_error: Optional[Exception] = None
        for path in self._saved_result_candidates():
            try:
                document = intrinsic_solver.load_intrinsic(path)
            except (OSError, CalibrationError) as error:
                load_error = error
                continue
            if self._saved_board_matches(document):
                saved.append((self._saved_result_time(document, path), path.name, path, document))
        if not saved:
            if load_error is not None:
                raise load_error
            return False
        _created_at, _name, path, document = max(saved, key=lambda item: (item[0], item[1]))
        self.output_file = str(path)
        distortion = document.get("distortion_coefficients", {})
        distortion_values = distortion.get("data") if isinstance(distortion, dict) else None
        image_size = (int(document["image_width"]), int(document["image_height"]))
        if (
            not isinstance(distortion_values, list)
            or not distortion_values
            or image_size[0] <= 0
            or image_size[1] <= 0
        ):
            raise CalibrationError("saved intrinsic result is incomplete")
        result = intrinsic_solver.IntrinsicResult(
            camera_matrix=document["camera_matrix_array"],
            distortion=np.asarray(distortion_values, dtype=np.float64),
            image_size=image_size,
            rms_reprojection_error_px=float(document["rms_reprojection_error_px"]),
            sample_count=int(document["sample_count"]),
        )
        self.result = result
        self.result_payload = self._result_document(result)
        self.result_restored = True
        self.image_size = image_size
        metadata = document.get("metadata")
        coverage = metadata.get("coverage") if isinstance(metadata, dict) else None
        if isinstance(coverage, list):
            self.restored_coverage = [
                {"label": str(item["label"]), "progress": float(item["progress"])}
                for item in coverage
                if isinstance(item, dict) and "label" in item and "progress" in item
            ]
        return True

    def _load_checkpoint(self) -> bool:
        path = Path(self.checkpoint_file)
        if not path.is_file():
            return False
        with np.load(str(path), allow_pickle=False) as archive:
            fingerprint = json.loads(str(archive["fingerprint"].item()))
            if fingerprint != self._recovery_fingerprint():
                return False
            samples = np.asarray(archive["samples"], dtype=np.float64)
            image_size_values = np.asarray(archive["image_size"], dtype=np.int64).reshape(-1)
            if samples.ndim != 2 or samples.shape[1] != 4 or len(image_size_values) != 2:
                raise CalibrationError("calibration checkpoint shape is invalid")
            image_points: List[np.ndarray] = []
            object_points: List[np.ndarray] = []
            for index in range(len(samples)):
                image = np.asarray(archive["image_points_{:03d}".format(index)], dtype=np.float32)
                objects = np.asarray(archive["object_points_{:03d}".format(index)], dtype=np.float32)
                if image.reshape(-1, 2).shape[0] != objects.reshape(-1, 3).shape[0]:
                    raise CalibrationError("calibration checkpoint correspondences do not match")
                image_points.append(image.reshape(-1, 1, 2))
                object_points.append(objects.reshape(-1, 3))
        self.samples = [tuple(float(value) for value in row) for row in samples]
        self.image_points = image_points
        self.object_points = object_points
        self.image_size = (int(image_size_values[0]), int(image_size_values[1]))
        return bool(self.samples)

    def _load_recovery(self) -> None:
        try:
            if self._load_saved_result():
                return
            self._load_checkpoint()
        # A damaged or stale recovery artifact must never prevent the camera
        # service from starting a fresh stage; expose the problem through state.
        except Exception as error:
            self._recovery_error = str(error) or error.__class__.__name__

    def _save_checkpoint_locked(self) -> None:
        if not self.samples or self.image_size is None or self.result is not None:
            return
        destination = Path(self.checkpoint_file)
        destination.parent.mkdir(parents=True, exist_ok=True)
        payload: Dict[str, Any] = {
            "fingerprint": np.asarray(json.dumps(self._recovery_fingerprint(), sort_keys=True)),
            "samples": np.asarray(self.samples, dtype=np.float64),
            "image_size": np.asarray(self.image_size, dtype=np.int64),
        }
        for index, (image, objects) in enumerate(zip(self.image_points, self.object_points)):
            payload["image_points_{:03d}".format(index)] = np.asarray(image, dtype=np.float32)
            payload["object_points_{:03d}".format(index)] = np.asarray(objects, dtype=np.float32)
        descriptor, temporary_name = tempfile.mkstemp(
            prefix="." + destination.name + ".", suffix=".tmp", dir=str(destination.parent)
        )
        try:
            with os.fdopen(descriptor, "wb") as stream:
                np.savez_compressed(stream, **payload)
                stream.flush()
                os.fsync(stream.fileno())
            os.chmod(temporary_name, 0o644)
            os.replace(temporary_name, str(destination))
            self._recovery_error = None
        except Exception:
            try:
                os.unlink(temporary_name)
            except FileNotFoundError:
                pass
            raise

    def _remove_checkpoint_locked(self) -> None:
        try:
            Path(self.checkpoint_file).unlink()
        except FileNotFoundError:
            pass
        except OSError as error:
            self._recovery_error = str(error) or error.__class__.__name__

    def _save_ref(self, index: int, jpeg: bytes) -> None:
        self.refs[index] = jpeg
        if not self.references_dir:
            return
        try:
            os.makedirs(self.references_dir, exist_ok=True)
            with open(os.path.join(self.references_dir, "{}.jpg".format(index)), "wb") as handle:
                handle.write(jpeg)
        except OSError:
            pass

    def ref(self, index: int) -> Optional[bytes]:
        with self.lock:
            return self.refs.get(index)

    def _nearest_target(self, position: Sequence[float]) -> Tuple[Optional[int], float]:
        best_index, best_distance = None, float("inf")
        for index, view in enumerate(self.views):
            target = view["position"]
            distance = (
                (position[0] - target[0]) ** 2
                + (position[1] - target[1]) ** 2
                + (position[2] - target[2]) ** 2
            ) ** 0.5
            if distance < best_distance:
                best_distance, best_index = distance, index
        return best_index, best_distance

    def _mark_aligned(
        self,
        display: np.ndarray,
        render_position: Optional[Sequence[float]] = None,
        render_orientation: Optional[Sequence[float]] = None,
    ) -> None:
        """Green the nearest target once the camera aligns to it and the board is
        visible this frame -- independent of whether this frame became a *new*
        sample (is_new_sample de-duplicates similar views, so a
        redundant-but-valid pose would otherwise stay grey).  Requires the sim
        camera adapter for the live pose; a no-op for a real camera.
        """
        if self._recording or self.camera is None:
            return
        if not self._render_pose_matches_camera(render_position, render_orientation):
            return
        position = tuple(render_position) if render_position is not None else self.camera.current_position()
        if position is None:
            return
        index = None
        if self.action is not None and self.action.get("status") == "running":
            candidate = self.action.get("target_index")
            if isinstance(candidate, int) and 0 <= candidate < len(self.views):
                index = candidate
        if index is None and self._selected_target_index is not None:
            if 0 <= self._selected_target_index < len(self.views):
                index = self._selected_target_index
        if index is None:
            index, _ = self._nearest_target(position)
        if index is None or self.target_done[index]:
            return
        target = self.views[index]["position"]
        distance = sum((position[axis] - target[axis]) ** 2 for axis in range(3)) ** 0.5
        if distance > self.align_threshold:
            return
        self.target_done[index] = True
        ok, encoded = cv2.imencode(
            ".jpg", display, [int(cv2.IMWRITE_JPEG_QUALITY), 75]
        )
        if ok:
            self._save_ref(index, encoded.tobytes())

    def _render_pose_matches_camera(
        self,
        render_position: Optional[Sequence[float]],
        render_orientation: Optional[Sequence[float]],
    ) -> bool:
        if render_position is None or render_orientation is None or self.camera is None:
            return True
        current_optical_pose = getattr(self.camera, "current_optical_pose", None)
        if not callable(current_optical_pose):
            return True
        current = current_optical_pose()
        if current is None:
            return False
        position_error = sum(
            (float(render_position[index]) - float(current["position"][index])) ** 2
            for index in range(3)
        ) ** 0.5
        dot = abs(sum(
            float(render_orientation[index]) * float(current["orientation"][index])
            for index in range(4)
        ))
        angle_error = 2.0 * math.acos(min(1.0, max(0.0, dot)))
        return position_error <= 0.12 and angle_error <= 0.04

    def _encode_jpeg(self, image: np.ndarray) -> bytes:
        ok, encoded = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), self.jpeg_quality])
        if not ok:
            raise ApiError(HTTPStatus.INTERNAL_SERVER_ERROR, "Could not encode camera frame")
        return encoded.tobytes()

    def process_frame(
        self,
        bgr: np.ndarray,
        render_position: Optional[Sequence[float]] = None,
        render_orientation: Optional[Sequence[float]] = None,
        source_image_size: Optional[Sequence[int]] = None,
        source_jpeg: Optional[bytes] = None,
    ) -> None:
        """Ingest one decoded BGR frame: detect the board, auto-collect, annotate."""
        if bgr.ndim != 3 or bgr.shape[2] != 3:
            return
        height, width = bgr.shape[:2]
        source_width = int(source_image_size[0]) if source_image_size is not None else width
        source_height = int(source_image_size[1]) if source_image_size is not None else height
        if source_width < width or source_height < height:
            return
        source_scale = np.asarray(
            [float(source_width) / float(width), float(source_height) / float(height)],
            dtype=np.float32,
        )
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        detection = intrinsic_solver.detect_board(
            gray,
            self.board_size,
            width if self.board_type == "aprilgrid" else self.maximum_detect_width,
            board_type=self.board_type,
            square=self.square,
            tag_spacing=self.tag_spacing,
            tag_family=self.tag_family,
            start_id=self.tag_start_id,
            min_tags=self.min_tags,
        )
        if (
            detection is None
            and self.board_type == "aprilgrid"
            and source_jpeg
            and source_width > width
            and intrinsic_solver.aprilgrid_has_candidate_evidence(
                gray, self.tag_family, self.min_tags
            )
        ):
            adaptive = self._decode_adaptive_aprilgrid_frame(source_jpeg, source_width)
            if adaptive is not None and adaptive.shape[1] > width:
                bgr = adaptive
                height, width = bgr.shape[:2]
                source_scale = np.asarray(
                    [float(source_width) / float(width), float(source_height) / float(height)],
                    dtype=np.float32,
                )
                gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
                detection = intrinsic_solver.detect_board(
                    gray,
                    self.board_size,
                    width,
                    board_type=self.board_type,
                    square=self.square,
                    tag_spacing=self.tag_spacing,
                    tag_family=self.tag_family,
                    start_id=self.tag_start_id,
                    min_tags=self.min_tags,
                )

        scale = 1.0
        if width > self.display_width:
            scale = float(self.display_width) / float(width)
            display = cv2.resize(bgr, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        else:
            display = bgr.copy()

        with self.lock:
            self.image_size = (source_width, source_height)
            self._frame_sequence += 1
            accepted = False
            duplicate = False
            if detection is not None:
                corners = detection.image_points
                params = detection.coverage
                calibration_corners = (
                    detection.calibration_image_points
                    if detection.calibration_image_points is not None
                    else corners
                )
                calibration_corners = (
                    np.asarray(calibration_corners, dtype=np.float32) * source_scale
                ).astype(np.float32)
                calibration_objects = (
                    detection.calibration_object_points
                    if detection.calibration_object_points is not None
                    else detection.object_points
                )
                scaled = (corners * scale).astype(np.float32)
                if self.board_type == "aprilgrid":
                    for point in scaled.reshape(-1, 2):
                        cv2.circle(
                            display, (int(round(point[0])), int(round(point[1]))), 4, (0, 255, 0), -1
                        )
                else:
                    cv2.drawChessboardCorners(display, self.board_size, scaled, True)
                pose_matches = self._render_pose_matches_camera(
                    render_position, render_orientation
                )
                if self.result is None and pose_matches:
                    accepted = intrinsic_solver.is_new_sample(
                        params, self.samples, self.sample_distance
                    )
                    duplicate = not accepted
                    if accepted:
                        if self.board_type == "aprilgrid" and source_jpeg:
                            try:
                                source_gray = cv2.imdecode(
                                    np.frombuffer(source_jpeg, dtype=np.uint8),
                                    cv2.IMREAD_GRAYSCALE,
                                )
                                if (
                                    not isinstance(source_gray, np.ndarray)
                                    or source_gray.shape[:2] != (source_height, source_width)
                                ):
                                    raise ValueError(
                                        "source JPEG dimensions do not match snapshot metadata"
                                    )
                                calibration_corners = (
                                    intrinsic_solver.refine_aprilgrid_calibration_centers(
                                        source_gray,
                                        np.asarray(corners, dtype=np.float32) * source_scale,
                                    )
                                )
                            except Exception as error:
                                # The bounded detection remains useful and the
                                # source-coordinate fallback is still valid.
                                # Surface refinement trouble without dropping
                                # the operator's newest detection frame.
                                self._recovery_error = str(error) or error.__class__.__name__
                        self.samples.append(params)
                        self.image_points.append(calibration_corners)
                        self.object_points.append(calibration_objects)
                        try:
                            self._save_checkpoint_locked()
                        except Exception as error:
                            self._recovery_error = str(error) or error.__class__.__name__
                self._mark_aligned(display, render_position, render_orientation)
                self.latest_detection = {
                    "status": "detected",
                    "corner_count": int(len(corners)),
                    "expected_corner_count": self.latest_detection["expected_corner_count"],
                    "frame_width": source_width,
                    "frame_height": source_height,
                    "sequence": self._frame_sequence,
                    "metrics": [
                        {"label": label, "value": float(value)}
                        for label, value in zip(intrinsic_solver.PARAM_NAMES, params)
                    ],
                    "accepted": accepted,
                    "duplicate": duplicate,
                }
            else:
                self.latest_detection = {
                    "status": "not_detected",
                    "corner_count": 0,
                    "expected_corner_count": self.latest_detection["expected_corner_count"],
                    "frame_width": source_width,
                    "frame_height": source_height,
                    "sequence": self._frame_sequence,
                    "metrics": [],
                    "accepted": False,
                    "duplicate": False,
                }
            self._display = display
            self._detection_condition.notify_all()

    @staticmethod
    def _decode_adaptive_aprilgrid_frame(
        source_jpeg: bytes,
        source_width: int,
    ) -> Optional[np.ndarray]:
        encoded = np.frombuffer(source_jpeg, dtype=np.uint8)
        target_width = min(APRILGRID_ADAPTIVE_DETECTION_WIDTH, source_width)
        decode_flag = (
            cv2.IMREAD_REDUCED_COLOR_2
            if source_width >= int(target_width * 1.5)
            else cv2.IMREAD_COLOR
        )
        image = cv2.imdecode(encoded, decode_flag)
        if not isinstance(image, np.ndarray) or image.ndim != 3 or image.shape[2] != 3:
            return None
        if image.shape[1] != target_width:
            scale = float(target_width) / float(image.shape[1])
            image = cv2.resize(
                image,
                (max(1, int(image.shape[1] * scale)), max(1, int(image.shape[0] * scale))),
                interpolation=cv2.INTER_CUBIC if scale > 1.0 else cv2.INTER_AREA,
            )
        return image

    def image_jpeg(self) -> bytes:
        with self.lock:
            display = self._display
        if display is None:
            raise ApiError(HTTPStatus.SERVICE_UNAVAILABLE, "No camera image has arrived")
        return self._encode_jpeg(display)

    def _board_document(self) -> Dict[str, Any]:
        document: Dict[str, Any] = {
            "type": self.board_type,
            "size": list(self.board_size),
            "square_size_m": self.square,
        }
        if self.board_type == "aprilgrid":
            last_id = self.tag_start_id + self.board_size[0] * self.board_size[1] - 1
            document.update(
                {
                    "tag_family": self.tag_family,
                    "tag_spacing_m": self.tag_spacing,
                    "start_id": self.tag_start_id,
                    "end_id": last_id,
                }
            )
        return document

    def targets_document(self) -> Dict[str, Any]:
        """Static guide geometry for the 3D scene: board + recommended views."""
        with self.lock:
            return {
                "board": dict(self.board_geometry),
                "views": [
                    {"name": view["name"], "position": view["position"]}
                    for view in self.views
                ],
                "camera_control": self.camera is not None,
            }

    def state(self) -> Dict[str, Any]:
        with self.lock:
            bars, sample_goodenough = intrinsic_solver.coverage(self.samples)
            if not self.samples and self.restored_coverage:
                bars = [dict(item) for item in self.restored_coverage]
            goodenough = self.result is not None or sample_goodenough
            guidance = intrinsic_solver.next_view_guidance(self.samples)
            targets = [{
                "name": view["name"],
                "position": view["position"],
                "done": self.target_done[index],
                "has_ref": index in self.refs,
            } for index, view in enumerate(self.views)]
            next_index = next((i for i, done in enumerate(self.target_done) if not done), None)
            pose = self.camera.current() if self.camera is not None else None
            return {
                "mode": "intrinsic",
                "samples": len(self.samples) if self.samples else (
                    self.result.sample_count if self.result is not None else 0
                ),
                "coverage": bars,
                "guidance": guidance,
                "goodenough": bool(goodenough),
                "calibrated": self.result is not None,
                "result_restored": self.result_restored,
                "result": self.result_payload,
                "output_file": self.output_file,
                "image_ready": self._display is not None,
                "media_source": self.media_source,
                "board": self._board_document(),
                "targets": targets,
                "next": next_index,
                "pose": pose,
                "camera_control": self.camera is not None,
                "auto_capture": self._auto_capture_document_locked(),
                "recovery": {
                    "checkpoint_file": self.checkpoint_file,
                    "checkpoint_available": Path(self.checkpoint_file).is_file(),
                    "result_restored": self.result_restored,
                    "last_error": self._recovery_error,
                },
                "action": dict(self.action) if self.action is not None else None,
                "detection": {
                    **self.latest_detection,
                    "metrics": [dict(metric) for metric in self.latest_detection["metrics"]],
                },
            }

    # -- sim camera guidance actions -----------------------------------------
    def _require_idle_locked(self) -> None:
        if self.action is not None and self.action.get("status") == "running":
            raise ApiError(
                HTTPStatus.CONFLICT,
                "Intrinsic calibration action '{}' is already running".format(
                    self.action.get("name", "unknown")
                ),
                details={"action": dict(self.action)},
            )
        # A failed background action remains visible until the operator makes
        # the next deliberate mutation, which also acknowledges the failure.
        if self.action is not None and self.action.get("status") in ("failed", "succeeded"):
            self.action = None

    def _clear_session_locked(self) -> None:
        self._remove_checkpoint_locked()
        self.samples = []
        self.image_points = []
        self.object_points = []
        self.result = None
        self.result_payload = None
        self.result_restored = False
        self.restored_coverage = []
        self.output_file = self.output_file_base
        self.refs = {}
        self.target_done = [False] * len(self.views)
        self._selected_target_index = None
        self._auto_capture_completed = False

    def _require_camera(self) -> Any:
        if self.camera is None:
            raise ApiError(HTTPStatus.NOT_FOUND, "No camera control is available")
        return self.camera

    def _capture_frame(self) -> Dict[str, Any]:
        with self._capture_lock:
            with self.lock:
                capture = self.frame_capture
            if capture is None:
                raise ApiError(
                    HTTPStatus.SERVICE_UNAVAILABLE,
                    "No calibration frame source is available",
                )
            try:
                frame = capture()
            except ApiError:
                raise
            except Exception as error:
                raise ApiError(
                    HTTPStatus.SERVICE_UNAVAILABLE,
                    "Could not capture a calibration frame: {}".format(error),
                ) from error
            render_position = getattr(frame, "render_position", None)
            render_orientation = getattr(frame, "render_orientation", None)
            source_width = getattr(frame, "width", None)
            source_height = getattr(frame, "height", None)
            source_jpeg = getattr(frame, "jpeg", None)
            if not isinstance(frame, np.ndarray):
                frame = getattr(frame, "bgr", None)
            if not isinstance(frame, np.ndarray):
                raise ApiError(
                    HTTPStatus.SERVICE_UNAVAILABLE,
                    "Calibration frame source returned no image",
                )
            source_image_size = (
                (int(source_width), int(source_height))
                if isinstance(source_width, int) and isinstance(source_height, int)
                else None
            )
            self.process_frame(
                frame,
                render_position,
                render_orientation,
                source_image_size=source_image_size,
                source_jpeg=source_jpeg if isinstance(source_jpeg, bytes) else None,
            )
        with self.lock:
            return {"ok": True, "samples": len(self.samples)}

    def capture(self) -> Dict[str, Any]:
        """Explicitly collect one calibration-board sample from Media Edge."""
        with self.lock:
            self._require_idle_locked()
        return self._capture_frame()

    def goto(self, index: int) -> Dict[str, Any]:
        with self.lock:
            self._require_idle_locked()
            camera = self._require_camera()
            if not 0 <= index < len(self.views):
                raise ApiError(HTTPStatus.UNPROCESSABLE_ENTITY, "Unknown target index")
            view = self.views[index]
            self._selected_target_index = index
            # Keep admission and the short camera command atomic with respect
            # to an auto-run starting on another HTTP worker thread.
            camera.goto(
                view["position"], view["yaw_offset"], view["pitch_offset"], view["roll"]
            )
        return {"ok": True, "name": view["name"]}

    def reset_pose(self) -> Dict[str, Any]:
        with self.lock:
            self._require_idle_locked()
            camera = self._require_camera()
            self._selected_target_index = None
            camera.reset()
        return {"ok": True}

    def auto_run(self, settle: float = 1.3, detection_timeout: float = 2.0) -> Dict[str, Any]:
        """Start a background sweep through every recommended sample view.

        The HTTP transport remains responsive while the camera dwells at each
        pose. The state event stream exposes the authoritative in-flight action;
        mutating operator actions are rejected until the sweep finishes.
        """
        if float(settle) < 0.0:
            raise ValueError("settle must be non-negative")
        if float(detection_timeout) <= 0.0:
            raise ValueError("detection_timeout must be positive")
        with self.lock:
            self._require_idle_locked()
            camera = self._require_camera()
            detector = self._auto_capture_thread
            if detector is None or not detector.is_alive() or self._auto_capture_stop.is_set():
                raise ApiError(
                    HTTPStatus.SERVICE_UNAVAILABLE,
                    "Continuous intrinsic detection is not running",
                )
            self._clear_session_locked()
            self.action = {
                "name": "auto_run",
                "status": "running",
                "target_index": None,
                "target_name": None,
                "error": None,
            }
            thread = threading.Thread(
                target=self._run_auto_sweep,
                args=(camera, float(settle), float(detection_timeout)),
                name="intrinsic-auto-run",
                daemon=True,
            )
            self._auto_run_thread = thread
            accepted = {"accepted": True, "action": dict(self.action)}
        try:
            thread.start()
        except RuntimeError as error:
            with self.lock:
                self.action = {
                    "name": "auto_run",
                    "status": "failed",
                    "target_index": None,
                    "target_name": None,
                    "error": str(error) or "Could not start the automatic coverage sweep",
                }
                self._auto_run_thread = None
            raise ApiError(
                HTTPStatus.INTERNAL_SERVER_ERROR,
                "Could not start the automatic coverage sweep",
            ) from error
        return accepted

    def _run_auto_sweep(self, camera: Any, settle: float, detection_timeout: float) -> None:
        try:
            for index, view in enumerate(self.views):
                with self.lock:
                    if self.action is None or self.action.get("status") != "running":
                        return
                    self.action["target_index"] = index
                    self.action["target_name"] = view["name"]
                    self._selected_target_index = index
                camera.goto(
                    view["position"], view["yaw_offset"], view["pitch_offset"], view["roll"]
                )
                time.sleep(settle)
                if not self._wait_for_target_detection(index, detection_timeout):
                    raise CalibrationError(
                        "continuous detection could not find the calibration board "
                        "at target '{}'".format(view["name"])
                    )
        except Exception as error:  # Camera-control failures are reported through state.
            with self.lock:
                self.action = {
                    "name": "auto_run",
                    "status": "failed",
                    "target_index": self.action.get("target_index") if self.action else None,
                    "target_name": self.action.get("target_name") if self.action else None,
                    "error": str(error) or error.__class__.__name__,
                }
                self._auto_run_thread = None
            return
        try:
            with self.lock:
                _bars, goodenough = intrinsic_solver.coverage(self.samples)
                if not goodenough:
                    raise CalibrationError(
                        "automatic sweep did not reach full X/Y/Size/Skew coverage "
                        "({} distinct samples)".format(len(self.samples))
                    )
                result = self._calibrate_locked()
                self.action = {
                    "name": "auto_run",
                    "status": "succeeded",
                    "target_index": len(self.views) - 1,
                    "target_name": self.views[-1]["name"],
                    "error": None,
                    "result": dict(result),
                }
                self._auto_run_thread = None
        except Exception as error:
            with self.lock:
                self.action = {
                    "name": "auto_run",
                    "status": "failed",
                    "target_index": self.action.get("target_index") if self.action else None,
                    "target_name": self.action.get("target_name") if self.action else None,
                    "error": str(error) or error.__class__.__name__,
                }
                self._auto_run_thread = None

    def _wait_for_target_detection(self, index: int, timeout: float) -> bool:
        deadline = time.monotonic() + timeout
        with self._detection_condition:
            while not self.target_done[index]:
                if self.action is None or self.action.get("status") != "running":
                    return False
                remaining = deadline - time.monotonic()
                if remaining <= 0.0:
                    return False
                self._detection_condition.wait(timeout=remaining)
            return True

    def record_references(self, settle: float = 1.3) -> Dict[str, Any]:
        """One-off: fly to every view, snapshot the annotated frame as its
        reference image, then start fresh so the operator still calibrates
        manually with all spheres grey.
        """
        with self.lock:
            self._require_idle_locked()
            camera = self._require_camera()
            self._recording = True
        saved = 0
        try:
            for index, view in enumerate(self.views):
                camera.goto(view["position"], view["yaw_offset"], view["pitch_offset"], view["roll"])
                time.sleep(settle)
                with self.lock:
                    display = None if self._display is None else self._display.copy()
                if display is not None:
                    ok, encoded = cv2.imencode(".jpg", display, [int(cv2.IMWRITE_JPEG_QUALITY), 75])
                    if ok:
                        with self.lock:
                            self._save_ref(index, encoded.tobytes())
                        saved += 1
            camera.reset()
        finally:
            with self.lock:
                self._recording = False
            self.reset()
        return {"ok": True, "saved": saved}

    def calibrate(self) -> Dict[str, Any]:
        with self.lock:
            self._require_idle_locked()
            return self._calibrate_locked()

    def _calibrate_locked(self) -> Dict[str, Any]:
        """Solve and atomically save while the caller owns ``self.lock``."""
        if self.result is not None:
            return self.result_payload  # type: ignore[return-value]
        if not self.image_points or self.image_size is None:
            raise ApiError(HTTPStatus.CONFLICT, "No calibration-board samples collected yet")
        try:
            result = intrinsic_solver.calibrate_intrinsic(
                self.image_points,
                self.board_size,
                self.square,
                self.image_size,
                object_points=self.object_points or None,
            )
        except (CalibrationError, cv2.error) as error:
            raise ApiError(HTTPStatus.UNPROCESSABLE_ENTITY, str(error)) from error
        try:
            output_file = self._versioned_output_path()
            intrinsic_solver.save_intrinsic(
                output_file,
                result,
                camera_name=self.camera_name,
                board_size=self.board_size,
                square=self.square,
                metadata={
                    "media_source": self.media_source,
                    "camera_name": self.camera_name,
                    "web_calibrator": True,
                    "coverage": intrinsic_solver.coverage(self.samples)[0],
                },
                board=self._board_document(),
            )
        except OSError as error:
            raise ApiError(
                HTTPStatus.INTERNAL_SERVER_ERROR,
                "Could not save calibration result: {}".format(error),
            ) from error
        self.output_file = str(output_file)
        self.result = result
        self.result_restored = False
        self.result_payload = self._result_document(result)
        try:
            Path(self.checkpoint_file).unlink()
        except FileNotFoundError:
            pass
        except OSError as error:
            self._recovery_error = str(error) or error.__class__.__name__
        return self.result_payload

    def reset(self) -> Dict[str, Any]:
        with self.lock:
            self._require_idle_locked()
            self._clear_session_locked()
            self.action = None
        return self.state()
