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

import os
import threading
import time
from http import HTTPStatus
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from xgc_camera_calibration import intrinsic_solver
from xgc_camera_calibration.solver import CalibrationError
from xgc_camera_calibration.web_service import ApiError


def recommended_views(board_center: Sequence[float]) -> List[Dict[str, Any]]:
    """Spatially-distinct sample poses that together fill X / Y / Size / Skew.

    Filling X and Y needs the board off-centre in the image, so those poses carry
    a yaw/pitch aim offset (the camera adapter applies it through
    look_at_orientation) -- a camera that simply aims at the board keeps it
    centred and never moves the X/Y bars.  Size needs near and far views; Skew
    needs the oblique corners.  Each pose sits at its own point so it is a
    distinct, clickable marker in the 3D guide.
    """
    tx, ty, tz = float(board_center[0]), float(board_center[1]), float(board_center[2])
    # Measured against the product's 1280x720, 110-degree Gazebo camera. Every
    # pose keeps the board detectable while the set spans both image edges,
    # near/far scale and perspective skew. The lowest camera is only 0.8 m
    # below the board centre (1.4 m at the default 2.2 m board height), so the
    # automatic path never grazes the ground.
    specs = [
        ("far lower left", (tx - 6.0, ty + 0.3, tz + 0.1), -0.76, -0.24),
        ("far lower right", (tx - 6.0, ty - 0.3, tz + 0.1), 0.76, -0.24),
        ("far bottom", (tx - 6.0, ty, tz - 0.1), 0.0, -0.48),
        ("far lower center", (tx - 6.0, ty, tz), 0.0, -0.24),
        ("upper perspective", (tx - 2.5, ty - 1.2, tz - 0.8), 0.0, 0.44),
        ("upper oblique", (tx - 3.5, ty + 2.0, tz + 1.0), 0.0, 0.44),
        ("center oblique left", (tx - 2.5, ty + 1.2, tz - 0.8), 0.0, 0.0),
        ("center oblique right", (tx - 2.5, ty - 1.2, tz + 0.8), 0.0, 0.0),
        ("medium center", (tx - 4.0, ty, tz), 0.0, -0.08),
        ("near center", (tx - 2.0, ty, tz), 0.0, -0.08),
        ("near large", (tx - 1.4, ty, tz), 0.0, -0.08),
        ("near maximum", (tx - 1.2, ty, tz), 0.0, -0.08),
        ("diagonal high", (tx - 2.5, ty + 2.6, tz + 1.7), 0.0, 0.0),
        ("lower right perspective", (tx - 4.0, ty - 1.0, tz - 0.8), 0.0, 0.20),
        ("upper left perspective", (tx - 2.5, ty + 1.2, tz + 0.8), 0.0, 0.28),
    ]
    return [{
        "name": name,
        "position": [round(value, 2) for value in position],
        "yaw_offset": yaw_offset,
        "pitch_offset": pitch_offset,
        "roll": 0.0,
    } for (name, position, yaw_offset, pitch_offset) in specs]


class IntrinsicCalibrationService:
    """Own one operator intrinsic-calibration session over a stream of frames."""

    def __init__(
        self,
        *,
        board_size: Sequence[int],
        square: float,
        output_file: str,
        image_topic: str = "",
        camera_info_topic: str = "",
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
        self.output_file = str(Path(output_file).expanduser())
        self.image_topic = str(image_topic)
        self.camera_info_topic = str(camera_info_topic)
        self.media_source = str(media_source).strip()
        self.jpeg_quality = int(jpeg_quality)
        self.sample_distance = float(sample_distance)
        self.maximum_detect_width = int(maximum_detect_width)
        self.display_width = int(display_width)
        self.lock = threading.RLock()
        self.samples: List[Tuple[float, float, float, float]] = []
        self.image_points: List[np.ndarray] = []
        self.object_points: List[np.ndarray] = []
        self.image_size: Optional[Tuple[int, int]] = None
        self._display: Optional[np.ndarray] = None
        self.result: Optional[intrinsic_solver.IntrinsicResult] = None
        self.result_payload: Optional[Dict[str, Any]] = None
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
        self.views: List[Dict[str, Any]] = recommended_views(self.board_center)
        self.target_done: List[bool] = [False] * len(self.views)
        self.references_dir = str(Path(references_dir).expanduser()) if references_dir else ""
        self.refs: Dict[int, bytes] = {}
        self.align_threshold = float(align_threshold)
        self.camera: Optional[Any] = None
        self.frame_capture: Optional[Callable[[], np.ndarray]] = None
        self._recording = False
        self.action: Optional[Dict[str, Any]] = None
        self._auto_run_thread: Optional[threading.Thread] = None
        self._load_refs()

    # -- guide wiring ---------------------------------------------------------
    def attach_camera_control(self, camera: Any) -> None:
        """Attach an optional sim camera adapter (goto/reset/current pose)."""
        with self.lock:
            self.camera = camera

    def attach_frame_capture(self, capture: Callable[[], np.ndarray]) -> None:
        """Attach an explicit immutable-frame transaction.

        It is intentionally a callback rather than a persistent subscriber:
        the browser owns the WebRTC live path, while the calibration algorithm
        pays for one RGB readback only when an operator captures a sample.
        """
        with self.lock:
            self.frame_capture = capture

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

    def _mark_aligned(self, display: np.ndarray) -> None:
        """Green the nearest target once the camera aligns to it and the board is
        visible this frame -- independent of whether this frame became a *new*
        sample (is_new_sample de-duplicates similar views, so a
        redundant-but-valid pose would otherwise stay grey).  Requires the sim
        camera adapter for the live pose; a no-op for a real camera.
        """
        if self._recording or self.camera is None:
            return
        position = self.camera.current_position()
        if position is None:
            return
        index = None
        if self.action is not None and self.action.get("status") == "running":
            candidate = self.action.get("target_index")
            if isinstance(candidate, int) and 0 <= candidate < len(self.views):
                index = candidate
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

    def _encode_jpeg(self, image: np.ndarray) -> bytes:
        ok, encoded = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), self.jpeg_quality])
        if not ok:
            raise ApiError(HTTPStatus.INTERNAL_SERVER_ERROR, "Could not encode camera frame")
        return encoded.tobytes()

    def process_frame(self, bgr: np.ndarray) -> None:
        """Ingest one decoded BGR frame: detect the board, auto-collect, annotate."""
        if bgr.ndim != 3 or bgr.shape[2] != 3:
            return
        height, width = bgr.shape[:2]
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        detection = intrinsic_solver.detect_board(
            gray,
            self.board_size,
            self.maximum_detect_width,
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
            self.image_size = (width, height)
            self._frame_sequence += 1
            accepted = False
            duplicate = False
            if detection is not None:
                corners = detection.image_points
                params = detection.coverage
                scaled = (corners * scale).astype(np.float32)
                if self.board_type == "aprilgrid":
                    for point in scaled.reshape(-1, 2):
                        cv2.circle(
                            display, (int(round(point[0])), int(round(point[1]))), 4, (0, 255, 0), -1
                        )
                else:
                    cv2.drawChessboardCorners(display, self.board_size, scaled, True)
                if self.result is None:
                    accepted = intrinsic_solver.is_new_sample(
                        params, self.samples, self.sample_distance
                    )
                    duplicate = not accepted
                    if accepted:
                        self.samples.append(params)
                        self.image_points.append(corners)
                        self.object_points.append(detection.object_points)
                self._mark_aligned(display)
                self.latest_detection = {
                    "status": "detected",
                    "corner_count": int(len(corners)),
                    "expected_corner_count": self.latest_detection["expected_corner_count"],
                    "frame_width": width,
                    "frame_height": height,
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
                    "frame_width": width,
                    "frame_height": height,
                    "sequence": self._frame_sequence,
                    "metrics": [],
                    "accepted": False,
                    "duplicate": False,
                }
            self._display = display

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
            bars, goodenough = intrinsic_solver.coverage(self.samples)
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
                "samples": len(self.samples),
                "coverage": bars,
                "goodenough": bool(goodenough),
                "calibrated": self.result is not None,
                "result": self.result_payload,
                "output_file": self.output_file,
                "image_ready": self._display is not None,
                "media_source": self.media_source or self.image_topic,
                "board": self._board_document(),
                "targets": targets,
                "next": next_index,
                "pose": pose,
                "camera_control": self.camera is not None,
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
        self.samples = []
        self.image_points = []
        self.object_points = []
        self.result = None
        self.result_payload = None
        self.target_done = [False] * len(self.views)

    def _require_camera(self) -> Any:
        if self.camera is None:
            raise ApiError(HTTPStatus.NOT_FOUND, "No camera control is available")
        return self.camera

    def _capture_frame(self) -> Dict[str, Any]:
        with self.lock:
            capture = self.frame_capture
        if capture is None:
            raise ApiError(HTTPStatus.SERVICE_UNAVAILABLE, "No calibration frame source is available")
        try:
            frame = capture()
        except ApiError:
            raise
        except Exception as error:
            raise ApiError(HTTPStatus.SERVICE_UNAVAILABLE, "Could not capture a calibration frame: {}".format(error)) from error
        if not isinstance(frame, np.ndarray):
            raise ApiError(HTTPStatus.SERVICE_UNAVAILABLE, "Calibration frame source returned no image")
        self.process_frame(frame)
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
            camera.reset()
        return {"ok": True}

    def auto_run(self, settle: float = 1.3) -> Dict[str, Any]:
        """Start a background sweep through every recommended sample view.

        The HTTP transport must remain responsive while the camera dwells at
        each pose.  State polling exposes the authoritative in-flight action;
        mutating operator actions are rejected until the sweep finishes.
        """
        if float(settle) < 0.0:
            raise ValueError("settle must be non-negative")
        with self.lock:
            self._require_idle_locked()
            camera = self._require_camera()
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
                args=(camera, float(settle)),
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

    def _run_auto_sweep(self, camera: Any, settle: float) -> None:
        try:
            for index, view in enumerate(self.views):
                with self.lock:
                    if self.action is None or self.action.get("status") != "running":
                        return
                    self.action["target_index"] = index
                    self.action["target_name"] = view["name"]
                camera.goto(
                    view["position"], view["yaw_offset"], view["pitch_offset"], view["roll"]
                )
                time.sleep(settle)
                # One bounded snapshot after the pose settles replaces the old
                # perpetual ROS/JPEG feeder. Automatic calibration is accepted
                # only when this transaction reaches full coverage and solves.
                with self.lock:
                    capture = self.frame_capture
                if capture is not None:
                    self._capture_frame()
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
            intrinsic_solver.save_intrinsic(
                self.output_file,
                result,
                board_size=self.board_size,
                square=self.square,
                metadata={"media_source": self.media_source or self.image_topic, "web_calibrator": True},
                board=self._board_document(),
            )
        except OSError as error:
            raise ApiError(
                HTTPStatus.INTERNAL_SERVER_ERROR,
                "Could not save calibration result: {}".format(error),
            ) from error
        matrix = result.camera_matrix
        self.result = result
        self.result_payload = {
            "camera_matrix": [float(v) for v in matrix.reshape(-1)],
            "distortion": [float(v) for v in result.distortion],
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
        return self.result_payload

    def reset(self) -> Dict[str, Any]:
        with self.lock:
            self._require_idle_locked()
            self._clear_session_locked()
            self.action = None
        return self.state()
