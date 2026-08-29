"""Robust 3D-to-2D camera extrinsic estimation and result persistence."""

from __future__ import annotations

import math
import os
import re
import tempfile
from itertools import chain, combinations, islice
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Sequence, Tuple

import cv2
import numpy as np
import yaml


CAMERA_NAME_PATTERN = re.compile(r"^[A-Za-z][A-Za-z0-9._-]{0,63}$")


class CalibrationError(RuntimeError):
    """Raised when the correspondence set cannot yield a trustworthy pose."""


def extrinsic_calibration_directory(root: str, mode: str, camera_name: str) -> Path:
    """Resolve the explicit storage identity shared by solver and TF consumer."""

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


def versioned_extrinsic_path(directory: os.PathLike) -> Path:
    """Allocate a concrete UTC-named result path without creating an alias."""

    destination = Path(directory).expanduser()
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    candidate = destination / "extrinsics-{}.yaml".format(timestamp)
    sequence = 1
    while candidate.exists():
        candidate = destination / "extrinsics-{}-{:02d}.yaml".format(timestamp, sequence)
        sequence += 1
    return candidate


@dataclass(frozen=True)
class ExtrinsicResult:
    """The world_T_camera pose and the world-to-camera projection model."""

    translation: np.ndarray
    quaternion_xyzw: np.ndarray
    rotation_world_to_camera: np.ndarray
    translation_world_to_camera: np.ndarray
    reprojection_errors_px: np.ndarray
    inlier_indices: np.ndarray
    warnings: Sequence[str]

    @property
    def mean_reprojection_error_px(self) -> float:
        return float(np.mean(self.reprojection_errors_px))

    @property
    def max_reprojection_error_px(self) -> float:
        return float(np.max(self.reprojection_errors_px))


def _as_points(values: Iterable[Iterable[float]], columns: int, name: str) -> np.ndarray:
    points = np.asarray(values, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != columns or not np.all(np.isfinite(points)):
        raise CalibrationError("{} must be a finite Nx{} array".format(name, columns))
    return points


def rotation_matrix_to_quaternion(rotation: np.ndarray) -> np.ndarray:
    matrix = np.asarray(rotation, dtype=np.float64).reshape(3, 3)
    trace = float(np.trace(matrix))
    if trace > 0.0:
        scale = math.sqrt(trace + 1.0) * 2.0
        quaternion = np.array(
            [
                (matrix[2, 1] - matrix[1, 2]) / scale,
                (matrix[0, 2] - matrix[2, 0]) / scale,
                (matrix[1, 0] - matrix[0, 1]) / scale,
                0.25 * scale,
            ]
        )
    else:
        diagonal = np.diag(matrix)
        index = int(np.argmax(diagonal))
        if index == 0:
            scale = math.sqrt(max(0.0, 1.0 + matrix[0, 0] - matrix[1, 1] - matrix[2, 2])) * 2.0
            quaternion = np.array(
                [0.25 * scale, (matrix[0, 1] + matrix[1, 0]) / scale,
                 (matrix[0, 2] + matrix[2, 0]) / scale,
                 (matrix[2, 1] - matrix[1, 2]) / scale]
            )
        elif index == 1:
            scale = math.sqrt(max(0.0, 1.0 + matrix[1, 1] - matrix[0, 0] - matrix[2, 2])) * 2.0
            quaternion = np.array(
                [(matrix[0, 1] + matrix[1, 0]) / scale, 0.25 * scale,
                 (matrix[1, 2] + matrix[2, 1]) / scale,
                 (matrix[0, 2] - matrix[2, 0]) / scale]
            )
        else:
            scale = math.sqrt(max(0.0, 1.0 + matrix[2, 2] - matrix[0, 0] - matrix[1, 1])) * 2.0
            quaternion = np.array(
                [(matrix[0, 2] + matrix[2, 0]) / scale,
                 (matrix[1, 2] + matrix[2, 1]) / scale, 0.25 * scale,
                 (matrix[1, 0] - matrix[0, 1]) / scale]
            )
    norm = float(np.linalg.norm(quaternion))
    if norm < 1e-12:
        raise CalibrationError("rotation produced an invalid quaternion")
    quaternion /= norm
    if quaternion[3] < 0.0:
        quaternion *= -1.0
    return quaternion


def _planar_pose_ransac(
    world: np.ndarray,
    pixels: np.ndarray,
    intrinsic: np.ndarray,
    coefficients: np.ndarray,
    reprojection_error_px: float,
    iterations: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Estimate a planar pose without relying on EPNP's ambiguous coplanar branch."""

    best = None
    index_sets = chain(
        (tuple(range(len(world))),),
        islice(combinations(range(len(world)), 4), int(iterations)),
    )
    for indices in index_sets:
        subset_world = world[list(indices)]
        singular_values = np.linalg.svd(
            subset_world - np.mean(subset_world, axis=0), compute_uv=False
        )
        scale = max(float(singular_values[0]), 1e-12)
        if int(np.count_nonzero(singular_values > scale * 1e-6)) < 2:
            continue
        candidates = []
        try:
            solved = cv2.solvePnPGeneric(
                subset_world,
                pixels[list(indices)],
                intrinsic,
                coefficients,
                flags=cv2.SOLVEPNP_IPPE,
            )
            if solved[0]:
                candidates.extend(zip(solved[1], solved[2]))
        except cv2.error:
            pass
        # OpenCV 4.2 can return no IPPE candidates for a valid noisy planar
        # subset. Its iterative planar initializer uses the same homography
        # geometry and is stable across the Focal/Noetic ABI.
        try:
            ok, rotation_vector, translation_vector = cv2.solvePnP(
                subset_world,
                pixels[list(indices)],
                intrinsic,
                coefficients,
                flags=cv2.SOLVEPNP_ITERATIVE,
            )
            if ok:
                candidates.append((rotation_vector, translation_vector))
        except cv2.error:
            pass
        for rotation_vector, translation_vector in candidates:
            if not (
                np.all(np.isfinite(rotation_vector))
                and np.all(np.isfinite(translation_vector))
            ):
                continue
            projected, _ = cv2.projectPoints(
                world.reshape(-1, 1, 3),
                rotation_vector,
                translation_vector,
                intrinsic,
                coefficients,
            )
            errors = np.linalg.norm(projected.reshape(-1, 2) - pixels, axis=1)
            rotation, _ = cv2.Rodrigues(rotation_vector)
            depths = (
                rotation.dot(world.T) + np.asarray(translation_vector).reshape(3, 1)
            )[2]
            inliers = np.flatnonzero(
                (errors <= float(reprojection_error_px)) & (depths > 0.0)
            )
            if len(inliers) < 4:
                continue
            score = (
                -len(inliers),
                float(np.mean(errors[inliers])),
                float(np.max(errors[inliers])),
            )
            if best is None or score < best[0]:
                best = (score, rotation_vector, translation_vector, inliers)
    if best is None:
        raise CalibrationError(
            "planar robot layout is degenerate; select at least two well-separated "
            "markers from both the UAV and UGV rows"
        )
    return best[1], best[2], np.asarray(best[3], dtype=np.int32)


def solve_extrinsic(
    world_points: Iterable[Iterable[float]],
    image_points: Iterable[Iterable[float]],
    camera_matrix: Iterable[Iterable[float]],
    distortion: Optional[Iterable[float]] = None,
    *,
    ransac_reprojection_error_px: float = 3.0,
    ransac_iterations: int = 300,
    confidence: float = 0.999,
    maximum_accepted_error_px: Optional[float] = None,
) -> ExtrinsicResult:
    """Estimate world_T_camera using RANSAC followed by iterative refinement."""

    world = _as_points(world_points, 3, "world_points")
    pixels = _as_points(image_points, 2, "image_points")
    if len(world) != len(pixels) or len(world) < 4:
        raise CalibrationError("at least four paired world/image points are required")
    intrinsic = np.asarray(camera_matrix, dtype=np.float64)
    if intrinsic.shape != (3, 3) or not np.all(np.isfinite(intrinsic)):
        raise CalibrationError("camera_matrix must be a finite 3x3 matrix")
    if intrinsic[0, 0] <= 0.0 or intrinsic[1, 1] <= 0.0 or abs(intrinsic[2, 2]) < 1e-12:
        raise CalibrationError("camera_matrix must contain positive focal lengths")
    coefficients = np.zeros((5, 1), dtype=np.float64) if distortion is None else np.asarray(
        distortion, dtype=np.float64
    ).reshape(-1, 1)
    if not np.all(np.isfinite(coefficients)):
        raise CalibrationError("distortion must contain only finite values")
    if ransac_reprojection_error_px <= 0.0 or ransac_iterations <= 0:
        raise CalibrationError("RANSAC threshold and iteration count must be positive")
    if not 0.0 < confidence < 1.0:
        raise CalibrationError("RANSAC confidence must be between zero and one")

    singular_values = np.linalg.svd(world - np.mean(world, axis=0), compute_uv=False)
    scale = max(float(singular_values[0]), 1e-12)
    rank = int(np.count_nonzero(singular_values > scale * 1e-6))
    if rank < 2:
        raise CalibrationError("world points are collinear or coincident")
    warnings = []
    if rank == 2:
        warnings.append("world points are coplanar; include depth-separated points when possible")

    if rank == 2:
        rvec, tvec, inlier_indices = _planar_pose_ransac(
            world,
            pixels,
            intrinsic,
            coefficients,
            ransac_reprojection_error_px,
            ransac_iterations,
        )
    else:
        ok, rvec, tvec, inliers = cv2.solvePnPRansac(
            world.reshape(-1, 1, 3),
            pixels.reshape(-1, 1, 2),
            intrinsic,
            coefficients,
            iterationsCount=int(ransac_iterations),
            reprojectionError=float(ransac_reprojection_error_px),
            confidence=float(confidence),
            flags=cv2.SOLVEPNP_EPNP,
        )
        if not ok or inliers is None or len(inliers) < 4:
            raise CalibrationError("solvePnPRansac could not find at least four inliers")
        inlier_indices = np.asarray(inliers, dtype=np.int32).reshape(-1)
    inlier_world = world[inlier_indices].reshape(-1, 1, 3)
    inlier_pixels = pixels[inlier_indices].reshape(-1, 1, 2)

    if hasattr(cv2, "solvePnPRefineLM"):
        rvec, tvec = cv2.solvePnPRefineLM(
            inlier_world, inlier_pixels, intrinsic, coefficients, rvec, tvec
        )
    else:
        ok, rvec, tvec = cv2.solvePnP(
            inlier_world,
            inlier_pixels,
            intrinsic,
            coefficients,
            rvec,
            tvec,
            True,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )
        if not ok:
            raise CalibrationError("iterative PnP refinement failed")

    projected, _ = cv2.projectPoints(world.reshape(-1, 1, 3), rvec, tvec, intrinsic, coefficients)
    errors = np.linalg.norm(projected.reshape(-1, 2) - pixels, axis=1)
    if maximum_accepted_error_px is not None and float(np.max(errors[inlier_indices])) > float(
        maximum_accepted_error_px
    ):
        raise CalibrationError(
            "inlier reprojection error {:.3f}px exceeds {:.3f}px".format(
                float(np.max(errors[inlier_indices])), float(maximum_accepted_error_px)
            )
        )

    rotation_world_to_camera, _ = cv2.Rodrigues(rvec)
    rotation_camera_to_world = rotation_world_to_camera.T
    translation_world_to_camera = np.asarray(tvec, dtype=np.float64).reshape(3)
    translation_camera_to_world = -rotation_camera_to_world.dot(translation_world_to_camera)
    quaternion = rotation_matrix_to_quaternion(rotation_camera_to_world)
    if len(inlier_indices) != len(world):
        warnings.append("RANSAC rejected {} correspondence(s)".format(len(world) - len(inlier_indices)))

    return ExtrinsicResult(
        translation=translation_camera_to_world,
        quaternion_xyzw=quaternion,
        rotation_world_to_camera=rotation_world_to_camera,
        translation_world_to_camera=translation_world_to_camera,
        reprojection_errors_px=errors,
        inlier_indices=inlier_indices,
        warnings=tuple(warnings),
    )


def result_document(
    result: ExtrinsicResult,
    *,
    calibration_mode: str,
    camera_name: str,
    parent_frame: str,
    child_frame: str,
    points: Optional[Sequence[Dict[str, Any]]] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    if calibration_mode not in ("sim", "phy"):
        raise CalibrationError("calibration_mode must be sim or phy")
    if not CAMERA_NAME_PATTERN.fullmatch(str(camera_name).strip()):
        raise CalibrationError("camera_name must be a stable identifier")
    if not isinstance(parent_frame, str) or not parent_frame.strip():
        raise CalibrationError("parent_frame must be a non-empty string")
    if not isinstance(child_frame, str) or not child_frame.strip():
        raise CalibrationError("child_frame must be a non-empty string")
    document: Dict[str, Any] = {
        "schema": "xgc2.camera.extrinsic.v1",
        "created_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "calibration_mode": calibration_mode,
        "camera_name": str(camera_name).strip(),
        "frame_convention": "parent_T_camera_optical",
        "parent_frame": parent_frame,
        "child_frame": child_frame,
        "translation": dict(zip(("x", "y", "z"), map(float, result.translation))),
        "quaternion_xyzw": dict(zip(("x", "y", "z", "w"), map(float, result.quaternion_xyzw))),
        "mean_reprojection_error_px": result.mean_reprojection_error_px,
        "max_reprojection_error_px": result.max_reprojection_error_px,
        "inlier_indices": [int(index) for index in result.inlier_indices],
        "warnings": list(result.warnings),
    }
    if points is not None:
        document["points"] = list(points)
    if metadata:
        document["metadata"] = dict(metadata)
    return document


def save_extrinsic(
    path: os.PathLike,
    result: ExtrinsicResult,
    *,
    calibration_mode: str,
    camera_name: str,
    parent_frame: str,
    child_frame: str,
    points: Optional[Sequence[Dict[str, Any]]] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> Path:
    """Atomically persist a versioned extrinsic document outside package share."""

    destination = Path(path).expanduser()
    destination.parent.mkdir(parents=True, exist_ok=True)
    document = result_document(
        result,
        calibration_mode=calibration_mode,
        camera_name=camera_name,
        parent_frame=parent_frame,
        child_frame=child_frame,
        points=points,
        metadata=metadata,
    )
    descriptor, temporary_name = tempfile.mkstemp(
        prefix="." + destination.name + ".", suffix=".tmp", dir=str(destination.parent)
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            yaml.safe_dump(document, stream, default_flow_style=False, sort_keys=False)
            stream.flush()
            os.fsync(stream.fileno())
        os.chmod(temporary_name, 0o644)
        os.replace(temporary_name, destination)
    except Exception:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise
    return destination


def _ordered_values(value: Any, names: Sequence[str], field: str) -> np.ndarray:
    if isinstance(value, dict):
        try:
            values = [value[name] for name in names]
        except KeyError as error:
            raise CalibrationError("{} is missing {}".format(field, error.args[0])) from error
    elif isinstance(value, (list, tuple)) and len(value) == len(names):
        values = value
    else:
        raise CalibrationError("{} must contain {}".format(field, ",".join(names)))
    result = np.asarray(values, dtype=np.float64)
    if not np.all(np.isfinite(result)):
        raise CalibrationError("{} contains a non-finite value".format(field))
    return result


def load_extrinsic(path: os.PathLike) -> Dict[str, Any]:
    source = Path(path).expanduser()
    with source.open("r", encoding="utf-8") as stream:
        document = yaml.safe_load(stream) or {}
    if not isinstance(document, dict):
        raise CalibrationError("extrinsic document must be a mapping")
    if document.get("schema") != "xgc2.camera.extrinsic.v1":
        raise CalibrationError("unsupported or missing extrinsic schema")
    if document.get("calibration_mode") not in ("sim", "phy"):
        raise CalibrationError("unsupported or missing calibration mode")
    if not CAMERA_NAME_PATTERN.fullmatch(str(document.get("camera_name", "")).strip()):
        raise CalibrationError("unsupported or missing camera name")
    if document.get("frame_convention") != "parent_T_camera_optical":
        raise CalibrationError("unsupported or missing extrinsic frame convention")
    for field in ("parent_frame", "child_frame"):
        if not isinstance(document.get(field), str) or not document[field].strip():
            raise CalibrationError("{} must be a non-empty string".format(field))
    translation = _ordered_values(document.get("translation"), ("x", "y", "z"), "translation")
    quaternion = _ordered_values(
        document.get("quaternion_xyzw"), ("x", "y", "z", "w"), "quaternion_xyzw"
    )
    norm = float(np.linalg.norm(quaternion))
    if norm < 1e-12:
        raise CalibrationError("quaternion has zero norm")
    document["translation_array"] = translation
    document["quaternion_xyzw_array"] = quaternion / norm
    return document
