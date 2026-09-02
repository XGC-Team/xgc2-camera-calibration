"""Intrinsic (pinhole) camera calibration, cv2-direct and cv_bridge-free.

Mirrors the ROS ``camera_calibration`` coverage heuristics (X / Y / Size / Skew)
and delegates the actual estimation to ``cv2.calibrateCameraExtended`` without
dragging in ``cv_bridge`` or any ROS calibration class. Frames arrive already
decoded (see ``web_service.image_message_to_bgr``); board detection runs on a
down-scaled copy for speed on large (4K) frames, then corners are refined at full
resolution.
"""

from __future__ import annotations

import math
import os
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import cv2
import numpy as np
import yaml

from xgc_camera_calibration.solver import CalibrationError

_DETECT_FLAGS = cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE
_SUBPIX_CRITERIA = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER, 30, 0.01)
# Same acceptance ranges the ROS camera_calibration GUI uses for goodenough.
PARAM_RANGES: Tuple[float, float, float, float] = (0.7, 0.7, 0.4, 0.5)
PARAM_NAMES: Tuple[str, str, str, str] = ("X", "Y", "Size", "Skew")
_APRILTAG_DICTIONARIES = {
    "tag36h11": "DICT_APRILTAG_36h11",
    "apriltag_36h11": "DICT_APRILTAG_36h11",
    "36h11": "DICT_APRILTAG_36h11",
}
_APRILGRID_MARKER_BORDER_BITS = 2
APRILGRID_FEATURE_MODEL = "aprilgrid_kalibr_tag_corners_v2"
APRILGRID_CORNER_DATUM = "kalibr_id0_lower_left_opencv_rotated_180_v1"
_APRILGRID_OPENCV_TO_KALIBR_CORNER_ORDER: Tuple[int, int, int, int] = (1, 0, 3, 2)
_APRILGRID_MIN_BORDER_DISTANCE = 6.0
_APRILGRID_MIN_EDGE_LENGTH_PX = 20.0
_APRILGRID_EDGE_SEARCH_RADIUS_PX = 3.0
_APRILGRID_EDGE_SAMPLE_STEP_PX = 0.5
_APRILGRID_EDGE_MIN_PROFILE_GRADIENT = 6.0
_APRILGRID_EDGE_MIN_MEDIAN_CONTRAST = 24.0
_APRILGRID_EDGE_MAX_LINE_RMS_PX = 0.65
_APRILGRID_EDGE_MAX_LINE_P90_PX = 0.75
_CALIBRATION_CRITERIA = (
    cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER,
    100,
    1.0e-12,
)
_DISTORTION_PARAMETER_NAMES: Tuple[str, ...] = (
    "k1",
    "k2",
    "p1",
    "p2",
    "k3",
    "k4",
    "k5",
    "k6",
    "s1",
    "s2",
    "s3",
    "s4",
    "tau_x",
    "tau_y",
)


@dataclass(frozen=True)
class BoardDetection:
    image_points: np.ndarray
    object_points: np.ndarray
    coverage: Tuple[float, float, float, float]
    # Keep every decoded outer corner for annotation only. Coverage and solve
    # both use this same quality-gated image/object mask and corner ordering.
    calibration_image_points: Optional[np.ndarray] = None
    calibration_object_points: Optional[np.ndarray] = None


@dataclass(frozen=True)
class IntrinsicFoldEstimate:
    omitted_view_index: int
    # Training RMS for the N-1 optimization.
    rms_reprojection_error_px: float
    parameters: Tuple[float, ...]
    held_out_rms_reprojection_error_px: Optional[float] = None
    held_out_mean_reprojection_error_px: Optional[float] = None
    held_out_max_reprojection_error_px: Optional[float] = None
    held_out_point_errors_px: Tuple[float, ...] = ()
    held_out_rotation_vector: Optional[Tuple[float, float, float]] = None
    held_out_translation_vector: Optional[Tuple[float, float, float]] = None
    undistorted_ray_rms_equivalent_px: Optional[float] = None
    undistorted_ray_max_equivalent_px: Optional[float] = None


@dataclass(frozen=True)
class IntrinsicRejectedView:
    original_view_index: int
    reason: str
    initial_rms_reprojection_error_px: float
    rejection_rms_reprojection_error_px: float
    rejection_envelope_px: float


@dataclass(frozen=True)
class IntrinsicStabilityDiagnostics:
    """Continuous leave-one-view-out sensitivity, never an admission gate."""

    method: str
    parameter_names: Tuple[str, ...]
    reference_parameters: Tuple[float, ...]
    folds: Tuple[IntrinsicFoldEstimate, ...]
    failed_omitted_view_indices: Tuple[int, ...]
    parameter_standard_deviation: Optional[Tuple[float, ...]]
    parameter_span: Optional[Tuple[float, ...]]
    maximum_absolute_delta: Optional[Tuple[float, ...]]
    maximum_relative_delta: Optional[Tuple[float, ...]]
    held_out_rms_mean_px: Optional[float] = None
    held_out_rms_max_px: Optional[float] = None
    undistorted_ray_rms_equivalent_px: Optional[float] = None
    undistorted_ray_max_equivalent_px: Optional[float] = None


@dataclass(frozen=True)
class IntrinsicCalibrationDiagnostics:
    """Optimizer, residual, observability and stability evidence for one solve."""

    finite: bool
    parameter_names: Tuple[str, ...]
    per_view_errors_px: Tuple[float, ...]
    intrinsic_standard_deviations: Tuple[float, ...]
    rotation_vectors: Tuple[Tuple[float, float, float], ...]
    translation_vectors: Tuple[Tuple[float, float, float], ...]
    projected_intrinsic_rank: int
    projected_intrinsic_parameter_count: int
    projected_intrinsic_rank_deficient: bool
    projected_intrinsic_condition_number: float
    projected_intrinsic_rank_tolerance: float
    projected_intrinsic_singular_values: Tuple[float, ...]
    projected_intrinsic_column_norms: Tuple[float, ...]
    stability: IntrinsicStabilityDiagnostics
    pool_sample_count: int = 0
    selected_view_indices: Tuple[int, ...] = ()
    rejected_views: Tuple[IntrinsicRejectedView, ...] = ()
    initial_per_view_errors_px: Tuple[float, ...] = ()
    observation_uncertainty_px: Optional[float] = None


@dataclass(frozen=True)
class IntrinsicResult:
    camera_matrix: np.ndarray          # 3x3
    distortion: np.ndarray             # (k1,k2,p1,p2,k3)
    image_size: Tuple[int, int]        # (width, height)
    rms_reprojection_error_px: float
    sample_count: int
    diagnostics: Optional[IntrinsicCalibrationDiagnostics] = None


@dataclass(frozen=True)
class _ExtendedCalibration:
    rms_reprojection_error_px: float
    camera_matrix: np.ndarray
    distortion: np.ndarray
    rotation_vectors: Tuple[np.ndarray, ...]
    translation_vectors: Tuple[np.ndarray, ...]
    intrinsic_standard_deviations: np.ndarray
    per_view_errors_px: np.ndarray


def observation_uncertainty_px(board_type: str) -> float:
    """Return the active detector's pixel localization uncertainty contract."""
    kind = str(board_type).strip().lower()
    if kind == "aprilgrid":
        return float(_APRILGRID_EDGE_MAX_LINE_P90_PX)
    if kind in ("checkerboard", "chessboard"):
        return float(_SUBPIX_CRITERIA[2])
    raise ValueError("unsupported calibration board type: {}".format(board_type))


def _opencv_corners_to_aprilgrid_datum(
    corners: np.ndarray, corner_datum: str
) -> np.ndarray:
    """Map decoded OpenCV marker corners to the one Kalibr board datum.

    Production artwork is ID0 lower-left with ids increasing +X then +Y and
    each raw OpenCV marker rotated 180 degrees. OpenCV consequently reports
    physical BR, BL, TL, TR; the Kalibr object model is BL, BR, TR, TL.
    """
    if str(corner_datum) != APRILGRID_CORNER_DATUM:
        raise ValueError("unsupported AprilGrid corner datum: {}".format(corner_datum))
    decoded = np.asarray(corners, dtype=np.float32).reshape(4, 2)
    return decoded[np.asarray(_APRILGRID_OPENCV_TO_KALIBR_CORNER_ORDER)]


def board_object_points(board_size: Sequence[int], square: float) -> np.ndarray:
    """3D corner grid (Z=0) for a (cols, rows) interior-corner board."""
    cols, rows = int(board_size[0]), int(board_size[1])
    grid = np.zeros((cols * rows, 3), dtype=np.float32)
    grid[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2)
    grid *= float(square)
    return grid


def detect_board(
    gray: np.ndarray,
    board_size: Sequence[int],
    maximum_width: int = 960,
    *,
    board_type: str = "checkerboard",
    square: float = 0.2,
    tag_spacing: float = 0.0,
    tag_family: str = "tag36h11",
    start_id: int = 0,
    min_tags: int = 6,
    require_refinement: bool = True,
) -> Optional[BoardDetection]:
    """Detect one calibration board and return image/object correspondences."""
    if gray.ndim != 2:
        raise ValueError("detect_board expects a single-channel image")
    kind = str(board_type or "checkerboard").strip().lower()
    if kind == "aprilgrid":
        return detect_aprilgrid(
            gray,
            board_size,
            square=square,
            tag_spacing=tag_spacing,
            tag_family=tag_family,
            start_id=start_id,
            min_tags=min_tags,
            maximum_width=maximum_width,
            require_refinement=require_refinement,
        )
    if kind not in ("checkerboard", "chessboard"):
        raise ValueError("unsupported calibration board type: {}".format(board_type))
    height, width = gray.shape[:2]
    scale = 1.0
    search = gray
    if width > maximum_width:
        scale = float(maximum_width) / float(width)
        search = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    found, corners = cv2.findChessboardCorners(search, tuple(board_size), _DETECT_FLAGS)
    if not found:
        return None
    corners = (corners / scale).astype(np.float32)
    corners = cv2.cornerSubPix(gray, corners, (5, 5), (-1, -1), _SUBPIX_CRITERIA)
    return BoardDetection(
        image_points=corners,
        object_points=board_object_points(board_size, square),
        coverage=_coverage_params(corners, board_size, width, height),
    )


def _detect_aruco_markers(image: np.ndarray, dictionary: Any):
    if not hasattr(cv2.aruco, "DetectorParameters") or not hasattr(
        cv2.aruco, "ArucoDetector"
    ):
        raise CalibrationError(
            "OpenCV contrib 4.12 ArUcoDetector contract is unavailable"
        )
    parameters = cv2.aruco.DetectorParameters()
    parameters.markerBorderBits = _APRILGRID_MARKER_BORDER_BITS
    detector = cv2.aruco.ArucoDetector(dictionary, parameters)
    return detector.detectMarkers(image)


def aprilgrid_has_candidate_evidence(
    gray: np.ndarray,
    tag_family: str = "tag36h11",
    minimum_quads: int = 1,
) -> bool:
    """Return whether a low-resolution frame justifies a higher-resolution retry.

    One decoded marker or rejected quad is enough. A4 24 mm tags at arm's
    length occupy ~8 px on the VGA search plane and often yield fewer than
    six recovered quads even though the source JPEG still decodes the board.
    """
    if gray.ndim != 2 or minimum_quads < 1 or not hasattr(cv2, "aruco"):
        return False
    dictionary_name = _APRILTAG_DICTIONARIES.get(str(tag_family).strip().lower())
    if not dictionary_name or not hasattr(cv2.aruco, dictionary_name):
        return False
    dictionary = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, dictionary_name))
    _corners, ids, rejected = _detect_aruco_markers(gray, dictionary)
    decoded = 0 if ids is None else len(ids)
    rejected_count = 0 if rejected is None else len(rejected)
    return decoded >= minimum_quads or rejected_count >= minimum_quads


def refine_aprilgrid_calibration_corners(
    source_gray: np.ndarray,
    source_tag_corners: np.ndarray,
    source_tag_objects: np.ndarray,
    minimum_tags: int = 1,
    allow_subpix_fallback: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return paired, full-resolution AprilGrid corner correspondences.

    Continuous detection stays on the small search plane. Only a geometrically
    distinct sample pays for this source-resolution grayscale refinement. A
    single validity mask is applied to image and object coordinates; an
    unrefined or ambiguously ordered point is never mixed into a solve sample.
    """
    if source_gray.ndim != 2:
        raise ValueError("AprilGrid source refinement expects a grayscale image")
    if int(minimum_tags) < 1:
        raise ValueError("AprilGrid source refinement needs a positive minimum tag count")
    corners = np.asarray(source_tag_corners, dtype=np.float32).reshape(-1, 4, 2)
    objects = np.asarray(source_tag_objects, dtype=np.float32).reshape(-1, 4, 3)
    if len(corners) != len(objects) or not len(corners):
        raise ValueError("AprilGrid image/object tag correspondences do not match")
    height, width = source_gray.shape[:2]
    flat = corners.reshape(-1, 2)
    finite = np.all(np.isfinite(corners), axis=(1, 2)) & np.all(
        np.isfinite(objects), axis=(1, 2)
    )
    inside = (
        (flat[:, 0] >= _APRILGRID_MIN_BORDER_DISTANCE)
        & (flat[:, 0] < float(width) - _APRILGRID_MIN_BORDER_DISTANCE)
        & (flat[:, 1] >= _APRILGRID_MIN_BORDER_DISTANCE)
        & (flat[:, 1] < float(height) - _APRILGRID_MIN_BORDER_DISTANCE)
    ).reshape(-1, 4).all(axis=1)
    convex = np.asarray(
        [cv2.isContourConvex(tag.reshape(-1, 1, 2)) for tag in corners],
        dtype=bool,
    )
    tag_mask = finite & inside & convex
    if not bool(np.any(tag_mask)):
        raise CalibrationError("AprilGrid source frame has no refinable tag corners")

    refined_tags = []
    accepted_objects = []
    for raw_tag, object_tag in zip(corners[tag_mask], objects[tag_mask]):
        try:
            refined_tag = _refine_aprilgrid_quad_edges(source_gray, raw_tag)
        except (ValueError, np.linalg.LinAlgError, cv2.error):
            if not allow_subpix_fallback or not _aprilgrid_tag_region_has_contrast(
                source_gray, raw_tag
            ):
                continue
            try:
                seed = np.asarray(raw_tag, dtype=np.float32).reshape(-1, 1, 2)
                refined_tag = cv2.cornerSubPix(
                    source_gray, seed, (5, 5), (-1, -1), _SUBPIX_CRITERIA
                ).reshape(4, 2)
            except cv2.error:
                continue
        orientation_ok = (
            cv2.isContourConvex(refined_tag.reshape(-1, 1, 2))
            and np.sign(cv2.contourArea(raw_tag, oriented=True))
            == np.sign(cv2.contourArea(refined_tag, oriented=True))
        )
        if not orientation_ok:
            continue
        refined_tags.append(refined_tag)
        accepted_objects.append(object_tag)
    if len(refined_tags) < int(minimum_tags):
        raise CalibrationError(
            "AprilGrid source refinement retained fewer than {} complete tags".format(
                int(minimum_tags)
            )
        )
    return (
        np.asarray(refined_tags, dtype=np.float32).reshape(-1, 1, 2),
        np.asarray(accepted_objects, dtype=np.float32).reshape(-1, 3),
    )


def _aprilgrid_tag_region_has_contrast(gray: np.ndarray, quad: np.ndarray) -> bool:
    points = np.asarray(quad, dtype=np.float32).reshape(4, 2)
    x0, y0 = np.floor(points.min(axis=0)).astype(np.int32)
    x1, y1 = np.ceil(points.max(axis=0)).astype(np.int32)
    x0 = int(max(0, x0))
    y0 = int(max(0, y0))
    x1 = int(min(gray.shape[1], x1))
    y1 = int(min(gray.shape[0], y1))
    if x1 - x0 < 4 or y1 - y0 < 4:
        return False
    return float(np.ptp(gray[y0:y1, x0:x1].astype(np.float32))) >= (
        _APRILGRID_EDGE_MIN_MEDIAN_CONTRAST
    )


def localize_aprilgrid_source_corners(
    source_gray: np.ndarray,
    search_tag_corners: np.ndarray,
    search_tag_objects: np.ndarray,
    board_size: Sequence[int],
    square: float,
    tag_spacing: float,
    start_id: int,
    min_tags: int,
) -> Tuple[np.ndarray, np.ndarray, Tuple[float, float, float, float]]:
    """Localize search-plane AprilGrid corners on the source grayscale image.

    The continuous detector already recovered unique tag IDs. Re-running ArUco
    on the 4K JPEG can invent duplicate IDs and drop a usable frame. Map the
    search corners, refine them on the source plane, and keep the observation
    only when the lattice stays geometrically consistent.
    """
    height, width = source_gray.shape[:2]
    try:
        calibration_pixels, calibration_objects = refine_aprilgrid_calibration_corners(
            source_gray,
            search_tag_corners,
            search_tag_objects,
            minimum_tags=min_tags,
            allow_subpix_fallback=False,
        )
    except CalibrationError:
        calibration_pixels, calibration_objects = refine_aprilgrid_calibration_corners(
            source_gray,
            search_tag_corners,
            search_tag_objects,
            minimum_tags=min_tags,
            allow_subpix_fallback=True,
        )
    image_tags = np.asarray(calibration_pixels, dtype=np.float32).reshape(-1, 4, 2)
    object_tags = np.asarray(calibration_objects, dtype=np.float32).reshape(-1, 4, 3)
    first_id = int(start_id)
    last_id = first_id + int(board_size[0]) * int(board_size[1])
    tag_ids: List[int] = []
    for object_tag in object_tags:
        matching = [
            tag_id
            for tag_id in range(first_id, last_id)
            if np.array_equal(
                object_tag,
                aprilgrid_tag_object_points(
                    board_size, square, tag_spacing, start_id, tag_id
                ),
            )
        ]
        if len(matching) != 1:
            raise CalibrationError("AprilGrid refined tags lost their lattice identity")
        tag_ids.append(matching[0])
    if not _strict_aprilgrid_observation(
        image_tags,
        object_tags,
        tag_ids,
        board_size,
        square,
        tag_spacing,
        start_id,
        observation_uncertainty_px("aprilgrid"),
    ):
        raise CalibrationError("AprilGrid observation failed geometric consistency")
    return (
        calibration_pixels,
        calibration_objects,
        _aprilgrid_coverage(calibration_pixels, calibration_objects, width, height),
    )


def _sample_gray_bilinear(gray: np.ndarray, points: np.ndarray) -> np.ndarray:
    coordinates = np.asarray(points, dtype=np.float32).reshape(-1, 2)
    return cv2.remap(
        gray,
        coordinates[:, 0].reshape(-1, 1),
        coordinates[:, 1].reshape(-1, 1),
        cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    ).reshape(-1)


def _fit_aprilgrid_edge(
    gray: np.ndarray,
    first: np.ndarray,
    second: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    edge = np.asarray(second, dtype=np.float64) - np.asarray(first, dtype=np.float64)
    length = float(np.linalg.norm(edge))
    if length < _APRILGRID_MIN_EDGE_LENGTH_PX:
        raise ValueError("AprilGrid edge is too short for line refinement")
    tangent = edge / length
    normal = np.asarray((-tangent[1], tangent[0]), dtype=np.float64)
    count = max(10, min(64, int(length / 3.0)))
    bases = np.asarray(first, dtype=np.float64)[None, :] + np.linspace(
        0.12, 0.88, count
    )[:, None] * edge[None, :]
    offsets = np.arange(
        -_APRILGRID_EDGE_SEARCH_RADIUS_PX,
        _APRILGRID_EDGE_SEARCH_RADIUS_PX + 1e-6,
        _APRILGRID_EDGE_SAMPLE_STEP_PX,
        dtype=np.float64,
    )
    probes = bases[:, None, :] + offsets[None, :, None] * normal[None, None, :]
    intensities = _sample_gray_bilinear(gray, probes).reshape(count, len(offsets)).astype(
        np.float64
    )
    # All four ArUco corners are ordered around the tag, so ``normal`` crosses
    # the same outer-border polarity along the edge. Estimate the two plateaus
    # and interpolate their 50% crossing. This selects one physical transition
    # instead of averaging an unrelated nearby texture, and avoids the
    # quarter-pixel first-index bias of argmax on equal JPEG gradient bins.
    outside = np.median(intensities[:, :3], axis=1)
    inside = np.median(intensities[:, -3:], axis=1)
    aggregate_contrast = float(np.median(outside - inside))
    polarity = 1.0 if aggregate_contrast >= 0.0 else -1.0
    profile_contrast = polarity * (outside - inside)
    if float(np.median(profile_contrast)) < _APRILGRID_EDGE_MIN_MEDIAN_CONTRAST:
        raise ValueError("AprilGrid edge contrast is too weak for line refinement")
    oriented = polarity * intensities
    targets = 0.5 * polarity * (outside + inside)
    edge_offsets = []
    edge_bases = []
    for base, values, target, contrast in zip(bases, oriented, targets, profile_contrast):
        if contrast < _APRILGRID_EDGE_MIN_MEDIAN_CONTRAST * 0.5:
            continue
        drops = values[:-1] - values[1:]
        crossing = (
            (values[:-1] >= target)
            & (values[1:] <= target)
            & (drops >= _APRILGRID_EDGE_MIN_PROFILE_GRADIENT)
        )
        # A transition touching the search boundary is not localized; the raw
        # detector seed needs to be close enough that both plateaus are visible.
        crossing[0] = False
        crossing[-1] = False
        candidates = np.flatnonzero(crossing)
        if not len(candidates):
            continue
        index = int(candidates[np.argmax(drops[candidates])])
        alpha = float((values[index] - target) / drops[index])
        edge_offsets.append(offsets[index] + alpha * _APRILGRID_EDGE_SAMPLE_STEP_PX)
        edge_bases.append(base)
    if len(edge_offsets) < max(8, int(math.ceil(0.75 * count))):
        raise ValueError("AprilGrid edge has too few consistent contrast profiles")
    edge_points = (
        np.asarray(edge_bases, dtype=np.float64)
        + np.asarray(edge_offsets, dtype=np.float64)[:, None] * normal[None, :]
    )
    vx, vy, x0, y0 = cv2.fitLine(
        edge_points.astype(np.float32), cv2.DIST_HUBER, 0.0, 0.01, 0.01
    ).reshape(-1)
    direction = np.asarray((vx, vy), dtype=np.float64)
    direction_norm = float(np.linalg.norm(direction))
    if direction_norm <= 1e-12:
        raise ValueError("AprilGrid refined edge line is degenerate")
    direction /= direction_norm
    line_normal = np.asarray((-direction[1], direction[0]), dtype=np.float64)
    residuals = np.matmul(
        edge_points - np.asarray((x0, y0), dtype=np.float64), line_normal
    )
    line_rms = float(np.sqrt(np.mean(np.square(residuals))))
    line_p90 = float(np.percentile(np.abs(residuals), 90.0))
    if (
        line_rms > _APRILGRID_EDGE_MAX_LINE_RMS_PX
        or line_p90 > _APRILGRID_EDGE_MAX_LINE_P90_PX
    ):
        raise ValueError("AprilGrid edge is not straight enough for line refinement")
    return np.asarray((x0, y0), dtype=np.float64), direction


def _line_intersection(
    first_point: np.ndarray,
    first_direction: np.ndarray,
    second_point: np.ndarray,
    second_direction: np.ndarray,
) -> np.ndarray:
    system = np.column_stack((first_direction, -second_direction))
    determinant = float(np.linalg.det(system))
    if abs(determinant) <= 1e-6:
        raise ValueError("AprilGrid refined edge lines are nearly parallel")
    parameters = np.linalg.solve(system, second_point - first_point)
    return first_point + first_direction * float(parameters[0])


def _refine_aprilgrid_quad_edges(gray: np.ndarray, quad: np.ndarray) -> np.ndarray:
    """Refine a tag as four gradient-fit lines and intersect adjacent edges.

    Generic cornerSubPix is biased by the large binary AprilTag border. Fitting
    each physical outer edge over its interior span uses many gradient samples,
    stays projectively valid, and retains four independent correspondences.
    """
    corners = np.asarray(quad, dtype=np.float64).reshape(4, 2)
    edge_lengths = np.asarray([
        np.linalg.norm(corners[(index + 1) % 4] - corners[index])
        for index in range(4)
    ], dtype=np.float64)
    minimum_edge_length = float(np.min(edge_lengths))
    if minimum_edge_length < _APRILGRID_MIN_EDGE_LENGTH_PX:
        raise ValueError("AprilGrid edge is too short for line refinement")
    lines = [
        _fit_aprilgrid_edge(gray, corners[index], corners[(index + 1) % 4])
        for index in range(4)
    ]
    # Adjacent search bands must separate over at least one raw edge. This gives
    # sin(theta) > 2r/L and prevents a numerically valid but ill-conditioned
    # intersection from sending the corner far away.
    minimum_adjacent_sine = min(
        1.0,
        2.0 * _APRILGRID_EDGE_SEARCH_RADIUS_PX / minimum_edge_length,
    )
    adjacent_cosines = []
    for index in range(4):
        previous_direction = lines[index - 1][1]
        direction = lines[index][1]
        adjacent_sine = abs(float(np.linalg.det(
            np.stack((previous_direction, direction), axis=1)
        )))
        if adjacent_sine <= minimum_adjacent_sine:
            raise ValueError("AprilGrid adjacent refined edges are too nearly parallel")
        adjacent_cosines.append(abs(float(np.dot(previous_direction, direction))))
    refined = np.asarray([
        _line_intersection(*lines[index - 1], *lines[index])
        for index in range(4)
    ], dtype=np.float32)
    if not bool(np.all(np.isfinite(refined))):
        raise ValueError("AprilGrid edge refinement produced non-finite corners")
    height, width = gray.shape[:2]
    if not bool(np.all(
        (refined[:, 0] >= 0.0)
        & (refined[:, 0] <= float(width - 1))
        & (refined[:, 1] >= 0.0)
        & (refined[:, 1] <= float(height - 1))
    )):
        raise ValueError("AprilGrid refined corners leave the source image")
    # Two fitted line offsets, each bounded by the normal search radius, move
    # their intersection by at most ||A^-1|| * sqrt(2)r. For unit line normals,
    # sigma_min(A)=sqrt(1-|cos(theta)|).
    displacements = np.linalg.norm(refined.astype(np.float64) - corners, axis=1)
    maximum_displacements = np.asarray([
        math.sqrt(2.0) * _APRILGRID_EDGE_SEARCH_RADIUS_PX
        / math.sqrt(max(np.finfo(np.float64).eps, 1.0 - cosine))
        for cosine in adjacent_cosines
    ])
    if bool(np.any(displacements > maximum_displacements)):
        raise ValueError("AprilGrid refined corner exceeds its search-band geometry")
    raw_area = abs(float(cv2.contourArea(corners.astype(np.float32))))
    refined_area = abs(float(cv2.contourArea(refined)))
    if raw_area <= np.finfo(np.float64).eps:
        raise ValueError("AprilGrid raw quad has degenerate area")
    edge_band_fraction = (
        2.0 * _APRILGRID_EDGE_SEARCH_RADIUS_PX / minimum_edge_length
    )
    area_ratio = refined_area / raw_area
    if not (
        (1.0 - edge_band_fraction) ** 2
        <= area_ratio
        <= (1.0 + edge_band_fraction) ** 2
    ):
        raise ValueError("AprilGrid refined quad area exceeds its edge-search band")
    if (
        not cv2.isContourConvex(refined.reshape(-1, 1, 2))
        or np.sign(cv2.contourArea(corners.astype(np.float32), oriented=True))
        != np.sign(cv2.contourArea(refined, oriented=True))
    ):
        raise ValueError("AprilGrid refined quad changed orientation")
    return refined


def _project_points(homography: np.ndarray, points: np.ndarray) -> np.ndarray:
    return cv2.perspectiveTransform(
        np.asarray(points, dtype=np.float32).reshape(-1, 1, 2), homography
    ).reshape(-1, 2)


def aprilgrid_tag_object_points(
    board_size: Sequence[int],
    tag_size: float,
    tag_spacing: float,
    start_id: int,
    tag_id: int,
) -> Optional[np.ndarray]:
    """Kalibr board corners BL, BR, TR, TL with ID0 at the lower-left."""
    cols, rows = int(board_size[0]), int(board_size[1])
    index = int(tag_id) - int(start_id)
    if index < 0 or index >= cols * rows:
        return None
    col = index % cols
    row = index // cols
    size = float(tag_size)
    pitch = size + float(tag_spacing)
    x0 = col * pitch
    y0 = row * pitch
    return np.array(
        (
            (x0, y0, 0.0),
            (x0 + size, y0, 0.0),
            (x0 + size, y0 + size, 0.0),
            (x0, y0 + size, 0.0),
        ),
        dtype=np.float32,
    )


def _aprilgrid_tag_lattice_is_two_dimensional(
    tag_ids: Sequence[int], board_size: Sequence[int], start_id: int
) -> bool:
    cols, rows = int(board_size[0]), int(board_size[1])
    first = int(start_id)
    ids = [int(value) for value in tag_ids]
    if (
        cols < 2
        or rows < 2
        or len(ids) < 4
        or len(set(ids)) != len(ids)
        or any(value < first or value >= first + cols * rows for value in ids)
    ):
        return False
    lattice = np.asarray(
        [((value - first) % cols, (value - first) // cols) for value in ids],
        dtype=np.float64,
    )
    return int(np.linalg.matrix_rank(lattice - np.mean(lattice, axis=0))) == 2


def _strict_aprilgrid_observation(
    image_points: Sequence[np.ndarray],
    object_points: Sequence[np.ndarray],
    tag_ids: Sequence[int],
    board_size: Sequence[int],
    tag_size: float,
    tag_spacing: float,
    start_id: int,
    detection_uncertainty_px: float,
) -> bool:
    """Validate one standard-datum AprilGrid observation without repairing it.

    IDs define one and only one board lattice, and decoded corner zero must
    already correspond to object corner zero. The checks below never permute a
    tag. Assignment bounds come from half the observed lattice separation (the
    nearest-neighbour ambiguity boundary) and the active detector uncertainty,
    rather than a fixed pixel residual chosen for a particular camera.
    """
    if len(image_points) != len(object_points) or len(image_points) != len(tag_ids):
        return False
    if not _aprilgrid_tag_lattice_is_two_dimensional(
        tag_ids, board_size, start_id
    ):
        return False
    try:
        image_tags = np.asarray(image_points, dtype=np.float64).reshape(-1, 4, 2)
        object_tags = np.asarray(object_points, dtype=np.float64).reshape(-1, 4, 3)
    except (TypeError, ValueError):
        return False
    if (
        len(image_tags) != len(tag_ids)
        or len(object_tags) != len(tag_ids)
        or not np.all(np.isfinite(image_tags))
        or not np.all(np.isfinite(object_tags))
    ):
        return False
    numeric_tolerance = (
        np.finfo(np.float32).eps
        * max(1.0, float(tag_size), float(tag_spacing))
        * 8.0
    )
    for marker_id, observed_object in zip(tag_ids, object_tags):
        expected_object = aprilgrid_tag_object_points(
            board_size, tag_size, tag_spacing, start_id, int(marker_id)
        )
        if expected_object is None or not np.allclose(
            observed_object,
            expected_object,
            rtol=0.0,
            atol=numeric_tolerance,
        ):
            return False

    object_centers = np.mean(object_tags[:, :, :2], axis=1)
    image_centers = np.mean(image_tags, axis=1)
    uncertainty = max(
        float(detection_uncertainty_px),
        np.finfo(np.float64).eps
        * max(1.0, float(np.max(np.abs(image_centers)))),
    )
    centered_image = image_centers - np.mean(image_centers, axis=0)
    try:
        image_singular_values = np.linalg.svd(centered_image, compute_uv=False)
    except np.linalg.LinAlgError:
        return False
    if (
        len(image_singular_values) < 2
        or float(image_singular_values[-1])
        <= uncertainty * math.sqrt(float(len(image_centers)))
    ):
        return False
    try:
        homography, _mask = cv2.findHomography(
            object_centers.astype(np.float32),
            image_centers.astype(np.float32),
            method=0,
        )
    except cv2.error:
        return False
    if homography is None or not np.all(np.isfinite(homography)):
        return False
    try:
        predicted_centers = _project_points(homography, object_centers)
    except cv2.error:
        return False
    pair_distances = np.linalg.norm(
        predicted_centers[:, None, :] - predicted_centers[None, :, :], axis=2
    )
    np.fill_diagonal(pair_distances, np.inf)
    minimum_lattice_separation = float(np.min(pair_distances))
    assignment_margin = 0.5 * minimum_lattice_separation - uncertainty
    if not np.isfinite(assignment_margin) or assignment_margin <= 0.0:
        return False
    center_residuals = np.linalg.norm(predicted_centers - image_centers, axis=1)
    if bool(np.any(center_residuals > assignment_margin)):
        return False

    for measured, tag_object in zip(image_tags, object_tags):
        try:
            predicted = _project_points(homography, tag_object[:, :2])
        except cv2.error:
            return False
        distances = np.linalg.norm(
            measured[:, None, :] - predicted[None, :, :], axis=2
        )
        # Canonical index i must be the unique nearest prediction for measured
        # corner i in both directions. This rejects rotated/reflected D4 corner
        # order instead of silently selecting whichever permutation fits best.
        if not np.array_equal(np.argmin(distances, axis=1), np.arange(4)):
            return False
        if not np.array_equal(np.argmin(distances, axis=0), np.arange(4)):
            return False
        for index in range(4):
            other = np.delete(distances[index], index)
            if float(distances[index, index]) + 2.0 * uncertainty >= float(
                np.min(other)
            ):
                return False
        predicted_area = float(
            cv2.contourArea(predicted.astype(np.float32), oriented=True)
        )
        measured_area = float(
            cv2.contourArea(measured.astype(np.float32), oriented=True)
        )
        if predicted_area == 0.0 or measured_area == 0.0:
            return False
        if math.copysign(1.0, predicted_area) != math.copysign(1.0, measured_area):
            return False
    return True


def detect_aprilgrid(
    gray: np.ndarray,
    board_size: Sequence[int],
    *,
    square: float,
    tag_spacing: float,
    tag_family: str = "tag36h11",
    start_id: int = 0,
    min_tags: int = 6,
    maximum_width: int = 960,
    corner_datum: str = APRILGRID_CORNER_DATUM,
    require_refinement: bool = True,
) -> Optional[BoardDetection]:
    """Detect a Kalibr-style AprilGrid and return the visible tag corners.

    Search planes may pass ``require_refinement=False`` so a decoded lattice
    can annotate the live result even when 20 px edge fitting is impossible
    at VGA. Sample admission still requires source-resolution refinement.
    """
    if float(square) <= 0.0:
        raise ValueError("AprilGrid tag size must be positive")
    if float(tag_spacing) < 0.0:
        raise ValueError("AprilGrid tag spacing must be non-negative")
    if int(min_tags) < 4:
        raise ValueError("AprilGrid detection needs at least four tags")
    if str(corner_datum) != APRILGRID_CORNER_DATUM:
        raise ValueError("unsupported AprilGrid corner datum: {}".format(corner_datum))
    if not hasattr(cv2, "aruco"):
        raise CalibrationError("OpenCV was built without the aruco module")
    dictionary_name = _APRILTAG_DICTIONARIES.get(str(tag_family).strip().lower())
    if not dictionary_name or not hasattr(cv2.aruco, dictionary_name):
        raise CalibrationError("unsupported AprilTag family: {}".format(tag_family))
    height, width = gray.shape[:2]
    scale = 1.0
    search = gray
    if width > maximum_width:
        scale = float(maximum_width) / float(width)
        search = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    dictionary = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, dictionary_name))
    corners, ids, _rejected = _detect_aruco_markers(search, dictionary)

    decoded_ids = [] if ids is None else [int(value) for value in ids.reshape(-1)]
    first_id = int(start_id)
    last_id = first_id + int(board_size[0]) * int(board_size[1])
    if (
        len(set(decoded_ids)) != len(decoded_ids)
        or any(value < first_id or value >= last_id for value in decoded_ids)
    ):
        return None
    # Do not upscale this search image. AprilTag payload bits discarded by the
    # low-resolution decode cannot be reconstructed by interpolation. The
    # service uses rejected quads as evidence and retries by decoding the
    # original JPEG into a larger, still-bounded working plane.
    image_points: List[np.ndarray] = []
    object_points: List[np.ndarray] = []
    tag_ids: List[int] = []
    if ids is not None:
        for marker_corners, marker_id in zip(corners, ids.reshape(-1)):
            obj = aprilgrid_tag_object_points(
                board_size, square, tag_spacing, start_id, int(marker_id)
            )
            if obj is None:
                continue
            image = (
                np.asarray(marker_corners, dtype=np.float32).reshape(4, 2) / scale
            ).astype(np.float32)
            image = _opencv_corners_to_aprilgrid_datum(image, corner_datum)
            image_points.append(image)
            object_points.append(obj)
            tag_ids.append(int(marker_id))
    if len(image_points) < int(min_tags):
        return None
    if not _aprilgrid_tag_lattice_is_two_dimensional(
        tag_ids, board_size, start_id
    ):
        return None
    tag_image_points = np.asarray(image_points, dtype=np.float32).reshape(-1, 4, 2)
    tag_object_points = np.asarray(object_points, dtype=np.float32).reshape(-1, 4, 3)
    raw_pixels = tag_image_points.reshape(-1, 1, 2)
    objects = tag_object_points.reshape(-1, 3)
    calibration_pixels = None
    calibration_objects = None
    try:
        calibration_pixels, calibration_objects = refine_aprilgrid_calibration_corners(
            gray, tag_image_points, tag_object_points, minimum_tags=min_tags
        )
        calibration_image_tags = np.asarray(
            calibration_pixels, dtype=np.float32
        ).reshape(-1, 4, 2)
        calibration_object_tags = np.asarray(
            calibration_objects, dtype=np.float32
        ).reshape(-1, 4, 3)
        calibration_tag_ids: List[int] = []
        for refined_object in calibration_object_tags:
            matching = [
                index
                for index, raw_object in enumerate(tag_object_points)
                if np.array_equal(refined_object, raw_object)
            ]
            if len(matching) != 1:
                raise CalibrationError(
                    "AprilGrid refined tags lost their lattice identity"
                )
            calibration_tag_ids.append(tag_ids[matching[0]])
        if not _strict_aprilgrid_observation(
            calibration_image_tags,
            calibration_object_tags,
            calibration_tag_ids,
            board_size,
            square,
            tag_spacing,
            start_id,
            observation_uncertainty_px("aprilgrid"),
        ):
            raise CalibrationError(
                "AprilGrid observation failed geometric consistency"
            )
    except CalibrationError:
        if require_refinement:
            return None
        calibration_pixels = None
        calibration_objects = None
    # Raw decoded corners remain annotation-only. Calibration and coverage use
    # the exact same complete-tag refinement mask when it exists; search planes
    # may keep decoded corners for live annotation without admitting them.
    pixels = raw_pixels.copy()
    flat = raw_pixels.reshape(-1, 2)
    valid = (
        (flat[:, 0] >= 4.0)
        & (flat[:, 0] < float(width - 4))
        & (flat[:, 1] >= 4.0)
        & (flat[:, 1] < float(height - 4))
    )
    if bool(np.any(valid)):
        pixels[valid] = cv2.cornerSubPix(
            gray, raw_pixels[valid].copy(), (3, 3), (-1, -1), _SUBPIX_CRITERIA
        )
    coverage_pixels = (
        calibration_pixels if calibration_pixels is not None else pixels
    )
    coverage_objects = (
        calibration_objects if calibration_objects is not None else objects
    )
    return BoardDetection(
        image_points=pixels,
        object_points=objects,
        coverage=_aprilgrid_coverage(
            coverage_pixels,
            coverage_objects,
            width,
            height,
        ),
        calibration_image_points=calibration_pixels,
        calibration_object_points=calibration_objects,
    )


def _aprilgrid_coverage(
    pixels: np.ndarray,
    objects: np.ndarray,
    width: int,
    height: int,
) -> Tuple[float, float, float, float]:
    """Describe actual solve support and its local projective obliqueness.

    X/Y/Size come from the visible refined-corner hull. The fourth component
    measures non-conformal local homography deformation on those same object
    points. A 2-D roll is a similarity and therefore contributes zero; no
    invisible full-board boundary is synthesized.
    """
    hull_coverage = _coverage_params_from_points(pixels, width, height)
    object_xy = np.asarray(objects, dtype=np.float32).reshape(-1, 3)[:, :2]
    image_xy = np.asarray(pixels, dtype=np.float32).reshape(-1, 2)
    try:
        homography, _mask = cv2.findHomography(object_xy, image_xy, method=0)
    except cv2.error:
        homography = None
    obliqueness = 0.0
    if homography is not None and np.all(np.isfinite(homography)):
        local_values = []
        h = np.asarray(homography, dtype=np.float64)
        for x_value, y_value in object_xy.astype(np.float64):
            denominator = h[2, 0] * x_value + h[2, 1] * y_value + h[2, 2]
            if abs(denominator) <= np.finfo(np.float64).eps:
                continue
            u_value = (
                h[0, 0] * x_value + h[0, 1] * y_value + h[0, 2]
            ) / denominator
            v_value = (
                h[1, 0] * x_value + h[1, 1] * y_value + h[1, 2]
            ) / denominator
            jacobian = np.asarray(
                (
                    (
                        (h[0, 0] - u_value * h[2, 0]) / denominator,
                        (h[0, 1] - u_value * h[2, 1]) / denominator,
                    ),
                    (
                        (h[1, 0] - v_value * h[2, 0]) / denominator,
                        (h[1, 1] - v_value * h[2, 1]) / denominator,
                    ),
                ),
                dtype=np.float64,
            )
            singular_values = np.linalg.svd(jacobian, compute_uv=False)
            if (
                len(singular_values) == 2
                and np.all(np.isfinite(singular_values))
                and singular_values[0] > 0.0
            ):
                local_values.append(
                    1.0 - float(singular_values[-1] / singular_values[0])
                )
        if local_values:
            obliqueness = min(1.0, max(0.0, float(max(local_values))))
    return (
        float(hull_coverage[0]),
        float(hull_coverage[1]),
        float(hull_coverage[2]),
        obliqueness,
    )


def _coverage_params_from_points(
    corners: np.ndarray, width: int, height: int
) -> Tuple[float, float, float, float]:
    """X/Y/Size/Skew from an unordered point set (partial AprilGrid views)."""
    points = np.asarray(corners, dtype=np.float32).reshape(-1, 2)
    if len(points) < 4:
        return (0.5, 0.5, 0.0, 0.0)
    mean_x = float(np.mean(points[:, 0]))
    mean_y = float(np.mean(points[:, 1]))
    span_x = float(np.max(points[:, 0]) - np.min(points[:, 0]))
    span_y = float(np.max(points[:, 1]) - np.min(points[:, 1]))
    hull = cv2.convexHull(points)
    area = float(cv2.contourArea(hull))
    rect = cv2.minAreaRect(points)
    angle = abs(float(rect[2]))
    if angle > 45.0:
        angle = 90.0 - angle
    # Match the checkerboard/ROS coverage meaning: zero and one represent the
    # board touching the two image edges, not its centroid leaving the frame.
    # Without the visible-span correction a large AprilGrid had to be mostly
    # cropped before X/Y could ever reach the 0.70 range gate.
    p_x = min(1.0, max(0.0, (mean_x - span_x / 2.0) / max(1.0, float(width) - span_x)))
    p_y = min(1.0, max(0.0, (mean_y - span_y / 2.0) / max(1.0, float(height) - span_y)))
    p_size = math.sqrt(area / float(width * height)) if area > 0 else 0.0
    p_skew = min(1.0, angle / 45.0)
    return (p_x, p_y, p_size, p_skew)


def _coverage_params(
    corners: np.ndarray, board_size: Sequence[int], width: int, height: int
) -> Tuple[float, float, float, float]:
    columns = int(board_size[0])
    upper_left = corners[0, 0]
    upper_right = corners[columns - 1, 0]
    lower_right = corners[-1, 0]
    lower_left = corners[-columns, 0]
    edge_a = upper_right - upper_left
    edge_b = lower_right - upper_right
    edge_c = lower_left - lower_right
    diagonal_p = edge_b + edge_c
    diagonal_q = edge_a + edge_b
    area = abs(diagonal_p[0] * diagonal_q[1] - diagonal_p[1] * diagonal_q[0]) / 2.0
    border = math.sqrt(area) if area > 0 else 0.0
    mean_x = float(np.mean(corners[:, :, 0]))
    mean_y = float(np.mean(corners[:, :, 1]))
    p_x = min(1.0, max(0.0, (mean_x - border / 2.0) / max(1e-6, width - border)))
    p_y = min(1.0, max(0.0, (mean_y - border / 2.0) / max(1e-6, height - border)))
    p_size = math.sqrt(area / float(width * height)) if area > 0 else 0.0
    vector_a = upper_left - upper_right
    vector_b = lower_right - upper_right
    norm = float(np.linalg.norm(vector_a)) * float(np.linalg.norm(vector_b))
    cosine = float(np.dot(vector_a, vector_b)) / norm if norm > 0 else 0.0
    angle = math.acos(min(1.0, max(-1.0, cosine)))
    p_skew = min(1.0, 2.0 * abs(math.pi / 2.0 - angle))
    return (p_x, p_y, p_size, p_skew)


def coverage(
    samples: Sequence[Sequence[float]], ranges: Sequence[float] = PARAM_RANGES
) -> Tuple[List[Dict[str, Any]], bool]:
    """Return ([{label, progress}]x4, goodenough), mirroring compute_goodenough."""
    if not samples:
        return [{"label": name, "progress": 0.0} for name in PARAM_NAMES], False
    minimum = [min(sample[i] for sample in samples) for i in range(4)]
    maximum = [max(sample[i] for sample in samples) for i in range(4)]
    minimum[2] = 0.0  # size / skew are rewarded by their maximum only
    minimum[3] = 0.0
    progress = [min(1.0, (hi - lo) / rng) for lo, hi, rng in zip(minimum, maximum, ranges)]
    # Repeated observations never substitute for geometric diversity.
    goodenough = all(value >= 1.0 for value in progress)
    bars = [{"label": name, "progress": float(value)} for name, value in zip(PARAM_NAMES, progress)]
    return bars, goodenough


def next_view_guidance(
    samples: Sequence[Sequence[float]], ranges: Sequence[float] = PARAM_RANGES,
    coverage_bars: Optional[Sequence[Mapping[str, Any]]] = None,
) -> Dict[str, Any]:
    """Describe the single view that expands the weakest coverage axis most.

    X and Y need samples on both sides of the image, while Size and Skew are
    rewarded by their largest observed value. AprilGrid Skew is out-of-plane
    tilt (``direction='tilt'``), not in-plane roll. Returning a small semantic
    document keeps presentation/localization in the WebUI and gives a physical
    operator a stable direction based on the whole sample history rather than
    whichever frame happened to arrive last.
    """
    if coverage_bars is None:
        bars, _goodenough = coverage(samples, ranges)
    else:
        bars = [
            {"label": name, "progress": float(item["progress"])}
            for name, item in zip(PARAM_NAMES, coverage_bars)
            if item.get("label") == name
        ]
        if len(bars) != len(PARAM_NAMES):
            raise ValueError("coverage_bars must contain ordered X/Y/Size/Skew entries")
    complete = bool(samples) and all(bar["progress"] >= 1.0 for bar in bars)
    if complete:
        return {
            "complete": True,
            "dimension": None,
            "direction": "complete",
            "progress": 1.0,
        }
    if not samples:
        return {
            "complete": False,
            "dimension": None,
            "direction": "center",
            "progress": 0.0,
        }

    index = min(range(len(bars)), key=lambda candidate: bars[candidate]["progress"])
    minimum = min(float(sample[index]) for sample in samples)
    maximum = max(float(sample[index]) for sample in samples)
    dimension = PARAM_NAMES[index]
    if dimension == "X":
        direction = "left" if minimum >= 1.0 - maximum else "right"
    elif dimension == "Y":
        direction = "top" if minimum >= 1.0 - maximum else "bottom"
    elif dimension == "Size":
        direction = "closer"
    else:
        direction = "tilt"
    return {
        "complete": False,
        "dimension": dimension,
        "direction": direction,
        "progress": float(bars[index]["progress"]),
    }


def _prepare_calibration_observations(
    image_points: Sequence[np.ndarray],
    board_size: Sequence[int],
    square: float,
    image_size: Sequence[int],
    object_points: Optional[Sequence[np.ndarray]],
) -> Tuple[List[np.ndarray], List[np.ndarray], Tuple[int, int]]:
    if object_points is None:
        board = board_object_points(board_size, square)
        object_points = [board for _ in image_points]
    if len(object_points) != len(image_points):
        raise CalibrationError("each sample must provide matching image and object points")
    if len(image_size) != 2:
        raise CalibrationError("image size must contain width and height")
    size = (int(image_size[0]), int(image_size[1]))
    if size[0] <= 0 or size[1] <= 0:
        raise CalibrationError("image width and height must be positive")

    prepared_object: List[np.ndarray] = []
    corners: List[np.ndarray] = []
    for image, obj in zip(image_points, object_points):
        try:
            pixels = np.asarray(image, dtype=np.float32).reshape(-1, 1, 2)
            world = np.asarray(obj, dtype=np.float32).reshape(-1, 3)
        except (TypeError, ValueError) as error:
            raise CalibrationError(
                "calibration correspondences have invalid dimensions"
            ) from error
        if len(pixels) != len(world) or len(pixels) < 4:
            raise CalibrationError("each sample must contain at least four corresponding points")
        if not np.all(np.isfinite(pixels)) or not np.all(np.isfinite(world)):
            raise CalibrationError("calibration correspondences must be finite")
        corners.append(pixels)
        prepared_object.append(world)
    return prepared_object, corners, size


def _run_extended_calibration(
    object_points: Sequence[np.ndarray],
    image_points: Sequence[np.ndarray],
    image_size: Tuple[int, int],
) -> _ExtendedCalibration:
    try:
        output = cv2.calibrateCameraExtended(
            object_points,
            image_points,
            image_size,
            None,
            None,
            flags=0,
            criteria=_CALIBRATION_CRITERIA,
        )
    except cv2.error as error:
        raise CalibrationError("OpenCV intrinsic optimization failed: {}".format(error)) from error
    if not isinstance(output, tuple) or len(output) != 8:
        raise CalibrationError("OpenCV intrinsic optimization returned an invalid result")
    (
        rms,
        camera_matrix,
        distortion,
        rotation_vectors,
        translation_vectors,
        intrinsic_standard_deviations,
        extrinsic_standard_deviations,
        per_view_errors,
    ) = output

    camera_matrix = np.asarray(camera_matrix, dtype=np.float64)
    distortion = np.asarray(distortion, dtype=np.float64).reshape(-1)
    intrinsic_standard_deviations = np.asarray(
        intrinsic_standard_deviations, dtype=np.float64
    ).reshape(-1)
    extrinsic_standard_deviations = np.asarray(
        extrinsic_standard_deviations, dtype=np.float64
    ).reshape(-1)
    per_view_errors = np.asarray(per_view_errors, dtype=np.float64).reshape(-1)
    rotations = tuple(
        np.asarray(vector, dtype=np.float64).reshape(-1) for vector in rotation_vectors
    )
    translations = tuple(
        np.asarray(vector, dtype=np.float64).reshape(-1) for vector in translation_vectors
    )
    sample_count = len(image_points)
    if (
        not np.isfinite(float(rms))
        or not np.all(np.isfinite(camera_matrix))
        or not np.all(np.isfinite(distortion))
        or not np.all(np.isfinite(intrinsic_standard_deviations))
        or not np.all(np.isfinite(extrinsic_standard_deviations))
        or not np.all(np.isfinite(per_view_errors))
        or any(not np.all(np.isfinite(vector)) for vector in rotations)
        or any(not np.all(np.isfinite(vector)) for vector in translations)
    ):
        raise CalibrationError("calibration produced non-finite optimizer diagnostics")
    if float(rms) < 0.0 or np.any(per_view_errors < 0.0):
        raise CalibrationError("calibration produced a negative reprojection error")
    if np.any(intrinsic_standard_deviations < 0.0) or np.any(
        extrinsic_standard_deviations < 0.0
    ):
        raise CalibrationError("calibration produced a negative standard deviation")
    if camera_matrix.shape != (3, 3) or np.linalg.matrix_rank(camera_matrix) != 3:
        raise CalibrationError("calibration produced a degenerate camera matrix")
    if camera_matrix[0, 0] <= 0.0 or camera_matrix[1, 1] <= 0.0:
        raise CalibrationError("calibration produced a non-positive focal length")
    if not distortion.size:
        raise CalibrationError("calibration produced no distortion parameters")
    if (
        len(rotations) != sample_count
        or len(translations) != sample_count
        or per_view_errors.size != sample_count
        or extrinsic_standard_deviations.size != sample_count * 6
        or any(vector.size != 3 for vector in rotations)
        or any(vector.size != 3 for vector in translations)
    ):
        raise CalibrationError("calibration returned inconsistent per-view diagnostics")
    parameter_count = 4 + distortion.size
    if intrinsic_standard_deviations.size < parameter_count:
        raise CalibrationError("calibration omitted intrinsic standard deviations")
    return _ExtendedCalibration(
        rms_reprojection_error_px=float(rms),
        camera_matrix=camera_matrix,
        distortion=distortion,
        rotation_vectors=rotations,
        translation_vectors=translations,
        intrinsic_standard_deviations=intrinsic_standard_deviations[:parameter_count],
        per_view_errors_px=per_view_errors,
    )


def _intrinsic_parameter_names(distortion_count: int) -> Tuple[str, ...]:
    if distortion_count > len(_DISTORTION_PARAMETER_NAMES):
        raise CalibrationError("calibration returned an unsupported distortion vector")
    return ("fx", "fy", "cx", "cy") + _DISTORTION_PARAMETER_NAMES[:distortion_count]


def _intrinsic_parameter_vector(calibration: _ExtendedCalibration) -> np.ndarray:
    matrix = calibration.camera_matrix
    return np.concatenate(
        (
            np.asarray((matrix[0, 0], matrix[1, 1], matrix[0, 2], matrix[1, 2])),
            calibration.distortion,
        )
    ).astype(np.float64)


def _projected_intrinsic_information(
    object_points: Sequence[np.ndarray], calibration: _ExtendedCalibration
) -> Tuple[int, float, float, np.ndarray, np.ndarray]:
    """Return intrinsic Jacobian evidence after eliminating per-view poses.

    The projection Jacobian contains each view's six nuisance pose columns and
    the shared intrinsic columns. Projecting the latter onto the orthogonal
    complement of the former prevents a pose change from masquerading as
    intrinsic information. Columns are then normalized before the SVD so the
    reported numerical rank is not an artefact of mixed parameter units.
    """
    distortion_count = calibration.distortion.size
    parameter_count = 4 + distortion_count
    projected_blocks = []
    for world, rotation, translation in zip(
        object_points, calibration.rotation_vectors, calibration.translation_vectors
    ):
        try:
            _projected, jacobian = cv2.projectPoints(
                np.asarray(world, dtype=np.float64).reshape(-1, 3),
                rotation,
                translation,
                calibration.camera_matrix,
                calibration.distortion,
            )
        except (ValueError, cv2.error) as error:
            raise CalibrationError("intrinsic projection Jacobian failed") from error
        jacobian = np.asarray(jacobian, dtype=np.float64)
        if jacobian.ndim != 2 or jacobian.shape[1] < 10 + distortion_count:
            raise CalibrationError("OpenCV returned an incomplete projection Jacobian")
        pose_jacobian = jacobian[:, :6]
        intrinsic_jacobian = np.concatenate(
            (jacobian[:, 6:10], jacobian[:, 10 : 10 + distortion_count]), axis=1
        )
        try:
            pose_projection = pose_jacobian @ np.linalg.lstsq(
                pose_jacobian, intrinsic_jacobian, rcond=None
            )[0]
        except np.linalg.LinAlgError as error:
            raise CalibrationError("intrinsic information projection failed") from error
        projected_blocks.append(intrinsic_jacobian - pose_projection)
    projected = np.concatenate(projected_blocks, axis=0)
    if not np.all(np.isfinite(projected)):
        raise CalibrationError("intrinsic information matrix is non-finite")
    column_norms = np.linalg.norm(projected, axis=0)
    normalized = np.zeros_like(projected)
    nonzero = column_norms > 0.0
    normalized[:, nonzero] = projected[:, nonzero] / column_norms[nonzero]
    try:
        singular_values = np.linalg.svd(normalized, compute_uv=False)
    except np.linalg.LinAlgError as error:
        raise CalibrationError("intrinsic information SVD failed") from error
    largest = float(singular_values[0]) if singular_values.size else 0.0
    tolerance = (
        float(max(normalized.shape) * np.finfo(np.float64).eps * largest)
        if largest > 0.0
        else 0.0
    )
    rank = int(np.count_nonzero(singular_values > tolerance))
    condition_number = (
        float("inf")
        if rank < parameter_count or singular_values[-1] <= tolerance
        else float(singular_values[0] / singular_values[-1])
    )
    return rank, condition_number, tolerance, singular_values, column_norms


def _held_out_reprojection(
    object_points: np.ndarray,
    image_points: np.ndarray,
    calibration: _ExtendedCalibration,
) -> Tuple[float, float, float, Tuple[float, ...], np.ndarray, np.ndarray]:
    """Fit only an omitted view's pose under fixed fold K/D and score pixels."""
    world = np.asarray(object_points, dtype=np.float32).reshape(-1, 3)
    pixels = np.asarray(image_points, dtype=np.float32).reshape(-1, 1, 2)
    try:
        solved, rotation, translation = cv2.solvePnP(
            world,
            pixels,
            calibration.camera_matrix,
            calibration.distortion,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )
        projected, _jacobian = cv2.projectPoints(
            world,
            rotation,
            translation,
            calibration.camera_matrix,
            calibration.distortion,
        )
    except cv2.error as error:
        raise CalibrationError("held-out pose optimization failed") from error
    rotation = np.asarray(rotation, dtype=np.float64).reshape(-1)
    translation = np.asarray(translation, dtype=np.float64).reshape(-1)
    projected = np.asarray(projected, dtype=np.float64).reshape(-1, 2)
    observed = pixels.astype(np.float64).reshape(-1, 2)
    errors = np.linalg.norm(projected - observed, axis=1)
    if (
        not solved
        or rotation.size != 3
        or translation.size != 3
        or not np.all(np.isfinite(rotation))
        or not np.all(np.isfinite(translation))
        or not np.all(np.isfinite(errors))
    ):
        raise CalibrationError("held-out pose diagnostics are invalid")
    rms = float(np.sqrt(np.mean(np.square(errors))))
    return (
        rms,
        float(np.mean(errors)),
        float(np.max(errors)),
        tuple(float(value) for value in errors),
        rotation,
        translation,
    )


def _undistorted_ray_stability(
    reference: _ExtendedCalibration,
    fold: _ExtendedCalibration,
    image_size: Tuple[int, int],
) -> Tuple[float, float]:
    """Compare pixel-to-unit-ray mappings and express chord error in pixels."""
    width, height = image_size
    x_coordinates = np.linspace(0.0, float(width - 1), 9, dtype=np.float64)
    y_coordinates = np.linspace(0.0, float(height - 1), 7, dtype=np.float64)
    xx, yy = np.meshgrid(x_coordinates, y_coordinates)
    pixels = np.column_stack((xx.reshape(-1), yy.reshape(-1))).reshape(-1, 1, 2)

    def unit_rays(calibration: _ExtendedCalibration) -> np.ndarray:
        try:
            normalized = cv2.undistortPoints(
                pixels,
                calibration.camera_matrix,
                calibration.distortion,
            ).reshape(-1, 2)
        except cv2.error as error:
            raise CalibrationError("undistorted-ray diagnostics failed") from error
        rays = np.column_stack((normalized, np.ones(len(normalized), dtype=np.float64)))
        norms = np.linalg.norm(rays, axis=1)
        if (
            not np.all(np.isfinite(rays))
            or not np.all(np.isfinite(norms))
            or bool(np.any(norms <= 0.0))
        ):
            raise CalibrationError("undistorted-ray diagnostics are non-finite")
        return rays / norms[:, None]

    reference_rays = unit_rays(reference)
    fold_rays = unit_rays(fold)
    equivalent_focal_px = math.sqrt(
        float(reference.camera_matrix[0, 0] * reference.camera_matrix[1, 1])
    )
    errors = np.linalg.norm(reference_rays - fold_rays, axis=1) * equivalent_focal_px
    if not np.all(np.isfinite(errors)):
        raise CalibrationError("undistorted-ray stability is non-finite")
    return float(np.sqrt(np.mean(np.square(errors)))), float(np.max(errors))


def _select_calibration_views(
    object_points: Sequence[np.ndarray],
    image_points: Sequence[np.ndarray],
    image_size: Tuple[int, int],
    observation_uncertainty: Optional[float],
) -> Tuple[
    _ExtendedCalibration,
    List[int],
    List[IntrinsicRejectedView],
    Tuple[float, ...],
]:
    """Iteratively remove only the worst robust per-view RMS outlier."""
    initial = _run_extended_calibration(object_points, image_points, image_size)
    initial_errors = tuple(float(value) for value in initial.per_view_errors_px)
    selected_indices = list(range(len(image_points)))
    rejected: List[IntrinsicRejectedView] = []
    calibration = initial
    detector_sigma = 0.0 if observation_uncertainty is None else float(
        observation_uncertainty
    )
    while len(selected_indices) > 3:
        errors = np.asarray(calibration.per_view_errors_px, dtype=np.float64)
        median = float(np.median(errors))
        mad = float(np.median(np.abs(errors - median)))
        robust_sigma = 1.482602218505602 * mad
        numerical_sigma = np.finfo(np.float64).eps * max(1.0, abs(median))
        sigma = max(math.hypot(robust_sigma, detector_sigma), numerical_sigma)
        envelope = median + 3.0 * sigma
        worst_local_index = int(np.argmax(errors))
        worst_error = float(errors[worst_local_index])
        if worst_error <= envelope:
            break
        original_index = selected_indices[worst_local_index]
        rejected.append(
            IntrinsicRejectedView(
                original_view_index=original_index,
                reason="per_view_rms_above_robust_3sigma_envelope",
                initial_rms_reprojection_error_px=initial_errors[original_index],
                rejection_rms_reprojection_error_px=worst_error,
                rejection_envelope_px=envelope,
            )
        )
        del selected_indices[worst_local_index]
        calibration = _run_extended_calibration(
            [object_points[index] for index in selected_indices],
            [image_points[index] for index in selected_indices],
            image_size,
        )
    return calibration, selected_indices, rejected, initial_errors


def _leave_one_out_stability(
    object_points: Sequence[np.ndarray],
    image_points: Sequence[np.ndarray],
    image_size: Tuple[int, int],
    reference: _ExtendedCalibration,
    original_view_indices: Optional[Sequence[int]] = None,
) -> IntrinsicStabilityDiagnostics:
    names = _intrinsic_parameter_names(reference.distortion.size)
    reference_parameters = _intrinsic_parameter_vector(reference)
    folds: List[IntrinsicFoldEstimate] = []
    failed: List[int] = []
    if original_view_indices is None:
        original_view_indices = tuple(range(len(image_points)))
    if len(original_view_indices) != len(image_points):
        raise CalibrationError("LOO original view indices do not match selected views")
    for omitted in range(len(image_points)):
        fold_objects = [
            value for index, value in enumerate(object_points) if index != omitted
        ]
        fold_images = [
            value for index, value in enumerate(image_points) if index != omitted
        ]
        try:
            fold = _run_extended_calibration(fold_objects, fold_images, image_size)
            parameters = _intrinsic_parameter_vector(fold)
            if parameters.shape != reference_parameters.shape:
                raise CalibrationError("fold returned a different intrinsic parameter vector")
            (
                held_out_rms,
                held_out_mean,
                held_out_max,
                held_out_errors,
                held_out_rotation,
                held_out_translation,
            ) = _held_out_reprojection(
                object_points[omitted], image_points[omitted], fold
            )
            ray_rms, ray_max = _undistorted_ray_stability(
                reference, fold, image_size
            )
        except CalibrationError:
            failed.append(int(original_view_indices[omitted]))
            continue
        folds.append(
            IntrinsicFoldEstimate(
                omitted_view_index=int(original_view_indices[omitted]),
                rms_reprojection_error_px=fold.rms_reprojection_error_px,
                parameters=tuple(float(value) for value in parameters),
                held_out_rms_reprojection_error_px=held_out_rms,
                held_out_mean_reprojection_error_px=held_out_mean,
                held_out_max_reprojection_error_px=held_out_max,
                held_out_point_errors_px=held_out_errors,
                held_out_rotation_vector=tuple(
                    float(value) for value in held_out_rotation
                ),
                held_out_translation_vector=tuple(
                    float(value) for value in held_out_translation
                ),
                undistorted_ray_rms_equivalent_px=ray_rms,
                undistorted_ray_max_equivalent_px=ray_max,
            )
        )
    if not folds:
        standard_deviation = span = maximum_delta = relative_delta = None
        held_out_rms_mean = held_out_rms_max = None
        ray_rms = ray_max = None
    else:
        estimates = np.asarray([fold.parameters for fold in folds], dtype=np.float64)
        standard_deviation_array = np.std(estimates, axis=0)
        span_array = np.ptp(estimates, axis=0)
        maximum_delta_array = np.max(np.abs(estimates - reference_parameters), axis=0)
        reference_scale = np.maximum(
            np.abs(reference_parameters), np.finfo(np.float64).eps
        )
        relative_delta_array = maximum_delta_array / reference_scale
        standard_deviation = tuple(float(value) for value in standard_deviation_array)
        span = tuple(float(value) for value in span_array)
        maximum_delta = tuple(float(value) for value in maximum_delta_array)
        relative_delta = tuple(float(value) for value in relative_delta_array)
        held_out_values = np.asarray(
            [fold.held_out_rms_reprojection_error_px for fold in folds],
            dtype=np.float64,
        )
        ray_rms_values = np.asarray(
            [fold.undistorted_ray_rms_equivalent_px for fold in folds],
            dtype=np.float64,
        )
        ray_max_values = np.asarray(
            [fold.undistorted_ray_max_equivalent_px for fold in folds],
            dtype=np.float64,
        )
        held_out_rms_mean = float(np.mean(held_out_values))
        held_out_rms_max = float(np.max(held_out_values))
        ray_rms = float(np.sqrt(np.mean(np.square(ray_rms_values))))
        ray_max = float(np.max(ray_max_values))
    return IntrinsicStabilityDiagnostics(
        method="leave_one_view_out",
        parameter_names=names,
        reference_parameters=tuple(float(value) for value in reference_parameters),
        folds=tuple(folds),
        failed_omitted_view_indices=tuple(failed),
        parameter_standard_deviation=standard_deviation,
        parameter_span=span,
        maximum_absolute_delta=maximum_delta,
        maximum_relative_delta=relative_delta,
        held_out_rms_mean_px=held_out_rms_mean,
        held_out_rms_max_px=held_out_rms_max,
        undistorted_ray_rms_equivalent_px=ray_rms,
        undistorted_ray_max_equivalent_px=ray_max,
    )


def calibrate_intrinsic(
    image_points: Sequence[np.ndarray],
    board_size: Sequence[int],
    square: float,
    image_size: Sequence[int],
    object_points: Optional[Sequence[np.ndarray]] = None,
    observation_uncertainty: Optional[float] = None,
) -> IntrinsicResult:
    """Batch-estimate free K/D and return continuous solve-quality evidence."""
    if len(image_points) < 3:
        raise CalibrationError("need at least three samples to calibrate")
    prepared_object, corners, size = _prepare_calibration_observations(
        image_points, board_size, square, image_size, object_points
    )
    if observation_uncertainty is not None and (
        not np.isfinite(float(observation_uncertainty))
        or float(observation_uncertainty) < 0.0
    ):
        raise CalibrationError("observation uncertainty must be finite and non-negative")
    (
        calibration,
        selected_indices,
        rejected_views,
        initial_per_view_errors,
    ) = _select_calibration_views(
        prepared_object,
        corners,
        size,
        observation_uncertainty,
    )
    selected_objects = [prepared_object[index] for index in selected_indices]
    selected_corners = [corners[index] for index in selected_indices]
    parameter_names = _intrinsic_parameter_names(calibration.distortion.size)
    rank, condition, rank_tolerance, singular_values, column_norms = (
        _projected_intrinsic_information(selected_objects, calibration)
    )
    stability = _leave_one_out_stability(
        selected_objects,
        selected_corners,
        size,
        calibration,
        original_view_indices=selected_indices,
    )
    diagnostics = IntrinsicCalibrationDiagnostics(
        finite=True,
        parameter_names=parameter_names,
        per_view_errors_px=tuple(
            float(value) for value in calibration.per_view_errors_px
        ),
        intrinsic_standard_deviations=tuple(
            float(value) for value in calibration.intrinsic_standard_deviations
        ),
        rotation_vectors=tuple(
            tuple(float(value) for value in vector)
            for vector in calibration.rotation_vectors
        ),
        translation_vectors=tuple(
            tuple(float(value) for value in vector)
            for vector in calibration.translation_vectors
        ),
        projected_intrinsic_rank=rank,
        projected_intrinsic_parameter_count=len(parameter_names),
        projected_intrinsic_rank_deficient=rank < len(parameter_names),
        projected_intrinsic_condition_number=condition,
        projected_intrinsic_rank_tolerance=rank_tolerance,
        projected_intrinsic_singular_values=tuple(
            float(value) for value in singular_values
        ),
        projected_intrinsic_column_norms=tuple(float(value) for value in column_norms),
        stability=stability,
        pool_sample_count=len(image_points),
        selected_view_indices=tuple(int(value) for value in selected_indices),
        rejected_views=tuple(rejected_views),
        initial_per_view_errors_px=initial_per_view_errors,
        observation_uncertainty_px=(
            float(observation_uncertainty)
            if observation_uncertainty is not None
            else None
        ),
    )
    return IntrinsicResult(
        camera_matrix=calibration.camera_matrix,
        distortion=calibration.distortion,
        image_size=size,
        rms_reprojection_error_px=calibration.rms_reprojection_error_px,
        sample_count=len(selected_indices),
        diagnostics=diagnostics,
    )


def intrinsic_document(
    result: IntrinsicResult,
    *,
    camera_name: str,
    board_size: Sequence[int],
    square: float,
    metadata: Optional[Dict[str, Any]] = None,
    board: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    identity = str(camera_name).strip()
    if not identity or "\n" in identity or "\r" in identity:
        raise ValueError("camera_name must be a non-empty single line")
    k = result.camera_matrix
    rectification = np.eye(3, dtype=np.float64)
    projection = np.zeros((3, 4), dtype=np.float64)
    projection[:, :3] = k
    board_payload: Dict[str, Any] = {
        "size": [int(board_size[0]), int(board_size[1])],
        "square_size_m": float(square),
        "type": "checkerboard",
    }
    if board:
        board_payload.update(board)
    document: Dict[str, Any] = {
        "schema": "xgc2.camera.intrinsic.v1",
        "created_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "camera_name": identity,
        "image_width": result.image_size[0],
        "image_height": result.image_size[1],
        "camera_matrix": {"rows": 3, "cols": 3, "data": [float(v) for v in k.reshape(-1)]},
        "distortion_model": "plumb_bob",
        "distortion_coefficients": {
            "rows": 1,
            "cols": int(result.distortion.size),
            "data": [float(v) for v in result.distortion],
        },
        "rectification_matrix": {
            "rows": 3,
            "cols": 3,
            "data": [float(value) for value in rectification.reshape(-1)],
        },
        "projection_matrix": {
            "rows": 3,
            "cols": 4,
            "data": [float(value) for value in projection.reshape(-1)],
        },
        "focal_length": {"fx": float(k[0, 0]), "fy": float(k[1, 1])},
        "principal_point": {"cx": float(k[0, 2]), "cy": float(k[1, 2])},
        "rms_reprojection_error_px": result.rms_reprojection_error_px,
        "sample_count": result.sample_count,
        "board": board_payload,
    }
    if metadata:
        document["metadata"] = dict(metadata)
    return document


def save_intrinsic(
    path: os.PathLike,
    result: IntrinsicResult,
    *,
    camera_name: str,
    board_size: Sequence[int],
    square: float,
    metadata: Optional[Dict[str, Any]] = None,
    board: Optional[Dict[str, Any]] = None,
) -> Path:
    """Atomically persist a versioned intrinsic document outside package share."""
    destination = Path(path).expanduser()
    destination.parent.mkdir(parents=True, exist_ok=True)
    document = intrinsic_document(
        result, camera_name=camera_name, board_size=board_size, square=square,
        metadata=metadata, board=board
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


def load_intrinsic(path: os.PathLike) -> Dict[str, Any]:
    source = Path(path).expanduser()
    with source.open("r", encoding="utf-8") as stream:
        document = yaml.safe_load(stream) or {}
    if not isinstance(document, dict):
        raise CalibrationError("intrinsic document must be a mapping")
    if document.get("schema") != "xgc2.camera.intrinsic.v1":
        raise CalibrationError("unsupported or missing intrinsic schema")
    matrix = document.get("camera_matrix", {})
    data = matrix.get("data") if isinstance(matrix, dict) else None
    if not isinstance(data, list) or len(data) != 9:
        raise CalibrationError("camera_matrix.data must contain nine values")
    document["camera_matrix_array"] = np.asarray(data, dtype=np.float64).reshape(3, 3)
    return document
