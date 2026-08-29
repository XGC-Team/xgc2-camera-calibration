"""Intrinsic (pinhole) camera calibration, cv2-direct and cv_bridge-free.

Mirrors the ROS ``camera_calibration`` coverage heuristics (X / Y / Size / Skew)
and delegates the actual estimation to ``cv2.calibrateCamera`` -- the same call
``MonoCalibrator`` makes internally -- without dragging in ``cv_bridge`` or any
ROS calibration class.  Frames arrive already decoded (see
``web_service.image_message_to_bgr``); board detection runs on a down-scaled copy
for speed on large (4K) frames, then corners are refined at full resolution.
"""

from __future__ import annotations

import math
import os
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import yaml

from xgc_camera_calibration.solver import CalibrationError

_DETECT_FLAGS = cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE
_SUBPIX_CRITERIA = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER, 30, 0.01)
# Same acceptance ranges the ROS camera_calibration GUI uses for goodenough.
PARAM_RANGES: Tuple[float, float, float, float] = (0.7, 0.7, 0.4, 0.5)
PARAM_NAMES: Tuple[str, str, str, str] = ("X", "Y", "Size", "Skew")
SAMPLE_DISTANCE = 0.2
_APRILTAG_DICTIONARIES = {
    "tag36h11": "DICT_APRILTAG_36h11",
    "apriltag_36h11": "DICT_APRILTAG_36h11",
    "36h11": "DICT_APRILTAG_36h11",
}
_APRILGRID_FALLBACK_MIN_SHARPNESS = 160.0
_APRILGRID_FALLBACK_MARKER_PIXELS = 80
_APRILGRID_FALLBACK_MAX_PAYLOAD_ERRORS = 5
_APRILGRID_FALLBACK_MAX_BORDER_ERRORS = 2


@dataclass(frozen=True)
class BoardDetection:
    image_points: np.ndarray
    object_points: np.ndarray
    coverage: Tuple[float, float, float, float]
    # AprilGrid renderers/detectors do not always agree on whether a tag has
    # one or two black border cells. The decoded centre is invariant to that
    # convention, while the reported outer corners are not. Keep every corner
    # for annotation and coverage, but allow calibration to use tag centres.
    calibration_image_points: Optional[np.ndarray] = None
    calibration_object_points: Optional[np.ndarray] = None


@dataclass(frozen=True)
class IntrinsicResult:
    camera_matrix: np.ndarray          # 3x3
    distortion: np.ndarray             # (k1,k2,p1,p2,k3)
    image_size: Tuple[int, int]        # (width, height)
    rms_reprojection_error_px: float
    sample_count: int


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
    if hasattr(cv2.aruco, "ArucoDetector"):
        detector = cv2.aruco.ArucoDetector(dictionary)
        return detector.detectMarkers(image)
    if hasattr(cv2.aruco, "detectMarkers"):
        return cv2.aruco.detectMarkers(image, dictionary)
    raise CalibrationError("OpenCV aruco marker detection is unavailable")


def aprilgrid_has_candidate_evidence(
    gray: np.ndarray,
    tag_family: str = "tag36h11",
    minimum_quads: int = 6,
) -> bool:
    """Return whether a low-resolution frame justifies a source-level retry."""
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


def refine_aprilgrid_calibration_centers(
    source_gray: np.ndarray,
    source_tag_corners: np.ndarray,
) -> np.ndarray:
    """Refine accepted tag corners locally on the source image, then center them.

    Continuous detection stays on the small search plane. Only a geometrically
    distinct sample pays for this source-resolution grayscale refinement.
    """
    if source_gray.ndim != 2:
        raise ValueError("AprilGrid source refinement expects a grayscale image")
    corners = np.asarray(source_tag_corners, dtype=np.float32).reshape(-1, 4, 2)
    pixels = corners.reshape(-1, 1, 2).copy()
    height, width = source_gray.shape[:2]
    flat = pixels.reshape(-1, 2)
    valid = (
        (flat[:, 0] >= 6.0)
        & (flat[:, 0] < float(width - 6))
        & (flat[:, 1] >= 6.0)
        & (flat[:, 1] < float(height - 6))
    )
    if bool(np.any(valid)):
        pixels[valid] = cv2.cornerSubPix(
            source_gray,
            pixels[valid].copy(),
            (5, 5),
            (-1, -1),
            _SUBPIX_CRITERIA,
        )
    return np.mean(pixels.reshape(-1, 4, 2), axis=1).reshape(-1, 1, 2).astype(np.float32)


def _draw_aruco_marker(dictionary: Any, marker_id: int, size: int) -> np.ndarray:
    """Render one marker across the OpenCV 4.2 and 4.7+ Python APIs."""
    if hasattr(cv2.aruco, "generateImageMarker"):
        return cv2.aruco.generateImageMarker(dictionary, marker_id, size)
    if hasattr(cv2.aruco, "drawMarker"):
        return cv2.aruco.drawMarker(dictionary, marker_id, size, borderBits=1)
    raise CalibrationError("OpenCV aruco marker rendering is unavailable")


def _ordered_quad(points: np.ndarray) -> np.ndarray:
    """Return a convex quad in image TL, TR, BR, BL order."""
    quad = np.asarray(points, dtype=np.float32).reshape(4, 2)
    center = np.mean(quad, axis=0)
    angles = np.arctan2(quad[:, 1] - center[1], quad[:, 0] - center[0])
    quad = quad[np.argsort(angles)]
    return np.roll(quad, -int(np.argmin(np.sum(quad, axis=1))), axis=0)


def _aprilgrid_marker_templates(
    dictionary: Any, start_id: int, count: int
) -> List[np.ndarray]:
    """Build 8x8 binary templates: one black border plus a 6x6 payload."""
    marker_pixels = _APRILGRID_FALLBACK_MARKER_PIXELS
    cell = marker_pixels // 8
    templates: List[np.ndarray] = []
    for marker_id in range(int(start_id), int(start_id) + int(count)):
        marker = _draw_aruco_marker(dictionary, marker_id, marker_pixels)
        bits = np.zeros((8, 8), dtype=np.uint8)
        for row in range(8):
            for col in range(8):
                patch = marker[
                    row * cell + 2 : (row + 1) * cell - 2,
                    col * cell + 2 : (col + 1) * cell - 2,
                ]
                bits[row, col] = int(float(np.mean(patch)) > 127.0)
        templates.append(bits)
    return templates


def _decode_aprilgrid_quad(
    gray: np.ndarray,
    quad: np.ndarray,
    templates: Sequence[np.ndarray],
    start_id: int,
) -> Tuple[int, int, int, int]:
    """Return payload errors, marker id, rotation and border errors."""
    marker_pixels = _APRILGRID_FALLBACK_MARKER_PIXELS
    destination = np.array(
        (
            (0, 0),
            (marker_pixels - 1, 0),
            (marker_pixels - 1, marker_pixels - 1),
            (0, marker_pixels - 1),
        ),
        dtype=np.float32,
    )
    transform = cv2.getPerspectiveTransform(quad, destination)
    marker = cv2.warpPerspective(gray, transform, (marker_pixels, marker_pixels))
    _threshold, marker = cv2.threshold(
        marker, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU
    )
    cell = marker_pixels // 8
    bits = np.zeros((8, 8), dtype=np.uint8)
    for row in range(8):
        for col in range(8):
            patch = marker[
                row * cell + 2 : (row + 1) * cell - 2,
                col * cell + 2 : (col + 1) * cell - 2,
            ]
            bits[row, col] = int(float(np.mean(patch)) > 127.0)
    border = np.concatenate(
        (bits[0, :], bits[-1, :], bits[1:-1, 0], bits[1:-1, -1])
    )
    border_errors = int(np.sum(border))
    best = (37, int(start_id), 0, border_errors)
    for offset, template in enumerate(templates):
        for rotation in range(4):
            rotated = np.rot90(template, rotation)
            errors = int(np.count_nonzero(bits[1:7, 1:7] != rotated[1:7, 1:7]))
            if errors < best[0]:
                best = (errors, int(start_id) + offset, rotation, border_errors)
    return best


def _extract_aprilgrid_quads(
    gray: np.ndarray, dictionary: Any, start_id: int, tag_count: int
) -> List[Dict[str, Any]]:
    """Extract and softly decode black AprilTag outer quads."""
    templates = _aprilgrid_marker_templates(dictionary, start_id, tag_count)
    binary = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        31,
        7,
    )
    contours, _hierarchy = cv2.findContours(
        binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE
    )
    image_area = float(gray.shape[0] * gray.shape[1])
    candidates: List[Dict[str, Any]] = []
    for contour in contours:
        area = float(cv2.contourArea(contour))
        if not image_area * 0.00035 < area < image_area * 0.03:
            continue
        perimeter = float(cv2.arcLength(contour, True))
        approximate = cv2.approxPolyDP(contour, 0.04 * perimeter, True)
        if len(approximate) != 4 or not cv2.isContourConvex(approximate):
            continue
        _center, dimensions, _angle = cv2.minAreaRect(approximate)
        short_edge, long_edge = sorted((float(dimensions[0]), float(dimensions[1])))
        if short_edge < 8.0 or long_edge / short_edge > 2.0:
            continue
        if area / max(1.0, short_edge * long_edge) < 0.6:
            continue
        quad = _ordered_quad(approximate)
        payload_errors, marker_id, rotation, border_errors = _decode_aprilgrid_quad(
            gray, quad, templates, start_id
        )
        candidates.append(
            {
                "quad": quad,
                "center": np.mean(quad, axis=0),
                "edge": float(
                    np.mean(
                        [
                            np.linalg.norm(quad[(index + 1) % 4] - quad[index])
                            for index in range(4)
                        ]
                    )
                ),
                "payload_errors": payload_errors,
                "marker_id": marker_id,
                "rotation": rotation,
                "border_errors": border_errors,
            }
        )
    return candidates


def _project_points(homography: np.ndarray, points: np.ndarray) -> np.ndarray:
    return cv2.perspectiveTransform(
        np.asarray(points, dtype=np.float32).reshape(-1, 1, 2), homography
    ).reshape(-1, 2)


def _match_aprilgrid_lattice(
    homography: np.ndarray,
    tag_objects: Sequence[np.ndarray],
    candidates: Sequence[Dict[str, Any]],
) -> List[Tuple[int, int, float]]:
    choices: List[Tuple[float, float, int, int]] = []
    for tag_index, tag_object in enumerate(tag_objects):
        predicted = _project_points(homography, tag_object[:, :2])
        predicted_center = np.mean(predicted, axis=0)
        predicted_edge = float(
            np.mean(
                [
                    np.linalg.norm(predicted[(index + 1) % 4] - predicted[index])
                    for index in range(4)
                ]
            )
        )
        limit = max(8.0, predicted_edge * 0.55)
        for candidate_index, candidate in enumerate(candidates):
            if int(candidate["border_errors"]) > 4:
                continue
            edge_ratio = float(candidate["edge"]) / max(1.0, predicted_edge)
            if not 0.6 <= edge_ratio <= 1.4:
                continue
            distance = float(np.linalg.norm(candidate["center"] - predicted_center))
            if distance < limit:
                choices.append(
                    (distance / limit, distance, tag_index, candidate_index)
                )
    matches: List[Tuple[int, int, float]] = []
    used_tags = set()
    used_candidates = set()
    for _relative, distance, tag_index, candidate_index in sorted(choices):
        if tag_index in used_tags or candidate_index in used_candidates:
            continue
        matches.append((tag_index, candidate_index, distance))
        used_tags.add(tag_index)
        used_candidates.add(candidate_index)
    return matches


def _align_aprilgrid_corners_to_lattice(
    image_points: Sequence[np.ndarray], object_points: Sequence[np.ndarray]
) -> List[np.ndarray]:
    """Align per-tag corner order with the board lattice encoded by tag ids.

    Some printed AprilGrid plates rotate every marker bitmap relative to the
    board axes. ArUco then decodes the correct ids and centers but returns a
    per-marker corner order that is inconsistent with the board geometry. A
    homography fitted only from decoded tag centers resolves the board axes;
    measured corners are then permuted, never synthesized, to match them.
    """
    if len(image_points) < 4 or len(image_points) != len(object_points):
        return [np.asarray(points, dtype=np.float32) for points in image_points]
    source_centers = np.asarray(
        [np.mean(points[:, :2], axis=0) for points in object_points],
        dtype=np.float32,
    )
    image_centers = np.asarray(
        [np.mean(points, axis=0) for points in image_points], dtype=np.float32
    )
    homography, mask = cv2.findHomography(
        source_centers, image_centers, cv2.RANSAC, 5.0
    )
    if homography is None or mask is None or int(np.sum(mask)) < 4:
        return [np.asarray(points, dtype=np.float32) for points in image_points]
    aligned_points: List[np.ndarray] = []
    for measured, tag_object in zip(image_points, object_points):
        predicted = _project_points(homography, tag_object[:, :2])
        measured = np.asarray(measured, dtype=np.float32).reshape(4, 2)
        permutations = [
            np.roll(measured, shift, axis=0) for shift in range(4)
        ] + [
            np.roll(measured[::-1], shift, axis=0) for shift in range(4)
        ]
        aligned_points.append(
            min(
                permutations,
                key=lambda quad: float(
                    np.sum(np.linalg.norm(quad - predicted, axis=1))
                ),
            ).astype(np.float32)
        )
    return aligned_points


def _aprilgrid_contour_fallback(
    gray: np.ndarray,
    dictionary: Any,
    board_size: Sequence[int],
    square: float,
    tag_spacing: float,
    start_id: int,
    min_tags: int,
) -> Optional[Tuple[List[np.ndarray], List[np.ndarray]]]:
    """Recover a decoded lattice when OpenCV 4.2 rejects small real tags."""
    sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    if sharpness < _APRILGRID_FALLBACK_MIN_SHARPNESS:
        return None
    tag_count = int(board_size[0]) * int(board_size[1])
    candidates = _extract_aprilgrid_quads(gray, dictionary, start_id, tag_count)
    anchors = [
        candidate
        for candidate in candidates
        if int(candidate["payload_errors"])
        <= _APRILGRID_FALLBACK_MAX_PAYLOAD_ERRORS
        and int(candidate["border_errors"])
        <= _APRILGRID_FALLBACK_MAX_BORDER_ERRORS
    ]
    if not anchors:
        return None
    tag_objects = [
        aprilgrid_tag_object_points(
            board_size, square, tag_spacing, start_id, start_id + offset
        )
        for offset in range(tag_count)
    ]
    if any(tag_object is None for tag_object in tag_objects):
        return None
    unique_anchors: Dict[int, Dict[str, Any]] = {}
    for anchor in anchors:
        marker_offset = int(anchor["marker_id"]) - int(start_id)
        current = unique_anchors.get(marker_offset)
        if current is None or int(anchor["payload_errors"]) < int(
            current["payload_errors"]
        ):
            unique_anchors[marker_offset] = anchor
    if len(unique_anchors) < 4:
        return None

    source_centers = np.asarray(
        [
            np.mean(tag_objects[tag_index][:, :2], axis=0)
            for tag_index in unique_anchors
        ],
        dtype=np.float32,
    )
    image_centers = np.asarray(
        [unique_anchors[tag_index]["center"] for tag_index in unique_anchors],
        dtype=np.float32,
    )
    initial_homography, mask = cv2.findHomography(
        source_centers, image_centers, cv2.RANSAC, 5.0
    )
    if initial_homography is None or mask is None or int(np.sum(mask)) < 4:
        return None
    initial_matches = _match_aprilgrid_lattice(
        initial_homography, tag_objects, candidates
    )
    if len(initial_matches) < int(min_tags):
        return None
    source_centers = np.asarray(
        [
            np.mean(tag_objects[tag_index][:, :2], axis=0)
            for tag_index, _candidate, _distance in initial_matches
        ],
        dtype=np.float32,
    )
    image_centers = np.asarray(
        [
            candidates[candidate_index]["center"]
            for _tag, candidate_index, _distance in initial_matches
        ],
        dtype=np.float32,
    )
    refined_homography, mask = cv2.findHomography(
        source_centers, image_centers, cv2.RANSAC, 5.0
    )
    if refined_homography is None or mask is None or int(np.sum(mask)) < 4:
        return None
    matches = _match_aprilgrid_lattice(
        refined_homography, tag_objects, candidates
    )
    if len(matches) < int(min_tags):
        return None

    image_points: List[np.ndarray] = []
    object_points: List[np.ndarray] = []
    for tag_index, candidate_index, _distance in sorted(matches):
        predicted = _project_points(
            refined_homography, tag_objects[tag_index][:, :2]
        )
        candidate_quad = candidates[candidate_index]["quad"]
        permutations = [
            np.roll(candidate_quad, shift, axis=0) for shift in range(4)
        ] + [
            np.roll(candidate_quad[::-1], shift, axis=0) for shift in range(4)
        ]
        aligned = min(
            permutations,
            key=lambda quad: float(np.sum(np.linalg.norm(quad - predicted, axis=1))),
        )
        image_points.append(aligned.astype(np.float32))
        object_points.append(tag_objects[tag_index])
    return image_points, object_points


def aprilgrid_tag_object_points(
    board_size: Sequence[int],
    tag_size: float,
    tag_spacing: float,
    start_id: int,
    tag_id: int,
) -> Optional[np.ndarray]:
    """Four tag corners (TL, TR, BR, BL) in the board frame, matching OpenCV ArUco order."""
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
) -> Optional[BoardDetection]:
    """Detect a Kalibr-style AprilGrid and return the visible tag corners."""
    if float(square) <= 0.0:
        raise ValueError("AprilGrid tag size must be positive")
    if float(tag_spacing) < 0.0:
        raise ValueError("AprilGrid tag spacing must be non-negative")
    if int(min_tags) < 4:
        raise ValueError("AprilGrid detection needs at least four tags")
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

    def in_board_count(marker_ids: Optional[np.ndarray]) -> int:
        if marker_ids is None:
            return 0
        first = int(start_id)
        last = first + int(board_size[0]) * int(board_size[1])
        return sum(first <= int(marker_id) < last for marker_id in marker_ids.reshape(-1))

    best_count = in_board_count(ids)
    base_scale = scale
    recovered = None
    if best_count < int(min_tags):
        # The contour/lattice path works at the already bounded search plane
        # and is the compatibility path for older OpenCV AprilTag decoders.
        # Try it before enlarging that plane through the recovery pyramid: a
        # missing or motion-blurred board is common during physical movement
        # and must not pay for ten large ArUco scans on every live cycle.
        recovered = _aprilgrid_contour_fallback(
            search,
            dictionary,
            board_size,
            square,
            tag_spacing,
            start_id,
            min_tags,
        )
    # Do not upscale this search image. AprilTag payload bits discarded by the
    # low-resolution decode cannot be reconstructed by interpolation. The
    # service uses rejected quads as evidence and retries by decoding the
    # original JPEG into a larger, still-bounded working plane.
    image_points: List[np.ndarray] = []
    object_points: List[np.ndarray] = []
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
            image_points.append(image)
            object_points.append(obj)
    if recovered is not None:
        image_points, object_points = recovered
        if base_scale != 1.0:
            image_points = [
                (np.asarray(points, dtype=np.float32) / base_scale).astype(np.float32)
                for points in image_points
            ]
    elif len(image_points) < int(min_tags):
        return None
    image_points = _align_aprilgrid_corners_to_lattice(
        image_points, object_points
    )
    tag_image_points = np.asarray(image_points, dtype=np.float32).reshape(-1, 4, 2)
    tag_object_points = np.asarray(object_points, dtype=np.float32).reshape(-1, 4, 3)
    pixels = tag_image_points.reshape(-1, 1, 2)
    refined = pixels.copy()
    flat = pixels.reshape(-1, 2)
    valid = (
        (flat[:, 0] >= 4.0)
        & (flat[:, 0] < float(width - 4))
        & (flat[:, 1] >= 4.0)
        & (flat[:, 1] < float(height - 4))
    )
    if bool(np.any(valid)):
        refined[valid] = cv2.cornerSubPix(
            gray, pixels[valid].copy(), (3, 3), (-1, -1), _SUBPIX_CRITERIA
        )
    pixels = refined
    objects = tag_object_points.reshape(-1, 3)
    return BoardDetection(
        image_points=pixels,
        object_points=objects,
        coverage=_aprilgrid_coverage(
            pixels,
            objects,
            board_size,
            square,
            tag_spacing,
            width,
            height,
        ),
        calibration_image_points=np.mean(pixels.reshape(-1, 4, 2), axis=1)
        .reshape(-1, 1, 2)
        .astype(np.float32),
        calibration_object_points=np.mean(tag_object_points, axis=1)
        .reshape(-1, 3)
        .astype(np.float32),
    )


def _aprilgrid_coverage(
    pixels: np.ndarray,
    objects: np.ndarray,
    board_size: Sequence[int],
    tag_size: float,
    tag_spacing: float,
    width: int,
    height: int,
) -> Tuple[float, float, float, float]:
    """Project the complete AprilGrid boundary from the detected tag lattice."""
    homography, _mask = cv2.findHomography(
        np.asarray(objects, dtype=np.float32)[:, :2],
        np.asarray(pixels, dtype=np.float32).reshape(-1, 2),
        method=0,
    )
    cols, rows = int(board_size[0]), int(board_size[1])
    pitch = float(tag_size) + float(tag_spacing)
    board_width = float(tag_size) + float(cols - 1) * pitch
    board_height = float(tag_size) + float(rows - 1) * pitch
    boundary = np.asarray(
        ((0.0, 0.0), (board_width, 0.0), (board_width, board_height), (0.0, board_height)),
        dtype=np.float32,
    ).reshape(-1, 1, 2)
    projected = cv2.perspectiveTransform(boundary, homography)
    return _coverage_params_from_points(projected, width, height)


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


def is_new_sample(
    params: Sequence[float], samples: Sequence[Sequence[float]], threshold: float = SAMPLE_DISTANCE
) -> bool:
    """True if params are far enough (L1) from every already-collected sample."""
    if not samples:
        return True
    distance = min(sum(abs(a - b) for a, b in zip(params, sample)) for sample in samples)
    return distance > threshold


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
    goodenough = (len(samples) >= 40) or all(value >= 1.0 for value in progress)
    bars = [{"label": name, "progress": float(value)} for name, value in zip(PARAM_NAMES, progress)]
    return bars, goodenough


def next_view_guidance(
    samples: Sequence[Sequence[float]], ranges: Sequence[float] = PARAM_RANGES
) -> Dict[str, Any]:
    """Describe the single view that expands the weakest coverage axis most.

    X and Y need samples on both sides of the image, while Size and Skew are
    rewarded by their largest observed value.  Returning a small semantic
    document keeps presentation/localization in the WebUI and gives a physical
    operator a stable direction based on the whole sample history rather than
    whichever frame happened to arrive last.
    """
    bars, _goodenough = coverage(samples, ranges)
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


def calibrate_intrinsic(
    image_points: Sequence[np.ndarray],
    board_size: Sequence[int],
    square: float,
    image_size: Sequence[int],
    object_points: Optional[Sequence[np.ndarray]] = None,
) -> IntrinsicResult:
    """Estimate K and distortion from the collected corner sets via cv2.calibrateCamera."""
    if len(image_points) < 3:
        raise CalibrationError("need at least three samples to calibrate")
    if object_points is None:
        obj = board_object_points(board_size, square)
        object_points = [obj for _ in image_points]
    if len(object_points) != len(image_points):
        raise CalibrationError("each sample must provide matching image and object points")
    prepared_object = []
    corners = []
    for image, obj in zip(image_points, object_points):
        pixels = np.asarray(image, dtype=np.float32).reshape(-1, 1, 2)
        world = np.asarray(obj, dtype=np.float32).reshape(-1, 3)
        if len(pixels) != len(world) or len(pixels) < 4:
            raise CalibrationError("each sample must contain at least four corresponding points")
        corners.append(pixels)
        prepared_object.append(world)
    size = (int(image_size[0]), int(image_size[1]))
    rms, camera_matrix, distortion, _rvecs, _tvecs = cv2.calibrateCamera(
        prepared_object, corners, size, None, None
    )
    camera_matrix = np.asarray(camera_matrix, dtype=np.float64)
    if not np.all(np.isfinite(camera_matrix)) or camera_matrix[0, 0] <= 0.0:
        raise CalibrationError("calibration produced a degenerate camera matrix")
    return IntrinsicResult(
        camera_matrix=camera_matrix,
        distortion=np.asarray(distortion, dtype=np.float64).reshape(-1),
        image_size=size,
        rms_reprojection_error_px=float(rms),
        sample_count=len(image_points),
    )


def intrinsic_document(
    result: IntrinsicResult,
    *,
    board_size: Sequence[int],
    square: float,
    metadata: Optional[Dict[str, Any]] = None,
    board: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    k = result.camera_matrix
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
        "image_width": result.image_size[0],
        "image_height": result.image_size[1],
        "camera_matrix": {"rows": 3, "cols": 3, "data": [float(v) for v in k.reshape(-1)]},
        "distortion_model": "plumb_bob",
        "distortion_coefficients": {
            "rows": 1,
            "cols": int(result.distortion.size),
            "data": [float(v) for v in result.distortion],
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
    board_size: Sequence[int],
    square: float,
    metadata: Optional[Dict[str, Any]] = None,
    board: Optional[Dict[str, Any]] = None,
) -> Path:
    """Atomically persist a versioned intrinsic document outside package share."""
    destination = Path(path).expanduser()
    destination.parent.mkdir(parents=True, exist_ok=True)
    document = intrinsic_document(
        result, board_size=board_size, square=square, metadata=metadata, board=board
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
