"""Projective pose-coverage primitives for planar intrinsic targets.

Image-plane rotation is not evidence of out-of-plane target diversity. This
module estimates the board plane normal from its object/image homography and a
provisional camera matrix, keeping signed tilt around both camera axes. Roll is
retained only as a diagnostic.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Sequence, Tuple

import cv2
import numpy as np


@dataclass(frozen=True)
class PlaneOrientation:
    """One planar target orientation in the OpenCV camera frame."""

    normal: Tuple[float, float, float]
    tilt_x_degrees: float
    tilt_y_degrees: float
    roll_degrees: float
    homography_rms_px: float


def estimate_provisional_camera_matrix(
    object_point_sets: Sequence[np.ndarray],
    image_point_sets: Sequence[np.ndarray],
    image_size: Sequence[int],
    aspect_ratio: float = 0.0,
) -> np.ndarray:
    """Initialize K from multiple planar homographies using Zhang's method.

    Physical calibration cannot assume that Media Edge's source K is already
    calibrated. This initializer provides the provisional K needed only for
    pose-coverage classification; the final distortion solve remains separate.
    """

    if (
        len(object_point_sets) != len(image_point_sets)
        or len(object_point_sets) < 3
    ):
        raise ValueError("provisional intrinsics need at least three paired views")
    if len(image_size) != 2 or int(image_size[0]) < 2 or int(image_size[1]) < 2:
        raise ValueError("image size must contain positive width and height")
    ratio = float(aspect_ratio)
    if not math.isfinite(ratio) or ratio < 0.0:
        raise ValueError("aspect ratio must be finite and non-negative")
    objects = []
    images = []
    for object_points, image_points in zip(object_point_sets, image_point_sets):
        object_array = np.asarray(object_points, dtype=np.float32).reshape(-1, 3)
        image_array = np.asarray(image_points, dtype=np.float32).reshape(-1, 1, 2)
        if len(object_array) != len(image_array) or len(object_array) < 4:
            raise ValueError("each provisional-intrinsic view needs four paired points")
        if float(np.ptp(object_array[:, 2])) > 1e-6:
            raise ValueError("provisional-intrinsic object points must be planar")
        if (
            not bool(np.all(np.isfinite(object_array)))
            or not bool(np.all(np.isfinite(image_array)))
        ):
            raise ValueError("provisional-intrinsic correspondences must be finite")
        objects.append(object_array)
        images.append(image_array)
    matrix = cv2.initCameraMatrix2D(
        objects,
        images,
        (int(image_size[0]), int(image_size[1])),
        aspectRatio=ratio,
    )
    matrix = np.asarray(matrix, dtype=np.float64)
    if (
        matrix.shape != (3, 3)
        or not bool(np.all(np.isfinite(matrix)))
        or float(matrix[0, 0]) <= 0.0
        or float(matrix[1, 1]) <= 0.0
    ):
        raise ValueError("planar views did not yield valid provisional intrinsics")
    return matrix


def estimate_plane_orientation(
    object_points: np.ndarray,
    image_points: np.ndarray,
    camera_matrix: np.ndarray,
) -> PlaneOrientation:
    """Estimate a planar board normal from paired object/image points.

    ``camera_matrix`` may be a provisional intrinsic estimate while samples are
    being collected. It affects tilt magnitude, but unlike a bounding-box angle
    the homography decomposition is invariant to in-plane roll and preserves
    the sign of perspective tilt.
    """

    object_array = np.asarray(object_points, dtype=np.float64)
    image_array = np.asarray(image_points, dtype=np.float64)
    if object_array.ndim < 2 or object_array.shape[-1] not in (2, 3):
        raise ValueError("object points must contain two or three coordinates")
    objects = object_array.reshape(-1, object_array.shape[-1])
    images = image_array.reshape(-1, 2)
    matrix = np.asarray(camera_matrix, dtype=np.float64)
    if len(objects) != len(images) or len(objects) < 4:
        raise ValueError("plane orientation needs at least four paired points")
    if objects.shape[1] == 3 and float(np.ptp(objects[:, 2])) > 1e-9:
        raise ValueError("object points must lie on one XY plane")
    if matrix.shape != (3, 3) or not bool(np.all(np.isfinite(matrix))):
        raise ValueError("camera matrix must be finite and 3x3")
    if (
        not bool(np.all(np.isfinite(objects)))
        or not bool(np.all(np.isfinite(images)))
    ):
        raise ValueError("plane correspondences must be finite")

    object_xy = objects[:, :2]
    homography, _mask = cv2.findHomography(object_xy, images, method=0)
    if homography is None or not bool(np.all(np.isfinite(homography))):
        raise ValueError("plane correspondences do not define a homography")
    try:
        inverse_camera = np.linalg.inv(matrix)
    except np.linalg.LinAlgError as error:
        raise ValueError("camera matrix must be invertible") from error

    axis_x = inverse_camera.dot(homography[:, 0])
    axis_y = inverse_camera.dot(homography[:, 1])
    translation = inverse_camera.dot(homography[:, 2])
    norm_x = float(np.linalg.norm(axis_x))
    norm_y = float(np.linalg.norm(axis_y))
    if min(norm_x, norm_y) <= 1e-12:
        raise ValueError("homography has a degenerate plane basis")
    scale = 2.0 / (norm_x + norm_y)
    axis_x *= scale
    axis_y *= scale
    translation *= scale
    # Homographies are defined only up to a global sign. Choose the solution
    # whose board is in front of the camera so roll is deterministic too.
    if float(translation[2]) < 0.0:
        axis_x *= -1.0
        axis_y *= -1.0

    axis_z = np.cross(axis_x, axis_y)
    approximate_rotation = np.column_stack((axis_x, axis_y, axis_z))
    left, _singular, right = np.linalg.svd(approximate_rotation)
    rotation = left.dot(right)
    if float(np.linalg.det(rotation)) < 0.0:
        left[:, -1] *= -1.0
        rotation = left.dot(right)

    normal = rotation[:, 2]
    # A plane normal has a two-way sign ambiguity. Canonicalize it toward the
    # camera's forward hemisphere before assigning signed tilt bins.
    if float(normal[2]) < 0.0:
        normal = -normal
    tilt_x = math.degrees(math.atan2(float(normal[0]), float(normal[2])))
    tilt_y = math.degrees(math.atan2(float(normal[1]), float(normal[2])))
    roll = math.degrees(math.atan2(float(rotation[1, 0]), float(rotation[0, 0])))

    projected = cv2.perspectiveTransform(
        object_xy.astype(np.float64).reshape(-1, 1, 2), homography
    ).reshape(-1, 2)
    residual = projected - images
    homography_rms = math.sqrt(float(np.mean(np.sum(residual * residual, axis=1))))
    return PlaneOrientation(
        normal=tuple(float(value) for value in normal),
        tilt_x_degrees=float(tilt_x),
        tilt_y_degrees=float(tilt_y),
        roll_degrees=float(roll),
        homography_rms_px=float(homography_rms),
    )


def signed_tilt_bins(
    observations: Sequence[PlaneOrientation], minimum_degrees: float
) -> Dict[str, bool]:
    """Return independent positive/negative coverage for both normal axes."""

    threshold = float(minimum_degrees)
    if not math.isfinite(threshold) or threshold <= 0.0 or threshold >= 90.0:
        raise ValueError("minimum tilt must be between zero and 90 degrees")
    bins = {
        "x_negative": any(item.tilt_x_degrees <= -threshold for item in observations),
        "x_positive": any(item.tilt_x_degrees >= threshold for item in observations),
        "y_negative": any(item.tilt_y_degrees <= -threshold for item in observations),
        "y_positive": any(item.tilt_y_degrees >= threshold for item in observations),
    }
    bins["complete"] = all(bins.values())
    return bins
