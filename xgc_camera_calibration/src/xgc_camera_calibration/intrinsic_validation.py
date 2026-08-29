"""Visual validation products for one solved intrinsic calibration.

This module applies an existing plumb_bob K/D model to one immutable camera
frame. It does not collect calibration samples or solve intrinsics again.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, Mapping, Tuple

import cv2
import numpy as np

from xgc_camera_calibration.solver import CalibrationError


@dataclass(frozen=True)
class IntrinsicValidationResult:
    report: Dict[str, Any]
    images: Dict[str, bytes]


_VIEWS = (
    ("overlay_checker", "Grid comparison", "Alternating raw and undistorted tiles expose line displacement."),
    ("overlay_redcyan", "Red / cyan overlay", "Raw red and undistorted cyan align to grey where warp is small."),
    ("overlay_corner_zoom", "Maximum-warp detail", "Three-times crop around the largest remap displacement."),
    ("overlay_diff", "Difference heatmap", "Amplified absolute pixel difference over the captured frame."),
    ("displacement", "Distortion displacement", "Pixel remap magnitude independent of scene texture."),
    ("compare", "Raw / undistorted", "Side-by-side reference view."),
    ("raw", "Raw capture", "Immutable source snapshot used by this validation."),
    ("undistorted", "Undistorted", "The same snapshot after applying the selected K/D model."),
)


def generate_intrinsic_validation(
    raw: np.ndarray,
    document: Mapping[str, Any],
    *,
    calibration_id: str,
    tile: int = 48,
    jpeg_quality: int = 90,
) -> IntrinsicValidationResult:
    if not isinstance(raw, np.ndarray) or raw.ndim != 3 or raw.shape[2] != 3:
        raise CalibrationError("intrinsic validation requires a BGR color frame")
    if tile < 8 or not 1 <= jpeg_quality <= 100:
        raise ValueError("invalid intrinsic validation render parameters")

    source_height, source_width = raw.shape[:2]
    image = raw
    height, width = image.shape[:2]
    matrix, distortion, calibration_size = intrinsic_parameters(document)
    matrix = scale_intrinsics(matrix, calibration_size, (width, height))

    map_x, map_y = cv2.initUndistortRectifyMap(
        matrix, distortion, None, matrix, (width, height), cv2.CV_32FC1
    )
    undistorted = cv2.remap(image, map_x, map_y, cv2.INTER_LINEAR)
    shift_x = map_x
    shift_x -= np.arange(width, dtype=np.float32)[None, :]
    shift_y = map_y
    shift_y -= np.arange(height, dtype=np.float32)[:, None]
    magnitude = np.sqrt(shift_x * shift_x + shift_y * shift_y)

    checker = undistorted.copy()
    x_tiles = (np.arange(width, dtype=np.int32) // tile)[None, :]
    y_tiles = (np.arange(height, dtype=np.int32) // tile)[:, None]
    checker_mask = (x_tiles + y_tiles) % 2 == 0
    checker[checker_mask] = image[checker_mask]
    for x in range(tile, width, tile):
        checker[:, x] = (checker[:, x] * 0.45 + (40, 40, 40)).astype(np.uint8)
    for y in range(tile, height, tile):
        checker[y, :] = (checker[y, :] * 0.45 + (40, 40, 40)).astype(np.uint8)

    red_cyan = np.zeros_like(image)
    red_cyan[:, :, 2] = image[:, :, 2]
    red_cyan[:, :, 1] = undistorted[:, :, 1]
    red_cyan[:, :, 0] = undistorted[:, :, 0]
    label(red_cyan, "RAW red + UNDISTORT cyan", (0, 255, 255))

    difference = cv2.absdiff(image, undistorted)
    amplified = np.clip(
        cv2.cvtColor(difference, cv2.COLOR_BGR2GRAY).astype(np.float32) * 8.0,
        0,
        255,
    ).astype(np.uint8)
    difference_heatmap = cv2.applyColorMap(amplified, cv2.COLORMAP_INFERNO)
    difference_heatmap = cv2.addWeighted(image, 0.35, difference_heatmap, 0.65, 0)
    label(difference_heatmap, "absolute difference x8", (255, 255, 255))

    magnitude_u8 = np.clip(
        magnitude / max(float(magnitude.max()), 1e-6) * 255.0,
        0,
        255,
    ).astype(np.uint8)
    displacement = cv2.applyColorMap(magnitude_u8, cv2.COLORMAP_JET)
    label(displacement, "remap displacement magnitude", (255, 255, 255))

    compare = np.full((height, width * 2 + 8, 3), 24, np.uint8)
    compare[:, :width] = image
    compare[:, width + 8:] = undistorted
    label(compare[:, :width], "raw", (0, 0, 255))
    label(compare[:, width + 8:], "undistorted", (0, 220, 80))

    corner_detail = corner_zoom(image, undistorted, checker, magnitude)
    encoded = {}
    encoded["overlay_checker"] = encode_jpeg(checker, jpeg_quality)
    del checker
    encoded["overlay_redcyan"] = encode_jpeg(red_cyan, jpeg_quality)
    del red_cyan
    encoded["overlay_corner_zoom"] = encode_jpeg(corner_detail, jpeg_quality)
    del corner_detail
    encoded["overlay_diff"] = encode_jpeg(difference_heatmap, jpeg_quality)
    del difference_heatmap
    encoded["displacement"] = encode_jpeg(displacement, jpeg_quality)
    del displacement
    encoded["compare"] = encode_jpeg(compare, jpeg_quality)
    del compare
    encoded["raw"] = encode_jpeg(image, jpeg_quality)
    encoded["undistorted"] = encode_jpeg(undistorted, jpeg_quality)
    corners = []
    for u, v in ((0, 0), (width - 1, 0), (0, height - 1), (width - 1, height - 1)):
        corners.append({
            "uv": [u, v],
            "shift_px": [float(shift_x[v, u]), float(shift_y[v, u])],
            "magnitude_px": float(magnitude[v, u]),
        })
    return IntrinsicValidationResult(
        report={
            "schema": "xgc2.camera.intrinsic-validation.v1",
            "captured_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "calibration_id": calibration_id,
            "calibration_created_at": str(document.get("created_at", "")),
            "source_image_size": [source_width, source_height],
            "analysis_image_size": [width, height],
            "remap_px": {
                "mean": float(magnitude.mean()),
                "maximum": float(magnitude.max()),
            },
            "corners": corners,
            "default_view": "overlay_checker",
            "views": [
                {"id": view_id, "label": view_label, "description": description}
                for view_id, view_label, description in _VIEWS
            ],
        },
        images=encoded,
    )


def intrinsic_parameters(
    document: Mapping[str, Any],
) -> Tuple[np.ndarray, np.ndarray, Tuple[int, int]]:
    matrix = document.get("camera_matrix_array")
    if matrix is None:
        camera_matrix = document.get("camera_matrix", {})
        matrix = camera_matrix.get("data") if isinstance(camera_matrix, Mapping) else None
    matrix = np.asarray(matrix, dtype=np.float64).reshape(-1)
    distortion_document = document.get("distortion_coefficients", {})
    distortion = (
        distortion_document.get("data")
        if isinstance(distortion_document, Mapping)
        else distortion_document
    )
    distortion = np.asarray(distortion, dtype=np.float64).reshape(-1)
    width = int(document.get("image_width", 0))
    height = int(document.get("image_height", 0))
    if matrix.size != 9 or distortion.size < 4 or width <= 0 or height <= 0:
        raise CalibrationError("intrinsic validation calibration is incomplete")
    matrix = matrix.reshape(3, 3)
    if not np.all(np.isfinite(matrix)) or not np.all(np.isfinite(distortion)):
        raise CalibrationError("intrinsic validation calibration contains non-finite values")
    return matrix, distortion, (width, height)


def scale_intrinsics(
    matrix: np.ndarray,
    calibration_size: Tuple[int, int],
    image_size: Tuple[int, int],
) -> np.ndarray:
    calibration_width, calibration_height = calibration_size
    width, height = image_size
    scaled = matrix.copy()
    scaled[0, 0] *= float(width) / float(calibration_width)
    scaled[0, 2] *= float(width) / float(calibration_width)
    scaled[1, 1] *= float(height) / float(calibration_height)
    scaled[1, 2] *= float(height) / float(calibration_height)
    return scaled


def encode_jpeg(image: np.ndarray, quality: int) -> bytes:
    ok, encoded = cv2.imencode(
        ".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)]
    )
    if not ok:
        raise CalibrationError("could not encode intrinsic validation image")
    return encoded.tobytes()


def label(image: np.ndarray, text: str, color: Tuple[int, int, int]) -> None:
    cv2.putText(
        image, text, (16, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2, cv2.LINE_AA
    )


def corner_zoom(
    raw: np.ndarray,
    undistorted: np.ndarray,
    checker: np.ndarray,
    magnitude: np.ndarray,
) -> np.ndarray:
    height, width = raw.shape[:2]
    crop_width, crop_height = min(280, width), min(200, height)
    row, column = np.unravel_index(int(np.argmax(magnitude)), magnitude.shape)
    x0 = int(np.clip(column - crop_width // 2, 0, width - crop_width))
    y0 = int(np.clip(row - crop_height // 2, 0, height - crop_height))
    scale, gap = 3, 8

    def zoom(image: np.ndarray) -> np.ndarray:
        crop = image[y0:y0 + crop_height, x0:x0 + crop_width]
        return cv2.resize(
            crop,
            (crop_width * scale, crop_height * scale),
            interpolation=cv2.INTER_NEAREST,
        )

    panels = (zoom(raw), zoom(undistorted), zoom(checker))
    zoom_height, zoom_width = panels[0].shape[:2]
    strip = np.full((zoom_height, zoom_width * 3 + gap * 2, 3), 18, np.uint8)
    strip[:, :zoom_width] = panels[0]
    strip[:, zoom_width + gap:zoom_width * 2 + gap] = panels[1]
    strip[:, zoom_width * 2 + gap * 2:] = panels[2]
    label(strip[:, :zoom_width], "RAW 3x", (0, 0, 255))
    label(strip[:, zoom_width + gap:zoom_width * 2 + gap], "UNDISTORT 3x", (0, 220, 80))
    label(strip[:, zoom_width * 2 + gap * 2:], "CHECKER 3x", (0, 200, 255))
    return strip
