"""Visual validation products for one solved intrinsic calibration.

This module applies an existing plumb_bob K/D model to one immutable camera
frame. It does not collect calibration samples or solve intrinsics again.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, Mapping, Optional, Tuple

import cv2
import numpy as np

from xgc_camera_calibration.solver import CalibrationError


@dataclass(frozen=True)
class IntrinsicValidationResult:
    report: Dict[str, Any]
    images: Dict[str, bytes]


_VIEWS = (
    ("overlay_checker", "Grid comparison", "Alternating reference and comparison tiles expose line displacement."),
    ("overlay_redcyan", "Red / cyan overlay", "Reference red and comparison cyan align to grey where both configurations agree."),
    ("overlay_corner_zoom", "Maximum-warp detail", "Three-times crop around the largest remap difference."),
    ("overlay_diff", "Difference heatmap", "Amplified absolute pixel difference between both configurations."),
    ("displacement", "Remap difference", "Pixel remap difference between reference and comparison, independent of scene texture."),
    ("compare", "Reference / comparison", "Side-by-side output from the two selected configurations."),
    ("reference", "Reference", "The immutable source snapshot after applying the reference configuration."),
    ("comparison", "Comparison", "The same immutable snapshot after applying the comparison configuration."),
)

_LABEL_SURFACE_BGR = (20, 22, 24)  # XGC dark surface #181614.
_LABEL_BORDER_BGR = (46, 51, 56)  # XGC dark border #38332e.
_LABEL_ACCENT_BGR = (111, 160, 208)  # XGC dark accent #d0a06f.
_LABEL_TEXT_BGR = (220, 228, 233)  # XGC dark heading text #e9e4dc.


def generate_intrinsic_comparison(
    raw: np.ndarray,
    reference_document: Optional[Mapping[str, Any]],
    comparison_document: Optional[Mapping[str, Any]],
    *,
    reference_calibration_id: Optional[str] = None,
    comparison_calibration_id: Optional[str] = None,
    tile: int = 48,
    jpeg_quality: int = 90,
) -> IntrinsicValidationResult:
    """Compare two configurations on exactly one immutable source frame.

    A ``None`` document is the raw/identity configuration. Calibration IDs are
    report provenance only and are required exactly when their document is
    present; path resolution remains the service's responsibility.
    """
    _validate_configuration(reference_document, reference_calibration_id, "reference")
    _validate_configuration(comparison_document, comparison_calibration_id, "comparison")
    same_configuration = (
        reference_document is None and comparison_document is None
    ) or (
        reference_document is not None
        and comparison_document is not None
        and reference_calibration_id == comparison_calibration_id
    )
    products = _generate_comparison_products(
        raw,
        reference_document,
        comparison_document,
        same_configuration=same_configuration,
        reference_legend="REFERENCE",
        comparison_legend="COMPARISON",
        tile=tile,
        jpeg_quality=jpeg_quality,
    )
    reference_report = _configuration_report(
        reference_document, reference_calibration_id
    )
    comparison_report = _configuration_report(
        comparison_document, comparison_calibration_id
    )
    return IntrinsicValidationResult(
        report={
            "schema": "xgc2.camera.intrinsic-validation.v2",
            "captured_at": products.report["captured_at"],
            "configurations": {
                "reference": reference_report,
                "comparison": comparison_report,
            },
            "source_image_size": products.report["source_image_size"],
            "analysis_image_size": products.report["analysis_image_size"],
            "remap_delta_px": products.report["remap_delta_px"],
            "corners": products.report["corners"],
            "default_view": "overlay_checker",
            "views": [
                {"id": view_id, "label": view_label, "description": description}
                for view_id, view_label, description in _VIEWS
            ],
        },
        images=products.images,
    )


def _generate_comparison_products(
    raw: np.ndarray,
    reference_document: Optional[Mapping[str, Any]],
    comparison_document: Optional[Mapping[str, Any]],
    *,
    same_configuration: bool,
    reference_legend: str,
    comparison_legend: str,
    tile: int,
    jpeg_quality: int,
) -> IntrinsicValidationResult:
    if not isinstance(raw, np.ndarray) or raw.ndim != 3 or raw.shape[2] != 3:
        raise CalibrationError("intrinsic validation requires a BGR color frame")
    if tile < 8 or not 1 <= jpeg_quality <= 100:
        raise ValueError("invalid intrinsic validation render parameters")

    source_height, source_width = raw.shape[:2]
    height, width = raw.shape[:2]
    reference, reference_maps = _apply_configuration(raw, reference_document)
    if same_configuration:
        comparison = reference
        reference_maps = None
        delta_x = np.zeros((height, width), dtype=np.float32)
        delta_y = np.zeros((height, width), dtype=np.float32)
    else:
        comparison, comparison_maps = _apply_configuration(raw, comparison_document)
        delta_x, delta_y = _remap_delta(
            reference_maps,
            comparison_maps,
            width,
            height,
        )
        reference_maps = None
        comparison_maps = None
    magnitude = cv2.magnitude(delta_x, delta_y)

    checker = comparison.copy()
    x_tiles = (np.arange(width, dtype=np.int32) // tile)[None, :]
    y_tiles = (np.arange(height, dtype=np.int32) // tile)[:, None]
    checker_mask = (x_tiles + y_tiles) % 2 == 0
    checker[checker_mask] = reference[checker_mask]
    for x in range(tile, width, tile):
        checker[:, x] = (checker[:, x] * 0.45 + (40, 40, 40)).astype(np.uint8)
    for y in range(tile, height, tile):
        checker[y, :] = (checker[y, :] * 0.45 + (40, 40, 40)).astype(np.uint8)
    label(checker, "{} EVEN  /  {} ODD".format(reference_legend, comparison_legend))

    red_cyan = np.zeros_like(raw)
    red_cyan[:, :, 2] = reference[:, :, 2]
    red_cyan[:, :, 1] = comparison[:, :, 1]
    red_cyan[:, :, 0] = comparison[:, :, 0]
    label(red_cyan, "{} RED  /  {} CYAN".format(reference_legend, comparison_legend))

    difference = cv2.absdiff(reference, comparison)
    amplified = np.clip(
        cv2.cvtColor(difference, cv2.COLOR_BGR2GRAY).astype(np.float32) * 8.0,
        0,
        255,
    ).astype(np.uint8)
    difference_heatmap = cv2.applyColorMap(amplified, cv2.COLORMAP_INFERNO)
    difference_heatmap = cv2.addWeighted(reference, 0.35, difference_heatmap, 0.65, 0)
    label(difference_heatmap, "ABSOLUTE DIFFERENCE  x8")
    del difference, amplified

    magnitude_u8 = np.clip(
        magnitude / max(float(magnitude.max()), 1e-6) * 255.0,
        0,
        255,
    ).astype(np.uint8)
    displacement = cv2.applyColorMap(magnitude_u8, cv2.COLORMAP_JET)
    label(displacement, "REMAP DIFFERENCE MAGNITUDE")
    del magnitude_u8

    compare = np.full((height, width * 2 + 8, 3), 24, np.uint8)
    compare[:, :width] = reference
    compare[:, width + 8:] = comparison
    label(compare[:, :width], reference_legend)
    label(compare[:, width + 8:], comparison_legend)

    corner_detail = corner_zoom(
        reference,
        comparison,
        checker,
        magnitude,
        reference_legend,
        comparison_legend,
    )
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
    encoded["reference"] = encode_jpeg(reference, jpeg_quality)
    encoded["comparison"] = (
        encoded["reference"] if comparison is reference
        else encode_jpeg(comparison, jpeg_quality)
    )
    corners = []
    for u, v in ((0, 0), (width - 1, 0), (0, height - 1), (width - 1, height - 1)):
        corners.append({
            "uv": [u, v],
            "delta_px": [float(delta_x[v, u]), float(delta_y[v, u])],
            "magnitude_px": float(magnitude[v, u]),
        })
    return IntrinsicValidationResult(
        report={
            "captured_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "source_image_size": [source_width, source_height],
            "analysis_image_size": [width, height],
            "remap_delta_px": {
                "mean": float(magnitude.mean()),
                "maximum": float(magnitude.max()),
            },
            "corners": corners,
        },
        images=encoded,
    )


def _validate_configuration(
    document: Optional[Mapping[str, Any]],
    calibration_id: Optional[str],
    name: str,
) -> None:
    if document is None:
        if calibration_id is not None:
            raise ValueError("{} raw configuration must not have calibration_id".format(name))
        return
    if not isinstance(calibration_id, str) or not calibration_id:
        raise ValueError("{} calibration configuration requires calibration_id".format(name))


def _configuration_report(
    document: Optional[Mapping[str, Any]],
    calibration_id: Optional[str],
) -> Dict[str, Any]:
    if document is None:
        return {"kind": "raw"}
    return {
        "kind": "calibration",
        "calibration_id": calibration_id,
        "calibration_created_at": str(document.get("created_at", "")),
    }


def _apply_configuration(
    raw: np.ndarray,
    document: Optional[Mapping[str, Any]],
) -> Tuple[np.ndarray, Optional[Tuple[np.ndarray, np.ndarray]]]:
    if document is None:
        return raw, None
    height, width = raw.shape[:2]
    matrix, distortion, calibration_size = intrinsic_parameters(document)
    matrix = scale_intrinsics(matrix, calibration_size, (width, height))
    map_x, map_y = cv2.initUndistortRectifyMap(
        matrix, distortion, None, matrix, (width, height), cv2.CV_32FC1
    )
    return cv2.remap(raw, map_x, map_y, cv2.INTER_LINEAR), (map_x, map_y)


def _remap_delta(
    reference_maps: Optional[Tuple[np.ndarray, np.ndarray]],
    comparison_maps: Optional[Tuple[np.ndarray, np.ndarray]],
    width: int,
    height: int,
) -> Tuple[np.ndarray, np.ndarray]:
    x_coordinates = np.arange(width, dtype=np.float32)[None, :]
    y_coordinates = np.arange(height, dtype=np.float32)[:, None]
    if comparison_maps is not None:
        delta_x, delta_y = comparison_maps
        if reference_maps is None:
            delta_x -= x_coordinates
            delta_y -= y_coordinates
        else:
            delta_x -= reference_maps[0]
            delta_y -= reference_maps[1]
        return delta_x, delta_y
    if reference_maps is not None:
        delta_x, delta_y = reference_maps
        delta_x *= -1.0
        delta_x += x_coordinates
        delta_y *= -1.0
        delta_y += y_coordinates
        return delta_x, delta_y
    return (
        np.zeros((height, width), dtype=np.float32),
        np.zeros((height, width), dtype=np.float32),
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
    distortion_model = str(document.get("distortion_model", "plumb_bob"))
    if (
        matrix.size != 9
        or distortion.size not in (4, 5, 8, 12, 14)
        or width <= 0
        or height <= 0
    ):
        raise CalibrationError("intrinsic validation calibration is incomplete")
    if distortion_model != "plumb_bob":
        raise CalibrationError(
            "intrinsic validation supports only plumb_bob calibration"
        )
    matrix = matrix.reshape(3, 3)
    if not np.all(np.isfinite(matrix)) or not np.all(np.isfinite(distortion)):
        raise CalibrationError("intrinsic validation calibration contains non-finite values")
    if matrix[0, 0] <= 0.0 or matrix[1, 1] <= 0.0:
        raise CalibrationError("intrinsic validation calibration has invalid focal length")
    return matrix, distortion, (width, height)


def scale_intrinsics(
    matrix: np.ndarray,
    calibration_size: Tuple[int, int],
    image_size: Tuple[int, int],
) -> np.ndarray:
    calibration_width, calibration_height = calibration_size
    width, height = image_size
    scale_x = float(width) / float(calibration_width)
    scale_y = float(height) / float(calibration_height)
    if not np.isclose(scale_x, scale_y, rtol=1e-6, atol=1e-9):
        raise CalibrationError(
            "intrinsic validation frame aspect ratio does not match calibration"
        )
    scaled = matrix.copy()
    scaled[0, 0] *= scale_x
    scaled[0, 2] *= scale_x
    scaled[1, 1] *= scale_y
    scaled[1, 2] *= scale_y
    return scaled


def encode_jpeg(image: np.ndarray, quality: int) -> bytes:
    ok, encoded = cv2.imencode(
        ".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)]
    )
    if not ok:
        raise CalibrationError("could not encode intrinsic validation image")
    return encoded.tobytes()


def label(image: np.ndarray, text: str) -> None:
    """Burn a compact XGC-themed legend into an exported validation image."""
    height, width = image.shape[:2]
    font_scale = float(np.clip(min(width, height) / 900.0, 0.65, 1.35))
    thickness = max(1, int(round(font_scale * 1.5)))
    margin = max(8, int(round(12 * font_scale)))
    padding_x = max(9, int(round(11 * font_scale)))
    padding_y = max(6, int(round(7 * font_scale)))
    accent_width = max(3, int(round(4 * font_scale)))
    (text_width, text_height), baseline = cv2.getTextSize(
        text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness
    )
    box_width = min(width - margin, accent_width + padding_x * 2 + text_width)
    box_height = min(height - margin, padding_y * 2 + text_height + baseline)
    if box_width <= 0 or box_height <= 0:
        return

    x0, y0 = margin, margin
    x1, y1 = x0 + box_width, y0 + box_height
    region = image[y0:y1, x0:x1]
    surface = np.full_like(region, _LABEL_SURFACE_BGR)
    cv2.addWeighted(surface, 0.84, region, 0.16, 0.0, region)
    cv2.rectangle(image, (x0, y0), (x1 - 1, y1 - 1), _LABEL_BORDER_BGR, 1, cv2.LINE_AA)
    cv2.rectangle(
        image,
        (x0 + 1, y0 + 1),
        (min(x0 + accent_width, x1 - 1), y1 - 2),
        _LABEL_ACCENT_BGR,
        -1,
        cv2.LINE_AA,
    )
    cv2.putText(
        image,
        text,
        (x0 + accent_width + padding_x, y0 + padding_y + text_height),
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        _LABEL_TEXT_BGR,
        thickness,
        cv2.LINE_AA,
    )


def corner_zoom(
    reference: np.ndarray,
    comparison: np.ndarray,
    checker: np.ndarray,
    magnitude: np.ndarray,
    reference_legend: str,
    comparison_legend: str,
) -> np.ndarray:
    height, width = reference.shape[:2]
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

    panels = (zoom(reference), zoom(comparison), zoom(checker))
    zoom_height, zoom_width = panels[0].shape[:2]
    strip = np.full((zoom_height, zoom_width * 3 + gap * 2, 3), 18, np.uint8)
    strip[:, :zoom_width] = panels[0]
    strip[:, zoom_width + gap:zoom_width * 2 + gap] = panels[1]
    strip[:, zoom_width * 2 + gap * 2:] = panels[2]
    label(strip[:, :zoom_width], "{} 3x".format(reference_legend))
    label(
        strip[:, zoom_width + gap:zoom_width * 2 + gap],
        "{} 3x".format(comparison_legend),
    )
    label(strip[:, zoom_width * 2 + gap * 2:], "CHECKER 3x")
    return strip
