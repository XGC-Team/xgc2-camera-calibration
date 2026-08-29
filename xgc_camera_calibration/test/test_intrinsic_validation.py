import unittest

import cv2
import numpy as np

from xgc_camera_calibration.intrinsic_validation import (
    generate_intrinsic_comparison,
)
from xgc_camera_calibration.solver import CalibrationError


def intrinsic_document(
    width: int,
    height: int,
    k1: float,
    created_at: str,
):
    return {
        "created_at": created_at,
        "image_width": width,
        "image_height": height,
        "distortion_model": "plumb_bob",
        "camera_matrix": {
            "data": [
                width * 0.7, 0.0, width / 2.0,
                0.0, width * 0.7, height / 2.0,
                0.0, 0.0, 1.0,
            ],
        },
        "distortion_coefficients": {"data": [k1, 0.03, 0.0, 0.0, 0.0]},
    }


class IntrinsicValidationTest(unittest.TestCase):
    def test_v2_compares_two_calibrations_on_one_source_native_frame(self):
        width, height = 640, 360
        raw = np.empty((height, width, 3), dtype=np.uint8)
        raw[:, :, 0] = np.arange(width, dtype=np.uint16)[None, :] % 256
        raw[:, :, 1] = np.arange(height, dtype=np.uint16)[:, None] % 256
        raw[:, :, 2] = 127
        reference = intrinsic_document(width, height, -0.08, "2026-08-30T12:00:00Z")
        comparison = intrinsic_document(width, height, -0.22, "2026-08-30T13:00:00Z")

        validation = generate_intrinsic_comparison(
            raw,
            reference,
            comparison,
            reference_calibration_id="intrinsics-reference.yaml",
            comparison_calibration_id="intrinsics-comparison.yaml",
            jpeg_quality=95,
        )

        self.assertEqual(validation.report["schema"], "xgc2.camera.intrinsic-validation.v2")
        self.assertEqual(validation.report["source_image_size"], [width, height])
        self.assertEqual(validation.report["analysis_image_size"], [width, height])
        self.assertEqual(validation.report["configurations"]["reference"], {
            "kind": "calibration",
            "calibration_id": "intrinsics-reference.yaml",
            "calibration_created_at": "2026-08-30T12:00:00Z",
        })
        self.assertEqual(validation.report["configurations"]["comparison"], {
            "kind": "calibration",
            "calibration_id": "intrinsics-comparison.yaml",
            "calibration_created_at": "2026-08-30T13:00:00Z",
        })
        self.assertGreater(validation.report["remap_delta_px"]["maximum"], 0.0)
        self.assertEqual(
            [view["id"] for view in validation.report["views"]],
            [
                "overlay_checker", "overlay_redcyan", "overlay_corner_zoom",
                "overlay_diff", "displacement", "compare", "reference", "comparison",
            ],
        )
        self.assertEqual(validation.report["views"][6]["label"], "Reference")
        self.assertEqual(validation.report["views"][7]["label"], "Comparison")
        decoded_reference = cv2.imdecode(
            np.frombuffer(validation.images["reference"], dtype=np.uint8), cv2.IMREAD_COLOR
        )
        decoded_comparison = cv2.imdecode(
            np.frombuffer(validation.images["comparison"], dtype=np.uint8), cv2.IMREAD_COLOR
        )
        self.assertEqual(decoded_reference.shape[:2], (height, width))
        self.assertEqual(decoded_comparison.shape[:2], (height, width))

    def test_v2_allows_raw_against_raw_as_an_explicit_capture(self):
        raw = np.full((180, 320, 3), 91, dtype=np.uint8)

        validation = generate_intrinsic_comparison(raw, None, None, jpeg_quality=90)

        self.assertEqual(validation.report["configurations"]["reference"], {"kind": "raw"})
        self.assertEqual(validation.report["configurations"]["comparison"], {"kind": "raw"})
        self.assertEqual(validation.report["remap_delta_px"], {
            "mean": 0.0, "maximum": 0.0,
        })
        self.assertEqual(validation.images["reference"], validation.images["comparison"])

    def test_rejects_calibration_with_a_different_source_aspect_ratio(self):
        raw = np.full((360, 640, 3), 91, dtype=np.uint8)
        document = intrinsic_document(640, 480, -0.1, "2026-08-30T12:00:00Z")

        with self.assertRaisesRegex(CalibrationError, "aspect ratio"):
            generate_intrinsic_comparison(
                raw,
                None,
                document,
                comparison_calibration_id="intrinsics-4x3.yaml",
            )

    def test_checker_uses_theme_legend_instead_of_baked_in_red_text(self):
        width, height = 640, 480
        raw = np.full((height, width, 3), 128, dtype=np.uint8)
        document = {
            "image_width": width,
            "image_height": height,
            "camera_matrix": {
                "data": [500.0, 0.0, width / 2.0, 0.0, 500.0, height / 2.0, 0.0, 0.0, 1.0],
            },
            "distortion_coefficients": {"data": [0.0, 0.0, 0.0, 0.0, 0.0]},
        }

        validation = generate_intrinsic_comparison(
            raw, None, document,
            comparison_calibration_id="intrinsics-neutral.yaml", jpeg_quality=100
        )
        for view_id in (
            "overlay_checker",
            "overlay_redcyan",
            "overlay_corner_zoom",
            "overlay_diff",
            "displacement",
            "compare",
        ):
            rendered = cv2.imdecode(
                np.frombuffer(validation.images[view_id], dtype=np.uint8),
                cv2.IMREAD_COLOR,
            )
            label_region = rendered[8:52, 20:360].astype(np.int16)
            red_dominance = label_region[:, :, 2] - np.maximum(
                label_region[:, :, 0], label_region[:, :, 1]
            )

            with self.subTest(view_id=view_id):
                self.assertLess(int(np.percentile(red_dominance, 99.5)), 20)
                self.assertLess(float(rendered[16:42, 20:360].mean()), 120.0)

    def test_preserves_source_native_4k_without_analysis_downscaling(self):
        width, height = 3840, 2160
        x = np.linspace(0, 255, width, dtype=np.uint8)[None, :]
        y = np.linspace(0, 255, height, dtype=np.uint8)[:, None]
        raw = np.empty((height, width, 3), dtype=np.uint8)
        raw[:, :, 0] = x
        raw[:, :, 1] = y
        raw[:, :, 2] = ((x.astype(np.uint16) + y.astype(np.uint16)) // 2).astype(np.uint8)
        document = {
            "created_at": "2026-08-30T12:00:00Z",
            "image_width": width,
            "image_height": height,
            "camera_matrix": {
                "data": [2100.0, 0.0, width / 2.0, 0.0, 2100.0, height / 2.0, 0.0, 0.0, 1.0],
            },
            "distortion_coefficients": {"data": [-0.18, 0.04, 0.0, 0.0, 0.0]},
        }

        validation = generate_intrinsic_comparison(
            raw, None, document,
            comparison_calibration_id="intrinsics-4k.yaml", jpeg_quality=85
        )

        self.assertEqual(validation.report["source_image_size"], [width, height])
        self.assertEqual(validation.report["analysis_image_size"], [width, height])
        self.assertEqual(set(validation.images), {
            "overlay_checker", "overlay_redcyan", "overlay_corner_zoom", "overlay_diff",
            "displacement", "compare", "reference", "comparison",
        })
        decoded_raw = cv2.imdecode(
            np.frombuffer(validation.images["reference"], dtype=np.uint8), cv2.IMREAD_COLOR
        )
        decoded_undistorted = cv2.imdecode(
            np.frombuffer(validation.images["comparison"], dtype=np.uint8), cv2.IMREAD_COLOR
        )
        decoded_compare = cv2.imdecode(
            np.frombuffer(validation.images["compare"], dtype=np.uint8), cv2.IMREAD_COLOR
        )
        self.assertEqual(decoded_raw.shape[:2], (height, width))
        self.assertEqual(decoded_undistorted.shape[:2], (height, width))
        self.assertEqual(decoded_compare.shape[:2], (height, width * 2 + 8))


if __name__ == "__main__":
    unittest.main()
