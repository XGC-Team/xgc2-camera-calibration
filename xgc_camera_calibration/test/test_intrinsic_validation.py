import unittest

import cv2
import numpy as np

from xgc_camera_calibration.intrinsic_validation import generate_intrinsic_validation


class IntrinsicValidationTest(unittest.TestCase):
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

        validation = generate_intrinsic_validation(
            raw, document, calibration_id="intrinsics-4k.yaml", jpeg_quality=85
        )

        self.assertEqual(validation.report["source_image_size"], [width, height])
        self.assertEqual(validation.report["analysis_image_size"], [width, height])
        self.assertEqual(set(validation.images), {
            "overlay_checker", "overlay_redcyan", "overlay_corner_zoom", "overlay_diff",
            "displacement", "compare", "raw", "undistorted",
        })
        decoded_raw = cv2.imdecode(
            np.frombuffer(validation.images["raw"], dtype=np.uint8), cv2.IMREAD_COLOR
        )
        decoded_undistorted = cv2.imdecode(
            np.frombuffer(validation.images["undistorted"], dtype=np.uint8), cv2.IMREAD_COLOR
        )
        decoded_compare = cv2.imdecode(
            np.frombuffer(validation.images["compare"], dtype=np.uint8), cv2.IMREAD_COLOR
        )
        self.assertEqual(decoded_raw.shape[:2], (height, width))
        self.assertEqual(decoded_undistorted.shape[:2], (height, width))
        self.assertEqual(decoded_compare.shape[:2], (height, width * 2 + 8))


if __name__ == "__main__":
    unittest.main()
