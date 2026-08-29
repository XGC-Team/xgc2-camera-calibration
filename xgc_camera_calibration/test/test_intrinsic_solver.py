#!/usr/bin/env python3

import tempfile
import unittest
from pathlib import Path
from unittest import mock

import cv2
import numpy as np

from xgc_camera_calibration import intrinsic_solver as solver
from xgc_camera_calibration.solver import CalibrationError


BOARD = (7, 5)
SQUARE = 0.20
WIDTH, HEIGHT = 3840, 2160
TRUTH_K = np.array([[2288.17, 0.0, 1920.5], [0.0, 2288.17, 1080.5], [0.0, 0.0, 1.0]])


def _project_views():
    obj = solver.board_object_points(BOARD, SQUARE)
    poses = [
        (0, 0, 0, 0.0, 0.0, 4.0),
        (0.3, 0, 0, -0.5, 0.0, 4.0),
        (-0.3, 0, 0, 0.5, 0.0, 4.0),
        (0, 0.25, 0, 0.0, -0.4, 4.0),
        (0, -0.3, 0, 0.0, 0.4, 4.0),
        (0.2, 0.2, 0, -0.3, -0.3, 3.5),
        (0, 0, 0, 0.0, 0.0, 2.2),
        (0.4, -0.2, 0.1, 0.2, 0.2, 3.0),
    ]
    image_points, params = [], []
    for rx, ry, rz, tx, ty, tz in poses:
        projected, _ = cv2.projectPoints(
            obj,
            np.array([rx, ry, rz], dtype=np.float64),
            np.array([tx, ty, tz], dtype=np.float64),
            TRUTH_K,
            np.zeros(5),
        )
        image_points.append(projected.reshape(-1, 1, 2).astype(np.float32))
        params.append(solver._coverage_params(projected.reshape(-1, 1, 2), BOARD, WIDTH, HEIGHT))
    return image_points, params


class IntrinsicSolverTest(unittest.TestCase):
    def test_recovers_known_intrinsics(self):
        image_points, _ = _project_views()
        result = solver.calibrate_intrinsic(image_points, BOARD, SQUARE, (WIDTH, HEIGHT))
        self.assertAlmostEqual(result.camera_matrix[0, 0], 2288.17, delta=2.0)
        self.assertAlmostEqual(result.camera_matrix[0, 2], 1920.5, delta=2.0)
        self.assertAlmostEqual(result.camera_matrix[1, 2], 1080.5, delta=2.0)
        self.assertLess(result.rms_reprojection_error_px, 0.5)
        self.assertEqual(result.image_size, (WIDTH, HEIGHT))

    def test_coverage_and_new_sample(self):
        bars, goodenough = solver.coverage([])
        self.assertEqual([b["label"] for b in bars], ["X", "Y", "Size", "Skew"])
        self.assertTrue(all(b["progress"] == 0.0 for b in bars))
        self.assertFalse(goodenough)
        _, params = _project_views()
        bars, _ = solver.coverage(params)
        self.assertEqual(len(bars), 4)
        self.assertTrue(solver.is_new_sample((0.9, 0.1, 0.35, 0.2), params))
        self.assertFalse(solver.is_new_sample(params[0], params))

    def test_aprilgrid_xy_coverage_uses_visible_board_extent(self):
        left = np.array([[10, 100], [310, 100], [310, 500], [10, 500]], np.float32)
        right = left + np.array([680, 0], np.float32)
        left_params = solver._coverage_params_from_points(left, 1000, 600)
        right_params = solver._coverage_params_from_points(right, 1000, 600)
        self.assertLess(left_params[0], 0.02)
        self.assertGreater(right_params[0], 0.98)
        bars, _ = solver.coverage([left_params, right_params])
        self.assertEqual(bars[0]["progress"], 1.0)

    def test_next_view_guidance_uses_history_to_name_the_missing_direction(self):
        self.assertEqual(solver.next_view_guidance([]), {
            "complete": False, "dimension": None, "direction": "center", "progress": 0.0,
        })
        guidance = solver.next_view_guidance([
            (0.40, 0.10, 0.45, 0.55),
            (0.82, 0.78, 0.45, 0.55),
        ])
        self.assertEqual(guidance["dimension"], "X")
        self.assertEqual(guidance["direction"], "left")
        self.assertAlmostEqual(guidance["progress"], 0.6)

        complete = solver.next_view_guidance([
            (0.05, 0.05, 0.45, 0.55),
            (0.85, 0.85, 0.45, 0.55),
        ])
        self.assertTrue(complete["complete"])
        self.assertEqual(complete["direction"], "complete")

    def test_rejects_too_few_samples(self):
        with self.assertRaises(CalibrationError):
            solver.calibrate_intrinsic([np.zeros((35, 1, 2), np.float32)], BOARD, SQUARE, (WIDTH, HEIGHT))

    def test_save_load_roundtrip(self):
        image_points, _ = _project_views()
        result = solver.calibrate_intrinsic(image_points, BOARD, SQUARE, (WIDTH, HEIGHT))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "intrinsics.yaml"
            solver.save_intrinsic(path, result, board_size=BOARD, square=SQUARE, metadata={"web_calibrator": True})
            document = solver.load_intrinsic(path)
        self.assertEqual(document["schema"], "xgc2.camera.intrinsic.v1")
        self.assertEqual(document["image_width"], WIDTH)
        self.assertAlmostEqual(document["camera_matrix_array"][0, 0], result.camera_matrix[0, 0], places=6)
        self.assertEqual(document["metadata"]["web_calibrator"], True)

    def test_aprilgrid_tag_geometry_matches_printed_board(self):
        # Station plate: 6x6 tag36h11, 88 mm tags, 26.4 mm gaps, ids 0..35.
        obj = solver.aprilgrid_tag_object_points((6, 6), 0.088, 0.0264, 0, 7)
        self.assertIsNotNone(obj)
        # id 7 is column 1, row 1. Pitch = 0.1144 m.
        self.assertAlmostEqual(float(obj[0, 0]), 0.1144, places=6)
        self.assertAlmostEqual(float(obj[0, 1]), 0.1144, places=6)
        self.assertAlmostEqual(float(obj[1, 0]), 0.2024, places=6)
        self.assertIsNone(solver.aprilgrid_tag_object_points((6, 6), 0.088, 0.0264, 0, 36))

    def test_detects_synthetic_aprilgrid_and_recovers_intrinsics(self):
        if not hasattr(cv2, "aruco") or not hasattr(cv2.aruco, "DICT_APRILTAG_36h11"):
            self.skipTest("OpenCV AprilTag 36h11 dictionary is unavailable")
        gray = _render_aprilgrid((3, 3), tag_pixels=80, gap_pixels=24)
        detection = solver.detect_aprilgrid(
            gray, (3, 3), square=0.088, tag_spacing=0.0264, min_tags=6, maximum_width=960
        )
        self.assertIsNotNone(detection)
        self.assertGreaterEqual(len(detection.image_points), 24)
        self.assertIsNotNone(detection.calibration_image_points)
        self.assertIsNotNone(detection.calibration_object_points)
        self.assertEqual(
            len(detection.calibration_image_points), len(detection.image_points) // 4
        )
        np.testing.assert_allclose(
            detection.calibration_image_points.reshape(-1, 2),
            detection.image_points.reshape(-1, 4, 2).mean(axis=1),
            atol=1e-5,
        )
        image_points = []
        object_points = []
        obj = np.concatenate(
            [
                solver.aprilgrid_tag_object_points((3, 3), 0.088, 0.0264, 0, tag_id)
                for tag_id in range(9)
            ],
            axis=0,
        )
        poses = [
            (0, 0, 0, 0.0, 0.0, 3.5),
            (0.25, 0, 0, -0.4, 0.0, 3.2),
            (-0.2, 0.15, 0, 0.3, -0.2, 2.8),
        ]
        for rx, ry, rz, tx, ty, tz in poses:
            projected, _ = cv2.projectPoints(
                obj,
                np.array([rx, ry, rz], dtype=np.float64),
                np.array([tx, ty, tz], dtype=np.float64),
                TRUTH_K,
                np.zeros(5),
            )
            image_points.append(projected.reshape(-1, 1, 2).astype(np.float32))
            object_points.append(obj)
        result = solver.calibrate_intrinsic(
            image_points, (3, 3), 0.088, (WIDTH, HEIGHT), object_points=object_points
        )
        self.assertAlmostEqual(result.camera_matrix[0, 0], 2288.17, delta=2.0)
        self.assertLess(result.rms_reprojection_error_px, 0.5)

    def test_retries_small_physical_aprilgrid_at_bounded_resolution(self):
        if not hasattr(cv2, "aruco") or not hasattr(cv2.aruco, "DICT_APRILTAG_36h11"):
            self.skipTest("OpenCV AprilTag 36h11 dictionary is unavailable")
        gray = _render_aprilgrid((6, 6), tag_pixels=12, gap_pixels=4, border=20)
        detection = solver.detect_aprilgrid(
            gray, (6, 6), square=0.088, tag_spacing=0.0264, min_tags=6, maximum_width=960
        )
        self.assertIsNotNone(detection)
        self.assertGreaterEqual(len(detection.image_points), 24)
        points = detection.image_points.reshape(-1, 2)
        self.assertGreaterEqual(float(points.min()), 0.0)
        self.assertLessEqual(float(points[:, 0].max()), float(gray.shape[1]))
        self.assertLessEqual(float(points[:, 1].max()), float(gray.shape[0]))

    def test_contour_fallback_supports_opencv_42_aprilgrid_detection(self):
        if not hasattr(cv2, "aruco") or not hasattr(cv2.aruco, "DICT_APRILTAG_36h11"):
            self.skipTest("OpenCV AprilTag 36h11 dictionary is unavailable")
        gray = _render_aprilgrid((6, 6), tag_pixels=45, gap_pixels=14, border=40)
        # OpenCV 4.2 frequently returns rejected quads but no decoded ids for
        # the real station plate. Force that result so the compatibility path
        # remains covered even when tests run with a newer host OpenCV.
        with mock.patch.object(
            solver, "_detect_aruco_markers", return_value=([], None, [])
        ):
            detection = solver.detect_aprilgrid(
                gray,
                (6, 6),
                square=0.088,
                tag_spacing=0.0264,
                min_tags=6,
                maximum_width=960,
            )
        self.assertIsNotNone(detection)
        self.assertEqual(len(detection.image_points), 144)
        self.assertEqual(len(detection.object_points), 144)

    def test_physical_station_tag_rotation_keeps_board_geometry_aligned(self):
        if not hasattr(cv2, "aruco") or not hasattr(cv2.aruco, "DICT_APRILTAG_36h11"):
            self.skipTest("OpenCV AprilTag 36h11 dictionary is unavailable")
        gray = _render_aprilgrid(
            (6, 6),
            tag_pixels=45,
            gap_pixels=14,
            border=40,
            physical_station_layout=True,
        )
        with mock.patch.object(
            solver, "_detect_aruco_markers", return_value=([], None, [])
        ):
            detection = solver.detect_aprilgrid(
                gray,
                (6, 6),
                square=0.088,
                tag_spacing=0.0264,
                min_tags=6,
                maximum_width=960,
            )
        self.assertIsNotNone(detection)
        homography, _mask = cv2.findHomography(
            detection.object_points[:, :2], detection.image_points.reshape(-1, 2)
        )
        projected = cv2.perspectiveTransform(
            detection.object_points[:, :2].reshape(-1, 1, 2), homography
        ).reshape(-1, 2)
        error = np.linalg.norm(projected - detection.image_points.reshape(-1, 2), axis=1)
        self.assertLess(float(np.mean(error)), 0.5)
        self.assertLess(float(np.max(error)), 1.5)


def _render_aprilgrid(
    board_size,
    tag_pixels=80,
    gap_pixels=24,
    border=40,
    physical_station_layout=False,
):
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
    cols, rows = board_size
    pitch = tag_pixels + gap_pixels
    width = border * 2 + cols * tag_pixels + (cols - 1) * gap_pixels
    height = border * 2 + rows * tag_pixels + (rows - 1) * gap_pixels
    image = np.full((height, width), 255, np.uint8)
    for row in range(rows):
        for col in range(cols):
            tag_row = rows - 1 - row if physical_station_layout else row
            tag_id = tag_row * cols + col
            if hasattr(cv2.aruco, "generateImageMarker"):
                tag = cv2.aruco.generateImageMarker(dictionary, tag_id, tag_pixels)
            else:
                tag = cv2.aruco.drawMarker(dictionary, tag_id, tag_pixels)
            if physical_station_layout:
                tag = np.rot90(tag, 2)
            y0 = border + row * pitch
            x0 = border + col * pitch
            image[y0:y0 + tag_pixels, x0:x0 + tag_pixels] = tag
    return image


if __name__ == "__main__":
    unittest.main()
