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
    def test_refines_paired_aprilgrid_corners_on_source_gray_plane(self):
        gray = np.zeros((1080, 1920), np.uint8)
        corners = np.asarray(
            [[100, 100], [200, 100], [200, 200], [100, 200]],
            dtype=np.float32,
        ).reshape(-1, 1, 2)
        objects = np.asarray(
            [[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]],
            dtype=np.float32,
        )
        refined = corners.reshape(4, 2) + np.asarray([0.5, 0.5], dtype=np.float32)
        with mock.patch.object(
            solver, "_refine_aprilgrid_quad_edges", return_value=refined
        ) as edge_refinement:
            image_points, object_points = solver.refine_aprilgrid_calibration_corners(
                gray, corners, objects
            )
        edge_refinement.assert_called_once()
        self.assertEqual(edge_refinement.call_args.args[0].shape, (1080, 1920))
        np.testing.assert_allclose(image_points.reshape(-1, 2), refined.reshape(-1, 2))
        np.testing.assert_allclose(object_points, objects)

    def test_source_refinement_applies_one_complete_tag_mask_to_both_sides(self):
        gray = np.zeros((1080, 1920), np.uint8)
        corners = np.asarray(
            [
                [[100, 100], [200, 100], [200, 200], [100, 200]],
                [[300, 100], [400, 100], [400, 200], [300, 200]],
            ],
            dtype=np.float32,
        )
        objects = np.arange(24, dtype=np.float32).reshape(2, 4, 3)
        first_refined = corners[0] + np.asarray([0.5, 0.5], dtype=np.float32)
        with mock.patch.object(
            solver,
            "_refine_aprilgrid_quad_edges",
            side_effect=(first_refined, ValueError("second tag edge fit failed")),
        ):
            image_points, object_points = solver.refine_aprilgrid_calibration_corners(
                gray, corners, objects
            )
        self.assertEqual(len(image_points), 4)
        self.assertEqual(len(object_points), 4)
        np.testing.assert_allclose(object_points, objects[0])

    def test_recovers_known_intrinsics(self):
        image_points, _ = _project_views()
        result = solver.calibrate_intrinsic(image_points, BOARD, SQUARE, (WIDTH, HEIGHT))
        self.assertAlmostEqual(result.camera_matrix[0, 0], 2288.17, delta=2.0)
        self.assertAlmostEqual(result.camera_matrix[0, 2], 1920.5, delta=2.0)
        self.assertAlmostEqual(result.camera_matrix[1, 2], 1080.5, delta=2.0)
        self.assertLess(result.rms_reprojection_error_px, 0.5)
        self.assertEqual(result.image_size, (WIDTH, HEIGHT))

    def test_calibration_leaves_camera_matrix_and_distortion_free(self):
        image_points, _params = _project_views()
        returned_matrix = np.asarray(
            ((901.0, 0.0, 639.0), (0.0, 899.0, 361.0), (0.0, 0.0, 1.0)),
            dtype=np.float64,
        )
        returned_distortion = np.asarray(
            (-0.12, 0.03, 0.001, -0.002, 0.004), dtype=np.float64
        )
        with mock.patch.object(
            cv2,
            "calibrateCamera",
            return_value=(
                0.25,
                returned_matrix,
                returned_distortion,
                [],
                [],
            ),
        ) as calibrate:
            result = solver.calibrate_intrinsic(
                image_points[:3], BOARD, SQUARE, (WIDTH, HEIGHT)
            )

        self.assertEqual(len(calibrate.call_args.args), 5)
        self.assertIsNone(calibrate.call_args.args[3])
        self.assertIsNone(calibrate.call_args.args[4])
        self.assertEqual(calibrate.call_args.kwargs, {})
        np.testing.assert_allclose(result.camera_matrix, returned_matrix)
        np.testing.assert_allclose(result.distortion, returned_distortion)

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

    def test_repeated_samples_do_not_bypass_missing_geometry(self):
        repeated = [(0.5, 0.5, 0.1, 0.1)] * 40
        bars, goodenough = solver.coverage(repeated)
        self.assertFalse(goodenough)
        self.assertTrue(any(item["progress"] < 1.0 for item in bars))

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

    def test_next_view_guidance_uses_the_same_final_bars_shown_to_the_operator(self):
        samples = [
            (0.05, 0.05, 0.40, 0.02),
            (0.622, 0.75, 0.40, 0.08),
        ]
        guidance = solver.next_view_guidance(samples, coverage_bars=[
            {"label": "X", "progress": 0.817},
            {"label": "Y", "progress": 1.0},
            {"label": "Size", "progress": 1.0},
            {"label": "Skew", "progress": 1.0},
        ])
        self.assertEqual(guidance, {
            "complete": False,
            "dimension": "X",
            "direction": "right",
            "progress": 0.817,
        })

    def test_rejects_too_few_samples(self):
        with self.assertRaises(CalibrationError):
            solver.calibrate_intrinsic([np.zeros((35, 1, 2), np.float32)], BOARD, SQUARE, (WIDTH, HEIGHT))

    def test_save_load_roundtrip(self):
        image_points, _ = _project_views()
        result = solver.calibrate_intrinsic(image_points, BOARD, SQUARE, (WIDTH, HEIGHT))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "intrinsics.yaml"
            solver.save_intrinsic(
                path,result,camera_name="usb_cam",board_size=BOARD,square=SQUARE,
                metadata={"web_calibrator": True},
            )
            document = solver.load_intrinsic(path)
        self.assertEqual(document["schema"], "xgc2.camera.intrinsic.v1")
        self.assertEqual(document["image_width"], WIDTH)
        self.assertEqual(document["camera_name"], "usb_cam")
        self.assertEqual(
            document["rectification_matrix"]["data"],
            [1.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0,1.0],
        )
        self.assertEqual(document["projection_matrix"]["rows"], 3)
        self.assertEqual(document["projection_matrix"]["cols"], 4)
        matrix = document["camera_matrix"]["data"]
        self.assertEqual(document["projection_matrix"]["data"], [
            matrix[0],matrix[1],matrix[2],0.0,
            matrix[3],matrix[4],matrix[5],0.0,
            matrix[6],matrix[7],matrix[8],0.0,
        ])
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
            len(detection.calibration_image_points),
            len(detection.calibration_object_points),
        )
        self.assertGreaterEqual(
            len(detection.calibration_image_points), len(detection.image_points) - 4
        )
        self.assertGreater(len(detection.calibration_image_points), len(detection.image_points) // 4)
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

    def test_rejects_tags_too_small_for_full_corner_refinement(self):
        if not hasattr(cv2, "aruco") or not hasattr(cv2.aruco, "DICT_APRILTAG_36h11"):
            self.skipTest("OpenCV AprilTag 36h11 dictionary is unavailable")
        # Kalibr's two-cell border plus 6x6 payload needs at least 2 pixels per
        # logical cell; a 16 px marker cannot preserve ten binary cells.
        gray = _render_aprilgrid((6, 6), tag_pixels=16, gap_pixels=5, border=20)
        detection = solver.detect_aprilgrid(
            gray, (6, 6), square=0.088, tag_spacing=0.0264, min_tags=6, maximum_width=960
        )
        self.assertIsNone(detection)

    def test_edge_line_refinement_rejects_a_nonstraight_tag_boundary(self):
        gray = np.full((160, 240), 255, dtype=np.uint8)
        for x in range(gray.shape[1]):
            boundary = 80 + (2 if (x // 5) % 2 else -2)
            gray[boundary:, x] = 0
        with self.assertRaisesRegex(ValueError, "not straight enough"):
            solver._fit_aprilgrid_edge(
                gray,
                np.asarray((30.0, 80.0), dtype=np.float32),
                np.asarray((210.0, 80.0), dtype=np.float32),
            )

    def test_edge_line_refinement_rejects_far_convex_intersections(self):
        gray = np.zeros((240, 240), dtype=np.uint8)
        raw = np.asarray(
            ((50.0, 50.0), (150.0, 50.0), (150.0, 150.0), (50.0, 150.0)),
            dtype=np.float32,
        )
        shifted_lines = (
            (np.asarray((60.0, 60.0)), np.asarray((1.0, 0.0))),
            (np.asarray((160.0, 60.0)), np.asarray((0.0, 1.0))),
            (np.asarray((160.0, 160.0)), np.asarray((-1.0, 0.0))),
            (np.asarray((60.0, 160.0)), np.asarray((0.0, -1.0))),
        )
        with mock.patch.object(
            solver, "_fit_aprilgrid_edge", side_effect=shifted_lines
        ), self.assertRaisesRegex(ValueError, "search-band geometry"):
            solver._refine_aprilgrid_quad_edges(gray, raw)

    def test_edge_line_refinement_rejects_nearly_parallel_adjacent_lines(self):
        gray = np.zeros((240, 240), dtype=np.uint8)
        raw = np.asarray(
            ((50.0, 50.0), (150.0, 50.0), (150.0, 150.0), (50.0, 150.0)),
            dtype=np.float32,
        )
        nearly_horizontal = np.asarray((1.0, 0.01), dtype=np.float64)
        nearly_horizontal /= np.linalg.norm(nearly_horizontal)
        ill_conditioned_lines = (
            (np.asarray((50.0, 50.0)), np.asarray((1.0, 0.0))),
            (np.asarray((150.0, 50.0)), nearly_horizontal),
            (np.asarray((150.0, 150.0)), np.asarray((-1.0, 0.0))),
            (np.asarray((50.0, 150.0)), np.asarray((0.0, -1.0))),
        )
        with mock.patch.object(
            solver, "_fit_aprilgrid_edge", side_effect=ill_conditioned_lines
        ), self.assertRaisesRegex(ValueError, "too nearly parallel"):
            solver._refine_aprilgrid_quad_edges(gray, raw)

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

    def test_contour_fallback_stays_at_detection_width_and_restores_full_coordinates(self):
        if not hasattr(cv2, "aruco") or not hasattr(cv2.aruco, "DICT_APRILTAG_36h11"):
            self.skipTest("OpenCV AprilTag 36h11 dictionary is unavailable")
        source = _render_aprilgrid((6, 6), tag_pixels=45, gap_pixels=14, border=40)
        gray = cv2.resize(source, None, fx=3.0, fy=3.0, interpolation=cv2.INTER_NEAREST)
        fallback_widths = []
        fallback = solver._aprilgrid_contour_fallback

        def observed_fallback(image, *args, **kwargs):
            fallback_widths.append(image.shape[1])
            return fallback(image, *args, **kwargs)

        with mock.patch.object(
            solver, "_detect_aruco_markers", return_value=([], None, [])
        ) as detect_markers, mock.patch.object(
            solver, "_aprilgrid_contour_fallback", side_effect=observed_fallback
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
        self.assertEqual(fallback_widths, [960])
        self.assertEqual(detect_markers.call_count, 1)
        points = detection.image_points.reshape(-1, 2)
        self.assertGreater(float(points[:, 0].max()), 960.0)
        self.assertLessEqual(float(points[:, 0].max()), float(gray.shape[1]))
        self.assertLessEqual(float(points[:, 1].max()), float(gray.shape[0]))

    def test_board_absent_frame_does_not_run_large_recovery_pyramid(self):
        if not hasattr(cv2, "aruco") or not hasattr(cv2.aruco, "DICT_APRILTAG_36h11"):
            self.skipTest("OpenCV AprilTag 36h11 dictionary is unavailable")
        gray = np.full((540, 960), 127, np.uint8)
        with mock.patch.object(
            solver, "_detect_aruco_markers", return_value=([], None, [])
        ) as detect_markers, mock.patch.object(
            solver, "_aprilgrid_contour_fallback", return_value=None
        ) as fallback:
            detection = solver.detect_aprilgrid(
                gray,
                (6, 6),
                square=0.088,
                tag_spacing=0.0264,
                min_tags=6,
                maximum_width=960,
            )
        self.assertIsNone(detection)
        self.assertEqual(detect_markers.call_count, 1)
        fallback.assert_called_once()

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
                tag = cv2.aruco.generateImageMarker(dictionary, tag_id, tag_pixels, None, 2)
            else:
                tag = cv2.aruco.drawMarker(dictionary, tag_id, tag_pixels, borderBits=2)
            if physical_station_layout:
                tag = np.rot90(tag, 2)
            y0 = border + row * pitch
            x0 = border + col * pitch
            image[y0:y0 + tag_pixels, x0:x0 + tag_pixels] = tag
    return image


if __name__ == "__main__":
    unittest.main()
