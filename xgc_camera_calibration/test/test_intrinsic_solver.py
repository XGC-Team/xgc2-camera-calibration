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


def _project_nonzero_distortion_views():
    board = (9, 6)
    square = 0.04
    camera_matrix = np.asarray(
        ((672.0, 0.0, 640.0), (0.0, 680.0, 360.0), (0.0, 0.0, 1.0)),
        dtype=np.float64,
    )
    distortion = np.asarray((-0.25, 0.07, 0.001, -0.001, 0.01), dtype=np.float64)
    poses = (
        (0.15, 0.10, 0.00, -0.16, -0.10, 1.00),
        (-0.18, 0.15, 0.05, -0.05, -0.08, 1.10),
        (0.20, -0.20, -0.05, -0.25, -0.02, 1.15),
        (-0.25, -0.18, 0.08, 0.05, -0.12, 1.20),
        (0.35, 0.05, 0.12, -0.20, 0.02, 0.90),
        (-0.32, 0.08, -0.10, -0.08, -0.18, 1.00),
        (0.08, 0.32, 0.15, -0.25, -0.10, 1.25),
        (0.05, -0.35, -0.12, 0.05, -0.05, 1.30),
        (0.22, 0.24, 0.20, -0.10, -0.02, 0.85),
        (-0.28, 0.25, -0.18, -0.20, -0.15, 1.10),
        (0.30, -0.22, 0.10, -0.05, -0.10, 1.05),
        (-0.20, -0.30, -0.15, -0.22, 0.02, 1.20),
        (0.10, 0.05, 0.25, -0.12, -0.08, 0.75),
    )
    obj = solver.board_object_points(board, square)
    image_points = []
    for pose in poses:
        projected, _ = cv2.projectPoints(
            obj,
            np.asarray(pose[:3], dtype=np.float64),
            np.asarray(pose[3:], dtype=np.float64),
            camera_matrix,
            distortion,
        )
        image_points.append(projected.astype(np.float32))
    return board, square, camera_matrix, distortion, image_points


class IntrinsicSolverTest(unittest.TestCase):
    def test_aprilgrid_feature_model_breaks_the_retired_corner_datum_epoch(self):
        self.assertEqual(
            solver.APRILGRID_FEATURE_MODEL,
            "aprilgrid_kalibr_tag_corners_v2",
        )
        self.assertEqual(
            solver.APRILGRID_CORNER_DATUM,
            "kalibr_id0_lower_left_opencv_rotated_180_v1",
        )

    def test_detector_uncertainty_is_exposed_from_each_detection_contract(self):
        self.assertEqual(
            solver.observation_uncertainty_px("aprilgrid"),
            solver._APRILGRID_EDGE_MAX_LINE_P90_PX,
        )
        self.assertEqual(
            solver.observation_uncertainty_px("checkerboard"),
            solver._SUBPIX_CRITERIA[2],
        )
        with self.assertRaises(ValueError):
            solver.observation_uncertainty_px("unknown")

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
        diagnostics = result.diagnostics
        self.assertIsNotNone(diagnostics)
        self.assertTrue(diagnostics.finite)
        self.assertEqual(len(diagnostics.per_view_errors_px), len(image_points))
        self.assertEqual(len(diagnostics.rotation_vectors), len(image_points))
        self.assertEqual(len(diagnostics.translation_vectors), len(image_points))
        self.assertEqual(
            len(diagnostics.intrinsic_standard_deviations),
            len(diagnostics.parameter_names),
        )
        self.assertEqual(
            diagnostics.projected_intrinsic_rank,
            diagnostics.projected_intrinsic_parameter_count,
        )
        self.assertFalse(diagnostics.projected_intrinsic_rank_deficient)
        self.assertEqual(len(diagnostics.stability.folds), len(image_points))
        self.assertEqual(diagnostics.stability.failed_omitted_view_indices, ())
        self.assertLess(diagnostics.stability.held_out_rms_max_px, 0.001)
        self.assertLess(
            diagnostics.stability.undistorted_ray_max_equivalent_px, 0.1
        )
        self.assertTrue(all(
            len(fold.held_out_point_errors_px) == len(image_points[fold.omitted_view_index])
            for fold in diagnostics.stability.folds
        ))

    def test_extended_optimizer_converges_on_nonzero_plumb_bob_truth(self):
        board, square, truth_k, truth_d, image_points = (
            _project_nonzero_distortion_views()
        )
        result = solver.calibrate_intrinsic(
            image_points, board, square, (1280, 720)
        )

        self.assertGreaterEqual(solver._CALIBRATION_CRITERIA[1], 50)
        np.testing.assert_allclose(result.camera_matrix, truth_k, atol=0.01)
        np.testing.assert_allclose(result.distortion, truth_d, atol=0.001)
        self.assertLess(result.rms_reprojection_error_px, 0.001)
        self.assertEqual(result.diagnostics.projected_intrinsic_rank, 9)
        self.assertLess(
            result.diagnostics.stability.held_out_rms_max_px, 0.001
        )
        self.assertLess(
            result.diagnostics.stability.undistorted_ray_max_equivalent_px, 1.0
        )

    def test_robust_view_selection_removes_one_bad_view_and_recovers_truth(self):
        board, square, truth_k, truth_d, image_points = (
            _project_nonzero_distortion_views()
        )
        corrupted = [points.copy() for points in image_points]
        corrupted[4][:4] += np.asarray(
            ((8.0, -8.0), (-8.0, 8.0), (8.0, 8.0), (-8.0, -8.0)),
            dtype=np.float32,
        ).reshape(4, 1, 2)
        result = solver.calibrate_intrinsic(
            corrupted,
            board,
            square,
            (1280, 720),
            observation_uncertainty=solver.observation_uncertainty_px("aprilgrid"),
        )
        diagnostics = result.diagnostics

        self.assertEqual(diagnostics.pool_sample_count, 13)
        self.assertEqual(result.sample_count, 12)
        self.assertEqual(diagnostics.selected_view_indices, tuple(
            index for index in range(13) if index != 4
        ))
        self.assertEqual(len(diagnostics.rejected_views), 1)
        rejected = diagnostics.rejected_views[0]
        self.assertEqual(rejected.original_view_index, 4)
        self.assertEqual(
            rejected.reason, "per_view_rms_above_robust_3sigma_envelope"
        )
        self.assertGreater(
            rejected.initial_rms_reprojection_error_px,
            rejected.rejection_envelope_px,
        )
        self.assertNotIn(4, {
            fold.omitted_view_index for fold in diagnostics.stability.folds
        })
        np.testing.assert_allclose(result.camera_matrix, truth_k, atol=0.01)
        np.testing.assert_allclose(result.distortion, truth_d, atol=0.001)

    def test_consistent_high_noise_is_not_rejected_by_an_absolute_rms_gate(self):
        board, square, _truth_k, _truth_d, image_points = (
            _project_nonzero_distortion_views()
        )
        random = np.random.RandomState(123)
        noisy = [
            points + random.normal(0.0, 3.0, points.shape).astype(np.float32)
            for points in image_points
        ]
        result = solver.calibrate_intrinsic(
            noisy,
            board,
            square,
            (1280, 720),
            observation_uncertainty=solver.observation_uncertainty_px("aprilgrid"),
        )

        self.assertGreater(result.rms_reprojection_error_px, 3.0)
        self.assertEqual(result.sample_count, len(noisy))
        self.assertEqual(result.diagnostics.rejected_views, ())
        self.assertEqual(
            result.diagnostics.selected_view_indices, tuple(range(len(noisy)))
        )

    def test_three_weak_views_report_ill_conditioning_without_a_lens_gate(self):
        board = (7, 5)
        square = 0.10
        object_points = solver.board_object_points(board, square)
        truth_k = np.asarray(
            ((700.0, 0.0, 640.0), (0.0, 700.0, 360.0), (0.0, 0.0, 1.0)),
            dtype=np.float64,
        )
        # These views translate a nearly fronto-parallel plane. The tiny
        # rotations keep OpenCV's output finite while leaving focal length and
        # distance almost interchangeable -- the audit's weak three-view case.
        poses = (
            (0.0, 0.0, 0.0, -0.30, -0.20, 3.0),
            (1.0e-4, 0.0, 0.0, 0.10, -0.10, 3.1),
            (0.0, 1.0e-4, 0.0, -0.10, 0.15, 2.9),
        )
        image_points = []
        for pose in poses:
            projected, _ = cv2.projectPoints(
                object_points,
                np.asarray(pose[:3], dtype=np.float64),
                np.asarray(pose[3:], dtype=np.float64),
                truth_k,
                np.zeros(5),
            )
            image_points.append(projected.astype(np.float32))

        result = solver.calibrate_intrinsic(
            image_points, board, square, (1280, 720)
        )
        diagnostics = result.diagnostics

        # A tiny training RMS does not erase the continuous evidence: the
        # pose-eliminated intrinsic system is badly conditioned and leave-one-
        # out parameters move by orders of magnitude. No fixed fx/D/RMS range
        # rejects or edits this result; callers receive the evidence verbatim.
        self.assertLess(result.rms_reprojection_error_px, 0.1)
        self.assertGreater(
            diagnostics.projected_intrinsic_condition_number, 1.0e4
        )
        self.assertEqual(len(diagnostics.stability.folds), 3)
        self.assertGreater(
            max(diagnostics.stability.maximum_relative_delta), 10.0
        )
        self.assertGreater(
            diagnostics.stability.undistorted_ray_max_equivalent_px, 100.0
        )
        self.assertTrue(all(
            fold.held_out_rms_reprojection_error_px is not None
            and len(fold.held_out_point_errors_px) == len(image_points[fold.omitted_view_index])
            for fold in diagnostics.stability.folds
        ))

    def test_calibration_leaves_camera_matrix_and_distortion_free(self):
        image_points, _params = _project_views()
        returned_matrix = np.asarray(
            ((901.0, 0.0, 639.0), (0.0, 899.0, 361.0), (0.0, 0.0, 1.0)),
            dtype=np.float64,
        )
        returned_distortion = np.asarray(
            (-0.12, 0.03, 0.001, -0.002, 0.004), dtype=np.float64
        )
        def extended_result(objects, images, _size, _matrix, _distortion, **_kwargs):
            sample_count = len(images)
            return (
                0.25,
                returned_matrix,
                returned_distortion,
                [np.zeros((3, 1), dtype=np.float64) for _ in images],
                [np.asarray(((0.0,), (0.0,), (4.0,))) for _ in images],
                np.ones((18, 1), dtype=np.float64),
                np.ones((sample_count * 6, 1), dtype=np.float64),
                np.full((sample_count, 1), 0.25, dtype=np.float64),
            )

        with mock.patch.object(
            cv2,
            "calibrateCameraExtended",
            side_effect=extended_result,
        ) as calibrate:
            result = solver.calibrate_intrinsic(
                image_points[:3], BOARD, SQUARE, (WIDTH, HEIGHT)
            )

        first_call = calibrate.call_args_list[0]
        self.assertEqual(len(first_call.args), 5)
        self.assertIsNone(first_call.args[3])
        self.assertIsNone(first_call.args[4])
        self.assertEqual(first_call.kwargs["flags"], 0)
        self.assertEqual(first_call.kwargs["criteria"], solver._CALIBRATION_CRITERIA)
        np.testing.assert_allclose(result.camera_matrix, returned_matrix)
        np.testing.assert_allclose(result.distortion, returned_distortion)

    def test_coverage_reports_geometric_extent(self):
        bars, goodenough = solver.coverage([])
        self.assertEqual([b["label"] for b in bars], ["X", "Y", "Size", "Skew"])
        self.assertTrue(all(b["progress"] == 0.0 for b in bars))
        self.assertFalse(goodenough)
        _, params = _project_views()
        bars, _ = solver.coverage(params)
        self.assertEqual(len(bars), 4)

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

    def test_strict_aprilgrid_observation_accepts_only_the_standard_datum(self):
        tag_ids = (0, 1, 2, 6, 7, 8)
        objects, images = _project_standard_aprilgrid_tags(tag_ids)
        self.assertTrue(
            solver._strict_aprilgrid_observation(
                images, objects, tag_ids, (6, 6), 0.088, 0.0264, 0, 0.75
            )
        )

        # candidate-upper-062 equivalent: one decoded tag arrives with a
        # cyclic corner datum. The retired D4 code used to rotate this back;
        # the single standard datum now rejects the observation unchanged.
        rotated = list(images)
        rotated[2] = np.roll(rotated[2], 1, axis=0)
        self.assertFalse(
            solver._strict_aprilgrid_observation(
                rotated, objects, tag_ids, (6, 6), 0.088, 0.0264, 0, 0.75
            )
        )

    def test_strict_aprilgrid_observation_rejects_id_and_lattice_ambiguity(self):
        one_row_ids = (0, 1, 2, 3, 4, 5)
        row_objects, row_images = _project_standard_aprilgrid_tags(one_row_ids)
        self.assertFalse(
            solver._strict_aprilgrid_observation(
                row_images,
                row_objects,
                one_row_ids,
                (6, 6),
                0.088,
                0.0264,
                0,
                0.75,
            )
        )

        tag_ids = (0, 1, 2, 6, 7, 8)
        objects, images = _project_standard_aprilgrid_tags(tag_ids)
        self.assertFalse(
            solver._strict_aprilgrid_observation(
                images,
                objects,
                (0, 1, 2, 6, 7, 7),
                (6, 6),
                0.088,
                0.0264,
                0,
                0.75,
            )
        )
        self.assertFalse(
            solver._strict_aprilgrid_observation(
                images,
                objects,
                (0, 1, 2, 6, 7, 36),
                (6, 6),
                0.088,
                0.0264,
                0,
                0.75,
            )
        )

        # Simulate a confidently decoded but wrong in-range id: the measured
        # last tag is physically id 8 while its object correspondence says 9.
        wrong_ids = (0, 1, 2, 6, 7, 9)
        wrong_objects = list(objects[:-1]) + [
            solver.aprilgrid_tag_object_points((6, 6), 0.088, 0.0264, 0, 9)
        ]
        self.assertFalse(
            solver._strict_aprilgrid_observation(
                images,
                wrong_objects,
                wrong_ids,
                (6, 6),
                0.088,
                0.0264,
                0,
                0.75,
            )
        )

    def test_detect_aprilgrid_rejects_duplicate_or_out_of_range_decoded_ids(self):
        objects, images = _project_standard_aprilgrid_tags((0, 1, 2, 6, 7, 8))
        del objects  # detector reconstructs the authoritative object lattice
        opencv_order = np.asarray(
            solver._APRILGRID_OPENCV_TO_KALIBR_CORNER_ORDER
        )
        corners = [image[opencv_order].reshape(1, 4, 2) for image in images]
        gray = np.full((800, 1200), 127, dtype=np.uint8)
        for invalid_ids in ((0, 1, 2, 6, 7, 7), (0, 1, 2, 6, 7, 36)):
            with self.subTest(invalid_ids=invalid_ids), mock.patch.object(
                solver,
                "_detect_aruco_markers",
                return_value=(
                    corners,
                    np.asarray(invalid_ids, dtype=np.int32).reshape(-1, 1),
                    [],
                ),
            ), mock.patch.object(
                solver, "_refine_aprilgrid_quad_edges"
            ) as refinement:
                detection = solver.detect_aprilgrid(
                    gray,
                    (6, 6),
                    square=0.088,
                    tag_spacing=0.0264,
                    min_tags=6,
                    maximum_width=1200,
                )
            self.assertIsNone(detection)
            refinement.assert_not_called()

    def test_aprilgrid_coverage_uses_the_same_six_tag_refinement_mask_as_solve(self):
        tag_ids = (0, 1, 2, 3, 6, 7, 8)
        objects, images = _project_standard_aprilgrid_tags(tag_ids)
        # Tag 3 is a raw annotation candidate with a large localization error;
        # source refinement rejects it. The remaining two-dimensional six-tag
        # observation is the only input to both coverage and calibration.
        images = list(images)
        images[3] = images[3] + np.asarray((80.0, 0.0), dtype=np.float32)
        opencv_order = np.asarray(
            solver._APRILGRID_OPENCV_TO_KALIBR_CORNER_ORDER
        )
        corners = [image[opencv_order].reshape(1, 4, 2) for image in images]
        ids = np.asarray(tag_ids, dtype=np.int32).reshape(-1, 1)
        refinement_index = {"value": 0}

        def refine(_gray, quad):
            index = refinement_index["value"]
            refinement_index["value"] += 1
            if index == 3:
                raise ValueError("controlled source refinement rejection")
            return np.asarray(quad, dtype=np.float32) + np.asarray(
                (0.25, 0.25), dtype=np.float32
            )

        gray = np.full((800, 1200), 127, dtype=np.uint8)
        with mock.patch.object(
            solver, "_detect_aruco_markers", return_value=(corners, ids, [])
        ), mock.patch.object(
            solver, "_refine_aprilgrid_quad_edges", side_effect=refine
        ), mock.patch.object(
            cv2, "cornerSubPix", side_effect=lambda _gray, points, *_args: points
        ):
            detection = solver.detect_aprilgrid(
                gray,
                (6, 6),
                square=0.088,
                tag_spacing=0.0264,
                min_tags=6,
                maximum_width=1200,
            )

        self.assertIsNotNone(detection)
        self.assertEqual(len(detection.image_points), 7 * 4)
        self.assertEqual(len(detection.calibration_image_points), 6 * 4)
        expected_hull = solver._coverage_params_from_points(
            detection.calibration_image_points,
            1200,
            800,
        )
        expected = solver._aprilgrid_coverage(
            detection.calibration_image_points,
            detection.calibration_object_points,
            1200,
            800,
        )
        raw = solver._aprilgrid_coverage(
            detection.image_points,
            detection.object_points,
            1200,
            800,
        )
        np.testing.assert_allclose(detection.coverage, expected, atol=1e-9)
        np.testing.assert_allclose(detection.coverage[:3], expected_hull[:3], atol=1e-9)
        self.assertGreater(np.linalg.norm(np.asarray(raw) - np.asarray(expected)), 0.01)

    def test_aprilgrid_obliqueness_ignores_roll_and_responds_to_plane_tilt(self):
        tag_ids = (0, 1, 2, 6, 7, 8, 12, 13, 14)
        objects = np.concatenate([
            solver.aprilgrid_tag_object_points(
                (6, 6), 0.088, 0.0264, 0, marker_id
            )
            for marker_id in tag_ids
        ])
        camera_matrix = np.asarray(
            ((700.0, 0.0, 640.0), (0.0, 700.0, 360.0), (0.0, 0.0, 1.0)),
            dtype=np.float64,
        )

        rolled, _ = cv2.projectPoints(
            objects,
            np.asarray((0.0, 0.0, 0.7), dtype=np.float64),
            np.asarray((-0.2, -0.2, 1.2), dtype=np.float64),
            camera_matrix,
            np.zeros(5),
        )
        tilted, _ = cv2.projectPoints(
            objects,
            np.asarray((0.5, 0.3, 0.7), dtype=np.float64),
            np.asarray((-0.2, -0.2, 1.2), dtype=np.float64),
            camera_matrix,
            np.zeros(5),
        )
        roll_coverage = solver._aprilgrid_coverage(
            rolled, objects, 1280, 720
        )
        tilt_coverage = solver._aprilgrid_coverage(
            tilted, objects, 1280, 720
        )

        self.assertLess(roll_coverage[3], 1.0e-5)
        self.assertGreater(tilt_coverage[3], 0.2)
        self.assertGreater(tilt_coverage[3], roll_coverage[3])

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

    def test_source_edge_refine_falls_back_to_cornersubpix(self):
        if not hasattr(cv2, "aruco") or not hasattr(cv2.aruco, "DICT_APRILTAG_36h11"):
            self.skipTest("OpenCV AprilTag 36h11 dictionary is unavailable")
        gray = _render_aprilgrid((6, 6), tag_pixels=80, gap_pixels=24, border=60)
        corners = []
        objects = []
        dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
        detected, ids, _rejected = solver._detect_aruco_markers(gray, dictionary)
        self.assertIsNotNone(ids)
        for marker_corners, marker_id in zip(detected, ids.reshape(-1)):
            obj = solver.aprilgrid_tag_object_points(
                (6, 6), 0.088, 0.0264, 0, int(marker_id)
            )
            if obj is None:
                continue
            image = solver._opencv_corners_to_aprilgrid_datum(
                np.asarray(marker_corners, dtype=np.float32).reshape(4, 2),
                solver.APRILGRID_CORNER_DATUM,
            )
            corners.append(image)
            objects.append(obj)
        with mock.patch.object(
            solver, "_refine_aprilgrid_quad_edges", side_effect=ValueError("too short")
        ):
            pixels, localized, _coverage = solver.localize_aprilgrid_source_corners(
                gray,
                np.asarray(corners, dtype=np.float32),
                np.asarray(objects, dtype=np.float32),
                (6, 6),
                0.088,
                0.0264,
                0,
                6,
            )
        self.assertGreaterEqual(len(pixels), 24)
        self.assertEqual(len(pixels), len(localized))

    def test_optional_refinement_keeps_decoded_search_when_edges_are_short(self):
        objects, images = _project_standard_aprilgrid_tags((0, 1, 2, 6, 7, 8))
        del objects
        opencv_order = np.asarray(solver._APRILGRID_OPENCV_TO_KALIBR_CORNER_ORDER)
        corners = [image[opencv_order].reshape(1, 4, 2) for image in images]
        ids = np.asarray((0, 1, 2, 6, 7, 8), dtype=np.int32).reshape(-1, 1)
        gray = np.full((800, 1200), 127, dtype=np.uint8)
        with mock.patch.object(
            solver, "_detect_aruco_markers", return_value=(corners, ids, [])
        ), mock.patch.object(
            solver,
            "refine_aprilgrid_calibration_corners",
            side_effect=CalibrationError("AprilGrid edge is too short"),
        ):
            required = solver.detect_aprilgrid(
                gray,
                (6, 6),
                square=0.088,
                tag_spacing=0.0264,
                min_tags=6,
                maximum_width=1200,
                require_refinement=True,
            )
            optional = solver.detect_aprilgrid(
                gray,
                (6, 6),
                square=0.088,
                tag_spacing=0.0264,
                min_tags=6,
                maximum_width=1200,
                require_refinement=False,
            )
        self.assertIsNone(required)
        self.assertIsNotNone(optional)
        self.assertEqual(len(optional.image_points), 24)
        self.assertIsNone(optional.calibration_image_points)
        self.assertIsNone(optional.calibration_object_points)

    def test_one_rejected_quad_justifies_a_higher_resolution_retry(self):
        gray = np.full((540, 960), 127, np.uint8)
        rejected = [
            np.asarray(
                (((10.0, 10.0), (20.0, 10.0), (20.0, 20.0), (10.0, 20.0)),),
                dtype=np.float32,
            )
        ]
        with mock.patch.object(
            solver, "_detect_aruco_markers", return_value=([], None, rejected)
        ):
            self.assertTrue(solver.aprilgrid_has_candidate_evidence(gray))
            self.assertFalse(
                solver.aprilgrid_has_candidate_evidence(gray, minimum_quads=6)
            )

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

    def test_rejected_quads_only_trigger_source_retry_and_never_enter_solve(self):
        if not hasattr(cv2, "aruco") or not hasattr(cv2.aruco, "DICT_APRILTAG_36h11"):
            self.skipTest("OpenCV AprilTag 36h11 dictionary is unavailable")
        gray = np.full((540, 960), 127, np.uint8)
        rejected = [
            np.asarray(
                (((10.0 + index, 10.0), (20.0 + index, 10.0),
                  (20.0 + index, 20.0), (10.0 + index, 20.0)),),
                dtype=np.float32,
            )
            for index in range(6)
        ]
        with mock.patch.object(
            solver, "_detect_aruco_markers", return_value=([], None, rejected)
        ) as detect_markers:
            self.assertTrue(solver.aprilgrid_has_candidate_evidence(gray))
            detection = solver.detect_aprilgrid(
                gray,
                (6, 6),
                square=0.088,
                tag_spacing=0.0264,
                min_tags=6,
                maximum_width=960,
            )
        self.assertIsNone(detection)
        self.assertEqual(detect_markers.call_count, 2)

    def test_direct_decode_below_minimum_never_promotes_random_quads(self):
        decoded_ids = (0, 1, 6, 7, 8)
        _objects, decoded_images = _project_standard_aprilgrid_tags(decoded_ids)
        opencv_order = np.asarray(
            solver._APRILGRID_OPENCV_TO_KALIBR_CORNER_ORDER
        )
        corners = [image[opencv_order].reshape(1, 4, 2) for image in decoded_images]
        ids = np.asarray(decoded_ids, dtype=np.int32).reshape(-1, 1)
        rejected = [np.random.default_rng(7).random((1, 4, 2)).astype(np.float32)] * 20
        gray = np.full((800, 1200), 127, dtype=np.uint8)
        with mock.patch.object(
            solver, "_detect_aruco_markers", return_value=(corners, ids, rejected)
        ), mock.patch.object(
            solver, "refine_aprilgrid_calibration_corners"
        ) as refinement:
            detection = solver.detect_aprilgrid(
                gray,
                (6, 6),
                square=0.088,
                tag_spacing=0.0264,
                min_tags=6,
                maximum_width=1200,
            )
        self.assertIsNone(detection)
        refinement.assert_not_called()

    def test_retired_top_left_raw_marker_datum_is_rejected(self):
        if not hasattr(cv2, "aruco") or not hasattr(cv2.aruco, "DICT_APRILTAG_36h11"):
            self.skipTest("OpenCV AprilTag 36h11 dictionary is unavailable")
        gray = _render_aprilgrid(
            (6, 6),
            tag_pixels=80,
            gap_pixels=24,
            border=60,
            kalibr_datum=False,
        )
        detection = solver.detect_aprilgrid(
            gray,
            (6, 6),
            square=0.088,
            tag_spacing=0.0264,
            min_tags=6,
            maximum_width=960,
        )
        self.assertIsNone(detection)

    def test_canonical_kalibr_datum_accepts_all_global_image_quarter_turns(self):
        if not hasattr(cv2, "aruco") or not hasattr(
            cv2.aruco, "DICT_APRILTAG_36h11"
        ):
            self.skipTest("OpenCV AprilTag 36h11 dictionary is unavailable")
        canonical = _render_aprilgrid(
            (6, 6), tag_pixels=80, gap_pixels=24, border=60
        )
        for quarter_turns in range(4):
            with self.subTest(quarter_turns=quarter_turns):
                gray = np.rot90(canonical, quarter_turns).copy()
                detection = solver.detect_aprilgrid(
                    gray,
                    (6, 6),
                    square=0.088,
                    tag_spacing=0.0264,
                    min_tags=6,
                    maximum_width=960,
                )
                self.assertIsNotNone(detection)
                self.assertEqual(len(detection.image_points), 36 * 4)
                self.assertEqual(len(detection.calibration_image_points), 36 * 4)


def _project_standard_aprilgrid_tags(tag_ids):
    homography = np.asarray(
        ((420.0, 32.0, 180.0), (18.0, 390.0, 120.0), (0.025, 0.018, 1.0)),
        dtype=np.float64,
    )
    objects = [
        solver.aprilgrid_tag_object_points((6, 6), 0.088, 0.0264, 0, tag_id)
        for tag_id in tag_ids
    ]
    images = [
        cv2.perspectiveTransform(
            np.asarray(points[:, :2], dtype=np.float32).reshape(-1, 1, 2),
            homography,
        ).reshape(4, 2)
        for points in objects
    ]
    return objects, images


def _render_aprilgrid(
    board_size,
    tag_pixels=80,
    gap_pixels=24,
    border=40,
    kalibr_datum=True,
):
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
    cols, rows = board_size
    pitch = tag_pixels + gap_pixels
    width = border * 2 + cols * tag_pixels + (cols - 1) * gap_pixels
    height = border * 2 + rows * tag_pixels + (rows - 1) * gap_pixels
    image = np.full((height, width), 255, np.uint8)
    for row in range(rows):
        for col in range(cols):
            tag_row = rows - 1 - row if kalibr_datum else row
            tag_id = tag_row * cols + col
            tag = cv2.aruco.generateImageMarker(
                dictionary, tag_id, tag_pixels, None, 2
            )
            if kalibr_datum:
                tag = np.rot90(tag, 2)
            y0 = border + row * pitch
            x0 = border + col * pitch
            image[y0:y0 + tag_pixels, x0:x0 + tag_pixels] = tag
    return image


if __name__ == "__main__":
    unittest.main()
