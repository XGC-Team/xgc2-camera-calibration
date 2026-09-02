#!/usr/bin/env python3

import hashlib
import io
import json
import math
import tempfile
import threading
import unittest
import urllib.error
import urllib.request
import zipfile
from http import HTTPStatus
from pathlib import Path
from time import monotonic, sleep
from types import SimpleNamespace
from unittest.mock import patch

import cv2
import numpy as np

from xgc_camera_calibration import intrinsic_solver, intrinsic_validation
from xgc_camera_calibration.intrinsic_service import (
    APRILGRID_ADAPTIVE_EVIDENCE_QUADS,
    APRILGRID_SEARCH_HOLD_FRAMES,
    IntrinsicCalibrationService,
    intrinsic_algorithm_provenance,
    intrinsic_calibration_directory,
    recommended_views,
)
from xgc_camera_calibration.media_snapshot import MediaSnapshotClient
from xgc_camera_calibration.web_service import ApiError, CalibrationHttpServer


WEB_ROOT = Path(__file__).resolve().parents[1] / "web" / "intrinsic"
ENTRYPOINT = Path(__file__).resolve().parents[1] / "scripts" / "intrinsic_calibrator_web.py"
WEB_SERVICE_SOURCE = Path(__file__).resolve().parents[1] / "src" / "xgc_camera_calibration" / "web_service.py"


def render_board(cols_squares=8, rows_squares=6, square=40, border=40):
    """A clean synthetic chessboard (cols_squares x rows_squares squares)."""
    width = cols_squares * square + 2 * border
    height = rows_squares * square + 2 * border
    image = np.full((height, width), 255, np.uint8)
    for row in range(rows_squares):
        for col in range(cols_squares):
            if (row + col) % 2 == 0:
                y0, x0 = border + row * square, border + col * square
                image[y0:y0 + square, x0:x0 + square] = 0
    return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)


APRILGRID_TEST_SIZE = (4, 4)
APRILGRID_TEST_TAG_SIZE = 0.088
APRILGRID_TEST_TAG_GAP = 0.0264
APRILGRID_TEST_IMAGE_SIZE = (1280, 720)
APRILGRID_TEST_K = np.asarray(
    ((900.0, 0.0, 639.5), (0.0, 900.0, 359.5), (0.0, 0.0, 1.0)),
    dtype=np.float64,
)


def render_aprilgrid_view(center_pixel, depth_m, rotation_vector):
    """Render a real tag36h11 image for process_frame-level pose admission."""
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
    tag_pixels = 100
    gap_pixels = 30
    margin = 48
    pitch_pixels = tag_pixels + gap_pixels
    content_width = (
        APRILGRID_TEST_SIZE[0] * tag_pixels
        + (APRILGRID_TEST_SIZE[0] - 1) * gap_pixels
    )
    content_height = (
        APRILGRID_TEST_SIZE[1] * tag_pixels
        + (APRILGRID_TEST_SIZE[1] - 1) * gap_pixels
    )
    printed_width = content_width + 2 * gap_pixels
    printed_height = content_height + 2 * gap_pixels
    board = np.full(
        (printed_height + 2 * margin, printed_width + 2 * margin),
        255,
        dtype=np.uint8,
    )
    for row in range(APRILGRID_TEST_SIZE[1] + 1):
        for col in range(APRILGRID_TEST_SIZE[0] + 1):
            y0 = margin + row * pitch_pixels
            x0 = margin + col * pitch_pixels
            board[y0:y0 + gap_pixels, x0:x0 + gap_pixels] = 0
    for visual_row in range(APRILGRID_TEST_SIZE[1]):
        for col in range(APRILGRID_TEST_SIZE[0]):
            tag_row = APRILGRID_TEST_SIZE[1] - 1 - visual_row
            marker_id = tag_row * APRILGRID_TEST_SIZE[0] + col
            if hasattr(cv2.aruco, "generateImageMarker"):
                marker = cv2.aruco.generateImageMarker(
                    dictionary, marker_id, tag_pixels, None, 2
                )
            else:
                marker = cv2.aruco.drawMarker(
                    dictionary, marker_id, tag_pixels, borderBits=2
                )
            marker = np.rot90(marker, 2)
            y0 = margin + gap_pixels + visual_row * pitch_pixels
            x0 = margin + gap_pixels + col * pitch_pixels
            board[y0:y0 + tag_pixels, x0:x0 + tag_pixels] = marker

    board_width = (
        APRILGRID_TEST_TAG_SIZE
        + (APRILGRID_TEST_SIZE[0] - 1)
        * (APRILGRID_TEST_TAG_SIZE + APRILGRID_TEST_TAG_GAP)
    )
    board_height = (
        APRILGRID_TEST_TAG_SIZE
        + (APRILGRID_TEST_SIZE[1] - 1)
        * (APRILGRID_TEST_TAG_SIZE + APRILGRID_TEST_TAG_GAP)
    )
    lo = float(margin + gap_pixels) - 0.5
    hi_x = lo + float(content_width)
    hi_y = lo + float(content_height)
    source = np.asarray(
        ((lo, hi_y), (hi_x, hi_y), (hi_x, lo), (lo, lo)),
        dtype=np.float32,
    )
    boundary = np.asarray(
        (
            (0.0, 0.0, 0.0),
            (board_width, 0.0, 0.0),
            (board_width, board_height, 0.0),
            (0.0, board_height, 0.0),
        ),
        dtype=np.float32,
    )
    delta_rotation = cv2.Rodrigues(
        np.asarray(rotation_vector, dtype=np.float64)
    )[0]
    rotation = delta_rotation.dot(np.diag((1.0, -1.0, -1.0)))
    rvec = cv2.Rodrigues(rotation)[0].reshape(3)
    board_center = np.asarray((board_width / 2.0, board_height / 2.0, 0.0))
    u, v = center_pixel
    camera_center = np.asarray(
        (
            (float(u) - APRILGRID_TEST_K[0, 2]) * depth_m / APRILGRID_TEST_K[0, 0],
            (float(v) - APRILGRID_TEST_K[1, 2]) * depth_m / APRILGRID_TEST_K[1, 1],
            depth_m,
        )
    )
    translation = camera_center - rotation.dot(board_center)
    projected, _jacobian = cv2.projectPoints(
        boundary,
        rvec,
        translation,
        APRILGRID_TEST_K,
        np.zeros(5, dtype=np.float64),
    )
    transform = cv2.getPerspectiveTransform(
        source, projected.reshape(4, 2).astype(np.float32)
    )
    gray = cv2.warpPerspective(
        board,
        transform,
        APRILGRID_TEST_IMAGE_SIZE,
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=255,
    )
    encoded, jpeg = cv2.imencode(".jpg", gray, (cv2.IMWRITE_JPEG_QUALITY, 94))
    if not encoded:
        raise AssertionError("could not encode synthetic AprilGrid frame")
    decoded = cv2.imdecode(jpeg, cv2.IMREAD_GRAYSCALE)
    return cv2.cvtColor(decoded, cv2.COLOR_GRAY2BGR)


PRODUCTION_IMAGE_SIZE = (3840, 2160)
PRODUCTION_K = np.asarray(
    (
        (1344.398473, 0.0, 1919.5),
        (0.0, 1344.398473, 1079.5),
        (0.0, 0.0, 1.0),
    ),
    dtype=np.float64,
)


def render_production_aprilgrid(tag_size, tag_gap, depth_m, rotation=(0.0, 0.0, 0.0)):
    """Warp a 6x6 tag36h11 board onto the local-fleet 4K 110° camera."""
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
    cols, rows = 6, 6
    tag_pixels = 80
    gap_pixels = 24
    margin = 48
    pitch = tag_pixels + gap_pixels
    content_width = cols * tag_pixels + (cols - 1) * gap_pixels
    content_height = rows * tag_pixels + (rows - 1) * gap_pixels
    board = np.full(
        (content_height + 2 * gap_pixels + 2 * margin,
         content_width + 2 * gap_pixels + 2 * margin),
        255,
        dtype=np.uint8,
    )
    for row in range(rows + 1):
        for col in range(cols + 1):
            y0 = margin + row * pitch
            x0 = margin + col * pitch
            board[y0:y0 + gap_pixels, x0:x0 + gap_pixels] = 0
    for visual_row in range(rows):
        for col in range(cols):
            marker_id = (rows - 1 - visual_row) * cols + col
            marker = cv2.aruco.generateImageMarker(
                dictionary, marker_id, tag_pixels, None, 2
            )
            marker = np.rot90(marker, 2)
            y0 = margin + gap_pixels + visual_row * pitch
            x0 = margin + gap_pixels + col * pitch
            board[y0:y0 + tag_pixels, x0:x0 + tag_pixels] = marker
    board_width = tag_size + (cols - 1) * (tag_size + tag_gap)
    board_height = tag_size + (rows - 1) * (tag_size + tag_gap)
    lo = float(margin + gap_pixels) - 0.5
    source = np.asarray(
        (
            (lo, lo + content_height),
            (lo + content_width, lo + content_height),
            (lo + content_width, lo),
            (lo, lo),
        ),
        dtype=np.float32,
    )
    boundary = np.asarray(
        (
            (0.0, 0.0, 0.0),
            (board_width, 0.0, 0.0),
            (board_width, board_height, 0.0),
            (0.0, board_height, 0.0),
        ),
        dtype=np.float32,
    )
    delta_rotation = cv2.Rodrigues(np.asarray(rotation, dtype=np.float64))[0]
    rotation_matrix = delta_rotation.dot(np.diag((1.0, -1.0, -1.0)))
    rvec = cv2.Rodrigues(rotation_matrix)[0].reshape(3)
    board_center = np.asarray((board_width / 2.0, board_height / 2.0, 0.0))
    camera_center = np.asarray((0.0, 0.0, float(depth_m)))
    translation = camera_center - rotation_matrix.dot(board_center)
    projected, _jacobian = cv2.projectPoints(
        boundary,
        rvec,
        translation,
        PRODUCTION_K,
        np.zeros(5, dtype=np.float64),
    )
    transform = cv2.getPerspectiveTransform(
        source, projected.reshape(4, 2).astype(np.float32)
    )
    gray = cv2.warpPerspective(
        board,
        transform,
        PRODUCTION_IMAGE_SIZE,
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=255,
    )
    encoded, jpeg = cv2.imencode(".jpg", gray, (cv2.IMWRITE_JPEG_QUALITY, 94))
    if not encoded:
        raise AssertionError("could not encode production AprilGrid frame")
    decoded = cv2.imdecode(jpeg, cv2.IMREAD_GRAYSCALE)
    return cv2.cvtColor(decoded, cv2.COLOR_GRAY2BGR), jpeg.tobytes()


def _project_aprilgrid_view_coverage(view, objects, extent):
    width, height = 3840, 2160
    camera_matrix = np.array(
        ((1344.398473, 0.0, 1919.5), (0.0, 1344.398473, 1079.5), (0.0, 0.0, 1.0)),
        dtype=np.float64,
    )
    board_center = np.array((2.0, 0.0, 2.2), dtype=np.float64)
    link_from_optical = np.array(((0.0, 0.0, 1.0), (-1.0, 0.0, 0.0), (0.0, -1.0, 0.0)))
    world_from_board = np.array(((0.0, 0.0, -1.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)))
    position = np.asarray(view["position"], dtype=np.float64)
    delta = board_center - position
    yaw = math.atan2(delta[1], delta[0]) + float(view["yaw_offset"])
    pitch = -math.atan2(delta[2], math.hypot(delta[0], delta[1])) + float(view["pitch_offset"])
    roll = float(view["roll"])
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    world_from_link = np.array(
        ((cy, -sy, 0.0), (sy, cy, 0.0), (0.0, 0.0, 1.0))
    ).dot(np.array(((cp, 0.0, sp), (0.0, 1.0, 0.0), (-sp, 0.0, cp)))).dot(
        np.array(((1.0, 0.0, 0.0), (0.0, cr, -sr), (0.0, sr, cr)))
    )
    optical_pos = position + 0.067 * world_from_link[:, 0]
    optical_from_world = (world_from_link.dot(link_from_optical)).T
    centered = objects - np.array((extent / 2.0, extent / 2.0, 0.0))
    world = world_from_board.dot(centered.T).T + board_center
    rotation, _ = cv2.Rodrigues(optical_from_world)
    translation = -optical_from_world.dot(optical_pos)
    pixels = cv2.projectPoints(
        world.astype(np.float64), rotation, translation, camera_matrix, np.zeros(5),
    )[0].reshape(-1, 2)
    camera_z = optical_from_world.dot((world - optical_pos).T).T[:, 2]
    visible = (
        (camera_z > 1.0e-6)
        & (pixels[:, 0] >= 0.0) & (pixels[:, 0] < width)
        & (pixels[:, 1] >= 0.0) & (pixels[:, 1] < height)
    )
    if int(np.count_nonzero(visible)) < 16:
        return (0.5, 0.5, 0.0, 0.0)
    return intrinsic_solver._aprilgrid_coverage(pixels[visible], objects[visible], width, height)


def make_service(output_file, **kwargs):
    # 8x6 squares -> 7x5 interior corners.
    return IntrinsicCalibrationService(
        board_size=(7, 5), square=0.20, output_file=str(output_file),
        camera_name="usb_cam",
        media_source="usb_cam", display_width=640,
        **kwargs,
    )


def make_diagnostic_result(
    image_size,
    sample_count,
    *,
    rank_deficient=False,
    failed=(),
    held_out_rms_max=0.405,
    ray_max=0.008,
):
    parameter_names = ("fx", "fy", "cx", "cy", "k1", "k2", "p1", "p2", "k3")
    folds = tuple(
        intrinsic_solver.IntrinsicFoldEstimate(
            omitted_view_index=index,
            rms_reprojection_error_px=0.4,
            parameters=(638.0, 637.0, 600.0, 390.0, 0.01, -0.02, 0.0, 0.0, 0.0),
            held_out_rms_reprojection_error_px=held_out_rms_max,
            held_out_mean_reprojection_error_px=0.4,
            held_out_max_reprojection_error_px=0.42,
            held_out_point_errors_px=(0.39, 0.40, 0.42),
            held_out_rotation_vector=(0.0, 0.0, 0.0),
            held_out_translation_vector=(0.0, 0.0, 1.0),
            undistorted_ray_rms_equivalent_px=0.005,
            undistorted_ray_max_equivalent_px=ray_max,
        )
        for index in range(sample_count)
        if index not in failed
    )
    stability = intrinsic_solver.IntrinsicStabilityDiagnostics(
        method="leave_one_view_out",
        parameter_names=parameter_names,
        reference_parameters=folds[0].parameters if folds else (0.0,) * 9,
        folds=folds,
        failed_omitted_view_indices=tuple(failed),
        parameter_standard_deviation=(0.0,) * 9 if folds else None,
        parameter_span=(0.0,) * 9 if folds else None,
        maximum_absolute_delta=(0.0,) * 9 if folds else None,
        maximum_relative_delta=(0.0,) * 9 if folds else None,
        held_out_rms_mean_px=0.405 if folds else None,
        held_out_rms_max_px=held_out_rms_max if folds else None,
        undistorted_ray_rms_equivalent_px=0.005 if folds else None,
        undistorted_ray_max_equivalent_px=ray_max if folds else None,
    )
    diagnostics = intrinsic_solver.IntrinsicCalibrationDiagnostics(
        finite=True,
        parameter_names=parameter_names,
        per_view_errors_px=(0.4,) * sample_count,
        intrinsic_standard_deviations=(0.01,) * 9,
        rotation_vectors=((0.0, 0.0, 0.0),) * sample_count,
        translation_vectors=((0.0, 0.0, 1.0),) * sample_count,
        projected_intrinsic_rank=8 if rank_deficient else 9,
        projected_intrinsic_parameter_count=9,
        projected_intrinsic_rank_deficient=rank_deficient,
        projected_intrinsic_condition_number=float("inf") if rank_deficient else 500.0,
        projected_intrinsic_rank_tolerance=1e-12,
        projected_intrinsic_singular_values=(1.0,) * (8 if rank_deficient else 9),
        projected_intrinsic_column_norms=(1.0,) * 9,
        stability=stability,
        pool_sample_count=sample_count,
        selected_view_indices=tuple(range(sample_count)),
        rejected_views=(),
        initial_per_view_errors_px=(0.4,) * sample_count,
        observation_uncertainty_px=intrinsic_solver.observation_uncertainty_px(
            "checkerboard"
        ),
    )
    return intrinsic_solver.IntrinsicResult(
        camera_matrix=np.array([
            [638.0, 0.0, 600.0],
            [0.0, 637.0, 390.0],
            [0.0, 0.0, 1.0],
        ]),
        distortion=np.array([0.01, -0.02, 0.0, 0.0, 0.0]),
        image_size=image_size,
        rms_reprojection_error_px=0.4,
        sample_count=sample_count,
        diagnostics=diagnostics,
    )


class FakeCameraControl:
    def __init__(self):
        self.positions = []
        self.current_pose = None

    def goto(self, position, yaw_offset, pitch_offset, roll):
        self.positions.append(list(position))
        self.current_pose = {"position": list(position)}

    def reset(self):
        self.current_pose = None

    def current(self):
        return self.current_pose

    def current_position(self):
        if self.current_pose is None:
            return None
        return self.current_pose["position"]

    def current_optical_pose(self):
        if self.current_pose is None:
            return None
        return {
            "position": tuple(self.current_pose["position"]),
            "orientation": (0.0, 0.0, 0.0, 1.0),
        }


class IntrinsicServiceTest(unittest.TestCase):
    def test_algorithm_provenance_preserves_the_selected_feature_model(self):
        feature_model = "checkerboard_corners_v1"
        self.assertEqual(
            intrinsic_algorithm_provenance(feature_model)["feature_model"],
            feature_model,
        )

    def test_storage_contract_is_explicitly_partitioned_by_mode_and_camera(self):
        root = "/home/operator/Documents/XGC/Calibration/camera"
        self.assertEqual(
            intrinsic_calibration_directory(root, "sim", "front_camera"),
            Path(root) / "sim/front_camera",
        )
        self.assertEqual(
            intrinsic_calibration_directory(root, "phy", "front_camera"),
            Path(root) / "phy/front_camera",
        )
        for mode, camera in (("simulation", "front_camera"), ("phy", "../camera"), ("phy", "")):
            with self.assertRaises(ValueError):
                intrinsic_calibration_directory(root, mode, camera)

    def test_90_degree_simulation_sweep_stays_on_the_viewing_side(self):
        views = recommended_views((2.0, 0.0, 2.2))
        near = next(view for view in views if view["name"] == "near maximum")
        oblique_high = next(view for view in views if view["name"] == "oblique high")
        self.assertEqual(near["position"], [0.7, 0.0, 2.2])
        self.assertGreater(oblique_high["position"][2], 2.2)
        self.assertEqual(oblique_high["roll"], 0.0)
        self.assertEqual(len({tuple(view["position"]) for view in views}), len(views))
        self.assertGreaterEqual(min(view["position"][2] for view in views), 1.5)
        self.assertTrue(all(view["position"][0] < 2.0 for view in views))
        for name in ("left oblique", "right oblique", "oblique high", "oblique low"):
            view = next(item for item in views if item["name"] == name)
            self.assertEqual(view["yaw_offset"], 0.0)
            self.assertEqual(view["pitch_offset"], 0.0)
            self.assertEqual(view["roll"], 0.0)

    def test_field_aprilgrid_scales_simulation_views_to_target_extent(self):
        views = recommended_views((2.0, 0.0, 2.2), 0.66, camera_optical_origin=0.067)
        left = next(view for view in views if view["name"] == "left edge")
        near = next(view for view in views if view["name"] == "near maximum")
        self.assertEqual(left["position"], [0.8, 0.02, 2.2])
        self.assertEqual(left["yaw_offset"], -0.58)
        self.assertEqual(near["position"], [1.46, 0.0, 2.2])
        self.assertGreaterEqual(min(view["position"][2] for view in views), 1.5)
        self.assertEqual(len({tuple(view["position"]) for view in views}), len(views))

    def test_a4_aprilgrid_scales_optical_distance_instead_of_camera_link_distance(self):
        field = recommended_views(
            (2.0, 0.0, 2.2), 0.66, camera_optical_origin=0.067
        )
        a4 = recommended_views(
            (2.0, 0.0, 2.2), 0.18, camera_optical_origin=0.067
        )
        field_near = next(view for view in field if view["name"] == "near maximum")
        a4_near = next(view for view in a4 if view["name"] == "near maximum")
        field_optical_distance = 2.0 - field_near["position"][0] - 0.067
        a4_optical_distance = 2.0 - a4_near["position"][0] - 0.067
        self.assertAlmostEqual(
            a4_optical_distance / field_optical_distance,
            0.18 / 0.66,
            delta=0.015,
        )
        self.assertEqual(a4_near["position"], [1.81, 0.0, 2.2])

    def test_aprilgrid_auto_sweep_fills_homography_skew_for_both_boards(self):
        # In-plane roll used to stall Skew at ~75% / ~83%. Off-axis look-at
        # must fill the AprilGrid anisotropy bar for both production plates.
        for extent, tag, gap in (
            (0.66, 0.088, 0.0264),
            (0.18, 0.024, 0.0072),
        ):
            views = recommended_views(
                (2.0, 0.0, 2.2), extent, camera_optical_origin=0.067
            )
            objects = np.concatenate(
                [
                    intrinsic_solver.aprilgrid_tag_object_points((6, 6), tag, gap, 0, tag_id)
                    for tag_id in range(36)
                ],
                axis=0,
            )
            samples = [
                _project_aprilgrid_view_coverage(view, objects, extent)
                for view in views
            ]
            bars, goodenough = intrinsic_solver.coverage(samples)
            skew = next(bar["progress"] for bar in bars if bar["label"] == "Skew")
            self.assertGreaterEqual(
                skew,
                1.0,
                msg="extent {} Skew stalled at {:.3f}".format(extent, skew),
            )
            self.assertTrue(goodenough, msg="extent {} coverage incomplete: {}".format(extent, bars))

    def test_web_assets_use_proxy_safe_relative_urls(self):
        index = (WEB_ROOT / "index.html").read_text(encoding="utf-8")
        app = (WEB_ROOT / "app.js").read_text(encoding="utf-8")
        styles = (WEB_ROOT / "styles.css").read_text(encoding="utf-8")
        self.assertIn('href="styles.css"', index)
        self.assertIn('src="app.js"', index)
        self.assertIn('type="module"', index)
        self.assertNotIn('"/api/v1/intrinsic/', app)
        self.assertIn("api/v1/intrinsic/state", app)
        self.assertIn("xgc-app-shell", app)
        self.assertIn("Board detection", app)
        self.assertIn("detection-status", app)
        self.assertIn("URL.createObjectURL", app)
        self.assertIn(".xgc-topbar", styles)

    def test_entrypoint_runs_one_continuous_detector_for_both_camera_origins(self):
        source = ENTRYPOINT.read_text(encoding="utf-8")
        self.assertIn('rospy.get_param("~auto_capture", True)', source)
        self.assertIn('rospy.get_param("~auto_capture_interval", 0.2)', source)
        self.assertNotIn('rospy.get_param("~auto_capture", not bool(', source)
        self.assertIn('rospy.get_param("~maximum_detect_width", display_width)', source)
        self.assertIn('rospy.get_param("~detection_target_pixels", 640 * 480)', " ".join(source.split()))
        self.assertIn('kwargs={"poll_interval": 0.05}', source)
        self.assertIn('rospy.get_param("~calibration_root"', source)
        self.assertIn('rospy.get_param("~calibration_mode", "sim")', source)
        self.assertIn('rospy.get_param("~camera_name", "usb_cam")', source)
        self.assertIn('intrinsic_calibration_directory(', source)
        self.assertNotIn('rospy.get_param("~output_file"', source)
        self.assertNotIn('rospy.get_param("~references_dir"', source)
        self.assertIn("time.sleep(0.1)", WEB_SERVICE_SOURCE.read_text(encoding="utf-8"))

    def test_process_frame_collects_a_board_sample(self):
        with tempfile.TemporaryDirectory() as directory:
            service = make_service(Path(directory) / "intrinsics.yaml")
            service.process_frame(render_board())
            state = service.state()
            self.assertEqual(state["mode"], "intrinsic")
            self.assertEqual(state["samples"], 1)
            self.assertEqual([bar["label"] for bar in state["coverage"]], ["X", "Y", "Size", "Skew"])
            self.assertEqual(state["detection"]["status"], "detected")
            self.assertEqual(state["detection"]["corner_count"], 35)
            self.assertEqual(state["detection"]["expected_corner_count"], 35)
            self.assertTrue(state["detection"]["accepted"])
            self.assertFalse(state["detection"]["duplicate"])
            self.assertEqual([metric["label"] for metric in state["detection"]["metrics"]], ["X", "Y", "Size", "Skew"])
            self.assertTrue(service.image_jpeg().startswith(b"\xff\xd8"))

    def test_simulation_alignment_uses_the_captured_render_pose(self):
        with tempfile.TemporaryDirectory() as directory:
            service = make_service(Path(directory) / "intrinsics.yaml")
            camera = FakeCameraControl()
            service.attach_camera_control(camera)
            target = service.views[0]["position"]
            service.goto(0)
            service.attach_frame_capture(lambda: SimpleNamespace(
                bgr=render_board(), render_position=(99.0, 99.0, 99.0),
                render_orientation=(0.0, 0.0, 0.0, 1.0),
            ))
            service._capture_frame()
            self.assertFalse(service.target_done[0])
            self.assertEqual(len(service.samples), 0)

            service.attach_frame_capture(lambda: SimpleNamespace(
                bgr=render_board(), render_position=tuple(target),
                render_orientation=(0.0, 0.0, 0.0, 1.0),
            ))
            service._capture_frame()
            self.assertTrue(service.target_done[0])

    def test_production_capture_anchors_real_sensor_offset_then_accepts_next_frame(self):
        with tempfile.TemporaryDirectory() as directory:
            service = make_service(Path(directory) / "intrinsics.yaml")
            camera = FakeCameraControl()
            service.attach_camera_control(camera)
            service.views[0]["position"] = [0.8, 0.02, 2.2]
            render_position = (0.862546, -0.00401979, 2.2)
            frame = SimpleNamespace(
                bgr=render_board(),
                render_position=render_position,
                render_orientation=(0.0, 0.0, 0.0, 1.0),
            )
            service.attach_frame_capture(lambda: frame)

            service.goto(0)
            service._capture_frame()
            self.assertEqual(service.samples, [])
            self.assertFalse(service.target_done[0])
            self.assertEqual(service._target_expected_pose["position"], render_position)
            self.assertTrue(service._target_expected_pose["render_pose_anchored"])

            service._capture_frame()
            self.assertTrue(service.target_done[0])
            self.assertEqual(service.sample_target_ids, [0])

    def test_sensor_offset_anchor_rejects_far_and_stale_orientation_frames(self):
        with tempfile.TemporaryDirectory() as directory:
            service = make_service(Path(directory) / "intrinsics.yaml")
            camera = FakeCameraControl()
            service.attach_camera_control(camera)
            target = np.asarray(service.views[0]["position"], dtype=np.float64)
            good_position = tuple(target + np.asarray((0.062546, -0.02401979, 0.0)))
            frames = iter((
                SimpleNamespace(
                    bgr=render_board(),
                    render_position=tuple(target + np.asarray((0.2, 0.0, 0.0))),
                    render_orientation=(0.0, 0.0, 0.0, 1.0),
                ),
                SimpleNamespace(
                    bgr=render_board(),
                    render_position=good_position,
                    render_orientation=(0.0, 0.0, np.sin(0.05), np.cos(0.05)),
                ),
                SimpleNamespace(
                    bgr=render_board(),
                    render_position=good_position,
                    render_orientation=(0.0, 0.0, 0.0, 1.0),
                ),
                SimpleNamespace(
                    bgr=render_board(),
                    render_position=good_position,
                    render_orientation=(0.0, 0.0, 0.0, 1.0),
                ),
            ))
            service.attach_frame_capture(lambda: next(frames))
            service.goto(0)

            service._capture_frame()
            service._capture_frame()
            self.assertEqual(service.samples, [])
            self.assertFalse(service._target_expected_pose["render_pose_anchored"])
            self.assertEqual(
                service._target_expected_pose["position"], tuple(target)
            )

            service._capture_frame()
            self.assertEqual(service.samples, [])
            self.assertTrue(service._target_expected_pose["render_pose_anchored"])
            service._capture_frame()
            self.assertEqual(service.sample_target_ids, [0])

    def test_sensor_offset_anchor_requires_the_current_target_capture_token(self):
        with tempfile.TemporaryDirectory() as directory:
            service = make_service(Path(directory) / "intrinsics.yaml")
            camera = FakeCameraControl()
            service.attach_camera_control(camera)
            capture_started = threading.Event()
            release_capture = threading.Event()

            def delayed_capture():
                capture_started.set()
                self.assertTrue(release_capture.wait(timeout=1.0))
                target = np.asarray(camera.current_position(), dtype=np.float64)
                return SimpleNamespace(
                    bgr=render_board(),
                    render_position=tuple(
                        target + np.asarray((0.062546, -0.02401979, 0.0))
                    ),
                    render_orientation=(0.0, 0.0, 0.0, 1.0),
                )

            service.goto(0)
            service.attach_frame_capture(delayed_capture)
            capture_thread = threading.Thread(target=service._capture_frame)
            capture_thread.start()
            self.assertTrue(capture_started.wait(timeout=1.0))
            service.goto(1)
            release_capture.set()
            capture_thread.join(timeout=1.0)
            self.assertFalse(capture_thread.is_alive())
            self.assertEqual(service.samples, [])
            self.assertFalse(service._target_expected_pose["render_pose_anchored"])

            service._capture_frame()
            self.assertEqual(service.samples, [])
            self.assertTrue(service._target_expected_pose["render_pose_anchored"])
            service._capture_frame()
            self.assertEqual(service.sample_target_ids, [1])

    def test_manual_goto_marks_the_explicit_target_when_guide_positions_overlap(self):
        with tempfile.TemporaryDirectory() as directory:
            service = make_service(Path(directory) / "intrinsics.yaml")
            camera = FakeCameraControl()
            service.attach_camera_control(camera)
            service.views[0]["position"] = [0.0, 0.0, 0.0]
            service.views[1]["position"] = [0.0, 0.01, 0.0]
            service.goto(1)
            service.process_frame(
                render_board(),
                render_position=tuple(service.views[1]["position"]),
                render_orientation=(0.0, 0.0, 0.0, 1.0),
            )
            self.assertFalse(service.target_done[0])
            self.assertTrue(service.target_done[1])
            self.assertEqual(len(service.samples), 1)

    def test_simulation_target_requires_fresh_pose_metadata_and_keeps_mirror_samples(self):
        with tempfile.TemporaryDirectory() as directory:
            service = make_service(Path(directory) / "intrinsics.yaml")
            camera = FakeCameraControl()
            service.attach_camera_control(camera)
            frame = render_board()

            service.goto(0)
            service.process_frame(frame)
            self.assertEqual(len(service.samples), 0)
            self.assertFalse(service.target_done[0])

            first_target = tuple(service.views[0]["position"])
            service.process_frame(
                frame,
                render_position=first_target,
                render_orientation=(0.0, 0.0, 0.0, 1.0),
            )
            self.assertEqual(len(service.samples), 1)
            self.assertTrue(service.target_done[0])

            service.goto(1)
            service.process_frame(
                frame,
                render_position=first_target,
                render_orientation=(0.0, 0.0, 0.0, 1.0),
            )
            # This is a valid pose from the previous target, not an artificial
            # orientation failure. Target identity must reject the stale replay.
            self.assertEqual(len(service.samples), 1)
            self.assertFalse(service.target_done[1])

            second_target = tuple(service.views[1]["position"])
            service.process_frame(
                frame,
                render_position=second_target,
                render_orientation=(0.0, 0.0, 0.0, 1.0),
            )
            # Identical image-space coverage is still one sample per authored
            # mirror target in simulation.
            self.assertEqual(len(service.samples), 2)
            self.assertTrue(service.target_done[1])
            self.assertEqual(service.sample_target_ids, [0, 1])

    def test_capture_started_before_next_goto_cannot_enter_the_new_target_epoch(self):
        with tempfile.TemporaryDirectory() as directory:
            service = make_service(Path(directory) / "intrinsics.yaml")
            camera = FakeCameraControl()
            service.attach_camera_control(camera)
            capture_started = threading.Event()
            release_capture = threading.Event()

            def delayed_capture():
                capture_started.set()
                self.assertTrue(release_capture.wait(timeout=1.0))
                return SimpleNamespace(
                    bgr=render_board(),
                    # Return the new pose deliberately: even matching metadata
                    # cannot make a transaction that began in the old epoch fresh.
                    render_position=tuple(camera.current_position()),
                    render_orientation=(0.0, 0.0, 0.0, 1.0),
                )

            service.goto(0)
            service.attach_frame_capture(delayed_capture)
            capture_errors = []

            def run_delayed_capture():
                try:
                    service._capture_frame()
                except Exception as error:
                    capture_errors.append(error)

            capture_thread = threading.Thread(target=run_delayed_capture)
            capture_thread.start()
            self.assertTrue(capture_started.wait(timeout=1.0))
            service.goto(1)
            release_capture.set()
            capture_thread.join(timeout=1.0)
            self.assertFalse(capture_thread.is_alive())
            self.assertEqual(capture_errors, [])
            self.assertEqual(service.samples, [])
            self.assertFalse(any(service.target_done))

            service.attach_frame_capture(lambda: SimpleNamespace(
                bgr=render_board(),
                render_position=tuple(camera.current_position()),
                render_orientation=(0.0, 0.0, 0.0, 1.0),
            ))
            service._capture_frame()
            self.assertEqual(service.sample_target_ids, [1])
            self.assertTrue(service.target_done[1])

    def test_reduced_detection_frame_keeps_source_calibration_coordinates(self):
        with tempfile.TemporaryDirectory() as directory:
            service = make_service(Path(directory) / "intrinsics.yaml")
            frame = render_board()
            reduced_height, reduced_width = frame.shape[:2]
            service.attach_frame_capture(lambda: SimpleNamespace(
                bgr=frame,
                width=reduced_width * 4,
                height=reduced_height * 4,
            ))
            service._capture_frame()
            self.assertEqual(service.image_size, (reduced_width * 4, reduced_height * 4))
            points = service.image_points[0].reshape(-1, 2)
            self.assertGreater(float(points[:, 0].max()), float(reduced_width))
            state = service.state()
            self.assertEqual(state["detection"]["frame_width"], reduced_width * 4)
            self.assertEqual(state["detection"]["frame_height"], reduced_height * 4)

    def test_distinct_aprilgrid_sample_refines_on_source_jpeg_before_storage(self):
        with tempfile.TemporaryDirectory() as directory:
            service = IntrinsicCalibrationService(
                board_size=(2, 2),
                square=0.088,
                output_file=str(Path(directory) / "intrinsics.yaml"),
                camera_name="usb_cam",
                board_type="aprilgrid",
                tag_spacing=0.0264,
                min_tags=4,
            )
            reduced = np.zeros((120, 160, 3), np.uint8)
            source = np.zeros((480, 640, 3), np.uint8)
            ok, encoded = cv2.imencode(".jpg", source)
            self.assertTrue(ok)
            corners = np.asarray(
                [[20, 20], [40, 20], [40, 40], [20, 40]],
                dtype=np.float32,
            ).reshape(-1, 1, 2)
            detection = intrinsic_solver.BoardDetection(
                image_points=corners,
                object_points=np.zeros((4, 3), np.float32),
                coverage=(0.2, 0.3, 0.4, 0.5),
                calibration_image_points=corners,
                calibration_object_points=np.zeros((4, 3), np.float32),
            )
            refined = corners * 4.0 + np.asarray([0.25, 0.5], np.float32)
            refined_objects = np.arange(12, dtype=np.float32).reshape(4, 3)
            with patch.object(
                intrinsic_solver,
                "detect_board",
                return_value=detection,
            ) as detect, patch.object(
                intrinsic_solver,
                "localize_aprilgrid_source_corners",
                return_value=(refined, refined_objects, detection.coverage),
            ) as localize:
                service.process_frame(
                    reduced,
                    source_image_size=(640, 480),
                    source_jpeg=encoded.tobytes(),
                )
            self.assertEqual(
                [call.args[2] for call in detect.call_args_list], [160]
            )
            self.assertEqual(localize.call_args.args[0].shape, (480, 640))
            np.testing.assert_allclose(service.image_points[0], refined)
            np.testing.assert_allclose(service.object_points[0], refined_objects)

    def test_aprilgrid_candidate_redecodes_original_jpeg_for_adaptive_search(self):
        with tempfile.TemporaryDirectory() as directory:
            service = IntrinsicCalibrationService(
                board_size=(2, 2), square=0.088,
                output_file=str(Path(directory) / "intrinsics.yaml"),
                camera_name="usb_cam",
                board_type="aprilgrid", tag_spacing=0.0264, min_tags=4,
            )
            reduced = np.zeros((270, 480, 3), np.uint8)
            source = np.zeros((1080, 1920, 3), np.uint8)
            ok, encoded = cv2.imencode(".jpg", source)
            self.assertTrue(ok)
            corners = np.asarray(
                [[100, 100], [140, 100], [140, 140], [100, 140]],
                dtype=np.float32,
            ).reshape(-1, 1, 2)
            detection = intrinsic_solver.BoardDetection(
                image_points=corners,
                object_points=np.zeros((4, 3), np.float32),
                coverage=(0.2, 0.3, 0.4, 0.5),
                calibration_image_points=corners,
                calibration_object_points=np.zeros((4, 3), np.float32),
            )
            with patch.object(
                intrinsic_solver,
                "detect_board",
                side_effect=[None, detection],
            ) as detect, patch.object(
                intrinsic_solver, "aprilgrid_has_candidate_evidence", return_value=True
            ):
                service.process_frame(
                    reduced,
                    source_image_size=(1920, 1080),
                    source_jpeg=encoded.tobytes(),
                )
            # Adaptive decode already sits on the source plane, so admission
            # must not run the same detector a third time.
            self.assertEqual(
                [call.args[2] for call in detect.call_args_list], [480, 1920]
            )
            self.assertEqual(service.state()["detection"]["corner_count"], 4)
            self.assertTrue(service.state()["detection"]["accepted"])
            self.assertEqual(service.state()["samples"], 1)

    def test_aprilgrid_search_hold_retries_source_when_vga_evidence_vanishes(self):
        with tempfile.TemporaryDirectory() as directory:
            service = IntrinsicCalibrationService(
                board_size=(2, 2), square=0.088,
                output_file=str(Path(directory) / "intrinsics.yaml"),
                camera_name="usb_cam",
                board_type="aprilgrid", tag_spacing=0.0264, min_tags=4,
            )
            reduced = np.zeros((270, 480, 3), np.uint8)
            source = np.zeros((2160, 3840, 3), np.uint8)
            ok, encoded = cv2.imencode(".jpg", source)
            self.assertTrue(ok)
            jpeg = encoded.tobytes()
            corners = np.asarray(
                [[100, 100], [140, 100], [140, 140], [100, 140]],
                dtype=np.float32,
            ).reshape(-1, 1, 2)
            detection = intrinsic_solver.BoardDetection(
                image_points=corners,
                object_points=np.zeros((4, 3), np.float32),
                coverage=(0.2, 0.3, 0.4, 0.5),
                calibration_image_points=corners,
                calibration_object_points=np.zeros((4, 3), np.float32),
            )
            localized = (
                corners * 8.0,
                np.arange(12, dtype=np.float32).reshape(4, 3),
                detection.coverage,
            )

            def process(detect_side_effect, evidence):
                with patch.object(
                    intrinsic_solver, "detect_board", side_effect=detect_side_effect
                ) as detect, patch.object(
                    intrinsic_solver, "aprilgrid_has_candidate_evidence", return_value=evidence
                ), patch.object(
                    intrinsic_solver,
                    "localize_aprilgrid_source_corners",
                    return_value=localized,
                ):
                    service.process_frame(
                        reduced,
                        source_image_size=(3840, 2160),
                        source_jpeg=jpeg,
                        source_snapshot_id="hold-{}".format(evidence),
                    )
                return [call.args[2] for call in detect.call_args_list]

            first_widths = process([None, None, detection], True)
            self.assertIn(3840, first_widths)
            self.assertEqual(service.state()["detection"]["status"], "detected")
            self.assertEqual(
                service._aprilgrid_search_hold_frames, APRILGRID_SEARCH_HOLD_FRAMES
            )

            second_widths = process([None, None, detection], False)
            self.assertIn(3840, second_widths)
            self.assertEqual(service.state()["detection"]["status"], "detected")

            service._aprilgrid_search_hold_frames = 0
            service._aprilgrid_search_hold_width = 0
            cold_widths = process([None, None], False)
            self.assertNotIn(3840, cold_widths)
            self.assertEqual(service.state()["detection"]["status"], "not_detected")

    def test_aprilgrid_empty_scene_does_not_search_source_without_a_hold(self):
        with tempfile.TemporaryDirectory() as directory:
            service = IntrinsicCalibrationService(
                board_size=(2, 2), square=0.088,
                output_file=str(Path(directory) / "intrinsics.yaml"),
                camera_name="usb_cam",
                board_type="aprilgrid", tag_spacing=0.0264, min_tags=4,
            )
            reduced = np.zeros((270, 480, 3), np.uint8)
            source = np.zeros((2160, 3840, 3), np.uint8)
            ok, encoded = cv2.imencode(".jpg", source)
            self.assertTrue(ok)
            with patch.object(
                intrinsic_solver, "detect_board", wraps=intrinsic_solver.detect_board
            ) as detect:
                service.process_frame(
                    reduced,
                    source_image_size=(3840, 2160),
                    source_jpeg=encoded.tobytes(),
                    source_snapshot_id="empty",
                )
            widths = [call.args[2] for call in detect.call_args_list]
            self.assertTrue(widths)
            self.assertNotIn(3840, widths)
            self.assertEqual(service.state()["detection"]["status"], "not_detected")
            self.assertEqual(service._aprilgrid_search_hold_frames, 0)

    def test_adaptive_aprilgrid_decode_does_not_upscale_a_reduced_jpeg(self):
        source = np.full((2160, 3840, 3), 127, np.uint8)
        ok, encoded = cv2.imencode(".jpg", source)
        self.assertTrue(ok)
        image = IntrinsicCalibrationService._decode_adaptive_aprilgrid_frame(
            encoded.tobytes(), 3840
        )
        self.assertIsNotNone(image)
        self.assertEqual(image.shape[1], 1920)
        self.assertEqual(image.shape[0], 1080)

    def test_unrefined_search_corners_never_enter_the_pool_without_source_jpeg(self):
        with tempfile.TemporaryDirectory() as directory:
            service = IntrinsicCalibrationService(
                board_size=(2, 2),
                square=0.088,
                output_file=str(Path(directory) / "intrinsics.yaml"),
                camera_name="usb_cam",
                board_type="aprilgrid",
                tag_spacing=0.0264,
                min_tags=4,
            )
            corners = np.asarray(
                [[20, 20], [40, 20], [40, 40], [20, 40]],
                dtype=np.float32,
            ).reshape(-1, 1, 2)
            detection = intrinsic_solver.BoardDetection(
                image_points=corners,
                object_points=np.zeros((4, 3), np.float32),
                coverage=(0.2, 0.3, 0.4, 0.5),
                calibration_image_points=None,
                calibration_object_points=None,
            )
            with patch.object(intrinsic_solver, "detect_board", return_value=detection):
                service.process_frame(np.zeros((120, 160, 3), np.uint8))
            state = service.state()
            self.assertEqual(state["samples"], 0)
            self.assertEqual(state["detection"]["status"], "detected")
            self.assertFalse(state["detection"]["accepted"])
            self.assertEqual(service._evidence_samples, [])

    def test_physical_vga_search_collects_both_aprilgrid_profiles_from_source_jpeg(self):
        if not hasattr(cv2, "aruco") or not hasattr(cv2.aruco, "DICT_APRILTAG_36h11"):
            self.skipTest("OpenCV AprilTag 36h11 dictionary is unavailable")
        self.assertEqual(APRILGRID_ADAPTIVE_EVIDENCE_QUADS, 1)
        cases = (
            ("a4", 0.024, 0.0072, 0.80),
            ("field", 0.088, 0.0264, 2.50),
        )
        for name, tag_size, tag_gap, depth in cases:
            with self.subTest(profile=name, depth=depth):
                source_bgr, source_jpeg = render_production_aprilgrid(
                    tag_size, tag_gap, depth
                )
                working = MediaSnapshotClient._decode_detection_jpeg(
                    source_jpeg,
                    PRODUCTION_IMAGE_SIZE[0],
                    PRODUCTION_IMAGE_SIZE[1],
                    640 * 480,
                )
                self.assertLess(working.shape[1], 960)
                with tempfile.TemporaryDirectory() as directory:
                    service = IntrinsicCalibrationService(
                        board_size=(6, 6),
                        square=tag_size,
                        output_file=str(Path(directory) / "intrinsics.yaml"),
                        camera_name="usb_cam",
                        calibration_mode="phy",
                        board_type="aprilgrid",
                        tag_spacing=tag_gap,
                        min_tags=6,
                        display_width=640,
                    )
                    self.assertIsNone(service.camera)
                    service.process_frame(
                        working,
                        source_image_size=PRODUCTION_IMAGE_SIZE,
                        source_jpeg=source_jpeg,
                        source_snapshot_id="physical-{}".format(name),
                    )
                    state = service.state()
                    self.assertEqual(state["detection"]["status"], "detected")
                    self.assertTrue(state["detection"]["accepted"])
                    self.assertEqual(state["samples"], 1)
                    self.assertFalse(state["camera_control"])
                    self.assertEqual(len(service._evidence_samples), 1)
                    self.assertEqual(
                        service.image_size, PRODUCTION_IMAGE_SIZE
                    )
                    evidence = Path(service._evidence_root) / "source/000.jpg"
                    self.assertEqual(evidence.read_bytes(), source_jpeg)

    def test_physical_admission_maps_search_corners_instead_of_rerunning_source_aruco(self):
        if not hasattr(cv2, "aruco") or not hasattr(cv2.aruco, "DICT_APRILTAG_36h11"):
            self.skipTest("OpenCV AprilTag 36h11 dictionary is unavailable")
        source_bgr, source_jpeg = render_production_aprilgrid(0.024, 0.0072, 0.50)
        del source_bgr
        working = MediaSnapshotClient._decode_detection_jpeg(
            source_jpeg,
            PRODUCTION_IMAGE_SIZE[0],
            PRODUCTION_IMAGE_SIZE[1],
            640 * 480,
        )
        with tempfile.TemporaryDirectory() as directory:
            service = IntrinsicCalibrationService(
                board_size=(6, 6),
                square=0.024,
                output_file=str(Path(directory) / "intrinsics.yaml"),
                camera_name="usb_cam",
                calibration_mode="phy",
                board_type="aprilgrid",
                tag_spacing=0.0072,
                min_tags=6,
                display_width=640,
            )
            with patch.object(
                intrinsic_solver,
                "detect_board",
                wraps=intrinsic_solver.detect_board,
            ) as detect:
                service.process_frame(
                    working,
                    source_image_size=PRODUCTION_IMAGE_SIZE,
                    source_jpeg=source_jpeg,
                    source_snapshot_id="map-refine",
                )
            widths = [call.args[2] for call in detect.call_args_list]
            self.assertTrue(widths)
            self.assertNotIn(PRODUCTION_IMAGE_SIZE[0], widths)
            state = service.state()
            self.assertEqual(state["detection"]["status"], "detected")
            self.assertTrue(state["detection"]["accepted"])
            self.assertEqual(state["samples"], 1)

    def test_source_jpeg_pixels_win_when_snapshot_metadata_size_is_wrong(self):
        if not hasattr(cv2, "aruco") or not hasattr(cv2.aruco, "DICT_APRILTAG_36h11"):
            self.skipTest("OpenCV AprilTag 36h11 dictionary is unavailable")
        source_bgr, _ignored = render_production_aprilgrid(0.024, 0.0072, 0.50)
        small = cv2.resize(source_bgr, (1920, 1080), interpolation=cv2.INTER_AREA)
        ok, encoded = cv2.imencode(".jpg", small, (cv2.IMWRITE_JPEG_QUALITY, 94))
        self.assertTrue(ok)
        jpeg = encoded.tobytes()
        working = small
        with tempfile.TemporaryDirectory() as directory:
            service = IntrinsicCalibrationService(
                board_size=(6, 6),
                square=0.024,
                output_file=str(Path(directory) / "intrinsics.yaml"),
                camera_name="usb_cam",
                calibration_mode="phy",
                board_type="aprilgrid",
                tag_spacing=0.0072,
                min_tags=6,
                display_width=640,
            )
            service.process_frame(
                working,
                source_image_size=PRODUCTION_IMAGE_SIZE,
                source_jpeg=jpeg,
                source_snapshot_id="meta-mismatch",
            )
            state = service.state()
            self.assertEqual(state["detection"]["status"], "detected")
            self.assertTrue(state["detection"]["accepted"])
            self.assertEqual(state["samples"], 1)
            self.assertEqual(service.image_size, (1920, 1080))

    def test_simulation_pose_gate_still_blocks_untargeted_physical_style_frames(self):
        if not hasattr(cv2, "aruco") or not hasattr(cv2.aruco, "DICT_APRILTAG_36h11"):
            self.skipTest("OpenCV AprilTag 36h11 dictionary is unavailable")
        source_bgr, source_jpeg = render_production_aprilgrid(0.024, 0.0072, 0.80)
        del source_bgr
        working = MediaSnapshotClient._decode_detection_jpeg(
            source_jpeg,
            PRODUCTION_IMAGE_SIZE[0],
            PRODUCTION_IMAGE_SIZE[1],
            640 * 480,
        )
        with tempfile.TemporaryDirectory() as directory:
            service = IntrinsicCalibrationService(
                board_size=(6, 6),
                square=0.024,
                output_file=str(Path(directory) / "intrinsics.yaml"),
                camera_name="usb_cam",
                calibration_mode="sim",
                board_type="aprilgrid",
                tag_spacing=0.0072,
                min_tags=6,
                display_width=640,
            )
            service.attach_camera_control(FakeCameraControl())
            service.process_frame(
                working,
                source_image_size=PRODUCTION_IMAGE_SIZE,
                source_jpeg=source_jpeg,
                source_snapshot_id="sim-untargeted",
            )
            state = service.state()
            self.assertEqual(state["detection"]["status"], "detected")
            self.assertFalse(state["detection"]["accepted"])
            self.assertEqual(state["samples"], 0)
            self.assertTrue(state["camera_control"])
            self.assertEqual(service._evidence_samples, [])

    def test_source_jpeg_refinement_is_the_candidate_pool_authority(self):
        if not hasattr(cv2, "aruco") or not hasattr(cv2.aruco, "DICT_APRILTAG_36h11"):
            self.skipTest("OpenCV AprilTag 36h11 dictionary is unavailable")
        with tempfile.TemporaryDirectory() as directory:
            service = IntrinsicCalibrationService(
                board_size=APRILGRID_TEST_SIZE,
                square=APRILGRID_TEST_TAG_SIZE,
                output_file=str(Path(directory) / "intrinsics.yaml"),
                camera_name="usb_cam",
                board_type="aprilgrid",
                tag_spacing=APRILGRID_TEST_TAG_GAP,
                min_tags=6,
                display_width=640,
            )
            for center, depth, rotation in (
                ((640, 360), 1.50, (0.00, 0.00, 0.00)),
                ((300, 360), 1.50, (0.08, 0.00, 0.00)),
                ((640, 180), 1.05, (0.00, 0.08, 0.00)),
            ):
                service.process_frame(render_aprilgrid_view(center, depth, rotation))
            self.assertEqual(len(service.samples), 3)

            frontal_source = render_aprilgrid_view(
                (640, 360), 1.50, (0.00, 0.00, 0.00)
            )
            positive_x_source = render_aprilgrid_view(
                (640, 360), 1.50, (0.00, 0.35, 0.00)
            )
            working_positive_x = cv2.resize(
                positive_x_source,
                (640, 360),
                interpolation=cv2.INTER_AREA,
            )
            stale_ok, stale_jpeg = cv2.imencode(
                ".jpg", frontal_source, (cv2.IMWRITE_JPEG_QUALITY, 94)
            )
            self.assertTrue(stale_ok)
            baseline_lengths = (
                len(service.samples),
                len(service.image_points),
                len(service.object_points),
            )

            service.process_frame(
                working_positive_x,
                source_image_size=APRILGRID_TEST_IMAGE_SIZE,
                source_jpeg=stale_jpeg.tobytes(),
            )
            admitted = service.state()
            self.assertEqual(
                (
                    len(service.samples),
                    len(service.image_points),
                    len(service.object_points),
                ),
                tuple(value + 1 for value in baseline_lengths),
            )
            self.assertTrue(admitted["detection"]["accepted"])
            self.assertIsNone(admitted["recovery"]["last_error"])

            matching_ok, matching_jpeg = cv2.imencode(
                ".jpg", positive_x_source, (cv2.IMWRITE_JPEG_QUALITY, 94)
            )
            self.assertTrue(matching_ok)
            service.process_frame(
                working_positive_x,
                source_image_size=APRILGRID_TEST_IMAGE_SIZE,
                source_jpeg=matching_jpeg.tobytes(),
            )
            accepted = service.state()
            self.assertEqual(len(service.samples), baseline_lengths[0] + 2)
            self.assertEqual(len(service.image_points), baseline_lengths[1] + 2)
            self.assertEqual(len(service.object_points), baseline_lengths[2] + 2)
            self.assertTrue(accepted["detection"]["accepted"])
            self.assertIsNone(accepted["recovery"]["last_error"])

    def test_only_an_explicit_snapshot_identity_is_deduplicated(self):
        with tempfile.TemporaryDirectory() as directory:
            service = make_service(Path(directory) / "intrinsics.yaml")
            frame = render_board()
            service.process_frame(frame, source_snapshot_id="snapshot-1")
            service.process_frame(frame, source_snapshot_id="snapshot-1")
            state = service.state()
            self.assertEqual(state["samples"], 1)
            self.assertEqual(state["detection"]["status"], "detected")
            self.assertFalse(state["detection"]["accepted"])
            self.assertTrue(state["detection"]["duplicate"])

            service.process_frame(frame, source_snapshot_id="snapshot-2")
            service.process_frame(frame)
            self.assertEqual(service.state()["samples"], 3)

    def test_physical_candidate_pool_keeps_thirty_plus_strict_observations(self):
        if not hasattr(cv2, "aruco") or not hasattr(cv2.aruco, "DICT_APRILTAG_36h11"):
            self.skipTest("OpenCV AprilTag 36h11 dictionary is unavailable")
        with tempfile.TemporaryDirectory() as directory:
            service = IntrinsicCalibrationService(
                board_size=APRILGRID_TEST_SIZE,
                square=APRILGRID_TEST_TAG_SIZE,
                output_file=str(Path(directory) / "intrinsics.yaml"),
                camera_name="usb_cam", board_type="aprilgrid",
                tag_spacing=APRILGRID_TEST_TAG_GAP, min_tags=6,
                display_width=640,
            )
            frame = render_aprilgrid_view((640, 360), 1.50, (0.0, 0.2, 0.0))
            history = []
            for index in range(32):
                service.process_frame(frame, source_snapshot_id="snapshot-{}".format(index))
                history.append([
                    item["progress"] for item in service.state()["coverage"]
                ])

            self.assertEqual(service.state()["samples"], 32)
            self.assertEqual(service.state()["candidate_pool"]["count"], 32)
            self.assertTrue(all(
                current[axis] >= previous[axis]
                for previous, current in zip(history, history[1:])
                for axis in range(4)
            ))

    def test_non_board_frame_adds_no_sample(self):
        with tempfile.TemporaryDirectory() as directory:
            service = make_service(Path(directory) / "intrinsics.yaml")
            service.process_frame(np.full((200, 320, 3), 127, np.uint8))
            state = service.state()
            self.assertEqual(state["samples"], 0)
            self.assertEqual(state["detection"], {
                "status": "not_detected", "corner_count": 0, "expected_corner_count": 35,
                "frame_width": 320, "frame_height": 200, "sequence": 1, "metrics": [],
                "accepted": False, "duplicate": False,
            })

    def test_first_candidate_freezes_resolution_and_mismatches_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            service = make_service(Path(directory) / "intrinsics.yaml")
            service.process_frame(np.full((720, 1280, 3), 127, np.uint8))
            self.assertIsNone(service.image_size)

            frame = render_board()
            service.process_frame(frame, source_snapshot_id="first")
            frozen = (frame.shape[1], frame.shape[0])
            self.assertEqual(service.image_size, frozen)
            self.assertEqual(service.state()["samples"], 1)

            larger = cv2.resize(frame, (frame.shape[1] * 2, frame.shape[0] * 2))
            service.process_frame(larger, source_snapshot_id="wrong-size")
            state = service.state()
            self.assertEqual(service.image_size, frozen)
            self.assertEqual(state["samples"], 1)
            self.assertFalse(state["detection"]["accepted"])
            self.assertIn("do not match the frozen", state["recovery"]["last_error"])

            service.reset()
            self.assertIsNone(service.image_size)

    def test_rank_deficient_candidate_is_structured_and_never_persisted(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory) / "intrinsics.yaml"
            service = make_service(base)
            frame = render_board()
            for index in range(3):
                service.process_frame(
                    frame, source_snapshot_id="weak-{}".format(index)
                )
            checkpoint = Path(service.checkpoint_file)
            self.assertTrue(checkpoint.is_file())
            weak = make_diagnostic_result(
                (frame.shape[1], frame.shape[0]), 3, rank_deficient=True
            )
            with patch.object(intrinsic_solver, "calibrate_intrinsic", return_value=weak):
                with self.assertRaises(ApiError) as caught:
                    service.calibrate()
            self.assertEqual(caught.exception.status, int(HTTPStatus.UNPROCESSABLE_ENTITY))
            self.assertEqual(
                caught.exception.details["code"], "intrinsic_candidate_unstable"
            )
            self.assertIn(
                "projected_intrinsic_rank_deficient",
                caught.exception.details["reasons"],
            )
            self.assertEqual(
                caught.exception.details["save_blocked"],
                "stability_validation_failed",
            )
            self.assertTrue(checkpoint.is_file())
            self.assertFalse(base.exists())
            self.assertEqual(list(Path(directory).glob("intrinsics-*.yaml")), [])

    def test_normalized_ray_instability_blocks_explicit_save(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory) / "intrinsics.yaml"
            service = make_service(base)
            frame = render_board()
            for index in range(3):
                service.process_frame(frame, source_snapshot_id="ray-{}".format(index))
            unstable = make_diagnostic_result(
                (frame.shape[1], frame.shape[0]), 3, ray_max=214.0
            )
            with patch.object(
                intrinsic_solver, "calibrate_intrinsic", return_value=unstable
            ):
                candidate = service.calibrate()
            self.assertEqual(candidate["quality"]["status"], "unstable")
            self.assertIn(
                "normalized_ray_stability_exceeds_confidence_envelope",
                candidate["quality"]["reasons"],
            )
            with self.assertRaises(ApiError) as caught:
                service.save(candidate["candidate_id"])
            self.assertEqual(caught.exception.status, int(HTTPStatus.UNPROCESSABLE_ENTITY))
            self.assertIn(
                "normalized_ray_stability_exceeds_confidence_envelope",
                caught.exception.details["reasons"],
            )
            self.assertGreater(
                caught.exception.details["assessment"][
                    "undistorted_ray_max_equivalent_px"
                ],
                caught.exception.details["assessment"][
                    "normalized_ray_confidence_limit_px"
                ],
            )
            self.assertFalse(base.exists())
            self.assertEqual(service.state()["phase"], "candidate_ready")

    def test_calibrate_without_samples_conflicts(self):
        with tempfile.TemporaryDirectory() as directory:
            service = make_service(Path(directory) / "intrinsics.yaml")
            with self.assertRaises(ApiError) as caught:
                service.calibrate()
            self.assertEqual(caught.exception.status, int(HTTPStatus.CONFLICT))

    def test_reset_clears_samples(self):
        with tempfile.TemporaryDirectory() as directory:
            service = make_service(Path(directory) / "intrinsics.yaml")
            service.process_frame(render_board())
            self.assertEqual(service.reset()["samples"], 0)

    def test_guide_targets_and_agnostic_state(self):
        with tempfile.TemporaryDirectory() as directory:
            service = make_service(Path(directory) / "intrinsics.yaml")
            document = service.targets_document()
            self.assertIn("center", document["board"])
            self.assertEqual(len(document["views"]), 15)
            self.assertFalse(document["camera_control"])
            state = service.state()
            self.assertEqual(len(state["targets"]), 15)
            self.assertIsNone(state["pose"])
            self.assertFalse(state["camera_control"])
            self.assertEqual(state["auto_capture"], {
                "enabled": False,
                "interval_seconds": 0.0,
                "last_error": None,
                "coverage_complete": False,
            })
            self.assertIsNone(state["action"])
            self.assertEqual(state["next"], 0)
            self.assertFalse(state["targets"][0]["done"])
            self.assertEqual(state["detection"]["status"], "waiting")
            self.assertEqual(state["guidance"]["direction"], "center")
            self.assertFalse(state["recovery"]["checkpoint_available"])

    def test_guidance_uses_the_final_coverage_bars_returned_by_state(self):
        with tempfile.TemporaryDirectory() as directory:
            service = make_service(Path(directory) / "intrinsics.yaml")
            service.samples = [
                (0.05, 0.05, 0.40, 0.02),
                (0.622, 0.75, 0.40, 0.08),
            ]
            final_bars = [
                {"label": "X", "progress": 0.817},
                {"label": "Y", "progress": 1.0},
                {"label": "Size", "progress": 1.0},
                {"label": "Skew", "progress": 1.0},
            ]
            with patch.object(
                service,
                "_coverage_state_locked",
                return_value=(final_bars, False),
            ):
                state = service.state()
            self.assertEqual(state["coverage"], final_bars)
            self.assertEqual(state["guidance"], {
                "complete": False,
                "dimension": "X",
                "direction": "right",
                "progress": 0.817,
            })

    def test_physical_guidance_uses_monotonic_image_plane_coverage(self):
        with tempfile.TemporaryDirectory() as directory:
            service = IntrinsicCalibrationService(
                board_size=(2, 2),
                square=0.088,
                output_file=str(Path(directory) / "intrinsics.yaml"),
                camera_name="usb_cam",
                board_type="aprilgrid",
                tag_spacing=0.0264,
                min_tags=4,
                calibration_mode="phy",
            )
            service.samples = [
                (0.06, 0.31, 0.24, 0.01),
                (0.86, 0.96, 0.42, 0.29),
            ]
            final_bars = [
                {"label": "X", "progress": 1.0},
                {"label": "Y", "progress": 0.92},
                {"label": "Size", "progress": 0.97},
                {"label": "Skew", "progress": 0.5},
            ]
            with patch.object(
                service,
                "_coverage_state_locked",
                return_value=(final_bars, False),
            ):
                state = service.state()
            self.assertEqual(state["coverage"], final_bars)
            self.assertEqual(state["guidance"], {
                "complete": False,
                "dimension": "Skew",
                "direction": "tilt",
                "progress": 0.5,
            })

    def test_l1_proximity_never_rejects_candidate_observations(self):
        with tempfile.TemporaryDirectory() as directory:
            service = make_service(Path(directory) / "intrinsics.yaml")
            frame = render_board()
            service.process_frame(frame)
            service.process_frame(frame)
            service.process_frame(frame)
            self.assertEqual(service.state()["samples"], 3)
            self.assertTrue(service.state()["detection"]["accepted"])
            self.assertFalse(service.state()["detection"]["duplicate"])

    def test_accepted_corner_checkpoint_restores_an_interrupted_stage(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "intrinsics.yaml"
            service = make_service(output)
            service.process_frame(render_board())
            checkpoint = Path(str(output) + ".session.npz")
            self.assertTrue(checkpoint.is_file())

            restored = make_service(output)
            state = restored.state()
            self.assertEqual(state["samples"], 1)
            self.assertEqual(len(restored.image_points), 1)
            self.assertEqual(len(restored.object_points), 1)
            self.assertEqual(restored.sample_target_ids, [None])
            self.assertEqual(restored.sample_snapshot_ids, [""])
            self.assertTrue(state["recovery"]["checkpoint_available"])
            self.assertNotIn("result_restored", state)

    def test_checkpoint_restores_exact_simulation_target_identity(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "intrinsics.yaml"
            service = make_service(output)
            camera = FakeCameraControl()
            service.attach_camera_control(camera)
            service.goto(3)
            service.process_frame(
                render_board(),
                render_position=tuple(service.views[3]["position"]),
                render_orientation=camera.current_optical_pose()["orientation"],
            )
            self.assertEqual(service.sample_target_ids, [3])

            restored = make_service(output)
            self.assertEqual(restored.sample_target_ids, [3])
            self.assertTrue(restored.target_done[3])
            self.assertEqual(sum(restored.target_done), 1)

    def test_center_feature_checkpoint_is_not_restored_as_full_corner_data(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "intrinsics.yaml"
            service = IntrinsicCalibrationService(
                board_size=(2, 2),
                square=0.088,
                output_file=str(output),
                camera_name="usb_cam",
                board_type="aprilgrid",
                tag_spacing=0.0264,
                min_tags=4,
            )
            old_fingerprint = service._recovery_fingerprint()
            old_fingerprint.pop("feature_model")
            old_fingerprint["schema"] = 1
            checkpoint = Path(service.checkpoint_file)
            np.savez_compressed(
                str(checkpoint),
                fingerprint=np.asarray(json.dumps(old_fingerprint, sort_keys=True)),
                samples=np.asarray([[0.2, 0.3, 0.4, 0.5]], dtype=np.float64),
                image_size=np.asarray([640, 480], dtype=np.int64),
                image_points_000=np.zeros((4, 1, 2), dtype=np.float32),
                object_points_000=np.zeros((4, 3), dtype=np.float32),
            )

            restored = IntrinsicCalibrationService(
                board_size=(2, 2),
                square=0.088,
                output_file=str(output),
                camera_name="usb_cam",
                board_type="aprilgrid",
                tag_spacing=0.0264,
                min_tags=4,
            )
            state = restored.state()
            self.assertEqual(state["samples"], 0)
            self.assertIn("incompatible observation feature model", state["recovery"]["last_error"])
            self.assertTrue(state["recovery"]["checkpoint_available"])

    def test_legacy_result_without_quality_contract_is_not_restored(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory) / "intrinsics.yaml"
            output = Path(directory) / "intrinsics-20260830T120000.000000Z.yaml"
            result = intrinsic_solver.IntrinsicResult(
                camera_matrix=np.array([[638.0, 0.0, 600.0], [0.0, 637.0, 390.0], [0.0, 0.0, 1.0]]),
                distortion=np.array([0.01, -0.02, 0.0, 0.0, 0.0]),
                image_size=(1280, 720), rms_reprojection_error_px=0.9, sample_count=40,
            )
            intrinsic_solver.save_intrinsic(
                output,result,camera_name="usb_cam",board_size=(7, 5),square=0.20,
            )

            restored = make_service(base)
            restored.attach_frame_capture(lambda: render_board())
            state = restored.state()
            self.assertEqual(state["phase"], "collecting")
            self.assertEqual(state["samples"], 0)
            self.assertNotIn("result", state)
            self.assertNotIn("output_file", state)
            history = restored.calibration_history()
            self.assertIsNone(history["selected"])
            self.assertFalse(history["items"][0]["validated"])
            restored_capture = restored.start_auto_capture(interval=0.1)["auto_capture"]
            self.assertTrue(restored_capture["enabled"])

            restored.reset()
            self.assertTrue(output.exists())
            self.assertEqual(restored.state()["phase"], "collecting")
            self.assertEqual(restored.state()["samples"], 0)
            self.assertTrue(restored.state()["auto_capture"]["enabled"])
            restored.stop_auto_capture()

    def test_latest_timestamped_result_restores_by_created_time(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory) / "intrinsics.yaml"
            older = Path(directory) / "intrinsics-20260829T120000.000000Z.yaml"
            newer = Path(directory) / "intrinsics-20260830T120000.000000Z.yaml"
            older_result = intrinsic_solver.IntrinsicResult(
                camera_matrix=np.array([[638.0, 0.0, 600.0], [0.0, 637.0, 390.0], [0.0, 0.0, 1.0]]),
                distortion=np.array([0.01, -0.02, 0.0, 0.0, 0.0]),
                image_size=(1280, 720), rms_reprojection_error_px=0.9, sample_count=40,
            )
            newer_result = intrinsic_solver.IntrinsicResult(
                camera_matrix=np.array([[742.0, 0.0, 601.0], [0.0, 741.0, 391.0], [0.0, 0.0, 1.0]]),
                distortion=np.array([0.02, -0.03, 0.0, 0.0, 0.0]),
                image_size=(1280, 720), rms_reprojection_error_px=0.6, sample_count=45,
            )
            intrinsic_solver.save_intrinsic(
                older,older_result,camera_name="usb_cam",board_size=(7, 5),square=0.20,
                metadata={
                    "quality_contract": "xgc2.camera.intrinsic-quality.v2",
                    "candidate_id": "intrinsic-candidate-older",
                    "stability_assessment": {"passed": True},
                    "coverage": [
                        {"label": "X", "progress": 0.8},
                        {"label": "Y", "progress": 0.9},
                        {"label": "Size", "progress": 0.7},
                        {"label": "Skew", "progress": 0.6},
                    ],
                },
            )
            intrinsic_solver.save_intrinsic(
                newer,newer_result,camera_name="usb_cam",board_size=(7, 5),square=0.20,
                metadata={
                    "quality_contract": "xgc2.camera.intrinsic-quality.v2",
                    "candidate_id": "intrinsic-candidate-newer",
                    "stability_assessment": {"passed": True},
                    "coverage": [
                        {"label": "X", "progress": 0.8},
                        {"label": "Y", "progress": 0.9},
                        {"label": "Size", "progress": 0.7},
                        {"label": "Skew", "progress": 0.6},
                    ],
                },
            )

            restored = make_service(base)
            state = restored.state()
            self.assertEqual(state["phase"], "saved")
            self.assertTrue(state["result_restored"])
            self.assertEqual(state["output_file"], str(newer))
            self.assertEqual(state["result"]["output_file"], str(newer))
            self.assertAlmostEqual(state["result"]["fx"], 742.0)
            self.assertEqual(state["saved_candidate_id"], "intrinsic-candidate-newer")
            self.assertEqual(state["coverage"][0]["progress"], 0.8)
            self.assertFalse(state["guidance"]["complete"])

    def test_in_progress_checkpoint_takes_precedence_over_an_older_saved_result(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory) / "intrinsics.yaml"
            stage = make_service(base)
            stage.samples = [(0.2, 0.3, 0.25, 0.1)]
            stage.image_points = [np.zeros((35, 1, 2), dtype=np.float32)]
            stage.object_points = [np.zeros((35, 3), dtype=np.float32)]
            stage.sample_target_ids = [None]
            stage.sample_snapshot_ids = [""]
            stage.image_size = (1280, 720)
            stage.collection_revision = 1
            with stage.lock:
                stage._save_checkpoint_locked()

            saved = Path(directory) / "intrinsics-20260829T120000.000000Z.yaml"
            saved_result = intrinsic_solver.IntrinsicResult(
                camera_matrix=np.array(
                    [[638.0, 0.0, 600.0], [0.0, 637.0, 390.0], [0.0, 0.0, 1.0]]
                ),
                distortion=np.array([0.01, -0.02, 0.0, 0.0, 0.0]),
                image_size=(1280, 720),
                rms_reprojection_error_px=0.9,
                sample_count=40,
            )
            intrinsic_solver.save_intrinsic(
                saved,
                saved_result,
                camera_name="usb_cam",
                board_size=(7, 5),
                square=0.20,
            )

            restored = make_service(base)
            state = restored.state()
            self.assertEqual(state["phase"], "collecting")
            self.assertNotIn("result", state)
            self.assertEqual(state["samples"], 1)
            self.assertEqual(restored.samples, stage.samples)
            self.assertNotIn("output_file", state)

    def test_calibrate_produces_immutable_candidate_without_writing_an_asset(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory) / "intrinsics.yaml"
            service = make_service(base)
            service.image_points = [np.zeros((35, 1, 2), dtype=np.float32)]
            service.object_points = [np.zeros((35, 3), dtype=np.float32)]
            service.samples = [(0.2, 0.3, 0.4, 0.1)]
            service.sample_target_ids = [None]
            service.sample_snapshot_ids = [""]
            service.image_size = (1280, 720)
            service.collection_revision = 1
            result = make_diagnostic_result((1280, 720), 1)
            with patch.object(intrinsic_solver, "calibrate_intrinsic", return_value=result):
                candidate = service.calibrate()

            self.assertFalse(candidate["saved"])
            self.assertIsNone(candidate["output_file"])
            self.assertEqual(
                candidate["save_blocked"], "explicit_save_required"
            )
            self.assertEqual(candidate["quality"]["status"], "save_ready")
            self.assertTrue(candidate["quality"]["assessment"]["passed"])
            self.assertEqual(
                candidate["diagnostics"]["stability"]["folds"][0][
                    "held_out_rms_reprojection_error_px"
                ],
                0.405,
            )
            self.assertFalse(base.exists())
            self.assertEqual(list(Path(directory).glob("intrinsics-*.yaml")), [])
            self.assertEqual(service.state()["phase"], "candidate_ready")
            self.assertNotIn("result", service.state())
            self.assertNotIn("output_file", service.state())
            self.assertTrue(service.state()["candidate_pool"]["solve_frozen"])
            self.assertEqual(service.calibrate(), candidate)
            service.process_frame(render_board(), source_snapshot_id="after-solve")
            self.assertEqual(service.state()["samples"], 1)
            self.assertEqual(service.state()["candidate"], candidate)

            reset = service.reset()
            self.assertEqual(reset["samples"], 0)
            self.assertEqual(reset["phase"], "collecting")
            self.assertNotIn("candidate", reset)
            self.assertIsNone(service.image_size)
            self.assertNotIn("output_file", reset)

    def test_continue_retains_pool_and_new_observation_changes_candidate_cas(self):
        with tempfile.TemporaryDirectory() as directory:
            service = make_service(Path(directory) / "intrinsics.yaml")
            frame = render_board()
            service.process_frame(frame, source_snapshot_id="one")
            first_result = make_diagnostic_result(
                (frame.shape[1], frame.shape[0]), 1
            )
            with patch.object(
                intrinsic_solver, "calibrate_intrinsic", return_value=first_result
            ):
                first = service.calibrate()
            first_session_revision = service.state()["session_revision"]
            with self.assertRaises(ApiError) as stale:
                service.save("intrinsic-candidate-stale")
            self.assertEqual(stale.exception.status, int(HTTPStatus.CONFLICT))

            continued = service.continue_collection()
            self.assertEqual(continued["phase"], "collecting")
            self.assertEqual(continued["samples"], 1)
            self.assertNotIn("candidate", continued)
            self.assertGreater(continued["session_revision"], first_session_revision)
            with patch.object(
                intrinsic_solver, "calibrate_intrinsic", return_value=first_result
            ):
                replay = service.calibrate()
            self.assertNotEqual(replay["candidate_id"], first["candidate_id"])
            service.continue_collection()
            service.process_frame(frame, source_snapshot_id="two")
            self.assertEqual(service.state()["collection_revision"], 2)

            second_result = make_diagnostic_result(
                (frame.shape[1], frame.shape[0]), 2
            )
            with patch.object(
                intrinsic_solver, "calibrate_intrinsic", return_value=second_result
            ):
                second = service.calibrate()
            self.assertNotEqual(first["candidate_id"], second["candidate_id"])
            self.assertEqual(second["collection_revision"], 2)

            service.reset()
            service.process_frame(frame, source_snapshot_id="one")
            with patch.object(
                intrinsic_solver, "calibrate_intrinsic", return_value=first_result
            ):
                after_reset = service.calibrate()
            self.assertNotEqual(first["candidate_id"], after_reset["candidate_id"])
            with self.assertRaises(ApiError) as stale_after_reset:
                service.save(first["candidate_id"])
            self.assertEqual(
                stale_after_reset.exception.status, int(HTTPStatus.CONFLICT)
            )

    def test_evidence_download_contains_exact_sources_annotations_manifest_and_yaml(self):
        with tempfile.TemporaryDirectory() as directory:
            service = make_service(
                Path(directory) / "intrinsics.yaml",
                calibration_mode="phy",
                board_profile_id="field_6x6_88mm_30pct",
            )
            frame = render_board()
            encoded, jpeg = cv2.imencode(
                ".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 96]
            )
            self.assertTrue(encoded)
            source_jpeg = jpeg.tobytes()
            service.process_frame(
                frame,
                source_image_size=(frame.shape[1], frame.shape[0]),
                source_jpeg=source_jpeg,
                source_snapshot_id="snapshot-1",
                source_frame_id="usb_cam",
                source_timestamp_nanoseconds=123456789,
            )
            self.assertEqual(service.state()["samples"], 1)
            self.assertFalse(service.state()["evidence"]["available"])

            result = make_diagnostic_result(
                (frame.shape[1], frame.shape[0]), 1
            )
            with patch.object(intrinsic_solver, "calibrate_intrinsic", return_value=result):
                candidate = service.calibrate()
            self.assertEqual(
                candidate["save_blocked"], "explicit_save_required"
            )
            evidence = service.state()["evidence"]
            self.assertTrue(evidence["available"])
            self.assertEqual(evidence["sample_count"], 1)
            self.assertEqual(
                evidence["filename"], candidate["candidate_id"] + "-evidence.zip"
            )

            server = CalibrationHttpServer(
                ("127.0.0.1", 0), object(), WEB_ROOT,
                frame_ancestors="'self'", intrinsic_service=service,
            )
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            try:
                url = "http://127.0.0.1:{}/api/v1/intrinsic/evidence.zip".format(
                    server.server_address[1]
                )
                with urllib.request.urlopen(url) as response:
                    self.assertEqual(response.headers.get_content_type(), "application/zip")
                    self.assertEqual(
                        response.headers["Content-Disposition"],
                        'attachment; filename="{}"'.format(evidence["filename"]),
                    )
                    archive_payload = response.read()
            finally:
                server.shutdown()
                server.server_close()

            with zipfile.ZipFile(io.BytesIO(archive_payload)) as archive:
                self.assertEqual(
                    sorted(archive.namelist()),
                    ["annotated/000.jpg", "manifest.json", "source/000.jpg"],
                )
                self.assertEqual(archive.read("source/000.jpg"), source_jpeg)
                annotated = cv2.imdecode(
                    np.frombuffer(archive.read("annotated/000.jpg"), dtype=np.uint8),
                    cv2.IMREAD_COLOR,
                )
                self.assertEqual(annotated.shape, frame.shape)
                manifest = json.loads(archive.read("manifest.json"))
            self.assertEqual(manifest["schema"], "xgc2.camera.intrinsic-evidence.v2")
            self.assertIsNone(manifest["result"])
            self.assertEqual(
                manifest["candidate"]["candidate_id"], candidate["candidate_id"]
            )
            self.assertNotIn(
                "held_out_point_errors_px",
                service.state()["candidate"]["diagnostics"]["stability"]["folds"][0],
            )
            self.assertIn(
                "held_out_point_errors_px",
                manifest["solver_diagnostics"]["stability"]["folds"][0],
            )
            self.assertEqual(manifest["mode"], "phy")
            self.assertEqual(manifest["camera_name"], "usb_cam")
            self.assertEqual(manifest["board_profile"], "field_6x6_88mm_30pct")
            self.assertEqual(manifest["algorithm"]["contract"], "xgc2.camera.intrinsic-algorithm.v1")
            self.assertRegex(manifest["algorithm"]["sha256"], r"^[0-9a-f]{64}$")
            self.assertEqual(manifest["samples"][0]["snapshot_id"], "snapshot-1")
            self.assertEqual(manifest["samples"][0]["timestamp_nanoseconds"], 123456789)
            self.assertEqual(
                manifest["samples"][0]["source_sha256"],
                hashlib.sha256(source_jpeg).hexdigest(),
            )

            saved = service.save(candidate["candidate_id"])
            self.assertTrue(saved["saved"])
            self.assertEqual(service.state()["phase"], "saved")
            saved_filename, saved_bundle = service.evidence_bundle()
            self.assertEqual(saved_filename, evidence["filename"])
            with zipfile.ZipFile(saved_bundle) as archive:
                self.assertIn("intrinsics.yaml", archive.namelist())
                saved_manifest = json.loads(archive.read("manifest.json"))
            self.assertEqual(
                saved_manifest["result"]["original_filename"],
                Path(saved["output_file"]).name,
            )
            saved_document = intrinsic_solver.load_intrinsic(saved["output_file"])
            self.assertEqual(
                saved_document["metadata"]["quality_contract"],
                "xgc2.camera.intrinsic-quality.v2",
            )
            self.assertEqual(
                saved_document["metadata"]["board_profile"],
                "field_6x6_88mm_30pct",
            )
            self.assertEqual(
                saved_document["metadata"]["algorithm"]["sha256"],
                manifest["algorithm"]["sha256"],
            )
            self.assertEqual(
                saved_document["metadata"]["candidate_id"],
                candidate["candidate_id"],
            )
            self.assertTrue(
                saved_document["metadata"]["stability_assessment"]["passed"]
            )
            self.assertEqual(service.save(candidate["candidate_id"]), saved)

            old_evidence_root = service._evidence_root
            service.reset()
            self.assertFalse(old_evidence_root.exists())
            with self.assertRaisesRegex(ApiError, "evidence is unavailable"):
                service.evidence_bundle()

    def test_source_sample_is_not_admitted_when_evidence_cannot_be_recorded(self):
        with tempfile.TemporaryDirectory() as directory:
            service = make_service(Path(directory) / "intrinsics.yaml")
            frame = render_board()
            encoded, jpeg = cv2.imencode(".jpg", frame)
            self.assertTrue(encoded)
            with patch.object(
                service,
                "_record_evidence_sample_locked",
                side_effect=OSError("evidence disk unavailable"),
            ):
                service.process_frame(
                    frame,
                    source_image_size=(frame.shape[1], frame.shape[0]),
                    source_jpeg=jpeg.tobytes(),
                )
            state = service.state()
            self.assertEqual(state["samples"], 0)
            self.assertFalse(state["detection"]["accepted"])
            self.assertEqual(state["evidence"]["sample_count"], 0)
            self.assertEqual(state["recovery"]["last_error"], "evidence disk unavailable")

    def test_continuous_detection_keeps_running_after_coverage_is_ready(self):
        with tempfile.TemporaryDirectory() as directory:
            service = make_service(Path(directory) / "intrinsics.yaml")
            service.attach_frame_capture(lambda: render_board())
            bars = [
                {"label": label, "progress": 1.0}
                for label in ("X", "Y", "Size", "Skew")
            ]
            with patch(
                "xgc_camera_calibration.intrinsic_service.intrinsic_solver.coverage",
                return_value=(bars, True),
            ):
                started = service.start_auto_capture(interval=0.1)
                self.assertTrue(started["auto_capture"]["enabled"])
                deadline = monotonic() + 2.0
                while not service.state()["auto_capture"]["coverage_complete"] and monotonic() < deadline:
                    sleep(0.01)
            state = service.state()
            self.assertEqual(state["samples"], 1)
            self.assertTrue(state["auto_capture"]["coverage_complete"])
            first_sequence = state["detection"]["sequence"]
            deadline = monotonic() + 1.0
            while service.state()["detection"]["sequence"] <= first_sequence and monotonic() < deadline:
                sleep(0.01)
            self.assertGreater(service.state()["detection"]["sequence"], first_sequence)
            self.assertTrue(service.state()["auto_capture"]["enabled"])
            service.stop_auto_capture()

    def test_advisory_coverage_completion_does_not_close_the_candidate_pool(self):
        with tempfile.TemporaryDirectory() as directory:
            service = make_service(Path(directory) / "intrinsics.yaml")
            bars = [
                {"label": label, "progress": 1.0}
                for label in ("X", "Y", "Size", "Skew")
            ]
            with patch.object(
                intrinsic_solver, "coverage", return_value=(bars, True)
            ):
                service.process_frame(render_board(), source_snapshot_id="complete-1")
                self.assertTrue(service._coverage_state_locked()[1])
                service.process_frame(render_board(), source_snapshot_id="complete-2")
            self.assertEqual(service.state()["samples"], 2)

    def test_simulation_and_physical_share_continuous_detection(self):
        with tempfile.TemporaryDirectory() as directory:
            service = make_service(Path(directory) / "intrinsics.yaml")
            service.attach_camera_control(FakeCameraControl())
            service.attach_frame_capture(lambda: render_board())
            service.start_auto_capture(interval=0.1)
            deadline = monotonic() + 1.0
            while service.state()["detection"]["sequence"] < 3 and monotonic() < deadline:
                sleep(0.01)
            self.assertGreaterEqual(service.state()["detection"]["sequence"], 3)
            self.assertTrue(service.state()["auto_capture"]["enabled"])
            service.stop_auto_capture()

    def test_continuous_detector_uses_fixed_start_cadence_without_backlog(self):
        with tempfile.TemporaryDirectory() as directory:
            service = make_service(Path(directory) / "intrinsics.yaml")
            starts = []

            def capture():
                starts.append(monotonic())
                sleep(0.04)
                return np.full((200, 320, 3), 127, np.uint8)

            service.attach_frame_capture(capture)
            service.start_auto_capture(interval=0.1)
            deadline = monotonic() + 1.5
            while len(starts) < 5 and monotonic() < deadline:
                sleep(0.01)
            service.stop_auto_capture()
            self.assertGreaterEqual(len(starts), 5)
            periods = [right - left for left, right in zip(starts, starts[1:])]
            self.assertLess(max(periods[:4]), 0.13)

    def test_continuous_detector_has_no_artificial_success_delay(self):
        with tempfile.TemporaryDirectory() as directory:
            service = make_service(Path(directory) / "intrinsics.yaml")
            starts = []

            def capture():
                starts.append(monotonic())
                sleep(0.02)
                return np.full((200, 320, 3), 127, np.uint8)

            service.attach_frame_capture(capture)
            with patch.object(service, "process_frame", return_value=None):
                service.start_auto_capture()
                deadline = monotonic() + 1.0
                while len(starts) < 5 and monotonic() < deadline:
                    sleep(0.01)
                service.stop_auto_capture()
            self.assertGreaterEqual(len(starts), 5)
            periods = [right - left for left, right in zip(starts, starts[1:])]
            self.assertLess(max(periods[:4]), 0.05)

    def test_physical_auto_capture_does_not_retain_invalid_frames(self):
        with tempfile.TemporaryDirectory() as directory:
            service = make_service(Path(directory) / "intrinsics.yaml")
            service.attach_frame_capture(
                lambda: np.full((200, 320, 3), 127, np.uint8)
            )
            service.start_auto_capture(interval=0.1)
            deadline = monotonic() + 1.0
            while service.state()["detection"]["sequence"] < 3 and monotonic() < deadline:
                sleep(0.01)
            stopped = service.stop_auto_capture()
            self.assertFalse(stopped["auto_capture"]["enabled"])
            self.assertEqual(service.state()["samples"], 0)
            self.assertEqual(service.image_points, [])
            self.assertEqual(service.object_points, [])
            self.assertFalse((Path(directory) / "intrinsics.yaml").exists())
            self.assertFalse((Path(directory) / "intrinsics.yaml.session.npz").exists())

    def test_camera_actions_require_control(self):
        with tempfile.TemporaryDirectory() as directory:
            service = make_service(Path(directory) / "intrinsics.yaml")
            for action in (lambda: service.goto(0), service.reset_pose, service.auto_run):
                with self.assertRaises(ApiError) as caught:
                    action()
                self.assertEqual(caught.exception.status, int(HTTPStatus.NOT_FOUND))

    def test_auto_run_is_nonblocking_and_serializes_mutating_actions(self):
        with tempfile.TemporaryDirectory() as directory:
            service = make_service(Path(directory) / "intrinsics.yaml")
            camera = FakeCameraControl()
            service.attach_camera_control(camera)

            def capture_at_current_pose():
                position = camera.current_position()
                return SimpleNamespace(
                    bgr=render_board(),
                    render_position=tuple(position) if position is not None else None,
                    render_orientation=(0.0, 0.0, 0.0, 1.0),
                )

            service.attach_frame_capture(capture_at_current_pose)
            service.start_auto_capture(interval=0.1)
            self.addCleanup(service.stop_auto_capture)

            with patch.object(
                service, "_calibrate_locked", return_value={"output_file": str(Path(directory) / "intrinsics.yaml")}
            ):
                started = monotonic()
                accepted = service.auto_run(settle=0.03)
                self.assertLess(monotonic() - started, 0.1)
                self.assertTrue(accepted["accepted"])
                self.assertEqual(accepted["action"]["status"], "running")
                for mutation in (
                    service.reset,
                    service.reset_pose,
                    service.calibrate,
                    lambda: service.goto(0),
                    lambda: service.auto_run(settle=0.03),
                ):
                    with self.assertRaises(ApiError) as caught:
                        mutation()
                    self.assertEqual(caught.exception.status, int(HTTPStatus.CONFLICT))
                    self.assertIn("already running", caught.exception.message)

                auto_thread = service._auto_run_thread
                self.assertIsNotNone(auto_thread)
                auto_thread.join(timeout=len(service.views) * 0.3 + 2.0)
                self.assertFalse(auto_thread.is_alive())
            self.assertEqual(service.state()["action"]["status"], "succeeded")
            self.assertEqual(len(camera.positions), 15)
            self.assertEqual(len(service.samples), 15)
            self.assertEqual(service.sample_target_ids, list(range(15)))
            self.assertTrue(all(target["done"] for target in service.state()["targets"]))
            service.stop_auto_capture()
            self.assertEqual(service.reset()["samples"], 0)
            self.assertIsNone(service.state()["action"])

    def test_auto_run_fails_when_a_guide_target_never_detects_the_board(self):
        with tempfile.TemporaryDirectory() as directory:
            service = make_service(Path(directory) / "intrinsics.yaml")
            service.attach_camera_control(FakeCameraControl())
            service.attach_frame_capture(lambda: np.full((200, 320, 3), 127, np.uint8))
            service.start_auto_capture(interval=0.1)
            with patch("xgc_camera_calibration.intrinsic_service.time.sleep", return_value=None):
                service.auto_run(settle=0, detection_timeout=0.2)
                deadline = monotonic() + 1.0
                while service.state()["action"]["status"] == "running" and monotonic() < deadline:
                    sleep(0.01)
            action = service.state()["action"]
            self.assertEqual(action["status"], "failed")
            self.assertIn("left edge", action["error"])
            self.assertIn("continuous detection could not find", action["error"])
            service.stop_auto_capture()

    def test_auto_run_camera_failure_is_reported_and_recoverable(self):
        class FailingCameraControl(FakeCameraControl):
            def goto(self, position, yaw_offset, pitch_offset, roll):
                raise RuntimeError("Gazebo camera rejected the pose")

        with tempfile.TemporaryDirectory() as directory:
            service = make_service(Path(directory) / "intrinsics.yaml")
            service.attach_camera_control(FailingCameraControl())
            service.attach_frame_capture(lambda: render_board())
            service.start_auto_capture(interval=0.1)
            service.auto_run(settle=0)
            deadline = monotonic() + 1.0
            while service.state()["action"]["status"] == "running" and monotonic() < deadline:
                sleep(0.01)
            action = service.state()["action"]
            self.assertEqual(action["status"], "failed")
            self.assertIn("rejected the pose", action["error"])
            service.stop_auto_capture()
            self.assertEqual(service.reset()["samples"], 0)
            self.assertIsNone(service.state()["action"])

    def test_auto_run_thread_start_failure_does_not_leave_permanent_busy_state(self):
        with tempfile.TemporaryDirectory() as directory:
            service = make_service(Path(directory) / "intrinsics.yaml")
            service.attach_camera_control(FakeCameraControl())
            service.attach_frame_capture(lambda: render_board())
            service.start_auto_capture(interval=0.1)
            with patch("xgc_camera_calibration.intrinsic_service.threading.Thread") as constructor:
                constructor.return_value.start.side_effect = RuntimeError("thread unavailable")
                with self.assertRaises(ApiError) as caught:
                    service.auto_run()
            self.assertEqual(caught.exception.status, int(HTTPStatus.INTERNAL_SERVER_ERROR))
            self.assertEqual(service.state()["action"]["status"], "failed")
            service.stop_auto_capture()
            self.assertEqual(service.reset()["samples"], 0)
            self.assertIsNone(service.state()["action"])

    def test_validation_v2_captures_once_and_keeps_generation_atomic(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory) / "intrinsics.yaml"
            calibration = Path(directory) / "intrinsics-20260830T120000.000000Z.yaml"
            intrinsic_solver.save_intrinsic(
                calibration,
                intrinsic_solver.IntrinsicResult(
                    camera_matrix=np.array([
                        [638.0, 0.0, 200.0],
                        [0.0, 637.0, 160.0],
                        [0.0, 0.0, 1.0],
                    ]),
                    distortion=np.array([-0.18, 0.04, 0.0, 0.0, 0.0]),
                    image_size=(400, 320),
                    rms_reprojection_error_px=0.7,
                    sample_count=40,
                ),
                camera_name="usb_cam",
                board_size=(7, 5),
                square=0.20,
            )
            service = make_service(base)
            captures = []

            def capture():
                captures.append(len(captures) + 1)
                frame = render_board()
                frame[0, 0] = captures[-1]
                return frame

            service.attach_frame_capture(capture)
            with patch.object(
                intrinsic_validation,
                "generate_intrinsic_comparison",
                wraps=intrinsic_validation.generate_intrinsic_comparison,
            ) as render:
                first = service.validate_intrinsic(
                    {"kind": "raw"},
                    {"kind": "calibration", "calibration_id": calibration.name},
                )
            self.assertEqual(render.call_args.kwargs["jpeg_quality"], 95)
            self.assertEqual(captures, [1])
            self.assertEqual(first["schema"], "xgc2.camera.intrinsic-validation.v2")
            self.assertEqual(first["configurations"]["reference"], {"kind": "raw"})
            self.assertEqual(
                first["configurations"]["comparison"]["calibration_id"], calibration.name
            )
            self.assertEqual(first["source_image_size"], [400, 320])
            self.assertEqual(first["analysis_image_size"], [400, 320])
            self.assertEqual(
                [view["id"] for view in first["views"][-2:]],
                ["reference", "comparison"],
            )
            self.assertTrue(service.validation_image("reference", first["generation"]).startswith(b"\xff\xd8"))

            same = service.validate_intrinsic(
                {"kind": "calibration", "calibration_id": calibration.name},
                {"kind": "calibration", "calibration_id": calibration.name},
            )
            self.assertEqual(captures, [1, 2])
            self.assertEqual(same["remap_delta_px"], {"mean": 0.0, "maximum": 0.0})
            with self.assertRaises(ApiError) as stale:
                service.validation_image("reference", first["generation"])
            self.assertEqual(stale.exception.status, int(HTTPStatus.CONFLICT))

            raw = service.validate_intrinsic(
                {"kind": "raw"}, {"kind": "raw"}
            )
            self.assertEqual(captures, [1, 2, 3])
            self.assertEqual(raw["remap_delta_px"], {"mean": 0.0, "maximum": 0.0})

            with self.assertRaises(ApiError) as unsafe:
                service.validate_intrinsic(
                    {"kind": "raw"},
                    {"kind": "calibration", "calibration_id": "../intrinsics.yaml"},
                )
            self.assertEqual(unsafe.exception.status, int(HTTPStatus.BAD_REQUEST))
            self.assertEqual(captures, [1, 2, 3])

    def test_validation_generation_is_serialized_to_bound_4k_work(self):
        with tempfile.TemporaryDirectory() as directory:
            service = make_service(Path(directory) / "intrinsics.yaml")
            service.attach_frame_capture(
                lambda: np.full((64, 96, 3), 127, dtype=np.uint8)
            )
            original = intrinsic_validation.generate_intrinsic_comparison
            active = 0
            maximum_active = 0
            observation_lock = threading.Lock()
            errors = []

            def observed(*args, **kwargs):
                nonlocal active, maximum_active
                with observation_lock:
                    active += 1
                    maximum_active = max(maximum_active, active)
                try:
                    sleep(0.05)
                    return original(*args, **kwargs)
                finally:
                    with observation_lock:
                        active -= 1

            def validate():
                try:
                    service.validate_intrinsic(
                        {"kind": "raw"}, {"kind": "raw"}
                    )
                except Exception as error:  # pragma: no cover - asserted below
                    errors.append(error)

            with patch(
                "xgc_camera_calibration.intrinsic_service.intrinsic_validation.generate_intrinsic_comparison",
                side_effect=observed,
            ):
                workers = [threading.Thread(target=validate) for _ in range(2)]
                for worker in workers:
                    worker.start()
                for worker in workers:
                    worker.join(timeout=2.0)

            self.assertFalse(errors)
            self.assertEqual(maximum_active, 1)
            self.assertEqual(service._validation_generation, 2)

    def test_transport_uses_strict_candidate_save_continue_routes_without_alias(self):
        with tempfile.TemporaryDirectory() as directory:
            service = make_service(Path(directory) / "intrinsics.yaml")
            server = CalibrationHttpServer(
                ("127.0.0.1", 0),
                object(),
                WEB_ROOT,
                frame_ancestors="'self'",
                intrinsic_service=service,
            )
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            base = "http://127.0.0.1:{}".format(server.server_address[1])

            def post(path, payload):
                request = urllib.request.Request(
                    base + path,
                    data=json.dumps(payload).encode("utf-8"),
                    headers={"Content-Type": "application/json"},
                    method="POST",
                )
                with urllib.request.urlopen(request) as response:
                    return json.loads(response.read())

            try:
                with patch.object(
                    service,
                    "calibrate",
                    return_value={"candidate_id": "intrinsic-candidate-abc"},
                ) as candidate, patch.object(
                    service,
                    "save",
                    return_value={"saved": True},
                ) as save, patch.object(
                    service,
                    "continue_collection",
                    return_value={"phase": "collecting"},
                ) as continue_collection:
                    self.assertEqual(
                        post("/api/v1/intrinsic/candidate", {}),
                        {"candidate_id": "intrinsic-candidate-abc"},
                    )
                    candidate.assert_called_once_with()
                    self.assertEqual(
                        post(
                            "/api/v1/intrinsic/save",
                            {"candidate_id": "intrinsic-candidate-abc"},
                        ),
                        {"saved": True},
                    )
                    save.assert_called_once_with("intrinsic-candidate-abc")
                    self.assertEqual(
                        post("/api/v1/intrinsic/continue", {}),
                        {"phase": "collecting"},
                    )
                    continue_collection.assert_called_once_with()

                    for path, payload, status in (
                        ("/api/v1/intrinsic/calibrate", {}, HTTPStatus.NOT_FOUND),
                        (
                            "/api/v1/intrinsic/candidate",
                            {"legacy": True},
                            HTTPStatus.BAD_REQUEST,
                        ),
                        (
                            "/api/v1/intrinsic/save",
                            {"candidate_id": "", "extra": True},
                            HTTPStatus.BAD_REQUEST,
                        ),
                        (
                            "/api/v1/intrinsic/continue",
                            {"legacy": True},
                            HTTPStatus.BAD_REQUEST,
                        ),
                    ):
                        with self.assertRaises(urllib.error.HTTPError) as caught:
                            post(path, payload)
                        self.assertEqual(caught.exception.code, int(status))
            finally:
                server.shutdown()
                server.server_close()

    def test_transport_routes_intrinsic_and_gates_when_absent(self):
        with tempfile.TemporaryDirectory() as directory:
            service = make_service(Path(directory) / "intrinsics.yaml")
            service.process_frame(render_board())

            server = CalibrationHttpServer(
                ("127.0.0.1", 0), object(), WEB_ROOT,
                frame_ancestors="'self'", intrinsic_service=service,
            )
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            try:
                base = "http://127.0.0.1:{}".format(server.server_address[1])
                with urllib.request.urlopen(base + "/api/v1/intrinsic/state") as response:
                    state = json.loads(response.read())
                self.assertEqual(state["samples"], 1)
                with urllib.request.urlopen(base + "/api/v1/intrinsic/image.jpg") as response:
                    self.assertEqual(response.headers.get_content_type(), "image/jpeg")
                request = urllib.request.Request(
                    base + "/api/v1/intrinsic/reset", data=b"{}",
                    headers={"Content-Type": "application/json"}, method="POST",
                )
                with urllib.request.urlopen(request) as response:
                    self.assertEqual(json.loads(response.read())["samples"], 0)

                service.attach_frame_capture(lambda: render_board())
                request = urllib.request.Request(
                    base + "/api/v1/intrinsic/auto_capture/start", data=b"{}",
                    headers={"Content-Type": "application/json"}, method="POST",
                )
                with urllib.request.urlopen(request) as response:
                    self.assertTrue(json.loads(response.read())["auto_capture"]["enabled"])
                request = urllib.request.Request(
                    base + "/api/v1/intrinsic/auto_capture/stop", data=b"{}",
                    headers={"Content-Type": "application/json"}, method="POST",
                )
                with urllib.request.urlopen(request) as response:
                    self.assertFalse(json.loads(response.read())["auto_capture"]["enabled"])

                calibration_path = Path(directory) / "intrinsics-20260830T120000.000000Z.yaml"
                intrinsic_solver.save_intrinsic(
                    calibration_path,
                    intrinsic_solver.IntrinsicResult(
                        camera_matrix=np.array([
                            [638.0, 0.0, 200.0],
                            [0.0, 637.0, 160.0],
                            [0.0, 0.0, 1.0],
                        ]),
                        distortion=np.array([-0.18, 0.04, 0.0, 0.0, 0.0]),
                        image_size=(400, 320),
                        rms_reprojection_error_px=0.7,
                        sample_count=40,
                    ),
                    camera_name="usb_cam",
                    board_size=(7, 5),
                    square=0.20,
                    metadata={
                        "quality_contract": "xgc2.camera.intrinsic-quality.v2",
                        "candidate_id": "intrinsic-candidate-transport",
                        "stability_assessment": {"passed": True},
                    },
                )
                with urllib.request.urlopen(base + "/api/v1/intrinsic/calibrations") as response:
                    history = json.loads(response.read())
                self.assertEqual(history["selected"], calibration_path.name)
                self.assertTrue(history["items"][0]["latest"])

                validation_captures = []

                def capture_validation():
                    validation_captures.append(len(validation_captures) + 1)
                    return render_board()

                service.attach_frame_capture(capture_validation)
                samples_before_validation = service.state()["samples"]
                request = urllib.request.Request(
                    base + "/api/v1/intrinsic/validation",
                    data=json.dumps({
                        "reference": {"kind": "raw"},
                        "comparison": {
                            "kind": "calibration",
                            "calibration_id": calibration_path.name,
                        },
                    }).encode("utf-8"),
                    headers={"Content-Type": "application/json"},
                    method="POST",
                )
                with urllib.request.urlopen(request) as response:
                    comparison = json.loads(response.read())
                self.assertEqual(validation_captures, [1])
                self.assertEqual(comparison["schema"], "xgc2.camera.intrinsic-validation.v2")
                self.assertEqual(comparison["configurations"]["reference"], {"kind": "raw"})
                self.assertEqual(
                    comparison["configurations"]["comparison"]["calibration_id"],
                    calibration_path.name,
                )
                self.assertEqual(
                    [view["id"] for view in comparison["views"][-2:]],
                    ["reference", "comparison"],
                )
                with urllib.request.urlopen(
                    base + "/api/v1/intrinsic/validation/image/reference.jpg?generation={}".format(
                        comparison["generation"]
                    )
                ) as response:
                    self.assertTrue(response.read().startswith(b"\xff\xd8"))
                second_request = urllib.request.Request(
                    base + "/api/v1/intrinsic/validation",
                    data=json.dumps({
                        "reference": {"kind": "raw"},
                        "comparison": {"kind": "raw"},
                    }).encode("utf-8"),
                    headers={"Content-Type": "application/json"},
                    method="POST",
                )
                with urllib.request.urlopen(second_request) as response:
                    second = json.loads(response.read())
                self.assertEqual(validation_captures, [1, 2])
                with self.assertRaises(urllib.error.HTTPError) as stale:
                    urllib.request.urlopen(
                        base + "/api/v1/intrinsic/validation/image/reference.jpg?generation={}".format(
                            comparison["generation"]
                        )
                    )
                self.assertEqual(stale.exception.code, int(HTTPStatus.CONFLICT))
                with self.assertRaises(urllib.error.HTTPError) as malformed:
                    urllib.request.urlopen(
                        base + "/api/v1/intrinsic/validation/image/reference.jpg?generation=not-an-integer"
                    )
                self.assertEqual(malformed.exception.code, int(HTTPStatus.BAD_REQUEST))
                legacy_request = urllib.request.Request(
                    base + "/api/v1/intrinsic/validation",
                    data=json.dumps({"calibration_id": calibration_path.name}).encode("utf-8"),
                    headers={"Content-Type": "application/json"},
                    method="POST",
                )
                with self.assertRaises(urllib.error.HTTPError) as legacy:
                    urllib.request.urlopen(legacy_request)
                self.assertEqual(legacy.exception.code, int(HTTPStatus.BAD_REQUEST))
                self.assertEqual(second["generation"], comparison["generation"] + 1)
                self.assertEqual(service.state()["samples"], samples_before_validation)

                service.attach_camera_control(FakeCameraControl())
                service.auto_run = lambda: {
                    "accepted": True,
                    "action": {"name": "auto_run", "status": "running"},
                }
                request = urllib.request.Request(
                    base + "/api/v1/intrinsic/auto_run", data=b"{}",
                    headers={"Content-Type": "application/json"}, method="POST",
                )
                with urllib.request.urlopen(request) as response:
                    self.assertEqual(response.status, int(HTTPStatus.ACCEPTED))
                    self.assertTrue(json.loads(response.read())["accepted"])
            finally:
                server.shutdown()
                server.server_close()

            # With no intrinsic service the route is gated off.
            gated = CalibrationHttpServer(
                ("127.0.0.1", 0), object(), WEB_ROOT, frame_ancestors="'self'",
            )
            thread = threading.Thread(target=gated.serve_forever, daemon=True)
            thread.start()
            try:
                base = "http://127.0.0.1:{}".format(gated.server_address[1])
                with self.assertRaises(urllib.error.HTTPError) as caught:
                    urllib.request.urlopen(base + "/api/v1/intrinsic/state")
                self.assertEqual(caught.exception.code, int(HTTPStatus.NOT_FOUND))
            finally:
                gated.shutdown()
                gated.server_close()


if __name__ == "__main__":
    unittest.main()
