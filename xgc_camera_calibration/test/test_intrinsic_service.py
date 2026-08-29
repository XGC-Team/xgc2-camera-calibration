#!/usr/bin/env python3

import json
import tempfile
import threading
import unittest
import urllib.error
import urllib.request
from http import HTTPStatus
from pathlib import Path
from time import monotonic, sleep
from types import SimpleNamespace
from unittest.mock import patch

import cv2
import numpy as np

from xgc_camera_calibration import intrinsic_solver
from xgc_camera_calibration.intrinsic_service import (
    IntrinsicCalibrationService,
    intrinsic_calibration_directory,
    recommended_views,
)
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


def make_service(output_file):
    # 8x6 squares -> 7x5 interior corners.
    return IntrinsicCalibrationService(
        board_size=(7, 5), square=0.20, output_file=str(output_file),
        camera_name="usb_cam",
        media_source="usb_cam", display_width=640,
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

    def test_90_degree_simulation_sweep_stays_high_and_near(self):
        views = recommended_views((2.0, 0.0, 2.2))
        near = next(view for view in views if view["name"] == "near maximum")
        oblique_high = next(view for view in views if view["name"] == "oblique high")
        self.assertEqual(near["position"], [0.7, 0.0, 2.2])
        self.assertEqual(oblique_high["position"], [-0.2, 0.35, 2.55])
        self.assertEqual(len({tuple(view["position"]) for view in views}), len(views))
        self.assertGreaterEqual(min(view["position"][2] for view in views), 1.75)
        self.assertLessEqual(max(view["position"][2] for view in views), 2.65)
        self.assertGreaterEqual(max(abs(view["roll"]) for view in views), 0.46)

    def test_field_aprilgrid_scales_simulation_views_to_target_extent(self):
        views = recommended_views((2.0, 0.0, 2.2), 0.66)
        left = next(view for view in views if view["name"] == "left edge")
        near = next(view for view in views if view["name"] == "near maximum")
        self.assertEqual(left["position"], [0.8, 0.02, 2.2])
        self.assertEqual(left["yaw_offset"], -0.58)
        self.assertEqual(near["position"], [1.46, 0.0, 2.2])
        self.assertGreaterEqual(min(view["position"][2] for view in views), 2.01)
        self.assertEqual(len({tuple(view["position"]) for view in views}), len(views))

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
            camera.current_pose = {"position": list(target)}
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

    def test_manual_goto_marks_the_explicit_target_when_guide_positions_overlap(self):
        with tempfile.TemporaryDirectory() as directory:
            service = make_service(Path(directory) / "intrinsics.yaml")
            camera = FakeCameraControl()
            service.attach_camera_control(camera)
            service.views[0]["position"] = [0.0, 0.0, 0.0]
            service.views[1]["position"] = [0.0, 0.01, 0.0]
            service.goto(1)
            service.process_frame(render_board())
            self.assertFalse(service.target_done[0])
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
                calibration_image_points=np.asarray([[[30, 30]]], np.float32),
                calibration_object_points=np.zeros((1, 3), np.float32),
            )
            refined = np.asarray([[[123.0, 234.0]]], np.float32)
            with patch.object(
                intrinsic_solver, "detect_board", return_value=detection
            ) as detect, patch.object(
                intrinsic_solver,
                "refine_aprilgrid_calibration_centers",
                return_value=refined,
            ) as refine:
                service.process_frame(
                    reduced,
                    source_image_size=(640, 480),
                    source_jpeg=encoded.tobytes(),
                )
            self.assertEqual(detect.call_args.args[2], 160)
            self.assertEqual(refine.call_args.args[0].shape, (480, 640))
            np.testing.assert_allclose(
                refine.call_args.args[1].reshape(-1, 2),
                corners.reshape(-1, 2) * 4.0,
            )
            np.testing.assert_allclose(service.image_points[0], refined)

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
                calibration_image_points=np.asarray([[[120, 120]]], np.float32),
                calibration_object_points=np.zeros((1, 3), np.float32),
            )
            with patch.object(
                intrinsic_solver, "detect_board", side_effect=[None, detection]
            ) as detect, patch.object(
                intrinsic_solver, "aprilgrid_has_candidate_evidence", return_value=True
            ), patch.object(
                intrinsic_solver, "refine_aprilgrid_calibration_centers",
                return_value=np.asarray([[[180, 180]]], np.float32),
            ):
                service.process_frame(
                    reduced,
                    source_image_size=(1920, 1080),
                    source_jpeg=encoded.tobytes(),
                )
            self.assertEqual([call.args[2] for call in detect.call_args_list], [480, 1920])
            self.assertEqual(service.state()["detection"]["corner_count"], 4)

    def test_repeated_board_frame_reports_geometric_duplicate(self):
        with tempfile.TemporaryDirectory() as directory:
            service = make_service(Path(directory) / "intrinsics.yaml")
            frame = render_board()
            service.process_frame(frame)
            service.process_frame(frame)
            state = service.state()
            self.assertEqual(state["samples"], 1)
            self.assertEqual(state["detection"]["status"], "detected")
            self.assertFalse(state["detection"]["accepted"])
            self.assertTrue(state["detection"]["duplicate"])

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
            self.assertTrue(state["recovery"]["checkpoint_available"])
            self.assertFalse(state["result_restored"])

    def test_saved_result_restores_without_recollecting_samples(self):
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
            self.assertTrue(state["calibrated"])
            self.assertTrue(state["result_restored"])
            self.assertEqual(state["samples"], 40)
            self.assertAlmostEqual(state["result"]["fx"], 638.0)
            self.assertTrue(restored.start_auto_capture(interval=0.1)["auto_capture"]["enabled"])
            restored.stop_auto_capture()

            restored.reset()
            self.assertTrue(output.exists())
            self.assertFalse(restored.state()["calibrated"])
            self.assertEqual(restored.state()["samples"], 0)

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
            )
            intrinsic_solver.save_intrinsic(
                newer,newer_result,camera_name="usb_cam",board_size=(7, 5),square=0.20,
            )

            restored = make_service(base)
            state = restored.state()
            self.assertTrue(state["result_restored"])
            self.assertEqual(state["output_file"], str(newer))
            self.assertEqual(state["result"]["output_file"], str(newer))
            self.assertAlmostEqual(state["result"]["fx"], 742.0)

    def test_calibrate_writes_timestamped_version_and_reset_preserves_it(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory) / "intrinsics.yaml"
            service = make_service(base)
            service.image_points = [np.zeros((35, 1, 2), dtype=np.float32)]
            service.object_points = [np.zeros((35, 3), dtype=np.float32)]
            service.image_size = (1280, 720)
            result = intrinsic_solver.IntrinsicResult(
                camera_matrix=np.array([[638.0, 0.0, 600.0], [0.0, 637.0, 390.0], [0.0, 0.0, 1.0]]),
                distortion=np.array([0.01, -0.02, 0.0, 0.0, 0.0]),
                image_size=(1280, 720), rms_reprojection_error_px=0.9, sample_count=1,
            )
            with patch.object(intrinsic_solver, "calibrate_intrinsic", return_value=result):
                solved = service.calibrate()

            saved = Path(solved["output_file"])
            self.assertRegex(saved.name, r"^intrinsics-\d{8}T\d{6}\.\d{6}Z\.yaml$")
            self.assertTrue(saved.is_file())
            self.assertFalse(base.exists())
            saved_document = intrinsic_solver.load_intrinsic(saved)
            self.assertEqual(saved_document["camera_name"], "usb_cam")
            self.assertEqual(saved_document["metadata"]["camera_name"], "usb_cam")

            reset = service.reset()
            self.assertEqual(reset["samples"], 0)
            self.assertFalse(reset["calibrated"])
            self.assertTrue(saved.is_file())
            self.assertEqual(reset["output_file"], str(base))

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
            service.attach_frame_capture(lambda: render_board())
            service.start_auto_capture(interval=0.1)

            bars = [{"label": label, "progress": 1.0} for label in ("X", "Y", "Size", "Skew")]
            with patch(
                "xgc_camera_calibration.intrinsic_service.intrinsic_solver.coverage",
                return_value=(bars, True),
            ), patch.object(
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

                deadline = monotonic() + 4.0
                while service.state()["action"]["status"] == "running" and monotonic() < deadline:
                    sleep(0.01)
            self.assertEqual(service.state()["action"]["status"], "succeeded")
            self.assertEqual(len(camera.positions), 15)
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
                )
                with urllib.request.urlopen(base + "/api/v1/intrinsic/calibrations") as response:
                    history = json.loads(response.read())
                self.assertEqual(history["selected"], calibration_path.name)
                self.assertTrue(history["items"][0]["latest"])

                samples_before_validation = service.state()["samples"]
                request = urllib.request.Request(
                    base + "/api/v1/intrinsic/validation",
                    data=json.dumps({"calibration_id": calibration_path.name}).encode("utf-8"),
                    headers={"Content-Type": "application/json"},
                    method="POST",
                )
                with urllib.request.urlopen(request) as response:
                    validation = json.loads(response.read())
                self.assertEqual(validation["schema"], "xgc2.camera.intrinsic-validation.v1")
                self.assertEqual(validation["calibration_id"], calibration_path.name)
                self.assertEqual(validation["default_view"], "overlay_checker")
                self.assertEqual(
                    [view["id"] for view in validation["views"][:5]],
                    [
                        "overlay_checker",
                        "overlay_redcyan",
                        "overlay_corner_zoom",
                        "overlay_diff",
                        "displacement",
                    ],
                )
                self.assertGreater(validation["remap_px"]["maximum"], 0.0)
                self.assertEqual(service.state()["samples"], samples_before_validation)
                with urllib.request.urlopen(
                    base + "/api/v1/intrinsic/validation/image/displacement.jpg"
                ) as response:
                    self.assertEqual(response.headers.get_content_type(), "image/jpeg")
                    self.assertTrue(response.read().startswith(b"\xff\xd8"))

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
