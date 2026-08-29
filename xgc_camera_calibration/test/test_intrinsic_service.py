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
    recommended_views,
)
from xgc_camera_calibration.web_service import ApiError, CalibrationHttpServer


WEB_ROOT = Path(__file__).resolve().parents[1] / "web" / "intrinsic"
WEB_SOURCE = Path(__file__).resolve().parents[2] / "web-src" / "src" / "intrinsic-legacy.ts"


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
        image_topic="/usb_cam/image_raw", display_width=640,
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
    def test_90_degree_simulation_sweep_stays_high_and_near(self):
        views = recommended_views((2.0, 0.0, 2.2))
        near = next(view for view in views if view["name"] == "near maximum")
        oblique_high = next(view for view in views if view["name"] == "oblique high")
        self.assertEqual(near["position"], [0.05, 0.0, 2.2])
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
        self.assertEqual(left["yaw_offset"], -0.46)
        self.assertEqual(near["position"], [1.2, 0.0, 2.2])
        self.assertGreaterEqual(min(view["position"][2] for view in views), 2.01)
        self.assertEqual(len({tuple(view["position"]) for view in views}), len(views))

    def test_web_assets_use_proxy_safe_relative_urls(self):
        index = (WEB_ROOT / "index.html").read_text(encoding="utf-8")
        app = (WEB_ROOT / "app.js").read_text(encoding="utf-8")
        styles = (WEB_ROOT / "styles.css").read_text(encoding="utf-8")
        source = WEB_SOURCE.read_text(encoding="utf-8")
        self.assertIn('href="styles.css"', index)
        self.assertIn('src="app.js"', index)
        self.assertIn('type="module"', index)
        self.assertNotIn('"/api/v1/intrinsic/', app)
        self.assertIn("api/v1/intrinsic/state", app)
        self.assertIn("xgc-app-shell", app)
        self.assertIn("Board detection", app)
        self.assertIn("detection-status", app)
        self.assertIn("URL.createObjectURL", app)
        self.assertIn("queueImageRefresh(s.detection, s.image_ready)", source)
        self.assertNotIn("setInterval(refreshImage", source)
        self.assertIn(".xgc-topbar", styles)

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
                "interval_seconds": 0.5,
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
            output = Path(directory) / "intrinsics.yaml"
            result = intrinsic_solver.IntrinsicResult(
                camera_matrix=np.array([[638.0, 0.0, 600.0], [0.0, 637.0, 390.0], [0.0, 0.0, 1.0]]),
                distortion=np.array([0.01, -0.02, 0.0, 0.0, 0.0]),
                image_size=(1280, 720), rms_reprojection_error_px=0.9, sample_count=40,
            )
            intrinsic_solver.save_intrinsic(output, result, board_size=(7, 5), square=0.20)

            restored = make_service(output)
            restored.attach_frame_capture(lambda: render_board())
            state = restored.state()
            self.assertTrue(state["calibrated"])
            self.assertTrue(state["result_restored"])
            self.assertEqual(state["samples"], 40)
            self.assertAlmostEqual(state["result"]["fx"], 638.0)
            self.assertFalse(restored.start_auto_capture(interval=0.1)["auto_capture"]["enabled"])

            restored.reset()
            self.assertFalse(output.exists())

    def test_physical_auto_capture_collects_without_manual_click_and_stops_when_ready(self):
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
                while service.state()["auto_capture"]["enabled"] and monotonic() < deadline:
                    sleep(0.01)
            state = service.state()
            self.assertEqual(state["samples"], 1)
            self.assertTrue(state["auto_capture"]["coverage_complete"])
            self.assertFalse(state["auto_capture"]["enabled"])

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

            bars = [{"label": label, "progress": 1.0} for label in ("X", "Y", "Size", "Skew")]
            def capture_at_current_target():
                with service.lock:
                    index = service.action["target_index"]
                    service.target_done[index] = True
                return {"ok": True, "samples": 15}

            with patch(
                "xgc_camera_calibration.intrinsic_service.intrinsic_solver.coverage",
                return_value=(bars, True),
            ), patch.object(service, "_capture_frame", side_effect=capture_at_current_target), patch.object(
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

                deadline = monotonic() + 2.0
                while service.state()["action"]["status"] == "running" and monotonic() < deadline:
                    sleep(0.01)
            self.assertEqual(service.state()["action"]["status"], "succeeded")
            self.assertEqual(len(camera.positions), 15)
            self.assertTrue(all(target["done"] for target in service.state()["targets"]))
            self.assertEqual(service.reset()["samples"], 0)
            self.assertIsNone(service.state()["action"])

    def test_auto_run_fails_when_a_guide_target_never_detects_the_board(self):
        with tempfile.TemporaryDirectory() as directory:
            service = make_service(Path(directory) / "intrinsics.yaml")
            service.attach_camera_control(FakeCameraControl())
            service.attach_frame_capture(lambda: render_board())
            with patch.object(service, "_capture_frame", return_value={"ok": True, "samples": 0}), patch(
                "xgc_camera_calibration.intrinsic_service.time.sleep", return_value=None,
            ):
                service.auto_run(settle=0)
                deadline = monotonic() + 1.0
                while service.state()["action"]["status"] == "running" and monotonic() < deadline:
                    sleep(0.01)
            action = service.state()["action"]
            self.assertEqual(action["status"], "failed")
            self.assertIn("left edge", action["error"])
            self.assertIn("could not detect", action["error"])

    def test_auto_run_camera_failure_is_reported_and_recoverable(self):
        class FailingCameraControl(FakeCameraControl):
            def goto(self, position, yaw_offset, pitch_offset, roll):
                raise RuntimeError("Gazebo camera rejected the pose")

        with tempfile.TemporaryDirectory() as directory:
            service = make_service(Path(directory) / "intrinsics.yaml")
            service.attach_camera_control(FailingCameraControl())
            service.auto_run(settle=0)
            deadline = monotonic() + 1.0
            while service.state()["action"]["status"] == "running" and monotonic() < deadline:
                sleep(0.01)
            action = service.state()["action"]
            self.assertEqual(action["status"], "failed")
            self.assertIn("rejected the pose", action["error"])
            self.assertEqual(service.reset()["samples"], 0)
            self.assertIsNone(service.state()["action"])

    def test_auto_run_thread_start_failure_does_not_leave_permanent_busy_state(self):
        with tempfile.TemporaryDirectory() as directory:
            service = make_service(Path(directory) / "intrinsics.yaml")
            service.attach_camera_control(FakeCameraControl())
            with patch("xgc_camera_calibration.intrinsic_service.threading.Thread") as constructor:
                constructor.return_value.start.side_effect = RuntimeError("thread unavailable")
                with self.assertRaises(ApiError) as caught:
                    service.auto_run()
            self.assertEqual(caught.exception.status, int(HTTPStatus.INTERNAL_SERVER_ERROR))
            self.assertEqual(service.state()["action"]["status"], "failed")
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
