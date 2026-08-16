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
from unittest.mock import patch

import cv2
import numpy as np

from xgc_camera_calibration.intrinsic_service import (
    IntrinsicCalibrationService,
    recommended_views,
)
from xgc_camera_calibration.web_service import ApiError, CalibrationHttpServer


WEB_ROOT = Path(__file__).resolve().parents[1] / "web" / "intrinsic"


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


class IntrinsicServiceTest(unittest.TestCase):
    def test_110_degree_simulation_near_view_fills_size_coverage(self):
        views = recommended_views((2.0, 0.0, 1.5))
        near = next(view for view in views if view["name"] == "near (big)")
        self.assertEqual(near["position"], [1.1, 0.0, 1.5])

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
        self.assertIn(".xgc-topbar", styles)

    def test_process_frame_collects_a_board_sample(self):
        with tempfile.TemporaryDirectory() as directory:
            service = make_service(Path(directory) / "intrinsics.yaml")
            service.process_frame(render_board())
            state = service.state()
            self.assertEqual(state["mode"], "intrinsic")
            self.assertEqual(state["samples"], 1)
            self.assertEqual([bar["label"] for bar in state["coverage"]], ["X", "Y", "Size", "Skew"])
            self.assertTrue(service.image_jpeg().startswith(b"\xff\xd8"))

    def test_non_board_frame_adds_no_sample(self):
        with tempfile.TemporaryDirectory() as directory:
            service = make_service(Path(directory) / "intrinsics.yaml")
            service.process_frame(np.full((200, 320, 3), 127, np.uint8))
            self.assertEqual(service.state()["samples"], 0)

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
            self.assertEqual(len(document["views"]), 10)
            self.assertFalse(document["camera_control"])
            state = service.state()
            self.assertEqual(len(state["targets"]), 10)
            self.assertIsNone(state["pose"])
            self.assertFalse(state["camera_control"])
            self.assertIsNone(state["action"])
            self.assertEqual(state["next"], 0)
            self.assertFalse(state["targets"][0]["done"])

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

            started = monotonic()
            accepted = service.auto_run(settle=0.01)
            self.assertLess(monotonic() - started, 0.1)
            self.assertTrue(accepted["accepted"])
            self.assertEqual(accepted["action"]["status"], "running")
            for mutation in (
                service.reset,
                service.reset_pose,
                service.calibrate,
                lambda: service.goto(0),
                lambda: service.auto_run(settle=0.01),
            ):
                with self.assertRaises(ApiError) as caught:
                    mutation()
                self.assertEqual(caught.exception.status, int(HTTPStatus.CONFLICT))
                self.assertIn("already running", caught.exception.message)

            deadline = monotonic() + 2.0
            while service.state()["action"] is not None and monotonic() < deadline:
                sleep(0.01)
            self.assertIsNone(service.state()["action"])
            self.assertEqual(len(camera.positions), 10)
            self.assertEqual(service.reset()["samples"], 0)

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
