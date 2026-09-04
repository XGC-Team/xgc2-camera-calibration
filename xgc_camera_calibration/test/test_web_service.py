#!/usr/bin/env python3

import json
import shutil
import tempfile
import threading
import unittest
import urllib.error
import urllib.request
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np

from xgc_camera_calibration.solver import load_extrinsic
from xgc_camera_calibration.web_service import (
    ApiError,
    CalibrationHttpServer,
    CalibrationService,
    FrameSnapshot,
    MarkerObservation,
    image_message_to_bgr,
)


class FakeSource:
    image_topic = "/camera/image_raw"
    preview_image_topic = "/camera/image_raw/compressed"
    intrinsic_file = "/camera/sim/usb_cam/intrinsics-20260830T010203.000000Z.yaml"
    pose_prefix = "/vrpn_client_node"
    preview_jpeg = b"\xff\xd8cached-compressed-preview\xff\xd9"

    def __init__(self, snapshot):
        self.snapshot = snapshot

    def status(self):
        return {
            "image_topic": self.image_topic,
            "preview_image_topic": self.preview_image_topic,
            "intrinsic_file": self.intrinsic_file,
            "pose_prefix": self.pose_prefix,
            "image_ready": True,
            "preview_ready": True,
            "intrinsic_ready": True,
            "marker_count": len(self.snapshot.markers),
            "marker_names": sorted(self.snapshot.markers),
            "latest_image_stamp_sec": self.snapshot.stamp_sec,
        }

    def freeze(self, parent_frame):
        if parent_frame != "map":
            raise AssertionError("unexpected freeze arguments")
        return self.snapshot

    def preview_jpeg_bytes(self):
        return self.preview_jpeg


class WebCalibrationServiceTest(unittest.TestCase):
    def test_ros_entrypoints_require_the_shared_storage_identity(self):
        package = Path(__file__).resolve().parents[1]
        calibrator = (package / "scripts" / "extrinsic_calibrator_web.py").read_text(
            encoding="utf-8"
        )
        publisher = (package / "scripts" / "extrinsic_tf_publisher.py").read_text(
            encoding="utf-8"
        )
        for source in (calibrator, publisher):
            self.assertIn('rospy.get_param("~calibration_root")', source)
            self.assertIn('rospy.get_param("~calibration_mode")', source)
            self.assertIn('rospy.get_param("~camera_name")', source)
        self.assertNotIn('rospy.get_param("~output_file"', calibrator)
        self.assertNotIn('rospy.get_param("~extrinsic_file"', publisher)
        self.assertIn("optional_selected_intrinsic_path", calibrator)
        self.assertIn("Select a timestamped intrinsic calibration YAML before freezing", calibrator)
        self.assertIn("default_transform_chain", publisher)
        self.assertIn("Publishing default camera extrinsic", publisher)

    def test_web_assets_use_proxy_safe_relative_urls(self):
        web_root = Path(__file__).resolve().parents[1] / "web" / "extrinsic"
        index = (web_root / "index.html").read_text(encoding="utf-8")
        app = (web_root / "app.js").read_text(encoding="utf-8")
        styles = (web_root / "styles.css").read_text(encoding="utf-8")
        self.assertIn('href="styles.css"', index)
        self.assertIn('src="app.js"', index)
        self.assertIn('type="module"', index)
        self.assertNotIn('"/api/v1/', app)
        self.assertNotIn('`/api/v1/', app)
        self.assertIn("api/v1/state", app)
        self.assertIn("xgc-app-shell", app)
        self.assertIn(".xgc-topbar", styles)

    def setUp(self):
        self.world = np.array(
            [
                [-1.0, -0.7, 0.0],
                [1.0, -0.7, 0.1],
                [1.1, 0.8, -0.1],
                [-0.9, 0.9, 0.2],
                [-0.6, -0.4, 1.0],
                [0.8, -0.5, 1.2],
            ],
            dtype=np.float64,
        )
        self.intrinsic = np.array(
            [[680.0, 0.0, 320.0], [0.0, 675.0, 240.0], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )
        self.distortion = np.zeros(5, dtype=np.float64)
        self.rvec = np.array([0.12, -0.08, 0.04], dtype=np.float64)
        self.tvec = np.array([0.15, -0.2, 4.5], dtype=np.float64)
        pixels, _ = cv2.projectPoints(
            self.world.reshape(-1, 1, 3),
            self.rvec,
            self.tvec,
            self.intrinsic,
            self.distortion,
        )
        self.pixels = pixels.reshape(-1, 2)
        markers = {
            "marker_{:02d}".format(index + 1): MarkerObservation(
                name="marker_{:02d}".format(index + 1),
                position=tuple(map(float, position)),
                frame_id="map",
            )
            for index, position in enumerate(self.world)
        }
        self.snapshot = FrameSnapshot(
            image=np.zeros((480, 640, 3), dtype=np.uint8),
            stamp_sec=12.34,
            frame_id="camera_optical_frame",
            camera_matrix=self.intrinsic,
            distortion=self.distortion,
            markers=markers,
        )
        self.temporary = tempfile.TemporaryDirectory()
        self.calibration_root = Path(self.temporary.name) / "calibrations"
        self.output_directory = self.calibration_root / "sim" / "usb_cam"
        self.service = CalibrationService(
            FakeSource(self.snapshot),
            calibration_root=str(self.calibration_root),
            calibration_mode="sim",
            camera_name="usb_cam",
            parent_frame="map",
            child_frame="camera_optical_frame",
            maximum_inlier_error_px=1.0,
        )

    def tearDown(self):
        self.temporary.cleanup()

    def point_request(self):
        return {
            "generation": self.service.generation,
            "points": [
                {"marker": name, "pixel": list(map(float, pixel))}
                for name, pixel in zip(sorted(self.snapshot.markers), self.pixels)
            ],
        }

    def test_freeze_solve_and_save_round_trip(self):
        state = self.service.freeze()
        self.assertEqual(state["mode"], "frozen")
        self.assertEqual(len(state["markers"]), 6)
        self.assertTrue(self.service.image_jpeg().startswith(b"\xff\xd8"))
        result = self.service.solve(self.point_request())
        self.assertLess(result["max_reprojection_error_px"], 1e-3)
        self.assertEqual(len(result["projections"]), 6)
        self.assertFalse(result["saved"])
        self.assertIsNone(result["output_file"])
        self.assertEqual(list(self.output_directory.glob("extrinsics-*.yaml")), [])
        saved = self.service.save(result["candidate_id"])
        output = Path(saved["output_file"])
        self.assertEqual(output.parent, self.output_directory)
        self.assertRegex(
            output.name,
            r"^extrinsics-\d{8}T\d{6}\.\d{6}Z(?:-\d{2})?\.yaml$",
        )
        self.assertTrue(output.is_file())
        self.assertFalse((self.output_directory / "extrinsics.yaml").exists())
        self.assertEqual(self.service.state()["output_file"], str(output))
        self.assertFalse(self.service.state()["result_restored"])
        self.assertTrue(Path(saved["selection_file"]).is_file())
        document = load_extrinsic(output)
        self.assertEqual(document["calibration_mode"], "sim")
        self.assertEqual(document["camera_name"], "usb_cam")
        self.assertTrue(document["metadata"]["web_calibrator"])
        self.assertEqual(document["metadata"]["candidate_id"], result["candidate_id"])
        self.assertEqual(document["metadata"]["image_topic"], FakeSource.image_topic)
        self.assertEqual(self.service.save(result["candidate_id"]), saved)

    def test_restart_restores_only_the_exact_shared_selection(self):
        self.service.freeze()
        candidate = self.service.solve(self.point_request())
        saved = self.service.save(candidate["candidate_id"])

        restored = CalibrationService(
            FakeSource(self.snapshot),
            calibration_root=str(self.calibration_root),
            calibration_mode="sim",
            camera_name="usb_cam",
            parent_frame="map",
            child_frame="camera_optical_frame",
            maximum_inlier_error_px=1.0,
        )
        state = restored.state()
        self.assertEqual(state["mode"], "live")
        self.assertTrue(state["result_restored"])
        self.assertEqual(state["output_file"], saved["output_file"])
        self.assertTrue(state["result"]["saved"])
        self.assertEqual(state["result"]["candidate_id"], candidate["candidate_id"])
        self.assertEqual(state["result"]["selection_file"], saved["selection_file"])

        restored.freeze()
        self.assertFalse(restored.state()["result_restored"])
        self.assertIsNone(restored.state()["result"])

    def test_corrupt_selection_is_visible_but_does_not_block_fresh_save(self):
        pointer = self.calibration_root / "selections" / "usb_cam" / "sim-extrinsic.json"
        pointer.parent.mkdir(parents=True)
        pointer.write_text('{"schema":"broken"}\n', encoding="utf-8")
        service = CalibrationService(
            FakeSource(self.snapshot), calibration_root=str(self.calibration_root),
            calibration_mode="sim", camera_name="usb_cam", parent_frame="map",
            child_frame="camera_optical_frame", maximum_inlier_error_px=1.0,
        )
        self.assertFalse(service.state()["result_restored"])
        self.assertIn("invalid shape", service.state()["recovery_error"])
        service.freeze()
        request = self.point_request()
        request["generation"] = service.generation
        candidate = service.solve(request)
        saved = service.save(candidate["candidate_id"])
        self.assertTrue(saved["saved"])
        self.assertIsNone(service.state()["recovery_error"])

    def test_saved_retry_rejects_a_superseding_shared_selection(self):
        self.service.freeze()
        candidate = self.service.solve(self.point_request())
        self.service.save(candidate["candidate_id"])
        original = Path(self.service.output_file)
        replacement = original.with_name("extrinsics-20990101T000000.000000Z.yaml")
        shutil.copyfile(original, replacement)
        from xgc_camera_calibration.solver import write_extrinsic_selection
        write_extrinsic_selection(
            str(self.calibration_root), "sim", "usb_cam", replacement,
            candidate["candidate_id"],
        )
        with self.assertRaisesRegex(ApiError, "superseded"):
            self.service.save(candidate["candidate_id"])

    def test_state_has_no_fabricated_output_alias_before_solve(self):
        state = self.service.state()
        self.assertIsNone(state["output_file"])
        self.assertEqual(state["calibration_mode"], "sim")
        self.assertEqual(state["camera_name"], "usb_cam")

    def test_solve_is_a_fixed_point_and_save_is_the_only_writer(self):
        self.service.freeze()
        first = self.service.solve(self.point_request())
        second = self.service.solve(self.point_request())
        self.assertEqual(first["candidate_id"], second["candidate_id"])
        self.assertEqual(len(list(self.output_directory.glob("extrinsics-*.yaml"))), 0)
        with self.assertRaises(ApiError) as context:
            self.service.save("extrinsic-candidate-stale")
        self.assertEqual(context.exception.status, 409)
        saved = self.service.save(first["candidate_id"])
        self.assertTrue(saved["saved"])
        self.assertEqual(len(list(self.output_directory.glob("extrinsics-*.yaml"))), 1)

    def test_pointer_failure_retries_the_same_immutable_output(self):
        self.service.freeze()
        candidate = self.service.solve(self.point_request())
        from xgc_camera_calibration import web_service as module

        real_write = module.write_extrinsic_selection
        attempts = 0

        def flaky_write(*args, **kwargs):
            nonlocal attempts
            attempts += 1
            if attempts == 1:
                raise OSError("pointer unavailable")
            return real_write(*args, **kwargs)

        with patch.object(module, "write_extrinsic_selection", side_effect=flaky_write):
            with self.assertRaisesRegex(ApiError, "Could not save or select"):
                self.service.save(candidate["candidate_id"])
            outputs = list(self.output_directory.glob("extrinsics-*.yaml"))
            self.assertEqual(len(outputs), 1)
            saved = self.service.save(candidate["candidate_id"])
        self.assertEqual(Path(saved["output_file"]), outputs[0])
        self.assertEqual(len(list(self.output_directory.glob("extrinsics-*.yaml"))), 1)

    def test_live_preview_reuses_compressed_jpeg_without_reencoding(self):
        with patch.object(
            self.service,
            "_encode_jpeg",
            side_effect=AssertionError("live preview must not be re-encoded"),
        ):
            self.assertEqual(
                self.service.image_jpeg(),
                FakeSource.preview_jpeg,
            )

    def test_rejects_duplicate_marker_and_stale_generation(self):
        self.service.freeze()
        request = self.point_request()
        request["generation"] -= 1
        with self.assertRaises(ApiError) as context:
            self.service.solve(request)
        self.assertEqual(context.exception.status, 409)

        request = self.point_request()
        request["points"][1]["marker"] = request["points"][0]["marker"]
        with self.assertRaises(ApiError) as context:
            self.service.solve(request)
        self.assertEqual(context.exception.status, 400)

    def test_converts_padded_rgb_and_mono_images_without_cv_bridge(self):
        class Message:
            pass

        rgb = Message()
        rgb.height = 1
        rgb.width = 2
        rgb.encoding = "rgb8"
        rgb.step = 8
        rgb.data = bytes([255, 0, 0, 0, 255, 0, 99, 99])
        converted = image_message_to_bgr(rgb)
        np.testing.assert_array_equal(
            converted, np.array([[[0, 0, 255], [0, 255, 0]]], dtype=np.uint8)
        )

        mono = Message()
        mono.height = 1
        mono.width = 2
        mono.encoding = "mono8"
        mono.step = 2
        mono.data = bytes([7, 201])
        converted = image_message_to_bgr(mono)
        np.testing.assert_array_equal(
            converted, np.array([[[7, 7, 7], [201, 201, 201]]], dtype=np.uint8)
        )

    def test_http_server_serves_assets_health_and_api(self):
        web_root = Path(__file__).resolve().parents[1] / "web" / "extrinsic"
        server = CalibrationHttpServer(
            ("127.0.0.1", 0),
            self.service,
            web_root,
            frame_ancestors="'self' http://localhost:*",
        )
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        base = "http://127.0.0.1:{}".format(server.server_address[1])
        try:
            with urllib.request.urlopen(base + "/healthz", timeout=3) as response:
                health = json.loads(response.read().decode("utf-8"))
                self.assertEqual(health["status"], "ok")
                self.assertEqual(health["marker_count"], 6)
                self.assertIn("frame-ancestors", response.headers["Content-Security-Policy"])
            request = urllib.request.Request(
                base + "/api/v1/freeze",
                data=b"{}",
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(request, timeout=3) as response:
                frozen = json.loads(response.read().decode("utf-8"))
                self.assertEqual(frozen["mode"], "frozen")
            with urllib.request.urlopen(base + "/api/v1/image.jpg", timeout=3) as response:
                self.assertEqual(response.headers.get_content_type(), "image/jpeg")
                self.assertTrue(response.read().startswith(b"\xff\xd8"))
            with urllib.request.urlopen(base + "/", timeout=3) as response:
                self.assertIn(b"Camera extrinsic calibration", response.read())
            with self.assertRaises(urllib.error.HTTPError) as context:
                urllib.request.urlopen(base + "/../package.xml", timeout=3)
            self.assertEqual(context.exception.code, 404)
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=3)


if __name__ == "__main__":
    unittest.main()
