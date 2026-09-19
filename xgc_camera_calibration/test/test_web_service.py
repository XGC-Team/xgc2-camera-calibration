#!/usr/bin/env python3

import hashlib
from dataclasses import replace
import json
import shutil
import tempfile
import threading
import unittest
import uuid
import urllib.error
import urllib.request
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np

from xgc_camera_calibration.solver import load_extrinsic
from xgc_camera_calibration.extrinsic_coordinates import coordinate_provenance, optical_translation_in_world
from xgc_camera_calibration.web_service import (
    ApiError,
    CalibrationHttpServer,
    CalibrationService,
    FrameSnapshot,
    MarkerObservation,
    image_message_to_bgr,
)


class FakeSource:
    source_id = "camera-source"
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

    def observe_marker(self, marker, width, height, parent_frame):
        snapshot = self.snapshot
        if (width, height) != (snapshot.width, snapshot.height):
            raise ApiError(409, "source dimensions changed")
        observed = snapshot.markers.get(marker)
        if observed is None:
            raise ApiError(409, "Selected marker has no fresh pose observation")
        return {"marker": replace(observed, observation_id=uuid.uuid4().hex,
                    source_stamp_sec=snapshot.stamp_sec, received_at_sec=123456., received_monotonic_sec=100.),
                "camera_matrix": snapshot.camera_matrix, "distortion": snapshot.distortion,
                "camera_model": snapshot.camera_model, "pose_coordinates": snapshot.pose_coordinates}

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
        self.assertIn('rospy.get_param("~intrinsic_file", "")', calibrator)
        self.assertIn("ideal_intrinsic_parameters", calibrator)
        self.assertIn("ideal-pinhole", calibrator)
        self.assertNotIn("Freeze/Solve stay closed", calibrator)
        self.assertIn("snapshot_health_timer", calibrator)
        self.assertIn("except MediaSnapshotError as error", calibrator)
        self.assertIn("without failing the Experiment", calibrator)
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

    def current(self, service=None):
        service = service or self.service
        return {"sampling_session_id": service.samples.session_id, "expected_revision": service.samples.revision}

    def begin(self, marker, pixel, service=None, **extra):
        service = service or self.service
        frame = service.source.snapshot
        request = {**self.current(service), "request_id": uuid.uuid4().hex, "marker": marker,
                   "pixel": list(map(float, pixel)), "display": {
                       "id": uuid.uuid4().hex, "source_id": service.source.source_id, "source_epoch": "stream-1",
                       "width": frame.width, "height": frame.height,
                       "clock_domain": "browser-performance", "presented_at_ms": 42.}}
        request.update(extra)
        return request, service.begin_sample(request)

    def admit(self, marker, pixel, service=None, **extra):
        service = service or self.service
        request, pending = self.begin(marker, pixel, service, **extra)
        ok, encoded = cv2.imencode(".png", service.source.snapshot.image)
        self.assertTrue(ok)
        service.commit_sample(pending["sample_id"], encoded.tobytes(), "image/png")
        return pending["sample_id"]

    def point_request(self, service=None, names=None, pixels=None):
        service = service or self.service
        if not service.samples.active:
            for marker, pixel in zip(names or sorted(service.source.snapshot.markers), self.pixels if pixels is None else pixels):
                self.admit(marker, pixel, service)
        return self.current(service)

    def test_actual_source_frame_model_matches_ideal_and_selected_inputs(self):
        import ast
        import threading
        from types import SimpleNamespace
        from xgc_camera_calibration.pose_freshness import pose_is_fresh
        from xgc_camera_calibration.intrinsic_validation import ideal_intrinsic_parameters
        # Compile the actual ROS adapter method with its numerical dependencies;
        # ROS subscription/transport setup is outside this offline regression.
        source_path = Path(__file__).resolve().parents[1] / "scripts" / "extrinsic_calibrator_web.py"
        tree = ast.parse(source_path.read_text())
        cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "RosCalibrationSource")
        methods = [node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name in ("_frame_snapshot", "_observation_context")]
        namespace = {"FrameSnapshot": FrameSnapshot, "MarkerObservation": MarkerObservation,
                     "ApiError": ApiError, "replace": replace, "ideal_intrinsic_parameters": ideal_intrinsic_parameters,
                     "coordinate_provenance": coordinate_provenance,
                     "pose_is_fresh": pose_is_fresh,
                     "time": SimpleNamespace(monotonic=lambda: 100.),
                     "rospy": SimpleNamespace(Time=SimpleNamespace(now=lambda: SimpleNamespace(to_sec=lambda: 12.34)))}
        exec(compile(ast.Module(body=methods, type_ignores=[]), str(source_path), "exec"), namespace)
        for ideal in (True, False):
            expected_k, expected_d, size = ideal_intrinsic_parameters(640, 480, 110.)
            model = {"intrinsic_source": "ideal-pinhole" if ideal else "selected-file"}
            source = SimpleNamespace(use_ideal_intrinsics=ideal, ideal_horizontal_fov_degrees=110.,
                intrinsic_matrix=expected_k, intrinsic_distortion=expected_d, intrinsic_size=size,
                intrinsic_provenance=model, lock=threading.RLock(), marker_latest=self.snapshot.markers,
                pose_coordinate_source="raw-vrpn", pose_world_offset=(10., -5., 2.),
                marker_receipts={name:(100.,12.34) for name in self.snapshot.markers}, pose_max_age=2.)
            source._observation_context = lambda width, height, parent: namespace["_observation_context"](source, width, height, parent)
            result = namespace["_frame_snapshot"](source, self.snapshot.image, 12.34, "optical", "map")
            np.testing.assert_array_equal(result.camera_matrix, expected_k)
            np.testing.assert_array_equal(result.distortion, expected_d)
            self.assertEqual(result.camera_model, model)
            self.assertIsNot(result.camera_model, model)
            self.assertEqual(result.pose_coordinates["kind"], "experiment-world")
            self.assertEqual(result.pose_coordinates["input_kind"], "raw-vrpn")
            for name, observation in result.markers.items():
                np.testing.assert_allclose(observation.position, np.asarray(self.snapshot.markers[name].position) + [10., -5., 2.])
                self.assertEqual(observation.source_position, self.snapshot.markers[name].position)

    def test_shifted_world_pose_is_saved_and_simulation_uses_it_directly(self):
        offset = np.asarray([-8., 3., 1.])
        provenance = coordinate_provenance("experiment-world", "map", offset)
        provenance["input_kind"] = "raw-vrpn"
        markers = {name: replace(marker, position=tuple(np.asarray(marker.position)+offset),
                                 source_position=marker.position)
                   for name, marker in self.snapshot.markers.items()}
        self.service.source.snapshot = replace(self.snapshot, markers=markers, pose_coordinates=provenance)
        self.service.freeze()
        request = self.point_request()
        provenance["world_offset"][0] = 999.
        candidate = self.service.solve(request)
        saved = self.service.save(candidate["candidate_id"])
        document = load_extrinsic(saved["output_file"])
        raw_camera = -cv2.Rodrigues(self.rvec)[0].T.dot(self.tvec)
        np.testing.assert_allclose(document["translation_array"], raw_camera + offset, atol=1e-5)
        np.testing.assert_allclose(optical_translation_in_world(document, None), raw_camera + offset, atol=1e-5)
        np.testing.assert_allclose(optical_translation_in_world(document, offset), raw_camera + offset, atol=1e-5)
        np.testing.assert_allclose(optical_translation_in_world(document, [2., 4., 6.]), raw_camera + [2., 4., 6.], atol=1e-5)
        self.assertEqual(document["metadata"]["pose_coordinates"]["world_offset"], offset.tolist())
        for point in document["points"]:
            np.testing.assert_allclose(point["world"], np.asarray(point["source_world"])+offset)

    def test_frozen_model_survives_source_changes_save_and_restart(self):
        for kind in ("ideal-pinhole", "selected-file"):
            with self.subTest(kind=kind):
                model = {"intrinsic_source": kind, "distortion_model": "plumb_bob"}
                if kind == "ideal-pinhole":
                    model.update(ideal_horizontal_fov_degrees=110., assumption="Assumed ideal pinhole", intrinsic_file="")
                else:
                    from xgc_camera_calibration.intrinsic_solver import load_intrinsic
                    asset = Path(self.temporary.name) / "intrinsics.yaml"
                    original = json.dumps({"schema": "xgc2.camera.intrinsic.v1", "camera_matrix": {"data": self.intrinsic.reshape(-1).tolist()}}).encode()
                    asset.write_bytes(original)
                    loaded = load_intrinsic(asset)
                    self.assertEqual(loaded["source_sha256"], hashlib.sha256(original).hexdigest())
                    model.update(intrinsic_file=str(asset), intrinsic_sha256=loaded["source_sha256"])
                self.service.source.snapshot = replace(self.snapshot, camera_matrix=self.intrinsic.copy(), camera_model=model)
                self.service.mutate_sample("clear", self.current())
                self.service.freeze()
                request = self.point_request()
                expected = json.loads(json.dumps(next(iter(self.service.samples.active.values()))["model"]))
                self.service.source.snapshot.camera_matrix[0, 0] *= 2
                model["intrinsic_source"] = "changed-after-freeze"
                candidate = self.service.solve(request)
                if kind == "selected-file":
                    asset.write_text("replaced after load and solve")
                self.service.source.intrinsic_file = "/different.yaml"
                saved = self.service.save(candidate["candidate_id"])
                document = load_extrinsic(saved["output_file"])
                self.assertEqual(document["metadata"]["camera_model"], expected)
                self.assertEqual(saved["camera_model"], expected)
                self.assertEqual(expected["camera_matrix"], self.intrinsic.reshape(-1).tolist())
                self.assertNotIn("stamp_sec", expected)
                self.assertEqual(saved["points"][0]["pose_observation"]["source_stamp_sec"], self.snapshot.stamp_sec)
                restored = CalibrationService(self.service.source, calibration_root=str(self.calibration_root),
                    calibration_mode="sim", camera_name="usb_cam", parent_frame="map", child_frame="camera_optical_frame")
                self.assertEqual(restored.state()["result"]["camera_model"], expected)

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

    def test_arbitrary_bodies_recalibrate_and_publish_selection_without_restart(self):
        from xgc_camera_calibration.extrinsic_file_watcher import ExtrinsicSelectionWatcher
        names = ["calibration_stand", "ceiling_fixture", "desk_reference",
                 "reference_bar", "tripod", "wall_target"]
        snapshot = replace(self.snapshot, markers={
            name: MarkerObservation(name=name, position=tuple(position), frame_id="map")
            for name, position in zip(names, self.world)
        })
        source = FakeSource(snapshot)
        service = CalibrationService(source, calibration_root=str(self.calibration_root),
            calibration_mode="phy", camera_name="usb_cam", parent_frame="map",
            child_frame="camera_optical_frame", maximum_inlier_error_px=1.0)
        watcher = ExtrinsicSelectionWatcher(str(self.calibration_root), "phy", "usb_cam")
        paths = []
        revisions = []
        for translation in (self.tvec, self.tvec + np.array([0.2, 0.05, 0.1])):
            service.mutate_sample("clear", self.current(service))
            service.live()
            state = service.freeze()
            self.assertEqual([m["name"] for m in state["markers"]], names)
            pixels, _ = cv2.projectPoints(self.world, self.rvec, translation,
                                          self.intrinsic, self.distortion)
            candidate = service.solve(self.point_request(service, names, pixels.reshape(-1, 2)))
            self.assertIsNone(watcher.next_revision(), "Solve must not activate a candidate")
            saved = service.save(candidate["candidate_id"])
            revision = watcher.next_revision()
            self.assertEqual(str(revision.path), saved["output_file"])
            self.assertIsNone(watcher.next_revision())
            paths.append(revision.path)
            revisions.append(load_extrinsic(revision.path))
        self.assertNotEqual(paths[0], paths[1])
        self.assertTrue(all(path.is_file() for path in paths))
        self.assertNotEqual(revisions[0]["translation"], revisions[1]["translation"])

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
        self.assertTrue(restored.state()["result_restored"])
        self.assertEqual(restored.state()["result"]["candidate_id"], candidate["candidate_id"])
        self.assertEqual(restored.state()["samples"], [])

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
        request = self.point_request(service)
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

    def test_rejects_stale_revision_and_legacy_solve_authority(self):
        self.service.freeze()
        request = self.point_request()
        request["expected_revision"] -= 1
        with self.assertRaises(ApiError) as context:
            self.service.solve(request)
        self.assertEqual(context.exception.status, 409)

        request = self.point_request()
        request = {"generation": self.service.generation, "points": []}
        with self.assertRaises(ApiError) as context:
            self.service.solve(request)
        self.assertEqual(context.exception.status, 400)

    def test_one_body_at_independent_positions_solves_without_freeze(self):
        from xgc_camera_calibration.web_service import CalibrationError
        original = self.service.source.snapshot
        identities = []
        for index, (world, pixel) in enumerate(zip(self.world, self.pixels)):
            self.service.source.snapshot = replace(original, stamp_sec=10. + index,
                markers={"one-body": MarkerObservation("one-body", tuple(world), "map")})
            identities.append(self.admit("one-body", pixel))
        self.assertEqual(self.service.state()["mode"], "live")
        samples = self.service.state()["samples"]
        self.assertEqual(len(set(identities)), 6)
        self.assertEqual([sample["marker"] for sample in samples], ["one-body"] * 6)
        np.testing.assert_allclose([point["world"] for point in samples], self.world)
        self.assertEqual([p["pose_observation"]["source_stamp_sec"] for p in samples], list(np.arange(10., 16.)))
        # Solve never obtains latest pose, image, or camera model from source.
        self.service.source.observe_marker = lambda *args: self.fail("Solve recaptured a pose")
        self.service.source.freeze = lambda *args: self.fail("Solve captured a new image")
        result = self.service.solve(self.current())
        self.assertLess(result["max_reprojection_error_px"], 1e-3)
        self.assertEqual([p["sample_id"] for p in result["projections"]], identities)
        self.service.live()
        saved = self.service.save(result["candidate_id"])
        self.assertTrue(saved["saved"])
        self.assertEqual(load_extrinsic(saved["output_file"])["points"][0]["sample_id"], identities[0])

    def test_begin_fixes_pose_before_image_upload_and_exact_image_is_reviewable(self):
        request, pending = self.begin("marker_01", self.pixels[0])
        before = self.service.source.snapshot
        image_a = np.full_like(before.image, 25)
        encoded_a = cv2.imencode(".png", image_a)[1].tobytes()
        self.service.source.snapshot = replace(before, stamp_sec=99., image=np.full_like(before.image, 200),
            markers={"marker_01": MarkerObservation("marker_01", (50., 60., 70.), "map")})
        self.service.commit_sample(pending["sample_id"], encoded_a, "image/png")
        sample = self.service.state()["samples"][0]
        self.assertEqual(sample["world"], list(before.markers["marker_01"].position))
        self.assertEqual(sample["pose_observation"]["source_stamp_sec"], before.stamp_sec)
        self.assertEqual(sample["display"], request["display"])
        self.assertEqual(self.service.samples.image(pending["sample_id"]), (encoded_a, "image/png"))
        self.assertEqual(sample["image"]["sha256"], hashlib.sha256(encoded_a).hexdigest())
        self.service.freeze()
        self.service.live()
        self.assertEqual(self.service.state()["samples"], [sample])

    def test_sample_retry_cancel_expiry_and_failed_replacement_leave_original(self):
        original = self.admit("marker_01", self.pixels[0])
        old = self.service.state()["samples"]
        request, pending = self.begin("marker_01", self.pixels[1], replaces_sample_id=original)
        self.assertEqual(self.service.begin_sample(request)["sample_id"], pending["sample_id"])
        with self.assertRaises(ApiError):
            self.service.begin_sample({**request, "pixel": [5., 6.]})
        with self.assertRaises(ApiError):
            self.service.commit_sample(pending["sample_id"], b"bad", "image/png")
        self.assertEqual(self.service.state()["samples"], old)
        self.service.mutate_sample("cancel", {"sampling_session_id": self.service.samples.session_id, "sample_id": pending["sample_id"]})
        with self.assertRaises(ApiError):
            self.service.begin_sample(request)
        with self.assertRaises(ApiError):
            self.service.commit_sample(pending["sample_id"], b"bad", "image/png")
        _, expires = self.begin("marker_01", self.pixels[0])
        self.service.samples.pending[expires["sample_id"]]["expires"] = 0.
        with self.assertRaises(ApiError) as error:
            self.service.commit_sample(expires["sample_id"], b"bad", "image/png")
        self.assertEqual(error.exception.status, 404)
        self.assertEqual(self.service.state()["samples"], old)
        replacement = self.admit("marker_01", self.pixels[1], replaces_sample_id=original)
        self.assertNotEqual(replacement, original)
        self.assertEqual(len(self.service.state()["samples"]), 1)
        with self.assertRaises(ApiError):
            self.service.samples.image(original)

    def test_pixel_edit_undo_and_clear_invalidate_candidate_without_changing_other_samples(self):
        candidate = self.service.solve(self.point_request())
        before = self.service.state()["samples"]
        identity = before[0]["sample_id"]
        unchanged = self.current()
        self.service.mutate_sample("pixel", {**unchanged, "sample_id": identity, "pixel": before[0]["pixel"]})
        self.assertEqual(self.current(), unchanged)
        self.service.mutate_sample("pixel", {**self.current(), "sample_id": identity, "pixel": [20., 30.]})
        after = self.service.state()["samples"]
        self.assertEqual(after[1:], before[1:])
        self.assertEqual(after[0]["pose_observation"], before[0]["pose_observation"])
        self.assertEqual(after[0]["image"], before[0]["image"])
        with self.assertRaises(ApiError):
            self.service.save(candidate["candidate_id"])
        with self.assertRaises(ApiError):
            self.service.mutate_sample("remove", {**unchanged, "sample_id": identity})
        self.service.mutate_sample("remove", {**self.current(), "sample_id": before[-1]["sample_id"]})
        self.assertEqual(len(self.service.state()["samples"]), 5)
        self.service.mutate_sample("clear", self.current())
        revision = self.current()
        self.service.mutate_sample("clear", revision)
        self.assertEqual(self.current(), revision)
        self.assertEqual(self.service.state()["samples"], [])

    def test_concurrent_commits_compare_the_same_dataset_revision_atomically(self):
        request_a, a = self.begin("marker_01", self.pixels[0])
        _, b = self.begin("marker_02", self.pixels[1])
        encoded = cv2.imencode(".png", self.snapshot.image)[1].tobytes()
        barrier = threading.Barrier(3)
        results = []
        def commit(identity):
            barrier.wait()
            try:
                self.service.commit_sample(identity, encoded, "image/png")
                results.append(200)
            except ApiError as error:
                results.append(error.status)
        threads = [threading.Thread(target=commit, args=(item["sample_id"],)) for item in (a,b)]
        for thread in threads: thread.start()
        barrier.wait()
        for thread in threads: thread.join(3)
        self.assertEqual(sorted(results), [200,409])
        state = self.service.state()
        self.assertEqual(state["dataset_revision"], 1)
        identity = state["samples"][0]["sample_id"]
        self.service.commit_sample(identity, encoded, "image/png")
        self.assertEqual(self.service.state()["dataset_revision"], 1)
        with self.assertRaises(ApiError):
            self.service.commit_sample(identity, encoded + b"changed", "image/png")

    def test_model_source_and_pixel_boundaries_reject_without_dropping_samples(self):
        self.admit("marker_01", self.pixels[0])
        before = self.service.state()["samples"]
        request, pending = self.begin("marker_02", self.pixels[1])
        for bad in (
            {**request, "request_id": uuid.uuid4().hex, "pixel": [float("nan"), 1]},
            {**request, "request_id": uuid.uuid4().hex, "pixel": [640, 2]},
            {**request, "request_id": uuid.uuid4().hex, "sampling_session_id": "other"},
            {**request, "request_id": uuid.uuid4().hex, "display": {**request["display"], "source_id": "other"}},
        ):
            with self.assertRaises(ApiError): self.service.begin_sample(bad)
        self.service.source.snapshot = replace(self.snapshot, camera_matrix=self.intrinsic * 2)
        with self.assertRaisesRegex(ApiError, "model"):
            self.begin("marker_02", self.pixels[1])
        self.assertEqual(self.service.state()["samples"], before)

    def test_one_body_coincident_samples_are_not_fabricated_geometric_information(self):
        for _ in range(4): self.admit("marker_01", self.pixels[0])
        with self.assertRaises(ApiError) as error: self.service.solve(self.current())
        self.assertEqual(error.exception.status, 422)
        self.assertIsNone(self.service.result_payload)

    def test_image_headers_limit_decode_allocation_and_preserve_native_jpeg(self):
        from xgc_camera_calibration.extrinsic_samples import IMAGE_BYTES
        import struct
        _, pending = self.begin("marker_01", self.pixels[0])
        huge = b"\x89PNG\r\n\x1a\n\x00\x00\x00\x0dIHDR" + struct.pack(">II",8192,8192) + b"0" * 16
        with patch("xgc_camera_calibration.extrinsic_samples.cv2.imdecode", side_effect=AssertionError("must reject before decode")):
            with self.assertRaises(ApiError) as error: self.service.commit_sample(pending["sample_id"], huge, "image/png")
            self.assertEqual(error.exception.status, 413)
        native = cv2.imencode(".jpg", self.snapshot.image)[1].tobytes()
        self.service.commit_sample(pending["sample_id"], native, "image/jpeg")
        self.assertEqual(self.service.samples.image(pending["sample_id"]), (native,"image/jpeg"))

    def test_real_ros_callback_retains_both_clocks_and_converts_coordinates_once(self):
        import ast
        from types import SimpleNamespace, MethodType
        from xgc_camera_calibration.pose_freshness import pose_is_fresh
        from xgc_camera_calibration.intrinsic_validation import ideal_intrinsic_parameters
        source_path = Path(__file__).resolve().parents[1] / "scripts" / "extrinsic_calibrator_web.py"
        cls = next(node for node in ast.parse(source_path.read_text()).body if isinstance(node, ast.ClassDef) and node.name == "RosCalibrationSource")
        methods = [node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name in ("_marker_callback", "observe_marker", "_observation_context")]
        clock = SimpleNamespace(monotonic=lambda: 100., time=lambda: 123456.)
        namespace = {"MarkerObservation": MarkerObservation, "ApiError": ApiError, "replace": replace,
            "uuid": uuid, "time": clock, "pose_is_fresh": pose_is_fresh,
            "coordinate_provenance": coordinate_provenance, "ideal_intrinsic_parameters": ideal_intrinsic_parameters,
            "rospy": SimpleNamespace(Time=SimpleNamespace(now=lambda:SimpleNamespace(to_sec=lambda:12.34)))}
        exec(compile(ast.Module(body=methods,type_ignores=[]),str(source_path),"exec"),namespace)
        source = SimpleNamespace(lock=threading.RLock(),marker_latest={},marker_receipts={},
            source_size=(640,480),snapshot_client=object(),snapshot_available=True,pose_max_age=2.,
            use_ideal_intrinsics=False,intrinsic_matrix=self.intrinsic,intrinsic_distortion=self.distortion,
            intrinsic_size=(640,480),intrinsic_provenance={},pose_coordinate_source="raw-vrpn",pose_world_offset=(10.,-5.,2.))
        source._observation_context=MethodType(namespace["_observation_context"],source)
        callback=namespace["_marker_callback"](source,"one")
        def message(position):
            return SimpleNamespace(pose=SimpleNamespace(position=SimpleNamespace(x=position[0],y=position[1],z=position[2])),
                header=SimpleNamespace(frame_id="map",stamp=SimpleNamespace(to_sec=lambda:12.34)))
        callback(message((1.,2.,3.)))
        first=namespace["observe_marker"](source,"one",640,480,"map")["marker"]
        callback(message((4.,5.,6.)))
        second=namespace["observe_marker"](source,"one",640,480,"map")["marker"]
        self.assertEqual(first.position,(11.,-3.,5.));self.assertEqual(first.source_position,(1.,2.,3.))
        self.assertEqual(second.position,(14.,0.,8.));self.assertNotEqual(first.observation_id,second.observation_id)
        self.assertEqual((first.source_stamp_sec,first.received_at_sec,first.received_monotonic_sec),(12.34,123456.,100.))
        clock.monotonic=lambda:104.
        with self.assertRaisesRegex(ApiError,"fresh"):
            namespace["observe_marker"](source,"one",640,480,"map")

    def test_http_native_4k_image_commit_review_and_tamper_rejection(self):
        # Real service transport, real codec decode and native >2MiB image; no ROS.
        image=np.random.default_rng(17).integers(0,256,(2160,3840,3),dtype=np.uint8)
        native=cv2.imencode(".jpg",image,[cv2.IMWRITE_JPEG_QUALITY,90])[1].tobytes()
        self.assertGreater(len(native),2<<20)
        self.service.source.snapshot=replace(self.snapshot,image=image)
        server=CalibrationHttpServer(("127.0.0.1",0),self.service,
            Path(__file__).resolve().parents[1]/"web"/"extrinsic",frame_ancestors="'self'")
        thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
        base="http://127.0.0.1:{}".format(server.server_address[1])
        try:
            request,unused=self.begin("marker_01",[100.,200.])
            self.service.mutate_sample("cancel",{"sampling_session_id":self.service.samples.session_id,"sample_id":unused["sample_id"]})
            request["request_id"]=uuid.uuid4().hex
            def post(path,body,mime):
                return urllib.request.urlopen(urllib.request.Request(base+path,data=body,
                    headers={"Content-Type":mime},method="POST"),timeout=5)
            with post("/api/v1/samples/begin",json.dumps(request).encode(),"application/json") as response:
                pending=json.load(response)
            path="/api/v1/samples/"+pending["sample_id"]+"/image"
            with post(path,native,"image/jpeg") as response:
                state=json.load(response)
                self.assertEqual(state["dataset_revision"],1)
            with urllib.request.urlopen(base+path,timeout=5) as response:
                self.assertEqual(response.headers["Content-Type"],"image/jpeg")
                self.assertEqual(response.headers["Cache-Control"],"no-store")
                self.assertEqual(response.read(),native)
            with self.assertRaises(urllib.error.HTTPError) as error:
                post(path,native+b"different","image/jpeg")
            self.assertEqual(error.exception.code,409)
            with self.assertRaises(urllib.error.HTTPError) as error:
                post(path,b"not an image","application/octet-stream")
            self.assertEqual(error.exception.code,415)
            with self.assertRaises(urllib.error.HTTPError) as error:
                post("/api/v1/solve",b" "*(128*1024+1),"application/json")
            self.assertEqual(error.exception.code,413)
            self.assertEqual(self.service.samples.revision,1)
        finally:
            server.shutdown();server.server_close();thread.join(3)

    def test_upload_has_an_absolute_deadline_and_sample_actions_validate_identity(self):
        from types import SimpleNamespace
        from unittest.mock import Mock
        from xgc_camera_calibration.web_service import CalibrationRequestHandler
        handler = object.__new__(CalibrationRequestHandler)
        handler.path = "/api/v1/samples/" + "a" * 32 + "/image"
        handler.command = "POST"
        handler.headers = {"Content-Type": "image/png", "Content-Length": "2"}
        handler.connection = Mock()
        handler.connection.gettimeout.return_value = None
        handler.rfile = Mock()
        handler.rfile.read1.return_value = b"a"
        handler.close_connection = False
        with patch("xgc_camera_calibration.web_service.time.monotonic", side_effect=[0.,0.,31.]):
            with self.assertRaises(ApiError) as error: handler._dispatch()
        self.assertEqual(error.exception.status, 408)
        self.assertTrue(handler.close_connection)
        self.assertEqual(handler.rfile.read1.call_count,1)
        handler.connection.settimeout.assert_called_with(None)
        for action in ("pixel", "remove", "cancel"):
            body = {**self.current(),"sample_id":[]}
            if action == "pixel": body["pixel"] = [1,2]
            if action == "cancel": del body["expected_revision"]
            with self.assertRaises(ApiError) as error: self.service.mutate_sample(action,body)
            self.assertEqual(error.exception.status,400)

    def test_browser_capture_clock_can_precede_the_page_time_origin(self):
        request, _ = self.begin("marker_01", self.pixels[0])
        request["request_id"] = uuid.uuid4().hex
        request["display"]["capture_time_ms"] = -25.5
        pending = self.service.begin_sample(request)
        self.assertEqual(pending["status"], "pending")
        self.assertEqual(self.service.samples.pending[pending["sample_id"]]["sample"]["display"]["capture_time_ms"], -25.5)

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
