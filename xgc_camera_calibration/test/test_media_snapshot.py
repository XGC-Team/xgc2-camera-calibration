"""Instance-fenced native multipart capture; no legacy browser snapshot routes."""
import json
import tempfile
import time
import unittest
import uuid
from dataclasses import asdict
from pathlib import Path

import cv2
import numpy as np
from xgc2_xrpc import Fault, Host, Limits, Response, Runtime, multipart
from xgc_camera_calibration.media_snapshot import MediaSnapshotClient, MediaSnapshotError


class FakeMediaEdge:
    def __init__(self, path=None, instance="edge-1"):
        self.provided_path = path
        self.instance = instance
        self.requests, self.deleted, self.snapshot_ids = [], [], []
        self.sources = [{"id": "usb_cam"}]
        self.image = np.full((216, 384, 3), (10, 20, 30), dtype=np.uint8)
        self.jpeg = cv2.imencode(".jpg", self.image)[1].tobytes()
        self.raw = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB).tobytes()
        self.metadata = {"sourceId": "usb_cam", "frameId": "usb_cam_optical_frame",
            "timestampNanoseconds": 123456789, "timestampClockDomain": "simulation",
            "frameSequence": 1, "width": 384, "height": 216, "pixelFormat": "rgb8",
            "cameraMatrix": [100., 0., 192., 0., 101., 108., 0., 0., 1.],
            "distortion": [0.1, -0.2, 0.01, -0.01, 0.], "calibrationState": "available",
            "poseFrameId": "world", "renderPose": {"position": {"x": 1.2, "y": -.3, "z": 2.1},
            "orientation": {"x": 0., "y": 0., "z": 0., "w": 1.}}}
        self.announced_rgb_delta = 0
        self.custom_capture = None

    def __enter__(self):
        self.directory = tempfile.TemporaryDirectory()
        self.path = self.provided_path or str(Path(self.directory.name) / "edge.sock")
        self.runtime = Runtime()
        ref = {"target_id": "test-target", "service": "media-edge", "api_version": "v1",
               "instance_id": self.instance, "profile": "http.v1",
               "endpoint": {"kind": "unix", "address": self.path}}
        def capture(context, value, snapshot_id):
            self.requests.append((context.request_id, value))
            if self.custom_capture:
                return self.custom_capture(value)
            raw = self.raw if value["includeRgb"] else b""
            metadata = {**self.metadata, "snapshotId": snapshot_id,
                "jpegBytes": len(self.jpeg), "rgbBytes": len(raw) + self.announced_rgb_delta}
            return multipart(metadata, self.jpeg, raw)
        # The provider admits and owns the snapshot; the request does not
        # choose its identity. Release the exact identity returned in metadata.
        def capture_with_release(context, value):
            snapshot_id = uuid.uuid4().hex
            self.snapshot_ids.append(snapshot_id)
            def release(context, value):
                self.deleted.append(snapshot_id)
                return Response(b"", 204)
            self.host.routes[("DELETE", "/v1/media/snapshots/" + snapshot_id)] = release
            return capture(context, value, snapshot_id)
        routes = {("GET", "/v1/describe"): lambda c, r: {"ready": True, "service_ref": ref, "sources": self.sources},
                  ("GET", "/v1/health"): lambda c, r: {"sources": self.sources},
                  ("POST", "/v1/media/sources/usb_cam/capture"): capture_with_release}
        self.host = Host(self.path, routes, runtime=self.runtime, instance_id=self.instance,
                         discovery_routes=("/v1/describe",), limits=Limits(response_bytes=40 << 20)).start()
        self.client = MediaSnapshotClient(self.path, "usb_cam", local_target="test-target")
        self.client.peer.activate()
        deadline = time.monotonic() + 3
        while not self.client.status()["ready"] and time.monotonic() < deadline:
            time.sleep(.01)
        if not self.client.status()["ready"]:
            self.__exit__()
            raise AssertionError("fake media discovery did not complete")
        return self

    def __exit__(self, *args):
        self.client.close()
        self.host.close()
        self.runtime.close()
        self.directory.cleanup()


class MediaSnapshotClientTest(unittest.TestCase):
    def test_first_capture_discovers_peer_without_an_events_subscription(self):
        with FakeMediaEdge() as edge:
            cold = MediaSnapshotClient(edge.path, "usb_cam", local_target="test-target")
            try:
                self.assertEqual(cold.status()["reason"], "not_requested")
                snapshot = cold.capture_detection()
                self.assertEqual(snapshot.jpeg, edge.jpeg)
                self.assertEqual(len(edge.requests), 1)
                self.assertEqual(edge.deleted, [snapshot.id])
                self.assertEqual(cold.status()["connect_attempts"], 1)
            finally:
                cold.close()

    def test_missing_peer_capture_has_a_finite_wait_and_no_side_effect(self):
        with tempfile.TemporaryDirectory() as directory:
            cold = MediaSnapshotClient(str(Path(directory) / "absent.sock"), "usb_cam",
                                       timeout_seconds=.05, local_target="test-target")
            try:
                start = time.monotonic()
                with self.assertRaises(MediaSnapshotError):
                    cold.capture_detection()
                self.assertLess(time.monotonic() - start, 1)
                self.assertEqual(cold.status()["connect_attempts"], 0)
            finally:
                cold.close()

    def test_source_failure_is_status_without_invalidating_media_service(self):
        with FakeMediaEdge() as edge:
            edge.client.capture()
            self.assertTrue(edge.client.status()["source_available"])
            def absent(_value):
                raise Fault("unavailable", "camera source is absent", 503)
            edge.custom_capture = absent
            with self.assertRaises(MediaSnapshotError):
                edge.client.capture()
            self.assertTrue(edge.client.status()["ready"])
            self.assertFalse(edge.client.status()["source_available"])
            self.assertIn("camera source is absent", edge.client.status()["capture_error"])
            edge.custom_capture = None
            edge.client.capture()
            self.assertTrue(edge.client.status()["source_available"])
            self.assertEqual(edge.client.status()["connect_attempts"], 1)

    def test_health_and_capture_consume_one_immutable_native_frame(self):
        with FakeMediaEdge() as edge:
            self.assertEqual(edge.client.health()["sources"], edge.sources)
            snapshot = edge.client.capture()
            np.testing.assert_array_equal(snapshot.bgr, edge.image)
            self.assertEqual(snapshot.jpeg, edge.jpeg)
            self.assertEqual(snapshot.render_position, (1.2, -.3, 2.1))
            self.assertEqual(snapshot.pose_frame_id, "world")
            self.assertEqual(snapshot.timestamp_clock_domain, "simulation")
            self.assertEqual(snapshot.frame_sequence, 1)
            self.assertEqual(len(edge.requests), 1)
            self.assertEqual(edge.deleted, [snapshot.id])
            self.assertEqual(edge.snapshot_ids, [snapshot.id])
            self.assertNotIn("snapshotId", edge.requests[0][1])
            self.assertTrue(edge.requests[0][1]["requireFresh"])

    def test_detection_capture_is_same_frame_jpeg_only_and_preserves_source_plane(self):
        with FakeMediaEdge() as edge:
            snapshot = edge.client.capture_detection(maximum_pixels=4096)
            self.assertEqual((snapshot.width, snapshot.height), (384, 216))
            self.assertLessEqual(snapshot.bgr.shape[0] * snapshot.bgr.shape[1], 4096)
            self.assertFalse(edge.requests[0][1]["includeRgb"])
            self.assertEqual(snapshot.jpeg, edge.jpeg)
            np.testing.assert_array_equal(snapshot.camera_matrix.reshape(-1), edge.metadata["cameraMatrix"])

    def test_size_mismatch_or_invalid_frame_releases_exact_snapshot(self):
        for corrupt in ("length", "pixel", "clock", "sequence", "pose"):
            with self.subTest(corrupt=corrupt), FakeMediaEdge() as edge:
                if corrupt == "length": edge.announced_rgb_delta = 1
                if corrupt == "pixel": edge.metadata["pixelFormat"] = "bgr8"
                if corrupt == "clock": edge.metadata["timestampClockDomain"] = "invented"
                if corrupt == "sequence": edge.metadata["frameSequence"] = 0
                if corrupt == "pose": edge.metadata.pop("poseFrameId")
                with self.assertRaises(MediaSnapshotError): edge.client.capture()
                self.assertEqual(edge.deleted, edge.snapshot_ids)
                self.assertEqual(len(edge.deleted), 1)

    def test_simulation_zero_time_and_unavailable_calibration_are_preserved(self):
        with FakeMediaEdge() as edge:
            edge.metadata.update(timestampNanoseconds=0, calibrationState="unavailable",
                                 cameraMatrix=[], distortion=[])
            snapshot = edge.client.capture()
            self.assertEqual(snapshot.timestamp_nanoseconds, 0)
            self.assertIsNone(snapshot.camera_matrix)
            self.assertIsNone(snapshot.distortion)

    def test_bound_instance_conflict_is_not_rediscovered_or_replayed(self):
        with FakeMediaEdge() as edge:
            edge.host.instance_id = "edge-2"
            with self.assertRaises(MediaSnapshotError): edge.client.capture()
            self.assertEqual(edge.requests, [])
            self.assertEqual(edge.deleted, [])
            self.assertIsNone(edge.client.last_cleanup_error)

    def test_health_requires_configured_source_and_bootstrap_rejects_http(self):
        with FakeMediaEdge() as edge:
            edge.sources.clear()
            with self.assertRaises(MediaSnapshotError): edge.client.health()
        for address in ("http://localhost:18090", "relative.sock"):
            with self.assertRaises(ValueError): MediaSnapshotClient(address, "usb_cam", local_target="test-target")
        with self.assertRaises(ValueError): MediaSnapshotClient("/private/edge.sock", "bad source", local_target="test-target")
        with self.assertRaises(ValueError): MediaSnapshotClient("/private/edge.sock", "usb_cam", float("inf"), local_target="test-target")

    def test_extra_or_encoded_mime_parts_are_rejected(self):
        with FakeMediaEdge() as edge:
            def malformed(value):
                body = b'--b\r\nContent-Type: application/json\r\nContent-Disposition: inline; name="metadata"\r\nContent-Transfer-Encoding: base64\r\n\r\ne30=\r\n--b--\r\n'
                return Response(body, content_type="multipart/mixed; boundary=b")
            edge.custom_capture = malformed
            with self.assertRaises(MediaSnapshotError): edge.client.capture()
            self.assertEqual(len(edge.requests), 1)
            self.assertEqual(edge.deleted, [])
            self.assertIsNone(edge.client.last_cleanup_error)


if __name__ == "__main__": unittest.main()
