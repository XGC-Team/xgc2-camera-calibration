"""Instance-fenced native multipart capture; no legacy browser snapshot routes."""
import json
import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path

import cv2
import numpy as np
from xgc2_xrpc import Host, Limits, Response, Runtime, multipart
from xgc_camera_calibration.media_snapshot import MediaSnapshotClient, MediaSnapshotError


class FakeMediaEdge:
    def __init__(self):
        self.requests, self.deleted = [], []
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
        self.path = str(Path(self.directory.name) / "edge.sock")
        self.runtime = Runtime()
        ref = {"target_id": "test-target", "service": "media-edge", "api_version": "v1",
               "instance_id": "edge-1", "profile": "http.v1",
               "endpoint": {"kind": "unix", "address": self.path}}
        def capture(context, value):
            self.requests.append((context.request_id, value))
            if self.custom_capture:
                return self.custom_capture(value)
            raw = self.raw if value["includeRgb"] else b""
            metadata = {**self.metadata, "snapshotId": value["snapshotId"],
                "jpegBytes": len(self.jpeg), "rgbBytes": len(raw) + self.announced_rgb_delta}
            return multipart(metadata, self.jpeg, raw)
        def release(context, value):
            self.deleted.append(context.request_id)
            return Response(b"", 204)
        # Runtime Host resolves explicit routes; capture identities are known
        # only after admission, so extend the release route in this fixture.
        def capture_with_release(context, value):
            self.host.routes[("DELETE", "/v1/media/snapshots/" + value["snapshotId"])] = release
            return capture(context, value)
        routes = {("GET", "/v1/describe"): lambda c, r: {"service_ref": ref, "sources": self.sources},
                  ("GET", "/v1/health"): lambda c, r: {"sources": self.sources},
                  ("POST", "/v1/media/sources/usb_cam/capture"): capture_with_release}
        self.host = Host(self.path, routes, runtime=self.runtime, instance_id="edge-1",
                         discovery_routes=("/v1/describe",), limits=Limits(response_bytes=40 << 20)).start()
        self.client = MediaSnapshotClient(self.path, "usb_cam", local_target="test-target")
        return self

    def __exit__(self, *args):
        self.client.close()
        self.host.close()
        self.runtime.close()
        self.directory.cleanup()


class MediaSnapshotClientTest(unittest.TestCase):
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
            self.assertEqual(len(edge.deleted), 1)
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
            self.assertEqual(edge.client.last_cleanup_error, "TransportError")

    def test_health_requires_configured_source_and_bootstrap_rejects_http(self):
        with FakeMediaEdge() as edge:
            edge.sources.clear()
            with self.assertRaises(MediaSnapshotError): edge.client.health()
        for address in ("http://localhost:18090", "relative.sock"):
            with self.assertRaises(ValueError): MediaSnapshotClient(address, "usb_cam")
        with self.assertRaises(ValueError): MediaSnapshotClient("/private/edge.sock", "bad source")
        with self.assertRaises(ValueError): MediaSnapshotClient("/private/edge.sock", "usb_cam", float("inf"))

    def test_extra_or_encoded_mime_parts_are_rejected(self):
        with FakeMediaEdge() as edge:
            def malformed(value):
                body = b'--b\r\nContent-Type: application/json\r\nContent-Disposition: inline; name="metadata"\r\nContent-Transfer-Encoding: base64\r\n\r\ne30=\r\n--b--\r\n'
                return Response(body, content_type="multipart/mixed; boundary=b")
            edge.custom_capture = malformed
            with self.assertRaises(MediaSnapshotError): edge.client.capture()
            self.assertEqual(len(edge.deleted), 1)


if __name__ == "__main__": unittest.main()
