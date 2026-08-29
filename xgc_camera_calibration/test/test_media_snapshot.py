#!/usr/bin/env python3

import json
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np

from xgc_camera_calibration.media_snapshot import (
    MediaSnapshotClient,
    MediaSnapshotError,
)


class FakeMediaEdge:
    def __init__(self):
        self.requests = []
        self.deleted = []
        self.sources = [{"id": "usb_cam"}]
        self.corrupt_raw_headers = False
        self.jpeg = b"\xff\xd8xgc2-snapshot\xff\xd9"
        self.raw = bytes([10, 20, 30]) * (16 * 16)
        self.metadata = {
            "snapshotId": "snapshot-1",
            "sourceId": "usb_cam",
            "frameId": "usb_cam_optical_frame",
            "timestampNanoseconds": 123456789,
            "width": 16,
            "height": 16,
            "pixelFormat": "rgb8",
            "cameraMatrix": [
                100.0, 0.0, 8.0,
                0.0, 101.0, 8.0,
                0.0, 0.0, 1.0,
            ],
            "distortion": [0.1, -0.2, 0.01, -0.01, 0.0],
            "renderPose": {
                "position": {"x": 1.2, "y": -0.3, "z": 2.1},
                "orientation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0},
            },
        }
        self.server = None
        self.thread = None

    def __enter__(self):
        edge = self

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                edge._record(self)
                if self.path == "/healthz":
                    self._json({"sources": edge.sources})
                    return
                if self.path == "/api/v1/snapshots/snapshot-1/jpeg":
                    self._reply(200, edge.jpeg, "image/jpeg")
                    return
                if self.path == "/api/v1/snapshots/snapshot-1/raw":
                    width = "17" if edge.corrupt_raw_headers else "16"
                    self._reply(
                        200,
                        edge.raw,
                        "application/x-xgc-rgb8",
                        {
                            "X-Xgc-Snapshot-Id": "snapshot-1",
                            "X-Xgc-Frame-Id": "usb_cam_optical_frame",
                            "X-Xgc-Width": width,
                            "X-Xgc-Height": "16",
                        },
                    )
                    return
                self._json({"error": "not found"}, status=404)

            def do_POST(self):
                body = edge._record(self)
                if (
                    self.path == "/api/v1/sources/usb_cam/snapshots"
                    and body == b"{}"
                ):
                    self._json(edge.metadata)
                    return
                self._json({"error": "invalid capture request"}, status=400)

            def do_DELETE(self):
                edge._record(self)
                if self.path == "/api/v1/snapshots/snapshot-1":
                    edge.deleted.append("snapshot-1")
                    self._reply(204, b"", "application/octet-stream")
                    return
                self._json({"error": "not found"}, status=404)

            def _json(self, payload, status=200):
                self._reply(
                    status,
                    json.dumps(payload).encode("utf-8"),
                    "application/json",
                )

            def _reply(self, status, payload, content_type, headers=None):
                self.send_response(status)
                self.send_header("Content-Type", content_type)
                self.send_header("Content-Length", str(len(payload)))
                for name, value in (headers or {}).items():
                    self.send_header(name, value)
                self.end_headers()
                if payload:
                    self.wfile.write(payload)

            def log_message(self, _format, *_args):
                return

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.thread = threading.Thread(
            target=self.server.serve_forever,
            name="fake-media-edge",
            daemon=True,
        )
        self.thread.start()
        return self

    def __exit__(self, _exception_type, _exception, _traceback):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=5.0)

    @property
    def address(self):
        return "http://127.0.0.1:{}".format(self.server.server_address[1])

    def _record(self, handler):
        length = int(handler.headers.get("Content-Length", "0"))
        body = handler.rfile.read(length) if length else b""
        self.requests.append((handler.command, handler.path, body))
        return body


class MediaSnapshotClientTest(unittest.TestCase):
    def test_health_and_capture_consume_one_immutable_snapshot(self):
        with FakeMediaEdge() as edge:
            client = MediaSnapshotClient(edge.address + "/", "usb_cam", 1.0)

            health = client.health()
            snapshot = client.capture()

        self.assertEqual(health["sources"], [{"id": "usb_cam"}])
        self.assertEqual(snapshot.id, "snapshot-1")
        self.assertEqual(snapshot.source_id, "usb_cam")
        self.assertEqual(snapshot.frame_id, "usb_cam_optical_frame")
        self.assertEqual(snapshot.timestamp_nanoseconds, 123456789)
        self.assertEqual((snapshot.width, snapshot.height), (16, 16))
        self.assertEqual(snapshot.jpeg, edge.jpeg)
        self.assertEqual(snapshot.bgr.shape, (16, 16, 3))
        self.assertEqual(snapshot.bgr[0, 0].tolist(), [30, 20, 10])
        self.assertEqual(snapshot.render_position, (1.2, -0.3, 2.1))
        self.assertEqual(snapshot.render_orientation, (0.0, 0.0, 0.0, 1.0))
        np.testing.assert_allclose(
            snapshot.camera_matrix,
            np.asarray(edge.metadata["cameraMatrix"]).reshape(3, 3),
        )
        np.testing.assert_allclose(
            snapshot.distortion,
            np.asarray(edge.metadata["distortion"]),
        )
        self.assertEqual(edge.deleted, ["snapshot-1"])
        self.assertEqual(
            [(method, path) for method, path, _body in edge.requests],
            [
                ("GET", "/healthz"),
                ("POST", "/api/v1/sources/usb_cam/snapshots"),
                ("GET", "/api/v1/snapshots/snapshot-1/jpeg"),
                ("GET", "/api/v1/snapshots/snapshot-1/raw"),
                ("DELETE", "/api/v1/snapshots/snapshot-1"),
            ],
        )

    def test_capture_deletes_snapshot_after_raw_metadata_validation_fails(self):
        with FakeMediaEdge() as edge:
            edge.corrupt_raw_headers = True
            client = MediaSnapshotClient(edge.address, "usb_cam", 1.0)

            with self.assertRaisesRegex(MediaSnapshotError, "metadata does not match"):
                client.capture()

        self.assertEqual(edge.deleted, ["snapshot-1"])
        self.assertEqual(edge.requests[-1][:2], ("DELETE", "/api/v1/snapshots/snapshot-1"))

    def test_capture_accepts_simulation_epoch_timestamp_zero(self):
        with FakeMediaEdge() as edge:
            edge.metadata["timestampNanoseconds"] = 0
            client = MediaSnapshotClient(edge.address, "usb_cam", 1.0)

            snapshot = client.capture()

        self.assertEqual(snapshot.timestamp_nanoseconds, 0)
        self.assertEqual(edge.deleted, ["snapshot-1"])

    def test_capture_rejects_a_non_rgb_snapshot_contract(self):
        with FakeMediaEdge() as edge:
            edge.metadata["pixelFormat"] = "bgr8"
            client = MediaSnapshotClient(edge.address, "usb_cam", 1.0)

            with self.assertRaisesRegex(MediaSnapshotError, "pixel format"):
                client.capture()

        self.assertEqual(edge.deleted, ["snapshot-1"])

    def test_health_requires_the_configured_source(self):
        with FakeMediaEdge() as edge:
            edge.sources = [{"id": "rear"}]
            client = MediaSnapshotClient(edge.address, "usb_cam", 1.0)

            with self.assertRaisesRegex(MediaSnapshotError, "source is unavailable"):
                client.health()

    def test_rejects_remote_addresses_unstable_ids_and_invalid_timeouts(self):
        invalid_addresses = [
            "https://127.0.0.1:18090",
            "http://192.0.2.20:18090",
            "http://operator:secret@127.0.0.1:18090",
            "http://127.0.0.1:18090/api",
            "http://127.0.0.1:18090?source=usb_cam",
            "http://127.0.0.1:18090#source",
        ]
        for address in invalid_addresses:
            with self.subTest(address=address):
                with self.assertRaises(ValueError):
                    MediaSnapshotClient(address, "usb_cam")
        for source_id in ("", ".hidden", "../camera", "front/camera"):
            with self.subTest(source_id=source_id):
                with self.assertRaisesRegex(ValueError, "stable identifier"):
                    MediaSnapshotClient("http://127.0.0.1:18090", source_id)
        for timeout in (0, -1, float("nan"), float("inf")):
            with self.subTest(timeout=timeout):
                with self.assertRaisesRegex(ValueError, "timeout must be positive"):
                    MediaSnapshotClient(
                        "http://127.0.0.1:18090",
                        "usb_cam",
                        timeout,
                    )


if __name__ == "__main__":
    unittest.main()
