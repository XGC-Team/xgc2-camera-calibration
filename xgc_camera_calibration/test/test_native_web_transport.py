"""Real aiohttp edges: slow bodies, disconnects and fixed worker ownership."""
import json
import socket
import tempfile
import threading
import time
import unittest
import urllib.error
import urllib.request
from pathlib import Path

from xgc2_xrpc import Client, Endpoint, Fault, Host, Limits, Runtime, ServiceRef
from xgc_camera_calibration.media_snapshot import CameraMetadataClient, MediaSnapshotClient
from xgc_camera_calibration.intrinsic_service import IntrinsicCalibrationService
from xgc_camera_calibration.intrinsic_solver import IntrinsicResult, save_intrinsic
from xgc_camera_calibration.solver import selected_intrinsic_path
from xgc_camera_calibration.web_service import CalibrationHttpServer


class NativeWebTransportTest(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.root = Path(self.directory.name)
        for name in ("index.html", "app.js", "styles.css"):
            (self.root / name).write_text("static asset")
        self.servers = []

    def tearDown(self):
        for server in reversed(self.servers):
            server.close()
        self.directory.cleanup()

    def server(self, service=None, **kwargs):
        server = CalibrationHttpServer(("127.0.0.1", 0), service, self.root,
                                      frame_ancestors="'self'", **kwargs).start()
        self.servers.append(server)
        return server

    def url(self, server, path):
        return "http://127.0.0.1:{}{}".format(server.server_address[1], path)

    def test_slow_upload_deadline_and_unread_body_are_closed_without_mutation(self):
        class Service:
            def commit_sample(self, *args):
                raise AssertionError("incomplete body must not be committed")
        server = self.server(Service(), upload_timeout=0.04)
        with socket.create_connection(server.server_address, timeout=1) as peer:
            peer.sendall(b"POST /api/v1/samples/" + b"a" * 32 + b"/image HTTP/1.1\r\n"
                b"Host: local\r\nContent-Type: image/png\r\nContent-Length: 2\r\n\r\na")
            started = time.monotonic()
            received = b""
            while True:
                chunk = peer.recv(4096)
                if not chunk:
                    break
                received += chunk
            self.assertLess(time.monotonic() - started, 0.5)
            self.assertIn(b" 408 ", received)
            self.assertEqual(received.count(b"HTTP/1.1"), 1)

    def test_head_json_validation_cors_and_state_stream_disconnect(self):
        class Intrinsic:
            def state(self):
                return {"phase": "collecting", "image_ready": True}
        server = self.server(intrinsic_service=Intrinsic(), allowed_origins=["http://viewer"])
        request = urllib.request.Request(self.url(server, "/"), method="HEAD",
                                         headers={"Origin": "http://viewer"})
        with urllib.request.urlopen(request, timeout=1) as response:
            self.assertEqual(response.read(), b"")
            self.assertEqual(response.headers["Content-Length"], "12")
            self.assertEqual(response.headers["Access-Control-Allow-Origin"], "http://viewer")
        request = urllib.request.Request(self.url(server, "/api/v1/intrinsic/candidate"),
            data=b'{"x":1,"x":2}', headers={"Content-Type": "application/json"})
        with self.assertRaises(urllib.error.HTTPError) as error:
            urllib.request.urlopen(request, timeout=1)
        self.assertEqual(error.exception.code, 400)
        response = urllib.request.urlopen(self.url(server, "/api/v1/intrinsic/events"), timeout=1)
        self.assertEqual(response.readline(), b"id: 1\n")
        response.close()
        server.close()
        self.assertTrue(server.runtime.closed)
        self.assertEqual(len(server.host._jobs), 0)
        self.assertEqual(len(server.host._active), 0)

    def test_cancelled_work_keeps_slot_until_real_completion_and_close_can_retry(self):
        entered, release, complete = threading.Event(), threading.Event(), threading.Event()
        class Service:
            def freeze(self):
                entered.set()
                release.wait(2)
                complete.set()
                return {"mode": "frozen"}
        runtime = Runtime(blocking_workers=1, max_calls=4, max_connections=8)
        server = self.server(Service(), runtime=runtime,
            limits=Limits(shutdown_timeout=0.03, in_flight=4, connections=8))
        peer = socket.create_connection(server.server_address, timeout=1)
        try:
            peer.sendall(b"POST /api/v1/freeze HTTP/1.1\r\nHost: local\r\n"
                b"Content-Type: application/json\r\nContent-Length: 2\r\n\r\n{}")
            self.assertTrue(entered.wait(1))
            peer.close()
            with self.assertRaises(RuntimeError):
                server.close()
            self.assertFalse(complete.is_set())
            self.assertEqual(len(runtime._jobs), 1)
            self.assertEqual(len(server.host._jobs), 1)
        finally:
            peer.close()
            release.set()
            self.assertTrue(complete.wait(1))
            deadline = time.monotonic() + 1
            while runtime._jobs and time.monotonic() < deadline:
                time.sleep(0.01)
            server.close()
            runtime.close()

    def test_private_facade_shares_domain_runtime_and_fences_every_internal_state_call(self):
        class Intrinsic:
            def __init__(self): self.calls = 0
            def state(self):
                self.calls += 1
                return {"phase": "collecting", "session_revision": 7}
            def start_candidate(self): return {"accepted": True, "job": {"id": "one"}}
            def save(self, candidate_id): return {"saved": True, "candidate_id": candidate_id}
        domain = Intrinsic()
        path = str(self.root / "calibration.sock")
        server = self.server(intrinsic_service=domain, rpc_socket=path, target_id="test-target")
        initial = ServiceRef("test-target", "xgc2.calibration.v1.Calibration", "1", "", "http.v1", Endpoint("unix", path))
        with Client.from_service(initial, runtime=server.runtime, local_target="test-target", discovery=True) as discovery:
            description = discovery.json("/v1/describe", method="GET")
            reference = ServiceRef.from_dict(description["service_ref"])
            with self.assertRaises(Fault): discovery.json("/api/v1/intrinsic/state", method="GET")
        self.assertEqual(domain.calls, 0)
        with Client.from_service(reference, runtime=server.runtime, local_target="test-target") as client:
            self.assertEqual(client.json("/api/v1/intrinsic/state", method="GET"),
                             {"phase": "collecting", "session_revision": 7})
            self.assertEqual(client.call("/api/v1/intrinsic/candidate", {}).status, 202)
            with self.assertRaises(Fault): client.json("/api/v1/intrinsic/save", {})
            self.assertEqual(client.json("/api/v1/intrinsic/save", {"candidate_id": "candidate-1"})["candidate_id"], "candidate-1")
            calls = domain.calls
            server.private_host.instance_id = "replaced-instance"
            with self.assertRaises(Exception): client.json("/api/v1/intrinsic/state", method="GET")
            self.assertEqual(domain.calls, calls)
        self.assertIs(server.host.runtime, server.private_host.runtime)
        with urllib.request.urlopen(self.url(server, "/api/v1/intrinsic/state"), timeout=1) as public:
            self.assertEqual(json.load(public)["session_revision"], 7)
        server.close()
        self.assertFalse(Path(path).exists())
        self.assertTrue(server.runtime.closed)

    def test_private_facade_validates_explicit_binding_before_owning_runtime(self):
        for path, target in (("relative.sock", "target"), (str(self.root / "calibration.sock"), None)):
            with self.subTest(path=path, target=target), self.assertRaises(ValueError):
                self.server(service=object(), rpc_socket=path, target_id=target)

    def test_private_admission_stops_while_public_work_and_endpoint_are_still_owned(self):
        entered, release = threading.Event(), threading.Event()
        calls = []
        class Service:
            def freeze(self):
                entered.set()
                release.wait(2)
                return {"frozen": True}
            def state(self):
                calls.append("state")
                return {}
            def solve(self, value): return {}
            def save(self, identity): return {}
        path = str(self.root / "calibration.sock")
        server = self.server(service=Service(), rpc_socket=path, target_id="test-target")
        errors = []
        def work():
            try:
                with urllib.request.urlopen(urllib.request.Request(self.url(server, "/api/v1/freeze"),
                        data=b"{}", headers={"Content-Type": "application/json"}), timeout=2) as response:
                    response.read()
            except (OSError, urllib.error.HTTPError): pass
        def close():
            try: server.close()
            except Exception as error: errors.append(error)
        worker = threading.Thread(target=work)
        closer = threading.Thread(target=close)
        worker.start()
        self.assertTrue(entered.wait(1))
        closer.start()
        try:
            deadline = time.monotonic() + 1
            while not server._stopping and time.monotonic() < deadline: time.sleep(.001)
            self.assertTrue(server._stopping)
            self.assertTrue(Path(path).exists())
            self.assertTrue(closer.is_alive())
            with Client.from_service(server.service_ref, runtime=server.runtime, local_target="test-target") as client:
                with self.assertRaises(Fault) as draining:
                    client.json("/api/v1/state", method="GET")
                self.assertEqual(draining.exception.status, 503)
            self.assertEqual(calls, [])
        finally:
            release.set()
            worker.join(2)
            closer.join(2)
        self.assertFalse(closer.is_alive())
        self.assertEqual(errors, [])
        self.assertFalse(Path(path).exists())

    def test_saved_candidate_applies_native_camera_info_receipt_once(self):
        import copy
        import numpy as np
        state = {"source_id": "camera", "desired_revision": 4, "desired": {},
            "applied_revision": 4, "applied": {}, "published_calibration_revision": 4}
        descriptor = {"sourceId": "camera", "width": 640, "height": 480, "capabilities": ["calibration-metadata"]}
        mutations = []
        valid_receipt = [True]
        path = str(self.root / "source.sock")
        runtime = Runtime(blocking_workers=4)
        reference = ServiceRef("test-target", "camera-source", "v1", "source-1", "http.v1", Endpoint("unix", path))
        def apply(context, value):
            mutations.append(copy.deepcopy(value))
            self.assertEqual(value["expected_revision"], 4)
            self.assertIs(value["persist"], False)
            state["desired_revision"] = state["applied_revision"] = 5
            state["desired"] = value["config"]
            state["applied"] = copy.deepcopy(state["desired"])
            state["published_calibration_revision"] = 5 if valid_receipt[0] else 4
            return {**state, "receipt": {"stage": "completed", "effects": {"applied": True}, "request_id": context.request_id}}
        native = Host(path, {("GET", "/v1/media/sources/camera/describe"): lambda c, v: copy.deepcopy(descriptor),
            ("GET", "/v1/media/sources/camera/status"): lambda c, v: copy.deepcopy(state),
            ("GET", "/v1/media/sources/camera/config"): lambda c, v: copy.deepcopy(state),
            ("PATCH", "/v1/media/sources/camera/config"): apply}, runtime=runtime, instance_id="source-1").start()
        from dataclasses import asdict
        edge_path = str(self.root / "edge.sock")
        edge_reference = ServiceRef("test-target", "media-edge", "v1", "edge-1", "http.v1", Endpoint("unix", edge_path))
        edge = Host(edge_path, {("GET", "/v1/describe"): lambda c, v: {"service_ref": asdict(edge_reference)},
            ("GET", "/v1/media/sources/camera/ref"): lambda c, v: {"source_id": "camera", "service_ref": asdict(reference)}},
            runtime=runtime, instance_id="edge-1", discovery_routes=("/v1/describe",)).start()
        application = server = snapshots = None
        try:
            snapshots = MediaSnapshotClient(edge_path, "camera", runtime=runtime, local_target="test-target")
            application = snapshots.camera_metadata_client()
            service = IntrinsicCalibrationService(output_file=str(self.root / "intrinsics.yaml"),
                camera_name="camera", board_size=(7, 5), square=.2, media_source="camera")
            service.attach_metadata_application(application)
            service.result = IntrinsicResult(np.array([[500., 0., 320.], [0., 501., 240.], [0., 0., 1.]]),
                np.zeros(5), (640, 480), .1, 3)
            service._saved_candidate_id = "saved-1"
            service.result_payload = {"candidate_id": "saved-1"}
            server = self.server(intrinsic_service=service, runtime=runtime)
            def request(identity):
                return urllib.request.Request(self.url(server, "/api/v1/intrinsic/apply-metadata"),
                    data=json.dumps({"candidate_id": identity}).encode(), headers={"Content-Type": "application/json"})
            with self.assertRaises(urllib.error.HTTPError) as mismatch:
                urllib.request.urlopen(request("stale"), timeout=1)
            self.assertEqual(mismatch.exception.code, 409)
            self.assertEqual(mutations, [])
            with urllib.request.urlopen(request("saved-1"), timeout=1) as response:
                receipt = json.load(response)
            self.assertEqual(receipt["published_calibration_revision"], 5)
            self.assertEqual(mutations[0]["config"]["calibration"]["model"], "plumb_bob")
            with urllib.request.urlopen(request("saved-1"), timeout=1) as response:
                self.assertEqual(json.load(response), receipt)
            self.assertEqual(len(mutations), 1)
            with urllib.request.urlopen(self.url(server, "/api/v1/intrinsic/state"), timeout=1) as response:
                self.assertEqual(json.load(response)["metadata_application"]["receipt"], receipt)
            # A 200 response and applied fields alone cannot prove ROS publication.
            service._saved_candidate_id = "saved-2"
            service._metadata_receipt = None
            state["desired_revision"] = 4
            valid_receipt[0] = False
            with self.assertRaises(urllib.error.HTTPError) as unconfirmed:
                urllib.request.urlopen(request("saved-2"), timeout=1)
            self.assertEqual(unconfirmed.exception.code, 502)
            with self.assertRaises(urllib.error.HTTPError) as repeated:
                urllib.request.urlopen(request("saved-2"), timeout=1)
            self.assertEqual(repeated.exception.code, 409)
            self.assertEqual(len(mutations), 2)
            server.close()
            self.assertFalse(runtime.closed)
            native.instance_id = "source-2"
            with self.assertRaises(Exception): application.apply(mutations[0]["config"]["calibration"])
            self.assertEqual(len(mutations), 2)
        finally:
            if server is not None: server.close()
            if application is not None: application.close()
            if snapshots is not None: snapshots.close()
            edge.close()
            native.close()
            runtime.close()

    def test_metadata_apply_declines_unadvertised_or_mismatched_source_without_mutation(self):
        from xgc_camera_calibration.media_snapshot import MediaSnapshotError
        runtime = Runtime()
        path = str(self.root / "source.sock")
        state = {"source_id": "camera", "desired_revision": 0}
        descriptor = {"sourceId": "camera", "width": 640, "height": 480, "capabilities": []}
        def mutate(*args): raise AssertionError("invalid source must not be mutated")
        native = Host(path, {("GET", "/v1/media/sources/camera/describe"): lambda c, v: dict(descriptor),
            ("GET", "/v1/media/sources/camera/status"): lambda c, v: dict(state),
            ("GET", "/v1/media/sources/camera/config"): lambda c, v: dict(state),
            ("PATCH", "/v1/media/sources/camera/config"): mutate}, runtime=runtime, instance_id="source-1").start()
        application = None
        try:
            reference = ServiceRef("test-target", "camera-source", "v1", "source-1", "http.v1", Endpoint("unix", path))
            application = CameraMetadataClient(reference, "camera", runtime=runtime, local_target="test-target")
            self.assertFalse(application.state()["available"])
            with self.assertRaises(MediaSnapshotError): application.apply({"scope": "calibration-metadata", "width": 640, "height": 480})
            descriptor["capabilities"] = ["calibration-metadata"]
            application.close()
            application = CameraMetadataClient(reference, "camera", runtime=runtime, local_target="test-target")
            with self.assertRaises(MediaSnapshotError): application.apply({"scope": "calibration-metadata", "width": 1280, "height": 720})
        finally:
            if application is not None: application.close()
            native.close()
            runtime.close()

class ManagedCalibrationStorageTest(unittest.TestCase):
    def test_readable_versions_are_immutable_and_atomic_failed_ref_keeps_old_bytes(self):
        import numpy as np
        from unittest.mock import patch
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            partition = root / "phy/camera"
            service = IntrinsicCalibrationService(output_file=str(partition / "intrinsics.yaml"),
                camera_name="camera", calibration_mode="phy", board_size=(7, 5), square=.2,
                references_dir=str(partition / "references"))
            first = service._versioned_output_path()
            result = IntrinsicResult(np.eye(3), np.zeros(5), (640, 480), .1, 3)
            save_intrinsic(first, result, camera_name="camera", calibration_mode="phy", board_size=(7, 5), square=.2)
            self.assertRegex(first.name, r"^intrinsics-\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}(?:-\d{2,4})?\.yaml$")
            self.assertEqual(selected_intrinsic_path(str(root), "phy", "camera", str(first)), first)
            old = first.read_bytes()
            with self.assertRaises(FileExistsError):
                save_intrinsic(first, result, camera_name="camera", board_size=(7, 5), square=.2)
            self.assertEqual(first.read_bytes(), old)
            second = service._versioned_output_path()
            self.assertNotEqual(first, second)
            self.assertRegex(service._evidence_root.name, r"^capture-\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}(?:-\d{2,4})?$")
            service._save_ref(0, b"old")
            with patch("xgc_camera_calibration.intrinsic_service.os.replace", side_effect=OSError("disk full")):
                with self.assertRaises(OSError): service._save_ref(0, b"new")
            self.assertEqual(service.refs[0], b"old")
            self.assertEqual((partition / "references/0.jpg").read_bytes(), b"old")
            self.assertEqual(list(partition.glob(".intrinsics-*")), [])



if __name__ == "__main__":
    unittest.main()
