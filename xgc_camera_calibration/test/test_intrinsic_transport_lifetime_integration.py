"""Real HTTP/XRPC detector lifetime with a controlled external frame source.

This is not an original Experiment or physical-camera receipt. The normal
calibration owners and actual XRPC transport consume the existing test Media
boundary's blank JPEG; no device, ROS master or host network is used.
"""

import json
import os
from pathlib import Path
import tempfile
import threading
import time
import unittest
import urllib.request
from unittest.mock import patch

from xgc2_xrpc import Client, Endpoint, Runtime, ServiceRef, multipart
from xgc_camera_calibration.board_profiles import FIELD_6X6_88MM_30PCT, resolve_aprilgrid_profile
from xgc_camera_calibration.intrinsic_service import IntrinsicCalibrationService
from xgc_camera_calibration.media_snapshot import MediaSnapshotClient
from xgc_camera_calibration.web_service import CalibrationHttpServer


def require_isolated_device_free_lab():
    """Fail before creating any socket unless the ordinary lab is isolated."""
    if os.geteuid() == 0 or os.getuid() == 0:
        raise RuntimeError("native calibration transport test requires a non-root lab")
    if sorted(os.listdir("/sys/class/net")) != ["lo"]:
        raise RuntimeError("native calibration transport test requires a loopback-only network namespace")
    routes = Path("/proc/net/route").read_text().splitlines()
    if len(routes) != 1:
        raise RuntimeError("native calibration transport test requires an empty IPv4 route table")
    fields = {}
    for line in Path("/proc/self/status").read_text().splitlines():
        if ":" in line:
            key, value = line.split(":", 1)
            fields[key] = value.strip()
    if int(fields.get("CapEff", "-1"), 16) != 0 or fields.get("NoNewPrivs") != "1":
        raise RuntimeError("native calibration transport test requires no capabilities and no-new-privileges")
    if os.environ.get("DOCKER_HOST"):
        raise RuntimeError("native calibration transport test cannot inherit a Docker endpoint")
    for path in (Path("/var/run/docker.sock"), Path("/dev/kvm"), Path("/dev/dri"),
                 Path("/dev/v4l"), Path("/dev/bus/usb")):
        if path.exists() or path.is_symlink():
            raise RuntimeError("native calibration transport test cannot expose {}".format(path))
    for pattern in ("video*", "ttyUSB*", "ttyACM*", "hidraw*"):
        if list(Path("/dev").glob(pattern)):
            raise RuntimeError("native calibration transport test cannot expose /dev/{}".format(pattern))


class IntrinsicTransportLifetimeIntegrationTest(unittest.TestCase):
    def setUp(self):
        require_isolated_device_free_lab()
        # Reuse the existing XRPC Media test boundary. Only its selected source
        # identity is rebound to the original Experiment's "front" camera.
        from test_media_snapshot import FakeMediaEdge

        self.directory = tempfile.TemporaryDirectory(prefix="xgc-camera-transport-")
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.edge = FakeMediaEdge(str(self.root / "media.sock")).__enter__()
        self.addCleanup(self.edge.__exit__)
        self.edge.sources = [{"id": "front"}]
        self.edge.metadata.update(sourceId="front", frameId="front_optical_frame",
                                  timestampClockDomain="unknown", calibrationState="unavailable",
                                  cameraMatrix=[], distortion=[])
        self.edge.metadata.pop("poseFrameId")
        self.edge.metadata.pop("renderPose")
        routes = self.edge.host.routes
        routes[("POST", "/v1/media/sources/front/capture")] = routes[("POST", "/v1/media/sources/usb_cam/capture")]

        self.runtime = Runtime(blocking_workers=4, max_calls=16, max_connections=24)
        self.addCleanup(self.runtime.close)
        self.snapshots = MediaSnapshotClient(self.edge.path, "front", runtime=self.runtime,
                                             local_target="test-target")
        self.addCleanup(self.snapshots.close)
        profile = resolve_aprilgrid_profile(FIELD_6X6_88MM_30PCT)
        self.service = IntrinsicCalibrationService(
            board_size=(profile.columns, profile.rows), square=profile.tag_size_m,
            board_type="aprilgrid", tag_spacing=profile.tag_gap_m, tag_family=profile.tag_family,
            tag_start_id=profile.start_id, min_tags=profile.min_tags,
            board_profile_id=profile.profile_id, output_file=str(self.root / "intrinsics.yaml"),
            camera_name="front", media_source="front", calibration_mode="phy")
        self.service.capture_peer = self.snapshots.peer
        self.service.attach_frame_capture(lambda: self.snapshots.capture_detection(640 * 480))
        self.service.attach_metadata_application(self.snapshots)
        self.service.live_capture_interval = 0.2
        web_root = Path(__file__).resolve().parents[1] / "web" / "intrinsic"
        self.server = CalibrationHttpServer(
            ("127.0.0.1", 0), None, web_root, frame_ancestors="'self'",
            intrinsic_service=self.service, runtime=self.runtime, target_id="test-target",
            rpc_socket=str(self.root / "calibration.sock"),
            peer_status=lambda: {"media": self.snapshots.status()}).start()
        self.addCleanup(self.server.close)
        self.addCleanup(self.service.stop_auto_capture)
        self.streams = []
        self.addCleanup(self.close_streams)

    def close_streams(self):
        for stream in self.streams:
            stream.close()
        self.streams.clear()

    def url(self, path):
        return "http://127.0.0.1:{}{}".format(self.server.server_address[1], path)

    def request(self, path, payload=None):
        request = urllib.request.Request(
            self.url(path), data=json.dumps(payload).encode() if payload is not None else None,
            headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(request, timeout=5) as response:
            return json.load(response)

    def viewer(self):
        response = urllib.request.urlopen(self.url("/api/v1/intrinsic/events"), timeout=5)
        self.streams.append(response)
        self.assertEqual(response.readline(), b"id: 1\n")
        return response

    def wait_until(self, description, predicate, timeout=5):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if predicate():
                return
            time.sleep(0.01)
        self.fail("{} did not join: viewers={} captures={} released={}".format(
            description, self.server._live_capture_viewers,
            len(self.edge.requests), len(self.edge.deleted)))

    def wait_last_view_joined(self):
        self.wait_until("last intrinsic observer", lambda:
            self.server._live_capture_viewers == 0 and self.server._live_capture_retirement is not None
            and self.server._live_capture_retirement.done())
        self.server._live_capture_retirement.result()
        self.assertEqual(self.service._auto_capture_observers, 0)

    def assert_no_new_captures(self):
        # Three normal 200 ms detector periods expose a continuing producer.
        before = len(self.edge.requests)
        time.sleep(0.65)
        self.assertEqual(len(self.edge.requests), before)

    def wait_snapshots_released(self):
        self.wait_until("actual XRPC snapshot releases", lambda:
            len(self.edge.deleted) == len(self.edge.snapshot_ids))
        self.assertCountEqual(self.edge.deleted, self.edge.snapshot_ids)

    def test_two_http_views_share_capture_and_reveal_waits_for_real_xrpc_join(self):
        entered, release, opened = threading.Event(), threading.Event(), threading.Event()
        self.addCleanup(release.set)

        def held_capture(value):
            entered.set()
            if not release.wait(5):
                raise RuntimeError("controlled XRPC frame was not released")
            raw = self.edge.raw if value["includeRgb"] else b""
            metadata = {**self.edge.metadata, "snapshotId": self.edge.snapshot_ids[-1],
                        "jpegBytes": len(self.edge.jpeg), "rgbBytes": len(raw)}
            return multipart(metadata, self.edge.jpeg, raw)

        self.edge.custom_capture = held_capture
        head = urllib.request.Request(self.url("/api/v1/intrinsic/events"), method="HEAD")
        with urllib.request.urlopen(head, timeout=5) as response:
            self.assertEqual(response.read(), b"")
        self.assertEqual(self.server._live_capture_viewers, 0)
        self.assertEqual(self.edge.requests, [])
        first = self.viewer()
        self.assertTrue(entered.wait(3), "the real Media XRPC capture was not entered")
        detector = self.service._auto_capture_thread
        second = self.viewer()
        self.wait_until("two HTTP readers", lambda: self.server._live_capture_viewers == 2)
        self.assertEqual(self.service._auto_capture_observers, 1)
        self.assertIs(self.service._auto_capture_thread, detector)
        self.assertEqual(len(self.edge.requests), 1)
        self.assertFalse(self.edge.requests[0][1]["includeRgb"])
        self.assertTrue(self.edge.requests[0][1]["requireFresh"])
        first.close()
        self.wait_until("one HTTP reader", lambda: self.server._live_capture_viewers == 1)
        self.assertIs(self.service._auto_capture_thread, detector)
        second.close()
        self.wait_until("retiring final HTTP reader", lambda:
            self.server._live_capture_viewers == 0 and self.server._live_capture_retirement is not None)
        retirement = self.server._live_capture_retirement
        self.assertFalse(retirement.done(), "last release returned before the held physical capture joined")

        result = {}

        def reveal():
            try:
                result["stream"] = self.viewer()
            except Exception as error:
                result["error"] = error
            finally:
                opened.set()

        worker = threading.Thread(target=reveal, name="controlled-reveal")
        worker.start()
        try:
            self.wait_until("replacement HTTP reader admission", lambda: self.server._live_capture_viewers == 1)
            self.assertFalse(opened.is_set(), "reveal overtook the still-owned XRPC capture")
            self.assertEqual(len(self.edge.requests), 1)
            self.edge.custom_capture = None
            release.set()
            self.assertTrue(opened.wait(4), "reveal did not resume after the real capture joined")
            self.assertNotIn("error", result)
            self.wait_until("replacement detector capture", lambda: len(self.edge.requests) >= 2)
            result["stream"].close()
            self.wait_last_view_joined()
            self.wait_snapshots_released()
            self.assertIsNone(self.service._auto_capture_thread)
            self.assertIsNone(self.snapshots.peer._thread)
            self.assertIsNone(self.snapshots.peer._watch_fd)
            self.assertEqual(self.snapshots.peer._owners, 0)
            self.assert_no_new_captures()
        finally:
            release.set()
            worker.join(5)
            self.assertFalse(worker.is_alive(), "reveal request did not join")

    def test_pause_survives_reconnect_and_explicit_collection_outlives_view(self):
        first = self.viewer()
        self.wait_until("first real snapshot", lambda: len(self.edge.requests) >= 1)
        paused = self.request("/api/v1/intrinsic/auto_capture/stop", {})
        self.assertTrue(paused["ok"])
        self.assertFalse(paused["auto_capture"]["enabled"])
        self.wait_snapshots_released()
        first.close()
        self.wait_last_view_joined()

        reveal = self.viewer()
        self.assertFalse(self.request("/api/v1/intrinsic/state")["auto_capture"]["enabled"])
        self.assert_no_new_captures()
        reveal.close()
        self.wait_last_view_joined()

        # The explicit capture is accepted without a viewer, transfers a real
        # XRPC frame and releases its exact snapshot. A blank controlled frame
        # cannot produce a calibration or stand in for a physical board.
        before = len(self.edge.requests)
        capture = self.request("/api/v1/intrinsic/capture", {})
        self.assertTrue(capture["ok"])
        self.assertEqual(capture["samples"], 0)
        self.assertEqual(len(self.edge.requests), before + 1)
        self.wait_snapshots_released()
        self.assert_no_new_captures()

        reveal = self.viewer()
        state = self.request("/api/v1/intrinsic/state")
        self.assertFalse(state["auto_capture"]["enabled"])
        self.assertTrue(state["image_ready"])
        self.assertEqual(state["media_source"], "front")
        resumed = self.request("/api/v1/intrinsic/auto_capture/start", {})
        self.assertTrue(resumed["ok"])
        self.assertTrue(resumed["auto_capture"]["enabled"])
        self.wait_until("resumed collection", lambda: len(self.edge.requests) >= before + 2)
        reveal.close()
        self.wait_last_view_joined()
        after_park = len(self.edge.requests)
        self.wait_until("accepted collection while parked", lambda: len(self.edge.requests) > after_park)

        discovery_ref = ServiceRef("test-target", "xgc2.calibration.v1.Calibration", "1", "", "http.v1",
                                   Endpoint("unix", str(self.root / "calibration.sock")))
        with Client.from_service(discovery_ref, runtime=self.runtime, local_target="test-target", discovery=True) as client:
            description = client.json("/v1/describe?wait_ready_ms=0", method="GET")
        self.assertTrue(description["ready"])
        self.assertEqual(description["facts"]["camera_id"], "front")
        self.assertEqual(description["facts"]["calibration_mode"], "phy")
        reference = ServiceRef.from_dict(description["service_ref"])
        self.assertEqual(reference, self.server.service_ref)
        with Client.from_service(reference, runtime=self.runtime, local_target="test-target") as client:
            state = client.json("/api/v1/intrinsic/state", method="GET")
        self.assertTrue(state["auto_capture"]["enabled"])
        self.assertEqual(state["media_source"], "front")
        paused = self.request("/api/v1/intrinsic/auto_capture/stop", {})
        self.assertTrue(paused["ok"])
        self.assertFalse(paused["auto_capture"]["enabled"])
        self.wait_snapshots_released()
        self.assert_no_new_captures()
        reveal = self.viewer()
        self.assertFalse(self.request("/api/v1/intrinsic/state")["auto_capture"]["enabled"])
        self.assert_no_new_captures()
        reveal.close()
        self.wait_last_view_joined()
        self.wait_snapshots_released()

    def test_two_http_views_receive_only_changed_native_state_and_join(self):
        class Camera:
            def current(self):
                return None

        self.service.stop_auto_capture()
        projections = []
        encoded = []
        original = self.server._state
        dumps = json.dumps
        def project(domain):
            state = original(domain)
            projections.append((state["camera_control"], state["peers"]["media"]["ready"]))
            return state
        def encode(value, *args, **kwargs):
            if isinstance(value, dict) and value.get("mode") == "intrinsic":
                encoded.append(1)
            return dumps(value, *args, **kwargs)

        def received_camera_control(stream, value):
            while True:
                line = stream.readline()
                if not line:
                    self.fail("state stream closed before the changed fact")
                if line.startswith(b"data: "):
                    state = json.loads(line[6:])
                    if state["camera_control"] is value:
                        return

        def received_media_ready(stream, value):
            while True:
                line = stream.readline()
                if not line:
                    self.fail("state stream closed before the changed peer fact")
                if line.startswith(b"data: "):
                    state = json.loads(line[6:])
                    if state["peers"]["media"]["ready"] is value:
                        return

        with patch.object(self.server, "_state", side_effect=project), \
             patch("xgc_camera_calibration.web_service.json.dumps", side_effect=encode):
            first = self.viewer()
            second = self.viewer()
            self.wait_until("media peer ready", lambda: self.snapshots.peer.status()["ready"])
            self.wait_until("ready state projected", lambda: projections and projections[-1] == (False, True))
            time.sleep(0.15)
            baseline = len(projections)
            encoded_baseline = len(encoded)
            time.sleep(0.35)
            self.assertEqual((len(projections), len(encoded)), (baseline, encoded_baseline))
            self.assertEqual(self.edge.requests, [])
            # Drain the initial discovery transition from each stream before
            # the later intentional disconnect/recovery assertions.
            received_media_ready(first, True)
            received_media_ready(second, True)
            self.service.attach_camera_control(Camera())
            self.wait_until("one changed camera-control fact", lambda: len(projections) > baseline)
            received_camera_control(first, True)
            received_camera_control(second, True)
            self.assertEqual(projections[-1], (True, True))
            self.assertEqual(len(projections), baseline + 1)
            self.assertEqual(len(encoded), encoded_baseline + 1)
            changed_delta = (len(projections) - baseline, len(encoded) - encoded_baseline)
            # One deliberate held rediscovery exposes the real peer transition;
            # this is a separate negative fault, not normal throughput evidence.
            held, release_peer = threading.Event(), threading.Event()
            self.addCleanup(release_peer.set)
            key = ("GET", "/v1/describe")
            describe = self.edge.host.routes[key]
            def held_describe(context, request):
                held.set()
                if not release_peer.wait(3):
                    raise RuntimeError("controlled peer discovery was not released")
                return describe(context, request)
            self.edge.host.routes[key] = held_describe
            try:
                self.snapshots.peer.invalidate(RuntimeError("controlled peer transition"))
                self.wait_until("media peer unavailable projection", lambda: projections and projections[-1] == (True, False))
                received_media_ready(first, False)
                received_media_ready(second, False)
                self.assertTrue(held.wait(3))
                release_peer.set()
                self.wait_until("media peer recovery", lambda: self.snapshots.peer.status()["ready"])
                received_media_ready(first, True)
                received_media_ready(second, True)
            finally:
                release_peer.set()
                self.edge.host.routes[key] = describe
            first.close()
            second.close()
            self.wait_last_view_joined()
            self.assertEqual(len(self.service._state_listeners), 0)
            self.assertEqual(len(self.snapshots.peer._state_listeners), 0)
            self.assertIsNone(self.snapshots.peer._thread)
            self.assertIsNone(self.snapshots.peer._watch_fd)
            self.assertEqual(self.snapshots.peer._owners, 0)
            joined = len(projections)
            joined_encodes = len(encoded)
            self.service.attach_camera_control(None)
            time.sleep(0.35)
            self.assertEqual((len(projections), len(encoded)), (joined, joined_encodes))
            hidden_delta = (len(projections) - joined, len(encoded) - joined_encodes)
            reveal = self.viewer()
            received_camera_control(reveal, False)
            self.assertEqual(len(projections), joined + 1)
            self.assertEqual(len(encoded), joined_encodes + 1)
            reveal_delta = (len(projections) - joined, len(encoded) - joined_encodes)
            reveal.close()
            self.wait_last_view_joined()
            print('T15_INTRINSIC_TRANSPORT_WORK ' + json.dumps({
                'visible_readers': 2, 'paused_media_captures': len(self.edge.requests),
                'changed_projection_encode_delta': changed_delta,
                'hidden_projection_encode_delta': hidden_delta,
                'reveal_projection_encode_delta': reveal_delta,
                'peer_owners_after_join': self.snapshots.peer._owners,
                'peer_thread_after_join': self.snapshots.peer._thread is not None,
            }, sort_keys=True))


if __name__ == "__main__":
    unittest.main()
