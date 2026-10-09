"""Simulation-v1 consumers exercised against the actual native UDS host."""
import copy
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

from xgc2_xrpc import Fault, Host, Runtime
from xgc_camera_calibration.board_profiles import FIELD_6X6_88MM_30PCT, PROFILES
from xgc_camera_calibration import camera_control


class NativeWorld:
    def __enter__(self):
        self.directory = tempfile.TemporaryDirectory()
        self.path = str(Path(self.directory.name) / "world.sock")
        self.runtime = Runtime()
        self.calls = []
        self.completed = 0
        self.pose = {"position": [0., 0., 1.], "orientation": [0., 0., 0., 1.]}
        self.models = {"camera": {"id": "camera", "generation": 7},
                       "intrinsic_aprilgrid": {"id": "intrinsic_aprilgrid", "generation": 3}}
        self.operation_state = "succeeded"
        self.pending = None
        ref = {"target_id": "test-target", "service": "xgc2.simulation", "api_version": "v1",
               "instance_id": "world-1", "profile": "http.v1",
               "endpoint": {"kind": "unix", "address": self.path}}
        def entity(identity):
            if identity not in self.models:
                raise Fault("not_found", "entity unavailable")
            return {"ref": self.models[identity], "state": {"pose": copy.deepcopy(self.pose)}}
        def query(context, value):
            self.calls.append(("query", None))
            return {"entities": [entity("camera")]}
        def mutation(kind, identity, value):
            self.calls.append((kind, copy.deepcopy(value)))
            if kind != "create" and value["generation"] != self.models[identity]["generation"]:
                raise Fault("conflict", "stale entity")
            self.pending = (kind, identity, value)
            return {"id": "op-1", "state": "accepted"}
        def wait(context, value):
            self.calls.append(("wait", value))
            if self.operation_state == "succeeded":
                kind, identity, body = self.pending
                if kind == "pose": self.pose = copy.deepcopy(body["state"]["pose"])
                elif kind == "delete": self.models.pop(identity)
                elif kind == "create": self.models[identity] = {"id": identity, "generation": 1}
                self.completed += 1
            return {"id": "op-1", "state": self.operation_state, "error": None}
        routes = {("GET", "/v1/describe"): lambda c, r: {"service_ref": ref},
                  ("GET", "/v1/entities"): lambda c, r: {"entities": [entity(i) for i in self.models]},
                  ("GET", "/v1/entities/camera"): query,
                  ("POST", "/v1/entities/camera/state"): lambda c, r: mutation("pose", "camera", r),
                  ("POST", "/v1/operations/op-1/wait"): wait}
        for identity in ("intrinsic_aprilgrid",) + tuple(p.gazebo_instance_name for p in PROFILES.values()):
            routes[("DELETE", "/v1/entities/" + identity)] = lambda c, r, identity=identity: mutation("delete", identity, r)
        routes[("POST", "/v1/entities")] = lambda c, r: mutation("create", r["entity"]["id"], r)
        self.host = Host(self.path, routes, runtime=self.runtime, instance_id="world-1",
                         discovery_routes=("/v1/describe",)).start()
        return self

    def control(self):
        return camera_control.GazeboCameraControl("camera", (2., 0., 2.),
            endpoint=self.path, local_target="test-target")

    def __exit__(self, *args):
        self.host.close()
        self.runtime.close()
        self.directory.cleanup()


class NativeCameraControlTest(unittest.TestCase):
    def test_pose_waits_for_native_completion_and_status_uses_cached_snapshot(self):
        with NativeWorld() as world:
            control = world.control()
            self.addCleanup(control.close)
            count = len(world.calls)
            self.assertTrue(control.available())
            self.assertEqual(control.current_position(), (0., 0., 1.))
            self.assertEqual(control.current()["qw"], 1.)
            self.assertEqual(len(world.calls), count)
            control._apply({"position": [3., 2., 1.], "orientation": [0., 0., 0., 1.]})
            self.assertEqual([c[0] for c in world.calls[-3:]], ["pose", "wait", "query"])
            self.assertEqual(world.calls[-3][1]["generation"], 7)
            self.assertEqual(control.current_position(), (3., 2., 1.))
            control.reset()
            self.assertEqual(control.current_position(), (0., 0., 1.))
            control.close()

    def test_stale_generation_and_failed_operation_never_change_cached_pose(self):
        with NativeWorld() as world:
            control = world.control()
            world.models["camera"]["generation"] = 8
            with self.assertRaises(Exception): control.reset()
            self.assertEqual(world.completed, 0)
            self.assertEqual(control.current_position(), (0., 0., 1.))
            control.close()
        with NativeWorld() as world:
            control = world.control()
            world.operation_state = "failed"
            with self.assertRaisesRegex(RuntimeError, "did not complete"): control.reset()
            self.assertEqual(world.completed, 0)
            self.assertEqual(world.calls[-1][0], "wait")
            control.close()

    def test_replaced_instance_is_not_rediscovered_or_replayed(self):
        with NativeWorld() as world:
            control = world.control()
            world.host.instance_id = "world-2"
            count = len(world.calls)
            with self.assertRaises(Exception): control.reset()
            self.assertEqual(len(world.calls), count)
            control.close()

    def test_selected_board_uses_generation_fenced_remove_and_native_create(self):
        with NativeWorld() as world:
            root = Path(world.directory.name) / "assets"
            profile = PROFILES[FIELD_6X6_88MM_30PCT]
            model = root / "models" / profile.gazebo_model / "model.sdf"
            model.parent.mkdir(parents=True)
            model.write_text("<sdf version='1.6'><model name='board'/></sdf>")
            rospkg = types.ModuleType("rospkg")
            rospkg.RosPack = lambda: types.SimpleNamespace(get_path=lambda _: str(root))
            with mock.patch.dict(sys.modules, {"rospkg": rospkg}):
                for _ in range(2):
                    camera_control.select_gazebo_board_profile(profile, (2., 0., 2.2),
                        endpoint=world.path, local_target="test-target")
            self.assertNotIn("intrinsic_aprilgrid", world.models)
            self.assertIn(profile.gazebo_instance_name, world.models)
            self.assertEqual([c[0] for c in world.calls], ["delete", "wait", "create", "wait"])
            self.assertEqual(world.calls[0][1]["generation"], 3)
            entity = world.calls[2][1]["entity"]
            self.assertEqual(entity["pose"]["position"], [2., 0., 2.2])
            self.assertEqual(entity["asset"]["realization"]["media_type"], "application/sdf+xml")

    def test_endpoint_is_required_and_http_is_rejected(self):
        with self.assertRaises(TypeError): camera_control.GazeboCameraControl("camera", (0., 0., 0.))
        with self.assertRaises(ValueError):
            camera_control.GazeboCameraControl("camera", (0., 0., 0.), endpoint="http://localhost:8080")


if __name__ == "__main__": unittest.main()
