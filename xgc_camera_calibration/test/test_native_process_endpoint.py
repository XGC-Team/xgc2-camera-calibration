"""Core's allocated socket is independent of the frozen bootstrap grants."""
import importlib.util
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import MagicMock, patch

from xgc2_xrpc import BootstrapBinding, Client, Endpoint, Fault, ServiceRef
from xgc_camera_calibration.web_service import CalibrationHttpServer


class NativeProcessEndpointTest(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.root = Path(self.directory.name)
        for name in ("index.html", "app.js", "styles.css"):
            (self.root / name).write_text("asset")
        self.binding = BootstrapBinding({
            "schema_version": 1, "target_id": "local-test",
            "service": "xgc2.calibration.v1.Calibration", "api_version": "1",
            "profile": "http.v1", "endpoint": {"kind": "unix", "address": str(self.root / "bootstrap.sock")},
            "runtime_grant": "calibration", "authentication": "local_private",
            "secret_handles": {}, "storage_grants": ["preferences"],
        })

    def tearDown(self):
        self.directory.cleanup()

    def test_both_domains_bind_allocated_socket_and_answer_unbound_readiness_query(self):
        class Domain:
            def state(self):
                return {"phase": "collecting"}
            def solve(self, value):
                return value
            def save(self, identity):
                return {"saved": identity}
        for intrinsic in (False, True):
            with self.subTest(intrinsic=intrinsic):
                path = str(self.root / "allocated.sock")
                domain = Domain()
                server = CalibrationHttpServer(("127.0.0.1", 0), None if intrinsic else domain,
                    self.root, frame_ancestors="'self'", binding=self.binding, rpc_socket=path,
                    intrinsic_service=domain if intrinsic else None).start()
                try:
                    initial = ServiceRef(self.binding.target_id, self.binding.service,
                        self.binding.api_version, "", self.binding.profile, Endpoint("unix", path))
                    with Client.from_service(initial, runtime=server.runtime,
                            local_target=self.binding.target_id, discovery=True) as discovery:
                        started = time.monotonic()
                        description = discovery.json("/v1/describe?wait_ready_ms=30000", method="GET", timeout=1)
                        self.assertLess(time.monotonic() - started, 1)
                        self.assertTrue(description["ready"])
                        self.assertEqual(description["service"], initial.service)
                        self.assertEqual(description["api_version"], initial.api_version)
                        self.assertRegex(description["instance_id"], r"^[0-9a-f]{32}$")
                        self.assertEqual(description["facts"]["capabilities"],
                                         {"intrinsic": intrinsic, "extrinsic": not intrinsic})
                        for query in ("wait_ready_ms=-1", "wait_ready_ms=30001", "wait_ready_ms=",
                                      "wait_ready_ms=1&wait_ready_ms=2", "unknown=1"):
                            with self.subTest(query=query), self.assertRaises(Fault) as error:
                                discovery.json("/v1/describe?" + query, method="GET")
                            self.assertEqual(error.exception.status, 400)
                    reference = ServiceRef.from_dict(description["service_ref"])
                    self.assertEqual(reference.endpoint, initial.endpoint)
                    self.assertEqual(reference.target_id, self.binding.target_id)
                    self.assertEqual(reference.instance_id, description["instance_id"])
                    with Client.from_service(reference, runtime=server.runtime,
                            local_target=self.binding.target_id) as client:
                        route = "/api/v1/intrinsic/state" if intrinsic else "/api/v1/state"
                        self.assertEqual(client.json(route, method="GET"), {"phase": "collecting"})
                    self.assertEqual(self.binding.storage_grants, ("preferences",))
                    self.assertEqual(self.binding.endpoint.address, str(self.root / "bootstrap.sock"))
                    self.assertFalse(Path(self.binding.endpoint.address).exists())
                finally:
                    server.close()
                self.assertFalse(Path(path).exists())
                self.assertTrue(server.runtime.closed)

    def test_invalid_socket_and_target_override_fail_before_runtime_allocation(self):
        for kwargs in ({"rpc_socket": "relative.sock"}, {"rpc_socket": ""},
                       {"rpc_socket": "/"}, {"rpc_socket": "//private/socket"},
                       {"rpc_socket": "/private/\x00.sock"}, {"rpc_socket": "/" + "a" * 107},
                       {"target_id": "different-target"}):
            with self.subTest(kwargs=kwargs), patch("xgc_camera_calibration.web_service.Runtime") as runtime:
                with self.assertRaises(ValueError):
                    CalibrationHttpServer(("127.0.0.1", 0), object(), self.root,
                        frame_ancestors="'self'", binding=self.binding, **kwargs)
                runtime.assert_not_called()

    def test_calibrator_arguments_forward_socket_and_exact_application_state_name(self):
        parameters = {"~bootstrap_input": "/private/bootstrap.json",
            "~socket_path": str(self.root / "allocated.sock"), "~calibration_root": str(self.root),
            "~calibration_mode": "phy", "~camera_name": "test-camera"}
        state_name = "/owned_camera_tf/extrinsic_application_state"
        for kind in ("intrinsic", "extrinsic"):
            with self.subTest(kind=kind):
                ros = MagicMock()
                ros.get_param.side_effect = lambda name, *default: parameters[name] if name in parameters else default[0]
                ros.myargv.return_value = ["calibrator", "--resolved-extrinsic-json", "{}",
                    "--frame-roles-json", "{}", "--application-state-param", state_name]
                path = Path(__file__).parents[1] / "scripts" / (kind + "_calibrator_web.py")
                spec = importlib.util.spec_from_file_location(kind + "_entrypoint_test", path)
                entry = importlib.util.module_from_spec(spec)
                with patch.dict("sys.modules", {"rospy": ros, "geometry_msgs.msg": MagicMock(),
                                               "sensor_msgs.msg": MagicMock()}):
                    spec.loader.exec_module(entry)
                bootstrap = MagicMock(binding=self.binding)
                with patch.object(entry, "load_bootstrap_input", return_value=bootstrap), \
                        patch.object(entry, "Runtime"), patch.object(entry, "MediaSnapshotClient"), \
                        patch.object(entry, "create_preferences", return_value=(object(), MagicMock())), \
                        patch.object(entry.rospkg, "RosPack"), \
                        patch.object(entry, "CalibrationHttpServer", side_effect=RuntimeError("stop test startup")) as server:
                    if kind == "intrinsic":
                        with patch.object(entry, "IntrinsicCalibrationService"):
                            self.assertEqual(entry.main(), 1)
                    else:
                        with patch.object(entry, "RosCalibrationSource") as source, \
                                patch.object(entry, "CalibrationService") as service, \
                                patch.object(entry, "parse_frame_roles"), patch.object(entry, "decode_frozen"):
                            source.return_value.pose_coordinate_source = "experiment-world"
                            source.return_value.pose_world_offset = (0, 0, 0)
                            service.return_value.frozen_extrinsic = {"targetCoordinates": {"worldOffset": [0, 0, 0]}}
                            self.assertEqual(entry.main(), 1)
                            service.call_args.kwargs["application_state_reader"]()
                            ros.get_param.assert_called_with(state_name, None)
                    self.assertEqual(server.call_args.kwargs["rpc_socket"], parameters["~socket_path"])
                    self.assertIs(server.call_args.kwargs["binding"], self.binding)


if __name__ == "__main__":
    unittest.main()
