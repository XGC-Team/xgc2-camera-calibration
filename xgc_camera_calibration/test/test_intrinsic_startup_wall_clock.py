"""No-ROS regression tests for optional simulation camera startup.

Execute the production entrypoint with only its external imports stubbed. A
frozen ROS clock must not affect the wall-clock deadline; no test really sleeps.
"""

import importlib.util
import math
from pathlib import Path
import sys
import types
import unittest
from unittest.mock import Mock, patch


ENTRYPOINT = Path(__file__).resolve().parents[1] / "scripts" / "intrinsic_calibrator_web.py"


class FakeWallClock:
    def __init__(self):
        self.now = 100.0
        self.sleeps = []
        self.on_sleep = None

    def monotonic(self):
        return self.now

    def sleep(self, seconds):
        if not 0.0 < seconds <= 0.1:
            raise AssertionError("poll sleeps must be positive and bounded")
        self.sleeps.append(seconds)
        self.now += seconds
        if self.on_sleep is not None:
            self.on_sleep()


class IntrinsicStartupWallClockTest(unittest.TestCase):
    def setUp(self):
        self.clock = FakeWallClock()
        self.params = {"~camera_control": True, "~camera_control_timeout": 0.25}
        self.ros = types.ModuleType("rospy")
        self.ros.get_param = lambda name, default=None: self.params.get(name, default)
        self.ros.is_shutdown = Mock(return_value=False)
        self.ros.logwarn = Mock()
        self.ros.loginfo = Mock()
        self.ros.Time = types.SimpleNamespace(now=Mock(return_value=0.0))
        self.ros.Duration = float
        self.ros.Rate = Mock(return_value=types.SimpleNamespace(sleep=Mock(
            side_effect=AssertionError("ROS-time sleep blocks when /clock is frozen")
        )))
        self.control = Mock()
        self.control.available.return_value = False
        self.control_factory = Mock(return_value=self.control)
        names = {
            "intrinsic_service": ("IntrinsicCalibrationService", "intrinsic_calibration_directory"),
            "board_profiles": ("FIELD_6X6_88MM_30PCT", "resolve_aprilgrid_profile"),
            "media_snapshot": ("MediaSnapshotClient",),
            "web_service": ("CalibrationHttpServer",),
            "camera_control": (),
        }
        modules = {"rospy": self.ros, "rospkg": types.ModuleType("rospkg")}
        package = types.ModuleType("xgc_camera_calibration")
        package.__path__ = []
        modules["xgc_camera_calibration"] = package
        for name, attributes in names.items():
            module_name = "xgc_camera_calibration." + name
            module = types.ModuleType(module_name)
            for attribute in attributes:
                setattr(module, attribute, Mock())
            modules[module_name] = module
        modules["xgc_camera_calibration.camera_control"].GazeboCameraControl = self.control_factory
        self.module_patch = patch.dict(sys.modules, modules)
        self.module_patch.start()
        self.addCleanup(self.module_patch.stop)
        spec = importlib.util.spec_from_file_location("intrinsic_startup_under_test", ENTRYPOINT)
        self.entrypoint = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.entrypoint)
        # Create the name for the pre-fix version too: that version still uses
        # rospy.Time/Rate and therefore fails rather than hanging the test.
        self.entrypoint.time = self.clock

    def test_disabled_control_does_not_construct_adapter(self):
        self.params["~camera_control"] = False
        self.assertIsNone(self.entrypoint.maybe_camera_control((2.0, 0.0, 2.2)))
        self.control_factory.assert_not_called()

    def test_available_model_attaches_immediately(self):
        self.control.available.return_value = True
        self.assertIs(self.entrypoint.maybe_camera_control((2.0, 0.0, 2.2)), self.control)
        self.assertEqual(self.clock.sleeps, [])

    def test_frozen_sim_time_expires_on_wall_clock(self):
        self.assertIsNone(self.entrypoint.maybe_camera_control((2.0, 0.0, 2.2)))
        self.assertAlmostEqual(self.clock.now, 100.25)
        self.ros.Time.now.assert_not_called()
        self.ros.Rate.assert_not_called()
        self.ros.logwarn.assert_called_once()

    def test_model_arriving_during_poll_attaches(self):
        self.control.available.side_effect = [False, True]
        self.assertIs(self.entrypoint.maybe_camera_control((2.0, 0.0, 2.2)), self.control)
        self.assertEqual(self.clock.sleeps, [0.1])

    def test_constructor_receives_configured_timeout(self):
        self.control.available.return_value = True
        self.entrypoint.maybe_camera_control((2.0, 0.0, 2.2))
        self.control_factory.assert_called_once_with(
            "gazebo_static_camera", (2.0, 0.0, 2.2), connection_timeout=0.25
        )

    def test_constructor_consumes_same_timeout_budget(self):
        def construct(*args, **kwargs):
            self.clock.now += 0.25
            return self.control
        self.control_factory.side_effect = construct
        self.assertIsNone(self.entrypoint.maybe_camera_control((2.0, 0.0, 2.2)))
        self.assertEqual(self.clock.sleeps, [])

    def test_adapter_error_falls_back(self):
        self.control_factory.side_effect = RuntimeError("Gazebo unavailable")
        self.assertIsNone(self.entrypoint.maybe_camera_control((2.0, 0.0, 2.2)))
        self.ros.logwarn.assert_called_once()

    def test_shutdown_stops_polling(self):
        self.clock.on_sleep = lambda: setattr(self.ros.is_shutdown, "return_value", True)
        self.assertIsNone(self.entrypoint.maybe_camera_control((2.0, 0.0, 2.2)))
        self.assertEqual(self.clock.sleeps, [0.1])

    def test_invalid_timeout_fails_before_constructing_adapter(self):
        for value in (0.0, -1.0, math.nan, math.inf, -math.inf):
            with self.subTest(timeout=value):
                self.params["~camera_control_timeout"] = value
                with self.assertRaisesRegex(ValueError, "finite and positive"):
                    self.entrypoint.maybe_camera_control((2.0, 0.0, 2.2))
        self.control_factory.assert_not_called()


if __name__ == "__main__":
    unittest.main()
