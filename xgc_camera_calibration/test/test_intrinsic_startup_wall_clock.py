"""Optional simulation control is configured without network or ROS-time waits."""
import importlib.util
import math
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch


class IntrinsicStartupWallClockTest(unittest.TestCase):
    def setUp(self):
        path = Path(__file__).parents[1] / "scripts/intrinsic_calibrator_web.py"
        spec = importlib.util.spec_from_file_location("intrinsic_startup_test", path)
        self.entry = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.entry)
        self.values = {"~camera_control": True, "~camera_control_timeout": .25,
                       "~simulation_endpoint": "/granted/world.sock"}
        self.parameters = SimpleNamespace(get=lambda name, default=None: self.values.get(name, default))

    def control(self):
        return self.entry.maybe_camera_control((2., 0., 2.2), "profile", self.parameters,
            runtime="runtime", local_target="test-target")

    def test_disabled_control_does_not_construct_adapter(self):
        self.values["~camera_control"] = False
        with patch("xgc_camera_calibration.camera_control.GazeboCameraControl") as factory:
            self.assertIsNone(self.control())
            factory.assert_not_called()

    def test_configured_control_is_retained_without_availability_probe(self):
        with patch("xgc_camera_calibration.camera_control.GazeboCameraControl") as factory:
            self.assertIs(self.control(), factory.return_value)
            factory.return_value.available.assert_not_called()
            factory.assert_called_once_with("gazebo_static_camera", (2., 0., 2.2),
                connection_timeout=.25, endpoint="/granted/world.sock", runtime="runtime",
                local_target="test-target", board_profile="profile")

    def test_invalid_timeout_fails_before_constructing_adapter(self):
        with patch("xgc_camera_calibration.camera_control.GazeboCameraControl") as factory:
            for value in (0., -1., math.nan, math.inf, -math.inf):
                self.values["~camera_control_timeout"] = value
                with self.assertRaisesRegex(ValueError, "finite and positive"):
                    self.control()
            factory.assert_not_called()
