"""Frozen context parsing has no file or persistence side effects."""
import json
import unittest
from xgc_camera_calibration.extrinsic_application import application_arguments, parse_frame_roles
from xgc_camera_calibration.solver import CalibrationError

class FrozenContextTest(unittest.TestCase):
    def test_exact_frame_roles_and_arguments(self):
        roles = {"parentFrame": "world", "opticalFrames": {"sim": "sim_optical", "phy": "front_optical"}}
        self.assertEqual(parse_frame_roles(json.dumps(roles)), roles)
        arguments = application_arguments(["--resolved-extrinsic-json", "{}", "--frame-roles-json", json.dumps(roles)])
        self.assertEqual(arguments.resolved_extrinsic_json, "{}")
        with self.assertRaises(CalibrationError):
            parse_frame_roles("{" + " " * 1024)
