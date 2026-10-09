"""Consumer format oracles transferred from the retired Go workflow renderer."""
import json
import math
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import yaml
from xgc_camera_calibration.asset_document import render

SPEC = {
 "name": "Bench camera", "camera": {"sourceId": "usb_cam"},
 "intrinsics": {"width": 1280, "height": 720, "model": "plumb_bob",
 "k": [900.5, 0, 640.25, 0, 901.5, 360.75, 0, 0, 1], "d": [-.31, .09, .001, -.002, 0]},
 "extrinsics": {"parentFrame": "map", "childFrame": "usb_cam_optical_frame",
 "translation": [1.5, -.25, 2], "rotationRpy": [0, 0, math.pi / 2]},
 "provenance": {"method": "extrinsic-service", "capturedAt": "2026-07-28T09:12:00Z"}
}
GOLDENS = {'cameraInfoGolden': 'image_width: 1280\nimage_height: 720\ncamera_name: usb_cam\ncamera_matrix:\n  rows: 3\n  cols: 3\n  data: [900.5, 0.0, 640.25, 0.0, 901.5, 360.75, 0.0, 0.0, 1.0]\ndistortion_model: plumb_bob\ndistortion_coefficients:\n  rows: 1\n  cols: 5\n  data: [-0.31, 0.09, 0.001, -0.002, 0.0]\nrectification_matrix:\n  rows: 3\n  cols: 3\n  data: [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]\nprojection_matrix:\n  rows: 3\n  cols: 4\n  data: [900.5, 0.0, 640.25, 0.0, 0.0, 901.5, 360.75, 0.0, 0.0, 0.0, 1.0, 0.0]\n', 'extrinsicsGolden': 'schema: xgc2.camera.extrinsic.v1\ncreated_at: "2026-07-28T09:12:00Z"\nframe_convention: parent_T_camera_optical\nparent_frame: map\nchild_frame: usb_cam_optical_frame\ntranslation:\n  x: 1.5\n  y: -0.25\n  z: 2.0\nquaternion_xyzw:\n  x: 0.0\n  y: 0.0\n  z: 0.7071067811865475\n  w: 0.7071067811865476\n'}

class AssetDocumentTests(unittest.TestCase):
    def test_existing_ros_documents(self):
        for output_format, golden in (("camera_info_yaml", "cameraInfoGolden"),
                                      ("extrinsics_yaml", "extrinsicsGolden")):
            self.assertEqual(yaml.safe_load(render(SPEC, output_format)), yaml.safe_load(GOLDENS[golden]))

    def test_decimal_and_measurement_provenance(self):
        spec = json.loads(json.dumps(SPEC))
        spec["intrinsics"]["d"][0] = 1e-12
        document = yaml.safe_load(render(spec, "camera_info_yaml"))
        self.assertEqual(document["distortion_coefficients"]["data"][0], 1e-12)
        spec["provenance"]["method"] = "intrinsic-service"
        self.assertNotIn("created_at", yaml.safe_load(render(spec, "extrinsics_yaml")))
        spec["intrinsics"]["k"][0] = float("inf")
        with self.assertRaises(ValueError):
            render(spec, "camera_info_yaml")

    def test_fixed_axis_rotation_at_poles(self):
        for pitch in (math.pi/2, -math.pi/2, math.pi/2-1e-7, math.pi/2+1e-7, -math.pi/2-1e-7, -math.pi/2+1e-7, .4):
            for roll, yaw in ((.7, 1.2), (-1.3, .9), (2.1, -2.4)):
                spec = json.loads(json.dumps(SPEC))
                spec["extrinsics"]["rotationRpy"] = [roll, pitch, yaw]
                q = yaml.safe_load(render(spec, "extrinsics_yaml"))["quaternion_xyzw"]
                # Independent Rz * Ry * Rx matrix oracle, including singular Euler poses.
                x, y, z, w = [q[a] for a in "xyzw"]
                actual = [[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
                          [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
                          [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]]
                sr, cr, sp, cp, sy, cy = math.sin(roll), math.cos(roll), math.sin(pitch), math.cos(pitch), math.sin(yaw), math.cos(yaw)
                expected = [[cy*cp, cy*sp*sr-sy*cr, cy*sp*cr+sy*sr],
                            [sy*cp, sy*sp*sr+cy*cr, sy*sp*cr-cy*sr], [-sp, cp*sr, cp*cr]]
                for row, oracle in zip(actual, expected):
                    for value, exact in zip(row, oracle):
                        self.assertAlmostEqual(value, exact, places=12)

    def test_plain_installed_cli_and_atomic_output(self):
        script = Path(__file__).resolve().parents[1] / "scripts" / "render_asset.py"
        with tempfile.TemporaryDirectory() as directory:
            source, output = Path(directory)/"asset.json", Path(directory)/"camera.yaml"
            source.write_text(json.dumps(SPEC))
            result = subprocess.run([sys.executable, str(script), "--input", str(source),
                "--format", "camera_info_yaml", "--output", str(output)], check=True, capture_output=True, text=True)
            receipt = json.loads(result.stdout)
            self.assertEqual(receipt["bytesWritten"], output.stat().st_size)
            self.assertEqual(output.stat().st_mode & 0o777, 0o600)
            self.assertEqual(yaml.safe_load(output.read_text()), yaml.safe_load(GOLDENS["cameraInfoGolden"]))

if __name__ == "__main__":
    unittest.main()
