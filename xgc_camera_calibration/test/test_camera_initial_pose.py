#!/usr/bin/env python3

import math
import tempfile
import unittest
from pathlib import Path

import numpy as np

from xgc_camera_calibration.camera_initial_pose import (
    replace_roslaunch_pose_arguments,
    resolve_gazebo_camera_pose_from_file,
)
from xgc_camera_calibration.solver import (
    CalibrationError,
    ExtrinsicResult,
    save_extrinsic,
)
from xgc_camera_calibration.transforms import (
    link_to_optical_rotation,
    quaternion_to_rotation_matrix,
    split_parent_to_optical_pose,
)


class CameraInitialPoseTest(unittest.TestCase):
    def test_resolves_explicit_physical_file_pose_before_roslaunch(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "camera"
            output = root / "phy" / "usb_cam" / "extrinsics-20260830T010000.000000Z.yaml"
            result = ExtrinsicResult(
                translation=np.asarray((1.0, 2.0, 3.0)),
                quaternion_xyzw=np.asarray((0.0, 0.0, 0.0, 1.0)),
                rotation_world_to_camera=np.eye(3),
                translation_world_to_camera=np.asarray((-1.0, -2.0, -3.0)),
                reprojection_errors_px=np.asarray((0.1, 0.1, 0.1, 0.1)),
                inlier_indices=np.asarray((0, 1, 2, 3)), warnings=(),
            )
            save_extrinsic(
                output, result, calibration_mode="phy", camera_name="usb_cam",
                parent_frame="world", child_frame="usb_cam_optical_frame",
            )
            pose, selected = resolve_gazebo_camera_pose_from_file(
                str(root), "usb_cam", "world",
                ("xgc_world_camera_optical_frame", "usb_cam_optical_frame"),
                (0.067, 0.0, 0.0), str(output),
            )
            self.assertEqual(selected, str(output.resolve()))
            link_rotation = quaternion_to_rotation_matrix(result.quaternion_xyzw).dot(
                link_to_optical_rotation().T
            )
            expected_translation = result.translation - link_rotation.dot(
                np.asarray((0.067, 0.0, 0.0))
            )
            np.testing.assert_allclose([pose["x"], pose["y"], pose["z"]], expected_translation)
            np.testing.assert_allclose(
                [pose["roll"], pose["pitch"], pose["yaw"]],
                (0.0, -math.pi / 2.0, math.pi / 2.0),
                atol=1e-12,
            )

            launch = replace_roslaunch_pose_arguments(
                [
                    "/opt/ros/noetic/bin/roslaunch", "gazebo_sim_camera", "static_camera.launch",
                    "x:=0", "y:=0", "z:=0", "roll:=0", "pitch:=0", "yaw:=0",
                ],
                pose,
            )
            self.assertTrue(any(value.startswith("x:=") and value != "x:=0" for value in launch))

    def test_fails_closed_without_complete_launch_args(self):
        with self.assertRaisesRegex(CalibrationError, "exactly one yaw"):
            replace_roslaunch_pose_arguments(
                ["/opt/ros/noetic/bin/roslaunch", "x:=0", "y:=0", "z:=0", "roll:=0", "pitch:=0"],
                {"x": 0, "y": 0, "z": 0, "roll": 0, "pitch": 0, "yaw": 0},
            )

    def test_resolves_simulation_partition_file_without_a_selection_pointer(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "camera"
            output = root / "sim" / "usb_cam" / "extrinsics-20260831T010000.000000Z.yaml"
            result = ExtrinsicResult(
                translation=np.asarray((1.0, 2.0, 3.0)),
                quaternion_xyzw=np.asarray((0.0, 0.0, 0.0, 1.0)),
                rotation_world_to_camera=np.eye(3),
                translation_world_to_camera=np.asarray((-1.0, -2.0, -3.0)),
                reprojection_errors_px=np.asarray((0.1, 0.1, 0.1, 0.1)),
                inlier_indices=np.asarray((0, 1, 2, 3)), warnings=(),
            )
            save_extrinsic(
                output, result, calibration_mode="sim", camera_name="usb_cam",
                parent_frame="world", child_frame="xgc_world_camera_optical_frame",
            )
            pose, selected = resolve_gazebo_camera_pose_from_file(
                str(root), "usb_cam", "world",
                ("xgc_world_camera_optical_frame", "usb_cam_optical_frame"),
                (0.067, 0.0, 0.0), str(output),
            )
            self.assertEqual(selected, str(output.resolve()))
            np.testing.assert_allclose([pose["x"], pose["y"], pose["z"]], (1.0, 2.0, 2.933))
            with self.assertRaisesRegex(CalibrationError, "required when pose source is file"):
                resolve_gazebo_camera_pose_from_file(
                    str(root), "usb_cam", "world",
                    ("xgc_world_camera_optical_frame",), (0.067, 0.0, 0.0), "",
                )
            latest = root / "sim" / "usb_cam" / "extrinsics.yaml"
            latest.write_text("schema: xgc2.camera.extrinsic.v1\n")
            with self.assertRaisesRegex(CalibrationError, "concrete extrinsics-UTC.yaml"):
                resolve_gazebo_camera_pose_from_file(
                    str(root), "usb_cam", "world",
                    ("xgc_world_camera_optical_frame",), (0.067, 0.0, 0.0), str(latest),
                )

    def test_rejects_nonfinite_extrinsic_and_link_offsets(self):
        with self.assertRaisesRegex(CalibrationError, "non-finite"):
            split_parent_to_optical_pose(
                (1.0, 2.0, 3.0), (0.0, 0.0, 0.0, 1.0), (float("nan"), 0.0, 0.0)
            )
        with self.assertRaisesRegex(CalibrationError, "non-finite"):
            split_parent_to_optical_pose(
                (float("inf"), 2.0, 3.0), (0.0, 0.0, 0.0, 1.0), (0.0, 0.0, 0.0)
            )


if __name__ == "__main__":
    unittest.main()
