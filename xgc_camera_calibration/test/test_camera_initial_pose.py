#!/usr/bin/env python3

import tempfile
import unittest
from pathlib import Path

import numpy as np

from xgc_camera_calibration.camera_initial_pose import (
    replace_roslaunch_pose_arguments,
    resolve_gazebo_camera_initial_pose,
)
from xgc_camera_calibration.solver import (
    CalibrationError,
    ExtrinsicResult,
    save_extrinsic,
    write_extrinsic_selection,
)
from xgc_camera_calibration.transforms import (
    link_to_optical_rotation,
    quaternion_to_rotation_matrix,
    split_parent_to_optical_pose,
)


class CameraInitialPoseTest(unittest.TestCase):
    def test_resolves_physical_optical_pose_before_roslaunch(self):
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
            candidate = "extrinsic-candidate-physical"
            save_extrinsic(
                output, result, calibration_mode="phy", camera_name="usb_cam",
                parent_frame="world", child_frame="usb_cam_optical_frame",
                metadata={"candidate_id": candidate},
            )
            write_extrinsic_selection(str(root), "phy", "usb_cam", output, candidate)

            pose, selected = resolve_gazebo_camera_initial_pose(
                str(root), "usb_cam", "world", "usb_cam_optical_frame", (0.067, 0.0, 0.0)
            )
            self.assertEqual(selected, str(output.resolve()))
            link_rotation = quaternion_to_rotation_matrix(result.quaternion_xyzw).dot(
                link_to_optical_rotation().T
            )
            expected_translation = result.translation - link_rotation.dot(
                np.asarray((0.067, 0.0, 0.0))
            )
            np.testing.assert_allclose([pose["x"], pose["y"], pose["z"]], expected_translation)

            launch = replace_roslaunch_pose_arguments(
                [
                    "/opt/ros/noetic/bin/roslaunch", "gazebo_sim_camera", "static_camera.launch",
                    "x:=0", "y:=0", "z:=0", "roll:=0", "pitch:=0", "yaw:=0",
                ],
                pose,
            )
            self.assertTrue(any(value.startswith("x:=") and value != "x:=0" for value in launch))

    def test_fails_closed_without_shared_selection_or_complete_launch_args(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "camera"
            root.mkdir()
            with self.assertRaisesRegex(CalibrationError, "no shared physical"):
                resolve_gazebo_camera_initial_pose(
                    str(root), "usb_cam", "world", "usb_cam_optical_frame", (0.067, 0.0, 0.0)
                )
        with self.assertRaisesRegex(CalibrationError, "exactly one yaw"):
            replace_roslaunch_pose_arguments(
                ["/opt/ros/noetic/bin/roslaunch", "x:=0", "y:=0", "z:=0", "roll:=0", "pitch:=0"],
                {"x": 0, "y": 0, "z": 0, "roll": 0, "pitch": 0, "yaw": 0},
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
