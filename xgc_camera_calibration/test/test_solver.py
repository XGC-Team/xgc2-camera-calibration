#!/usr/bin/env python3

import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from xgc_camera_calibration.solver import CalibrationError, load_extrinsic, save_extrinsic, solve_extrinsic
from xgc_camera_calibration.transforms import (
    link_to_optical_rotation,
    quaternion_to_rotation_matrix,
    split_parent_to_optical_pose,
)


class ExtrinsicSolverTest(unittest.TestCase):
    def setUp(self):
        self.world = np.array(
            [
                [-1.0, -0.7, 0.0],
                [1.0, -0.7, 0.1],
                [1.1, 0.8, -0.1],
                [-0.9, 0.9, 0.2],
                [-0.6, -0.4, 1.0],
                [0.8, -0.5, 1.2],
                [0.7, 0.7, 0.9],
                [-0.7, 0.6, 1.1],
            ],
            dtype=np.float64,
        )
        self.intrinsic = np.array([[680.0, 0.0, 320.0], [0.0, 675.0, 240.0], [0.0, 0.0, 1.0]])
        self.rvec = np.array([0.12, -0.08, 0.04], dtype=np.float64)
        self.tvec = np.array([0.15, -0.2, 4.5], dtype=np.float64)
        self.pixels, _ = cv2.projectPoints(
            self.world.reshape(-1, 1, 3), self.rvec, self.tvec, self.intrinsic, np.zeros(5)
        )
        self.pixels = self.pixels.reshape(-1, 2)

    def test_recovers_pose_and_rejects_outlier(self):
        observed = self.pixels.copy()
        observed += np.random.RandomState(7).normal(scale=0.1, size=observed.shape)
        observed[-1] += np.array([50.0, -35.0])
        result = solve_extrinsic(
            self.world,
            observed,
            self.intrinsic,
            np.zeros(5),
            ransac_reprojection_error_px=1.5,
        )
        expected_rotation, _ = cv2.Rodrigues(self.rvec)
        expected_translation = -expected_rotation.T.dot(self.tvec)
        np.testing.assert_allclose(result.translation, expected_translation, atol=0.01)
        self.assertNotIn(len(self.world) - 1, result.inlier_indices.tolist())
        self.assertLess(np.max(result.reprojection_errors_px[result.inlier_indices]), 0.5)

    def test_rejects_collinear_points(self):
        world = np.array([[value, 0.0, 0.0] for value in range(5)], dtype=np.float64)
        with self.assertRaises(CalibrationError):
            solve_extrinsic(world, self.pixels[:5], self.intrinsic)

    def test_recovers_pose_from_planar_robot_rows_and_rejects_outlier(self):
        world = np.array(
            [[float(x), -1.0, 0.0] for x in range(5)]
            + [[float(x), 1.0, 0.0] for x in range(5)],
            dtype=np.float64,
        )
        pixels, _ = cv2.projectPoints(
            world.reshape(-1, 1, 3), self.rvec, self.tvec, self.intrinsic, np.zeros(5)
        )
        observed = pixels.reshape(-1, 2)
        observed += np.random.RandomState(9).normal(scale=0.1, size=observed.shape)
        observed[-1] += np.array([45.0, -30.0])

        result = solve_extrinsic(
            world,
            observed,
            self.intrinsic,
            np.zeros(5),
            ransac_reprojection_error_px=1.5,
        )

        expected_rotation, _ = cv2.Rodrigues(self.rvec)
        expected_translation = -expected_rotation.T.dot(self.tvec)
        np.testing.assert_allclose(result.translation, expected_translation, atol=0.02)
        self.assertNotIn(len(world) - 1, result.inlier_indices.tolist())
        self.assertIn("world points are coplanar; include depth-separated points when possible", result.warnings)

    def test_rejects_uncalibrated_intrinsics(self):
        with self.assertRaises(CalibrationError):
            solve_extrinsic(self.world, self.pixels, np.zeros((3, 3)))

    def test_persists_versioned_result_atomically(self):
        result = solve_extrinsic(self.world, self.pixels, self.intrinsic)
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "nested" / "extrinsics-20260830T010000.000000Z.yaml"
            save_extrinsic(
                output,
                result,
                calibration_mode="phy",
                camera_name="usb_cam",
                parent_frame="map",
                child_frame="usb_cam_optical_frame",
            )
            loaded = load_extrinsic(output)
            self.assertEqual(loaded["schema"], "xgc2.camera.extrinsic.v1")
            self.assertEqual(loaded["calibration_mode"], "phy")
            self.assertEqual(loaded["camera_name"], "usb_cam")
            self.assertEqual(loaded["parent_frame"], "map")
            self.assertEqual(loaded["child_frame"], "usb_cam_optical_frame")
            np.testing.assert_allclose(loaded["translation_array"], result.translation)

            output.write_text("translation: [0, 0, 0]\nquaternion_xyzw: [0, 0, 0, 1]\n")
            with self.assertRaises(CalibrationError):
                load_extrinsic(output)

    def test_parent_link_optical_chain_recomposes_calibrated_optical_pose(self):
        result = solve_extrinsic(self.world, self.pixels, self.intrinsic)
        chain = split_parent_to_optical_pose(result.translation, result.quaternion_xyzw)
        parent_r_link = quaternion_to_rotation_matrix(chain["parent_q_link_xyzw"])
        recomposed_rotation = parent_r_link.dot(link_to_optical_rotation())
        expected_rotation = quaternion_to_rotation_matrix(result.quaternion_xyzw)
        np.testing.assert_allclose(recomposed_rotation, expected_rotation, atol=1e-12)
        np.testing.assert_allclose(
            chain["parent_t_link"] + parent_r_link.dot(chain["link_t_optical"]),
            result.translation,
            atol=1e-12,
        )
        np.testing.assert_allclose(
            link_to_optical_rotation().dot(np.array([1.0, 0.0, 0.0])),
            np.array([0.0, -1.0, 0.0]),
            atol=1e-12,
        )


if __name__ == "__main__":
    unittest.main()
