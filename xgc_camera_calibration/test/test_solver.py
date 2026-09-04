#!/usr/bin/env python3

import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from xgc_camera_calibration.solver import (
    CalibrationError,
    extrinsic_selection_path,
    load_extrinsic_selection,
    load_extrinsic,
    optional_selected_intrinsic_path,
    save_extrinsic,
    selected_extrinsic_path,
    selected_intrinsic_path,
    solve_extrinsic,
    write_extrinsic_selection,
)
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
        with self.assertRaisesRegex(CalibrationError, "world points are collinear or coincident") as raised:
            solve_extrinsic(world, self.pixels[:5], self.intrinsic)
        self.assertNotRegex(str(raised.exception), r"UAV|UGV|rows")

    def test_recovers_pose_from_same_kind_planar_hexagon(self):
        world = np.array(
            [
                [1.2, -1.5, 0.0],
                [0.6, -0.4608, 0.0],
                [-0.6, -0.4608, 0.0],
                [-1.2, -1.5, 0.0],
                [-0.6, -2.5392, 0.0],
                [0.6, -2.5392, 0.0],
            ],
            dtype=np.float64,
        )
        pixels, _ = cv2.projectPoints(
            world.reshape(-1, 1, 3), self.rvec, self.tvec, self.intrinsic, np.zeros(5)
        )
        observed = pixels.reshape(-1, 2)
        observed += np.random.RandomState(3).normal(scale=0.1, size=observed.shape)
        result = solve_extrinsic(
            world, observed, self.intrinsic, np.zeros(5), ransac_reprojection_error_px=1.5,
        )
        expected_rotation, _ = cv2.Rodrigues(self.rvec)
        expected_translation = -expected_rotation.T.dot(self.tvec)
        np.testing.assert_allclose(result.translation, expected_translation, atol=0.02)
        self.assertEqual(len(result.inlier_indices), len(world))

    def test_planar_pose_failure_is_geometry_only(self):
        world = np.array(
            [
                [1.2, -1.5, 0.0],
                [0.6, -0.4608, 0.0],
                [-0.6, -0.4608, 0.0],
                [-1.2, -1.5, 0.0],
                [-0.6, -2.5392, 0.0],
                [0.6, -2.5392, 0.0],
            ],
            dtype=np.float64,
        )
        pixels = np.tile(np.array([[320.0, 240.0]], dtype=np.float64), (len(world), 1))
        with self.assertRaisesRegex(
            CalibrationError,
            "could not estimate a planar camera pose from the selected markers",
        ) as raised:
            solve_extrinsic(world, pixels, self.intrinsic)
        self.assertNotRegex(str(raised.exception), r"UAV|UGV|rows")

    def test_recovers_pose_from_planar_markers_and_rejects_outlier(self):
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

    def test_selects_only_concrete_intrinsic_in_explicit_camera_partition(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "camera"
            selected_directory = root / "phy" / "usb_cam"
            other_directory = root / "phy" / "outside"
            sim_directory = root / "sim" / "usb_cam"
            for directory in (selected_directory, other_directory, sim_directory):
                directory.mkdir(parents=True, exist_ok=True)
            filename = "intrinsics-20260830T010203.000000Z.yaml"
            selected_file = selected_directory / filename
            selected_file.write_text("schema: xgc2.camera.intrinsic.v1\n", encoding="utf-8")
            other_file = other_directory / filename
            other_file.write_text("schema: xgc2.camera.intrinsic.v1\n", encoding="utf-8")
            sim_file = sim_directory / filename
            sim_file.write_text("schema: xgc2.camera.intrinsic.v1\n", encoding="utf-8")
            outside_symlink = (
                selected_directory / "intrinsics-20260830T020304.000000Z.yaml"
            )
            outside_symlink.symlink_to(other_file)

            self.assertEqual(
                selected_intrinsic_path(str(root), "phy", "usb_cam", str(selected_file)),
                selected_file,
            )
            for invalid in (
                selected_directory / "intrinsics.yaml",
                sim_file,
                other_file,
                selected_directory / "../outside" / filename,
                outside_symlink,
            ):
                with self.assertRaises(ValueError):
                    selected_intrinsic_path(str(root), "phy", "usb_cam", str(invalid))
            for invalid_camera_name in ("../outside", "usb/cam", "usb cam", ".usb_cam"):
                with self.assertRaisesRegex(ValueError, "camera name"):
                    selected_intrinsic_path(
                        str(root), "phy", invalid_camera_name, str(selected_file)
                    )
            self.assertIsNone(
                optional_selected_intrinsic_path(str(root), "phy", "usb_cam", "")
            )
            self.assertIsNone(
                optional_selected_intrinsic_path(str(root), "phy", "usb_cam", "  ")
            )
            self.assertEqual(
                optional_selected_intrinsic_path(
                    str(root), "phy", "usb_cam", str(selected_file)
                ),
                selected_file,
            )
            with self.assertRaises(ValueError):
                optional_selected_intrinsic_path(
                    str(root), "phy", "usb_cam", str(other_file)
                )

    def test_default_identity_optical_pose_splits_into_rep103_link(self):
        chain = split_parent_to_optical_pose(
            [0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]
        )
        parent_r_link = quaternion_to_rotation_matrix(chain["parent_q_link_xyzw"])
        recomposed = parent_r_link.dot(link_to_optical_rotation())
        np.testing.assert_allclose(recomposed, np.eye(3), atol=1e-12)
        np.testing.assert_allclose(chain["parent_t_link"], [0.0, 0.0, 0.0])
        np.testing.assert_allclose(chain["link_t_optical"], [0.0, 0.0, 0.0])

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
        np.testing.assert_allclose(chain["link_t_optical"], [0.0, 0.0, 0.0])

        gazebo_chain = split_parent_to_optical_pose(
            result.translation, result.quaternion_xyzw, [0.067, 0.0, 0.0]
        )
        gazebo_parent_r_link = quaternion_to_rotation_matrix(
            gazebo_chain["parent_q_link_xyzw"]
        )
        np.testing.assert_allclose(
            gazebo_chain["parent_t_link"]
            + gazebo_parent_r_link.dot(gazebo_chain["link_t_optical"]),
            result.translation,
            atol=1e-12,
        )
        np.testing.assert_allclose(
            link_to_optical_rotation().dot(np.array([1.0, 0.0, 0.0])),
            np.array([0.0, -1.0, 0.0]),
            atol=1e-12,
        )

    def test_shared_selection_resolves_one_exact_immutable_extrinsic(self):
        result = solve_extrinsic(self.world, self.pixels, self.intrinsic)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "camera"
            output = root / "phy" / "usb_cam" / "extrinsics-20260830T010000.000000Z.yaml"
            candidate_id = "extrinsic-candidate-selection"
            save_extrinsic(
                output, result, calibration_mode="phy", camera_name="usb_cam",
                parent_frame="world", child_frame="usb_cam_optical_frame",
                metadata={"candidate_id": candidate_id},
            )
            pointer = write_extrinsic_selection(
                str(root), "phy", "usb_cam", output, candidate_id
            )
            self.assertEqual(
                pointer,
                extrinsic_selection_path(str(root), "phy", "usb_cam"),
            )
            selected, document, selection = load_extrinsic_selection(
                str(root), "phy", "usb_cam"
            )
            self.assertEqual(selected, output.resolve())
            self.assertEqual(document["metadata"]["candidate_id"], candidate_id)
            self.assertEqual(selection["relative_path"], "phy/usb_cam/" + output.name)
            self.assertEqual(
                selected_extrinsic_path(str(root), "phy", "usb_cam", str(output)),
                output.resolve(),
            )

            output.write_text(output.read_text(encoding="utf-8") + "# drift\n", encoding="utf-8")
            with self.assertRaisesRegex(CalibrationError, "digest does not match"):
                load_extrinsic_selection(str(root), "phy", "usb_cam")

    def test_shared_selection_rejects_alias_or_cross_partition_result(self):
        result = solve_extrinsic(self.world, self.pixels, self.intrinsic)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "camera"
            physical = root / "phy" / "usb_cam" / "extrinsics-20260830T010000.000000Z.yaml"
            simulation = root / "sim" / "usb_cam" / physical.name
            for output, mode in ((physical, "phy"), (simulation, "sim")):
                save_extrinsic(
                    output, result, calibration_mode=mode, camera_name="usb_cam",
                    parent_frame="world", child_frame="usb_cam_optical_frame",
                    metadata={"candidate_id": "candidate-" + mode},
                )
            alias = physical.parent / "extrinsics.yaml"
            alias.symlink_to(physical)
            timestamp_alias = (
                physical.parent / "extrinsics-20260830T030000.000000Z.yaml"
            )
            timestamp_alias.symlink_to(physical)
            for invalid in (alias, timestamp_alias, simulation):
                with self.assertRaises(ValueError):
                    selected_extrinsic_path(str(root), "phy", "usb_cam", str(invalid))


if __name__ == "__main__":
    unittest.main()
