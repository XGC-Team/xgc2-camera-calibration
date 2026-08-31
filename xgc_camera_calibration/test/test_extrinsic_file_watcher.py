#!/usr/bin/env python3

import tempfile
import unittest
from pathlib import Path

import numpy as np

from xgc_camera_calibration.extrinsic_file_watcher import (
    ExtrinsicSelectionWatcher,
)
from xgc_camera_calibration.solver import (
    ExtrinsicResult,
    save_extrinsic,
    write_extrinsic_selection,
)


class ExtrinsicSelectionWatcherTest(unittest.TestCase):
    def test_selection_watcher_follows_pointer_not_directory_mtime(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "camera"
            selected = root / "phy" / "usb_cam" / "extrinsics-20260830T010000.000000Z.yaml"
            newer = root / "phy" / "usb_cam" / "extrinsics-20260830T020000.000000Z.yaml"
            for path, candidate in ((selected, "candidate-selected"), (newer, "candidate-newer")):
                save_extrinsic(
                    path,
                    ExtrinsicResult(
                        translation=np.asarray((1.0, 2.0, 3.0)),
                        quaternion_xyzw=np.asarray((0.0, 0.0, 0.0, 1.0)),
                        rotation_world_to_camera=np.eye(3),
                        translation_world_to_camera=np.asarray((-1.0, -2.0, -3.0)),
                        reprojection_errors_px=np.asarray((0.1, 0.2, 0.3, 0.4)),
                        inlier_indices=np.asarray((0, 1, 2, 3)),
                        warnings=(),
                    ),
                    calibration_mode="phy", camera_name="usb_cam",
                    parent_frame="world", child_frame="usb_cam_optical_frame",
                    metadata={"candidate_id": candidate},
                )
            write_extrinsic_selection(
                str(root), "phy", "usb_cam", selected, "candidate-selected"
            )
            watcher = ExtrinsicSelectionWatcher(str(root), "phy", "usb_cam")
            self.assertEqual(watcher.next_revision().path, selected.resolve())
            self.assertIsNone(watcher.next_revision())
            write_extrinsic_selection(
                str(root), "phy", "usb_cam", newer, "candidate-newer"
            )
            self.assertEqual(watcher.next_revision().path, newer.resolve())

    def test_selection_watcher_can_require_pointer_update(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "camera"
            result = root / "phy" / "usb_cam" / "extrinsics-20260830T010000.000000Z.yaml"
            save_extrinsic(
                result,
                ExtrinsicResult(
                    translation=np.zeros(3), quaternion_xyzw=np.asarray((0.0, 0.0, 0.0, 1.0)),
                    rotation_world_to_camera=np.eye(3), translation_world_to_camera=np.zeros(3),
                    reprojection_errors_px=np.asarray((0.1, 0.1, 0.1, 0.1)),
                    inlier_indices=np.asarray((0, 1, 2, 3)), warnings=(),
                ),
                calibration_mode="phy", camera_name="usb_cam",
                parent_frame="world", child_frame="usb_cam_optical_frame",
                metadata={"candidate_id": "candidate"},
            )
            write_extrinsic_selection(str(root), "phy", "usb_cam", result, "candidate")
            watcher = ExtrinsicSelectionWatcher(
                str(root), "phy", "usb_cam", require_update=True
            )
            self.assertIsNone(watcher.next_revision())


if __name__ == "__main__":
    unittest.main()
