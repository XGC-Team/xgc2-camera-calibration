#!/usr/bin/env python3

import unittest

import cv2
import numpy as np

from xgc_camera_calibration.intrinsic_pose_coverage import (
    estimate_plane_orientation,
    estimate_provisional_camera_matrix,
    signed_tilt_bins,
)


CAMERA_MATRIX = np.asarray(
    ((1344.398473, 0.0, 1919.5), (0.0, 1344.398473, 1079.5), (0.0, 0.0, 1.0)),
    dtype=np.float64,
)
OBJECT_POINTS = np.asarray(
    [
        (0.0, 0.0, 0.0),
        (0.7, 0.0, 0.0),
        (0.7, 0.5, 0.0),
        (0.0, 0.5, 0.0),
        (0.2, 0.1, 0.0),
        (0.5, 0.4, 0.0),
    ],
    dtype=np.float64,
)


def orientation(rx=0.0, ry=0.0, rz=0.0, indices=None):
    points = OBJECT_POINTS if indices is None else OBJECT_POINTS[indices]
    projected, _jacobian = cv2.projectPoints(
        points,
        np.asarray((rx, ry, rz), dtype=np.float64),
        np.asarray((-0.35, -0.25, 3.0), dtype=np.float64),
        CAMERA_MATRIX,
        np.zeros(5, dtype=np.float64),
    )
    return estimate_plane_orientation(points, projected, CAMERA_MATRIX)


class IntrinsicPoseCoverageTest(unittest.TestCase):
    def test_planar_seed_views_initialize_coverage_camera_matrix(self):
        object_sets = []
        image_sets = []
        for rvec, tvec in (
            ((0.0, 0.0, 0.0), (-0.35, -0.25, 3.0)),
            ((0.2, 0.0, 0.0), (-0.2, -0.2, 2.7)),
            ((-0.2, 0.1, 0.0), (-0.4, -0.1, 3.4)),
            ((0.0, -0.25, 0.1), (-0.1, -0.4, 3.1)),
        ):
            projected, _jacobian = cv2.projectPoints(
                OBJECT_POINTS,
                np.asarray(rvec, dtype=np.float64),
                np.asarray(tvec, dtype=np.float64),
                CAMERA_MATRIX,
                np.zeros(5, dtype=np.float64),
            )
            object_sets.append(OBJECT_POINTS)
            image_sets.append(projected)
        provisional = estimate_provisional_camera_matrix(
            object_sets, image_sets, (3840, 2160), aspect_ratio=1.0
        )
        np.testing.assert_allclose(provisional, CAMERA_MATRIX, atol=0.01)

    def test_in_plane_roll_does_not_create_out_of_plane_tilt(self):
        clockwise = orientation(rz=np.deg2rad(45.0))
        counter_clockwise = orientation(rz=np.deg2rad(-45.0))
        for observed in (clockwise, counter_clockwise):
            self.assertAlmostEqual(observed.tilt_x_degrees, 0.0, delta=0.01)
            self.assertAlmostEqual(observed.tilt_y_degrees, 0.0, delta=0.01)
            self.assertAlmostEqual(abs(observed.roll_degrees), 45.0, delta=0.01)
        self.assertFalse(signed_tilt_bins((clockwise, counter_clockwise), 10.0)["complete"])

    def test_mirrored_plane_tilts_fill_distinct_signed_bins(self):
        observations = (
            orientation(ry=np.deg2rad(-18.0)),
            orientation(ry=np.deg2rad(18.0)),
            orientation(rx=np.deg2rad(-16.0)),
            orientation(rx=np.deg2rad(16.0)),
        )
        bins = signed_tilt_bins(observations, 10.0)
        self.assertEqual(bins, {
            "x_negative": True,
            "x_positive": True,
            "y_negative": True,
            "y_positive": True,
            "complete": True,
        })

    def test_partial_target_preserves_plane_normal_and_sign(self):
        complete = orientation(rx=np.deg2rad(14.0), ry=np.deg2rad(-17.0))
        partial = orientation(
            rx=np.deg2rad(14.0),
            ry=np.deg2rad(-17.0),
            indices=np.asarray((0, 1, 2, 4), dtype=np.int64),
        )
        self.assertAlmostEqual(partial.tilt_x_degrees, complete.tilt_x_degrees, delta=0.01)
        self.assertAlmostEqual(partial.tilt_y_degrees, complete.tilt_y_degrees, delta=0.01)
        self.assertLess(partial.homography_rms_px, 1e-3)


if __name__ == "__main__":
    unittest.main()
