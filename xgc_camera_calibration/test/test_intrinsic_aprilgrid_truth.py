#!/usr/bin/env python3

"""End-to-end accuracy contract for the production AprilGrid observation path.

The lower-level solver test deliberately feeds perfect ``projectPoints``
correspondences.  That is useful for isolating OpenCV, but it cannot catch a
bias introduced while XGC turns detected tag pixels into calibration
correspondences.  This test renders the station's real 6x6 tag36h11 geometry
through perspective views, JPEG round-trips every frame like a Media Edge
snapshot, runs ``detect_aprilgrid``, and calibrates only from the detector's
production ``calibration_*`` outputs.
"""

import unittest

import cv2
import numpy as np

from xgc_camera_calibration import intrinsic_solver as solver


IMAGE_SIZE = (1920, 1080)
BOARD_SIZE = (6, 6)
TAG_SIZE_M = 0.088
TAG_GAP_M = 0.0264
TRUTH_FOCAL_PX = 672.199
TRUTH_K = np.asarray(
    (
        (TRUTH_FOCAL_PX, 0.0, 959.5),
        (0.0, TRUTH_FOCAL_PX, 539.5),
        (0.0, 0.0, 1.0),
    ),
    dtype=np.float64,
)
TAG_PIXELS = 160
GAP_PIXELS = 48
BOARD_MARGIN_PIXELS = 64

# Signed tilts, image-edge coverage, and near/far observations are all
# represented.  Translation is derived from the requested board-centre pixel,
# so these remain readable as an acceptance dataset rather than opaque rvec /
# tvec constants.
VIEW_SPECS = (
    ((960, 540), 1.50, (0.00, 0.00, 0.00)),
    ((350, 540), 1.40, (0.00, 0.35, 0.00)),
    ((1570, 540), 1.40, (0.00, -0.35, 0.00)),
    ((960, 240), 1.30, (-0.35, 0.00, 0.00)),
    ((960, 840), 1.30, (0.35, 0.00, 0.00)),
    ((420, 270), 1.25, (0.28, 0.30, 0.10)),
    ((1500, 270), 1.25, (0.28, -0.30, -0.10)),
    ((420, 810), 1.25, (-0.28, 0.30, -0.10)),
    ((1500, 810), 1.25, (-0.28, -0.30, 0.10)),
    ((960, 540), 0.85, (0.12, 0.08, 0.00)),
    ((960, 540), 1.90, (-0.12, -0.08, 0.00)),
    ((650, 400), 1.05, (0.30, -0.18, 0.22)),
    ((1300, 680), 1.05, (-0.30, 0.18, -0.22)),
)


def _render_station_aprilgrid():
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
    # The printed station board uses a 0.3 tag-spacing ratio.  A white margin
    # preserves the complete black border around edge tags after warping.
    pitch = TAG_PIXELS + GAP_PIXELS
    content_side = BOARD_SIZE[0] * TAG_PIXELS + (BOARD_SIZE[0] - 1) * GAP_PIXELS
    side = content_side + 2 * BOARD_MARGIN_PIXELS
    board = np.full((side, side), 255, dtype=np.uint8)
    for row in range(BOARD_SIZE[1]):
        for col in range(BOARD_SIZE[0]):
            tag_id = row * BOARD_SIZE[0] + col
            if hasattr(cv2.aruco, "generateImageMarker"):
                marker = cv2.aruco.generateImageMarker(
                    dictionary, tag_id, TAG_PIXELS, None, 2
                )
            else:
                marker = cv2.aruco.drawMarker(
                    dictionary, tag_id, TAG_PIXELS, borderBits=2
                )
            y0 = BOARD_MARGIN_PIXELS + row * pitch
            x0 = BOARD_MARGIN_PIXELS + col * pitch
            board[y0:y0 + TAG_PIXELS, x0:x0 + TAG_PIXELS] = marker
    return board


def _render_view(board, center_pixel, depth_m, rotation_vector):
    width, height = IMAGE_SIZE
    extent = TAG_SIZE_M + (BOARD_SIZE[0] - 1) * (TAG_SIZE_M + TAG_GAP_M)
    content_side = BOARD_SIZE[0] * TAG_PIXELS + (BOARD_SIZE[0] - 1) * GAP_PIXELS
    # ArUco reports the raster corner coordinates themselves (the first marker
    # spans margin..margin+TAG_PIXELS-1), so anchor exactly those coordinates to
    # the physical outer-tag extent.  Mixing pixel-edge and detected-corner
    # conventions here would manufacture a focal-scale error in the oracle.
    lo = float(BOARD_MARGIN_PIXELS)
    hi = lo + float(content_side - 1)
    source = np.asarray(
        ((lo, lo), (hi, lo), (hi, hi), (lo, hi)),
        dtype=np.float32,
    )
    boundary = np.asarray(
        ((0, 0, 0), (extent, 0, 0), (extent, extent, 0), (0, extent, 0)),
        dtype=np.float32,
    )
    rvec = np.asarray(rotation_vector, dtype=np.float64)
    rotation = cv2.Rodrigues(rvec)[0]
    board_center = np.asarray((extent / 2.0, extent / 2.0, 0.0))
    u, v = center_pixel
    camera_center = np.asarray(
        (
            (float(u) - TRUTH_K[0, 2]) * depth_m / TRUTH_K[0, 0],
            (float(v) - TRUTH_K[1, 2]) * depth_m / TRUTH_K[1, 1],
            depth_m,
        )
    )
    tvec = camera_center - rotation.dot(board_center)
    projected, _ = cv2.projectPoints(
        boundary, rvec, tvec, TRUTH_K, np.zeros(5, dtype=np.float64)
    )
    transform = cv2.getPerspectiveTransform(
        source, projected.reshape(4, 2).astype(np.float32)
    )
    gray = cv2.warpPerspective(
        board,
        transform,
        IMAGE_SIZE,
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=255,
    )
    encoded, jpeg = cv2.imencode(
        ".jpg", gray, (cv2.IMWRITE_JPEG_QUALITY, 94)
    )
    if not encoded:
        raise AssertionError("could not encode synthetic Media Edge snapshot")
    return cv2.imdecode(jpeg, cv2.IMREAD_GRAYSCALE)


@unittest.skipUnless(
    hasattr(cv2, "aruco") and hasattr(cv2.aruco, "DICT_APRILTAG_36h11"),
    "OpenCV AprilTag 36h11 dictionary is unavailable",
)
class AprilGridProductionTruthTest(unittest.TestCase):
    def test_image_detection_chain_recovers_zero_distortion_camera_truth(self):
        board = _render_station_aprilgrid()
        image_points = []
        object_points = []
        for view_index, (center, depth, rotation) in enumerate(VIEW_SPECS):
            gray = _render_view(board, center, depth, rotation)
            detection = solver.detect_aprilgrid(
                gray,
                BOARD_SIZE,
                square=TAG_SIZE_M,
                tag_spacing=TAG_GAP_M,
                min_tags=6,
                # This is the service's source/adaptive AprilGrid plane.  It is
                # still bounded, but retains enough tag payload for all signed
                # edge views before source-pixel refinement.
                maximum_width=IMAGE_SIZE[0],
            )
            self.assertIsNotNone(detection, "view {} was not detected".format(view_index))
            self.assertIsNotNone(detection.calibration_image_points)
            self.assertIsNotNone(detection.calibration_object_points)
            # AprilGrid calibration must preserve every independent tag corner.
            # Compressing four projective observations to an arithmetic centre
            # is not a pinhole-preserving operation and is the regression this
            # end-to-end fixture is intended to catch.
            self.assertEqual(
                len(detection.calibration_image_points),
                len(detection.calibration_object_points),
            )
            self.assertEqual(len(detection.calibration_image_points) % 4, 0)
            self.assertGreaterEqual(
                len(detection.calibration_image_points),
                int(0.75 * len(detection.image_points)),
            )
            image_points.append(detection.calibration_image_points)
            object_points.append(detection.calibration_object_points)

        result = solver.calibrate_intrinsic(
            image_points,
            BOARD_SIZE,
            TAG_SIZE_M,
            IMAGE_SIZE,
            object_points=object_points,
        )

        focal_relative_error = np.abs(
            np.diag(result.camera_matrix)[:2] / np.diag(TRUTH_K)[:2] - 1.0
        )
        principal_error_px = np.linalg.norm(
            result.camera_matrix[:2, 2] - TRUTH_K[:2, 2]
        )
        self.assertLess(float(np.max(focal_relative_error)), 0.01)
        self.assertLess(float(principal_error_px), 1.0)
        self.assertLess(float(np.max(np.abs(result.distortion[:2]))), 0.01)
        self.assertLess(float(np.max(np.abs(result.distortion[2:4]))), 5e-4)
        if result.distortion.size > 4:
            self.assertLess(abs(float(result.distortion[4])), 0.01)
        self.assertLess(result.rms_reprojection_error_px, 0.6)


if __name__ == "__main__":
    unittest.main()
