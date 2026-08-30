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

import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from xgc_camera_calibration import intrinsic_solver as solver
from xgc_camera_calibration.intrinsic_service import IntrinsicCalibrationService


IMAGE_SIZE = (1920, 1080)
BOARD_SIZE = (6, 6)
FIELD_TAG_SIZE_M = 0.088
FIELD_TAG_GAP_M = 0.0264
A4_TAG_SIZE_M = 0.024
A4_TAG_GAP_M = 0.0072
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

# The two supported profiles share the standard Kalibr tag datum but are not
# aliases for one synthetic board.  The A4 camera is physically closer while
# its active grid still projects about 15% smaller than the field board.  This
# catches a profile implementation that changes only a label or accidentally
# reuses the field metric geometry.
BOARD_PROFILES = (
    (
        "field_6x6_88mm_30pct",
        FIELD_TAG_SIZE_M,
        FIELD_TAG_GAP_M,
        1.0,
    ),
    (
        "a4_6x6_24mm_30pct_kalibr_v1",
        A4_TAG_SIZE_M,
        A4_TAG_GAP_M,
        0.32,
    ),
)


def _render_station_aprilgrid():
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
    # This is the one printable/Gazebo/field datum, not OpenCV's convenient
    # default marker layout: ID 0 is lower-left, IDs advance +X then +Y, each
    # tag uses Kalibr rotation=2, and all 7x7 locator blocks are present.
    pitch = TAG_PIXELS + GAP_PIXELS
    printed_side = (
        BOARD_SIZE[0] * TAG_PIXELS + (BOARD_SIZE[0] + 1) * GAP_PIXELS
    )
    side = printed_side + 2 * BOARD_MARGIN_PIXELS
    board = np.full((side, side), 255, dtype=np.uint8)
    for row in range(BOARD_SIZE[1] + 1):
        for col in range(BOARD_SIZE[0] + 1):
            y0 = BOARD_MARGIN_PIXELS + row * pitch
            x0 = BOARD_MARGIN_PIXELS + col * pitch
            board[y0:y0 + GAP_PIXELS, x0:x0 + GAP_PIXELS] = 0
    for visual_row in range(BOARD_SIZE[1]):
        for col in range(BOARD_SIZE[0]):
            tag_row = BOARD_SIZE[1] - 1 - visual_row
            tag_id = tag_row * BOARD_SIZE[0] + col
            if hasattr(cv2.aruco, "generateImageMarker"):
                marker = cv2.aruco.generateImageMarker(
                    dictionary, tag_id, TAG_PIXELS, None, 2
                )
            else:
                marker = cv2.aruco.drawMarker(
                    dictionary, tag_id, TAG_PIXELS, borderBits=2
                )
            marker = np.rot90(marker, 2)
            y0 = BOARD_MARGIN_PIXELS + GAP_PIXELS + visual_row * pitch
            x0 = BOARD_MARGIN_PIXELS + GAP_PIXELS + col * pitch
            board[y0:y0 + TAG_PIXELS, x0:x0 + TAG_PIXELS] = marker
    return board


def _render_view(
    board,
    center_pixel,
    depth_m,
    rotation_vector,
    *,
    tag_size_m,
    tag_gap_m,
):
    width, height = IMAGE_SIZE
    extent = tag_size_m + (BOARD_SIZE[0] - 1) * (tag_size_m + tag_gap_m)
    content_side = BOARD_SIZE[0] * TAG_PIXELS + (BOARD_SIZE[0] - 1) * GAP_PIXELS
    # The physical border lies on the white/black pixel boundaries, half a pixel
    # outside the first/last black pixel centres. Its continuous raster span is
    # therefore exactly ``content_side`` pixels. Mapping the physical extent to
    # the black pixel centres would shorten the oracle by one pixel and
    # manufacture a focal-scale error before the production detector runs.
    lo = float(BOARD_MARGIN_PIXELS + GAP_PIXELS) - 0.5
    hi = lo + float(content_side)
    # Object coordinates follow Kalibr: origin at the lower-left, +Y upward.
    # Raster rows point downward, so the source order matching object
    # BL/BR/TR/TL is bottom-left, bottom-right, top-right, top-left.
    source = np.asarray(
        ((lo, hi), (hi, hi), (hi, lo), (lo, lo)),
        dtype=np.float32,
    )
    boundary = np.asarray(
        ((0, 0, 0), (extent, 0, 0), (extent, extent, 0), (0, extent, 0)),
        dtype=np.float32,
    )
    delta_rotation = cv2.Rodrigues(
        np.asarray(rotation_vector, dtype=np.float64)
    )[0]
    # A front-facing board with +Y upward has +Z toward the camera, whereas the
    # OpenCV camera looks along +Z with image +Y downward.  The pi-X base pose
    # preserves the printed tag payload instead of viewing a mirrored backside.
    rotation = delta_rotation.dot(np.diag((1.0, -1.0, -1.0)))
    rvec = cv2.Rodrigues(rotation)[0].reshape(3)
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
    def test_reduced_working_frame_admits_source_jpeg_correspondences(self):
        board = _render_station_aprilgrid()
        source_gray = _render_view(
            board,
            *VIEW_SPECS[9],
            tag_size_m=FIELD_TAG_SIZE_M,
            tag_gap_m=FIELD_TAG_GAP_M,
        )
        encoded, source_jpeg = cv2.imencode(
            ".jpg", source_gray, (cv2.IMWRITE_JPEG_QUALITY, 94)
        )
        self.assertTrue(encoded)
        reduced_gray = cv2.resize(
            source_gray, (640, 360), interpolation=cv2.INTER_AREA
        )
        reduced_bgr = cv2.cvtColor(reduced_gray, cv2.COLOR_GRAY2BGR)

        with tempfile.TemporaryDirectory() as directory:
            service = IntrinsicCalibrationService(
                board_size=BOARD_SIZE,
                square=FIELD_TAG_SIZE_M,
                output_file=str(Path(directory) / "intrinsics.yaml"),
                camera_name="sim_truth_camera",
                board_type="aprilgrid",
                tag_spacing=FIELD_TAG_GAP_M,
                min_tags=6,
                display_width=640,
            )
            service.process_frame(
                reduced_bgr,
                source_image_size=IMAGE_SIZE,
                source_jpeg=source_jpeg.tobytes(),
            )

            self.assertEqual(len(service.image_points), 1)
            self.assertEqual(len(service.object_points), 1)
            self.assertEqual(len(service.image_points[0]), len(service.object_points[0]))
            self.assertEqual(len(service.image_points[0]) % 4, 0)
            self.assertGreaterEqual(len(service.image_points[0]), 108)
            # Stored solve coordinates come from a fresh full-source detection,
            # not a VGA observation multiplied by a scale factor.
            stored = service.image_points[0].reshape(-1, 2)
            self.assertGreater(float(np.max(stored[:, 0])), 640.0)
            self.assertEqual(service.image_size, IMAGE_SIZE)

    def test_image_detection_chain_recovers_both_supported_board_profiles(self):
        board = _render_station_aprilgrid()
        median_image_spans = {}
        for profile_id, tag_size_m, tag_gap_m, depth_scale in BOARD_PROFILES:
            with self.subTest(profile=profile_id):
                image_points = []
                object_points = []
                image_spans = []
                for view_index, (center, depth, rotation) in enumerate(VIEW_SPECS):
                    gray = _render_view(
                        board,
                        center,
                        depth * depth_scale,
                        rotation,
                        tag_size_m=tag_size_m,
                        tag_gap_m=tag_gap_m,
                    )
                    detection = solver.detect_aprilgrid(
                        gray,
                        BOARD_SIZE,
                        square=tag_size_m,
                        tag_spacing=tag_gap_m,
                        min_tags=6,
                        # This is the service's source/adaptive AprilGrid plane.
                        # It is still bounded, but retains enough tag payload
                        # for all signed edge views before source-pixel
                        # refinement.
                        maximum_width=IMAGE_SIZE[0],
                    )
                    self.assertIsNotNone(
                        detection,
                        "{} view {} was not detected".format(profile_id, view_index),
                    )
                    self.assertIsNotNone(detection.calibration_image_points)
                    self.assertIsNotNone(detection.calibration_object_points)
                    # Strict AprilGrid calibration preserves one common mask
                    # and standard-datum ordering for image/object corners.
                    self.assertEqual(
                        len(detection.calibration_image_points),
                        len(detection.calibration_object_points),
                    )
                    self.assertEqual(len(detection.calibration_image_points) % 4, 0)
                    self.assertGreaterEqual(
                        len(detection.calibration_image_points),
                        6 * 4,
                    )
                    image_points.append(detection.calibration_image_points)
                    object_points.append(detection.calibration_object_points)
                    pixels = detection.calibration_image_points.reshape(-1, 2)
                    image_spans.append(float(np.ptp(pixels[:, 0])))

                result = solver.calibrate_intrinsic(
                    image_points,
                    BOARD_SIZE,
                    tag_size_m,
                    IMAGE_SIZE,
                    object_points=object_points,
                    observation_uncertainty=solver.observation_uncertainty_px(
                        "aprilgrid"
                    ),
                )

                focal_relative_error = np.abs(
                    np.diag(result.camera_matrix)[:2] / np.diag(TRUTH_K)[:2] - 1.0
                )
                principal_error_px = np.linalg.norm(
                    result.camera_matrix[:2, 2] - TRUTH_K[:2, 2]
                )
                self.assertLessEqual(float(np.max(focal_relative_error)), 0.005)
                self.assertTrue(
                    np.all(
                        np.abs(result.camera_matrix[:2, 2] - TRUTH_K[:2, 2])
                        <= 2.0
                    )
                )
                self.assertLess(float(principal_error_px), 1.0)
                self.assertLess(float(np.max(np.abs(result.distortion[:2]))), 0.01)
                self.assertLess(float(np.max(np.abs(result.distortion[2:4]))), 5e-4)
                if result.distortion.size > 4:
                    self.assertLess(abs(float(result.distortion[4])), 0.01)
                self.assertLessEqual(result.rms_reprojection_error_px, 0.5)

                diagnostics = result.diagnostics
                self.assertIsNotNone(diagnostics)
                self.assertTrue(diagnostics.finite)
                self.assertEqual(diagnostics.pool_sample_count, len(VIEW_SPECS))
                self.assertEqual(result.sample_count, len(VIEW_SPECS))
                self.assertEqual(
                    diagnostics.projected_intrinsic_rank,
                    diagnostics.projected_intrinsic_parameter_count,
                )
                self.assertFalse(diagnostics.projected_intrinsic_rank_deficient)
                self.assertEqual(diagnostics.rejected_views, ())
                self.assertEqual(
                    len(diagnostics.stability.folds), result.sample_count
                )
                self.assertEqual(
                    diagnostics.stability.failed_omitted_view_indices, ()
                )
                self.assertLess(
                    diagnostics.stability.held_out_rms_max_px, 1.0
                )
                self.assertLess(
                    diagnostics.stability.undistorted_ray_max_equivalent_px,
                    2.0,
                )
                median_image_spans[profile_id] = float(np.median(image_spans))

        # Different metric boards and camera distances must reach this gate as
        # different optical observations, not as profile-name aliases.
        self.assertGreater(
            median_image_spans["field_6x6_88mm_30pct"],
            1.1 * median_image_spans["a4_6x6_24mm_30pct_kalibr_v1"],
        )


if __name__ == "__main__":
    unittest.main()
