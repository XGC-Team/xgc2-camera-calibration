"""Planar solver regressions with projected inputs, not camera/ROS acceptance."""

import itertools
import json
import random
import tempfile
import threading
import urllib.error
import urllib.request
import unittest
from pathlib import Path

import cv2
import numpy as np

from xgc_camera_calibration import solver
from xgc_camera_calibration.web_service import (
    CalibrationHttpServer, CalibrationService, FrameSnapshot, MarkerObservation,
)


def scene(count=16, distortion=None, tilted=False):
    world = np.column_stack((
        np.mgrid[0:4, 0:count // 4].T.reshape(-1, 2) * 0.3,
        np.zeros(count),
    ))
    if tilted:
        world = world @ cv2.Rodrigues(np.array([0.35, 0.1, -0.2]))[0].T
        world += np.array([0.1, -0.2, 0.3])
    intrinsic = np.array([[800., 0., 640.], [0., 800., 480.], [0., 0., 1.]])
    coefficients = np.zeros(5) if distortion is None else np.asarray(distortion)
    rvec = np.array([0.2, -0.15, 0.1])
    tvec = np.array([-0.45, -0.45, 2.5])
    rotation = cv2.Rodrigues(rvec)[0]
    pixels = cv2.projectPoints(world, rvec, tvec, intrinsic, coefficients)[0].reshape(-1, 2)
    return world, pixels, intrinsic, coefficients, rotation, -rotation.T @ tvec


class PlanarRansacTest(unittest.TestCase):
    def assert_pose(self, result, rotation, position, expected_inliers, tolerance=1e-7):
        self.assertEqual(set(result.inlier_indices), set(expected_inliers))
        np.testing.assert_allclose(result.translation, position, atol=tolerance, rtol=0)
        np.testing.assert_allclose(result.rotation_world_to_camera, rotation, atol=tolerance, rtol=0)
        self.assertLess(float(max(result.reprojection_errors_px[result.inlier_indices])), 1.)

    def test_first_middle_last_misclick(self):
        for count in (16, 24):
            world, pixels, intrinsic, distortion, rotation, position = scene(count)
            for offset in ([400., 0.], [200., 150.]):
                observed = pixels.copy()
                observed[0] += offset
                for bad_index in (0, count // 2, count - 1):
                    with self.subTest(count=count, offset=offset, bad_index=bad_index):
                        order = list(range(1, count))
                        order.insert(bad_index, 0)
                        result = solver.solve_extrinsic(
                            world[order], observed[order], intrinsic, distortion,
                            maximum_accepted_error_px=5.,
                        )
                        self.assert_pose(result, rotation, position,
                                         set(range(count)) - {bad_index})

    def test_permutations_do_not_hide_majority_consensus(self):
        world, pixels, intrinsic, distortion, rotation, position = scene(24)
        pixels[0] += [400., 0.]
        for seed in range(8):
            with self.subTest(seed=seed):
                order = np.random.RandomState(seed).permutation(len(world))
                result = solver.solve_extrinsic(world[order], pixels[order], intrinsic, distortion)
                self.assert_pose(result, rotation, position, np.flatnonzero(order != 0))

    def test_noisy_distorted_tilted_plane_with_multiple_outliers(self):
        world, pixels, intrinsic, distortion, rotation, position = scene(
            24, [-0.12, 0.025, 0.001, -0.002, 0.005], tilted=True,
        )
        pixels += np.random.RandomState(17).normal(0., 0.08, pixels.shape)
        bad = {0, 9, 23}
        pixels[list(bad)] += [70., -50.]
        result = solver.solve_extrinsic(world, pixels, intrinsic, distortion,
                                        ransac_reprojection_error_px=1.5)
        self.assert_pose(result, rotation, position, set(range(24)) - bad, tolerance=0.015)

    def test_all_inliers_and_minimal_four_point_set(self):
        world, pixels, intrinsic, distortion, rotation, position = scene()
        for indices in (np.arange(16), np.array([0, 3, 12, 15])):
            with self.subTest(count=len(indices)):
                result = solver.solve_extrinsic(world[indices], pixels[indices], intrinsic, distortion)
                self.assert_pose(result, rotation, position, range(len(indices)))

    def test_geometry_and_quality_gates_remain_fail_closed(self):
        world, pixels, intrinsic, distortion, _, _ = scene()
        with self.assertRaisesRegex(solver.CalibrationError, 'collinear'):
            solver.solve_extrinsic(world[:4], pixels[:4], intrinsic)
        with self.assertRaises(solver.CalibrationError):
            solver.solve_extrinsic(world, np.tile([640., 480.], (16, 1)), intrinsic)
        noisy = pixels + np.random.RandomState(2).normal(0., 0.1, pixels.shape)
        with self.assertRaisesRegex(solver.CalibrationError, 'exceeds'):
            solver.solve_extrinsic(world, noisy, intrinsic, distortion,
                                    maximum_accepted_error_px=0.001)
        for confidence in (0., 1., float('nan')):
            with self.subTest(confidence=confidence), self.assertRaises(solver.CalibrationError):
                solver.solve_extrinsic(world, pixels, intrinsic, confidence=confidence)

    def test_sampling_is_unique_bounded_and_covers_complete_small_spaces(self):
        for count in range(4, 9):
            expected = set(itertools.combinations(range(count), 4))
            actual = list(solver._planar_sample_indices(count, len(expected) + 3))
            self.assertEqual(len(actual), len(expected))
            self.assertEqual(set(actual), expected)
        actual = list(solver._planar_sample_indices(1000, 300))
        self.assertEqual(len(actual), 300)
        self.assertEqual(len(set(actual)), 300)
        self.assertTrue(all(len(set(indices)) == 4 and 0 <= min(indices) < max(indices) < 1000
                            for indices in actual))
        self.assertTrue(any(0 not in indices for indices in actual))
        self.assertTrue(any(max(indices) > 500 for indices in actual))

    def test_sampling_is_repeatable_without_global_random_side_effects(self):
        state = random.getstate()
        first = list(solver._planar_sample_indices(24, 300))
        self.assertEqual(first, list(solver._planar_sample_indices(24, 300)))
        self.assertEqual(state, random.getstate())

    def test_nonplanar_solver_is_preserved(self):
        world, _, intrinsic, distortion, rotation, position = scene()
        world[::2, 2] = 0.3
        pixels = cv2.projectPoints(world, cv2.Rodrigues(rotation)[0], -rotation @ position,
                                  intrinsic, distortion)[0].reshape(-1, 2)
        result = solver.solve_extrinsic(world, pixels, intrinsic, distortion)
        self.assert_pose(result, rotation, position, range(len(world)), tolerance=1e-5)

    def test_http_candidate_then_explicit_save_uses_recovered_consensus(self):
        world, pixels, intrinsic, distortion, _, position = scene()
        pixels[0] += [400., 0.]
        markers = {str(index): MarkerObservation(str(index), tuple(point), 'world')
                   for index, point in enumerate(world)}
        snapshot = FrameSnapshot(np.zeros((960, 1280, 3), dtype=np.uint8), 123.,
                                 'camera_optical', intrinsic, distortion, markers)

        class ProjectedFrameSource:
            # Only the frame-source boundary is a fixture. The real service,
            # OpenCV solver, JPEG encoder and filesystem persistence run below.
            image_topic = '/camera/image'
            intrinsic_file = ''
            pose_prefix = '/markers'

            def freeze(self, parent_frame):
                assert parent_frame == 'world'
                return snapshot

            def status(self):
                return {'ready': True}

        with tempfile.TemporaryDirectory() as root:
            service = CalibrationService(ProjectedFrameSource(), calibration_root=root,
                                         calibration_mode='sim', camera_name='camera',
                                         parent_frame='world', child_frame='camera_optical')
            web_root = Path(__file__).resolve().parents[1] / 'web' / 'extrinsic'
            server = CalibrationHttpServer(
                ('127.0.0.1', 0), service, web_root, frame_ancestors="'self'",
            )
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            base = 'http://127.0.0.1:{}'.format(server.server_address[1])

            def post(path, body):
                request = urllib.request.Request(
                    base + path, data=json.dumps(body).encode('utf-8'),
                    headers={'Content-Type': 'application/json'}, method='POST',
                )
                with urllib.request.urlopen(request, timeout=3) as response:
                    self.assertEqual(response.status, 200)
                    return json.loads(response.read().decode('utf-8'))

            try:
                frozen = post('/api/v1/freeze', {})
                candidate = post('/api/v1/solve', {'generation': frozen['generation'], 'points': [
                    {'marker': str(index), 'pixel': pixel.tolist()} for index, pixel in enumerate(pixels)
                ]})
                self.assertFalse(candidate['saved'])
                self.assertEqual(list(Path(root).rglob('*.yaml')), [])
                self.assertEqual(set(candidate['inlier_indices']), set(range(1, 16)))
                np.testing.assert_allclose(candidate['translation'], position, atol=1e-7, rtol=0)
                with self.assertRaises(urllib.error.HTTPError) as rejected:
                    post('/api/v1/save', {'candidate_id': 'wrong-candidate'})
                self.assertEqual(rejected.exception.code, 409)
                self.assertEqual(list(Path(root).rglob('*.yaml')), [])
                saved = post('/api/v1/save', {'candidate_id': candidate['candidate_id']})
                output = Path(saved['output_file'])
                self.assertEqual(output.parent, Path(root) / 'sim' / 'camera')
                self.assertRegex(output.name, r'^extrinsics-\d{8}T\d{6}\.\d{6}Z(?:-\d{2})?\.yaml$')
                document = solver.load_extrinsic(output)
                np.testing.assert_allclose(document['translation_array'], position, atol=1e-7, rtol=0)
                self.assertFalse(document['points'][0]['inlier'])
                self.assertEqual(len(document['inlier_indices']), 15)
                repeated = post('/api/v1/save', {'candidate_id': candidate['candidate_id']})
                self.assertEqual(repeated['output_file'], str(output))
            finally:
                server.shutdown()
                server.server_close()
                thread.join(timeout=3)
                self.assertFalse(thread.is_alive())


if __name__ == '__main__':
    unittest.main()
