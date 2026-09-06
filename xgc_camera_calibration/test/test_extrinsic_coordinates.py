import copy
import unittest
import numpy as np
from xgc_camera_calibration.extrinsic_coordinates import coordinate_provenance, optical_translation_in_world
from xgc_camera_calibration.solver import CalibrationError


class ExtrinsicCoordinatesTest(unittest.TestCase):
    def document(self, provenance=None):
        return {'translation_array': np.asarray([1., 2., 3.]), 'parent_frame': 'world',
                'metadata': {} if provenance is None else {'pose_coordinates': provenance}}

    def test_raw_vrpn_applies_offset_once_without_mutating_source(self):
        document = self.document(coordinate_provenance('raw-vrpn', 'world'))
        before = copy.deepcopy(document)
        for _ in range(2):
            np.testing.assert_allclose(optical_translation_in_world(document, [-10., 4., .5]), [-9., 6., 3.5])
        np.testing.assert_array_equal(document['translation_array'], before['translation_array'])
        self.assertEqual(document['metadata'], before['metadata'])

    def test_rebase_saved_world_subtracts_old_offset_before_adding_new(self):
        document = self.document(coordinate_provenance('experiment-world', 'world', [10., -2., 4.]))
        np.testing.assert_allclose(optical_translation_in_world(document, [20., 5., 1.]), [11., 9., 0.])
        np.testing.assert_allclose(optical_translation_in_world(document, [10., -2., 4.]), [1., 2., 3.])

    def test_legacy_does_not_invent_coordinate_origin(self):
        document = self.document()
        np.testing.assert_allclose(optical_translation_in_world(document, [0., 0., 0.]), [1., 2., 3.])
        with self.assertRaisesRegex(CalibrationError, 'provenance is missing'):
            optical_translation_in_world(document, [1., 0., 0.])

    def test_malformed_or_relabelled_provenance_is_rejected(self):
        for provenance in [[], {'schema_version': 2}, coordinate_provenance('raw-vrpn', 'map'),
                           {'schema_version': 1, 'kind': 'raw-vrpn', 'frame': 'world', 'world_offset': [1, 0, 0]}]:
            with self.subTest(provenance=provenance), self.assertRaises(CalibrationError):
                optical_translation_in_world(self.document(provenance), [0., 0., 0.])
        for target in [[float('nan'), 0, 0], [1, 2], 'unknown']:
            with self.assertRaises(CalibrationError):
                optical_translation_in_world(self.document(), target)


if __name__ == '__main__':
    unittest.main()
