"""Publisher control-flow contracts; real ROS acceptance lives in tools/."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch

# Load native extensions before the temporary ROS module stubs are installed.
from xgc_camera_calibration import extrinsic_file_watcher, transforms  # noqa: F401


class ExtrinsicPublisherTest(unittest.TestCase):
    def setUp(self):
        self.ros = MagicMock()
        self.ros.is_shutdown.return_value = False
        path = Path(__file__).parents[1] / 'scripts' / 'extrinsic_tf_publisher.py'
        spec = importlib.util.spec_from_file_location('publisher_under_test', path)
        self.publisher = importlib.util.module_from_spec(spec)
        with patch.dict('sys.modules', {'rospy': self.ros, 'tf2_ros': MagicMock(),
                                      'geometry_msgs': MagicMock(), 'geometry_msgs.msg': MagicMock()}):
            spec.loader.exec_module(self.publisher)

    def test_invalid_live_selection_does_not_prevent_next_valid_update(self):
        watcher = MagicMock()
        revision = SimpleNamespace(path=Path('/result.yaml'), document={})
        watcher.next_revision.side_effect = [ValueError('digest mismatch'), revision]
        with patch.object(self.publisher, 'load_transform_chain', return_value=('chain',)), \
             patch.object(self.publisher.time, 'sleep') as sleep:
            result = self.publisher.wait_for_transform_chain('/root', 'phy', 'cam', watcher, True, .2)
        self.assertEqual(result, (('chain',), revision.path))
        sleep.assert_called_once_with(.2)
        self.ros.set_param.assert_called_with('~extrinsic_update_error', 'digest mismatch')
        self.ros.Rate.assert_not_called()

    def test_invalid_initial_selection_fails_instead_of_defaulting(self):
        watcher = MagicMock()
        watcher.next_revision.side_effect = ValueError('digest mismatch')
        with self.assertRaisesRegex(ValueError, 'digest mismatch'):
            self.publisher.wait_for_transform_chain('/root', 'phy', 'cam', watcher, False, .2,
                                                    default_transforms=('default',))

    def test_no_file_uses_explicit_default_without_writing_selection(self):
        watcher = MagicMock()
        watcher.next_revision.return_value = None
        result = self.publisher.wait_for_transform_chain('/root', 'phy', 'cam', watcher, False, .2,
                                                       default_transforms=('default',))
        self.assertEqual(result, (('default',), None))
        self.ros.set_param.assert_not_called()

    def test_invalid_transform_does_not_poison_next_live_revision(self):
        watcher = MagicMock()
        revision = SimpleNamespace(path=Path('/result.yaml'), document={})
        watcher.next_revision.side_effect = [revision, revision]
        with patch.object(self.publisher, 'load_transform_chain', side_effect=[ValueError('frame mismatch'), ('chain',)]), \
             patch.object(self.publisher.time, 'sleep'):
            result = self.publisher.wait_for_transform_chain('/root', 'phy', 'cam', watcher, True, .2)
        self.assertEqual(result, (('chain',), revision.path))


if __name__ == '__main__':
    unittest.main()
