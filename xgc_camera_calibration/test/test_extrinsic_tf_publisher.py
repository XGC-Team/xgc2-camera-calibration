"""Both TF segments must publish before the application callback succeeds."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch

from xgc_camera_calibration import transforms
assert transforms is not None  # Load native extensions before ROS stubs.


class ExtrinsicPublisherTest(unittest.TestCase):
    def setUp(self):
        self.ros = MagicMock()
        path = Path(__file__).parents[1] / 'scripts' / 'extrinsic_tf_publisher.py'
        spec = importlib.util.spec_from_file_location('publisher_under_test', path)
        self.publisher = importlib.util.module_from_spec(spec)
        with patch.dict('sys.modules', {'rospy': self.ros, 'tf2_ros': MagicMock(),
                'geometry_msgs': MagicMock(), 'geometry_msgs.msg': MagicMock()}):
            spec.loader.exec_module(self.publisher)
        self.chain = tuple(SimpleNamespace(header=SimpleNamespace(stamp=None)) for _ in range(2))

    def test_static_sends_both_segments_with_one_stamp(self):
        broadcaster = MagicMock()
        self.publisher.publish_chain(self.chain, broadcaster)
        broadcaster.sendTransform.assert_called_once_with(list(self.chain))
        self.assertIs(self.chain[0].header.stamp, self.chain[1].header.stamp)

    def test_dynamic_cannot_return_success_if_either_segment_fails(self):
        for failed in (0, 1):
            dynamic, optical = MagicMock(), MagicMock()
            (dynamic if failed == 0 else optical).sendTransform.side_effect = OSError('send failed')
            with self.assertRaises(OSError): self.publisher.publish_chain(self.chain, dynamic, optical)
            if failed == 1: dynamic.sendTransform.assert_not_called()

    def test_frozen_pose_is_consumed_without_another_world_offset(self):
        frozen = {'resolvedOpticalPose': {'translation': [7., 8., 9.], 'quaternionXyzw': [0., 0., 0., 1.]},
                  'targetCoordinates': {'worldOffset': [100., 200., 300.]}}
        with patch.object(self.publisher, 'default_transform_chain', return_value='chain') as convert:
            result = self.publisher.frozen_transform_chain(frozen, 'world', 'link', 'optical', (.067, 0, 0))
        self.assertEqual(result, 'chain')
        convert.assert_called_once_with('world', 'link', 'optical', [7., 8., 9.], [0., 0., 0., 1.],
                                        link_to_optical_translation=(.067, 0, 0))
