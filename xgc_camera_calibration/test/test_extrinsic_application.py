"""Real file owner + serial publisher handshake, without ROS transport mocks."""
import copy
from pathlib import Path
import sys
from unittest.mock import Mock, patch

# catkin/nose loads this test by path without adding its sibling directory.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_extrinsic_selection import SelectionFixture, ROLES, TARGET
from xgc_camera_calibration.extrinsic_application import ExtrinsicApplication, SavedExtrinsicApplication
from xgc_camera_calibration.extrinsic_resolver import encode_frozen, resolve_selection
from xgc_camera_calibration.extrinsic_selection import SelectionConflict


class ApplicationTest(SelectionFixture):
    def setUp(self):
        super().setUp()
        self.frozen = resolve_selection(str(self.root), 'usb_cam', {'mode': 'auto'}, TARGET, ROLES, 'resolution-a')
        self.payload = encode_frozen(self.frozen)
        self.states = []
        self.owner = ExtrinsicApplication(str(self.root), 'usb_cam', self.payload, ROLES, self.states.append, 'instance-a')
        self.owner.initial_published()
        self.save = SavedExtrinsicApplication(str(self.root), 'usb_cam', self.payload, ROLES, self.owner.state)

    def test_saved_pending_is_not_applied_until_complete_chain_publishes(self):
        staged = self.save.stage(self.path, 'phy', 'candidate-a')
        self.assertEqual(staged['status'], 'pending')
        self.assertIsNone(self.store.read()['applied'])
        poses = []
        def publish(frozen):
            self.assertIsNone(self.store.read()['applied'])
            poses.append(frozen['resolvedOpticalPose'])
        self.owner.tick(publish)
        self.assertEqual(poses[0]['translation'], [4., 5., 6.])  # saved(1,2,3)+target(7,8,9)-source(4,5,6)
        self.assertEqual(self.save.status()['status'], 'applied')
        current = self.store.read()
        self.save.stage(self.path, 'phy', 'candidate-a')
        self.assertEqual(self.store.read(), current)
        self.owner.tick(lambda value: self.fail('applied must not hot-follow'))

    def test_confirm_between_stage_request_and_return_is_already_applied(self):
        original_stage = self.save.store.stage
        def stage_then_confirm(request, expected):
            original_stage(request, expected)
            self.owner.tick(lambda frozen: None)
            return original_stage(request, expected)
        with patch.object(self.save.store, 'stage', side_effect=stage_then_confirm):
            result = self.save.stage(self.path, 'phy', 'candidate-a')
        self.assertEqual(result['status'], 'applied')

    def test_failed_publish_does_not_confirm(self):
        self.save.stage(self.path, 'phy', 'candidate-a')
        with self.assertRaisesRegex(OSError, 'transport'):
            self.owner.tick(Mock(side_effect=OSError('transport')))
        self.assertIsNone(self.store.read()['applied'])
        self.assertEqual(self.save.status()['status'], 'pending')
        self.owner.tick(lambda value: None)
        self.assertEqual(self.save.status()['status'], 'applied')

    def test_confirm_io_retry_never_publishes_again(self):
        self.save.stage(self.path, 'phy', 'candidate-a')
        publish = Mock()
        with patch.object(self.owner.store, 'confirm', side_effect=OSError('directory fsync')):
            with self.assertRaises(OSError):
                self.owner.tick(publish)
        self.assertEqual(self.owner.state()['active']['status'], 'confirmation_error')
        self.assertIsNone(self.store.read()['applied'])
        self.owner.tick(publish)
        publish.assert_called_once()
        self.assertEqual(self.save.status()['status'], 'applied')

    def test_restarted_and_other_run_producers_cannot_apply_old_pending(self):
        self.save.stage(self.path, 'phy', 'candidate-a')
        restarted = ExtrinsicApplication(str(self.root), 'usb_cam', self.payload, ROLES, lambda value: None, 'instance-b')
        restarted.initial_published()
        restarted.tick(lambda value: self.fail('old epoch'))
        other = copy.deepcopy(self.frozen)
        other['resolutionId'] = 'another-run'
        another = ExtrinsicApplication(str(self.root), 'usb_cam', encode_frozen(other), ROLES, lambda value: None, 'instance-a')
        another.initial_published()
        another.tick(lambda value: self.fail('foreign resolution'))
        self.assertIsNone(self.store.read()['applied'])

    def test_late_confirm_cannot_erase_new_pending_or_republish_old(self):
        self.save.stage(self.path, 'phy', 'candidate-a')
        publish = Mock()
        with patch.object(self.owner.store, 'confirm', side_effect=OSError('fsync')):
            with self.assertRaises(OSError): self.owner.tick(publish)
        newer = self.request('new-application', 'instance-b')
        self.store.stage(newer, 1)
        with self.assertRaises(SelectionConflict): self.owner.tick(publish)
        self.owner.tick(publish)
        publish.assert_called_once()
        self.assertEqual(self.store.read()['pending']['applicationId'], 'new-application')
        self.assertEqual(self.save.status()['status'], 'conflict')

    def test_stage_retry_preserves_request_and_cas(self):
        with patch.object(self.save.store, 'stage', side_effect=OSError('disk')):
            with self.assertRaises(OSError): self.save.stage(self.path, 'phy', 'candidate-a')
        request = copy.deepcopy(self.save.request)
        self.assertEqual(self.save.status()["status"], "unavailable")
        self.save.stage(self.path, 'phy', 'candidate-a')
        self.assertEqual(self.save.request, request)
        self.assertEqual(self.save.expected_revision, 0)
        self.store.stage(self.request('newer'), 1)
        with self.assertRaises(SelectionConflict): self.save.stage(self.path, 'phy', 'candidate-a')
        self.assertEqual(self.save.request, request)

    def test_unavailable_producer_does_not_stage(self):
        self.owner.ready = False
        self.assertEqual(self.save.stage(self.path, 'phy', 'candidate-a'), {'status': 'unavailable'})
        self.assertEqual(self.store.read()['revision'], 0)
        self.owner.ready = True
        self.owner.producer['resolutionId'] = 'foreign'
        self.assertEqual(self.save.stage(self.path, 'phy', 'candidate-a'), {'status': 'unavailable'})

    def test_manual_frozen_does_not_follow_another_runs_new_global_applied(self):
        selected = resolve_selection(str(self.root), 'usb_cam', {'mode': 'version', 'result': self.result}, TARGET, ROLES, 'manual-run')
        manual = ExtrinsicApplication(str(self.root), 'usb_cam', encode_frozen(selected), ROLES, lambda value: None)
        manual.initial_published()
        self.save.stage(self.path, 'phy', 'candidate-a')
        self.owner.tick(lambda value: None)
        manual.tick(lambda value: self.fail('manual selection must stay frozen'))
        self.assertEqual(manual.frozen, selected)
        self.assertEqual(resolve_selection(str(self.root), 'usb_cam', {'mode': 'auto'}, TARGET, ROLES)['result'], self.result)
