import unittest
from xgc_camera_calibration.pose_freshness import pose_is_fresh


class PoseFreshnessTest(unittest.TestCase):
    def test_recent_stationary_pose_is_usable(self):
        self.assertTrue(pose_is_fresh((99., 11.), 100., 12., 2.))

    def test_disconnect_expires_even_when_simulation_clock_is_paused(self):
        self.assertFalse(pose_is_fresh((97., 12.), 100., 12., 2.))

    def test_fresh_delivery_of_old_or_future_source_stamp_is_rejected(self):
        for stamp in (8., 13., 0., float('nan')):
            self.assertFalse(pose_is_fresh((100., stamp), 100., 12., 2.))

    def test_missing_receipt_and_clock_reset_are_rejected(self):
        self.assertFalse(pose_is_fresh(None, 100., 12., 2.))
        self.assertFalse(pose_is_fresh((101., 12.), 100., 12., 2.))
