#!/usr/bin/env python3

import unittest
from types import SimpleNamespace
from unittest import mock

from xgc_camera_calibration.camera_control import _wait_for_gazebo_model_presence


class CameraControlTest(unittest.TestCase):
    def test_waits_for_asynchronous_model_deletion(self):
        world = mock.Mock(
            side_effect=[
                SimpleNamespace(model_names=["intrinsic_aprilgrid"]),
                SimpleNamespace(model_names=[]),
                SimpleNamespace(model_names=[]),
                SimpleNamespace(model_names=[]),
                SimpleNamespace(model_names=[]),
            ]
        )
        _wait_for_gazebo_model_presence(
            world,
            model_name="intrinsic_aprilgrid",
            expected=False,
            timeout=0.1,
            poll_interval=0.0,
        )
        self.assertEqual(world.call_count, 5)

    def test_waits_for_asynchronous_model_insertion(self):
        world = mock.Mock(
            side_effect=[
                SimpleNamespace(model_names=[]),
                SimpleNamespace(model_names=["intrinsic_aprilgrid"]),
                SimpleNamespace(model_names=["intrinsic_aprilgrid"]),
                SimpleNamespace(model_names=["intrinsic_aprilgrid"]),
                SimpleNamespace(model_names=["intrinsic_aprilgrid"]),
            ]
        )
        _wait_for_gazebo_model_presence(
            world,
            model_name="intrinsic_aprilgrid",
            expected=True,
            timeout=0.1,
            poll_interval=0.0,
        )
        self.assertEqual(world.call_count, 5)

    def test_fails_closed_when_model_state_never_commits(self):
        world = mock.Mock(return_value=SimpleNamespace(model_names=[]))
        with self.assertRaisesRegex(RuntimeError, "did not appear"):
            _wait_for_gazebo_model_presence(
                world,
                model_name="intrinsic_aprilgrid",
                expected=True,
                timeout=0.001,
                poll_interval=0.0,
            )


if __name__ == "__main__":
    unittest.main()
