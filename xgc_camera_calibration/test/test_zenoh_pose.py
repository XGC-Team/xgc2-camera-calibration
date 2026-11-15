"""Real loopback Zenoh transport using the generated public protobuf schema."""
import socket
import threading
import time
import unittest

from xgc2.vrpn_zenoh.v1.tracker_pb2 import Pose
from xgc_camera_calibration.zenoh_pose import ZenohPoses, pose_key, tracker_from_pose_key


class ZenohPoseTest(unittest.TestCase):
    def test_discovery_decodes_only_the_selected_source_and_sensor(self):
        self.assertEqual(tracker_from_pose_key(pose_key("calibration", "标定杆 1", 0), "calibration", 0), "标定杆 1")
        for key in (pose_key("other", "wand", 0), pose_key("calibration", "wand", 1),
                    pose_key("calibration", "wand", 0).replace("/pose", "/velocity"),
                    "xgc2/localization/vrpn/v1/63616c6962726174696f6e/zz/0/pose",
                    "xgc2/localization/vrpn/v1/63616c6962726174696f6e/7761/6e64/0/pose"):
            with self.subTest(key=key):
                self.assertIsNone(tracker_from_pose_key(key, "calibration", 0))

    def test_discovery_accepts_late_bodies_without_crossing_source_or_sensor(self):
        import zenoh
        with socket.socket() as temporary:
            temporary.bind(("127.0.0.1", 0))
            endpoint = "tcp/127.0.0.1:{}".format(temporary.getsockname()[1])
        config = zenoh.Config()
        config.insert_json5("mode", '"peer"')
        config.insert_json5("listen/endpoints", '["' + endpoint + '"]')
        config.insert_json5("scouting/multicast/enabled", "false")
        config.insert_json5("scouting/gossip/enabled", "false")
        publisher = zenoh.open(config)
        received, ready = [], threading.Event()
        receiver = None
        try:
            receiver = ZenohPoses(endpoint, "calibration", [], 0,
                lambda *value: (received.append(value), ready.set()))
            message = Pose()
            message.report_time.nanoseconds = 1234567890123
            message.position_m.x, message.position_m.y, message.position_m.z = 1.25, -2, 3
            for source, name, sensor in (("other", "wand", 0), ("calibration", "wrong sensor", 1)):
                publisher.put(pose_key(source, name, sensor), message.SerializeToString())
            self.assertFalse(ready.wait(.15))
            for name in ("wand", "标定杆 1"):
                deadline = time.monotonic() + 3
                while not any(item[0] == name for item in received) and time.monotonic() < deadline:
                    publisher.put(pose_key("calibration", name, 0), message.SerializeToString())
                    time.sleep(.02)
                self.assertTrue(any(item == (name, 1234.567890123, (1.25, -2., 3.)) for item in received))
            self.assertEqual({item[0] for item in received}, {"wand", "标定杆 1"})
            receiver.close()
            receiver = None
            count = len(received)
            publisher.put(pose_key("calibration", "late", 0), message.SerializeToString())
            time.sleep(.05)
            self.assertEqual(len(received), count)
        finally:
            if receiver is not None:
                receiver.close()
            publisher.close()

    def test_exact_trackers_preserve_raw_position_and_source_time(self):
        import zenoh
        with socket.socket() as temporary:
            temporary.bind(("127.0.0.1", 0))
            endpoint = "tcp/127.0.0.1:{}".format(temporary.getsockname()[1])
        config = zenoh.Config()
        config.insert_json5("mode", '"peer"')
        config.insert_json5("listen/endpoints", '["' + endpoint + '"]')
        config.insert_json5("scouting/multicast/enabled", "false")
        config.insert_json5("scouting/gossip/enabled", "false")
        publisher = zenoh.open(config)
        received, ready = [], threading.Event()
        receiver = None
        try:
            receiver = ZenohPoses(endpoint, "physical", ["wand", "second"], 0,
                lambda *value: (received.append(value), ready.set()))
            message = Pose()
            message.report_time.nanoseconds = 1234567890123
            message.position_m.x, message.position_m.y, message.position_m.z = 1.25, -2, 3
            publisher.put(pose_key("other", "wand", 0), message.SerializeToString())
            self.assertFalse(ready.wait(.15))
            deadline = time.monotonic() + 3
            while not ready.is_set() and time.monotonic() < deadline:
                publisher.put(pose_key("physical", "wand", 0), message.SerializeToString())
                ready.wait(.02)
            self.assertTrue(ready.is_set())
            self.assertEqual(received[0], ("wand", 1234.567890123, (1.25, -2., 3.)))
            receiver.close()
            receiver = None
            count = len(received)
            publisher.put(pose_key("physical", "second", 0), message.SerializeToString())
            time.sleep(.05)
            self.assertEqual(len(received), count)
        finally:
            if receiver is not None:
                receiver.close()
            publisher.close()


if __name__ == "__main__":
    unittest.main()
