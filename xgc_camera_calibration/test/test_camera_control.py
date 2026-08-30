#!/usr/bin/env python3

import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

from xgc_camera_calibration.board_profiles import (
    A4_6X6_24MM_30PCT_KALIBR_V1,
    FIELD_6X6_88MM_30PCT,
    PROFILES,
)
from xgc_camera_calibration import camera_control


class _Pose:
    def __init__(self):
        self.position = types.SimpleNamespace(x=0.0, y=0.0, z=0.0)
        self.orientation = types.SimpleNamespace(w=0.0)


class GazeboBoardSelectionTest(unittest.TestCase):
    def test_profile_instances_replace_the_legacy_name_and_are_idempotent(self):
        with tempfile.TemporaryDirectory() as directory:
            package_root = Path(directory)
            for profile in PROFILES.values():
                model = package_root / "models" / profile.gazebo_model / "model.sdf"
                model.parent.mkdir(parents=True)
                model.write_text("<sdf version='1.6'><model name='board'/></sdf>")

            models = {"intrinsic_aprilgrid"}
            deleted = []
            spawned = []
            logs = []

            def world():
                return types.SimpleNamespace(model_names=sorted(models))

            def delete(name):
                deleted.append(name)
                models.discard(name)
                return types.SimpleNamespace(success=True, status_message="deleted")

            def spawn(name, _sdf, _namespace, pose, reference_frame):
                spawned.append((name, pose.position.x, pose.position.y, pose.position.z, reference_frame))
                models.add(name)
                return types.SimpleNamespace(success=True, status_message="spawned")

            proxies = {
                "/gazebo/delete_model": delete,
                "/gazebo/get_world_properties": world,
                "/gazebo/spawn_sdf_model": spawn,
            }
            rospy = types.ModuleType("rospy")
            rospy.wait_for_service = lambda *_args, **_kwargs: None
            rospy.ServiceProxy = lambda name, _kind: proxies[name]
            rospy.loginfo = lambda *args: logs.append(args)
            rospkg = types.ModuleType("rospkg")
            rospkg.RosPack = lambda: types.SimpleNamespace(
                get_path=lambda name: str(package_root) if name == "gazebo_sim_worlds" else ""
            )
            gazebo_msgs = types.ModuleType("gazebo_msgs")
            gazebo_srv = types.ModuleType("gazebo_msgs.srv")
            gazebo_srv.DeleteModel = object
            gazebo_srv.GetWorldProperties = object
            gazebo_srv.SpawnModel = object
            geometry_msgs = types.ModuleType("geometry_msgs")
            geometry_msg = types.ModuleType("geometry_msgs.msg")
            geometry_msg.Pose = _Pose

            modules = {
                "rospy": rospy,
                "rospkg": rospkg,
                "gazebo_msgs": gazebo_msgs,
                "gazebo_msgs.srv": gazebo_srv,
                "geometry_msgs": geometry_msgs,
                "geometry_msgs.msg": geometry_msg,
            }
            field = PROFILES[FIELD_6X6_88MM_30PCT]
            a4 = PROFILES[A4_6X6_24MM_30PCT_KALIBR_V1]
            with mock.patch.dict(sys.modules, modules), mock.patch.object(
                camera_control.time, "sleep", return_value=None
            ):
                camera_control.select_gazebo_board_profile(field, (2.0, 0.0, 2.2))
                self.assertEqual(models, {field.gazebo_instance_name})
                self.assertEqual(deleted, ["intrinsic_aprilgrid"])
                self.assertEqual(spawned[-1], (field.gazebo_instance_name, 2.0, 0.0, 2.2, "world"))

                calls = (len(deleted), len(spawned))
                camera_control.select_gazebo_board_profile(field, (2.0, 0.0, 2.2))
                self.assertEqual((len(deleted), len(spawned)), calls)
                self.assertIn("already selected", logs[-1][0])

                camera_control.select_gazebo_board_profile(a4, (2.0, 0.0, 2.2))
                self.assertEqual(models, {a4.gazebo_instance_name})
                self.assertEqual(deleted[-1], field.gazebo_instance_name)
                self.assertEqual(spawned[-1][0], a4.gazebo_instance_name)


class GazeboModelPresenceTest(unittest.TestCase):
    def test_waits_for_asynchronous_model_deletion(self):
        world = mock.Mock(
            side_effect=[
                types.SimpleNamespace(model_names=["intrinsic_aprilgrid"]),
                types.SimpleNamespace(model_names=[]),
                types.SimpleNamespace(model_names=[]),
                types.SimpleNamespace(model_names=[]),
                types.SimpleNamespace(model_names=[]),
            ]
        )
        camera_control._wait_for_gazebo_model_presence(
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
                types.SimpleNamespace(model_names=[]),
                types.SimpleNamespace(model_names=["intrinsic_aprilgrid"]),
                types.SimpleNamespace(model_names=["intrinsic_aprilgrid"]),
                types.SimpleNamespace(model_names=["intrinsic_aprilgrid"]),
                types.SimpleNamespace(model_names=["intrinsic_aprilgrid"]),
            ]
        )
        camera_control._wait_for_gazebo_model_presence(
            world,
            model_name="intrinsic_aprilgrid",
            expected=True,
            timeout=0.1,
            poll_interval=0.0,
        )
        self.assertEqual(world.call_count, 5)

    def test_fails_closed_when_model_state_never_commits(self):
        world = mock.Mock(return_value=types.SimpleNamespace(model_names=[]))
        with self.assertRaisesRegex(RuntimeError, "did not appear"):
            camera_control._wait_for_gazebo_model_presence(
                world,
                model_name="intrinsic_aprilgrid",
                expected=True,
                timeout=0.001,
                poll_interval=0.0,
            )


if __name__ == "__main__":
    unittest.main()
