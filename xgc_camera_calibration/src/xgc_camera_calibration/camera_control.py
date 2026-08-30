"""Optional Gazebo camera-control adapter for the intrinsic sample guide.

The intrinsic calibrator is camera-agnostic: it works against any image source
with no camera control at all.  When it happens to run against a Gazebo
simulation, this adapter lights up the guide's interactive parts -- fly the
camera to a sample pose (``goto``), send it home (``reset``), and read its live
pose -- by talking to Gazebo's own standard topics (``/gazebo/set_model_state``
and ``/gazebo/model_states``).  It depends only on ``gazebo_msgs`` (a standard
message package), never on the simulator product packages, and everything ROS
is imported lazily so non-simulation builds never load it.
"""

from __future__ import annotations

import copy
import math
import threading
import time
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

from xgc_camera_calibration.board_profiles import AprilGridProfile


def select_gazebo_board_profile(
    profile: AprilGridProfile,
    board_center: Sequence[float],
    connection_timeout: float = 10.0,
) -> None:
    """Replace the calibration target with the selected exact-size model."""
    import rospkg
    import rospy
    from gazebo_msgs.srv import DeleteModel, SpawnModel
    from geometry_msgs.msg import Pose

    package_root = rospkg.RosPack().get_path("gazebo_sim_worlds")
    model_path = (
        Path(package_root) / "models" / profile.gazebo_model / "model.sdf"
    )
    if not model_path.is_file():
        raise RuntimeError("Gazebo calibration board model is unavailable: {}".format(model_path))
    delete_name = "/gazebo/delete_model"
    spawn_name = "/gazebo/spawn_sdf_model"
    rospy.wait_for_service(delete_name, timeout=connection_timeout)
    rospy.wait_for_service(spawn_name, timeout=connection_timeout)
    delete = rospy.ServiceProxy(delete_name, DeleteModel)
    spawn = rospy.ServiceProxy(spawn_name, SpawnModel)
    try:
        delete("intrinsic_aprilgrid")
    except Exception:
        pass
    pose = Pose()
    pose.position.x, pose.position.y, pose.position.z = (
        float(board_center[0]), float(board_center[1]), float(board_center[2])
    )
    pose.orientation.w = 1.0
    response = spawn(
        "intrinsic_aprilgrid", model_path.read_text(encoding="utf-8"), "", pose, "world"
    )
    if not response.success:
        raise RuntimeError("Could not spawn Gazebo calibration board: {}".format(response.status_message))
    rospy.loginfo(
        "Gazebo calibration board selected: profile=%s model=%s",
        profile.profile_id,
        profile.gazebo_model,
    )


def look_at_orientation(position, target, yaw_offset=0.0, pitch_offset=0.0, roll=0.0):
    """Quaternion that aims the camera's +x axis from ``position`` at ``target``.

    Mirrors the simulator's own ``look_at_orientation`` so a pose sent here lands
    the board where the sample-guide expects it; the yaw/pitch offsets push the
    board off-centre to fill the X/Y coverage that a centred view never moves.
    """
    from tf.transformations import quaternion_from_euler

    delta_x = target[0] - position[0]
    delta_y = target[1] - position[1]
    delta_z = target[2] - position[2]
    horizontal = math.hypot(delta_x, delta_y)
    yaw = math.atan2(delta_y, delta_x) + yaw_offset
    pitch = -math.atan2(delta_z, horizontal) + pitch_offset
    return quaternion_from_euler(roll, pitch, yaw)


class GazeboCameraControl:
    """Move and read one Gazebo model over its standard fire-and-forget topics.

    ``/gazebo/set_model_state`` only takes effect on a non-static model, so the
    simulated camera must be spawned with ``static:=false``.  Poses are read from
    the latest ``/gazebo/model_states`` sample, so ``current()`` is lock-free for
    callers and never blocks the calibration feeder thread.
    """

    def __init__(self, model_name, board_center, reference_frame="world",
                 connection_timeout=10.0):
        import rospy
        from gazebo_msgs.msg import ModelState, ModelStates

        self._rospy = rospy
        self._model_state_cls = ModelState
        self.model_name = str(model_name)
        self.board_center = tuple(float(value) for value in board_center)
        self.reference_frame = str(reference_frame)
        self._lock = threading.Lock()
        self._latest_pose = None
        self._initial_pose = None
        self._publisher = rospy.Publisher(
            "/gazebo/set_model_state", ModelState, queue_size=1
        )
        self._subscriber = rospy.Subscriber(
            "/gazebo/model_states", ModelStates, self._on_states, queue_size=1
        )
        self._wait_for_subscriber(connection_timeout)

    def _on_states(self, message):
        try:
            index = message.name.index(self.model_name)
        except ValueError:
            return
        pose = message.pose[index]
        with self._lock:
            self._latest_pose = pose
            if self._initial_pose is None:
                self._initial_pose = copy.deepcopy(pose)

    def _wait_for_subscriber(self, timeout):
        deadline = time.monotonic() + timeout
        while not self._rospy.is_shutdown() and self._publisher.get_num_connections() == 0:
            if time.monotonic() >= deadline:
                return
            time.sleep(0.05)
        # get_num_connections() flips to 1 before the TCP handshake is fully
        # ready to carry a message, so let it settle -- otherwise the very first
        # teleport (a lone click) can be dropped on a fire-and-forget topic.
        if self._publisher.get_num_connections() > 0:
            time.sleep(0.3)

    def available(self) -> bool:
        """True once a pose for the model has been seen on /gazebo/model_states."""
        with self._lock:
            return self._latest_pose is not None

    def current(self) -> Optional[Dict[str, float]]:
        with self._lock:
            pose = self._latest_pose
        if pose is None:
            return None
        return {
            "x": float(pose.position.x),
            "y": float(pose.position.y),
            "z": float(pose.position.z),
            "qx": float(pose.orientation.x),
            "qy": float(pose.orientation.y),
            "qz": float(pose.orientation.z),
            "qw": float(pose.orientation.w),
        }

    def current_position(self) -> Optional[Tuple[float, float, float]]:
        with self._lock:
            pose = self._latest_pose
        if pose is None:
            return None
        return (float(pose.position.x), float(pose.position.y), float(pose.position.z))

    def current_optical_pose(self):
        from tf.transformations import quaternion_from_euler, quaternion_multiply

        with self._lock:
            pose = copy.deepcopy(self._latest_pose)
        if pose is None:
            return None
        link_orientation = (
            float(pose.orientation.x), float(pose.orientation.y),
            float(pose.orientation.z), float(pose.orientation.w),
        )
        optical_orientation = quaternion_multiply(
            link_orientation,
            quaternion_from_euler(-math.pi / 2.0, 0.0, -math.pi / 2.0),
        )
        return {
            "position": (
                float(pose.position.x), float(pose.position.y), float(pose.position.z)
            ),
            "orientation": tuple(float(value) for value in optical_orientation),
        }

    def _publish(self, pose):
        state = self._model_state_cls()
        state.model_name = self.model_name
        state.reference_frame = self.reference_frame
        state.pose = pose
        self._publisher.publish(state)

    def goto(self, position: Sequence[float], yaw_offset=0.0, pitch_offset=0.0, roll=0.0):
        from geometry_msgs.msg import Pose

        quaternion = look_at_orientation(
            position, self.board_center, yaw_offset=yaw_offset,
            pitch_offset=pitch_offset, roll=roll,
        )
        pose = Pose()
        pose.position.x, pose.position.y, pose.position.z = (
            float(position[0]), float(position[1]), float(position[2])
        )
        (
            pose.orientation.x,
            pose.orientation.y,
            pose.orientation.z,
            pose.orientation.w,
        ) = quaternion
        self._publish(pose)

    def reset(self):
        with self._lock:
            initial = copy.deepcopy(self._initial_pose)
        if initial is not None:
            self._publish(initial)
