"""Optional Gazebo camera-control adapter for the intrinsic sample guide.

The intrinsic calibrator is camera-agnostic: it works against any image source
with no camera control at all.  When it happens to run against a Gazebo
simulation, this adapter lights up the guide's interactive parts -- fly the
camera to a sample pose (``goto``), send it home (``reset``), and read its live
pose through the world-owned simulation-v1 XRPC service. ROS remains the
image/pose observation data edge. Native operation completion, instance and
entity-generation fencing belong to the simulation provider.
"""

from __future__ import annotations

import copy
import math
import threading
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple
from urllib.parse import quote
from xgc2_xrpc import Client, Endpoint, Runtime, ServiceRef

from xgc_camera_calibration.board_profiles import AprilGridProfile, PROFILES


_LEGACY_GAZEBO_BOARD_INSTANCE_NAME = "intrinsic_aprilgrid"


class _SimulationWorld:
    """Domain facade for an explicitly granted simulation-v1 world."""

    def __init__(self, endpoint, *, runtime=None, timeout=10.0, local_target=None):
        if not isinstance(local_target, str) or not local_target:
            raise ValueError("an explicit local target binding is required")
        self.runtime = runtime or Runtime(blocking_workers=4)
        self._owns_runtime = runtime is None
        self.timeout = float(timeout)
        if not math.isfinite(self.timeout) or self.timeout <= 0:
            raise ValueError("simulation timeout must be finite and positive")
        self._closed = False
        target = local_target
        initial = ServiceRef(target, "xgc2.simulation", "v1", "", "http.v1", Endpoint("unix", str(endpoint)))
        initial.validate(discovery=True)
        try:
            with Client.from_service(initial, runtime=self.runtime, local_target=target, discovery=True) as discovery:
                description = discovery.json("/v1/describe", method="GET", timeout=self.timeout)
            reference = ServiceRef.from_dict(description["service_ref"])
            reference.validate()
            if (reference.service != initial.service or reference.api_version != initial.api_version
                    or reference.target_id != target or reference.endpoint != initial.endpoint):
                raise RuntimeError("simulation ServiceRef does not match the granted world")
            self.client = Client.from_service(reference, runtime=self.runtime, local_target=target)
        except BaseException:
            if self._owns_runtime:
                self.runtime.close()
            raise

    def close(self):
        if self._closed:
            return
        self.client.close()
        if self._owns_runtime:
            self.runtime.close()
        self._closed = True

    def entities(self, identity=None):
        path = "/v1/entities" if identity is None else "/v1/entities/" + quote(identity, safe="")
        value = self.client.json(path, method="GET", timeout=self.timeout)
        entities = value.get("entities") if isinstance(value, dict) else None
        if not isinstance(entities, list):
            raise RuntimeError("simulation entity response is invalid")
        return entities

    def mutate(self, path, value, method="POST"):
        operation = self.client.json(path, {**value, "operation_timeout_ms": int(self.timeout * 1000)},
                                     method=method, timeout=self.timeout)
        if operation.get("state") in ("accepted", "running"):
            operation = self.client.json("/v1/operations/" + quote(operation["id"], safe="") + "/wait",
                                         {}, timeout=self.timeout)
        if operation.get("state") != "succeeded":
            raise RuntimeError("simulation operation did not complete: {}".format(operation.get("error")))
        return operation


def select_gazebo_board_profile(profile, board_center, connection_timeout=10.0, *, endpoint,
                               runtime=None, local_target=None):
    """Replace the selected board only after native remove/create completion."""
    import rospkg
    root = Path(rospkg.RosPack().get_path("gazebo_sim_worlds"))
    model_path = root / "models" / profile.gazebo_model / "model.sdf"
    if not model_path.is_file() or not profile.gazebo_instance_name:
        raise RuntimeError("Calibration board realization is unavailable")
    world = _SimulationWorld(endpoint, runtime=runtime, timeout=connection_timeout,
                             local_target=local_target)
    try:
        present = {item["ref"]["id"]: item["ref"] for item in world.entities()}
        known = {_LEGACY_GAZEBO_BOARD_INSTANCE_NAME}
        known.update(item.gazebo_instance_name for item in PROFILES.values() if item.gazebo_instance_name)
        for identity in sorted((set(present) & known) - {profile.gazebo_instance_name}):
            world.mutate("/v1/entities/" + quote(identity, safe=""),
                         {"generation": present[identity]["generation"]}, method="DELETE")
        if profile.gazebo_instance_name in present:
            return
        world.mutate("/v1/entities", {"entity": {"id": profile.gazebo_instance_name,
            "role": "object", "asset": {"id": profile.gazebo_model,
                "realization": {"media_type": "application/sdf+xml",
                                "content": model_path.read_text(encoding="utf-8")}},
            "pose": {"position": [float(x) for x in board_center], "orientation": [0., 0., 0., 1.]}}})
    finally:
        world.close()


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
    """Generation-fenced camera control with native completion and cached pose.

    Only explicit operations query/apply world state. Status and feeder reads
    use the last native-confirmed snapshot and never issue a background probe.
    """

    def __init__(self, model_name, board_center, reference_frame="world", connection_timeout=10.0,
                 *, endpoint, runtime=None, local_target=None):
        if reference_frame != "world":
            raise ValueError("simulation-v1 camera uses the declared world frame")
        self.model_name = str(model_name)
        self.board_center = tuple(float(x) for x in board_center)
        self._lock = threading.RLock()
        self.world = _SimulationWorld(endpoint, runtime=runtime, timeout=connection_timeout,
                                      local_target=local_target)
        try:
            entity = self._entity()
            self._reference = dict(entity["ref"])
            self._latest_pose = copy.deepcopy(entity["state"]["pose"])
            self._initial_pose = copy.deepcopy(self._latest_pose)
        except BaseException:
            self.world.close()
            raise

    def _entity(self):
        items = self.world.entities(self.model_name)
        if len(items) != 1 or items[0]["ref"]["id"] != self.model_name:
            raise RuntimeError("simulation provider did not return the camera entity")
        return items[0]

    def close(self):
        self.world.close()

    def available(self):
        with self._lock:
            return self._latest_pose is not None

    def current(self):
        with self._lock:
            pose = self._latest_pose
            position, orientation = pose["position"], pose["orientation"]
            return dict(zip(("x", "y", "z", "qx", "qy", "qz", "qw"),
                            [float(x) for x in position + orientation]))

    def current_position(self):
        with self._lock:
            return tuple(float(x) for x in self._latest_pose["position"])

    def current_optical_pose(self):
        from tf.transformations import quaternion_from_euler, quaternion_multiply
        with self._lock:
            pose = copy.deepcopy(self._latest_pose)
        orientation = quaternion_multiply(pose["orientation"],
            quaternion_from_euler(-math.pi / 2., 0., -math.pi / 2.))
        return {"position": tuple(float(x) for x in pose["position"]),
                "orientation": tuple(float(x) for x in orientation)}

    def _apply(self, pose):
        self.world.mutate("/v1/entities/" + quote(self.model_name, safe="") + "/state",
                          {"generation": self._reference["generation"], "state": {"pose": pose}})
        # A single postcondition snapshot, never a completion polling loop.
        entity = self._entity()
        if entity["ref"] != self._reference:
            raise RuntimeError("camera entity was replaced during pose application")
        with self._lock:
            self._latest_pose = copy.deepcopy(entity["state"]["pose"])

    def goto(self, position, yaw_offset=0.0, pitch_offset=0.0, roll=0.0):
        quaternion = look_at_orientation(position, self.board_center,
            yaw_offset=yaw_offset, pitch_offset=pitch_offset, roll=roll)
        self._apply({"position": [float(x) for x in position],
                     "orientation": [float(x) for x in quaternion]})

    def reset(self):
        self._apply(copy.deepcopy(self._initial_pose))
