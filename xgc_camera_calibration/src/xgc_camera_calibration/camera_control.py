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
from xgc2_xrpc import Endpoint, Runtime, ServiceRef, TransportError
from xgc_camera_calibration.peer_connection import UnixPeer

from xgc_camera_calibration.board_profiles import AprilGridProfile, PROFILES


_LEGACY_GAZEBO_BOARD_INSTANCE_NAME = "intrinsic_aprilgrid"


class _SimulationWorld:
    """Domain facade for an explicitly granted simulation-v1 world."""

    def __init__(self, endpoint, *, runtime=None, timeout=10.0, local_target=None):
        if not isinstance(local_target, str) or not local_target:
            raise ValueError("an explicit local target binding is required")
        ServiceRef(local_target, "xgc2.simulation", "v1", "", "http.v1",
                   Endpoint("unix", str(endpoint))).validate(discovery=True)
        self.timeout = float(timeout)
        if not math.isfinite(self.timeout) or self.timeout <= 0:
            raise ValueError("simulation timeout must be finite and positive")
        self.runtime = runtime or Runtime(blocking_workers=4)
        self._owns_runtime = runtime is None
        self._closed = False
        self.peer = UnixPeer(endpoint, "xgc2.simulation", "v1", runtime=self.runtime,
                             local_target=local_target)

    @property
    def client(self):
        return self.peer.require_client()

    def close(self):
        if self._closed:
            return
        self.peer.close()
        if self._owns_runtime:
            self.runtime.close()
        self._closed = True

    def entities(self, identity=None, *, client=None):
        path = "/v1/entities" if identity is None else "/v1/entities/" + quote(identity, safe="")
        try:
            value = (client or self.client).json(path, method="GET", timeout=self.timeout)
        except (TransportError, OSError, TimeoutError) as error:
            self.peer.invalidate(error)
            raise
        entities = value.get("entities") if isinstance(value, dict) else None
        if not isinstance(entities, list):
            raise RuntimeError("simulation entity response is invalid")
        return entities

    def mutate(self, path, value, method="POST", *, client=None):
        bound = client or self.client
        try:
            operation = bound.json(path, {**value, "operation_timeout_ms": int(self.timeout * 1000)},
                                   method=method, timeout=self.timeout)
            if operation.get("state") in ("accepted", "running"):
                operation = bound.json("/v1/operations/" + quote(operation["id"], safe="") + "/wait",
                                       {}, timeout=self.timeout)
        except (TransportError, OSError, TimeoutError) as error:
            self.peer.invalidate(error)
            raise
        if operation.get("state") != "succeeded":
            raise RuntimeError("simulation operation did not complete: {}".format(operation.get("error")))
        return operation



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
                 *, endpoint, runtime=None, local_target=None, board_profile=None):
        if reference_frame != "world":
            raise ValueError("simulation-v1 camera uses the declared world frame")
        self.model_name = str(model_name)
        self.board_center = tuple(float(x) for x in board_center)
        self._lock = threading.RLock()
        self.world = _SimulationWorld(endpoint, runtime=runtime, timeout=connection_timeout,
                                      local_target=local_target)
        self._reference = None
        self._latest_pose = None
        self._initial_pose = None
        self._desired_pose = None
        self._desired_revision = 0
        self._applied_revision = 0
        self._applied_instance = None
        self._board_profile = board_profile
        self.world.peer.connected = self._connected

    def _select_board(self, client):
        profile = self._board_profile
        if profile is None:
            return
        import rospkg
        root = Path(rospkg.RosPack().get_path("gazebo_sim_worlds"))
        model_path = root / "models" / profile.gazebo_model / "model.sdf"
        if not model_path.is_file() or not profile.gazebo_instance_name:
            raise RuntimeError("Calibration board realization is unavailable")
        present = {item["ref"]["id"]: item["ref"] for item in self.world.entities(client=client)}
        known = {_LEGACY_GAZEBO_BOARD_INSTANCE_NAME}
        known.update(item.gazebo_instance_name for item in PROFILES.values() if item.gazebo_instance_name)
        for identity in sorted((set(present) & known) - {profile.gazebo_instance_name}):
            self.world.mutate("/v1/entities/" + quote(identity, safe=""),
                {"generation": present[identity]["generation"]}, method="DELETE", client=client)
        if profile.gazebo_instance_name not in present:
            self.world.mutate("/v1/entities", {"entity": {"id": profile.gazebo_instance_name,
                "role": "object", "asset": {"id": profile.gazebo_model,
                    "realization": {"media_type": "application/sdf+xml",
                                    "content": model_path.read_text(encoding="utf-8")}},
                "pose": {"position": list(self.board_center), "orientation": [0., 0., 0., 1.]}}}, client=client)

    def _connected(self, client, reference):
        self._select_board(client)
        entity = self._entity(client)
        with self._lock:
            self._reference = dict(entity["ref"])
            self._latest_pose = copy.deepcopy(entity["state"]["pose"])
            if self._initial_pose is None:
                self._initial_pose = copy.deepcopy(self._latest_pose)
            desired, revision = copy.deepcopy(self._desired_pose), self._desired_revision
        # Reconnect work never holds the lock used by status/capture reads.
        if desired is not None and entity["state"]["pose"] != desired:
            self._apply_confirmed(desired, client)
        with self._lock:
            self._applied_revision, self._applied_instance = revision, reference.instance_id

    def status(self):
        value = self.world.peer.status()
        with self._lock:
            return {**value, "pose_available": bool(value["ready"] and self._latest_pose is not None),
                    "desired_revision": self._desired_revision,
                    "applied_revision": self._applied_revision, "applied_instance": self._applied_instance,
                    "board_revision": 1 if self._board_profile is not None else 0}

    def _entity(self, client=None):
        items = self.world.entities(self.model_name, client=client)
        if len(items) != 1 or items[0]["ref"]["id"] != self.model_name:
            raise RuntimeError("simulation provider did not return the camera entity")
        return items[0]

    def close(self):
        self.world.close()

    def available(self):
        with self._lock:
            return self.world.peer.status()["ready"] and self._latest_pose is not None

    def current(self):
        with self._lock:
            pose = self._latest_pose
            if pose is None or not self.world.peer.status()["ready"]:
                return None
            position, orientation = pose["position"], pose["orientation"]
            return dict(zip(("x", "y", "z", "qx", "qy", "qz", "qw"),
                            [float(x) for x in position + orientation]))

    def current_position(self):
        with self._lock:
            return tuple(float(x) for x in self._latest_pose["position"]) if self.available() else None

    def current_optical_pose(self):
        from tf.transformations import quaternion_from_euler, quaternion_multiply
        with self._lock:
            pose = copy.deepcopy(self._latest_pose) if self.available() else None
        if pose is None:
            return None
        orientation = quaternion_multiply(pose["orientation"],
            quaternion_from_euler(-math.pi / 2., 0., -math.pi / 2.))
        return {"position": tuple(float(x) for x in pose["position"]),
                "orientation": tuple(float(x) for x in orientation)}

    def _apply(self, pose):
        with self._lock:
            self._desired_pose = copy.deepcopy(pose)
            self._desired_revision += 1
            revision = self._desired_revision
        self._apply_confirmed(pose, self.world.client)
        with self._lock:
            self._applied_revision = revision
            self._applied_instance = self.world.peer.status()["instance_id"]

    def _apply_confirmed(self, pose, client):
        self.world.mutate("/v1/entities/" + quote(self.model_name, safe="") + "/state",
                          {"generation": self._reference["generation"], "state": {"pose": pose}}, client=client)
        # A single postcondition snapshot, never a completion polling loop.
        entity = self._entity(client)
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
        self.world.client
        self._apply(copy.deepcopy(self._initial_pose))
