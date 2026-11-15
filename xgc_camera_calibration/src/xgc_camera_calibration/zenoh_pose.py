"""Raw VRPN poses from one camera-calibration source and sensor."""
import json
import math


def pose_key(source, tracker, sensor):
    return "xgc2/localization/vrpn/v1/{}/{}/{}/pose".format(
        source.encode("utf-8").hex(), tracker.encode("utf-8").hex(), sensor)


def tracker_from_pose_key(key, source, sensor):
    """Recover the actual tracker identity inside the selected source only."""
    prefix = "xgc2/localization/vrpn/v1/{}/".format(source.encode("utf-8").hex())
    suffix = "/{}/pose".format(sensor)
    if not key.startswith(prefix) or not key.endswith(suffix):
        return None
    encoded = key[len(prefix):-len(suffix)]
    try:
        name = bytes.fromhex(encoded).decode("utf-8")
    except (ValueError, UnicodeDecodeError):
        return None
    return name if name and name.encode("utf-8").hex() == encoded else None


class ZenohPoses:
    def __init__(self, endpoint, source, trackers, sensor, callback):
        import zenoh
        from xgc2.vrpn_zenoh.v1.tracker_pb2 import Pose
        from google.protobuf.message import DecodeError
        self._session = None
        self._subscribers = []
        config = zenoh.Config()
        config.insert_json5("mode", '"client"')
        config.insert_json5("connect/endpoints", json.dumps([endpoint]))
        config.insert_json5("scouting/multicast/enabled", "false")
        config.insert_json5("scouting/gossip/enabled", "false")
        try:
            self._session = zenoh.open(config)
            for tracker in sorted(trackers) if trackers else [None]:
                def receive(sample, name=tracker):
                    if name is None:
                        name = tracker_from_pose_key(str(sample.key_expr), source, sensor)
                        if name is None:
                            return
                    message = Pose()
                    try:
                        message.ParseFromString(sample.payload.to_bytes())
                        stamp = message.report_time.nanoseconds / 1.0e9
                        position = (message.position_m.x, message.position_m.y, message.position_m.z)
                        if stamp <= 0 or not all(math.isfinite(v) for v in position):
                            return
                        callback(name, stamp, position)
                    except (ValueError, TypeError, DecodeError):
                        return
                self._subscribers.append(self._session.declare_subscriber(
                    pose_key(source, tracker, sensor) if tracker is not None else
                    "xgc2/localization/vrpn/v1/{}/*/{}/pose".format(source.encode("utf-8").hex(), sensor),
                    receive))
        except BaseException:
            self.close()
            raise

    def close(self):
        for subscriber in self._subscribers:
            subscriber.undeclare()
        self._subscribers.clear()
        if self._session is not None:
            self._session.close()
            self._session = None
