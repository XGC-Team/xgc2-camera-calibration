#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PREFIX="/opt/ros/${ROS_DISTRO:-noetic}"
source "${PREFIX}/setup.bash"
python3 "${SCRIPT_DIR}/check_python_xrpc.py"
dpkg -s ros-noetic-xgc2-camera-calibration >/dev/null
for definition in xgc2-camera-extrinsic-calibrator-ros1 xgc2-camera-extrinsic-tf-ros1 xgc2-camera-intrinsic-calibrator-ros1; do
  test -s "/usr/share/xgc2/process-definitions/${definition}.json"
done
test "$(rospack find xgc_camera_calibration)" = "${PREFIX}/share/xgc_camera_calibration"
for script in extrinsic_calibrator_web intrinsic_calibrator_web extrinsic_tf_publisher gazebo_camera_from_extrinsic resolve_extrinsic; do
  test -x "${PREFIX}/lib/xgc_camera_calibration/${script}.py"
done
for page in extrinsic intrinsic; do
  for asset in index.html app.js styles.css; do
    test -f "${PREFIX}/share/xgc_camera_calibration/web/${page}/${asset}"
  done
done
python3 -c 'from xgc_camera_calibration.extrinsic_resolver import decode_frozen; from xgc_camera_calibration.intrinsic_solver import calibrate_intrinsic; from xgc_camera_calibration.media_snapshot import MediaSnapshotClient; from xgc_camera_calibration.solver import solve_extrinsic; from xgc_camera_calibration.web_service import CalibrationHttpServer'

RUNTIME="$(mktemp -d)"
ROSCORE_PID=""
EXTRINSIC_PID=""
TF_PID=""
INTRINSIC_PID=""
MEDIA_EDGE_PID=""
cleanup() {
  for task_pid in "${INTRINSIC_PID}" "${EXTRINSIC_PID}" "${TF_PID}" "${MEDIA_EDGE_PID}" "${ROSCORE_PID}"; do
    if [[ -n "${task_pid}" ]]; then kill "${task_pid}" 2>/dev/null || true; fi
  done
  for task_pid in "${INTRINSIC_PID}" "${EXTRINSIC_PID}" "${TF_PID}" "${MEDIA_EDGE_PID}" "${ROSCORE_PID}"; do
    if [[ -n "${task_pid}" ]]; then wait "${task_pid}" 2>/dev/null || true; fi
  done
  rm -rf "${RUNTIME}"
}
trap cleanup EXIT
read -r MASTER_PORT EXTRINSIC_PORT INTRINSIC_PORT < <(python3 - <<'PY'
import random, socket
sockets = []
try:
    while len(sockets) < 3:
        peer = socket.socket()
        try:
            peer.bind(('127.0.0.1', random.randint(20000, 40000)))
        except OSError:
            peer.close()
            continue
        sockets.append(peer)
    print(*(peer.getsockname()[1] for peer in sockets))
finally:
    for peer in sockets:
        peer.close()
PY
)
export ROS_MASTER_URI="http://127.0.0.1:${MASTER_PORT}"
export ROS_HOME="${RUNTIME}/ros-home"
export ROS_LOG_DIR="${RUNTIME}/ros-log"
export ROS_IP=127.0.0.1
unset ROS_HOSTNAME
mkdir -p "${ROS_HOME}" "${ROS_LOG_DIR}" "${RUNTIME}/input"
FRAME_ROLES_JSON='{"parentFrame":"world","opticalFrames":{"sim":"camera_optical","phy":"camera_optical"}}'
RESOLVED_EXTRINSIC_JSON="$(env -u PYTHONPATH "${PREFIX}/lib/xgc_camera_calibration/resolve_extrinsic.py" \
  --root "${RUNTIME}/input" --camera package_smoke --selection-json '{"mode":"auto"}' \
  --target-offset-json '{"x":0,"y":0,"z":0}' --frame-roles-json "${FRAME_ROLES_JSON}")"
roslaunch --files xgc_camera_calibration intrinsic_calibrator.launch \
  socket_path:="${RUNTIME}/intrinsic.sock" target_id:=package-smoke calibration_mode:=sim camera_name:=package_smoke \
  media_edge_rpc_socket:="${RUNTIME}/media-edge.sock" >/dev/null
roslaunch --files xgc_camera_calibration extrinsic_calibrator.launch \
  socket_path:="${RUNTIME}/extrinsic.sock" target_id:=package-smoke \
  resolved_extrinsic_json:="${RESOLVED_EXTRINSIC_JSON}" frame_roles_json:="${FRAME_ROLES_JSON}" \
  pose_coordinate_source:=experiment-world parent_frame:=world child_frame:=camera_optical \
  calibration_mode:=sim camera_name:=package_smoke >/dev/null
roslaunch --files xgc_camera_calibration extrinsic_tf.launch \
  resolved_extrinsic_json:="${RESOLVED_EXTRINSIC_JSON}" frame_roles_json:="${FRAME_ROLES_JSON}" \
  calibration_mode:=sim camera_name:=package_smoke parent_frame:=world camera_link_frame:=camera_link optical_frame:=camera_optical >/dev/null
roscore -p "${MASTER_PORT}" >"${RUNTIME}/roscore.log" 2>&1 &
ROSCORE_PID="$!"
for _ in $(seq 1 100); do
  if rosparam list >/dev/null 2>&1; then break; fi
  if ! kill -0 "${ROSCORE_PID}" 2>/dev/null; then exit 1; fi
  sleep 0.1
done
rosparam list >/dev/null

# Discovery and health fixture only; no camera frame or calibration is fabricated.
MEDIA_EDGE_RPC_SOCKET="${RUNTIME}/media-edge.sock"
python3 - "${MEDIA_EDGE_RPC_SOCKET}" <<'PYMEDIA' >"${RUNTIME}/media-edge.log" 2>&1 &
import signal, sys, threading
from xgc2_xrpc import Host, Runtime
path = sys.argv[1]
ref = {"target_id": "package-smoke", "service": "media-edge", "api_version": "v1",
       "instance_id": "package-smoke", "profile": "http.v1", "endpoint": {"kind": "unix", "address": path}}
runtime = Runtime()
host = Host(path, {("GET", "/v1/describe"): lambda c, r: {"service_ref": ref},
    ("GET", "/v1/health"): lambda c, r: {"sources": [{"id": "package_smoke"}]}},
    runtime=runtime, instance_id="package-smoke", discovery_routes=("/v1/describe",)).start()
stop = threading.Event()
for signum in (signal.SIGTERM, signal.SIGINT):
    signal.signal(signum, lambda *_: stop.set())
try:
    stop.wait()
finally:
    host.close()
    runtime.close()
PYMEDIA
MEDIA_EDGE_PID="$!"
for _ in $(seq 1 100); do
  if [[ -S "${MEDIA_EDGE_RPC_SOCKET}" ]]; then break; fi
  if ! kill -0 "${MEDIA_EDGE_PID}" 2>/dev/null; then exit 1; fi
  sleep 0.1
done
[[ -S "${MEDIA_EDGE_RPC_SOCKET}" ]]
"${PREFIX}/lib/xgc_camera_calibration/extrinsic_tf_publisher.py" \
  --resolved-extrinsic-json "${RESOLVED_EXTRINSIC_JSON}" --frame-roles-json "${FRAME_ROLES_JSON}" \
  __name:=xgc_camera_extrinsic_tf _calibration_mode:=sim _camera_name:=package_smoke \
  _parent_frame:=world _camera_link_frame:=camera_link _optical_frame:=camera_optical >"${RUNTIME}/extrinsic-tf.log" 2>&1 &
TF_PID="$!"
"${PREFIX}/lib/xgc_camera_calibration/extrinsic_calibrator_web.py" \
  --resolved-extrinsic-json "${RESOLVED_EXTRINSIC_JSON}" --frame-roles-json "${FRAME_ROLES_JSON}" \
  __name:=xgc_camera_extrinsic_calibrator_web _target_id:=package-smoke _socket_path:="${RUNTIME}/extrinsic.sock" \
  _image_topic:=/package_smoke/image_raw/compressed _camera_info_topic:=/package_smoke/camera_info \
  _calibration_mode:=sim _camera_name:=package_smoke _http_port:="${EXTRINSIC_PORT}" \
  _parent_frame:=world _child_frame:=camera_optical _pose_coordinate_source:=experiment-world >"${RUNTIME}/extrinsic.log" 2>&1 &
EXTRINSIC_PID="$!"
"${PREFIX}/lib/xgc_camera_calibration/intrinsic_calibrator_web.py" \
  __name:=xgc_camera_intrinsic_calibrator_web _target_id:=package-smoke _socket_path:="${RUNTIME}/intrinsic.sock" \
  _media_edge_rpc_socket:="${MEDIA_EDGE_RPC_SOCKET}" _media_source_id:=package_smoke _snapshot_timeout:=1 _auto_capture:=false \
  _http_port:="${INTRINSIC_PORT}" _calibration_mode:=sim _camera_name:=package_smoke >"${RUNTIME}/intrinsic.log" 2>&1 &
INTRINSIC_PID="$!"
python3 - "${RUNTIME}" "${EXTRINSIC_PORT}" "${INTRINSIC_PORT}" <<'PYCHECK'
import json, sys, time, urllib.request
from pathlib import Path
from xgc2_xrpc import Client, Endpoint, ServiceRef
root = Path(sys.argv[1])
for kind, port in zip(("extrinsic", "intrinsic"), sys.argv[2:]):
    for _ in range(100):
        try:
            with urllib.request.urlopen('http://127.0.0.1:' + port + '/healthz', timeout=.5) as reply:
                assert json.load(reply)['status'] == 'ok'
            break
        except OSError:
            time.sleep(.1)
    else:
        raise RuntimeError(kind + ' did not start')
    ref = ServiceRef('package-smoke', 'xgc2.calibration.v1.Calibration', '1', '', 'http.v1',
                     Endpoint('unix', str(root / (kind + '.sock'))))
    with Client.from_service(ref, local_target='package-smoke', discovery=True) as client:
        started = time.monotonic()
        description = client.json('/v1/describe?wait_ready_ms=100', method='GET', timeout=1)
        assert time.monotonic() - started >= .09
        assert not description['ready']
        assert description['facts']['result_path'] == '/v1/result'
    with Client.from_service(ServiceRef.from_dict(description['service_ref']), local_target='package-smoke') as client:
        result = client.json('/v1/result', method='GET')
        assert result['kind'] == kind and result['camera_id'] == 'package_smoke' and result['result'] is None
        assert client.json('/v1/result', method='GET')['revision'] == result['revision']
import rospy
from tf2_msgs.msg import TFMessage
rospy.init_node('camera_package_tf_check', anonymous=True, disable_signals=True)
message = rospy.wait_for_message('/tf_static', TFMessage, timeout=5)
assert {'camera_link', 'camera_optical'} <= {item.child_frame_id for item in message.transforms}
assert rospy.get_param('/xgc_camera_extrinsic_tf/extrinsic_ready') is True
PYCHECK

echo "Installed standalone ROS1 camera calibration package passed"
