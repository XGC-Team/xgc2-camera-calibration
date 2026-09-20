#!/usr/bin/env bash
set -Eeuo pipefail

report_failure() {
  local code=$?
  echo "Installed-package check failed near line ${BASH_LINENO[0]} (exit ${code})" >&2
  if [[ -n "${RUNTIME:-}" && -d "${RUNTIME}" ]]; then
    for log in "${RUNTIME}"/*.log; do
      [[ -f "${log}" ]] || continue
      echo "Diagnostic: ${log##*/}" >&2
      tail -n 80 "${log}" >&2
    done
  fi
  return "${code}"
}
trap report_failure ERR

ROS_DISTRO="${ROS_DISTRO:-noetic}"
PREFIX="/opt/ros/${ROS_DISTRO}"

# shellcheck disable=SC1090
source "${PREFIX}/setup.bash"
dpkg -s ros-noetic-xgc2-camera-calibration >/dev/null
test "$(rospack find xgc_camera_calibration)" = "${PREFIX}/share/xgc_camera_calibration"
test -x "${PREFIX}/lib/xgc_camera_calibration/extrinsic_calibrator_web.py"
test -x "${PREFIX}/lib/xgc_camera_calibration/intrinsic_calibrator_web.py"
test -x "${PREFIX}/lib/xgc_camera_calibration/extrinsic_tf_publisher.py"
test -x "${PREFIX}/lib/xgc_camera_calibration/gazebo_camera_from_extrinsic.py"
test -x "${PREFIX}/lib/xgc_camera_calibration/resolve_extrinsic.py"
for page in extrinsic intrinsic; do
  test -f "${PREFIX}/share/xgc_camera_calibration/web/${page}/index.html"
  test -f "${PREFIX}/share/xgc_camera_calibration/web/${page}/app.js"
  test -f "${PREFIX}/share/xgc_camera_calibration/web/${page}/styles.css"
done
python3 -c 'from xgc_camera_calibration.extrinsic_application import ExtrinsicApplication, SavedExtrinsicApplication; from xgc_camera_calibration.extrinsic_resolver import decode_frozen; from xgc_camera_calibration.camera_initial_pose import resolve_gazebo_camera_pose_from_file; from xgc_camera_calibration.extrinsic_file_watcher import ExtrinsicSelectionWatcher; from xgc_camera_calibration.intrinsic_solver import calibrate_intrinsic; from xgc_camera_calibration.media_snapshot import MediaSnapshotClient; from xgc_camera_calibration.solver import load_extrinsic_selection, solve_extrinsic, write_extrinsic_selection; from xgc_camera_calibration.transforms import split_parent_to_optical_pose'
RUNTIME="$(mktemp -d)"
CALIBRATION_ROOT="${RUNTIME}/calibrations"
CAMERA_NAME="package_smoke"
INTRINSIC_FILE="${CALIBRATION_ROOT}/sim/${CAMERA_NAME}/intrinsics-20260101T000000.000000Z.yaml"
mkdir -p "$(dirname "${INTRINSIC_FILE}")"
cat >"${INTRINSIC_FILE}" <<'YAML'
schema: xgc2.camera.intrinsic.v1
created_at: '2026-01-01T00:00:00Z'
camera_name: package_smoke
image_width: 640
image_height: 480
camera_matrix:
  rows: 3
  cols: 3
  data: [500.0, 0.0, 319.5, 0.0, 500.0, 239.5, 0.0, 0.0, 1.0]
distortion_model: plumb_bob
distortion_coefficients:
  rows: 1
  cols: 5
  data: [0.0, 0.0, 0.0, 0.0, 0.0]
rectification_matrix:
  rows: 3
  cols: 3
  data: [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
projection_matrix:
  rows: 3
  cols: 4
  data: [500.0, 0.0, 319.5, 0.0, 0.0, 500.0, 239.5, 0.0, 0.0, 0.0, 1.0, 0.0]
YAML
FRAME_ROLES_JSON='{"parentFrame":"world","opticalFrames":{"sim":"xgc_world_camera_optical_frame","phy":"usb_cam_optical_frame"}}'
# Also verify the installed finite resolver works without a ROS PYTHONPATH.
RESOLVED_EXTRINSIC_JSON="$(env -u PYTHONPATH "${PREFIX}/lib/xgc_camera_calibration/resolve_extrinsic.py" \
  --root "${CALIBRATION_ROOT}" --camera "${CAMERA_NAME}" --selection-json '{"mode":"auto"}' \
  --target-offset-json '{"x":0,"y":0,"z":0}' --frame-roles-json "${FRAME_ROLES_JSON}")"
APPLICATION_STATE_PARAM="/xgc_camera_extrinsic_tf/extrinsic_application_state"
roslaunch --files xgc_camera_calibration extrinsic_calibrator.launch \
  resolved_extrinsic_json:="${RESOLVED_EXTRINSIC_JSON}" frame_roles_json:="${FRAME_ROLES_JSON}" \
  application_state_param:="${APPLICATION_STATE_PARAM}" pose_coordinate_source:=experiment-world \
  child_frame:=xgc_world_camera_optical_frame \
  calibration_root:="${CALIBRATION_ROOT}" calibration_mode:=sim \
  camera_name:="${CAMERA_NAME}" intrinsic_file:="${INTRINSIC_FILE}" >/dev/null
roslaunch --files xgc_camera_calibration intrinsic_calibrator.launch \
  calibration_root:="${CALIBRATION_ROOT}" calibration_mode:=sim \
  camera_name:=package_smoke >/dev/null
roslaunch --files xgc_camera_calibration extrinsic_tf.launch \
  resolved_extrinsic_json:="${RESOLVED_EXTRINSIC_JSON}" frame_roles_json:="${FRAME_ROLES_JSON}" \
  optical_frame:=xgc_world_camera_optical_frame \
  calibration_root:="${CALIBRATION_ROOT}" calibration_mode:=sim \
  camera_name:="${CAMERA_NAME}" >/dev/null

ROSCORE_PID=""
EXTRINSIC_PID=""
TF_PID=""
INTRINSIC_PID=""
MEDIA_EDGE_PID=""
cleanup() {
  if [[ -n "${INTRINSIC_PID}" ]]; then kill "${INTRINSIC_PID}" 2>/dev/null || true; fi
  if [[ -n "${EXTRINSIC_PID}" ]]; then kill "${EXTRINSIC_PID}" 2>/dev/null || true; fi
  if [[ -n "${TF_PID}" ]]; then kill "${TF_PID}" 2>/dev/null || true; fi
  if [[ -n "${MEDIA_EDGE_PID}" ]]; then kill "${MEDIA_EDGE_PID}" 2>/dev/null || true; fi
  if [[ -n "${ROSCORE_PID}" ]]; then kill "${ROSCORE_PID}" 2>/dev/null || true; fi
  wait "${INTRINSIC_PID}" 2>/dev/null || true
  wait "${EXTRINSIC_PID}" 2>/dev/null || true
  wait "${TF_PID}" 2>/dev/null || true
  wait "${MEDIA_EDGE_PID}" 2>/dev/null || true
  wait "${ROSCORE_PID}" 2>/dev/null || true
  rm -rf "${RUNTIME}"
}
trap cleanup EXIT
export ROS_MASTER_URI="http://127.0.0.1:11359"
export ROS_HOME="${RUNTIME}/ros-home"
export ROS_LOG_DIR="${RUNTIME}/ros-log"
mkdir -p "${ROS_HOME}" "${ROS_LOG_DIR}"
wait_http() {
  local port="$1" pid="$2"
  for _ in $(seq 1 100); do
    if python3 -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:${port}/healthz', timeout=1)" >/dev/null 2>&1; then
      return 0
    fi
    if ! kill -0 "${pid}" 2>/dev/null; then return 1; fi
    sleep 0.1
  done
  return 1
}
roscore -p 11359 >"${RUNTIME}/roscore.log" 2>&1 &
ROSCORE_PID="$!"
for _ in $(seq 1 50); do
  if rosparam list >/dev/null 2>&1; then break; fi
  sleep 0.1
done
rosparam list >/dev/null

python3 -c '
import json
from http.server import BaseHTTPRequestHandler, HTTPServer

class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path != "/healthz":
            self.send_error(404)
            return
        payload = json.dumps({"sources": [{"id": "usb_cam"}]}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, _format, *_args):
        pass

HTTPServer(("127.0.0.1", 18790), Handler).serve_forever()
' >"${RUNTIME}/media-edge.log" 2>&1 &
MEDIA_EDGE_PID="$!"
wait_http 18790 "${MEDIA_EDGE_PID}"

"${PREFIX}/lib/xgc_camera_calibration/extrinsic_tf_publisher.py" \
  --resolved-extrinsic-json "${RESOLVED_EXTRINSIC_JSON}" --frame-roles-json "${FRAME_ROLES_JSON}" \
  __name:=xgc_camera_extrinsic_tf \
  _calibration_root:="${CALIBRATION_ROOT}" _calibration_mode:=sim _camera_name:="${CAMERA_NAME}" \
  _parent_frame:=world _camera_link_frame:=package_smoke_link _optical_frame:=xgc_world_camera_optical_frame \
  >"${RUNTIME}/extrinsic-tf.log" 2>&1 &
TF_PID="$!"
python3 - "${APPLICATION_STATE_PARAM}" "${RESOLVED_EXTRINSIC_JSON}" <<'PYREADY'
import json, sys, time
import rospy
rospy.init_node("camera_package_wait_application", anonymous=True, disable_signals=True)
expected = json.loads(sys.argv[2])
for _ in range(100):
    value = rospy.get_param(sys.argv[1], {})
    if value.get("ready") is True and value.get("resolutionId") == expected["resolutionId"] and value.get("cameraName") == expected["cameraName"]:
        break
    time.sleep(.1)
else:
    raise RuntimeError("exact camera TF producer did not become ready")
PYREADY
"${PREFIX}/lib/xgc_camera_calibration/extrinsic_calibrator_web.py" \
  --resolved-extrinsic-json "${RESOLVED_EXTRINSIC_JSON}" --frame-roles-json "${FRAME_ROLES_JSON}" \
  --application-state-param "${APPLICATION_STATE_PARAM}" \
  __name:=xgc_camera_extrinsic_calibrator_web \
  _image_topic:=/not_installed_by_this_product/image_raw \
  _intrinsic_file:="${INTRINSIC_FILE}" \
  _calibration_root:="${CALIBRATION_ROOT}" _calibration_mode:=sim \
  _camera_name:="${CAMERA_NAME}" _http_port:=18765 \
  _parent_frame:=world _child_frame:=xgc_world_camera_optical_frame _pose_coordinate_source:=experiment-world \
  >"${RUNTIME}/extrinsic.log" 2>&1 &
EXTRINSIC_PID="$!"
"${PREFIX}/lib/xgc_camera_calibration/intrinsic_calibrator_web.py" \
  __name:=xgc_camera_intrinsic_calibrator_web \
  _media_edge_address:=http://127.0.0.1:18790 \
  _media_source_id:=usb_cam _snapshot_timeout:=1 \
  _http_port:=18766 _calibration_root:="${RUNTIME}/calibrations" \
  _calibration_mode:=sim _camera_name:=usb_cam \
  >"${RUNTIME}/intrinsic.log" 2>&1 &
INTRINSIC_PID="$!"

wait_http 18765 "${EXTRINSIC_PID}"
wait_http 18766 "${INTRINSIC_PID}"
python3 -c 'import json, urllib.request; p=json.load(urllib.request.urlopen("http://127.0.0.1:18765/healthz")); assert p["status"] == "ok" and not p["image_ready"] and p["intrinsic_ready"]'
python3 -c 'import json, urllib.request; p=json.load(urllib.request.urlopen("http://127.0.0.1:18766/healthz")); assert p["status"] == "ok" and not p["image_ready"] and not p["camera_control"]'
python3 -c 'import urllib.request; assert b"Camera extrinsic calibration" in urllib.request.urlopen("http://127.0.0.1:18765/").read()'
python3 -c 'import urllib.request; assert b"Camera intrinsic calibration" in urllib.request.urlopen("http://127.0.0.1:18766/").read()'

# Exercise the installed Save application owner against the real isolated ROS
# broadcaster. No camera image is fabricated and no user calibration is touched.
python3 - "${CALIBRATION_ROOT}" "${CAMERA_NAME}" "${RESOLVED_EXTRINSIC_JSON}" "${FRAME_ROLES_JSON}" "${APPLICATION_STATE_PARAM}" <<'PYAPPLY'
import json, sys, time
from pathlib import Path
import numpy as np
import rospy
from tf2_msgs.msg import TFMessage
from xgc_camera_calibration.extrinsic_application import SavedExtrinsicApplication
from xgc_camera_calibration.solver import ExtrinsicResult, save_extrinsic
from xgc_camera_calibration.extrinsic_coordinates import coordinate_provenance
root, camera, frozen, roles_json, parameter = sys.argv[1:]
roles = json.loads(roles_json)
rospy.init_node("camera_package_apply_check", anonymous=True, disable_signals=True)
output = Path(root) / "sim" / camera / "extrinsics-20260101T000001.000000Z.yaml"
result = ExtrinsicResult(translation=np.asarray([1., 2., 3.]), quaternion_xyzw=np.asarray([0., 0., 0., 1.]),
    rotation_world_to_camera=np.eye(3), translation_world_to_camera=np.asarray([-1., -2., -3.]),
    reprojection_errors_px=np.asarray([0., 0., 0., 0.]), inlier_indices=np.asarray([0, 1, 2, 3]), warnings=())
save_extrinsic(output, result, calibration_mode="sim", camera_name=camera, parent_frame="world",
    child_frame=roles["opticalFrames"]["sim"], metadata={"candidate_id":"package-smoke-candidate",
    "pose_coordinates":coordinate_provenance("experiment-world", "world", [0., 0., 0.])})
application = SavedExtrinsicApplication(root, camera, frozen, roles, lambda: rospy.get_param(parameter, {}))
receipt = application.stage(output, "sim", "package-smoke-candidate")
assert receipt["status"] in ("pending", "applied"), receipt
for _ in range(100):
    status = application.status()
    if status["status"] == "applied":
        break
    time.sleep(.1)
else:
    raise RuntimeError("saved result never reached applied: " + repr(status))
state = rospy.get_param(parameter)
assert state["active"]["applicationId"] == status["applicationId"]
message = rospy.wait_for_message("/tf_static", TFMessage, timeout=5)
transforms = {item.child_frame_id: item for item in message.transforms}
assert {"package_smoke_link", roles["opticalFrames"]["sim"]} <= set(transforms)
pose = transforms["package_smoke_link"].transform.translation
assert np.allclose([pose.x, pose.y, pose.z], [1., 2., 3.])
PYAPPLY

echo "Installed standalone ROS1 camera calibration package passed"
