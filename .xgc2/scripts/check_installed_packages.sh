#!/usr/bin/env bash
set -euo pipefail

ROS_DISTRO="${ROS_DISTRO:-noetic}"
PREFIX="/opt/ros/${ROS_DISTRO}"

# shellcheck disable=SC1090
source "${PREFIX}/setup.bash"
dpkg -s ros-noetic-xgc2-camera-calibration >/dev/null
test "$(rospack find xgc_camera_calibration)" = "${PREFIX}/share/xgc_camera_calibration"
test -x "${PREFIX}/lib/xgc_camera_calibration/extrinsic_calibrator_web.py"
test -x "${PREFIX}/lib/xgc_camera_calibration/intrinsic_calibrator_web.py"
test -x "${PREFIX}/lib/xgc_camera_calibration/extrinsic_tf_publisher.py"
for page in extrinsic intrinsic; do
  test -f "${PREFIX}/share/xgc_camera_calibration/web/${page}/index.html"
  test -f "${PREFIX}/share/xgc_camera_calibration/web/${page}/app.js"
  test -f "${PREFIX}/share/xgc_camera_calibration/web/${page}/styles.css"
done
python3 -c 'from xgc_camera_calibration.extrinsic_file_watcher import ExtrinsicDirectoryWatcher; from xgc_camera_calibration.intrinsic_solver import calibrate_intrinsic; from xgc_camera_calibration.media_snapshot import MediaSnapshotClient; from xgc_camera_calibration.solver import solve_extrinsic; from xgc_camera_calibration.transforms import split_parent_to_optical_pose'
RUNTIME="$(mktemp -d)"
INTRINSIC_FILE="${RUNTIME}/intrinsics.yaml"
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
roslaunch --files xgc_camera_calibration extrinsic_calibrator.launch \
  intrinsic_file:="${INTRINSIC_FILE}" >/dev/null
roslaunch --files xgc_camera_calibration intrinsic_calibrator.launch \
  calibration_root:="${RUNTIME}/calibrations" calibration_mode:=sim \
  camera_name:=package_smoke >/dev/null
roslaunch --files xgc_camera_calibration extrinsic_tf.launch >/dev/null

ROSCORE_PID=""
EXTRINSIC_PID=""
INTRINSIC_PID=""
MEDIA_EDGE_PID=""
cleanup() {
  if [[ -n "${INTRINSIC_PID}" ]]; then kill "${INTRINSIC_PID}" 2>/dev/null || true; fi
  if [[ -n "${EXTRINSIC_PID}" ]]; then kill "${EXTRINSIC_PID}" 2>/dev/null || true; fi
  if [[ -n "${MEDIA_EDGE_PID}" ]]; then kill "${MEDIA_EDGE_PID}" 2>/dev/null || true; fi
  if [[ -n "${ROSCORE_PID}" ]]; then kill "${ROSCORE_PID}" 2>/dev/null || true; fi
  wait "${INTRINSIC_PID}" 2>/dev/null || true
  wait "${EXTRINSIC_PID}" 2>/dev/null || true
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

"${PREFIX}/lib/xgc_camera_calibration/extrinsic_calibrator_web.py" \
  __name:=xgc_camera_extrinsic_calibrator_web \
  _image_topic:=/not_installed_by_this_product/image_raw \
  _intrinsic_file:="${INTRINSIC_FILE}" \
  _http_port:=18765 _output_file:="${RUNTIME}/extrinsics.yaml" \
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

echo "Installed standalone ROS1 camera calibration package passed"
