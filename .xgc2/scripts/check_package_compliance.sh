#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

for script in .xgc2/scripts/*.sh; do
  bash -n "${script}"
done

PYTHONPYCACHEPREFIX="${TMPDIR:-/tmp}/xgc2-camera-pycache" python3 -m py_compile \
  xgc_camera_calibration/scripts/*.py \
  xgc_camera_calibration/src/xgc_camera_calibration/*.py \
  xgc_camera_calibration/test/*.py \
  .xgc2/scripts/xgc2_artifact_manifest.py

MANIFEST_TEST_ROOT="$(mktemp -d)"
trap 'rm -rf "${MANIFEST_TEST_ROOT}"' EXIT
MANIFEST_TEST_ARCH="$(dpkg --print-architecture)"
mkdir -p \
  "${MANIFEST_TEST_ROOT}/package/DEBIAN" \
  "${MANIFEST_TEST_ROOT}/debs" \
  "${MANIFEST_TEST_ROOT}/manifests"
printf '%s\n' \
  'Package: xgc2-camera-calibration-manifest-contract' \
  'Version: 0.3.0-2' \
  'Section: misc' \
  'Priority: optional' \
  "Architecture: ${MANIFEST_TEST_ARCH}" \
  'Maintainer: XGC2 <dev@xiaokang.ink>' \
  'Description: XGC2 camera calibration build-manifest contract test' \
  >"${MANIFEST_TEST_ROOT}/package/DEBIAN/control"
dpkg-deb --build \
  "${MANIFEST_TEST_ROOT}/package" \
  "${MANIFEST_TEST_ROOT}/debs/xgc2-camera-calibration-manifest-contract_0.3.0-2_${MANIFEST_TEST_ARCH}.deb" \
  >/dev/null
python3 .xgc2/scripts/xgc2_artifact_manifest.py build \
  --deb-dir "${MANIFEST_TEST_ROOT}/debs" \
  --output-dir "${MANIFEST_TEST_ROOT}/manifests" \
  --product xgc2-camera-calibration-ros1 \
  --product-version 0.3.0-2 \
  --distribution focal \
  --architecture "${MANIFEST_TEST_ARCH}" \
  --source-sha 0000000000000000000000000000000000000000 \
  --ci-run-id compliance \
  --ci-workflow ci \
  --ci-workflow-ref refs/heads/main

MANIFEST_TEST_ROOT="${MANIFEST_TEST_ROOT}" python3 - <<'PY'
import json
import os
import pathlib
import xml.etree.ElementTree as ET

root = pathlib.Path(".")
for path in sorted(root.glob("xgc_camera_*/package.xml")):
    ET.parse(path)
for path in sorted(root.glob("xgc_camera_*/*/*.launch")):
    ET.parse(path)

manifest_paths = list(
    (pathlib.Path(os.environ["MANIFEST_TEST_ROOT"]) / "manifests").glob("*.json")
)
assert len(manifest_paths) == 1
manifest = json.loads(manifest_paths[0].read_text(encoding="utf-8"))
assert set(manifest) == {
    "schema", "product", "source_sha", "version", "distribution",
    "architecture", "ci", "created_at", "debs",
}
assert manifest["schema"] == "xgc2.build-artifact.v1"
assert manifest["product"] == "xgc2-camera-calibration-ros1"
assert manifest["version"] == "0.3.0-2"
assert set(manifest["ci"]) == {"run_id", "workflow", "workflow_ref"}
assert len(manifest["debs"]) == 1
deb = manifest["debs"][0]
assert set(deb) == {
    "file", "package", "version", "architecture", "sha256", "size",
}
assert deb["package"] == "xgc2-camera-calibration-manifest-contract"
assert deb["version"] == "0.3.0-2"
assert len(deb["sha256"]) == 64
assert deb["size"] > 0
PY

grep -q '^id: xgc2-camera-calibration-ros1$' .xgc2/product.yml
grep -q '^version: 0.3.0-30$' .xgc2/product.yml
grep -q '^    focal: 0.3.0-30$' .xgc2/product.yml
grep -Fq 'DOCKER_IMAGE="${DOCKER_IMAGE:-ghcr.io/xgc-team/xgc2-images/xgc2-build-focal-full-noetic:1.0.1}"' \
  .xgc2/scripts/build_debs_in_docker.sh
if grep -q '^    focal: .*~focal' .xgc2/product.yml; then
  echo "single-distribution ROS1 package version must not retain a focal suffix" >&2
  exit 1
fi
grep -q '<exec_depend>gazebo_msgs</exec_depend>' xgc_camera_calibration/package.xml
grep -q '<exec_depend>tf</exec_depend>' xgc_camera_calibration/package.xml
grep -q 'catkin_add_nosetests(test/test_media_snapshot.py)' xgc_camera_calibration/CMakeLists.txt
grep -q 'http://127.0.0.1:18090' \
  xgc_camera_calibration/scripts/intrinsic_calibrator_web.py \
  xgc_camera_calibration/launch/intrinsic_calibrator.launch
grep -q '<arg name="media_source_id" default="usb_cam"' \
  xgc_camera_calibration/launch/intrinsic_calibrator.launch
grep -q '<arg name="snapshot_timeout" default="5.0"' \
  xgc_camera_calibration/launch/intrinsic_calibrator.launch
if grep -Eq '<arg name="(image_topic|camera_info_topic|intrinsic_rate)"' \
  xgc_camera_calibration/launch/intrinsic_calibrator.launch; then
  echo "intrinsic launch retained a retired ROS image polling parameter" >&2
  exit 1
fi
if grep -Eq 'camera_info_topic|_camera_info_topic' \
  xgc_camera_calibration/launch/extrinsic_calibrator.launch \
  xgc_camera_calibration/scripts/extrinsic_calibrator_web.py \
  .xgc2/scripts/check_installed_packages.sh; then
  echo "extrinsic calibration retained the removed CameraInfo topic contract" >&2
  exit 1
fi
if grep -Eq 'camera_info_(topic|ready)|CameraInfo|info-topic' \
  web-src/src/extrinsic-legacy.ts web-src/src/main.tsx \
  xgc_camera_calibration/src/xgc_camera_calibration/web_service.py; then
  echo "extrinsic WebUI retained the removed CameraInfo compatibility state" >&2
  exit 1
fi
if grep -Eq '<arg name="(output_file|references_dir)"|<param name="(output_file|references_dir)"' \
  xgc_camera_calibration/launch/intrinsic_calibrator.launch; then
  echo "intrinsic launch retained ignored output/reference compatibility parameters" >&2
  exit 1
fi
for required_arg in calibration_root calibration_mode camera_name; do
  grep -q "<arg name=\"${required_arg}\" />" \
    xgc_camera_calibration/launch/intrinsic_calibrator.launch
done
if grep -R -E 'intrinsic-validation\.v1|generate_intrinsic_validation|validate_intrinsic_comparison' \
  xgc_camera_calibration/src xgc_camera_calibration/scripts; then
  echo "intrinsic validation retained the removed v1 compatibility contract" >&2
  exit 1
fi
grep -q 'CompressedImage' xgc_camera_calibration/scripts/extrinsic_calibrator_web.py
grep -q 'wait_for_message' xgc_camera_calibration/scripts/extrinsic_calibrator_web.py
if grep -q 'Subscriber(.*self.image_topic' \
  xgc_camera_calibration/scripts/extrinsic_calibrator_web.py; then
  echo "extrinsic calibrator must not retain a steady-state raw image subscription" >&2
  exit 1
fi
grep -q 'ros-noetic-gazebo-msgs' .xgc2/product.yml
grep -q 'ros-noetic-tf$' .xgc2/product.yml
grep -q 'ros-noetic-gazebo-msgs.*ros-noetic-geometry-msgs' .xgc2/scripts/package_debs.sh
grep -q 'ros-noetic-sensor-msgs.*ros-noetic-tf,.*ros-noetic-tf2-ros' .xgc2/scripts/package_debs.sh
for page in extrinsic intrinsic; do
  test -f "xgc_camera_calibration/web/${page}/index.html"
  test -f "xgc_camera_calibration/web/${page}/app.js"
  test -f "xgc_camera_calibration/web/${page}/styles.css"
done
if grep -R --exclude-dir=__pycache__ -E '(PyQt|python3-pyqt5|extrinsic_calibrator_ui)' \
  .xgc2/product.yml .xgc2/scripts/package_debs.sh \
  .xgc2/scripts/build_debs_in_docker.sh \
  .xgc2/scripts/check_installed_packages.sh >/dev/null; then
  echo "desktop Qt dependency leaked into the WebUI camera calibrator" >&2
  exit 1
fi
if grep -R -E -i '(xgc_camera_driver|libxgc2-camera-dev|ros-noetic-xgc-camera-driver)' \
  .xgc2/scripts/build_debs_in_docker.sh .xgc2/scripts/package_debs.sh \
  .xgc2/scripts/check_installed_packages.sh >/dev/null; then
  echo "camera driver dependency leaked into the independent calibration product" >&2
  exit 1
fi

BUILD_SCRIPT_PATH=.xgc2/scripts/build_debs_in_docker.sh python3 - <<'PY'
import os
from pathlib import Path

lines = Path(os.environ["BUILD_SCRIPT_PATH"]).read_text(encoding="utf-8").splitlines()
source_dir = "      /workspace/work/src/xgc_camera_calibration"
index = lines.index(source_dir)
assert not lines[index].rstrip().endswith("\\"), (
    "the final mkdir argument must not continue into rsync"
)
assert lines[index + 1].startswith("    rsync -a --delete "), (
    "rsync must be its own Docker build command"
)
PY

echo "ROS1 camera product compliance passed"
