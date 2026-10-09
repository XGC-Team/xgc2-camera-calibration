#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
ROS_DISTRO="${ROS_DISTRO:-noetic}"
INSTALL_ROOT=""
OUTPUT_DIR=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --install-root) INSTALL_ROOT="$2"; shift 2 ;;
    --output-dir) OUTPUT_DIR="$2"; shift 2 ;;
    *) echo "unknown argument: $1" >&2; exit 1 ;;
  esac
done

if [[ -z "${INSTALL_ROOT}" || -z "${OUTPUT_DIR}" ]]; then
  echo "--install-root and --output-dir are required" >&2
  exit 1
fi

VERSION="${PACKAGE_VERSION:-$(sed -nE 's/^version:[[:space:]]*([^[:space:]#]+).*/\1/p' "${REPO_ROOT}/.xgc2/product.yml" | head -n1)}"
if [[ -z "${VERSION}" ]]; then
  echo "package version is missing" >&2
  exit 1
fi

ARCH="$(dpkg --print-architecture)"
PREFIX="/opt/ros/${ROS_DISTRO}"
BUILD_ROOT="$(mktemp -d)"
trap 'rm -rf "${BUILD_ROOT}"' EXIT
PACKAGE_NAME="ros-noetic-xgc2-camera-calibration"
PACKAGE_ROOT="${BUILD_ROOT}/${PACKAGE_NAME}"
mkdir -p "${PACKAGE_ROOT}" "${OUTPUT_DIR}"

copy_path() {
  local source="$1"
  if [[ ! -e "${source}" ]]; then return; fi
  local relative="${source#${INSTALL_ROOT}}"
  mkdir -p "${PACKAGE_ROOT}$(dirname "${relative}")"
  cp -a "${source}" "${PACKAGE_ROOT}${relative}"
}

copy_path "${INSTALL_ROOT}${PREFIX}/share/xgc_camera_calibration"
copy_path "${INSTALL_ROOT}${PREFIX}/lib/xgc_camera_calibration"
copy_path "${INSTALL_ROOT}${PREFIX}/lib/python3/dist-packages/xgc_camera_calibration"

mkdir -p "${PACKAGE_ROOT}/DEBIAN" "${PACKAGE_ROOT}/usr/share/doc/${PACKAGE_NAME}"
cat >"${PACKAGE_ROOT}/DEBIAN/control" <<EOF
Package: ${PACKAGE_NAME}
Version: ${VERSION}
Section: misc
Priority: optional
Architecture: ${ARCH}
Maintainer: XGC2 <dev@xiaokang.ink>
Depends: python3 (>= 3.8), python3-numpy, python3-opencv, python3-rospkg, python3-yaml, ros-noetic-gazebo-msgs, ros-noetic-geometry-msgs, ros-noetic-rosbash, ros-noetic-roslaunch, ros-noetic-rospy, ros-noetic-sensor-msgs, ros-noetic-tf, ros-noetic-tf2-ros
Provides: ros-noetic-xgc-camera-calibration
Conflicts: ros-noetic-xgc-camera-calibration
Replaces: ros-noetic-xgc-camera-calibration
Description: Independent intrinsic and fixed-world-camera extrinsic calibration for ROS Noetic
EOF
install -m 0644 "${REPO_ROOT}/LICENSE" "${PACKAGE_ROOT}/usr/share/doc/${PACKAGE_NAME}/copyright"

test -f "${PACKAGE_ROOT}${PREFIX}/lib/python3/dist-packages/xgc_camera_calibration/solver.py"
test -f "${PACKAGE_ROOT}${PREFIX}/share/xgc_camera_calibration/web/intrinsic/index.html"
test -f "${PACKAGE_ROOT}${PREFIX}/share/xgc_camera_calibration/web/extrinsic/index.html"
find "${PACKAGE_ROOT}" -type d -name __pycache__ -prune -exec rm -rf {} +
find "${PACKAGE_ROOT}" -type d -exec chmod 0755 {} +
find "${PACKAGE_ROOT}" -type f -exec chmod 0644 {} +
if [[ -d "${PACKAGE_ROOT}${PREFIX}/lib/xgc_camera_calibration" ]]; then
  find "${PACKAGE_ROOT}${PREFIX}/lib/xgc_camera_calibration" -type f -exec chmod 0755 {} +
fi
chmod 0755 "${PACKAGE_ROOT}/DEBIAN"
fakeroot dpkg-deb --build "${PACKAGE_ROOT}" "${OUTPUT_DIR}/${PACKAGE_NAME}_${VERSION}_${ARCH}.deb" >/dev/null
find "${OUTPUT_DIR}" -maxdepth 1 -type f -name '*.deb' -print | sort
