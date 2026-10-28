#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
DOCKER_IMAGE="${DOCKER_IMAGE:-ghcr.io/xgc-team/xgc2-images/xgc2-build-focal-full-noetic:1.0.8@sha256:fce2d76fddf4f6439bf0a188249b731650febdc163befc360bed186b269d252a}"
WORK_DIR="${WORK_DIR:-${REPO_ROOT}/.work/docker}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/debs}"
INSTALL_CHECK="${INSTALL_CHECK:-true}"
SDK_WHEEL=""
LOCAL_IMAGE=false
NETWORK_NONE=false
CIDFILE=""
BUILD_LABEL=""
BUILD_USER=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --image) DOCKER_IMAGE="$2"; shift 2 ;;
    --work-dir) WORK_DIR="$2"; shift 2 ;;
    --output-dir) OUTPUT_DIR="$2"; shift 2 ;;
    --sdk-wheel) SDK_WHEEL="$2"; shift 2 ;;
    --local-image) LOCAL_IMAGE=true; shift ;;
    --network-none) NETWORK_NONE=true; shift ;;
    --cidfile) CIDFILE="$2"; shift 2 ;;
    --build-label) BUILD_LABEL="$2"; shift 2 ;;
    --build-user) BUILD_USER="$2"; shift 2 ;;
    --skip-install-check) INSTALL_CHECK=false; shift ;;
    *) echo "unknown argument: $1" >&2; exit 1 ;;
  esac
done

mkdir -p "${WORK_DIR}" "${OUTPUT_DIR}"
if [[ "${LOCAL_IMAGE}" == true ]]; then
  docker image inspect "${DOCKER_IMAGE}" >/dev/null
else
  docker pull "${DOCKER_IMAGE}"
fi
docker_args=(--rm --entrypoint /bin/bash)
if [[ -n "${SDK_WHEEL}" ]]; then
  SDK_WHEEL="$(realpath -e "${SDK_WHEEL}")"
  test -f "${SDK_WHEEL}"
  docker_args+=(--mount "type=bind,source=${SDK_WHEEL},target=/workspace/sdk-wheel.whl,readonly")
fi
if [[ "${NETWORK_NONE}" == true ]]; then docker_args+=(--network none); fi
if [[ -n "${CIDFILE}" ]]; then docker_args+=(--cidfile "${CIDFILE}"); fi
if [[ -n "${BUILD_LABEL}" ]]; then docker_args+=(--label "${BUILD_LABEL}"); fi
if [[ -n "${BUILD_USER}" ]]; then docker_args+=(--user "${BUILD_USER}"); fi
docker run "${docker_args[@]}" \
  -e DEBIAN_FRONTEND=noninteractive \
  -e INSTALL_CHECK="${INSTALL_CHECK}" \
  -e SDK_WHEEL_SUPPLIED="$([[ -n "${SDK_WHEEL}" ]] && echo true || echo false)" \
  -e PACKAGE_VERSION="${PACKAGE_VERSION:-}" \
  -e XGC2_APT_OVERLAY_URL="${XGC2_APT_OVERLAY_URL:-}" \
  -v "${REPO_ROOT}:/workspace/repo:ro" \
  -v "${WORK_DIR}:/workspace/work" \
  -v "${OUTPUT_DIR}:/workspace/out" \
  "${DOCKER_IMAGE}" -lc '
    set -euo pipefail
    rm -rf /workspace/work/src /workspace/work/build /workspace/work/devel /workspace/work/install-root
    mkdir -p \
      /workspace/work/src/xgc_camera_calibration
    rsync -a --delete --exclude="__pycache__/" --exclude="*.pyc" --exclude=".pytest_cache/" /workspace/repo/xgc_camera_calibration/ /workspace/work/src/xgc_camera_calibration/
    rsync -a --delete /workspace/repo/process-definitions/ /workspace/work/src/process-definitions/
    cd /workspace/work
    source /opt/ros/noetic/setup.bash
    # First-party official SDK 0.2.1 (path-only discovery, unbound describe query);
    # third-party modules remain image-owned.
    sdk_wheel=/workspace/work/xgc2_xrpc-0.2.1-py3-none-any.whl
    if [[ "${SDK_WHEEL_SUPPLIED}" == true ]]; then
      cp /workspace/sdk-wheel.whl "$sdk_wheel"
    else
      curl -fsSL --retry 5 \
        https://github.com/XGC-Team/xgc2-xrpc/releases/download/v0.2.1-1/xgc2_xrpc-0.2.1-py3-none-any.whl \
        -o "$sdk_wheel"
    fi
    echo "d2b8acd6677d6471b967d4724b1a3457a43893f4a97c582f3a189e039eff7d5e  $sdk_wheel" | sha256sum -c -
    rm -rf /workspace/work/sdk
    python3 -m zipfile -e "$sdk_wheel" /workspace/work/sdk
    printf "%s\n" "{\"url\":\"https://github.com/XGC-Team/xgc2-xrpc/releases/download/v0.2.1-1/xgc2_xrpc-0.2.1-py3-none-any.whl\",\"archive_info\":{\"hashes\":{\"sha256\":\"d2b8acd6677d6471b967d4724b1a3457a43893f4a97c582f3a189e039eff7d5e\"}}}" \
      > /workspace/work/sdk/xgc2_xrpc-0.2.1.dist-info/direct_url.json
    export PYTHONPATH="/workspace/work/sdk:${PYTHONPATH:-}"
    python3 /workspace/repo/.xgc2/scripts/check_python_xrpc.py
    python3 /workspace/repo/.xgc2/scripts/check_python_opencv.py
    catkin_make -j1 -DCMAKE_BUILD_TYPE=RelWithDebInfo
    ROS_HOME=/workspace/work/ros-home ROS_LOG_DIR=/workspace/work/ros-log catkin_make -j1 run_tests
    catkin_test_results --verbose
    DESTDIR=/workspace/work/install-root catkin_make -j1 install -DCMAKE_INSTALL_PREFIX=/opt/ros/noetic -DCATKIN_ENABLE_TESTING=OFF
    /workspace/repo/.xgc2/scripts/package_debs.sh --install-root /workspace/work/install-root --output-dir /workspace/out
    if [[ "${INSTALL_CHECK}" == true ]]; then
      apt-get install -y /workspace/out/ros-noetic-xgc2-camera-calibration_*.deb
      /workspace/repo/.xgc2/scripts/check_installed_packages.sh
    fi
  '

find "${OUTPUT_DIR}" -maxdepth 1 -type f -name '*.deb' -print | sort
