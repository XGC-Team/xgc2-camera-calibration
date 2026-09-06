#!/usr/bin/env bash
set -eo pipefail
# The workflow passes literal argv; selection policy belongs to that workflow.
source /opt/ros/noetic/setup.bash
resolver_source="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../src" 2>/dev/null && pwd)" || resolver_source=""
if [[ -n "$resolver_source" && -d "$resolver_source/xgc_camera_calibration" ]]; then
  export PYTHONPATH="$resolver_source${PYTHONPATH:+:$PYTHONPATH}"
fi
exec python3 -m xgc_camera_calibration.intrinsic_selection "$@"
