#!/usr/bin/env python3
"""Installed finite resolver; no ROS shell setup or network is required."""
from pathlib import Path
import sys

# A finite Python Job does not inherit roslaunch's PYTHONPATH. Resolve only
# this script's own source/install package, never another mutable workspace.
_here = Path(__file__).resolve()
for _candidate in (_here.parents[1] / "src", _here.parents[1] / "python3" / "dist-packages"):
    if (_candidate / "xgc_camera_calibration" / "extrinsic_resolver.py").is_file():
        sys.path.insert(0, str(_candidate))
        break

from xgc_camera_calibration.extrinsic_resolver import main

if __name__ == "__main__":
    raise SystemExit(main())
