#!/usr/bin/env python3
"""Finite installed calibration renderer; accepts an exact spec, never looks up a head."""
from pathlib import Path
import sys

_here = Path(__file__).resolve()
for _candidate in (_here.parents[1] / "src", _here.parents[1] / "python3" / "dist-packages"):
    if (_candidate / "xgc_camera_calibration" / "asset_document.py").is_file():
        sys.path.insert(0, str(_candidate))
        break

from xgc_camera_calibration.asset_document import main

if __name__ == "__main__":
    main()
