"""Resolve one immutable intrinsic input for an explicitly authored workflow."""
from __future__ import annotations

import argparse
import json
import sys
import yaml
from pathlib import Path

from .intrinsic_solver import load_intrinsic
from .intrinsic_validation import intrinsic_parameters
from .solver import CalibrationError, extrinsic_calibration_directory, selected_intrinsic_path


def read_candidate(root, mode, camera, path, image_size):
    selected = selected_intrinsic_path(root, mode, camera, str(path))
    document = load_intrinsic(selected)
    if document.get("camera_name") != camera:
        raise CalibrationError("intrinsic camera identity does not match")
    declared_mode = document.get("calibration_mode")
    if declared_mode is not None and declared_mode != mode:
        raise CalibrationError("intrinsic mode identity does not match")
    _, _, actual_size = intrinsic_parameters(document)
    if actual_size != tuple(image_size):
        raise CalibrationError("intrinsic image size {} does not match {}".format(actual_size, image_size))
    metadata = document.get("metadata", {})
    if not isinstance(metadata, dict):
        raise CalibrationError("intrinsic metadata must be a mapping")
    assessment = metadata.get("stability_assessment", {})
    if not isinstance(assessment, dict):
        raise CalibrationError("intrinsic assessment must be a mapping")
    if assessment and assessment.get("passed") is not True:
        raise CalibrationError("intrinsic quality assessment did not pass")
    return selected, document


def resolve_intrinsic(root, mode, camera, explicit, image_size, policy="latest"):
    """No writes; an explicit invalid selection never falls back to another file."""
    directory = extrinsic_calibration_directory(root, mode, camera)
    if policy not in ("latest", "default"):
        raise ValueError("intrinsic selection policy must be latest or default")
    rejected = []
    if explicit:
        candidates = [Path(explicit)]
    elif policy == "latest":
        candidates = sorted(directory.glob("intrinsics-*.yaml"), key=lambda path: path.name, reverse=True)
    else:
        candidates = []
    for path in candidates:
        try:
            selected, document = read_candidate(root, mode, camera, path, image_size)
        except (ValueError, TypeError, OSError, CalibrationError, yaml.YAMLError) as error:
            if explicit:
                raise
            rejected.append({"file": str(path), "reason": str(error)})
            continue
        return {
            "file": str(selected), "sha256": document["source_sha256"],
            "source": "explicit" if explicit else "latest",
            "calibration_mode": mode, "camera_name": camera,
            "image_size": list(image_size),
            "mode_identity": "document" if document.get("calibration_mode") else "legacy-directory",
            "rejected": rejected,
        }
    return {"file": "", "sha256": None, "source": "default",
            "calibration_mode": mode, "camera_name": camera,
            "image_size": list(image_size), "rejected": rejected}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    parser.add_argument("--mode", choices=("sim", "phy"), required=True)
    parser.add_argument("--camera", required=True)
    parser.add_argument("--file", default="")
    parser.add_argument("--policy", choices=("latest", "default"), required=True)
    parser.add_argument("--width", type=int, required=True)
    parser.add_argument("--height", type=int, required=True)
    args = parser.parse_args()
    receipt = resolve_intrinsic(args.root, args.mode, args.camera, args.file,
                                (args.width, args.height), args.policy)
    # stdout is the exact downstream binding; evidence stays in the Job log.
    print(json.dumps(receipt, sort_keys=True), file=sys.stderr)
    sys.stdout.write(receipt["file"])


if __name__ == "__main__":
    main()
