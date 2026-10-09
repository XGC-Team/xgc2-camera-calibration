"""Render an explicitly supplied immutable calibration spec for ROS consumers."""

import argparse
from decimal import Decimal
import hashlib
import json
import math
import os
from pathlib import Path
import stat
import tempfile


def numbers(value, size, label):
    if not isinstance(value, list) or len(value) != size or any(
        isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v)
        for v in value
    ):
        raise ValueError(label + " must contain {} finite numbers".format(size))
    return value


def scalar(value):
    # PyYAML treats some exponent spellings as strings. Emit decimal floats.
    if value == 0:
        return "0.0"
    text = format(Decimal(str(value)), "f")
    return text if "." in text else text + ".0"


def name(value, label):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(label + " is required")
    return json.dumps(value, ensure_ascii=False)


def matrix(label, rows, cols, values):
    return "{}:\n  rows: {}\n  cols: {}\n  data: [{}]\n".format(
        label, rows, cols, ", ".join(map(scalar, values)))


def render(spec, output_format):
    if not isinstance(spec, dict):
        raise ValueError("input must be one calibration spec object")
    if output_format == "camera_info_yaml":
        intrinsic = spec.get("intrinsics")
        if not isinstance(intrinsic, dict):
            raise ValueError("camera_info_yaml requires intrinsics")
        width, height = intrinsic.get("width"), intrinsic.get("height")
        if any(type(v) is not int or v <= 0 for v in (width, height)):
            raise ValueError("image dimensions must be positive integers")
        model = intrinsic.get("model")
        if model not in ("plumb_bob", "rational_polynomial"):
            raise ValueError("unsupported distortion model")
        k = numbers(intrinsic.get("k"), 9, "camera matrix")
        d = intrinsic.get("d")
        if not isinstance(d, list) or not 1 <= len(d) <= 16:
            raise ValueError("distortion requires 1 to 16 coefficients")
        numbers(d, len(d), "distortion")
        r = numbers(intrinsic.get("r", [1, 0, 0, 0, 1, 0, 0, 0, 1]), 9, "rectification")
        p = numbers(intrinsic.get("p", [k[0], k[1], k[2], 0, k[3], k[4], k[5], 0,
                                            k[6], k[7], k[8], 0]), 12, "projection")
        return ("image_width: {}\nimage_height: {}\ncamera_name: {}\n".format(
            width, height, name(spec.get("camera", {}).get("sourceId"), "camera source"))
            + matrix("camera_matrix", 3, 3, k) + "distortion_model: " + model + "\n"
            + matrix("distortion_coefficients", 1, len(d), d)
            + matrix("rectification_matrix", 3, 3, r) + matrix("projection_matrix", 3, 4, p))
    if output_format != "extrinsics_yaml":
        raise ValueError("unsupported output format")
    extrinsic = spec.get("extrinsics")
    if not isinstance(extrinsic, dict):
        raise ValueError("extrinsics_yaml requires extrinsics")
    translation = numbers(extrinsic.get("translation"), 3, "translation")
    roll, pitch, yaw = numbers(extrinsic.get("rotationRpy"), 3, "rotation")
    sr, cr = math.sin(roll / 2), math.cos(roll / 2)
    sp, cp = math.sin(pitch / 2), math.cos(pitch / 2)
    sy, cy = math.sin(yaw / 2), math.cos(yaw / 2)
    quaternion = [sr * cp * cy - cr * sp * sy, cr * sp * cy + sr * cp * sy,
                  cr * cp * sy - sr * sp * cy, cr * cp * cy + sr * sp * sy]
    text = "schema: xgc2.camera.extrinsic.v1\n"
    provenance = spec.get("provenance", {})
    if provenance.get("method") == "extrinsic-service" and provenance.get("capturedAt"):
        text += "created_at: " + name(provenance["capturedAt"], "capture time") + "\n"
    text += ("frame_convention: parent_T_camera_optical\nparent_frame: {}\nchild_frame: {}\n".format(
        name(extrinsic.get("parentFrame"), "parent frame"),
        name(extrinsic.get("childFrame"), "child frame")))
    for label, axes, values in (("translation", "xyz", translation),
                                ("quaternion_xyzw", "xyzw", quaternion)):
        text += label + ":\n" + "".join("  {}: {}\n".format(a, scalar(v)) for a, v in zip(axes, values))
    if extrinsic.get("cameraModel") is not None:
        text += "metadata:\n  camera_model: " + json.dumps(
            extrinsic["cameraModel"], ensure_ascii=False, allow_nan=False) + "\n"
    return text


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path, help="JSON file containing the exact pinned calibration spec")
    parser.add_argument("--format", required=True, choices=("camera_info_yaml", "extrinsics_yaml"))
    parser.add_argument("--output", required=True, type=Path, help="absolute output file under an existing granted directory")
    args = parser.parse_args()
    with args.input.open("rb") as stream:
        raw = stream.read(256 * 1024 + 1)
    if len(raw) > 256 * 1024:
        parser.error("calibration spec exceeds 256 KiB")
    payload = render(json.loads(raw), args.format).encode("utf-8")
    path = args.output
    if not path.is_absolute() or path.is_symlink():
        parser.error("output must be an absolute file path, not a symbolic link")
    if path.exists() and not stat.S_ISREG(path.stat().st_mode):
        parser.error("output must be a regular file")
    fd, temporary = tempfile.mkstemp(prefix="." + path.name + ".", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        directory = os.open(str(path.parent), os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    print(json.dumps({"outputPath": str(path), "bytesWritten": len(payload),
                      "digest": hashlib.sha256(payload).hexdigest()}))


if __name__ == "__main__":
    main()
