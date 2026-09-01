"""Resolve an explicit extrinsic YAML into a Gazebo camera-link pose."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

from .solver import (
    EXTRINSIC_FILENAME_PATTERN,
    CalibrationError,
    load_extrinsic,
)
from .transforms import (
    quaternion_to_rotation_matrix,
    rotation_matrix_to_rpy,
    split_parent_to_optical_pose,
)


def gazebo_camera_link_pose_from_optical(
    translation: Sequence[float],
    quaternion_xyzw: Sequence[float],
    link_to_optical_translation: Sequence[float],
) -> Dict[str, float]:
    chain = split_parent_to_optical_pose(
        translation,
        quaternion_xyzw,
        link_to_optical_translation,
    )
    rotation = quaternion_to_rotation_matrix(chain["parent_q_link_xyzw"])
    roll, pitch, yaw = rotation_matrix_to_rpy(rotation)
    link_translation = chain["parent_t_link"]
    return {
        "x": float(link_translation[0]),
        "y": float(link_translation[1]),
        "z": float(link_translation[2]),
        "roll": float(roll),
        "pitch": float(pitch),
        "yaw": float(yaw),
    }


def authored_gazebo_extrinsic_path(
    calibration_root: str,
    camera_name: str,
    extrinsic_file: str,
) -> Path:
    """Return one concrete extrinsics-UTC.yaml under {root}/{sim|phy}/{camera}/."""

    authored = Path(str(extrinsic_file).strip()).expanduser()
    if not str(extrinsic_file).strip():
        raise CalibrationError("extrinsic file is required when pose source is file")
    if not authored.is_absolute():
        raise CalibrationError("extrinsic file must be absolute")
    if authored.is_symlink():
        raise CalibrationError("extrinsic file must not be a symbolic link")
    try:
        canonical_root = Path(str(calibration_root)).expanduser().resolve(strict=True)
        selected = authored.resolve(strict=True)
    except OSError as error:
        raise CalibrationError("extrinsic file must resolve to an existing file") from error
    try:
        relative = selected.relative_to(canonical_root)
    except ValueError as error:
        raise CalibrationError("extrinsic file must be under the calibration root") from error
    identity = str(camera_name).strip()
    parts = relative.parts
    if (
        len(parts) != 3
        or parts[0] not in ("sim", "phy")
        or parts[1] != identity
        or not selected.is_file()
        or not EXTRINSIC_FILENAME_PATTERN.fullmatch(selected.name)
    ):
        raise CalibrationError(
            "extrinsic file must be a concrete extrinsics-UTC.yaml under {}/{{sim|phy}}/{}/".format(
                canonical_root.as_posix(), identity
            )
        )
    return selected


def resolve_gazebo_camera_pose_from_file(
    calibration_root: str,
    camera_name: str,
    parent_frame: str,
    allowed_optical_frames: Sequence[str],
    link_to_optical_translation: Sequence[float],
    extrinsic_file: str,
) -> Tuple[Dict[str, float], str]:
    if parent_frame != "world":
        raise CalibrationError(
            "Gazebo camera file pose requires world-frame extrinsics"
        )
    selected = authored_gazebo_extrinsic_path(
        calibration_root, camera_name, extrinsic_file
    )
    document = load_extrinsic(selected)
    if document.get("camera_name") != str(camera_name).strip():
        raise CalibrationError("extrinsic camera name does not match the simulation camera")
    _require_world_optical_document(document, parent_frame, allowed_optical_frames)
    return gazebo_camera_link_pose_from_optical(
        document["translation_array"],
        document["quaternion_xyzw_array"],
        link_to_optical_translation,
    ), str(selected)


def replace_roslaunch_pose_arguments(
    arguments: Iterable[str], pose: Dict[str, float]
) -> List[str]:
    result = list(arguments)
    for name in ("x", "y", "z", "roll", "pitch", "yaw"):
        matches = [index for index, value in enumerate(result) if value.startswith(name + ":=")]
        if len(matches) != 1:
            raise CalibrationError(
                "camera roslaunch arguments require exactly one {} assignment".format(name)
            )
        result[matches[0]] = "{}:={}".format(name, _roslaunch_decimal(pose[name]))
    return result


def _roslaunch_decimal(value: float) -> str:
    """Serialize a finite float without exponent syntax for ROS argparse."""

    number = float(value)
    if not math.isfinite(number):
        raise CalibrationError("camera roslaunch pose values must be finite")
    text = format(number, ".17f").rstrip("0").rstrip(".")
    return "0" if text in ("", "-0") else text


def _require_world_optical_document(
    document: Dict[str, object],
    parent_frame: str,
    allowed_optical_frames: Sequence[str],
) -> None:
    if document.get("parent_frame") != parent_frame:
        raise CalibrationError(
            "extrinsic parent frame does not match the simulation world"
        )
    allowed = {str(frame).strip() for frame in allowed_optical_frames if str(frame).strip()}
    child = document.get("child_frame")
    if child not in allowed:
        raise CalibrationError(
            "extrinsic optical frame does not match the simulation camera"
        )
