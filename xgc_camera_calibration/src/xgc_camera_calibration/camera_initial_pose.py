"""Resolve an explicit physical extrinsic into a Gazebo camera-link pose."""

from __future__ import annotations

from typing import Dict, Iterable, List, Sequence, Tuple

from .solver import CalibrationError, load_extrinsic_selection
from .transforms import (
    quaternion_to_rotation_matrix,
    rotation_matrix_to_rpy,
    split_parent_to_optical_pose,
)


def resolve_gazebo_camera_initial_pose(
    calibration_root: str,
    camera_name: str,
    parent_frame: str,
    optical_frame: str,
    link_to_optical_translation: Sequence[float],
) -> Tuple[Dict[str, float], str]:
    if parent_frame != "world":
        raise CalibrationError(
            "Gazebo camera physical-selection requires world-frame extrinsics"
        )
    selected = load_extrinsic_selection(calibration_root, "phy", camera_name)
    if selected is None:
        raise CalibrationError(
            "no shared physical extrinsic selection exists for camera {}".format(
                camera_name
            )
        )
    path, document, _selection = selected
    if document.get("parent_frame") != parent_frame:
        raise CalibrationError(
            "physical extrinsic parent frame does not match the simulation world"
        )
    if document.get("child_frame") != optical_frame:
        raise CalibrationError(
            "physical extrinsic optical frame does not match the simulation camera"
        )
    chain = split_parent_to_optical_pose(
        document["translation_array"],
        document["quaternion_xyzw_array"],
        link_to_optical_translation,
    )
    rotation = quaternion_to_rotation_matrix(chain["parent_q_link_xyzw"])
    roll, pitch, yaw = rotation_matrix_to_rpy(rotation)
    translation = chain["parent_t_link"]
    return {
        "x": float(translation[0]),
        "y": float(translation[1]),
        "z": float(translation[2]),
        "roll": float(roll),
        "pitch": float(pitch),
        "yaw": float(yaw),
    }, str(path)


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
        result[matches[0]] = "{}:={:.17g}".format(name, float(pose[name]))
    return result
