"""Camera-link and optical-frame transform helpers."""

import math

import numpy as np

from .solver import CalibrationError, rotation_matrix_to_quaternion


CAMERA_LINK_TO_OPTICAL_TRANSLATION = np.zeros(3, dtype=np.float64)


def quaternion_to_rotation_matrix(quaternion_xyzw):
    quaternion = np.asarray(quaternion_xyzw, dtype=np.float64).reshape(4)
    if not np.all(np.isfinite(quaternion)):
        raise CalibrationError("quaternion contains a non-finite value")
    norm = float(np.linalg.norm(quaternion))
    if norm < 1e-12:
        raise CalibrationError("quaternion has zero norm")
    x, y, z, w = quaternion / norm
    return np.array(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def _rpy_matrix(roll, pitch, yaw):
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    rx = np.array([[1.0, 0.0, 0.0], [0.0, cr, -sr], [0.0, sr, cr]])
    ry = np.array([[cp, 0.0, sp], [0.0, 1.0, 0.0], [-sp, 0.0, cp]])
    rz = np.array([[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]])
    return rz.dot(ry).dot(rx)


def link_to_optical_rotation():
    """Return camera_link_R_camera_optical per REP-103."""

    return _rpy_matrix(-math.pi / 2.0, 0.0, -math.pi / 2.0)


def rotation_matrix_to_rpy(rotation):
    """Return fixed-axis roll, pitch, yaw for a finite rotation matrix."""

    matrix = np.asarray(rotation, dtype=np.float64).reshape(3, 3)
    if not np.all(np.isfinite(matrix)):
        raise CalibrationError("rotation matrix contains a non-finite value")
    pitch = math.asin(max(-1.0, min(1.0, -float(matrix[2, 0]))))
    if abs(math.cos(pitch)) > 1.0e-9:
        roll = math.atan2(float(matrix[2, 1]), float(matrix[2, 2]))
        yaw = math.atan2(float(matrix[1, 0]), float(matrix[0, 0]))
    else:
        roll = 0.0
        yaw = math.atan2(-float(matrix[0, 1]), float(matrix[1, 1]))
    return roll, pitch, yaw


def split_parent_to_optical_pose(
    translation, quaternion_xyzw,
    link_to_optical_translation=CAMERA_LINK_TO_OPTICAL_TRANSLATION,
):
    """Convert parent_T_optical into parent_T_link and link_T_optical.

    camera_link uses x-forward, y-left, z-up; camera_optical uses x-right,
    y-down, z-forward. Callers must supply their measured/model-owned optical
    offset; the generic physical-camera default remains coincident origins.
    """

    parent_r_optical = quaternion_to_rotation_matrix(quaternion_xyzw)
    link_r_optical = link_to_optical_rotation()
    parent_r_link = parent_r_optical.dot(link_r_optical.T)
    link_t_optical = np.asarray(link_to_optical_translation, dtype=np.float64).reshape(3)
    parent_t_optical = np.asarray(translation, dtype=np.float64).reshape(3)
    if not np.all(np.isfinite(link_t_optical)) or not np.all(np.isfinite(parent_t_optical)):
        raise CalibrationError("camera translation contains a non-finite value")
    return {
        "parent_t_link": parent_t_optical - parent_r_link.dot(link_t_optical),
        "parent_q_link_xyzw": rotation_matrix_to_quaternion(parent_r_link),
        "link_t_optical": link_t_optical,
        "link_q_optical_xyzw": rotation_matrix_to_quaternion(link_r_optical),
    }
