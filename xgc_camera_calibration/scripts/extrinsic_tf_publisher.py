#!/usr/bin/env python3
"""Publish one frozen camera pose; acknowledge only this Run's exact saves."""
import sys
import math
import time

import rospy
import tf2_ros
from geometry_msgs.msg import TransformStamped

from xgc_camera_calibration.extrinsic_application import (
    ExtrinsicApplication, application_arguments, parse_frame_roles,
)
from xgc_camera_calibration.transforms import split_parent_to_optical_pose


def make_transform(parent_frame, child_frame, translation, quaternion):
    message = TransformStamped()
    message.header.frame_id = parent_frame
    message.child_frame_id = child_frame
    message.transform.translation.x = float(translation[0])
    message.transform.translation.y = float(translation[1])
    message.transform.translation.z = float(translation[2])
    message.transform.rotation.x = float(quaternion[0])
    message.transform.rotation.y = float(quaternion[1])
    message.transform.rotation.z = float(quaternion[2])
    message.transform.rotation.w = float(quaternion[3])
    return message


def default_transform_chain(
    parent_frame,
    camera_link_frame,
    optical_frame,
    parent_t_optical=(0.0, 0.0, 0.0),
    parent_q_optical_xyzw=(0.0, 0.0, 0.0, 1.0),
    parent_offsets=(0.0, 0.0, 0.0),
    link_to_optical_translation=(0.0, 0.0, 0.0),
):
    optical_translation = tuple(
        float(parent_t_optical[index]) + float(parent_offsets[index])
        for index in range(3)
    )
    chain = split_parent_to_optical_pose(
        optical_translation,
        parent_q_optical_xyzw,
        tuple(float(value) for value in link_to_optical_translation),
    )
    return (
        make_transform(
            parent_frame,
            camera_link_frame,
            chain["parent_t_link"],
            chain["parent_q_link_xyzw"],
        ),
        make_transform(
            camera_link_frame,
            optical_frame,
            chain["link_t_optical"],
            chain["link_q_optical_xyzw"],
        ),
    )


def frozen_transform_chain(frozen, parent_frame, camera_link_frame, optical_frame,
                           link_offset):
    pose = frozen["resolvedOpticalPose"]
    return default_transform_chain(parent_frame, camera_link_frame, optical_frame,
        pose["translation"], pose["quaternionXyzw"], link_to_optical_translation=link_offset)


def publish_chain(transforms, broadcaster, optical_broadcaster=None):
    stamp = rospy.Time.now()
    for transform in transforms:
        transform.header.stamp = stamp
    if optical_broadcaster is None:
        broadcaster.sendTransform(list(transforms))
    else:
        optical_broadcaster.sendTransform(transforms[1])
        broadcaster.sendTransform(transforms[0])


def main():
    args = application_arguments(rospy.myargv()[1:])
    rospy.init_node("xgc_camera_extrinsic_tf")
    try:
        root = str(rospy.get_param("~calibration_root")).strip()
        mode = str(rospy.get_param("~calibration_mode")).strip()
        camera_name = str(rospy.get_param("~camera_name")).strip()
        roles = parse_frame_roles(args.frame_roles_json)
        parent = str(rospy.get_param("~parent_frame", "world"))
        optical = str(rospy.get_param("~optical_frame", "usb_cam_optical_frame"))
        link = str(rospy.get_param("~camera_link_frame", "usb_cam_link"))
        if mode not in ("sim", "phy") or parent != roles["parentFrame"] or optical != roles["opticalFrames"][mode]:
            raise ValueError("output frames do not match the controlled camera roles")
        link_offset = tuple(float(rospy.get_param("~link_to_optical_" + axis, 0.0)) for axis in ("x", "y", "z"))
        if not all(math.isfinite(value) for value in link_offset):
            raise ValueError("camera link offset must be finite")
        application = ExtrinsicApplication(root, camera_name, args.resolved_extrinsic_json, roles,
            lambda state: rospy.set_param("~extrinsic_application_state", state))
        frozen = application.frozen
        # Only explicit uncalibrated auto uses the declared raw-origin identity.
        # The target offset comes from the frozen resolver, not mutable ROS params.
        if frozen["status"] == "uncalibrated":
            transforms = default_transform_chain(parent, link, optical,
                parent_offsets=frozen["targetCoordinates"]["worldOffset"],
                link_to_optical_translation=link_offset)
            rospy.logwarn("Publishing default camera extrinsic; no applied calibration exists")
        else:
            transforms = frozen_transform_chain(frozen, parent, link, optical, link_offset)
        static = bool(rospy.get_param("~static", True))
        rate = float(rospy.get_param("~file_poll_rate" if static else "~publish_rate", 5.0 if static else 10.0))
        if not math.isfinite(rate) or rate <= 0:
            raise ValueError("camera publication rate must be finite and positive")
        broadcaster = tf2_ros.StaticTransformBroadcaster() if static else tf2_ros.TransformBroadcaster()
        optical_broadcaster = None if static else tf2_ros.StaticTransformBroadcaster()
        publish_chain(transforms, broadcaster, optical_broadcaster)
        application.initial_published()
    except Exception as error:
        rospy.logfatal("Could not start frozen camera extrinsic publisher: %s", error)
        return 2

    def apply(resolved):
        nonlocal transforms
        candidate = frozen_transform_chain(resolved, parent, link, optical, link_offset)
        publish_chain(candidate, broadcaster, optical_broadcaster)
        transforms = candidate

    try:
        while not rospy.is_shutdown():
            try:
                application.tick(apply)
                rospy.set_param("~extrinsic_update_error", "")
            except Exception as error:
                rospy.set_param("~extrinsic_update_error", str(error))
                rospy.logwarn_throttle(5.0, "Camera application is not confirmed: %s", error)
            if not static:
                publish_chain(transforms, broadcaster, optical_broadcaster)
            time.sleep(1.0 / rate)
    finally:
        application.ready = False
        application.project(application.state())
    return 0


if __name__ == "__main__":
    sys.exit(main())
