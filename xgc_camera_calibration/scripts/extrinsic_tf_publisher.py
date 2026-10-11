#!/usr/bin/env python3
"""Publish the immutable camera pose supplied by Core for this Run."""
import sys
import math
import uuid

import rospy
import tf2_ros
from geometry_msgs.msg import TransformStamped

from xgc_camera_calibration.extrinsic_application import (
    application_arguments, parse_frame_roles,
)
from xgc_camera_calibration.extrinsic_resolver import decode_frozen
from xgc_camera_calibration.transforms import split_parent_to_optical_pose
from xgc_camera_calibration.record_facts import AppliedTransformFacts, transform_values


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
    return stamp


def main():
    args = application_arguments(rospy.myargv()[1:])
    rospy.init_node("xgc_camera_extrinsic_tf")
    try:
        mode = str(rospy.get_param("~calibration_mode")).strip()
        camera_name = str(rospy.get_param("~camera_name")).strip()
        roles = parse_frame_roles(args.frame_roles_json)
        parent = str(rospy.get_param("~parent_frame"))
        optical = str(rospy.get_param("~optical_frame"))
        link = str(rospy.get_param("~camera_link_frame"))
        if mode not in ("sim", "phy") or parent != roles["parentFrame"] or optical != roles["opticalFrames"][mode]:
            raise ValueError("output frames do not match the controlled camera roles")
        link_offset = tuple(float(rospy.get_param("~link_to_optical_" + axis, 0.0)) for axis in ("x", "y", "z"))
        if not all(math.isfinite(value) for value in link_offset):
            raise ValueError("camera link offset must be finite")
        frozen = decode_frozen(args.resolved_extrinsic_json, camera_name, roles)
        instance_id = uuid.uuid4().hex
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
        rate = float(rospy.get_param("~publish_rate", 10.0))
        if not math.isfinite(rate) or rate <= 0:
            raise ValueError("camera publication rate must be finite and positive")
        broadcaster = tf2_ros.StaticTransformBroadcaster() if static else tf2_ros.TransformBroadcaster()
        optical_broadcaster = None if static else tf2_ros.StaticTransformBroadcaster()
        # Publication evidence records this process and the supplied resolution.
        fact_topic = [None]
        def emit_fact(text):
            from std_msgs.msg import String
            if fact_topic[0] is None:
                fact_topic[0] = rospy.Publisher(AppliedTransformFacts.TOPIC, String, queue_size=64, latch=True)
            fact_topic[0].publish(String(data=text))
        facts = AppliedTransformFacts({"id": camera_name + ":" + parent + ":" + optical,
            "instanceId": instance_id, "kind": "camera-extrinsic",
            "resolutionId": frozen["resolutionId"]},
            emit_fact,
            lambda message: rospy.logwarn_throttle(5.0, message))
        stamp = publish_chain(transforms, broadcaster, optical_broadcaster)
        facts.applied(transform_values(transforms), frozen, stamp.to_nsec())
        rospy.set_param("~extrinsic_ready", True)
    except Exception as error:
        rospy.logfatal("Could not start frozen camera extrinsic publisher: %s", error)
        return 2

    try:
        while not rospy.is_shutdown():
            if not static:
                stamp = publish_chain(transforms, broadcaster, optical_broadcaster)
                facts.applied(transform_values(transforms), frozen, stamp.to_nsec())
            facts.flush()
            if static:
                rospy.spin()
                break
            rospy.rostime.wallsleep(1.0 / rate)
    finally:
        facts.stop(rospy.Time.now().to_nsec())
        rospy.set_param("~extrinsic_ready", False)
    return 0


if __name__ == "__main__":
    sys.exit(main())
