#!/usr/bin/env python3

import sys
import time

import rospy
import tf2_ros
from geometry_msgs.msg import TransformStamped

from xgc_camera_calibration.extrinsic_file_watcher import ExtrinsicSelectionWatcher
from xgc_camera_calibration.solver import (
    extrinsic_calibration_directory,
    load_extrinsic,
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


def load_transform_chain(
    extrinsic_file,
    calibration_mode,
    camera_name,
    expected_parent_frame=None,
    expected_optical_frame=None,
    document=None,
):
    document = load_extrinsic(extrinsic_file) if document is None else dict(document)
    if document["calibration_mode"] != calibration_mode:
        raise ValueError("extrinsic calibration mode does not match the requested storage identity")
    if document["camera_name"] != camera_name:
        raise ValueError("extrinsic camera name does not match the requested storage identity")
    if expected_parent_frame and document.get("parent_frame") != expected_parent_frame:
        raise ValueError("extrinsic source parent frame does not match")
    if expected_optical_frame and document.get("child_frame") != expected_optical_frame:
        raise ValueError("extrinsic source optical frame does not match")
    parent_frame = rospy.get_param("~parent_frame", document.get("parent_frame", "map"))
    optical_frame = rospy.get_param(
        "~optical_frame", document.get("child_frame", "usb_cam_optical_frame")
    )
    camera_link_frame = rospy.get_param("~camera_link_frame", "usb_cam_link")
    offsets = tuple(
        float(rospy.get_param("~{}_offset".format(axis), 0.0)) for axis in ("x", "y", "z")
    )
    optical_translation = document["translation_array"] + offsets
    chain = split_parent_to_optical_pose(
        optical_translation,
        document["quaternion_xyzw_array"],
        tuple(
            float(rospy.get_param("~link_to_optical_{}".format(axis), 0.0))
            for axis in ("x", "y", "z")
        ),
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


def wait_for_transform_chain(
    extrinsic_directory,
    calibration_mode,
    camera_name,
    watcher,
    wait_for_file,
    poll_rate,
    expected_parent_frame=None,
    expected_optical_frame=None,
):
    announced_wait = False
    while not rospy.is_shutdown():
        result_revision = watcher.next_revision()
        if result_revision is None:
            if not wait_for_file:
                raise RuntimeError("calibration asset does not exist")
            if not announced_wait:
                rospy.loginfo(
                    "Waiting for a newly solved camera extrinsic under %s",
                    extrinsic_directory,
                )
                announced_wait = True
            poll_rate.sleep()
            continue
        extrinsic_file = result_revision.path
        try:
            return (
                load_transform_chain(
                    extrinsic_file, calibration_mode, camera_name,
                    expected_parent_frame, expected_optical_frame,
                    result_revision.document,
                ),
                extrinsic_file,
            )
        except Exception as error:
            if not wait_for_file:
                raise
            rospy.logwarn("Ignoring unreadable camera extrinsic %s: %s", extrinsic_file, error)
            poll_rate.sleep()
    return None


def log_transform_chain(extrinsic_file, transforms):
    parent_to_link, link_to_optical = transforms
    rospy.loginfo(
        "Publishing camera extrinsic chain %s -> %s -> %s from %s",
        parent_to_link.header.frame_id,
        parent_to_link.child_frame_id,
        link_to_optical.child_frame_id,
        extrinsic_file,
    )


def main():
    rospy.init_node("xgc_camera_extrinsic_tf")
    try:
        calibration_root = str(rospy.get_param("~calibration_root")).strip()
        calibration_mode = str(rospy.get_param("~calibration_mode")).strip()
        camera_name = str(rospy.get_param("~camera_name")).strip()
        selection_source = str(
            rospy.get_param("~selection_source", "authored")
        ).strip()
        if selection_source not in ("authored", "physical-selection"):
            raise ValueError("selection_source must be authored or physical-selection")
        selected_mode = "phy" if selection_source == "physical-selection" else calibration_mode
        extrinsic_directory = extrinsic_calibration_directory(
            calibration_root, selected_mode, camera_name
        )
    except Exception as error:
        rospy.logfatal("Invalid camera extrinsic storage identity: %s", error)
        return 2
    wait_for_file = bool(rospy.get_param("~wait_for_file", False))
    require_file_update = bool(rospy.get_param("~require_file_update", False))
    watch_file = bool(rospy.get_param("~watch_file", False))
    if selection_source == "physical-selection" and (require_file_update or watch_file):
        rospy.logfatal(
            "physical-selection is frozen for one Run and requires "
            "require_file_update=false, watch_file=false"
        )
        return 2
    expected_parent_frame = str(rospy.get_param(
        "~physical_source_parent_frame" if selection_source == "physical-selection"
        else "~runtime_source_parent_frame",
        "map",
    )).strip()
    expected_optical_frame = str(rospy.get_param(
        "~physical_source_optical_frame" if selection_source == "physical-selection"
        else "~runtime_source_optical_frame",
        "usb_cam_optical_frame",
    )).strip()
    if require_file_update and not wait_for_file:
        rospy.logfatal("~require_file_update requires ~wait_for_file=true")
        return 2
    file_poll_rate = float(rospy.get_param("~file_poll_rate", 5.0))
    if file_poll_rate <= 0.0:
        rospy.logfatal("~file_poll_rate must be positive")
        return 2
    watcher = ExtrinsicSelectionWatcher(
        calibration_root,
        selected_mode,
        camera_name,
        require_update=require_file_update,
    )
    poll_rate = rospy.Rate(file_poll_rate)
    try:
        loaded = wait_for_transform_chain(
            extrinsic_directory,
            selected_mode,
            camera_name,
            watcher,
            wait_for_file,
            poll_rate,
            expected_parent_frame,
            expected_optical_frame,
        )
    except Exception as error:
        rospy.logfatal("Could not load camera extrinsic under %s: %s", extrinsic_directory, error)
        return 1
    if loaded is None:
        return 0
    transforms, extrinsic_file = loaded

    static = bool(rospy.get_param("~static", True))
    log_transform_chain(extrinsic_file, transforms)
    rospy.set_param("~active_extrinsic_file", str(extrinsic_file))
    if static:
        broadcaster = tf2_ros.StaticTransformBroadcaster()
        while not rospy.is_shutdown():
            parent_to_link, link_to_optical = transforms
            stamp = rospy.Time.now()
            parent_to_link.header.stamp = stamp
            link_to_optical.header.stamp = stamp
            broadcaster.sendTransform([parent_to_link, link_to_optical])
            if not watch_file:
                rospy.spin()
                return 0
            loaded = wait_for_transform_chain(
                extrinsic_directory,
                selected_mode,
                camera_name,
                watcher,
                True,
                poll_rate,
                expected_parent_frame,
                expected_optical_frame,
            )
            if loaded is not None:
                transforms, extrinsic_file = loaded
                log_transform_chain(extrinsic_file, transforms)
                rospy.set_param("~active_extrinsic_file", str(extrinsic_file))
        return 0

    broadcaster = tf2_ros.TransformBroadcaster()
    optical_broadcaster = tf2_ros.StaticTransformBroadcaster()
    parent_to_link, link_to_optical = transforms
    link_to_optical.header.stamp = rospy.Time.now()
    optical_broadcaster.sendTransform(link_to_optical)
    rate = rospy.Rate(float(rospy.get_param("~publish_rate", 10.0)))
    next_file_poll = time.monotonic()
    while not rospy.is_shutdown():
        if watch_file and time.monotonic() >= next_file_poll:
            next_file_poll = time.monotonic() + (1.0 / file_poll_rate)
            result_revision = watcher.next_revision()
            if result_revision is not None:
                candidate_file = result_revision.path
                try:
                    transforms = load_transform_chain(
                        candidate_file, selected_mode, camera_name,
                        expected_parent_frame, expected_optical_frame,
                        result_revision.document,
                    )
                    extrinsic_file = candidate_file
                    parent_to_link, link_to_optical = transforms
                    link_to_optical.header.stamp = rospy.Time.now()
                    optical_broadcaster.sendTransform(link_to_optical)
                    log_transform_chain(extrinsic_file, transforms)
                    rospy.set_param("~active_extrinsic_file", str(extrinsic_file))
                except Exception as error:
                    rospy.logwarn(
                        "Ignoring unreadable camera extrinsic %s: %s", candidate_file, error
                    )
        parent_to_link.header.stamp = rospy.Time.now()
        broadcaster.sendTransform(parent_to_link)
        rate.sleep()
    return 0


if __name__ == "__main__":
    sys.exit(main())
