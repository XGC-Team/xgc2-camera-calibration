#!/usr/bin/env python3
"""Exec Gazebo camera roslaunch with an authored or explicit file pose."""

import argparse
import os
import sys

from xgc_camera_calibration.camera_initial_pose import (
    replace_roslaunch_pose_arguments,
    resolve_gazebo_camera_pose_from_file,
)


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pose-source",
        choices=("authored", "file"),
        required=True,
    )
    parser.add_argument("--calibration-root", required=True)
    parser.add_argument("--camera-name", required=True)
    parser.add_argument("--parent-frame", required=True)
    parser.add_argument("--optical-frame", default="")
    parser.add_argument("--selected-physical-optical-frame", required=True)
    parser.add_argument("--extrinsic-file", default="")
    parser.add_argument("--optical-offset-x", type=float, required=True)
    parser.add_argument("--optical-offset-y", type=float, required=True)
    parser.add_argument("--optical-offset-z", type=float, required=True)
    parser.add_argument("launch", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    launch = list(args.launch)
    if launch and launch[0] == "--":
        launch = launch[1:]
    if not launch or not os.path.isabs(launch[0]):
        parser.error("an absolute roslaunch executable is required after --")
    offset = (args.optical_offset_x, args.optical_offset_y, args.optical_offset_z)
    if args.pose_source == "file":
        pose, selected = resolve_gazebo_camera_pose_from_file(
            args.calibration_root,
            args.camera_name,
            args.parent_frame,
            (args.optical_frame, args.selected_physical_optical_frame),
            offset,
            args.extrinsic_file,
        )
        launch = replace_roslaunch_pose_arguments(launch, pose)
        print(
            "Applying explicit extrinsic {} as Gazebo camera initial pose".format(
                selected
            ),
            flush=True,
        )
    os.execv(launch[0], launch)
    return 0


if __name__ == "__main__":
    sys.exit(main())
