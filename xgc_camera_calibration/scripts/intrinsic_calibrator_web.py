#!/usr/bin/env python3
"""ROS1 pose-control adapter and Media Edge entrypoint for intrinsics.

Live imagery belongs to the browser WebRTC session. Simulation requests an
automatic camera pose; a person moves the physical camera. Both origins feed
the same continuous, lower-rate Media Edge snapshot detection loop, which
updates the annotated result and coverage independently from WebRTC FPS.
Frames themselves are never persisted.
"""

import math
import sys
from pathlib import Path

import rospkg
import rospy
from xgc2_xrpc import Runtime

from xgc_camera_calibration.intrinsic_service import (
    IntrinsicCalibrationService,
    intrinsic_calibration_directory,
)
from xgc_camera_calibration.board_profiles import (
    FIELD_6X6_88MM_30PCT,
    resolve_aprilgrid_profile,
)
from xgc_camera_calibration.media_snapshot import MediaSnapshotClient
from xgc_camera_calibration.web_service import CalibrationHttpServer


def split_list_parameter(value):
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    return [item.strip() for item in str(value).split(",") if item.strip()]


def maybe_camera_control(board_center, *, runtime):
    """Attach the optional Gazebo camera adapter, or run camera-agnostic.

    The explicit simulation endpoint supplies one bounded entity query. An
    unavailable optional camera leaves the sample guide camera-agnostic.
    """
    if not bool(rospy.get_param("~camera_control", False)):
        return None
    model_name = str(rospy.get_param("~camera_model_name", "gazebo_static_camera"))
    timeout = float(rospy.get_param("~camera_control_timeout", 8.0))
    if not math.isfinite(timeout) or timeout <= 0.0:
        raise ValueError("~camera_control_timeout must be finite and positive")
    endpoint = str(rospy.get_param("~simulation_endpoint"))
    try:
        from xgc_camera_calibration.camera_control import GazeboCameraControl

        control = GazeboCameraControl(
            model_name, board_center, connection_timeout=timeout,
            endpoint=endpoint, runtime=runtime,
        )
    except Exception as error:
        rospy.logwarn("Sim camera control unavailable (%s); running camera-agnostic", error)
        return None
    rospy.loginfo("Sim camera control attached for model '%s'", model_name)
    return control


def main():
    rospy.init_node("xgc_camera_intrinsic_calibrator_web")
    runtime = Runtime(blocking_workers=4, max_calls=16, max_connections=24)
    snapshot_client = None
    metadata_client = None
    camera = None
    service = None
    server = None
    try:
        snapshot_client = MediaSnapshotClient(
            rospy.get_param("~media_edge_rpc_socket"),
            rospy.get_param("~media_source_id", "usb_cam"),
            float(rospy.get_param("~snapshot_timeout", 5.0)),
            runtime=runtime,
            local_target=str(rospy.get_param("~target_id")),
        )
        snapshot_client.health()
        package_root = Path(rospkg.RosPack().get_path("xgc_camera_calibration"))
        web_root = Path(rospy.get_param("~web_root", str(package_root / "web" / "intrinsic")))
        calibration_root = Path(str(rospy.get_param("~calibration_root")))
        if not calibration_root.is_absolute():
            raise ValueError("~calibration_root must be an explicitly granted absolute path")
        calibration_mode = str(rospy.get_param("~calibration_mode")).strip()
        camera_name = str(rospy.get_param("~camera_name")).strip()
        calibrations = intrinsic_calibration_directory(
            str(calibration_root), calibration_mode, camera_name
        )
        board_center = (
            float(rospy.get_param("~board_x", 2.0)),
            float(rospy.get_param("~board_y", 0.0)),
            float(rospy.get_param("~board_z", 2.2)),
        )
        board_profile = resolve_aprilgrid_profile(
            rospy.get_param("~board_profile", FIELD_6X6_88MM_30PCT)
        )
        if bool(rospy.get_param("~camera_control", False)):
            from xgc_camera_calibration.camera_control import select_gazebo_board_profile

            select_gazebo_board_profile(
                board_profile,
                board_center,
                float(rospy.get_param("~camera_control_timeout", 8.0)),
                endpoint=str(rospy.get_param("~simulation_endpoint")),
                runtime=runtime,
            )
        display_width = int(rospy.get_param("~display_width", 720))
        service = IntrinsicCalibrationService(
            board_size=(board_profile.columns, board_profile.rows),
            square=board_profile.tag_size_m,
            output_file=str(calibrations / "intrinsics.yaml"),
            camera_name=camera_name,
            calibration_mode=calibration_mode,
            board_profile_id=board_profile.profile_id,
            media_source=snapshot_client.source_id,
            jpeg_quality=int(rospy.get_param("~jpeg_quality", 80)),
            maximum_detect_width=int(rospy.get_param("~maximum_detect_width", display_width)),
            display_width=display_width,
            board_center=board_center,
            board_type="aprilgrid",
            tag_spacing=board_profile.tag_gap_m,
            tag_family=board_profile.tag_family,
            tag_start_id=board_profile.start_id,
            min_tags=board_profile.min_tags,
        )
        try:
            metadata_client = snapshot_client.camera_metadata_client()
            service.attach_metadata_application(metadata_client)
        except Exception as error:
            rospy.logwarn("Source CameraInfo metadata application unavailable: %s", error)
        camera = maybe_camera_control(board_center, runtime=runtime)
        if camera is not None:
            service.attach_camera_control(camera)
        if service.board_type == "aprilgrid":
            detection_target_pixels = int(
                rospy.get_param("~detection_target_pixels", 640 * 480)
            )
            service.attach_frame_capture(
                lambda: snapshot_client.capture_detection(detection_target_pixels)
            )
        else:
            service.attach_frame_capture(snapshot_client.capture)
        automatic_detection = bool(rospy.get_param("~auto_capture", True))
        if automatic_detection:
            # Five fresh snapshots per second are sufficient for human motion
            # and the simulation sweep. An unbounded loop can occupy an entire
            # CPU core decoding/detecting 4K JPEGs and compete with the 4K30
            # H264 adapter without improving geometric sample diversity.
            service.start_auto_capture(
                float(rospy.get_param("~auto_capture_interval", 0.2))
            )
        bind_address = str(rospy.get_param("~bind_address", "127.0.0.1"))
        http_port = int(rospy.get_param("~http_port", 8766))
        if not 1 <= http_port <= 65535:
            raise ValueError("~http_port must be between 1 and 65535")
        server = CalibrationHttpServer(
            (bind_address, http_port),
            None,
            web_root,
            frame_ancestors=str(
                rospy.get_param(
                    "~frame_ancestors", "'self' http://127.0.0.1:* http://localhost:*"
                )
            ),
            allowed_origins=split_list_parameter(rospy.get_param("~allowed_origins", [])),
            logger=lambda message: rospy.logdebug("Intrinsic web: %s", message),
            intrinsic_service=service,
            runtime=runtime,
            rpc_socket=rospy.get_param("~rpc_socket"),
            target_id=rospy.get_param("~target_id"),
        )
        server.start()
    except Exception as error:
        rospy.logfatal("Could not start intrinsic calibration WebUI: %s", error)
        if service is not None:
            service.stop_auto_capture()
        if server is not None:
            server.close()
        if camera is not None:
            camera.close()
        if snapshot_client is not None:
            snapshot_client.close()
        if metadata_client is not None:
            metadata_client.close()
        runtime.close()
        return 1

    rospy.loginfo(
        "Intrinsic calibration WebUI on http://%s:%d (media=%s, camera_control=%s)",
        bind_address,
        http_port,
        snapshot_client.source_id,
        camera is not None,
    )
    try:
        rospy.spin()
    finally:
        service.stop_auto_capture()
        server.close()
        if camera is not None:
            camera.close()
        snapshot_client.close()
        if metadata_client is not None:
            metadata_client.close()
        runtime.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
