#!/usr/bin/env python3
"""ROS1 pose-control adapter and Media Edge entrypoint for intrinsics.

Live imagery belongs to the browser WebRTC session. Simulation requests an
automatic camera pose; a person moves the physical camera. Both origins feed
the same continuous, lower-rate Media Edge snapshot detection loop, which
updates the annotated result and coverage independently from WebRTC FPS.
Frame evidence stays in the private runtime directory until shutdown.
"""

import math
import sys
import tempfile
from pathlib import Path

import rospkg
import rospy
from xgc2_xrpc import Runtime

from xgc_camera_calibration.intrinsic_service import (
    IntrinsicCalibrationService,
)
from xgc_camera_calibration.board_profiles import (
    FIELD_6X6_88MM_30PCT,
    resolve_aprilgrid_profile,
)
from xgc_camera_calibration.media_snapshot import MediaSnapshotClient
from xgc_camera_calibration.web_service import CalibrationHttpServer
from xgc_camera_calibration.ros_lifecycle import NodeParameters, RosMasterBinding


def split_list_parameter(value):
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    return [item.strip() for item in str(value).split(",") if item.strip()]


def maybe_camera_control(board_center, board_profile, parameters, *, runtime, local_target):
    """Attach the optional Gazebo camera adapter, or run camera-agnostic.

    The selected board belongs to the running calibration, including when no
    guide is viewed. Its existing peer connects asynchronously and survives an
    absent or restarted world.
    """
    if not bool(parameters.get("~camera_control", False)):
        return None
    model_name = str(parameters.get("~camera_model_name", "gazebo_static_camera"))
    timeout = float(parameters.get("~camera_control_timeout", 8.0))
    if not math.isfinite(timeout) or timeout <= 0.0:
        raise ValueError("~camera_control_timeout must be finite and positive")
    endpoint = str(parameters.get("~simulation_endpoint"))
    from xgc_camera_calibration.camera_control import GazeboCameraControl
    camera = GazeboCameraControl(model_name, board_center, connection_timeout=timeout,
        endpoint=endpoint, runtime=runtime, local_target=local_target, board_profile=board_profile)
    camera.world.peer.activate()
    return camera


def main():
    parameters = NodeParameters("xgc_camera_intrinsic_calibrator_web")
    binding = RosMasterBinding("xgc_camera_intrinsic_calibrator_web")
    runtime = None
    snapshot_client = None
    camera = None
    service = None
    server = None
    work_directory = None
    try:
        target_id = str(parameters.get("~target_id"))
        socket_path = str(parameters.get("~socket_path"))
        runtime = Runtime(blocking_workers=4, max_calls=16, max_connections=24)
        snapshot_client = MediaSnapshotClient(
            parameters.get("~media_edge_rpc_socket"),
            parameters.get("~media_source_id", "usb_cam"),
            float(parameters.get("~snapshot_timeout", 5.0)),
            runtime=runtime,
            local_target=target_id,
        )
        package_root = Path(rospkg.RosPack().get_path("xgc_camera_calibration"))
        web_root = Path(parameters.get("~web_root", str(package_root / "web" / "intrinsic")))
        calibration_mode = str(parameters.get("~calibration_mode")).strip()
        camera_name = str(parameters.get("~camera_name")).strip()
        work_directory = tempfile.TemporaryDirectory(prefix="intrinsic-", dir=str(Path(socket_path).parent))
        board_center = (
            float(parameters.get("~board_x", 2.0)),
            float(parameters.get("~board_y", 0.0)),
            float(parameters.get("~board_z", 2.2)),
        )
        board_profile = resolve_aprilgrid_profile(
            parameters.get("~board_profile", FIELD_6X6_88MM_30PCT)
        )
        display_width = int(parameters.get("~display_width", 720))
        service = IntrinsicCalibrationService(
            board_size=(board_profile.columns, board_profile.rows),
            square=board_profile.tag_size_m,
            output_file=str(Path(work_directory.name) / "intrinsics.yaml"),
            camera_name=camera_name,
            calibration_mode=calibration_mode,
            board_profile_id=board_profile.profile_id,
            media_source=snapshot_client.source_id,
            jpeg_quality=int(parameters.get("~jpeg_quality", 80)),
            maximum_detect_width=int(parameters.get("~maximum_detect_width", display_width)),
            display_width=display_width,
            board_center=board_center,
            board_type="aprilgrid",
            tag_spacing=board_profile.tag_gap_m,
            tag_family=board_profile.tag_family,
            tag_start_id=board_profile.start_id,
            min_tags=board_profile.min_tags,
        )
        service.attach_metadata_application(snapshot_client)
        service.capture_peer = snapshot_client.peer
        camera = maybe_camera_control(board_center, board_profile, parameters,
                                      runtime=runtime, local_target=target_id)
        if camera is not None:
            service.attach_camera_control(camera)
        if service.board_type == "aprilgrid":
            detection_target_pixels = int(
                parameters.get("~detection_target_pixels", 640 * 480)
            )
            service.attach_frame_capture(
                lambda: snapshot_client.capture_detection(detection_target_pixels)
            )
        else:
            service.attach_frame_capture(snapshot_client.capture)
        service.live_capture_interval = (float(parameters.get("~auto_capture_interval", 0.2))
            if bool(parameters.get("~auto_capture", True)) else None)
        def peer_status():
            media = snapshot_client.status()
            return {"ros_master": binding.status(), "media": media,
                    "camera_info": {"received": media["camera_info_received"]},
                    "world": camera.status() if camera is not None else {"reason": "not_configured", "pose_available": False}}
        def activate_peers():
            if camera is not None:
                camera.world.peer.activate()
        bind_address = str(parameters.get("~bind_address", "127.0.0.1"))
        http_port = int(parameters.get("~http_port", 8766))
        if not 1 <= http_port <= 65535:
            raise ValueError("~http_port must be between 1 and 65535")
        server = CalibrationHttpServer(
            (bind_address, http_port),
            None,
            web_root,
            frame_ancestors=str(
                parameters.get(
                    "~frame_ancestors", "'self' http://127.0.0.1:* http://localhost:*"
                )
            ),
            allowed_origins=split_list_parameter(parameters.get("~allowed_origins", [])),
            logger=lambda message: rospy.logdebug("Intrinsic web: %s", message),
            intrinsic_service=service,
            runtime=runtime,
            target_id=target_id,
            rpc_socket=socket_path,
            lifecycle=binding, peer_status=peer_status, activate_peers=activate_peers,
        )
        server.start()
        if not binding.initialize():
            return 0
        binding.mark_ready()
        rospy.loginfo("Intrinsic calibration WebUI on http://%s:%d (media=%s)",
                      bind_address, http_port, snapshot_client.source_id)
        rospy.spin()
        return 0
    except Exception as error:
        rospy.logfatal("Could not start intrinsic calibration WebUI: %s", error)
        return 1
    finally:
        binding.stop_readiness()
        if service is not None:
            service.stop_auto_capture()
        if server is not None:
            server.close()
        if camera is not None:
            camera.close()
        if snapshot_client is not None:
            snapshot_client.close()
        binding.close()
        if runtime is not None:
            runtime.close()
        if work_directory is not None:
            work_directory.cleanup()


if __name__ == "__main__":
    sys.exit(main())
