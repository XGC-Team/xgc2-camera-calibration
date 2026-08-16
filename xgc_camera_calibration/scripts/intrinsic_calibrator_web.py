#!/usr/bin/env python3
"""ROS1 pose-control adapter and Media Edge entrypoint for intrinsics.

Live imagery belongs to the browser WebRTC session. This process asks the
target-local Media Edge for one RGB snapshot only when a manual capture or an
automatic pose sweep needs it; it never subscribes to a ROS camera image topic.
Gazebo pose control remains a small ROS-only branch.
"""

import sys
import threading
from pathlib import Path

import rospkg
import rospy

from xgc_camera_calibration.intrinsic_service import IntrinsicCalibrationService
from xgc_camera_calibration.media_snapshot import MediaSnapshotClient
from xgc_camera_calibration.web_service import CalibrationHttpServer


def split_list_parameter(value):
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    return [item.strip() for item in str(value).split(",") if item.strip()]


def maybe_camera_control(board_center):
    """Attach the optional Gazebo camera adapter, or run camera-agnostic.

    Only attaches when ``~camera_control`` is requested and the model actually
    appears on /gazebo/model_states within the timeout, so a real-camera run and
    a simulation without the model both fall back cleanly to guidance-only.
    """
    if not bool(rospy.get_param("~camera_control", False)):
        return None
    model_name = str(rospy.get_param("~camera_model_name", "gazebo_static_camera"))
    timeout = float(rospy.get_param("~camera_control_timeout", 8.0))
    try:
        from xgc_camera_calibration.camera_control import GazeboCameraControl

        control = GazeboCameraControl(model_name, board_center)
    except Exception as error:
        rospy.logwarn("Sim camera control unavailable (%s); running camera-agnostic", error)
        return None
    deadline = rospy.Time.now() + rospy.Duration(timeout)
    poll = rospy.Rate(10)
    while not rospy.is_shutdown() and rospy.Time.now() < deadline:
        if control.available():
            rospy.loginfo("Sim camera control attached for model '%s'", model_name)
            return control
        poll.sleep()
    rospy.logwarn(
        "Gazebo model '%s' not seen in %.1fs; running camera-agnostic", model_name, timeout
    )
    return None


def main():
    rospy.init_node("xgc_camera_intrinsic_calibrator_web")
    try:
        snapshot_client = MediaSnapshotClient(
            rospy.get_param("~media_edge_address", "http://127.0.0.1:18090"),
            rospy.get_param("~media_source_id", "usb_cam"),
            float(rospy.get_param("~snapshot_timeout", 5.0)),
        )
        snapshot_client.health()
        package_root = Path(rospkg.RosPack().get_path("xgc_camera_calibration"))
        web_root = Path(rospy.get_param("~web_root", str(package_root / "web" / "intrinsic")))
        calibrations = Path.home() / ".local/state/xgc2/camera/calibrations/usb_cam"
        board_center = (
            float(rospy.get_param("~board_x", 2.0)),
            float(rospy.get_param("~board_y", 0.0)),
            float(rospy.get_param("~board_z", 1.5)),
        )
        # Simulation uses the shared checkerboard_8x6 model (7x5 interior
        # corners). Physical uses the station AprilGrid printed on the plate:
        # 6x6 tag36h11, 88 mm tags, 26.4 mm gaps, ids 0..35.
        service = IntrinsicCalibrationService(
            board_size=(
                int(rospy.get_param("~board_cols", 7)),
                int(rospy.get_param("~board_rows", 5)),
            ),
            square=float(rospy.get_param("~square_size", 0.20)),
            output_file=rospy.get_param("~output_file", str(calibrations / "intrinsics.yaml")),
            image_topic="media:{}".format(snapshot_client.source_id),
            camera_info_topic="snapshot metadata",
            media_source=snapshot_client.source_id,
            jpeg_quality=int(rospy.get_param("~jpeg_quality", 80)),
            display_width=int(rospy.get_param("~display_width", 720)),
            board_center=board_center,
            references_dir=str(
                rospy.get_param("~references_dir", str(calibrations / "intrinsic_refs"))
            ),
            board_type=str(rospy.get_param("~board_type", "checkerboard")),
            tag_spacing=float(rospy.get_param("~tag_spacing", 0.0)),
            tag_family=str(rospy.get_param("~tag_family", "tag36h11")),
            tag_start_id=int(rospy.get_param("~tag_start_id", 0)),
            min_tags=int(rospy.get_param("~min_tags", 6)),
        )
        camera = maybe_camera_control(board_center)
        if camera is not None:
            service.attach_camera_control(camera)
        service.attach_frame_capture(lambda: snapshot_client.capture().bgr)
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
        )
    except Exception as error:
        rospy.logfatal("Could not start intrinsic calibration WebUI: %s", error)
        return 1

    server_thread = threading.Thread(
        target=server.serve_forever, name="intrinsic-calibration-http", daemon=True
    )
    server_thread.start()
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
        server.shutdown()
        server.server_close()
        server_thread.join(timeout=5.0)
    return 0


if __name__ == "__main__":
    sys.exit(main())
