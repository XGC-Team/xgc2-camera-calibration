# XGC2 ROS1 Camera Calibration

Public ROS Noetic camera-calibration tools. Camera capture and ROS driver
adaptation deliberately live in separate products. Intrinsic calibration asks
a co-located XGC Media Edge for bounded immutable snapshots; fixed-camera
extrinsic calibration retains its ROS image, camera-info, and pose contracts.

## Calibration capabilities

### General intrinsic calibration

The intrinsic calibrator works with fixed, onboard, and vehicle-mounted
cameras. It auto-collects geometrically diverse chessboard views, tracks
X/Y/size/skew coverage, solves directly with OpenCV, and atomically writes a
standard camera calibration YAML. It does not assume that the camera is fixed
in a world frame.

```bash
roslaunch xgc_camera_calibration intrinsic_calibrator.launch \
  media_edge_address:=http://127.0.0.1:18090 \
  media_source_id:=usb_cam snapshot_timeout:=5.0 \
  board_cols:=7 board_rows:=5 square_size:=0.20 \
  bind_address:=127.0.0.1 http_port:=8766
```

Open `http://127.0.0.1:8766/`. The optional `camera_control:=true` adapter can
move a named Gazebo camera through the sample guide, but simulation control is
not required by the intrinsic algorithm.

The Media Edge must run on the same host and expose the configured source
before the calibrator starts. Each manual or automatic sample creates one
immutable snapshot, reads its RGB8 pixels and intrinsic metadata from that same
capture, then deletes it immediately. Live video remains WebRTC and is never
polled through this calibration HTTP path.

The product-facing intrinsic service is available under
`/api/v1/intrinsic/`: `state`, `image.jpg`, `targets`, and `ref/<index>.jpg`
are read endpoints; `goto`, `reset_pose`, `auto_run`, `calibrate`, and `reset`
are JSON actions. `auto_run` returns HTTP 202 immediately and exposes its
authoritative progress in `state.action`. Conflicting mutation requests return
HTTP 409 until the sweep finishes; the OpenCV solver itself remains unchanged.

### Fixed-world-camera extrinsic calibration

The extrinsic calibrator is for a camera fixed in an experiment site's world
frame. It associates world-frame marker poses with pixels in a frozen image,
then solves and persists `parent_T_camera_optical` using robust PnP.

```bash
roslaunch xgc_camera_calibration extrinsic_calibrator.launch \
  image_topic:=/usb_cam/image_raw \
  preview_image_topic:=/usb_cam/image_raw/compressed \
  camera_info_topic:=/usb_cam/camera_info \
  pose_prefix:=/vrpn_client_node \
  bind_address:=127.0.0.1 http_port:=8765
```

Open `http://127.0.0.1:8765/`. The solver requires valid intrinsic values, but
those values are an input contract rather than a package dependency. They may
come from the intrinsic tool above, a vendor calibration, or an existing
calibration asset.

The live view consumes the canonical JPEG-compressed image transport and
returns those bytes directly to the browser. The raw image topic is subscribed
only while handling Freeze, so the synchronized frame used by PnP retains full
source quality without continuously moving or re-encoding raw frames in the
WebUI process. A camera must publish both configured topics; there is no raw
preview fallback.

Publish a solved fixed-camera transform with:

```bash
roslaunch xgc_camera_calibration extrinsic_tf.launch \
  extrinsic_file:=/var/lib/xgc2/camera/calibrations/usb_cam/extrinsics.yaml
```

An Automation can start the publisher before the operator solves the camera and
activate the new result without restarting any process:

```bash
roslaunch xgc_camera_calibration extrinsic_tf.launch \
  extrinsic_file:=/tmp/xgc2/camera/calibrations/usb_cam/extrinsics.yaml \
  wait_for_file:=true require_file_update:=true watch_file:=true
```

`require_file_update` ignores a stale file that existed when the node started;
the calibrator's atomic save then activates exactly the result from the current
run. `watch_file` also applies later re-solves while the workflow remains open.

The stable REP-103 chain is:

```text
map -> usb_cam_link -> usb_cam_optical_frame
```

## Independence and release boundary

Intrinsic calibration and fixed-camera extrinsic calibration are separate
logical workflows. Neither declares a build or release dependency on the
other. XGC2 Automation may run intrinsic calibration first when no usable
intrinsic asset exists, but an existing valid intrinsic asset lets the
extrinsic workflow start directly.

This repository releases `ros-noetic-xgc2-camera-calibration` independently
from both `libxgc2-camera-dev` and `ros-noetic-xgc-camera-driver`.

## Automation

XGC2's separately owned process catalog registers three independent
process-definition IDs for this product:

- `xgc2-camera-intrinsic-calibrator-ros1`
- `xgc2-camera-extrinsic-calibrator-ros1`
- `xgc2-camera-extrinsic-tf-ros1`

Both WebUIs bind to loopback by default and require no desktop session or
`DISPLAY`. The intrinsic process also talks only to a loopback Media Edge.
Managed definitions write runtime calibration assets outside the package share
directory under `/tmp/xgc2/camera/calibrations`. This product package does not
ship or own XGC2 process definitions.

## Build and test

CI tests the Python solvers and Web services, builds the standalone Debian
package for Focal `amd64` and `arm64`, installs it in a clean container, and
checks its ROS launch files, process definitions, Python imports, and local
HTTP endpoints without installing or launching a camera driver.
