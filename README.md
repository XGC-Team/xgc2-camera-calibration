# XGC2 ROS1 Camera Calibration

Public ROS Noetic camera-calibration tools. Camera capture and ROS driver
adaptation deliberately live in separate products. Both calibration modes can
ask a co-located XGC Media Edge for bounded immutable snapshots. Fixed-camera
extrinsic calibration also retains its ROS image and camera-info contract for
physical cameras while marker poses continue to arrive through ROS.

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
`/api/v1/intrinsic/`: `state`, `targets`, and `ref/<index>.jpg` are read
endpoints; `capture`, `goto`, `reset_pose`, `auto_run`, `calibrate`, and
`reset` are JSON actions. `capture` performs one explicit immutable snapshot;
`auto_run` returns HTTP 202 immediately and exposes its authoritative progress
in `state.action`. Conflicting mutation requests return HTTP 409 until the
sweep finishes; the OpenCV solver itself remains unchanged. `image.jpg` remains
a compatibility view of the most recently processed sample, not a live-video
transport.

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

For a managed Gazebo world camera, use the snapshot mode instead:

```bash
roslaunch xgc_camera_calibration extrinsic_calibrator.launch \
  media_edge_address:=http://127.0.0.1:18090 \
  media_source_id:=gazebo_world_camera \
  pose_prefix:=/vrpn_client_node \
  bind_address:=127.0.0.1 http_port:=8765
```

The panel's live view remains direct WebRTC. Freeze captures one immutable
Media Edge frame whose Gazebo timestamp, RGB pixels, frame ID, camera matrix,
and distortion values come from the same render pass; marker histories are
then interpolated at that timestamp before robust PnP. No ROS image publisher
or calibration JPEG polling is required in this mode.

When `media_edge_address` is empty, the compatibility path consumes the
canonical JPEG-compressed ROS preview and subscribes to the raw image only
while handling Freeze. A physical camera using that path must publish the raw,
compressed, and CameraInfo topics.

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
from both `libxgc2-camera-dev` and `ros-noetic-xgc2-camera-driver`.

## Automation

XGC2's separately owned process catalog registers three independent
process-definition IDs for this product:

- `xgc2-camera-intrinsic-calibrator-ros1`
- `xgc2-camera-extrinsic-calibrator-ros1`
- `xgc2-camera-extrinsic-tf-ros1`

Both WebUIs bind to loopback by default and require no desktop session or
`DISPLAY`. Media Edge snapshot mode accepts only a loopback Edge address.
Managed definitions write runtime calibration assets outside the package share
directory under `/tmp/xgc2/camera/calibrations`. This product package does not
ship or own XGC2 process definitions.

## Build and test

CI tests the Python solvers and Web services, builds the standalone Debian
package for Focal `amd64` and `arm64`, installs it in a clean container, and
checks its ROS launch files, process definitions, Python imports, and local
HTTP endpoints without installing or launching a camera driver.

The intrinsic and extrinsic pages are two deterministic builds of one React
entry. Both consume the immutable `@xgc2/ui-react` `0.15.8` release for their
shell, single-title topbar, themes, panels, controls, feedback, progress,
tables, structured details, code results, responsive layout, and scrollbars.
The imperative camera/ROS transport and canvas interaction stay in small
page-specific modules behind that shared view.

```bash
npm --prefix web-src ci
npm --prefix web-src run build
```

Generated `app.js` and `styles.css` files remain beside each packaged HTML
entry. CI and release jobs rebuild them and reject source/generated drift.
