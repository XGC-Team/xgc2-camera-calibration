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
  calibration_root:=/home/user/Documents/XGC/Calibration/camera \
  calibration_mode:=phy camera_name:=usb_cam \
  board_profile:=field_6x6_88mm_30pct \
  bind_address:=127.0.0.1 http_port:=8766
```

Open `http://127.0.0.1:8766/`. The optional `camera_control:=true` adapter can
move a named Gazebo camera through the sample guide, but simulation control is
not required by the intrinsic algorithm.

`board_profile` accepts exactly two atomic profiles:

- `field_6x6_88mm_30pct`;
- `a4_6x6_24mm_30pct_kalibr_v1`.

Both are 6×6 tag36h11 grids with ID 0 at the lower-left, IDs increasing along
+X then +Y, and every tag rendered with Kalibr `rotation=2` relative to
OpenCV's raw marker image. This datum matches the established Gazebo textures;
it does not select the Kalibr solver or require rosbag. Detection and solving
remain on the same XGC OpenCV path for both profiles. The retired pre-datum A4
identifier is rejected rather than translated or restored.

Gazebo gives each selected profile a distinct runtime model instance name.
Switching profiles removes only the other known calibration-board instances;
reselecting the current profile is idempotent. The world-file legacy instance
is never deleted and respawned under the same name, avoiding asynchronous
delete events that could remove a freshly selected board after startup.

The Media Edge must run on the same host and expose the configured source
before the calibrator starts. Each manual or automatic sample creates one
immutable snapshot, reads its RGB8 pixels and intrinsic metadata from that same
capture, then deletes it immediately. Live video remains WebRTC and is never
polled through this calibration HTTP path.

The product-facing intrinsic service is available under
`/api/v1/intrinsic/`: `state`, `targets`, `ref/<index>.jpg`, and
`evidence.zip` are read
endpoints; `capture`, `goto`, `reset_pose`, `auto_run`, `candidate`, `save`,
`continue`, and `reset` are JSON actions. `capture` performs one explicit
immutable snapshot;
`auto_run` returns HTTP 202 immediately and exposes its authoritative progress
in `state.action`. Conflicting mutation requests return HTTP 409 until the
sweep finishes. `candidate` freezes the strict observation pool and performs
the robust batch solve plus leave-one-view-out validation without writing a
file. `save` accepts only that candidate's exact ID and is the sole operation
that creates a timestamped YAML; `continue` discards the candidate while
retaining its observations. There is no `calibrate` alias. `image.jpg` is the
most recently annotated detector snapshot, not a live-video transport.

Every solver-admitted sample retains its exact source JPEG and annotation under
`<calibration directory>/captures/<capture identity>/`, together with a
correspondence checkpoint and checksum manifest. Reset and process exit retain
these captures; deletion is an explicit user file-management operation.
Evidence ZIP download is available during collection, before solving or saving.

`POST /api/v1/intrinsic/candidate` accepts a frozen observation revision and
returns HTTP 202 with a job receipt. Repeated submissions during that solve
return the same job. State includes `solve_job` (stage and progress); computation
runs outside the state lock, so live detection and short status requests remain
responsive. No new sample enters the frozen pool while solving. Reconnecting
clients observe the job; only the exact resulting candidate ID can be saved.
An interrupted process retains the original capture and exposes a failed job on
checkpoint recovery; it does not silently resume computation or save an asset.

The optimizer uses OpenCV QR for its linear solves while retaining free K/D,
the same convergence criteria, iterative robust view exclusion, projected-rank
checks and complete leave-one-view-out validation. A 30-minute cooperative job
deadline is checked between fits; it does not forcibly interrupt a native OpenCV
call. A failed solve retains evidence and never creates a successful YAML.

Replay a preserved physical capture through real HTTP and an explicit Save:

```bash
python3 tools/verify_intrinsic_capture.py --capture /path/to/capture --output /new/analysis/directory
```

The output must be new; the source capture is verified and remains unchanged.
This regression does not activate its resulting YAML in an experiment.

### Fixed-world-camera extrinsic calibration

The extrinsic calibrator is for a camera fixed in an experiment site's world
frame. It associates world-frame marker poses with pixels in a frozen image,
then solves and persists `parent_T_camera_optical` using robust PnP.

```bash
roslaunch xgc_camera_calibration extrinsic_calibrator.launch \
  image_topic:=/usb_cam/image_raw \
  preview_image_topic:=/usb_cam/image_raw/compressed \
  intrinsic_file:=/home/user/Documents/XGC/Calibration/camera/phy/usb_cam/intrinsics-20260830T010203.000000Z.yaml \
  pose_prefix:=/vrpn_client_node \
  calibration_root:=/home/user/Documents/XGC/Calibration/camera \
  calibration_mode:=phy camera_name:=usb_cam \
  bind_address:=127.0.0.1 http_port:=8765
```

Open `http://127.0.0.1:8765/`. The solver requires valid intrinsic values, but
those values are an input contract rather than a package dependency. They may
come from the intrinsic tool above, a vendor calibration, or an existing
calibration asset.

The extrinsic HTTP contract separates computation from persistence. `freeze`
captures one immutable image plus the latest static marker map; `solve`
returns an in-memory candidate with a content-bound `candidate_id` and writes
nothing; `save` accepts only that exact candidate ID and is the sole operation
that creates a timestamped YAML. A stale candidate is rejected and repeating
Save for the same candidate is idempotent. There is no solve-and-save alias.

For a managed Gazebo world camera, use the snapshot mode instead:

```bash
roslaunch xgc_camera_calibration extrinsic_calibrator.launch \
  media_edge_address:=http://127.0.0.1:18090 \
  media_source_id:=gazebo_world_camera \
  intrinsic_file:=/home/user/Documents/XGC/Calibration/camera/sim/usb_cam/intrinsics-20260830T010203.000000Z.yaml \
  pose_prefix:=/vrpn_client_node \
  calibration_root:=/home/user/Documents/XGC/Calibration/camera \
  calibration_mode:=sim camera_name:=usb_cam \
  bind_address:=127.0.0.1 http_port:=8765
```

The panel's live view remains direct WebRTC. Freeze captures one immutable
Media Edge frame, then copies the latest pose of every discovered marker once
before robust PnP. This contract deliberately has no pose history, timestamp
interpolation, or marker-age gate: it applies only to the current extrinsic
experiment, where the calibration camera and all markers remain static during
capture. No ROS image publisher or calibration JPEG polling is required in
this mode.

When `media_edge_address` is empty, the compatibility path consumes the
canonical JPEG-compressed ROS preview and subscribes to the raw image only
while handling Freeze. A physical camera using that path must publish the raw,
compressed, and CameraInfo topics.

Publish a solved fixed-camera transform with:

```bash
roslaunch xgc_camera_calibration extrinsic_tf.launch \
  calibration_root:=/home/user/Documents/XGC/Calibration/camera \
  calibration_mode:=phy camera_name:=usb_cam
```

An Automation can start the publisher before the operator solves the camera and
activate the new result without restarting any process:

```bash
roslaunch xgc_camera_calibration extrinsic_tf.launch \
  calibration_root:=/home/user/Documents/XGC/Calibration/camera \
  calibration_mode:=phy camera_name:=usb_cam \
  wait_for_file:=true require_file_update:=true watch_file:=true
```

The calibrator writes exactly one immutable
`<root>/<sim|phy>/<cameraName>/extrinsics-<UTC>.yaml` per explicit Save. It does not
create or update an `extrinsics.yaml` alias. `require_file_update` ignores every
timestamped result that existed when the node started; the directory watcher
then activates the concrete result from the current run. `watch_file` also
applies later re-solves while the workflow remains open, and the publisher
exposes the active absolute path as its private `active_extrinsic_file` ROS
parameter. When `wait_for_file` is false and the partition has no timestamped
YAML, the node publishes a default identity parent→link→optical chain immediately
so 3D/AR still have a camera frame; `watch_file` later hot-loads the first saved
file. That default is not a calibration result.

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
Managed definitions pass an explicit Documents calibration root, `sim` or
`phy`, and the stable camera name. The media source ID is not a storage
identity. This product package does not ship or own XGC2 process definitions.

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

## Interactive calibration acceptance

Intrinsic Analyze starts a revision-bound background job and returns immediately.
Save commits the completed candidate; it does not start another solve. Selection
removes all observations outside the unchanged robust envelope in each refit.
Independent leave-one-out fits use bounded parallel workers (at most eight), with
canonical observation identities and result order. All retained observations still
receive a full free-parameter QR fit; final points and intrinsics stay in the source
image coordinate system. Parallel speed is machine-dependent, not a one-minute SLA.

Extrinsic correspondences use arbitrary discovered pose markers, independent of
Experiment participants or robot type. The selected pixel must represent the rigid
body coordinate origin. Freeze captures a static scene; keep bodies stationary.
Solve estimates a candidate and rejects outliers; Save alone persists a versioned
`<calibrationRoot>/<sim|phy>/<cameraName>/extrinsics-<UTC>.yaml` and advances the
exact shared selection. A running TF publisher with `watch_file=true` consumes that
selection, allowing repeated calibration after moving the camera without restarting
the Experiment. This file/service contract does not by itself certify downstream
browser rendering or capture/pose time synchronization.
