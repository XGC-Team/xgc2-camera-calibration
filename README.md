# XGC2 ROS1 Camera Calibration

This independent ROS Noetic product provides intrinsic and fixed-camera
extrinsic calibration. Core owns Storage and commits calibration results.
The calibrators compute candidates, serve them through their own private
http.v1 service, and show collection and solve progress in their Web pages.

Each calibrator requires `socketPath` and `targetId` from Core. Its published
service is `xgc2.calibration.v1.Calibration`, API version `1`, on the private
Unix socket. No credentials or persistence service reference enter a product
process. Appearance preferences in the standalone pages live in memory.

## Result and readiness contract

An unbound `GET /v1/describe?wait_ready_ms=N` accepts `N` from 0 through 30000.
The SDK matches discovery by method and path. The service holds the request
on a condition until camera inputs become ready or the wait expires; the
response contains `service`, `api_version`, `instance_id`, `ready`, and
`facts`. Intrinsic readiness requires a captured frame. Extrinsic readiness
requires an image and a usable model from CameraInfo or an explicit intrinsic
file. Marker selection and calibration quality remain separate solve state.

Facts include `camera_id`, `calibration_mode`, `result_kind`,
`result_path: "/v1/result"`, `commit_owner: "core"`, `ros_master_uri`, and
intrinsic/extrinsic capabilities. Core pins later requests to the discovered
instance. Describe remains unbound when it carries the readiness query.

`GET /v1/result` returns:

```json
{
  "kind": "intrinsic",
  "camera_id": "front_camera",
  "calibration_mode": "phy",
  "revision": "<SHA-256 of the current result>",
  "result": {
    "candidate_id": "<computed candidate identity>",
    "quality": {},
    "document": {}
  }
}
```

Before a solve, `result` is null. Repeated reads of an unchanged result have
an unchanged revision. A changed candidate or an invalidated result changes
it. `document` is the intrinsic or extrinsic calibration document that Core
needs to commit; Core supplies commit time and durable identity. Reads leave
all computed and collection state unchanged.

Intrinsic documents carry camera matrix, distortion, image dimensions,
sample counts, board identity, sample identities, source frame identities,
algorithm provenance, full residual and stability diagnostics, and the
candidate quality assessment. Extrinsic documents carry parent and optical
frame ids, translation, quaternion, sample/inlier counts, every residual,
point observations, intrinsic model identity, and coordinate provenance.
Core checks intrinsic quality and reviews the candidate before committing.

The product pages analyze candidates and apply reviewed intrinsic estimates
as ephemeral CameraInfo metadata when the source advertises that capability.
Operator Save belongs to the Core calibration panel or workflow. There is
no product Save route. Source metadata application does not persist a result
or source configuration and never replays an uncertain mutation.

## Native camera inputs

One `xgc.system.camera` fragment per camera publishes `camera:<id>` and
`media:<id>`. The workflow supplies topic names, frame roles, source identity,
private endpoints, and ROS master from these registry entries. Panels bind
by instance id and those service names. Calibration never opens a device or
starts another camera producer.

Intrinsic calibration requests immutable captures through the target-local
Media Edge endpoint. Live preview remains the browser's WebRTC session. The
AprilGrid detector uses reduced detection pixels and keeps source JPEGs for
candidate analysis and the evidence download. Optional simulation camera
control uses the explicitly supplied simulation endpoint; physical cameras
remain camera-agnostic.

Extrinsic calibration consumes ROS `image_raw/compressed` and `camera_info`,
or camera `capture` and an explicit intrinsic YAML input. CameraInfo supplies
the measured model when `intrinsicFile` is empty. Uncalibrated or mismatched
CameraInfo leaves the model unavailable; it does not invent a pinhole model.
Image and CameraInfo optical frames and dimensions must agree. Pose markers
consume `geometry_msgs/PoseStamped` under `posePrefix`. Coordinate provenance
is explicit: `raw-vrpn` converts the frozen world offset once;
`experiment-world` already uses that coordinate system.

The extrinsic TF node publishes the immutable resolved pose supplied by
Core. It validates exact parent and optical frame roles and publishes both
segments of the parent/link/optical chain. It reports applied transform facts
on ROS. Core supplies a new resolved pose when it starts the camera fragment
with a new calibration; the TF node does not update a selection file.

## Launch inputs

The installed process definitions are authoritative. The calibrators need
`targetId`, `socketPath`, `cameraName`, and `calibrationMode`; native wiring
supplies camera and media facts. Extrinsic consumers additionally need
`resolvedExtrinsicJson`, `frameRolesJson`, and exact frame parameters.

```bash
roslaunch xgc_camera_calibration intrinsic_calibrator.launch \
  target_id:="$TARGET_ID" socket_path:="$INTRINSIC_SOCKET" \
  camera_name:=front_camera calibration_mode:=phy \
  media_edge_rpc_socket:="$MEDIA_SOCKET" media_source_id:=front_camera

roslaunch xgc_camera_calibration extrinsic_calibrator.launch \
  target_id:="$TARGET_ID" socket_path:="$EXTRINSIC_SOCKET" \
  camera_name:=front_camera calibration_mode:=phy \
  image_topic:=/front_camera/image_raw/compressed \
  preview_image_topic:=/front_camera/image_raw/compressed \
  camera_info_topic:=/front_camera/camera_info \
  resolved_extrinsic_json:="$RESOLVED_EXTRINSIC_JSON" \
  frame_roles_json:="$FRAME_ROLES_JSON" \
  parent_frame:=world child_frame:=front_camera_optical \
  pose_coordinate_source:=experiment-world
```

Intrinsic evidence and observation checkpoints use a temporary directory
under the private socket's runtime directory and are removed at process
shutdown. They support progress, candidate analysis, and evidence downloads;
they do not commit calibration assets. Read-only YAML import, resolver, and
asset export utilities accept explicit data files independently of the
calibrators. Package installation does not write user calibration files.

## Python SDK and verification

Build and runtime images supply Python XRPC **0.2.0**, from released commit
`4b257c97887b115eafbf20f7739b7072c5dc468d`, and its pinned native dependencies.
`.xgc2/scripts/check_python_xrpc.py` verifies the installed version and
`Context.query` before package builds and installed checks. Version 0.1.0
cannot serve the required unbound readiness query. Product automation does
not install toolchain or third-party Python packages.

The currently published XRPC wheel is 0.1.0. The central SDK installer gate
only allows that checksum-pinned wheel for scene generation and Gazebo
scenes. CI image provisioning for 0.2.0 is a central prerequisite. This
repository does not vendor the SDK or guess an unpublished wheel checksum.

Light checks run serially without a ROS master:

```bash
.xgc2/scripts/check_package_compliance.sh
PYTHONPATH="$XRPC_SOURCE/python:$PWD/xgc_camera_calibration/src:$PWD/xgc_camera_calibration/test" \
  OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 -m pytest -q xgc_camera_calibration/test
npm --prefix web-src ci
npm --prefix web-src run check:ui-policy
npm --prefix web-src run build
```

Package CI builds with `catkin_make -j1` in the locked XGC2 image. The
installed check uses an isolated ROS_HOME and a random master port in
20000–40000, starts real entrypoints, checks held unbound describe and empty
result reads, verifies TF publication, and stops and waits for its processes.
Real headless camera calibration and obstacle-scene modeling runs belong to
the later R5 quiet-window regression gate.
