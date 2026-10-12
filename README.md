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
on a condition until its own socket/state and ROS master binding are ready,
or the wait expires. CameraInfo, images, Media Edge and the world never gate
startup or readiness. The response contains `service`, `api_version`,
`instance_id`, `ready`, and `facts`. Marker selection and calibration quality
remain separate solve state.

All three entrypoints bind their own sockets before waiting for ROS master.
They wait without a startup deadline, report `ready:false` while the master
is absent, and exit when its PID/run ID changes after binding. Core supplies
private command parameters before the master exists; roslaunch uses its
already running master's parameter namespace.

`facts.peers` and web state report ROS master, media, CameraInfo and world
pose independently. Pages render immediately and consume one SSE connection
with full-jitter retries from 250 ms to 30 s. No reload is needed after a
peer returns. A viewer or explicit capture first activates the shared media
watcher; a guide viewer or camera command activates the world watcher.
Neither readiness nor status issues domain probes. Each Unix peer has one
inotify directory watch and at most one held unbound describe in flight.
Once describe is ready it returns immediately, so socket events and domain
transport failures drive later discovery rather than continuously issuing
ready describes. Early not-ready replies and connection failures use
exponential full jitter from 250 ms to 30 s. A missing socket waits for an
inotify event without connect attempts; ROS master has no lifecycle event
and uses bounded XMLRPC backoff, then one identity check every 10 s.

Capture buffers retain the latest frame. A restarted world receives the
selected board and latest desired camera pose once per instance. Pose
revisions are monotonic; the confirmed revision and instance appear in
status. World entity state is compared before replaying the idempotent pose.
Source-owned metadata mutations and snapshot transactions are never replayed.
Outage logs summarize suppressed attempts at most once every 10 s.

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
The native calibrator owns intrinsic quality assessment. `/v1/result` refuses a
candidate that is not ready to save; Core maps the accepted document into the
configuration store without interpreting quality or algorithm provenance.

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
consume `geometry_msgs/PoseStamped` under `posePrefix`, with tracker names
supplied through `trackers` (a comma-separated process parameter or a ROS
list). Subscribers stay registered across publisher outages and receive
late publishers through normal ROS updates. ROS1 has no wildcard topic
subscription; the helper does not poll the master to discover arbitrary
tracker names. Core must supply the Run's tracker names. Coordinate provenance
is explicit: `raw-vrpn` converts the frozen world offset once;
`experiment-world` already uses that coordinate system.

The extrinsic TF node publishes the immutable resolved pose supplied by
Core. It validates exact parent and optical frame roles and publishes both
segments of the parent/link/optical chain. It reports applied transform facts
on ROS. Core supplies a new resolved pose when it starts the camera fragment
with a new calibration; the TF node does not update a selection file.
Its private service is `xgc2.calibration.v1.ExtrinsicTf`, API version `1`,
with the same unbound held readiness contract. TF uses its supplied frozen
resolution and reports media/CameraInfo as unconsumed peers.

## Launch inputs

The installed process definitions are authoritative. All three helpers need
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
  pose_coordinate_source:=experiment-world trackers:=body_01,body_02
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

The build script fetches the released 0.2.0 wheel and verifies its pinned
checksum. Tests use that released SDK on real Unix sockets; the repository
does not vendor the SDK.

Checks run serially. Native endpoint tests own private ROS masters, runtime
directories and all child processes:

```bash
.xgc2/scripts/check_package_compliance.sh
PYTHONPATH="$XRPC_SOURCE/python:$PWD/xgc_camera_calibration/src:$PWD/xgc_camera_calibration/test" \
  OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 -m pytest -q xgc_camera_calibration/test
npm --prefix web-src ci
npm --prefix web-src run check:ui-policy
npm --prefix web-src run build
PLAYWRIGHT_PACKAGE=/path/to/browser-toolchain/package.json \
  node web-src/scripts/check-peer-recovery.mjs
PLAYWRIGHT_PACKAGE=/path/to/browser-toolchain/package.json \
  python3 web-src/scripts/check-extrinsic-candidate.py
```

The two 60-second F11 tests are skipped by default in the fast cell. Run a
long cell with the same released SDK, OpenCV/NumPy and ROS environment:

```bash
XGC_CALIBRATION_LONG_TESTS=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  nice -n 19 python3 -m pytest -q -s \
  xgc_camera_calibration/test/test_native_process_endpoint.py -k sixty_seconds
```

The long cell measures process CPU (under 1.2 s per 60 s), connect attempts,
log lines, threads and FDs for ROS master and media/world outages. It then
asserts recovery and one revisioned world-state replay.

Package CI builds with `catkin_make -j1` in the locked XGC2 image. The
installed check uses an isolated ROS_HOME and a random master port in
20000–40000, starts real entrypoints, checks held unbound describe and empty
result reads, verifies TF publication, and stops and waits for its processes.
Real headless camera calibration and obstacle-scene modeling runs belong to
the later R5 quiet-window regression gate.
