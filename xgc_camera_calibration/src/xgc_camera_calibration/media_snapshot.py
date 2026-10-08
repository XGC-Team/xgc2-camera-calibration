"""Explicit, target-local calibration snapshots from XGC Media Edge.

The live video path is WebRTC in the browser.  Calibration is intentionally a
different operation: an algorithm asks the co-located Media Edge for one
immutable snapshot, consumes it, then releases it. Continuous intrinsic
detection requests a fresh JPEG without RGB; operations that truly require raw
pixels can still request RGB8. This module never polls JPEG previews and never
creates a ROS image subscriber.
"""

from __future__ import annotations

import json
import math
import re
import time
import uuid
from email import policy
from email.parser import BytesParser
from dataclasses import dataclass
from typing import Any, Dict, Optional, Sequence, Tuple
from urllib.parse import quote
from xgc2_xrpc import Client, Endpoint, Fault, Limits, Runtime, ServiceRef, TransportError

import cv2
import numpy as np


_SOURCE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_MAX_JPEG_BYTES = 32 << 20
_MAX_RGB_BYTES = 128 << 20


class MediaSnapshotError(RuntimeError):
    """An expected Media Edge snapshot failure suitable for an API response."""


class CameraMetadataClient:
    """Apply estimates to CameraInfo through an explicitly granted source ref.

    This is source-owned ephemeral metadata, independent of immutable local
    calibration assets and of optical/render truth. No mutation is replayed.
    """

    def __init__(self, reference, source_id, *, runtime, local_target, timeout_seconds=5.0):
        reference.validate()
        if (reference.service != "camera-source" or reference.api_version != "v1"
                or reference.target_id != local_target
                or not _SOURCE_ID.fullmatch(str(source_id))):
            raise ValueError("an explicit local camera-source binding is required")
        if not math.isfinite(float(timeout_seconds)) or float(timeout_seconds) <= 0:
            raise ValueError("metadata timeout must be positive")
        self.source_id = str(source_id)
        self.timeout_seconds = float(timeout_seconds)
        self.path = "/v1/media/sources/" + quote(self.source_id, safe="")
        self.client = Client.from_service(reference, runtime=runtime, local_target=local_target,
            limits=Limits(connections=2, in_flight=2, body_bytes=64 << 10, response_bytes=128 << 10))
        self._status = None
        try:
            self._description = self.client.json(self.path + "/describe", method="GET", timeout=self.timeout_seconds)
            if (not isinstance(self._description, dict) or self._description.get("sourceId") != self.source_id
                    or not isinstance(self._description.get("capabilities"), list)):
                raise MediaSnapshotError("camera descriptor differs from its binding")
            self._status = self._read()
        except BaseException:
            self.client.close()
            raise

    def close(self):
        self.client.close()

    def _read(self):
        value = self.client.json(self.path + "/status", method="GET", timeout=self.timeout_seconds)
        if not isinstance(value, dict) or value.get("source_id") != self.source_id:
            raise MediaSnapshotError("camera source identity differs from its binding")
        return value

    def state(self):
        value = self._status or {}
        return {"available": "calibration-metadata" in self._description["capabilities"],
            "scope": "calibration-metadata", "source_id": self.source_id,
            "published_calibration_revision": value.get("published_calibration_revision")}

    def apply(self, calibration):
        current = self.client.json(self.path + "/config", method="GET", timeout=self.timeout_seconds)
        self._status = current
        if "calibration-metadata" not in self._description["capabilities"]:
            raise MediaSnapshotError("camera source does not advertise calibration-metadata")
        if (calibration.get("scope") != "calibration-metadata"
                or calibration.get("width") != self._description.get("width")
                or calibration.get("height") != self._description.get("height")):
            raise MediaSnapshotError("saved intrinsic dimensions do not match the camera source")
        revision = current.get("desired_revision")
        if not isinstance(revision, int) or isinstance(revision, bool) or revision < 0:
            raise MediaSnapshotError("camera source omitted its desired revision")
        identity = uuid.uuid4().hex
        result = self.client.json(self.path + "/config",
            {"expected_revision": revision, "persist": False,
                  "config": {"calibration": calibration}}, method="PATCH", request_id=identity, timeout=self.timeout_seconds)
        receipt = result.get("receipt") if isinstance(result, dict) else None
        desired = result.get("desired_revision") if isinstance(result, dict) else None
        applied = result.get("applied_revision") if isinstance(result, dict) else None
        published = result.get("published_calibration_revision") if isinstance(result, dict) else None
        if (not isinstance(receipt, dict) or receipt.get("stage") != "completed"
                or receipt.get("effects", {}).get("applied") is not True
                or receipt.get("request_id") != identity
                or desired != revision + 1 or applied != desired or published != applied
                or result.get("source_id") != self.source_id
                or result.get("applied", {}).get("calibration") != calibration):
            raise MediaSnapshotError("camera source did not prove CameraInfo metadata publication")
        self._status = result
        return {"scope": "calibration-metadata", "source_id": self.source_id,
            "persisted": False, "receipt": receipt,
            "desired_revision": desired, "applied_revision": applied,
            "published_calibration_revision": published}


@dataclass(frozen=True)
class MediaSnapshot:
    """One decoded working frame and source metadata from the same snapshot."""

    id: str
    source_id: str
    frame_id: str
    timestamp_nanoseconds: int
    width: int
    height: int
    camera_matrix: Optional[np.ndarray]
    distortion: Optional[np.ndarray]
    jpeg: bytes
    bgr: np.ndarray
    timestamp_clock_domain: str = "unknown"
    frame_sequence: int = 0
    pose_frame_id: Optional[str] = None
    render_position: Optional[Tuple[float, float, float]] = None
    render_orientation: Optional[Tuple[float, float, float, float]] = None


class MediaSnapshotClient:
    """One target-local, instance-fenced XRPC camera capture client.

    The process owner grants the private socket. Discovery is explicit and
    bounded; domain calls never fall back to the browser signaling listener.
    The SDK owns reusable HTTP sessions, framing, deadlines and no-replay policy.
    """

    def __init__(self, rpc_socket, source_id, timeout_seconds=5.0, *, runtime=None,
                 local_target=None):
        if not _SOURCE_ID.fullmatch(str(source_id).strip()):
            raise ValueError("media_source_id must be a stable identifier")
        if not math.isfinite(float(timeout_seconds)) or float(timeout_seconds) <= 0:
            raise ValueError("snapshot timeout must be positive")
        if not isinstance(local_target, str) or not local_target:
            raise ValueError("an explicit local target binding is required")
        discovery = ServiceRef(local_target, "media-edge", "v1", "", "http.v1",
                               Endpoint("unix", str(rpc_socket)))
        discovery.validate(discovery=True)
        self.source_id = str(source_id).strip()
        self.timeout_seconds = float(timeout_seconds)
        self.runtime = runtime or Runtime(blocking_workers=4)
        self._owns_runtime = runtime is None
        self._closed = False
        self.last_cleanup_error = None
        self.local_target = local_target
        self.limits = Limits(connections=2, in_flight=4,
            response_bytes=_MAX_JPEG_BYTES + _MAX_RGB_BYTES + (128 << 10))
        try:
            with Client.from_service(discovery, runtime=self.runtime,
                                     local_target=self.local_target, discovery=True,
                                     limits=self.limits) as client:
                document = client.json("/v1/describe", method="GET", timeout=self.timeout_seconds)
            if not isinstance(document, dict) or not isinstance(document.get("service_ref"), dict):
                raise ValueError("media edge discovery omitted service_ref")
            service = ServiceRef.from_dict(document["service_ref"])
            service.validate()
            if (service.service != "media-edge" or service.api_version != "v1"
                    or service.target_id != self.local_target or service.endpoint != discovery.endpoint):
                raise ValueError("media edge discovery identity does not match the granted endpoint")
            self.client = Client.from_service(service, runtime=self.runtime,
                                             local_target=self.local_target, limits=self.limits)
        except BaseException:
            if self._owns_runtime:
                self.runtime.close()
            raise

    def close(self):
        if self._closed:
            return
        self.client.close()
        if self._owns_runtime:
            self.runtime.close()
        self._closed = True

    def health(self):
        try:
            payload = self.client.json("/v1/health", method="GET", timeout=self.timeout_seconds)
        except (Fault, TransportError, OSError, TimeoutError) as error:
            raise MediaSnapshotError("media edge health is unavailable") from error
        sources = payload.get("sources") if isinstance(payload, dict) else None
        if not isinstance(sources, list) or not any(
            isinstance(item, dict) and item.get("id") == self.source_id for item in sources
        ):
            raise MediaSnapshotError("configured media source is unavailable")
        return payload

    def capture(self):
        return self._capture(include_rgb=True, maximum_pixels=None)

    def camera_metadata_client(self):
        """Use the source owner's observed ref, never construct its endpoint."""
        value = self.client.json("/v1/media/sources/{}/ref".format(quote(self.source_id, safe="")),
            method="GET", timeout=self.timeout_seconds)
        if not isinstance(value, dict) or value.get("source_id") != self.source_id:
            raise MediaSnapshotError("media edge omitted the configured source binding")
        reference = ServiceRef.from_dict(value.get("service_ref"))
        return CameraMetadataClient(reference, self.source_id, runtime=self.runtime,
            local_target=self.local_target, timeout_seconds=self.timeout_seconds)

    def capture_detection(self, maximum_pixels=640 * 480):
        if not isinstance(maximum_pixels, int) or isinstance(maximum_pixels, bool) or maximum_pixels < 4096:
            raise ValueError("maximum detection pixels must be an integer of at least 4096")
        return self._capture(include_rgb=False, maximum_pixels=maximum_pixels)

    def _parts(self, response, include_rgb):
        # Python's maintained MIME parser owns binary framing. The SDK bounds
        # the complete response; each decoded part is independently bounded.
        # This buffered API has bounded copies, not a zero-copy claim.
        message = BytesParser(policy=policy.default).parsebytes(
            ("Content-Type: " + response.content_type + "\r\n\r\n").encode("ascii")
            + response.body)
        boundary = message.get_boundary()
        if (message.get_content_type() != "multipart/mixed" or not boundary
                or len(boundary) > 70 or not boundary.isascii()
                or not message.is_multipart() or message.defects
                or message.preamble or message.epilogue):
            raise MediaSnapshotError("media snapshot requires bounded multipart/mixed")
        parts = list(message.iter_parts())
        expected = [("metadata", "application/json", 64 << 10),
                    ("jpeg", "image/jpeg", _MAX_JPEG_BYTES)]
        if include_rgb:
            expected.append(("rgb", "application/octet-stream", _MAX_RGB_BYTES))
        if len(parts) != len(expected):
            raise MediaSnapshotError("media snapshot has unexpected parts")
        payloads = []
        for part, (name, mime, maximum) in zip(parts, expected):
            if (part.defects or part.is_multipart()
                    or any(key.lower() not in {"content-type", "content-disposition", "content-length"} for key, _ in part.items())
                    or len(part.get_all("Content-Length", [])) > 1
                    or len(part.get_all("Content-Type", [])) != 1
                    or len(part.get_all("Content-Disposition", [])) != 1
                    or part.get_content_type() != mime
                    or part.get_content_disposition() != "inline"
                    or part.get_param("name", header="Content-Disposition") != name
                    or part.get_filename() is not None
                    or part.get("Content-Transfer-Encoding") is not None):
                raise MediaSnapshotError("media snapshot part headers are invalid")
            payload = part.get_payload(decode=True)
            if not isinstance(payload, bytes) or not 0 < len(payload) <= maximum:
                raise MediaSnapshotError("media snapshot part exceeds its byte limit")
            announced = part.get("Content-Length")
            if announced is not None and (not announced.isascii() or not announced.isdigit() or int(announced) != len(payload)):
                raise MediaSnapshotError("media snapshot part length is invalid")
            payloads.append(payload)
        try:
            metadata = json.loads(payloads[0])
        except (ValueError, UnicodeError) as error:
            raise MediaSnapshotError("media snapshot metadata is invalid JSON") from error
        return metadata, payloads[1], payloads[2] if include_rgb else b""

    def _capture(self, include_rgb, maximum_pixels):
        identity = uuid.uuid4().hex
        try:
            response = self.client.call("/v1/media/sources/{}/capture".format(quote(self.source_id, safe="")),
                {"snapshotId": identity, "includeRgb": include_rgb,
                 "requireFresh": True, "requestKeyframe": False}, timeout=self.timeout_seconds)
            metadata, jpeg, raw = self._parts(response, include_rgb)
            parsed = self._metadata(metadata)
            if parsed["id"] != identity:
                raise MediaSnapshotError("media snapshot identity does not match the capture request")
            if (metadata.get("jpegBytes") != len(jpeg) or metadata.get("rgbBytes") != len(raw)
                    or len(jpeg) < 2):
                raise MediaSnapshotError("media snapshot announced lengths do not match frame bytes")
            if include_rgb:
                if len(raw) != parsed["width"] * parsed["height"] * 3:
                    raise MediaSnapshotError("media snapshot RGB size does not match its dimensions")
                rgb = np.frombuffer(raw, dtype=np.uint8).reshape(parsed["height"], parsed["width"], 3)
                bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            else:
                bgr = self._decode_detection_jpeg(jpeg, parsed["width"], parsed["height"],
                    maximum_pixels or parsed["width"] * parsed["height"])
            return MediaSnapshot(id=identity, source_id=parsed["source_id"], frame_id=parsed["frame_id"],
                timestamp_nanoseconds=parsed["timestamp_nanoseconds"], width=parsed["width"],
                height=parsed["height"], camera_matrix=None if parsed["camera_matrix"] is None else
                    np.asarray(parsed["camera_matrix"], dtype=np.float64).reshape(3, 3),
                distortion=None if parsed["distortion"] is None else np.asarray(parsed["distortion"], dtype=np.float64),
                jpeg=jpeg, bgr=bgr, timestamp_clock_domain=parsed["timestamp_clock_domain"],
                frame_sequence=parsed["frame_sequence"], pose_frame_id=parsed["pose_frame_id"],
                render_position=parsed["render_position"], render_orientation=parsed["render_orientation"])
        except (Fault, TransportError, OSError, TimeoutError) as error:
            # No mutation replay. In particular outcome_unknown remains a
            # transport cause, not a claim that capture never happened.
            raise MediaSnapshotError("media edge capture is unavailable") from error
        finally:
            try:
                self.client.call("/v1/media/snapshots/" + identity, method="DELETE",
                                 timeout=min(2.0, self.timeout_seconds))
                self.last_cleanup_error = None
            except (Fault, TransportError, OSError, TimeoutError) as error:
                self.last_cleanup_error = type(error).__name__

    @staticmethod
    def _decode_detection_jpeg(
        jpeg: bytes,
        source_width: int,
        source_height: int,
        maximum_pixels: int,
    ) -> np.ndarray:
        ratio = math.sqrt(float(source_width * source_height) / float(maximum_pixels))
        if ratio >= 8.0:
            flag = cv2.IMREAD_REDUCED_COLOR_8
        elif ratio >= 4.0:
            flag = cv2.IMREAD_REDUCED_COLOR_4
        elif ratio >= 2.0:
            flag = cv2.IMREAD_REDUCED_COLOR_2
        else:
            flag = cv2.IMREAD_COLOR
        bgr = cv2.imdecode(np.frombuffer(jpeg, dtype=np.uint8), flag)
        if not isinstance(bgr, np.ndarray) or bgr.ndim != 3 or bgr.shape[2] != 3:
            raise MediaSnapshotError("media snapshot JPEG could not be decoded")
        if bgr.shape[0] * bgr.shape[1] > maximum_pixels:
            scale = math.sqrt(float(maximum_pixels) / float(bgr.shape[0] * bgr.shape[1]))
            target_width = max(1, int(bgr.shape[1] * scale))
            target_height = max(1, int(bgr.shape[0] * scale))
            bgr = cv2.resize(
                bgr,
                (target_width, target_height),
                interpolation=cv2.INTER_AREA,
            )
        return bgr

    def _snapshot_id(self, value: Any) -> str:
        if not isinstance(value, dict):
            raise MediaSnapshotError("media snapshot response is invalid")
        snapshot_id = value.get("snapshotId")
        if not _SOURCE_ID.fullmatch(snapshot_id if isinstance(snapshot_id, str) else ""):
            raise MediaSnapshotError("media snapshot ID is invalid")
        return snapshot_id

    def _metadata(self, value: Any) -> Dict[str, Any]:
        if not isinstance(value, dict):
            raise MediaSnapshotError("media snapshot response is invalid")
        snapshot_id = self._snapshot_id(value)
        source_id = value.get("sourceId")
        frame_id = value.get("frameId")
        width = value.get("width")
        height = value.get("height")
        timestamp = value.get("timestampNanoseconds")
        pixel_format = value.get("pixelFormat")
        matrix = value.get("cameraMatrix")
        distortion = value.get("distortion")
        render_pose = value.get("renderPose")
        if source_id != self.source_id or not isinstance(frame_id, str) or not frame_id:
            raise MediaSnapshotError("media snapshot source metadata is invalid")
        if isinstance(width, bool) or isinstance(height, bool) or not isinstance(width, int) or not isinstance(height, int) or not (16 <= width <= 8192 and 16 <= height <= 8192):
            raise MediaSnapshotError("media snapshot dimensions are invalid")
        # Simulation time begins at exactly zero. The Gazebo source and Media
        # Edge contract both preserve that valid first-frame timestamp; treating
        # it as missing made automatic calibration fail nondeterministically
        # whenever an on-demand camera was activated at the simulation epoch.
        if isinstance(timestamp, bool) or not isinstance(timestamp, int) or timestamp < 0:
            raise MediaSnapshotError("media snapshot clock is invalid")
        if pixel_format != "rgb8":
            raise MediaSnapshotError("media snapshot pixel format is invalid")
        calibration = value.get("calibrationState")
        if calibration == "unavailable":
            if matrix or distortion:
                raise MediaSnapshotError("unavailable calibration must not fabricate camera metadata")
            matrix, distortion = None, None
        elif not isinstance(calibration, str) or not calibration or not _finite_vector(matrix, 9) or len(matrix) != 9 or not _finite_vector(distortion, 4) or len(distortion) > 16:
            raise MediaSnapshotError("media snapshot camera metadata is invalid")
        clock = value.get("timestampClockDomain")
        sequence = value.get("frameSequence")
        if clock not in {"simulation", "system_realtime", "monotonic", "device", "unknown"}:
            raise MediaSnapshotError("media snapshot clock domain is invalid")
        if isinstance(sequence, bool) or not isinstance(sequence, int) or sequence <= 0:
            raise MediaSnapshotError("media snapshot frame sequence is invalid")
        pose_frame = value.get("poseFrameId")
        render_position = None
        render_orientation = None
        if render_pose is not None:
            if (
                not isinstance(pose_frame, str) or not pose_frame
                or not isinstance(render_pose, dict)
                or not _finite_xyz(render_pose.get("position"))
                or not _finite_xyzw(render_pose.get("orientation"))
            ):
                raise MediaSnapshotError("media snapshot render pose is invalid")
            position = render_pose["position"]
            orientation = render_pose["orientation"]
            render_position = (
                float(position["x"]), float(position["y"]), float(position["z"])
            )
            render_orientation = (
                float(orientation["x"]), float(orientation["y"]),
                float(orientation["z"]), float(orientation["w"]),
            )
        return {
            "id": snapshot_id,
            "source_id": source_id,
            "frame_id": frame_id,
            "timestamp_nanoseconds": timestamp,
            "width": width,
            "height": height,
            "timestamp_clock_domain": clock,
            "frame_sequence": sequence,
            "pose_frame_id": pose_frame,
            "camera_matrix": matrix,
            "distortion": distortion,
            "render_position": render_position,
            "render_orientation": render_orientation,
        }


def _finite_vector(value: Any, minimum: int) -> bool:
    if not isinstance(value, list) or len(value) < minimum:
        return False
    try:
        array = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError):
        return False
    return array.ndim == 1 and bool(np.all(np.isfinite(array)))


def _finite_xyz(value: Any) -> bool:
    if not isinstance(value, dict) or set(value) != {"x", "y", "z"}:
        return False
    return _finite_vector([value["x"], value["y"], value["z"]], 3)


def _finite_xyzw(value: Any) -> bool:
    if not isinstance(value, dict) or set(value) != {"x", "y", "z", "w"}:
        return False
    return _finite_vector([value["x"], value["y"], value["z"], value["w"]], 4)
