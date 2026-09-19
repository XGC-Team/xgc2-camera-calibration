"""Bounded, process-owned independent extrinsic observations; no image acquisition."""
from __future__ import annotations

import copy
import hashlib
import json
import math
import re
import struct
import time
import uuid
from collections import OrderedDict
from typing import Any, Dict, Optional

import cv2
import numpy as np

IMAGE_BYTES = 32 << 20
IMAGE_PIXELS = 16 * 1024 * 1024
LIMITS = {"image_bytes": IMAGE_BYTES, "image_pixels": IMAGE_PIXELS, "samples": 64,
          "pending": 4, "total_image_bytes": 256 << 20, "pending_seconds": 120}
IMAGE_PATH = re.compile(r"^/api/v1/samples/([0-9a-f]{32})/image$")
TOKEN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$")


class ApiError(RuntimeError):
    """An expected request or calibration-input failure."""
    def __init__(self, status: int, message: str, *, details: Optional[Dict[str, Any]] = None):
        super().__init__(message)
        self.status, self.message, self.details = int(status), str(message), dict(details or {})


def cloned(value):
    return copy.deepcopy(value)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def finite(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ApiError(400, name + " must be a finite number")
    return float(value)


def pixel(value, width, height):
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ApiError(400, "pixel must contain two coordinates")
    result = [finite(item, "pixel") for item in value]
    if not (0 <= result[0] < width and 0 <= result[1] < height):
        raise ApiError(400, "pixel is outside the displayed image")
    return result


def image_dimensions(payload, mime):
    """Read bounded encoded headers before asking OpenCV to allocate pixels."""
    if mime == "image/png":
        if len(payload) < 33 or payload[:8] != b"\x89PNG\r\n\x1a\n" or payload[8:16] != b"\x00\x00\x00\x0dIHDR":
            raise ApiError(400, "Invalid PNG image")
        return struct.unpack(">II", payload[16:24])
    if mime != "image/jpeg":
        raise ApiError(415, "Sample image must be PNG or JPEG")
    if not payload.startswith(b"\xff\xd8"):
        raise ApiError(400, "Invalid JPEG image")
    at = 2
    while at < len(payload):
        if payload[at] != 0xff:
            break
        while at < len(payload) and payload[at] == 0xff:
            at += 1
        if at >= len(payload):
            break
        marker = payload[at]
        at += 1
        if marker in (0xd9, 0xda):
            break
        if marker == 0x01 or 0xd0 <= marker <= 0xd7:
            continue
        if at + 2 > len(payload):
            break
        size = int.from_bytes(payload[at:at + 2], "big")
        if size < 2 or at + size > len(payload):
            break
        if marker in (0xc0, 0xc1, 0xc2, 0xc3, 0xc5, 0xc6, 0xc7, 0xc9, 0xca, 0xcb, 0xcd, 0xce, 0xcf):
            if size < 8:
                break
            return int.from_bytes(payload[at + 5:at + 7], "big"), int.from_bytes(payload[at + 3:at + 5], "big")
        at += size
    raise ApiError(400, "JPEG image dimensions are unavailable")


def dimensions(width, height):
    if any(type(item) is not int or not 1 <= item <= 8192 for item in (width, height)):
        raise ApiError(400, "Image dimensions must be integers between 1 and 8192")
    if width * height > IMAGE_PIXELS:
        raise ApiError(413, "Sample image exceeds the pixel limit")


class SampleCollection:
    """The service lock also serializes sample mutation with candidate Save."""
    def __init__(self, source, parent_frame, lock, changed):
        self.source, self.parent_frame, self.lock, self.changed = source, parent_frame, lock, changed
        self.session_id = uuid.uuid4().hex
        self.revision = 0
        self.active = OrderedDict()
        self.pending = {}
        self.requests = OrderedDict()

    def _check(self, request, fields, optional=(), revision=True):
        if not isinstance(request, dict) or set(request) != (set(fields) | (set(request) & set(optional))):
            raise ApiError(400, "Invalid sample request fields")
        if request.get("sampling_session_id") != self.session_id:
            raise ApiError(409, "Sampling session changed", details={"dataset_revision": self.revision})
        if revision and (type(request.get("expected_revision")) is not int or request["expected_revision"] != self.revision):
            raise ApiError(409, "Sample dataset changed", details={"dataset_revision": self.revision})

    def _expire(self):
        now = time.monotonic()
        for identity, pending in list(self.pending.items()):
            if now >= pending["expires"]:
                del self.pending[identity]
        for request_id, record in list(self.requests.items()):
            if record["id"] not in self.pending and record["id"] not in self.active and now >= record["expires"]:
                del self.requests[request_id]

    def state(self):
        with self.lock:
            self._expire()
            return {"sampling_session_id": self.session_id, "dataset_revision": self.revision,
                    "samples": [cloned(item["sample"]) for item in self.active.values()],
                    "sample_limits": dict(LIMITS)}

    def _display(self, value):
        required = {"id", "source_id", "source_epoch", "width", "height", "clock_domain", "presented_at_ms"}
        optional = {"time_origin_ms", "media_time_sec", "presented_frames", "capture_time_ms", "receive_time_ms", "rtp_timestamp"}
        if not isinstance(value, dict) or not required <= set(value) or set(value) - required - optional:
            raise ApiError(400, "Invalid display observation")
        for name in ("id", "source_epoch"):
            if not isinstance(value[name], str) or not TOKEN.fullmatch(value[name]):
                raise ApiError(400, "Invalid display " + name)
        if value["source_id"] != self.source.source_id:
            raise ApiError(409, "Displayed source does not match the calibration source")
        if value["clock_domain"] != "browser-performance":
            raise ApiError(400, "Display clock must be browser-performance")
        dimensions(value["width"], value["height"])
        for name in optional | {"presented_at_ms"}:
            if name in value:
                finite(value[name], name)
                # Estimated capture time may precede this page's time origin.
                # Do not turn a valid clock offset into a motion/readiness gate.
                if name in ("presented_frames", "rtp_timestamp") and (type(value[name]) is not int or value[name] < 0):
                    raise ApiError(400, name + " must be a nonnegative integer")
        return cloned(value)

    def begin(self, request):
        with self.lock:
            self._expire()
            fields = {"sampling_session_id", "expected_revision", "request_id", "marker", "pixel", "display"}
            self._check(request, fields, {"replaces_sample_id"}, revision=False)
            request_id = request["request_id"]
            if not isinstance(request_id, str) or not TOKEN.fullmatch(request_id):
                raise ApiError(400, "Invalid sample request_id")
            try:
                request_digest = digest(request)
            except (ValueError, TypeError) as error:
                raise ApiError(400, "Sample request is not finite JSON") from error
            old = self.requests.get(request_id)
            if old is not None:
                if old["digest"] != request_digest:
                    raise ApiError(409, "Sample request_id was reused with different content")
                if old["id"] in self.pending or old["id"] in self.active:
                    return self._admission(old["id"])
                raise ApiError(409, "Sample request was cancelled or expired")
            self._check(request, fields, {"replaces_sample_id"})
            display = self._display(request["display"])
            position = pixel(request["pixel"], display["width"], display["height"])
            marker = request["marker"]
            if not isinstance(marker, str) or not marker or len(marker) > 256:
                raise ApiError(400, "Invalid marker")
            replacement = request.get("replaces_sample_id")
            if replacement is not None and (not isinstance(replacement, str) or not re.fullmatch(r"[0-9a-f]{32}", replacement)):
                raise ApiError(400, "Invalid replacement sample identity")
            if replacement is not None and replacement not in self.active:
                raise ApiError(404, "Replacement sample is unavailable")
            if len(self.pending) >= LIMITS["pending"] or len(self.requests) >= 256 or (replacement is None and len(self.active) >= LIMITS["samples"]):
                raise ApiError(413, "Sample capacity reached")
            context = self.source.observe_marker(marker, display["width"], display["height"], self.parent_frame)
            matrix = np.asarray(context["camera_matrix"], dtype=np.float64)
            distortion = np.asarray(context["distortion"], dtype=np.float64).reshape(-1)
            if matrix.shape != (3, 3) or not np.all(np.isfinite(matrix)) or matrix[0, 0] <= 0 or matrix[1, 1] <= 0 or not np.all(np.isfinite(distortion)):
                raise ApiError(409, "Selected camera intrinsics are invalid")
            model = cloned(dict(context["camera_model"]))
            model.update(camera_matrix=matrix.reshape(-1).tolist(), distortion=distortion.tolist(),
                         image_width=display["width"], image_height=display["height"])
            model.setdefault("intrinsic_source", "unspecified")
            model.setdefault("distortion_model", "plumb_bob")
            coordinates = cloned(dict(context["pose_coordinates"]))
            model_id, coordinate_id = digest(model), digest(coordinates)
            if self.active:
                first = next(iter(self.active.values()))["sample"]
                if first["camera_model_id"] != model_id or first["pose_coordinate_id"] != coordinate_id:
                    raise ApiError(409, "Sample camera model or coordinate basis changed")
            observed = context["marker"]
            if observed.name != marker or (observed.frame_id and observed.frame_id != self.parent_frame):
                raise ApiError(409, "Marker observation frame does not match")
            world = [finite(item, "world") for item in observed.position]
            raw = [finite(item, "source_world") for item in (observed.source_position or observed.position)]
            if len(world) != 3 or len(raw) != 3:
                raise ApiError(409, "Marker observation position is invalid")
            pose = {"observation_id": observed.observation_id, "frame_id": observed.frame_id,
                    "source_stamp_sec": finite(observed.source_stamp_sec, "pose source stamp"), "source_clock": "ros",
                    "received_at_sec": finite(observed.received_at_sec, "pose received time"), "received_clock": "unix",
                    "received_monotonic_sec": finite(observed.received_monotonic_sec, "pose receipt")}
            if not pose["observation_id"]:
                raise ApiError(409, "Marker observation identity is unavailable")
            identity = uuid.uuid4().hex
            sample = {"sample_id": identity, "marker": marker, "pixel": position, "world": world,
                      "source_world": raw, "display": display, "pose_observation": pose,
                      "camera_model_id": model_id, "pose_coordinate_id": coordinate_id}
            expires = time.monotonic() + LIMITS["pending_seconds"]
            self.pending[identity] = {"sample": sample, "model": model, "coordinates": coordinates,
                                      "revision": self.revision, "replacement": replacement, "expires": expires}
            self.requests[request_id] = {"digest": request_digest, "id": identity, "expires": expires}
            return self._admission(identity)

    def _admission(self, identity):
        return {"sample_id": identity, "sampling_session_id": self.session_id, "dataset_revision": self.revision,
                "status": "pending" if identity in self.pending else "completed",
                "expires_in_seconds": max(0, math.ceil(self.pending[identity]["expires"] - time.monotonic())) if identity in self.pending else 0}

    def commit(self, identity, payload, mime):
        with self.lock:
            self._expire()
            if len(payload) > IMAGE_BYTES:
                raise ApiError(413, "Sample image exceeds the byte limit")
            sha = hashlib.sha256(payload).hexdigest()
            old = self.active.get(identity)
            if old is not None:
                if old["sample"]["image"]["sha256"] != sha or old["mime"] != mime:
                    raise ApiError(409, "Sample image is immutable")
                return
            pending = self.pending.get(identity)
            if pending is None:
                raise ApiError(404, "Pending sample is unavailable")
            if pending["revision"] != self.revision:
                raise ApiError(409, "Sample dataset changed", details={"dataset_revision": self.revision})
            width, height = image_dimensions(payload, mime)
            dimensions(width, height)
            display = pending["sample"]["display"]
            if (width, height) != (display["width"], display["height"]):
                raise ApiError(409, "Sample image dimensions do not match the displayed observation")
            try:
                decoded = cv2.imdecode(np.frombuffer(payload, dtype=np.uint8), cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
            except cv2.error as error:
                raise ApiError(400, "Sample image cannot be decoded") from error
            if decoded is None or decoded.shape[:2] != (height, width):
                raise ApiError(400, "Sample image cannot be decoded")
            replacement = pending["replacement"]
            total = sum(len(item["bytes"]) for key, item in self.active.items() if key != replacement) + len(payload)
            if total > LIMITS["total_image_bytes"]:
                raise ApiError(413, "Sample image storage capacity reached")
            sample = pending["sample"]
            sample["image"] = {"path": "api/v1/samples/" + identity + "/image", "mime_type": mime,
                               "sha256": sha, "width": width, "height": height}
            item = {"sample": sample, "model": pending["model"], "coordinates": pending["coordinates"],
                    "bytes": bytes(payload), "mime": mime}
            if replacement:
                self.active = OrderedDict((identity, item) if key == replacement else (key, value)
                                          for key, value in self.active.items())
            else:
                self.active[identity] = item
            del self.pending[identity]
            self._changed()

    def _changed(self):
        self.revision += 1
        self.changed()

    def image(self, identity):
        with self.lock:
            item = self.active.get(identity)
            if item is None:
                raise ApiError(404, "Sample image is unavailable")
            return item["bytes"], item["mime"]

    def mutate(self, action, request):
        with self.lock:
            self._expire()
            fields = {"sampling_session_id"}
            if action != "cancel":
                fields.add("expected_revision")
            if action != "clear":
                fields.add("sample_id")
            if action == "pixel":
                fields.add("pixel")
            self._check(request, fields, revision=action != "cancel")
            if action != "clear" and (not isinstance(request["sample_id"], str) or not re.fullmatch(r"[0-9a-f]{32}", request["sample_id"])):
                raise ApiError(400, "Invalid sample identity")
            if action == "cancel":
                self.pending.pop(request["sample_id"], None)
                return
            if action == "clear":
                self.pending.clear()
                if self.active:
                    self.active.clear()
                    self._changed()
                return
            identity = request["sample_id"]
            if identity not in self.active:
                raise ApiError(404, "Sample is unavailable")
            if action == "remove":
                del self.active[identity]
            elif action == "pixel":
                item = self.active[identity]
                display = item["sample"]["display"]
                value = pixel(request["pixel"], display["width"], display["height"])
                if value == item["sample"]["pixel"]:
                    return
                item["sample"] = {**item["sample"], "pixel": value}
            else:
                raise ApiError(404, "Unknown sample operation")
            self._changed()

    def solve_inputs(self, request):
        self._check(request, {"sampling_session_id", "expected_revision"})
        if len(self.active) < 4:
            raise ApiError(400, "At least four independent correspondences are required")
        first = next(iter(self.active.values()))
        return [cloned(item["sample"]) for item in self.active.values()], cloned(first["model"]), cloned(first["coordinates"])
