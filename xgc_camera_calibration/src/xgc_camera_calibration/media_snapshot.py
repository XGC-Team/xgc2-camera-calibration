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
from dataclasses import dataclass
from typing import Any, Dict, Optional, Sequence, Tuple
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlsplit
from urllib.request import Request, urlopen

import cv2
import numpy as np


_SOURCE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_MAX_JPEG_BYTES = 32 << 20
_MAX_RGB_BYTES = 128 << 20


class MediaSnapshotError(RuntimeError):
    """An expected Media Edge snapshot failure suitable for an API response."""


@dataclass(frozen=True)
class MediaSnapshot:
    """One decoded working frame and source metadata from the same snapshot."""

    id: str
    source_id: str
    frame_id: str
    timestamp_nanoseconds: int
    width: int
    height: int
    camera_matrix: np.ndarray
    distortion: np.ndarray
    jpeg: bytes
    bgr: np.ndarray
    render_position: Optional[Tuple[float, float, float]] = None
    render_orientation: Optional[Tuple[float, float, float, float]] = None


class MediaSnapshotClient:
    """Strict loopback client for one Media Edge source.

    The control listener is deliberately loopback-only.  Rejecting a remote
    URL here keeps the calibration process from silently becoming a second
    remote camera transport or bypassing the Media Edge trust boundary.
    """

    def __init__(self, edge_address: str, source_id: str, timeout_seconds: float = 5.0):
        parsed = urlsplit(str(edge_address).strip())
        host = (parsed.hostname or "").lower()
        if parsed.scheme != "http" or host not in {"127.0.0.1", "localhost", "::1"}:
            raise ValueError("media_edge_address must be an absolute loopback http URL")
        if parsed.username or parsed.password or parsed.query or parsed.fragment:
            raise ValueError("media_edge_address must not contain credentials, query, or fragment")
        if parsed.path not in ("", "/"):
            raise ValueError("media_edge_address must not contain a path")
        if not _SOURCE_ID.fullmatch(str(source_id).strip()):
            raise ValueError("media_source_id must be a stable identifier")
        if not np.isfinite(float(timeout_seconds)) or float(timeout_seconds) <= 0.0:
            raise ValueError("snapshot timeout must be positive")
        self.edge_address = "{}://{}".format(parsed.scheme, parsed.netloc)
        self.source_id = str(source_id).strip()
        self.timeout_seconds = float(timeout_seconds)

    def health(self) -> Dict[str, Any]:
        payload = self._json("GET", "/healthz")
        if not isinstance(payload, dict):
            raise MediaSnapshotError("media edge health response is invalid")
        sources = payload.get("sources")
        source = next((
            item for item in sources
            if isinstance(item, dict) and item.get("id") == self.source_id
        ), None) if isinstance(sources, list) else None
        if source is None:
            raise MediaSnapshotError("configured media source is unavailable")
        return payload

    def capture(self) -> MediaSnapshot:
        """Capture and consume one immutable frame, releasing Edge memory.

        RGB and JPEG are fetched only for this operation.  The ``finally`` is
        important at 4K: it releases the ~24 MiB RGB transaction immediately
        instead of waiting for TTL-based cleanup.
        """
        return self._capture(include_rgb=True, maximum_pixels=None)

    def capture_detection(self, maximum_pixels: int = 640 * 480) -> MediaSnapshot:
        """Capture one fresh JPEG-only frame near the detector pixel budget.

        Source metadata and camera intrinsics retain their original dimensions;
        only ``bgr`` is reduced. This avoids the 4K RGB payload, preserves the
        source aspect ratio at an approximately VGA budget, and lets the
        calibration service map detected corners back to the source plane.
        """
        if (
            not isinstance(maximum_pixels, int)
            or isinstance(maximum_pixels, bool)
            or maximum_pixels < 64 * 64
        ):
            raise ValueError("maximum detection pixels must be an integer of at least 4096")
        return self._capture(include_rgb=False, maximum_pixels=maximum_pixels)

    def _capture(self, include_rgb: bool, maximum_pixels: Optional[int]) -> MediaSnapshot:
        request = b"{}" if include_rgb else json.dumps(
            {"includeRgb": False, "requestKeyframe": False, "requireFresh": True},
            separators=(",", ":"),
        ).encode("utf-8")
        metadata = self._json(
            "POST",
            "/api/v1/sources/{}/snapshots".format(quote(self.source_id, safe="")),
            request,
        )
        snapshot_id = self._snapshot_id(metadata)
        try:
            parsed = self._metadata(metadata)
            jpeg = self._bytes("GET", "/api/v1/snapshots/{}/jpeg".format(quote(snapshot_id, safe="")), _MAX_JPEG_BYTES)
            if include_rgb:
                raw, headers = self._bytes_with_headers(
                    "GET", "/api/v1/snapshots/{}/raw".format(quote(snapshot_id, safe="")), _MAX_RGB_BYTES
                )
                self._validate_raw_headers(headers, parsed)
                expected_size = parsed["width"] * parsed["height"] * 3
                if len(raw) != expected_size:
                    raise MediaSnapshotError("media snapshot RGB size does not match its dimensions")
                rgb = np.frombuffer(raw, dtype=np.uint8).reshape(parsed["height"], parsed["width"], 3)
                bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            else:
                bgr = self._decode_detection_jpeg(
                    jpeg,
                    parsed["width"],
                    parsed["height"],
                    maximum_pixels or parsed["width"] * parsed["height"],
                )
            return MediaSnapshot(
                id=snapshot_id,
                source_id=parsed["source_id"],
                frame_id=parsed["frame_id"],
                timestamp_nanoseconds=parsed["timestamp_nanoseconds"],
                width=parsed["width"],
                height=parsed["height"],
                camera_matrix=np.asarray(parsed["camera_matrix"], dtype=np.float64).reshape(3, 3),
                distortion=np.asarray(parsed["distortion"], dtype=np.float64),
                jpeg=jpeg,
                bgr=bgr,
                render_position=parsed["render_position"],
                render_orientation=parsed["render_orientation"],
            )
        finally:
            # Deletion is best-effort: a failed cleanup still expires quickly
            # in Media Edge, but a successful path must not retain large RGB.
            try:
                self._bytes("DELETE", "/api/v1/snapshots/{}".format(quote(snapshot_id, safe="")), 0)
            except MediaSnapshotError:
                pass

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
        if not isinstance(width, int) or not isinstance(height, int) or not (16 <= width <= 8192 and 16 <= height <= 8192):
            raise MediaSnapshotError("media snapshot dimensions are invalid")
        # Simulation time begins at exactly zero. The Gazebo source and Media
        # Edge contract both preserve that valid first-frame timestamp; treating
        # it as missing made automatic calibration fail nondeterministically
        # whenever an on-demand camera was activated at the simulation epoch.
        if not isinstance(timestamp, int) or timestamp < 0:
            raise MediaSnapshotError("media snapshot clock is invalid")
        if pixel_format != "rgb8":
            raise MediaSnapshotError("media snapshot pixel format is invalid")
        if not _finite_vector(matrix, 9) or not _finite_vector(distortion, 4):
            raise MediaSnapshotError("media snapshot camera metadata is invalid")
        render_position = None
        render_orientation = None
        if render_pose is not None:
            if (
                not isinstance(render_pose, dict)
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
            "camera_matrix": matrix,
            "distortion": distortion,
            "render_position": render_position,
            "render_orientation": render_orientation,
        }

    def _validate_raw_headers(self, headers: Any, metadata: Dict[str, Any]) -> None:
        if headers.get_content_type() != "application/x-xgc-rgb8":
            raise MediaSnapshotError("media snapshot raw response has an unexpected content type")
        expected = {
            "X-Xgc-Snapshot-Id": metadata["id"],
            "X-Xgc-Frame-Id": metadata["frame_id"],
            "X-Xgc-Width": str(metadata["width"]),
            "X-Xgc-Height": str(metadata["height"]),
        }
        for name, value in expected.items():
            if headers.get(name) != value:
                raise MediaSnapshotError("media snapshot raw response metadata does not match capture metadata")

    def _json(self, method: str, path: str, body: bytes | None = None) -> Any:
        payload = self._bytes(method, path, _MAX_JPEG_BYTES, body)
        try:
            return json.loads(payload.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise MediaSnapshotError("media edge returned invalid JSON") from error

    def _bytes(self, method: str, path: str, maximum: int, body: bytes | None = None) -> bytes:
        payload, _headers = self._bytes_with_headers(method, path, maximum, body)
        return payload

    def _bytes_with_headers(
        self, method: str, path: str, maximum: int, body: bytes | None = None
    ) -> tuple[bytes, Any]:
        request = Request(
            self.edge_address + path,
            data=body,
            method=method,
            headers={"Accept": "application/json", **({"Content-Type": "application/json"} if body is not None else {})},
        )
        try:
            with urlopen(request, timeout=self.timeout_seconds) as response:
                payload = response.read(maximum + 1) if maximum else response.read(1)
                if maximum and len(payload) > maximum:
                    raise MediaSnapshotError("media edge response exceeds the allowed size")
                return payload, response.headers
        except HTTPError as error:
            message = _http_error_message(error)
            raise MediaSnapshotError("media edge {} {} failed: {}".format(method, path, message)) from error
        except (URLError, OSError, TimeoutError) as error:
            raise MediaSnapshotError("media edge {} {} is unavailable: {}".format(method, path, error)) from error


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


def _http_error_message(error: HTTPError) -> str:
    try:
        payload = json.loads(error.read().decode("utf-8"))
        if isinstance(payload, dict) and isinstance(payload.get("error"), str):
            return payload["error"]
    except (UnicodeDecodeError, json.JSONDecodeError, OSError):
        pass
    return "HTTP {}".format(error.code)
