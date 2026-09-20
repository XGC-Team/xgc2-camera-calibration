"""Exact immutable extrinsic references and the single pending/applied owner.

This module does not migrate or read legacy mode-specific selection pointers.
Only a producer's exact application acknowledgement can advance applied.
"""
from contextlib import contextmanager
import copy
import fcntl
import hashlib
import json
import math
import os
import re
import stat
import time
import uuid

import yaml

from .extrinsic_coordinates import coordinate_provenance, optical_translation_in_world
from .solver import CAMERA_NAME_PATTERN, EXTRINSIC_FILENAME_PATTERN, CalibrationError, _validate_extrinsic_document

MAX_RESULT_BYTES = 8 * 1024 * 1024
MAX_SELECTION_BYTES = 32 * 1024
MAX_REVISION = 2 ** 53 - 1
_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$")
_FRAME = re.compile(r"^[A-Za-z][A-Za-z0-9_/-]{0,127}$")
_REQUEST_FIELDS = {"applicationId", "candidateId", "result", "producer", "targetCoordinates"}


class SelectionConflict(CalibrationError):
    """The requested transition no longer owns the current selection revision."""


def closed_object(value, fields, label):
    if not isinstance(value, dict) or set(value) != set(fields):
        raise CalibrationError(label + " has an invalid shape")
    return value


def identifier(value, label):
    if not isinstance(value, str) or not _ID.fullmatch(value):
        raise CalibrationError(label + " is invalid")
    return value


def finite_vector(value, size, label):
    if not isinstance(value, (list, tuple)) or len(value) != size:
        raise CalibrationError(label + " has an invalid size")
    try:
        valid = all(type(v) in (int, float) and math.isfinite(v) for v in value)
    except OverflowError:
        valid = False
    if not valid:
        raise CalibrationError(label + " must contain finite numbers")
    return [float(v) for v in value]


def revision(value):
    if type(value) is not int or not 0 <= value <= MAX_REVISION:
        raise CalibrationError("selection revision is invalid")
    return value


def compact_json(value):
    try:
        return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    except (TypeError, ValueError, OverflowError, RecursionError) as error:
        raise CalibrationError("value is not finite JSON") from error


def strict_json(payload):
    def object_pairs(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise CalibrationError("JSON has duplicate object keys")
            result[key] = value
        return result

    def invalid_constant(_):
        raise CalibrationError("JSON contains a non-finite number")

    try:
        return json.loads(payload, object_pairs_hook=object_pairs, parse_constant=invalid_constant)
    except (UnicodeError, ValueError, TypeError, RecursionError) as error:
        raise CalibrationError("JSON is unreadable") from error


def validate_roles(value):
    closed_object(value, {"parentFrame", "opticalFrames"}, "frame roles")
    if value["parentFrame"] != "world":
        raise CalibrationError("extrinsic parent frame must be world")
    frames = closed_object(value["opticalFrames"], {"sim", "phy"}, "optical frame roles")
    if any(not isinstance(frame, str) or not _FRAME.fullmatch(frame) for frame in frames.values()):
        raise CalibrationError("optical frame role is invalid")
    return copy.deepcopy(value)


def validate_target(value):
    closed_object(value, {"frame", "worldOffset"}, "target coordinates")
    if value["frame"] != "world":
        raise CalibrationError("target frame must be world")
    return {"frame": "world", "worldOffset": finite_vector(value["worldOffset"], 3, "target offset")}


def validate_result(value):
    closed_object(value, {"sourceMode", "fileName", "sha256"}, "extrinsic version")
    if value["sourceMode"] not in ("sim", "phy"):
        raise CalibrationError("extrinsic source mode is invalid")
    if not isinstance(value["fileName"], str) or not EXTRINSIC_FILENAME_PATTERN.fullmatch(value["fileName"]):
        raise CalibrationError("extrinsic version must name an immutable result")
    if not isinstance(value["sha256"], str) or not re.fullmatch(r"[0-9a-f]{64}", value["sha256"]):
        raise CalibrationError("extrinsic version digest is invalid")
    return dict(value)


def validate_storage(root, camera_name):
    try:
        root = os.fspath(root)
    except TypeError as error:
        raise CalibrationError("calibration root must be an explicit absolute directory") from error
    if not isinstance(root, str) or not root.startswith("/") or "\0" in root or any(
        part in (".", "..") for part in root.split("/")
    ) or not [part for part in root.split("/") if part]:
        raise CalibrationError("calibration root must be an explicit absolute directory")
    if not isinstance(camera_name, str) or not CAMERA_NAME_PATTERN.fullmatch(camera_name):
        raise CalibrationError("camera name must be a stable identifier")
    return root


@contextmanager
def _directory(root, components=(), create=False):
    """Anchor every component with O_NOFOLLOW; do not follow renamed parents."""
    fd = os.open("/", os.O_RDONLY | os.O_DIRECTORY)
    try:
        for name, may_create in [(part, False) for part in root.split("/") if part] + [
            (part, create) for part in components
        ]:
            if may_create:
                try:
                    os.mkdir(name, mode=0o700, dir_fd=fd)
                    os.fsync(fd)
                except FileExistsError:
                    pass
            next_fd = os.open(name, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=fd)
            os.close(fd)
            fd = next_fd
        yield fd
    finally:
        os.close(fd)


def _read_file(directory_fd, name, limit):
    fd = os.open(name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=directory_fd)
    try:
        info = os.fstat(fd)
        if not stat.S_ISREG(info.st_mode) or info.st_size > limit:
            raise CalibrationError("calibration file is not regular or exceeds its byte limit")
        with os.fdopen(fd, "rb", closefd=False) as stream:
            data = stream.read(limit + 1)
        if len(data) > limit:
            raise CalibrationError("calibration file exceeds its byte limit")
        return data
    finally:
        os.close(fd)


def read_version(root, camera_name, result, frame_roles):
    """Verify bytes and declared identity, preserving the original YAML file."""
    root = validate_storage(root, camera_name)
    result = validate_result(result)
    roles = validate_roles(frame_roles)
    try:
        with _directory(root, (result["sourceMode"], camera_name)) as fd:
            payload = _read_file(fd, result["fileName"], MAX_RESULT_BYTES)
    except OSError as error:
        raise CalibrationError("extrinsic version is unavailable or contains a symbolic link") from error
    if hashlib.sha256(payload).hexdigest() != result["sha256"]:
        raise CalibrationError("extrinsic version digest does not match")
    try:
        raw = yaml.safe_load(payload.decode("utf-8"))
        if not isinstance(raw, dict):
            raise CalibrationError("extrinsic version must be an object")
        for field, names in (("translation", "xyz"), ("quaternion_xyzw", "xyzw")):
            values = raw.get(field)
            if isinstance(values, dict):
                values = [values.get(name) for name in names]
            finite_vector(values, len(names), field)
        document = _validate_extrinsic_document(raw)
    except (UnicodeError, yaml.YAMLError, TypeError, ValueError, RecursionError) as error:
        raise CalibrationError("extrinsic version is unreadable") from error
    if document["camera_name"] != camera_name or document["calibration_mode"] != result["sourceMode"]:
        raise CalibrationError("extrinsic version camera or mode does not match")
    if document["parent_frame"] != roles["parentFrame"] or document["child_frame"] != roles["opticalFrames"][result["sourceMode"]]:
        raise CalibrationError("extrinsic version frame role does not match")
    pose_from_document(document, {"frame": "world", "worldOffset": [0, 0, 0]})
    return document


def pose_from_document(document, target_coordinates):
    target = validate_target(target_coordinates)
    metadata = document.get("metadata", {})
    if not isinstance(metadata, dict):
        raise CalibrationError("extrinsic metadata must be an object")
    provenance = metadata.get("pose_coordinates")
    source = None
    if provenance is not None:
        if not isinstance(provenance, dict) or type(provenance.get("schema_version")) is not int or provenance["schema_version"] != 1:
            raise CalibrationError("extrinsic coordinate provenance schema is invalid")
        validated = coordinate_provenance(provenance.get("kind"), provenance.get("frame"),
                                           finite_vector(provenance.get("world_offset"), 3, "saved offset"))
        source = {"kind": validated["kind"], "frame": validated["frame"], "worldOffset": validated["world_offset"]}
    translation = finite_vector(document["translation_array"].tolist(), 3, "optical translation")
    quaternion = finite_vector(document["quaternion_xyzw_array"].tolist(), 4, "optical quaternion")
    if not math.isclose(sum(v * v for v in quaternion), 1.0, rel_tol=0, abs_tol=1e-6):
        raise CalibrationError("optical quaternion must be unit length")
    resolved = finite_vector(optical_translation_in_world(document, target["worldOffset"]).tolist(), 3, "resolved translation")
    return source, {"translation": translation, "quaternionXyzw": quaternion}, {
        "translation": resolved, "quaternionXyzw": list(quaternion),
    }


class SelectionStore:
    """Cross-process CAS around one global applied/pending selection document."""

    def __init__(self, root, camera_name, frame_roles, lock_timeout=2.0):
        self.root = validate_storage(root, camera_name)
        self.camera_name = camera_name
        self.frame_roles = validate_roles(frame_roles)
        if type(lock_timeout) not in (int, float) or not math.isfinite(lock_timeout) or not 0 < lock_timeout <= 15:
            raise CalibrationError("selection lock timeout must be in (0, 15] seconds")
        self.lock_timeout = lock_timeout
        try:
            with _directory(self.root):
                pass
        except OSError as error:
            raise CalibrationError("calibration root is unavailable or contains a symbolic link") from error

    def _empty(self):
        return {"schemaVersion": 2, "cameraName": self.camera_name, "revision": 0, "applied": None, "pending": None}

    def _request(self, request):
        closed_object(request, _REQUEST_FIELDS, "application request")
        producer = closed_object(request["producer"], {"resolutionId", "instanceEpoch"}, "producer identity")
        value = {"applicationId": identifier(request["applicationId"], "application id"),
                 "candidateId": identifier(request["candidateId"], "candidate id"),
                 "result": validate_result(request["result"]),
                 "producer": {key: identifier(producer[key], key) for key in producer},
                 "targetCoordinates": validate_target(request["targetCoordinates"])}
        document = read_version(self.root, self.camera_name, value["result"], self.frame_roles)
        if document.get("metadata", {}).get("candidate_id") != value["candidateId"]:
            raise CalibrationError("application candidate does not match immutable result")
        pose_from_document(document, value["targetCoordinates"])
        return value

    def _record(self, record, applied):
        extra = {"requestedRevision", "appliedRevision", "appliedAtUnixNs"} if applied else {"requestedRevision"}
        closed_object(record, _REQUEST_FIELDS | extra, "selection application")
        self._request({key: record[key] for key in _REQUEST_FIELDS})
        if revision(record["requestedRevision"]) == 0:
            raise CalibrationError("application revision must be positive")
        if applied:
            if revision(record["appliedRevision"]) != record["requestedRevision"] + 1:
                raise CalibrationError("applied revision does not follow its request")
            if not isinstance(record["appliedAtUnixNs"], str) or not re.fullmatch(r"[1-9][0-9]{0,19}", record["appliedAtUnixNs"]):
                raise CalibrationError("application timestamp is invalid")

    def _read_at(self, fd):
        try:
            payload = _read_file(fd, "extrinsic.json", MAX_SELECTION_BYTES)
        except FileNotFoundError:
            return self._empty()
        value = strict_json(payload)
        closed_object(value, {"schemaVersion", "cameraName", "revision", "applied", "pending"}, "selection")
        if type(value["schemaVersion"]) is not int or value["schemaVersion"] != 2 or value["cameraName"] != self.camera_name:
            raise CalibrationError("selection schema or camera does not match")
        current = revision(value["revision"])
        applied, pending = value["applied"], value["pending"]
        if applied is not None:
            self._record(applied, True)
            if applied["appliedRevision"] > current:
                raise CalibrationError("applied revision exceeds selection")
        if pending is not None:
            self._record(pending, False)
            if pending["requestedRevision"] != current or (applied is not None and applied["appliedRevision"] >= current):
                raise CalibrationError("pending revision does not own selection")
            if applied is not None and pending["applicationId"] == applied["applicationId"]:
                raise CalibrationError("pending application identity was already applied")
        elif applied is not None and applied["appliedRevision"] != current:
            raise CalibrationError("applied revision does not own selection")
        if (applied is None and pending is None) != (current == 0):
            raise CalibrationError("selection revision has no application")
        return value

    def read(self):
        try:
            with _directory(self.root, ("selections", self.camera_name)) as fd:
                return self._read_at(fd)
        except FileNotFoundError:
            # Missing selection directories mean first use; a vanished root is
            # a storage failure, not evidence that calibration never existed.
            try:
                with _directory(self.root):
                    pass
            except OSError as error:
                raise CalibrationError("calibration root is unavailable") from error
            return self._empty()
        except OSError as error:
            raise CalibrationError("selection is unavailable or contains a symbolic link") from error

    @contextmanager
    def _locked(self):
        try:
            with _directory(self.root, ("selections", self.camera_name), create=True) as fd:
                lock_fd = os.open(".extrinsic.lock", os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW | os.O_NONBLOCK,
                                  0o600, dir_fd=fd)
                try:
                    info = os.fstat(lock_fd)
                    if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
                        raise CalibrationError("selection lock must be an unaliased regular file")
                    deadline = time.monotonic() + self.lock_timeout
                    while True:
                        try:
                            fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                            break
                        except BlockingIOError:
                            if time.monotonic() >= deadline:
                                raise SelectionConflict("selection lock deadline exceeded")
                            time.sleep(min(0.01, self.lock_timeout))
                    yield fd
                finally:
                    os.close(lock_fd)
        except OSError as error:
            raise CalibrationError("selection storage operation failed") from error

    def _write_at(self, fd, value):
        payload = compact_json(value).encode("utf-8")
        if len(payload) > MAX_SELECTION_BYTES:
            raise CalibrationError("selection exceeds its byte limit")
        name = ".extrinsic-" + uuid.uuid4().hex + ".tmp"
        temporary_fd = os.open(name, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600, dir_fd=fd)
        try:
            with os.fdopen(temporary_fd, "wb") as stream:
                stream.write(payload)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(name, "extrinsic.json", src_dir_fd=fd, dst_dir_fd=fd)
            os.fsync(fd)
        finally:
            try:
                os.unlink(name, dir_fd=fd)
            except FileNotFoundError:
                pass

    @staticmethod
    def _same(record, request):
        return record is not None and all(record[key] == request[key] for key in _REQUEST_FIELDS)

    def stage(self, request, expected_revision):
        expected = revision(expected_revision)
        request = self._request(request)
        with self._locked() as fd:
            value = self._read_at(fd)
            current = value["pending"] or value["applied"]
            if self._same(current, request) and current["requestedRevision"] == expected + 1:
                os.fsync(fd)  # Complete durability after an uncertain replace acknowledgement.
                return value
            if value["revision"] != expected:
                raise SelectionConflict("selection revision changed; request is stale or superseded")
            if any(record and record["applicationId"] == request["applicationId"] for record in (value["applied"], value["pending"])):
                raise SelectionConflict("application identity was already used")
            value["revision"] = revision(expected + 1)
            value["pending"] = dict(request, requestedRevision=value["revision"])
            self._write_at(fd, value)
            return value

    def confirm(self, request, expected_revision):
        expected = revision(expected_revision)
        request = self._request(request)
        with self._locked() as fd:
            value = self._read_at(fd)
            if value["pending"] is None and self._same(value["applied"], request) and value["applied"]["requestedRevision"] == expected:
                os.fsync(fd)
                return value
            if value["revision"] != expected or not self._same(value["pending"], request):
                raise SelectionConflict("application acknowledgement is stale, superseded or from another producer")
            value["revision"] = revision(expected + 1)
            value["applied"] = dict(value["pending"], appliedRevision=value["revision"], appliedAtUnixNs=str(time.time_ns()))
            value["pending"] = None
            self._write_at(fd, value)
            return value
