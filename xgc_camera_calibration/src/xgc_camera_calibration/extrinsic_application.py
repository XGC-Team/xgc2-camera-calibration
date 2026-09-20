"""One Run's publication handshake with the shared applied-selection owner.

The ROS parameter is a rendezvous projection, never selection authority. A
publisher consumes only requests addressed to its exact process epoch. Global
applied changes do not alter another Run's frozen camera pose.
"""

import argparse
import copy
import hashlib
import uuid

from .extrinsic_resolver import decode_frozen, encode_frozen, resolve_selection
from .extrinsic_selection import (
    SelectionConflict, SelectionStore, identifier, read_version, strict_json,
    validate_roles,
)
from .solver import CalibrationError


def parse_frame_roles(payload):
    if not isinstance(payload, str) or len(payload.encode("utf-8")) > 1024:
        raise CalibrationError("frame roles JSON exceeds its byte limit")
    return validate_roles(strict_json(payload))


def application_arguments(argv, with_state_parameter=False):
    parser = argparse.ArgumentParser()
    parser.add_argument("--resolved-extrinsic-json", required=True)
    parser.add_argument("--frame-roles-json", required=True)
    if with_state_parameter:
        parser.add_argument("--application-state-param", required=True)
    return parser.parse_args(argv)


def request_from_record(record):
    return {key: copy.deepcopy(record[key]) for key in (
        "applicationId", "candidateId", "result", "producer", "targetCoordinates")}


class ExtrinsicApplication:
    """Serial publisher-side state. Call tick only from the publication loop."""

    def __init__(self, root, camera_name, frozen_json, frame_roles, project, epoch=None):
        self.roles = validate_roles(frame_roles)
        self.frozen = decode_frozen(frozen_json, camera_name, self.roles)
        self.store = SelectionStore(root, camera_name, self.roles)
        self.producer = {"resolutionId": self.frozen["resolutionId"],
                         "instanceEpoch": identifier(epoch or uuid.uuid4().hex, "producer epoch")}
        self.project = project
        self.active = {"origin": "initial", "status": self.frozen["status"]}
        self.ready = False
        self._published = None

    def state(self):
        return {"schemaVersion": 1, "cameraName": self.frozen["cameraName"],
                **self.producer, "ready": self.ready, "active": copy.deepcopy(self.active)}

    def initial_published(self):
        self.ready = True
        self.project(self.state())

    def tick(self, publish):
        if not self.ready:
            raise CalibrationError("initial camera transform has not been published")
        if self._published is not None:
            self._confirm()
            return
        pending = self.store.read()["pending"]
        if (pending is None or pending["producer"] != self.producer
                or pending["targetCoordinates"] != self.frozen["targetCoordinates"]):
            return
        request = request_from_record(pending)
        resolved = resolve_selection(self.store.root, self.frozen["cameraName"],
            {"mode": "version", "result": request["result"]},
            self.frozen["targetCoordinates"], self.roles,
            resolution_id=self.producer["resolutionId"])
        resolved = decode_frozen(encode_frozen(resolved), self.frozen["cameraName"], self.roles)
        # The callback must publish the complete chain before returning. Failed
        # sends never advance the global applied owner.
        publish(resolved)
        self._published = (request, pending["requestedRevision"])
        self.active = {"origin": "application", "status": "published",
                       "applicationId": request["applicationId"],
                       "requestedRevision": pending["requestedRevision"]}
        self.project(self.state())
        self._confirm()

    def _confirm(self):
        request, revision = self._published
        try:
            document = self.store.confirm(request, revision)
        except Exception as error:
            self.active["status"] = "confirmation_error"
            self.active["error"] = str(error)
            # An exact CAS conflict cannot be retried into a newer request. Keep
            # the local publish fact, then admit a subsequent addressed request.
            if isinstance(error, SelectionConflict):
                self._published = None
            self.project(self.state())
            raise
        self.active["status"] = "applied"
        self.active["appliedRevision"] = document["applied"]["appliedRevision"]
        self.active.pop("error", None)
        self._published = None
        self.project(self.state())


class SavedExtrinsicApplication:
    """A saved candidate's fixed request/CAS, including durability retries."""

    def __init__(self, root, camera_name, frozen_json, frame_roles, read_producer):
        self.roles = validate_roles(frame_roles)
        self.frozen = decode_frozen(frozen_json, camera_name, self.roles)
        self.store = SelectionStore(root, camera_name, self.roles)
        self.read_producer = read_producer
        self.request = None
        self.expected_revision = None

    def stage(self, path, source_mode, candidate_id):
        if self.request is None:
            producer = self.read_producer()
            if (not isinstance(producer, dict) or producer.get("schemaVersion") != 1
                    or producer.get("ready") is not True
                    or producer.get("cameraName") != self.frozen["cameraName"]
                    or producer.get("resolutionId") != self.frozen["resolutionId"]):
                return {"status": "unavailable"}
            epoch = identifier(producer.get("instanceEpoch"), "producer epoch")
            result = {"sourceMode": source_mode, "fileName": path.name,
                      "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
            document = read_version(self.store.root, self.frozen["cameraName"], result, self.roles)
            if document.get("metadata", {}).get("candidate_id") != candidate_id:
                raise CalibrationError("saved candidate identity does not match")
            revision = self.store.read()["revision"]
            self.request = {"applicationId": uuid.uuid4().hex, "candidateId": candidate_id,
                "result": result, "producer": {"resolutionId": self.frozen["resolutionId"],
                                                "instanceEpoch": epoch},
                "targetCoordinates": self.frozen["targetCoordinates"]}
            self.expected_revision = revision
        # stage owns the retry check under its file lock. Confirmation may race
        # this call, so its exact idempotent receipt can already be applied.
        current = self.store.stage(self.request, self.expected_revision)
        if self._matches(current["applied"]) and current["pending"] is None:
            return {"status": "applied", "applicationId": self.request["applicationId"],
                    "appliedRevision": current["applied"]["appliedRevision"]}
        return {"status": "pending", "applicationId": self.request["applicationId"],
                "requestedRevision": current["pending"]["requestedRevision"]}

    def _matches(self, record):
        return record is not None and request_from_record(record) == self.request

    def status(self):
        if self.request is None:
            return {"status": "unavailable"}
        current = self.store.read()
        identity = {"applicationId": self.request["applicationId"]}
        if self._matches(current["applied"]):
            return {**identity, "status": "applied",
                    "appliedRevision": current["applied"]["appliedRevision"]}
        if self._matches(current["pending"]):
            return {**identity, "status": "pending",
                    "requestedRevision": current["pending"]["requestedRevision"]}
        # A pre-stage I/O failure keeps the same request/CAS retryable. It is
        # superseded only when the persistent revision has actually advanced.
        if current["revision"] == self.expected_revision:
            return {**identity, "status": "unavailable"}
        return {**identity, "status": "conflict"}
