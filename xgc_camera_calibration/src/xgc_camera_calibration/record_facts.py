"""Camera publication facts for Record; not an extrinsic-selection authority.

The caller invokes ``applied`` only after BOTH TF sends return successfully,
using the same ROS stamp and transform values. A fact describes publication,
not an independent acknowledgement by every TF consumer or SelectionStore.
"""

from collections import deque
import copy
import hashlib
import json
import time


class AppliedTransformFacts:
    TOPIC = "/xgc/record_facts"
    MAX_PENDING = 64

    def __init__(self, source, emit, warn, clock=None):
        self.source = copy.deepcopy(source)
        self.emit = emit
        self._warn = warn
        self.clock = clock or self._clock
        self.sequence = 0
        self.pending = deque()
        self.current = None
        self.digest = None
        self.stopped = False

    def warn(self, message):
        try:
            self._warn(message)
        except Exception:
            # A failing diagnostic sink cannot change camera application.
            pass

    @staticmethod
    def _clock(ros_time_ns):
        return {"rosTimeNs": str(ros_time_ns), "unixTimeNs": str(time.time_ns()),
                "monotonicTimeNs": str(time.monotonic_ns())}

    def _enqueue(self, event, values, ros_time_ns):
        self.sequence += 1
        payload = {"schemaVersion": 1, "source": self.source,
                   "sequence": self.sequence, "event": event,
                   "evidence": "producer-applied", "effectiveAt": self.clock(ros_time_ns),
                   "values": values,
                   "provenance": {"boundary": ("producer-shutdown" if event == "stopped" else "complete-tf-chain-broadcast-returned"),
                                  "appliesTo": "camera-transform-publication",
                                  "resolutionId": self.source.get("resolutionId"),
                                  "selectionConfirmation": "owned-by-SelectionStore-not-this-fact"}}
        text = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
        if len(self.pending) >= self.MAX_PENDING:
            # Do not block TF or silently renumber lost events. A recorder
            # already observing this epoch can detect the sequence gap.
            self.pending.popleft()
            self.warn("Record publication queue overflow; an application fact was lost")
        self.pending.append(text)
        self.flush()

    def applied(self, transforms, frozen, ros_time_ns):
        if self.stopped:
            return
        try:
            values = {"transforms": copy.deepcopy(transforms), "resolvedExtrinsic": copy.deepcopy(frozen),
                      "calibrated": frozen["status"] != "uncalibrated"}
            data = json.dumps(values, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
            digest = hashlib.sha256(data).hexdigest()
            if self.digest == digest:
                self.flush()
                return
            self.current = values
            self.digest = digest
            self._enqueue("applied", values, ros_time_ns)
        except Exception as exc:
            # Recording diagnostics must not turn a completed TF publication
            # into a failed application or alter the selection CAS contract.
            self.warn("Could not retain Record publication fact: " + type(exc).__name__)

    def flush(self):
        while self.pending:
            try:
                self.emit(self.pending[0])
            except Exception as exc:
                self.warn("Could not publish Record fact: " + type(exc).__name__)
                return
            self.pending.popleft()

    def stop(self, ros_time_ns):
        if self.stopped:
            return
        self.stopped = True
        if self.current is not None:
            try:
                self._enqueue("stopped", self.current, ros_time_ns)
            except Exception as exc:
                self.warn("Could not retain Record producer stop: " + type(exc).__name__)


def transform_values(transforms):
    """Copy exactly what is broadcast; no rebase, optical conversion or YAML read."""
    result = []
    for message in transforms:
        translation = message.transform.translation
        rotation = message.transform.rotation
        result.append({"parentFrame": message.header.frame_id, "childFrame": message.child_frame_id,
                       "translation": [translation.x, translation.y, translation.z],
                       "quaternionXyzw": [rotation.x, rotation.y, rotation.z, rotation.w]})
    return result
