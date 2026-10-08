"""Product-owned appearance documents over the formal Storage v1 contract.

Scope, authenticated native client and schema are supplied by the composition
owner. This domain never opens a database, parses bootstrap files or owns files.
"""
import copy
from collections import OrderedDict
from datetime import datetime
import re
import threading
import time
import uuid

NAMESPACE = "camera-calibration"
SCHEMA = "camera-calibration.preferences.v1"
COLLECTION = "preferences"
KEY = "appearance"
_ID = re.compile(r"^[A-Za-z0-9._-]{1,128}$")
_DECIMAL = re.compile(r"^(0|[1-9][0-9]{0,19})$")
_RFC3339 = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d{1,9})?Z$")


def _revision(value):
    return isinstance(value, str) and _DECIMAL.fullmatch(value) and int(value) <= 9223372036854775807


class PreferenceError(RuntimeError):
    def __init__(self, code, message, status=503, *, outcome=None, request_id=None):
        super().__init__(message)
        self.code, self.status = code, status
        self.outcome, self.request_id = outcome, request_id


def _token(value):
    if (not isinstance(value, dict) or set(value) != {"database_id", "schema", "revision"}
            or not isinstance(value["database_id"], str) or not _ID.fullmatch(value["database_id"])
            or value["schema"] != SCHEMA or not _revision(value["revision"])):
        raise PreferenceError("invalid_argument", "Current preference storage token required", 409)
    return dict(value)


class NativeStorageCalls:
    """Thin use of an owner-authenticated maintained XRPC Client.

    Credential resolution and native headers belong to the SDK composition
    root. Until that official API is supplied, no unauthenticated fallback exists.
    """
    def __init__(self, client):
        self.client = client

    def __call__(self, path, value, *, request_id):
        from xgc2_xrpc import Fault, TransportError
        try:
            return self.client.json(path, value, timeout=5.0, request_id=request_id)
        except Fault as error:
            raise PreferenceError(error.code, str(error), error.status,
                outcome="outcome_unknown" if error.code == "outcome_unknown" else None,
                request_id=request_id) from error
        except TransportError as error:
            raise PreferenceError("unavailable", "Preference storage call did not complete", 503,
                outcome="outcome_unknown" if path == "/v1/batch" and error.outcome != "not_sent" else error.outcome,
                request_id=request_id) from error
        except ValueError as error:
            raise PreferenceError("invalid_response", "Preference storage returned invalid JSON", 502,
                outcome="outcome_unknown" if path == "/v1/batch" else None,
                request_id=request_id) from error


class PreferencesDomain:
    def __init__(self, call, scope):
        if (not callable(call) or not isinstance(scope, dict) or set(scope) != {"namespace", "user", "workspace"}
                or scope["namespace"] != NAMESPACE
                or any(not isinstance(scope[name], str) or not _ID.fullmatch(scope[name]) for name in ("user", "workspace"))):
            raise ValueError("Explicit camera-calibration user/workspace storage grant required")
        self.call, self.scope = call, dict(scope)
        self.lock = threading.RLock()
        self._cached, self._error = None, None
        self._next_refresh = 0.0
        self._uncertain = {}
        self._recent = OrderedDict()

    def get(self):
        with self.lock:
            result = self.call("/v1/snapshot", {"scope": self.scope,
                "queries": [{"collection": COLLECTION, "keys": [KEY]}]}, request_id=uuid.uuid4().hex)
            if not isinstance(result, dict) or result.get("scope") != self.scope:
                raise PreferenceError("invalid_response", "Preference snapshot scope differs", 502)
            try:
                token = _token(result.get("token"))
            except PreferenceError as error:
                raise PreferenceError("invalid_response", "Preference snapshot token is invalid", 502) from error
            queries = result.get("results")
            if (not isinstance(queries, list) or len(queries) != 1 or not isinstance(queries[0], dict) or queries[0].get("collection") != COLLECTION
                    or not isinstance(queries[0].get("records"), list) or len(queries[0]["records"]) != 1):
                raise PreferenceError("invalid_response", "Preference snapshot omitted its exact record", 502)
            record = queries[0]["records"][0]
            if not isinstance(record, dict):
                raise PreferenceError("invalid_response", "Preference record must be an object", 502)
            version = record.get("version")
            if (record.get("key") != KEY or record.get("collection", COLLECTION) != COLLECTION
                    or not _revision(version)
                    or record.get("deleted") or int(version) > int(token["revision"])):
                raise PreferenceError("invalid_response", "Preference record identity/version differs", 502)
            if record.get("missing") is True and version == "0":
                if "data" in record:
                    raise PreferenceError("invalid_response", "Missing preference record contains data", 502)
                skin = "dark"  # Explicit product choice for a never-created document.
            else:
                data = record.get("data")
                if record.get("missing") or version == "0" or not isinstance(data, dict) or set(data) != {"skin"} or data["skin"] not in ("light", "dark"):
                    raise PreferenceError("invalid_response", "Stored appearance is invalid", 502)
                skin = data["skin"]
            value = {"skin": skin, "version": version, "token": token}
            self._publish(value)
            return copy.deepcopy(self._cached)

    def _publish(self, value):
        if self._cached is not None:
            old = self._cached["token"]
            if old["database_id"] != value["token"]["database_id"]:
                raise PreferenceError("conflict", "Preference database binding changed; restart required", 409)
            if int(value["token"]["revision"]) < int(old["revision"]):
                return
            if (int(value["version"]) < int(self._cached["version"])
                    or value["version"] == self._cached["version"] and value["skin"] != self._cached["skin"]):
                raise PreferenceError("invalid_response", "Preference record changed without a new version", 502)
        self._cached, self._error = copy.deepcopy(value), None
        self._next_refresh = time.monotonic() + 1.0

    def _confirm(self, result, plan):
        try:
            if (not isinstance(result, dict) or not isinstance(result.get("digest"), str)
                    or re.fullmatch(r"[0-9a-f]{64}", result["digest"]) is None
                    or any(not isinstance(result.get(name), str) or not _RFC3339.fullmatch(result[name])
                           for name in ("committed_at", "expires_at"))
                    or datetime.fromisoformat(result["expires_at"].replace("Z", "+00:00")) <=
                       datetime.fromisoformat(result["committed_at"].replace("Z", "+00:00"))):
                raise ValueError("incomplete receipt")
        except ValueError as error:
            raise PreferenceError("outcome_unknown", "Preference commit receipt is incomplete", 502,
                outcome="outcome_unknown", request_id=plan["request_id"]) from error
        if (not isinstance(result, dict) or result.get("request_id") != plan["request_id"]
                or result.get("durability") != "sqlite-full"):
            raise PreferenceError("outcome_unknown", "Preference commit has no FULL durability receipt", 502,
                outcome="outcome_unknown", request_id=plan["request_id"])
        try:
            token = _token(result.get("token"))
        except PreferenceError as error:
            raise PreferenceError("outcome_unknown", "Preference commit has an invalid token", 502,
                outcome="outcome_unknown", request_id=plan["request_id"]) from error
        versions = result.get("versions")
        if (token["database_id"] != plan["expected"]["database_id"]
                or int(token["revision"]) != int(plan["expected"]["revision"]) + 1
                or not isinstance(versions, list) or len(versions) != 1 or not isinstance(versions[0], dict)
                or versions[0].get("collection") != COLLECTION or versions[0].get("key") != KEY
                or versions[0].get("version") != token["revision"] or versions[0].get("deleted")):
            raise PreferenceError("outcome_unknown", "Preference commit receipt differs from its plan", 502,
                outcome="outcome_unknown", request_id=plan["request_id"])
        value = {"skin": plan["mutations"][0]["data"]["skin"], "version": token["revision"], "token": token}
        self._publish(value)
        self._uncertain.pop(plan["request_id"], None)
        self._recent[plan["request_id"]] = copy.deepcopy(plan)
        self._recent.move_to_end(plan["request_id"])
        while len(self._recent) > 64:
            self._recent.popitem(last=False)
        return {**copy.deepcopy(self._cached), "committed": value,
                "request_id": plan["request_id"], "durability": "sqlite-full"}

    def put(self, value):
        if (not isinstance(value, dict) or set(value) != {"skin", "expected_version", "expected", "request_id"}
                or value["skin"] not in ("light", "dark")
                or not _revision(value["expected_version"])
                or not isinstance(value["request_id"], str) or not _ID.fullmatch(value["request_id"])):
            raise PreferenceError("invalid_argument", "A revision-bound appearance plan is required", 400)
        token = _token(value["expected"])
        if int(value["expected_version"]) > int(token["revision"]):
            raise PreferenceError("invalid_argument", "Preference version exceeds its snapshot", 400)
        plan = {"scope": self.scope, "expected": token, "request_id": value["request_id"],
            "mutations": [{"collection": COLLECTION, "key": KEY,
                "expected_version": value["expected_version"], "data": {"skin": value["skin"]}}]}
        with self.lock:
            previous = self._uncertain.get(plan["request_id"], self._recent.get(plan["request_id"]))
            if previous is not None:
                if previous != plan:
                    raise PreferenceError("conflict", "Uncertain identity cannot change its plan", 409)
                return self.resolve(plan["request_id"])
            if len(self._uncertain) >= 16:
                raise PreferenceError("resource_exhausted", "Uncertain preference receipts require resolution", 429)
            try:
                result = self.call("/v1/batch", plan, request_id=plan["request_id"])
                return self._confirm(result, plan)
            except PreferenceError as error:
                if error.outcome == "outcome_unknown" or error.code == "outcome_unknown":
                    self._uncertain[plan["request_id"]] = copy.deepcopy(plan)
                raise

    def resolve(self, identity):
        with self.lock:
            plan = self._uncertain.get(identity, self._recent.get(identity))
            if plan is None:
                raise PreferenceError("not_found", "No unresolved preference identity is retained", 404)
            try:
                result = self.call("/v1/receipt", {"scope": self.scope, "request_id": identity}, request_id=uuid.uuid4().hex)
            except PreferenceError as error:
                if error.code == "not_found":
                    raise PreferenceError("outcome_unknown", "Missing receipt does not prove the earlier call cannot commit", 503,
                        outcome="outcome_unknown", request_id=identity) from error
                raise
            return self._confirm(result, plan)

    def event_state(self):
        with self.lock:
            if time.monotonic() >= self._next_refresh:
                self._next_refresh = time.monotonic() + 1.0
                try:
                    self.get()
                except PreferenceError as error:
                    self._error = {"code": error.code, "message": str(error)}
            return {"available": self._error is None and self._cached is not None,
                "snapshot": copy.deepcopy(self._cached), "error": copy.deepcopy(self._error)}
