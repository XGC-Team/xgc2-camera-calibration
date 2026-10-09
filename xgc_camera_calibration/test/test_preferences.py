"""Storage CAS/receipt domain and the real native Client -> product HTTP seam."""
import copy
from dataclasses import asdict
import json
import tempfile
import unittest
import urllib.error
import urllib.request
from pathlib import Path

from xgc2_xrpc import Client, Endpoint, Fault, Host, Limits, Runtime, ServiceRef, load_bootstrap_input
from xgc_camera_calibration.preferences import NativeStorageCalls, PreferenceError, PreferencesDomain, SCHEMA, create_preferences
from xgc_camera_calibration.web_service import CalibrationHttpServer

SCOPE = {"namespace": "camera-calibration", "user": "user-one", "workspace": "workspace-one"}


class StorageFixture:
    """Finite formal-contract double; it does not claim deployment auth/storage durability."""
    def __init__(self):
        self.revision, self.skin, self.calls, self.receipts = 0, None, [], {}
        self.lost, self.bad = False, None

    def token(self):
        return {"database_id": "db-one", "schema": SCHEMA, "revision": str(self.revision)}

    def __call__(self, path, value, *, request_id):
        self.calls.append((path, copy.deepcopy(value), request_id))
        if value["scope"] != SCOPE:
            raise PreferenceError("permission_denied", "scope denied", 403)
        if path == "/v1/snapshot":
            record = {"key": "appearance", "version": str(self.revision)}
            if self.skin is None:
                record["missing"] = True
            else:
                record["data"] = {"skin": self.skin}
            return {"scope": SCOPE, "token": self.token(), "results": [{"collection": "preferences", "records": [record]}]}
        if path == "/v1/receipt":
            if value["request_id"] not in self.receipts:
                raise PreferenceError("not_found", "receipt absent", 404)
            return copy.deepcopy(self.receipts[value["request_id"]])
        if path != "/v1/batch":
            raise AssertionError(path)
        assert request_id == value["request_id"]
        if value["expected"] != self.token() or value["mutations"][0]["expected_version"] != str(self.revision):
            raise PreferenceError("conflict", "CAS differs", 409)
        self.revision += 1
        self.skin = value["mutations"][0]["data"]["skin"]
        receipt = {"request_id": request_id, "digest": "a" * 64, "token": self.token(), "durability": "sqlite-full",
            "committed_at": "2026-10-09T01:00:00.123456789Z", "expires_at": "2026-10-16T01:00:00.123456789Z",
            "versions": [{"collection": "preferences", "key": "appearance", "version": str(self.revision)}]}
        self.receipts[request_id] = copy.deepcopy(receipt)
        if self.bad:
            self.bad(receipt)
        if self.lost:
            raise PreferenceError("unavailable", "reply lost", outcome="outcome_unknown", request_id=request_id)
        return receipt


def plan(value, skin="light", identity="request-one"):
    return {"skin": skin, "expected_version": value["version"], "expected": value["token"], "request_id": identity}


class PreferenceDomainTest(unittest.TestCase):
    def setUp(self):
        self.storage = StorageFixture()
        self.domain = PreferencesDomain(self.storage, SCOPE)

    def test_never_created_default_and_commit_exact_cas_full_receipt(self):
        initial = self.domain.get()
        self.assertEqual((initial["skin"], initial["version"]), ("dark", "0"))
        saved = self.domain.put(plan(initial))
        self.assertEqual((saved["skin"], saved["version"], saved["durability"]), ("light", "1", "sqlite-full"))
        self.assertEqual(saved["committed"]["skin"], "light")
        batch = self.storage.calls[-1]
        self.assertEqual(batch[2], batch[1]["request_id"])
        self.assertEqual(batch[1]["mutations"], [{"collection": "preferences", "key": "appearance", "expected_version": "0", "data": {"skin": "light"}}])

    def test_conflict_does_not_publish_or_replace_current_authority(self):
        initial = self.domain.get()
        self.storage.revision, self.storage.skin = 1, "dark"
        with self.assertRaises(PreferenceError) as error:
            self.domain.put(plan(initial))
        self.assertEqual(error.exception.code, "conflict")
        self.assertEqual(self.domain._cached, initial)

    def test_unknown_commit_queries_original_receipt_and_never_replays_batch(self):
        initial = self.domain.get()
        self.storage.lost = True
        with self.assertRaises(PreferenceError):
            self.domain.put(plan(initial))
        self.assertEqual(self.domain._cached, initial)
        receipt = self.storage.receipts.pop("request-one")
        with self.assertRaises(PreferenceError) as error:
            self.domain.put(plan(initial))
        self.assertEqual(error.exception.outcome, "outcome_unknown")
        self.assertEqual(sum(path == "/v1/batch" for path, _, _ in self.storage.calls), 1)
        self.storage.receipts["request-one"] = receipt
        self.assertEqual(self.domain.resolve("request-one")["version"], "1")
        self.assertEqual(sum(path == "/v1/batch" for path, _, _ in self.storage.calls), 1)

    def test_old_receipt_keeps_new_authority_and_separates_accurate_commit(self):
        first = self.domain.put(plan(self.domain.get()))
        second = self.domain.put(plan(first, "dark", "request-two"))
        recovered = self.domain.resolve("request-one")
        self.assertEqual((recovered["skin"], recovered["version"]), ("dark", "2"))
        self.assertEqual((recovered["committed"]["skin"], recovered["committed"]["version"]), ("light", "1"))
        self.assertEqual(self.domain._cached["token"], second["token"])

    def test_incomplete_nonfull_or_cross_revision_receipts_do_not_publish(self):
        changes = [lambda r: r.pop("digest"), lambda r: r.pop("committed_at"), lambda r: r.pop("expires_at"),
            lambda r: r.update(durability="memory"), lambda r: r["token"].update(revision="10")]
        for change in changes:
            with self.subTest(change=change):
                storage = StorageFixture()
                domain = PreferencesDomain(storage, SCOPE)
                initial = domain.get()
                storage.bad = change
                with self.assertRaises(PreferenceError) as error:
                    domain.put(plan(initial))
                self.assertEqual(error.exception.outcome, "outcome_unknown")
                self.assertEqual(domain._cached, initial)

    def test_identifier_revision_and_malformed_snapshot_fail_closed(self):
        with self.assertRaises(ValueError):
            PreferencesDomain(self.storage, {**SCOPE, "user": "user:other"})
        initial = self.domain.get()
        for bad in [{**plan(initial), "request_id": "bad:id"}, {**plan(initial), "expected_version": "9223372036854775808"}]:
            with self.assertRaises(PreferenceError):
                self.domain.put(bad)
        self.assertFalse(any(path == "/v1/batch" for path, _, _ in self.storage.calls))
        for record in [None, {"key": "appearance", "version": "0", "missing": True, "data": {"skin": "dark"}}]:
            domain = PreferencesDomain(lambda *a, **kw: {"scope": SCOPE, "token": self.storage.token(),
                "results": [{"collection": "preferences", "records": [record]}]}, SCOPE)
            with self.assertRaises(PreferenceError) as error:
                domain.get()
            self.assertEqual(error.exception.code, "invalid_response")

    def test_storage_error_event_preserves_truth_without_memory_fallback(self):
        initial = self.domain.get()
        self.domain._next_refresh = 0
        def failed(*args, **kwargs):
            raise PreferenceError("unavailable", "storage offline")
        self.domain.call = failed
        event = self.domain.event_state()
        self.assertFalse(event["available"])
        self.assertEqual(event["snapshot"], initial)
        self.assertEqual(event["error"]["code"], "unavailable")


class NativePreferenceSeamTest(unittest.TestCase):
    def test_native_storage_client_cas_product_http_and_shared_events(self):
        with tempfile.TemporaryDirectory() as directory:
            for asset in ("index.html", "app.js", "styles.css"):
                (Path(directory) / asset).write_text("test asset")
            runtime = Runtime(blocking_workers=4, max_calls=16, max_connections=16)
            fixture = StorageFixture()
            def route_call(route):
                def call(context, value):
                    try:
                        return fixture(route, value, request_id=context.request_id)
                    except PreferenceError as error:
                        raise Fault(error.code, str(error), error.status)
                return call
            path = str(Path(directory) / "storage.sock")
            reference = ServiceRef("calibration-one", "xgc2.storage.v1.Storage", "1", "storage-1", "http.v1", Endpoint("unix", path))
            host = Host(path, {("POST", route): route_call(route) for route in ("/v1/snapshot", "/v1/batch", "/v1/receipt")},
                        runtime=runtime, instance_id=reference.instance_id, limits=Limits()).start()
            root = Path(directory)
            token = root / "token"
            token.write_text("fixture-authorization")
            token.chmod(0o600)
            bootstrap_file = root / "bootstrap.json"
            bootstrap_file.write_text(json.dumps({"schema_version": 1, "binding": {
                "schema_version": 1, "target_id": "calibration-one",
                "service": "xgc2.calibration.v1.Calibration", "api_version": "1", "profile": "http.v1",
                "endpoint": {"kind": "unix", "address": str(root / "calibration.sock")},
                "runtime_grant": "fixture-runtime", "authentication": "local_private",
                "secret_handles": {}, "storage_grants": ["fixture-preferences"]},
                "grants": {"fixture-authorization": {"kind": "bearer", "token_file": str(token)}},
                "application": {"storage": {"grant": "fixture-preferences", "authorization": "fixture-authorization",
                    "reference": asdict(reference), "scope": SCOPE}}}))
            bootstrap_file.chmod(0o600)
            startup = load_bootstrap_input(str(bootstrap_file), role="client")
            domain, client = create_preferences(startup, runtime=runtime)
            server = CalibrationHttpServer(("127.0.0.1", 0), object(), directory, preferences=domain, runtime=runtime, frame_ancestors="'self'").start()
            base = "http://127.0.0.1:" + str(server.server_address[1])
            try:
                with urllib.request.urlopen(base + "/api/v1/preferences", timeout=2) as response:
                    initial = json.load(response)
                request = urllib.request.Request(base + "/api/v1/preferences", data=json.dumps(plan(initial)).encode(),
                    method="PUT", headers={"Content-Type": "application/json"})
                with urllib.request.urlopen(request, timeout=2) as response:
                    self.assertEqual(json.load(response)["committed"]["skin"], "light")
                with urllib.request.urlopen(base + "/api/v1/events", timeout=2) as response:
                    while True:
                        line = response.readline()
                        if line.startswith(b"data: "):
                            event = json.loads(line[6:])
                            break
                self.assertTrue(event["preferences"]["available"])
                self.assertEqual(event["preferences"]["snapshot"]["skin"], "light")
            finally:
                server.close()
                client.close()
                host.close()
                runtime.close()


if __name__ == "__main__":
    unittest.main()
