"""The application pointer records acknowledged ownership, never latest files."""
import copy
import errno
import fcntl
import hashlib
import multiprocessing
import os
from pathlib import Path
import stat
import tempfile
import time
import unittest
from unittest.mock import patch

import yaml

from xgc_camera_calibration.extrinsic_selection import CalibrationError, SelectionConflict, SelectionStore

ROLES = {"parentFrame": "world", "opticalFrames": {"sim": "xgc_world_camera_optical_frame", "phy": "usb_cam_optical_frame"}}
TARGET = {"frame": "world", "worldOffset": [7., 8., 9.]}


def stage_worker(root, request, ready, start, results):
    store = SelectionStore(root, "usb_cam", ROLES)
    ready.put(True)
    if not start.wait(5):
        results.put("timeout")
        return
    try:
        results.put(("staged", store.stage(request, 0)["revision"]))
    except SelectionConflict:
        results.put(("conflict", None))


def hold_lock(path, ready, release):
    with open(path, "a") as stream:
        fcntl.flock(stream, fcntl.LOCK_EX)
        ready.set()
        release.wait(5)


class SelectionFixture(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.store = SelectionStore(str(self.root), "usb_cam", ROLES)
        self.result, self.path = self.version()

    def version(self, mode="phy", name="extrinsics-20260920T010203.000001Z.yaml", candidate="candidate-a", provenance=True):
        document = {"schema": "xgc2.camera.extrinsic.v1", "calibration_mode": mode,
                    "camera_name": "usb_cam", "frame_convention": "parent_T_camera_optical",
                    "parent_frame": "world", "child_frame": ROLES["opticalFrames"][mode],
                    "translation": {"x": 1., "y": 2., "z": 3.},
                    "quaternion_xyzw": {"x": 0., "y": 0., "z": 0., "w": 1.},
                    "metadata": {"candidate_id": candidate}}
        if provenance:
            document["metadata"]["pose_coordinates"] = {"schema_version": 1, "kind": "experiment-world",
                                                         "frame": "world", "world_offset": [4., 5., 6.]}
        path = self.root / mode / "usb_cam" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(yaml.safe_dump(document), encoding="utf-8")
        return {"sourceMode": mode, "fileName": name, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}, path

    def request(self, application="application-a", producer="instance-a", result=None, candidate="candidate-a"):
        return {"applicationId": application, "candidateId": candidate, "result": result or self.result,
                "producer": {"resolutionId": "resolution-a", "instanceEpoch": producer},
                "targetCoordinates": copy.deepcopy(TARGET)}

    def pointer(self):
        return self.root / "selections" / "usb_cam" / "extrinsic.json"


class SelectionStateTest(SelectionFixture):
    def test_absent_state_is_read_only_and_does_not_migrate_legacy(self):
        legacy = self.pointer().parent / "phy-extrinsic.json"
        legacy.parent.mkdir(parents=True)
        legacy.write_bytes(b'{"selected_at":"old-only-not-applied"}')
        before = {p.relative_to(self.root): p.read_bytes() for p in self.root.rglob("*") if p.is_file()}
        self.assertEqual(self.store.read(), {"schemaVersion": 2, "cameraName": "usb_cam", "revision": 0, "applied": None, "pending": None})
        after = {p.relative_to(self.root): p.read_bytes() for p in self.root.rglob("*") if p.is_file()}
        self.assertEqual(before, after)
        self.assertFalse((legacy.parent / ".extrinsic.lock").exists())

    def test_pending_and_applied_are_separate_and_retries_do_not_write(self):
        original = self.path.read_bytes()
        request = self.request()
        pending = self.store.stage(request, 0)
        self.assertEqual(pending["revision"], 1)
        self.assertIsNone(pending["applied"])
        self.assertEqual(pending["pending"]["requestedRevision"], 1)
        mtime = self.pointer().stat().st_mtime_ns
        self.assertEqual(self.store.stage(request, 0), pending)
        self.assertEqual(self.pointer().stat().st_mtime_ns, mtime)
        applied = self.store.confirm(request, 1)
        self.assertEqual(applied["revision"], 2)
        self.assertIsNone(applied["pending"])
        self.assertEqual(applied["applied"]["appliedRevision"], 2)
        self.assertGreater(int(applied["applied"]["appliedAtUnixNs"]), 0)
        mtime = self.pointer().stat().st_mtime_ns
        self.assertEqual(self.store.confirm(request, 1), applied)
        self.assertEqual(self.store.stage(request, 0), applied)
        self.assertEqual(self.pointer().stat().st_mtime_ns, mtime)
        self.assertEqual(self.path.read_bytes(), original)

    def test_supersession_and_foreign_producer_ack_cannot_change_applied(self):
        first, second = self.request(), self.request("application-b", "instance-b")
        self.store.stage(first, 0)
        self.store.confirm(first, 1)
        pending = self.store.stage(second, 2)
        self.assertEqual(pending["applied"]["applicationId"], "application-a")
        before = self.pointer().read_bytes()
        for request, expected in [(first, 1), (first, 3), (self.request("application-b", "wrong-instance"), 3)]:
            with self.subTest(request=request["producer"], expected=expected), self.assertRaises(SelectionConflict):
                self.store.confirm(request, expected)
            self.assertEqual(self.pointer().read_bytes(), before)
        self.assertEqual(self.store.confirm(second, 3)["applied"]["applicationId"], "application-b")
        with self.assertRaises(SelectionConflict):
            self.store.stage(first, 0)

    def test_pending_supersession_rejects_old_ack_even_for_same_result(self):
        first, second = self.request(), self.request("application-b")
        self.store.stage(first, 0)
        self.store.stage(second, 1)
        with self.assertRaises(SelectionConflict):
            self.store.confirm(first, 2)
        self.assertIsNone(self.store.read()["applied"])
        self.assertEqual(self.store.confirm(second, 2)["applied"]["applicationId"], "application-b")

    def test_candidate_and_revision_must_be_exact_before_any_state_write(self):
        for request, expected in [(self.request(candidate="another"), 0), (self.request(), True),
                                  (self.request(), -1), (dict(self.request(), extra=True), 0)]:
            with self.subTest(expected=expected), self.assertRaises(CalibrationError):
                self.store.stage(request, expected)
        self.assertFalse(self.pointer().exists())
        self.store.stage(self.request(), 0)
        with self.assertRaises(SelectionConflict):
            self.store.stage(self.request(producer="changed"), 1)

    def test_real_processes_with_same_revision_have_exactly_one_winner(self):
        ctx = multiprocessing.get_context("fork")
        ready, results, start = ctx.Queue(), ctx.Queue(), ctx.Event()
        processes = [ctx.Process(target=stage_worker, args=(str(self.root), self.request("application-" + str(i)), ready, start, results)) for i in range(4)]
        try:
            for process in processes:
                process.start()
            for _ in processes:
                self.assertTrue(ready.get(timeout=5))
            start.set()
            outcomes = [results.get(timeout=5) for _ in processes]
            for process in processes:
                process.join(5)
                self.assertEqual(process.exitcode, 0)
            self.assertEqual(outcomes.count(("staged", 1)), 1)
            self.assertEqual(outcomes.count(("conflict", None)), 3)
            self.assertEqual(self.store.read()["revision"], 1)
        finally:
            start.set()
            for process in processes:
                if process.is_alive():
                    process.terminate()
                process.join(5)
            ready.close()
            results.close()

    def test_cross_process_lock_wait_has_a_deadline(self):
        self.pointer().parent.mkdir(parents=True)
        ctx = multiprocessing.get_context("fork")
        ready, release = ctx.Event(), ctx.Event()
        process = ctx.Process(target=hold_lock, args=(str(self.pointer().parent / ".extrinsic.lock"), ready, release))
        process.start()
        try:
            self.assertTrue(ready.wait(5))
            started = time.monotonic()
            with self.assertRaisesRegex(SelectionConflict, "deadline"):
                SelectionStore(str(self.root), "usb_cam", ROLES, lock_timeout=.04).stage(self.request(), 0)
            self.assertLess(time.monotonic() - started, 1)
            self.assertFalse(self.pointer().exists())
        finally:
            release.set()
            process.join(5)
            if process.is_alive():
                process.terminate()
                process.join(5)

    def test_replace_failure_preserves_pointer_and_cleans_temporary(self):
        self.store.stage(self.request(), 0)
        before = self.pointer().read_bytes()
        with patch("xgc_camera_calibration.extrinsic_selection.os.replace", side_effect=OSError(errno.EIO, "injected write failure")):
            with self.assertRaises(CalibrationError):
                self.store.confirm(self.request(), 1)
        self.assertEqual(self.pointer().read_bytes(), before)
        self.assertEqual(list(self.pointer().parent.glob("*.tmp")), [])
        self.assertEqual(self.store.confirm(self.request(), 1)["revision"], 2)

    def test_post_replace_sync_failure_is_retryable_without_second_transition(self):
        self.store.stage(self.request(), 0)
        real_fsync = os.fsync
        def fail_directory(fd):
            if stat.S_ISDIR(os.fstat(fd).st_mode):
                raise OSError(errno.EIO, "injected directory sync failure")
            return real_fsync(fd)
        with patch("xgc_camera_calibration.extrinsic_selection.os.fsync", side_effect=fail_directory):
            with self.assertRaises(CalibrationError):
                self.store.confirm(self.request(), 1)
            # A retry must finish the directory durability barrier, not merely
            # see matching bytes and report a false successful acknowledgement.
            with self.assertRaises(CalibrationError):
                self.store.confirm(self.request(), 1)
        visible = self.store.read()
        self.assertEqual(visible["revision"], 2)
        self.assertEqual(self.store.confirm(self.request(), 1), visible)
        self.assertEqual(list(self.pointer().parent.glob("*.tmp")), [])

    def test_corrupt_state_or_source_never_falls_back(self):
        self.store.stage(self.request(), 0)
        good = self.pointer().read_bytes()
        for payload in [b'{', b'{"revision":0,"revision":1}', good.replace(b'"cameraName":"usb_cam"', b'"cameraName":"other"')]:
            self.pointer().write_bytes(payload)
            with self.assertRaises(CalibrationError):
                self.store.read()
        self.pointer().write_bytes(good)
        self.path.write_bytes(self.path.read_bytes() + b'\n# tamper\n')
        with self.assertRaisesRegex(CalibrationError, "digest"):
            self.store.read()
        with self.assertRaises(CalibrationError):
            self.store.confirm(self.request(), 1)
        self.assertEqual(self.pointer().read_bytes(), good)

    def test_byte_bounds_and_missing_root_fail_without_inventing_first_use(self):
        self.store.stage(self.request(), 0)
        with self.pointer().open("wb") as stream:
            stream.truncate(32 * 1024 + 1)
        with self.assertRaisesRegex(CalibrationError, "byte limit"):
            self.store.read()
        self.pointer().unlink()
        with self.path.open("wb") as stream:
            stream.truncate(8 * 1024 * 1024 + 1)
        with self.assertRaisesRegex(CalibrationError, "byte limit"):
            self.store.stage(self.request(), 0)
        moved = self.root.with_name(self.root.name + "-moved")
        self.root.rename(moved)
        try:
            with self.assertRaisesRegex(CalibrationError, "root is unavailable"):
                self.store.read()
        finally:
            moved.rename(self.root)

    def test_symlink_state_directory_and_lock_are_rejected(self):
        outside = self.root / "outside"
        outside.mkdir()
        selections = self.root / "selections"
        selections.symlink_to(outside, target_is_directory=True)
        with self.assertRaises(CalibrationError):
            self.store.stage(self.request(), 0)
        self.assertEqual(list(outside.iterdir()), [])
        selections.unlink()
        self.pointer().parent.mkdir(parents=True)
        lock = self.pointer().parent / ".extrinsic.lock"
        victim = outside / "victim"
        victim.write_text("unchanged")
        lock.symlink_to(victim)
        with self.assertRaises(CalibrationError):
            self.store.stage(self.request(), 0)
        lock.unlink()
        self.pointer().symlink_to(victim)
        with self.assertRaises(CalibrationError):
            self.store.read()
        with self.assertRaises(CalibrationError):
            self.store.stage(self.request(), 0)
        self.assertEqual(victim.read_text(), "unchanged")


if __name__ == "__main__":
    unittest.main()
