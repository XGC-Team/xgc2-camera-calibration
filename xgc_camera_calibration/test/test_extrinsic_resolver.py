"""Frozen pose consumption is independent of later global selection changes."""
import copy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from unittest.mock import patch
import unittest

import yaml

from xgc_camera_calibration.extrinsic_resolver import decode_frozen, encode_frozen, resolve_selection
from xgc_camera_calibration.extrinsic_selection import CalibrationError, SelectionStore
from test_extrinsic_selection import ROLES, TARGET, SelectionFixture

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "resolve_extrinsic.py"


class FrozenResolverTest(SelectionFixture):
    def resolve(self, choice, target=None):
        return resolve_selection(str(self.root), "usb_cam", choice, target or TARGET, ROLES, "resolution-test")

    def test_auto_without_applied_is_uncalibrated_not_pending_or_legacy_latest(self):
        result = self.resolve({"mode": "auto"})
        self.assertEqual(result["status"], "uncalibrated")
        self.assertEqual(result["globalAppliedRevision"], 0)
        self.assertNotIn("resolvedOpticalPose", result)
        self.assertFalse((self.root / "selections").exists())
        self.store.stage(self.request(), 0)
        self.assertEqual(self.resolve({"mode": "auto"}), result)

    def test_cross_mode_applied_version_rebases_once_and_freezes_independently(self):
        self.store.stage(self.request(), 0)
        self.store.confirm(self.request(), 1)
        original = self.path.read_bytes()
        first = self.resolve({"mode": "auto"})
        self.assertEqual(first["globalAppliedRevision"], 2)
        self.assertEqual(first["result"]["sourceMode"], "phy")
        self.assertEqual(first["originalOpticalPose"]["translation"], [1, 2, 3])
        self.assertEqual(first["resolvedOpticalPose"]["translation"], [4, 5, 6])
        frozen = encode_frozen(first)
        sim, _ = self.version("sim", candidate="candidate-sim")
        next_request = self.request("application-sim", result=sim, candidate="candidate-sim")
        self.store.stage(next_request, 2)
        self.assertEqual(self.resolve({"mode": "auto"}), first)
        self.store.confirm(next_request, 3)
        self.assertEqual(self.resolve({"mode": "auto"})["result"]["sourceMode"], "sim")
        self.assertEqual(decode_frozen(frozen, "usb_cam", ROLES), first)
        self.assertEqual(self.path.read_bytes(), original)
        # A consumer validates the frozen receipt, never current files or state.
        self.pointer().write_bytes(b'broken')
        self.path.unlink()
        self.assertEqual(decode_frozen(frozen, "usb_cam", ROLES), first)

    def test_manual_version_does_not_read_corrupt_global_selection(self):
        self.pointer().parent.mkdir(parents=True)
        self.pointer().write_bytes(b'broken')
        result = self.resolve({"mode": "version", "result": self.result}, {"frame": "world", "worldOffset": [4, 5, 6]})
        self.assertEqual(result["originalOpticalPose"], result["resolvedOpticalPose"])
        self.assertNotIn("globalAppliedRevision", result)
        self.assertEqual(self.pointer().read_bytes(), b'broken')
        with self.assertRaises(CalibrationError):
            self.resolve({"mode": "auto"})

    def test_manual_optical_pose_has_explicit_saved_origin(self):
        choice = {"mode": "pose", "pose": {"convention": "world_T_camera_optical", "translation": [1, 2, 3],
                  "quaternionXyzw": [0, 0, 0, 1], "coordinates": {"schemaVersion": 1, "kind": "experiment-world",
                  "frame": "world", "savedWorldOffset": [4, 5, 6]}}}
        before = copy.deepcopy(choice)
        result = self.resolve(choice)
        self.assertEqual(result["status"], "manual")
        self.assertEqual(result["resolvedOpticalPose"]["translation"], [4, 5, 6])
        self.assertEqual(choice, before)
        self.assertFalse((self.root / "selections").exists())
        for quaternion in [[0, 0, 0, 0], [0, 0, 0, 2], [0, 0, 0, True], [float('nan'), 0, 0, 1]]:
            invalid = copy.deepcopy(choice)
            invalid["pose"]["quaternionXyzw"] = quaternion
            with self.subTest(quaternion=quaternion), self.assertRaises(CalibrationError):
                self.resolve(invalid)

    def test_choices_are_mutually_exclusive_and_offsets_must_be_finite(self):
        for choice in [{"mode": "auto", "result": self.result}, {"mode": "version"},
                       {"mode": "pose", "result": self.result}, {"mode": "latest"},
                       {"mode": "auto", "unused": "x" * 3072}]:
            with self.subTest(choice=choice.get("mode")), self.assertRaises(CalibrationError):
                self.resolve(choice)
        for offset in [[True, 0, 0], [float("inf"), 0, 0], [10 ** 400, 0, 0], [0, 0], ["0", 0, 0]]:
            with self.assertRaises(CalibrationError):
                self.resolve({"mode": "version", "result": self.result}, {"frame": "world", "worldOffset": offset})

    def test_legacy_has_unknown_provenance_and_cannot_be_rebased(self):
        legacy, path = self.version(name="extrinsics-20260920T010203.000002Z.yaml", provenance=False)
        original = path.read_bytes()
        value = self.resolve({"mode": "version", "result": legacy}, {"frame": "world", "worldOffset": [0, 0, 0]})
        self.assertIsNone(value["sourceCoordinates"])
        self.assertEqual(value["resolvedOpticalPose"], value["originalOpticalPose"])
        with self.assertRaisesRegex(CalibrationError, "provenance"):
            self.resolve({"mode": "version", "result": legacy})
        self.assertEqual(path.read_bytes(), original)

    def test_digest_camera_mode_and_frame_role_are_all_required(self):
        good = self.path.read_bytes()
        invalid_ref = dict(self.result, sha256="0" * 64)
        with self.assertRaisesRegex(CalibrationError, "digest"):
            self.resolve({"mode": "version", "result": invalid_ref})
        for field, value in [("camera_name", "other"), ("calibration_mode", "sim"), ("parent_frame", "map"),
                             ("child_frame", "other_optical"), ("frame_convention", "camera_T_world")]:
            doc = yaml.safe_load(good)
            doc[field] = value
            self.path.write_text(yaml.safe_dump(doc))
            ref = dict(self.result, sha256=hashlib.sha256(self.path.read_bytes()).hexdigest())
            with self.subTest(field=field), self.assertRaises(CalibrationError):
                self.resolve({"mode": "version", "result": ref})
        self.path.write_bytes(good)

    def test_result_pose_cannot_coerce_booleans_or_numeric_strings(self):
        original = self.path.read_bytes()
        for value in (True, "1.0", float("nan")):
            document = yaml.safe_load(original)
            document["translation"]["x"] = value
            self.path.write_text(yaml.safe_dump(document))
            result = dict(self.result, sha256=hashlib.sha256(self.path.read_bytes()).hexdigest())
            with self.subTest(value=value), self.assertRaises(CalibrationError):
                self.resolve({"mode": "version", "result": result})

    def test_source_and_parent_symlinks_and_unsafe_identities_are_rejected(self):
        saved = self.path.with_suffix(".original")
        self.path.rename(saved)
        self.path.symlink_to(saved)
        with self.assertRaises(CalibrationError):
            self.resolve({"mode": "version", "result": self.result})
        self.path.unlink()
        saved.rename(self.path)
        mode = self.root / "phy"
        mode.rename(self.root / "real-phy")
        mode.symlink_to(self.root / "real-phy", target_is_directory=True)
        with self.assertRaises(CalibrationError):
            self.resolve({"mode": "version", "result": self.result})
        for root, camera in [("relative", "usb_cam"), (str(self.root / ".."), "usb_cam"), (str(self.root), "../camera")]:
            with self.assertRaises(CalibrationError):
                SelectionStore(root, camera, ROLES)
        alias = self.root / "root-alias"
        alias.symlink_to(self.root, target_is_directory=True)
        with self.assertRaises(CalibrationError):
            SelectionStore(str(alias), "usb_cam", ROLES)

    def test_frozen_parser_rejects_identity_drift_and_second_coordinate_conversion(self):
        result = self.resolve({"mode": "version", "result": self.result})
        for field, value in [("cameraName", "other"), ("sourceOpticalFrame", "other_optical")]:
            invalid = copy.deepcopy(result)
            invalid[field] = value
            with self.assertRaises(CalibrationError):
                decode_frozen(json.dumps(invalid), "usb_cam", ROLES)
        invalid = copy.deepcopy(result)
        invalid["resolvedOpticalPose"]["translation"][0] += 3
        with self.assertRaisesRegex(CalibrationError, "one coordinate"):
            encode_frozen(invalid)
        with self.assertRaises(CalibrationError):
            decode_frozen(" " * 3585, "usb_cam", ROLES)
        with patch("xgc_camera_calibration.extrinsic_resolver.MAX_FROZEN_BYTES", 64):
            with self.assertRaisesRegex(CalibrationError, "stdout budget"):
                encode_frozen(result)

    def test_installed_cli_stdout_is_only_compact_frozen_json_and_errors_are_stderr(self):
        env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
        env["PYTHONPATH"] = str(SCRIPT.parents[1] / "src")
        command = [sys.executable, str(SCRIPT), "--root", str(self.root), "--camera", "usb_cam",
                   "--frame-roles-json", json.dumps(ROLES), "--target-offset-json", '{"x":7,"y":8,"z":9}',
                   "--resolution-id", "resolution-cli", "--selection-json"]
        for raw in [json.dumps({"mode": "version", "result": self.result}), '{"mode":"auto"}']:
            process = subprocess.run(command + [raw], capture_output=True, text=True, env=env, timeout=10)
            self.assertEqual(process.returncode, 0, process.stderr)
            self.assertEqual(process.stderr, "")
            self.assertLessEqual(len(process.stdout.encode()), 3584)
            self.assertNotIn('\n', process.stdout)
            decode_frozen(process.stdout, "usb_cam", ROLES)
        for raw in ['{"mode":"auto","mode":"version"}', '{"mode":NaN}', '{"mode":"auto","result":{}}', ' ' * 3073]:
            process = subprocess.run(command + [raw], capture_output=True, text=True, env=env, timeout=10)
            self.assertNotEqual(process.returncode, 0)
            self.assertEqual(process.stdout, "")
            self.assertIn("Extrinsic resolution failed", process.stderr)


if __name__ == "__main__":
    unittest.main()
