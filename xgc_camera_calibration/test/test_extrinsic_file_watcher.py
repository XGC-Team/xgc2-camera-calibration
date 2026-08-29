#!/usr/bin/env python3

import tempfile
import unittest
from pathlib import Path

from xgc_camera_calibration.extrinsic_file_watcher import ExtrinsicDirectoryWatcher


class ExtrinsicDirectoryWatcherTest(unittest.TestCase):
    def test_reports_latest_existing_timestamped_file_once(self):
        with tempfile.TemporaryDirectory() as directory:
            older = Path(directory) / "extrinsics-20260830T010000.000000Z.yaml"
            newer = Path(directory) / "extrinsics-20260830T020000.000000Z.yaml"
            older.write_text("first")
            newer.write_text("second")
            watcher = ExtrinsicDirectoryWatcher(directory)

            self.assertEqual(watcher.next_revision().path, newer.resolve())
            self.assertIsNone(watcher.next_revision())

    def test_ignores_prestart_results_until_a_new_timestamped_file_arrives(self):
        with tempfile.TemporaryDirectory() as directory:
            stale = Path(directory) / "extrinsics-20260830T010000.000000Z.yaml"
            solved = Path(directory) / "extrinsics-20260830T020000.000000Z.yaml"
            stale.write_text("stale")
            watcher = ExtrinsicDirectoryWatcher(directory, require_update=True)

            self.assertIsNone(watcher.next_revision())
            solved.write_text("solved")
            self.assertEqual(watcher.next_revision().path, solved.resolve())
            self.assertIsNone(watcher.next_revision())

    def test_accepts_first_result_created_after_start_and_ignores_alias(self):
        with tempfile.TemporaryDirectory() as directory:
            alias = Path(directory) / "extrinsics.yaml"
            result = Path(directory) / "extrinsics-20260830T010000.000000Z.yaml"
            alias.write_text("not part of the result contract")
            watcher = ExtrinsicDirectoryWatcher(directory, require_update=True)

            self.assertIsNone(watcher.next_revision())
            result.write_text("solved")
            self.assertEqual(watcher.next_revision().path, result.resolve())


if __name__ == "__main__":
    unittest.main()
