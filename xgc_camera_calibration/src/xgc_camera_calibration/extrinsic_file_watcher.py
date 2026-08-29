"""Detect immutable, timestamped camera-extrinsic calibration assets."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple, Union


@dataclass(frozen=True)
class FileFingerprint:
    """Filesystem identity used to distinguish newly solved calibration assets."""

    device: int
    inode: int
    size: int
    modified_ns: int


@dataclass(frozen=True)
class ExtrinsicRevision:
    """One concrete result discovered in the camera storage directory."""

    path: Path
    fingerprint: FileFingerprint


def file_fingerprint(path: Union[str, Path]) -> Optional[FileFingerprint]:
    """Return a stable fingerprint, or ``None`` while the asset is absent."""

    try:
        status = Path(path).stat()
    except FileNotFoundError:
        return None
    return FileFingerprint(
        device=int(status.st_dev),
        inode=int(status.st_ino),
        size=int(status.st_size),
        modified_ns=int(status.st_mtime_ns),
    )


class ExtrinsicDirectoryWatcher:
    """Yield the latest unseen ``extrinsics-<UTC>.yaml`` result once."""

    def __init__(self, directory: Union[str, Path], require_update: bool = False):
        self.directory = Path(directory)
        self._seen: Dict[Tuple[str, FileFingerprint], None] = {}
        if require_update:
            self._seen.update((key, None) for key in self._revisions())

    def _revisions(self) -> Dict[Tuple[str, FileFingerprint], ExtrinsicRevision]:
        if not self.directory.is_dir():
            return {}
        expected_parent = self.directory.resolve()
        revisions: Dict[Tuple[str, FileFingerprint], ExtrinsicRevision] = {}
        for candidate in self.directory.glob("extrinsics-*.yaml"):
            if candidate.is_symlink():
                continue
            try:
                path = candidate.resolve(strict=True)
            except OSError:
                continue
            if path.parent != expected_parent or not path.is_file():
                continue
            fingerprint = file_fingerprint(path)
            if fingerprint is None:
                continue
            key = (str(path), fingerprint)
            revisions[key] = ExtrinsicRevision(path=path, fingerprint=fingerprint)
        return revisions

    def next_revision(self) -> Optional[ExtrinsicRevision]:
        """Return the newest result not present at the previous observation."""

        revisions = self._revisions()
        unseen = [revision for key, revision in revisions.items() if key not in self._seen]
        if not unseen:
            return None
        self._seen.update((key, None) for key in revisions)
        return max(
            unseen,
            key=lambda revision: (
                revision.fingerprint.modified_ns,
                revision.path.name,
            ),
        )
