"""Detect immutable, timestamped camera-extrinsic calibration assets."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

from .solver import extrinsic_selection_path, load_extrinsic_selection


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
    document: Optional[Dict[str, Any]] = None


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


class ExtrinsicSelectionWatcher:
    """Yield only the exact immutable result named by the shared pointer."""

    def __init__(
        self,
        calibration_root: Union[str, Path],
        calibration_mode: str,
        camera_name: str,
        require_update: bool = False,
    ):
        self.calibration_root = str(calibration_root)
        self.calibration_mode = str(calibration_mode)
        self.camera_name = str(camera_name)
        self.pointer = extrinsic_selection_path(
            self.calibration_root, self.calibration_mode, self.camera_name
        )
        self._seen: Dict[Tuple[FileFingerprint, str, FileFingerprint], None] = {}
        if require_update:
            revision = self._revision()
            if revision is not None:
                self._seen[self._key(revision)] = None

    def _revision(self) -> Optional[ExtrinsicRevision]:
        selected = load_extrinsic_selection(
            self.calibration_root, self.calibration_mode, self.camera_name
        )
        if selected is None:
            return None
        path, document, _selection = selected
        fingerprint = file_fingerprint(path)
        if fingerprint is None:
            return None
        return ExtrinsicRevision(path=path, fingerprint=fingerprint, document=document)

    def _key(
        self, revision: ExtrinsicRevision
    ) -> Tuple[FileFingerprint, str, FileFingerprint]:
        pointer_fingerprint = file_fingerprint(self.pointer)
        if pointer_fingerprint is None:
            raise RuntimeError("extrinsic selection pointer disappeared")
        return pointer_fingerprint, str(revision.path), revision.fingerprint

    def next_revision(self) -> Optional[ExtrinsicRevision]:
        revision = self._revision()
        if revision is None:
            return None
        key = self._key(revision)
        if key in self._seen:
            return None
        self._seen[key] = None
        return revision
