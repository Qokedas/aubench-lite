"""Submission validation and immutable snapshot creation."""

from __future__ import annotations

import hashlib
import os
import shutil
import stat
from pathlib import Path
from typing import Any

from .formats import FormatError, write_json

MAX_SUBMISSION_BYTES = 100 * 1024 * 1024
MAX_SUBMISSION_FILES = 1_000


def validate_submission(root: str | Path) -> dict[str, Any]:
    root = Path(root)
    if not root.is_dir() or root.is_symlink():
        raise FormatError(f"submission must be a real directory: {root}")
    entrypoint = root / "solution.py"
    if not entrypoint.is_file() or entrypoint.is_symlink():
        raise FormatError("submission must contain a regular solution.py")
    files: list[dict[str, Any]] = []
    total = 0
    root_real = root.resolve()
    for current, directory_names, file_names in os.walk(root, followlinks=False):
        current_path = Path(current)
        for name in directory_names:
            value = current_path / name
            if value.is_symlink():
                raise FormatError(f"symlink directory is forbidden: {value}")
        for name in file_names:
            value = current_path / name
            info = value.lstat()
            if not stat.S_ISREG(info.st_mode) or value.is_symlink():
                raise FormatError(f"only regular files are allowed: {value}")
            resolved = value.resolve()
            if root_real not in resolved.parents:
                raise FormatError(f"submission path escapes root: {value}")
            if info.st_nlink != 1:
                raise FormatError(f"hard-linked file is forbidden: {value}")
            total += info.st_size
            if total > MAX_SUBMISSION_BYTES:
                raise FormatError(
                    f"submission exceeds {MAX_SUBMISSION_BYTES} bytes ({total} observed)"
                )
            relative = value.relative_to(root).as_posix()
            digest = hashlib.sha256(value.read_bytes()).hexdigest()
            files.append({"path": relative, "bytes": info.st_size, "sha256": digest})
            if len(files) > MAX_SUBMISSION_FILES:
                raise FormatError(f"submission exceeds {MAX_SUBMISSION_FILES} files")
    files.sort(key=lambda item: item["path"])
    manifest_digest = hashlib.sha256()
    for item in files:
        manifest_digest.update(item["path"].encode("utf-8"))
        manifest_digest.update(b"\0")
        manifest_digest.update(item["sha256"].encode("ascii"))
        manifest_digest.update(b"\n")
    return {
        "schema_version": "cam1k-submission-v1",
        "file_count": len(files),
        "total_bytes": total,
        "sha256": manifest_digest.hexdigest(),
        "files": files,
    }


def freeze_submission(source: str | Path, destination: str | Path) -> dict[str, Any]:
    source = Path(source)
    destination = Path(destination)
    manifest = validate_submission(source)
    if destination.exists():
        raise FileExistsError(f"refusing to replace frozen submission: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source, destination, symlinks=False)
    copied = validate_submission(destination)
    if copied["sha256"] != manifest["sha256"]:
        shutil.rmtree(destination)
        raise RuntimeError("submission changed while it was being frozen")
    # Keep the trusted freeze receipt adjacent to, not inside, the exact submitted tree.
    write_json(destination.parent / f"{destination.name}.freeze.json", manifest)
    for value in sorted(destination.rglob("*"), reverse=True):
        value.chmod(0o555 if value.is_dir() else 0o444)
    destination.chmod(0o555)
    return manifest
