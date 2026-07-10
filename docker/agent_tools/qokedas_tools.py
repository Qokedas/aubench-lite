"""Standalone submission tooling available inside the agent container.

These tools validate the delivery contract only: they run the program on the visible
capture and check the output tensor's shape, dtype, and value constraints. They do
not measure quality — that is the submitter's own problem, as it would be for a real
client deliverable.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import stat
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

BLOCK_LENGTH = 8_192
BAND_COUNT = 11
GRID_HEIGHT = 24
GRID_WIDTH = 32
MICROPHONE_COUNT = 1_024
MAX_SUBMISSION_BYTES = 100 * 1024 * 1024
MAX_SUBMISSION_FILES = 1_000
DEFAULT_CAPTURE = "/data/capture"


class CheckError(ValueError):
    pass


def validate_submission(root: Path) -> dict:
    if not root.is_dir() or root.is_symlink():
        raise CheckError(f"submission must be a real directory: {root}")
    entrypoint = root / "solution.py"
    if not entrypoint.is_file() or entrypoint.is_symlink():
        raise CheckError("submission must contain a regular solution.py")
    files = []
    total = 0
    root_real = root.resolve()
    for current, directory_names, file_names in os.walk(root, followlinks=False):
        current_path = Path(current)
        for name in directory_names:
            if (current_path / name).is_symlink():
                raise CheckError(f"symlink directory is forbidden: {current_path / name}")
        for name in file_names:
            value = current_path / name
            info = value.lstat()
            if not stat.S_ISREG(info.st_mode) or value.is_symlink():
                raise CheckError(f"only regular files are allowed: {value}")
            if root_real not in value.resolve().parents:
                raise CheckError(f"submission path escapes root: {value}")
            if info.st_nlink != 1:
                raise CheckError(f"hard-linked file is forbidden: {value}")
            total += info.st_size
            if total > MAX_SUBMISSION_BYTES:
                raise CheckError(f"submission exceeds {MAX_SUBMISSION_BYTES} bytes")
            digest = hashlib.sha256(value.read_bytes()).hexdigest()
            files.append({"path": value.relative_to(root).as_posix(), "sha256": digest})
            if len(files) > MAX_SUBMISSION_FILES:
                raise CheckError(f"submission exceeds {MAX_SUBMISSION_FILES} files")
    files.sort(key=lambda item: item["path"])
    manifest = hashlib.sha256()
    for item in files:
        manifest.update(item["path"].encode() + b"\0" + item["sha256"].encode() + b"\n")
    return {"file_count": len(files), "total_bytes": total, "sha256": manifest.hexdigest()}


def expected_frames(capture: Path) -> int:
    sound = capture / "sound"
    if not sound.is_file() or not (capture / "container.json").is_file():
        raise CheckError(f"capture must contain sound and container.json: {capture}")
    samples, remainder = divmod(sound.stat().st_size, 4 * MICROPHONE_COUNT)
    if remainder:
        raise CheckError(f"sound stream size is not a whole number of frames: {sound}")
    if samples < BLOCK_LENGTH:
        raise CheckError(f"capture has {samples} samples; at least {BLOCK_LENGTH} required")
    return int(samples) // BLOCK_LENGTH


def run_and_validate(root: Path, capture: Path, timeout: int = 240) -> dict:
    frames = expected_frames(capture)
    with tempfile.TemporaryDirectory(prefix="qokedas-check-") as temporary:
        output = Path(temporary) / "maps.npy"
        result = subprocess.run(
            [
                sys.executable,
                str(root / "solution.py"),
                "--capture",
                str(capture),
                "--output",
                str(output),
            ],
            cwd=root,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
        if result.returncode:
            raise CheckError(
                f"solution exited {result.returncode}: "
                f"{result.stdout[-4096:]} {result.stderr[-4096:]}"
            )
        maps = np.load(output, allow_pickle=False)
        expected = (frames, BAND_COUNT, GRID_HEIGHT, GRID_WIDTH)
        if maps.shape != expected:
            raise CheckError(f"maps shape must be {expected}, got {maps.shape}")
        if maps.dtype != np.dtype("float32"):
            raise CheckError(f"maps dtype must be float32, got {maps.dtype}")
        if not np.isfinite(maps).all():
            raise CheckError("maps contains NaN or infinity")
        if np.any(maps < 0):
            raise CheckError("maps contains negative power")
        return {"output_shape": list(maps.shape), "dtype": str(maps.dtype)}


def _arguments(description: str) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("submission", nargs="?", default="/workspace/submission")
    parser.add_argument("--capture", default=DEFAULT_CAPTURE)
    return parser.parse_args()


def check_main() -> None:
    arguments = _arguments("Validate a submission against the delivery contract")
    root = Path(arguments.submission)
    manifest = validate_submission(root)
    execution = run_and_validate(root, Path(arguments.capture))
    print(json.dumps({"status": "ok", "submission": manifest, "run": execution}, indent=2))


def submit_main() -> None:
    arguments = _arguments("Validate the final submission; the host performs the freeze")
    manifest = validate_submission(Path(arguments.submission))
    print(json.dumps({"status": "ready_for_host_freeze", "manifest": manifest}, indent=2))
