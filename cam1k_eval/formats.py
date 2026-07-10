"""Strict, pickle-free validation and raw Sorama capture I/O."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np

from .constants import (
    BAND_COUNT,
    BLOCK_LENGTH,
    GRID_HEIGHT,
    GRID_WIDTH,
    MICROPHONE_COUNT,
    SAMPLE_RATE_HZ,
)


class FormatError(ValueError):
    """Raised when an evaluation artifact violates the benchmark contract."""


def sha256_file(path: str | Path, *, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: str | Path, *, max_bytes: int = 4_000_000) -> dict[str, Any]:
    path = Path(path)
    if path.stat().st_size > max_bytes:
        raise FormatError(f"JSON file exceeds {max_bytes} bytes: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise FormatError(f"Expected a JSON object: {path}")
    return value


def write_json(path: str | Path, value: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def load_npy(
    path: str | Path,
    *,
    max_bytes: int,
    mmap_mode: str | None = "r",
) -> np.ndarray:
    path = Path(path)
    size = path.stat().st_size
    if size > max_bytes:
        raise FormatError(f"NPY file is {size} bytes; limit is {max_bytes}: {path}")
    try:
        value = np.load(path, allow_pickle=False, mmap_mode=mmap_mode)
    except Exception as exc:  # NumPy exposes several format-specific exception types.
        raise FormatError(f"Unsafe or invalid NPY file: {path}: {exc}") from exc
    if not isinstance(value, np.ndarray) or value.dtype.hasobject:
        raise FormatError(f"Object and pickled arrays are forbidden: {path}")
    return value


def validate_audio(audio: np.ndarray, *, max_samples: int = 600_000) -> None:
    if audio.ndim != 2:
        raise FormatError(f"audio must have shape [microphone, sample], got {audio.shape}")
    if audio.shape[0] < 2 or audio.shape[0] > MICROPHONE_COUNT:
        raise FormatError(f"audio microphone count must be in [2, 1024], got {audio.shape[0]}")
    if audio.shape[1] < 256 or audio.shape[1] > max_samples:
        raise FormatError(
            f"audio sample count must be in [256, {max_samples}], got {audio.shape[1]}"
        )
    if audio.dtype not in (np.dtype("float32"), np.dtype("float64")):
        raise FormatError(f"audio dtype must be float32 or float64, got {audio.dtype}")
    if not np.isfinite(audio).all():
        raise FormatError("audio contains NaN or infinity")


def validate_microphones(microphones: np.ndarray, channel_count: int) -> None:
    if microphones.shape != (channel_count, 3):
        raise FormatError(
            f"microphones must have shape [{channel_count}, 3], got {microphones.shape}"
        )
    if microphones.dtype not in (np.dtype("float32"), np.dtype("float64")):
        raise FormatError(f"microphones dtype must be float32 or float64, got {microphones.dtype}")
    if not np.isfinite(microphones).all():
        raise FormatError("microphones contains NaN or infinity")
    aperture = np.ptp(microphones.astype(np.float64), axis=0)
    if np.max(aperture) <= 0 or np.max(aperture) > 10:
        raise FormatError(f"implausible microphone aperture in metres: {aperture.tolist()}")


def validate_grid(grid: np.ndarray, *, max_pixels: int = 16_384) -> None:
    if grid.ndim != 3 or grid.shape[2] != 3:
        raise FormatError(f"grid must have shape [height, width, 3], got {grid.shape}")
    if grid.shape[0] * grid.shape[1] > max_pixels:
        raise FormatError(f"grid has too many pixels: {grid.shape[0] * grid.shape[1]}")
    if grid.dtype not in (np.dtype("float32"), np.dtype("float64")):
        raise FormatError(f"grid dtype must be float32 or float64, got {grid.dtype}")
    if not np.isfinite(grid).all():
        raise FormatError("grid contains NaN or infinity")
    norm = np.linalg.norm(grid.astype(np.float64), axis=2)
    if not np.allclose(norm, 1.0, atol=1e-5, rtol=1e-5):
        raise FormatError("far-field grid entries must be unit direction vectors")


def expected_num_frames(num_samples: int) -> int:
    """One map per full non-overlapping block; the trailing partial block is dropped."""
    if num_samples < BLOCK_LENGTH:
        raise FormatError(f"capture has {num_samples} samples; at least {BLOCK_LENGTH} required")
    return num_samples // BLOCK_LENGTH


def expected_map_shape(num_samples: int) -> tuple[int, int, int, int]:
    return (expected_num_frames(num_samples), BAND_COUNT, GRID_HEIGHT, GRID_WIDTH)


def validate_maps(maps: np.ndarray, expected_shape: tuple[int, ...]) -> None:
    if maps.shape != expected_shape:
        raise FormatError(f"maps shape must be {expected_shape}, got {maps.shape}")
    if maps.dtype != np.dtype("float32"):
        raise FormatError(f"maps dtype must be float32, got {maps.dtype}")
    if not np.isfinite(maps).all():
        raise FormatError("maps contains NaN or infinity")
    if np.any(maps < 0):
        raise FormatError("maps contains negative power")
    expected_bytes = int(np.prod(expected_shape, dtype=np.int64)) * 4
    if maps.nbytes != expected_bytes:
        raise FormatError(f"maps has unexpected storage size: {maps.nbytes} != {expected_bytes}")


def assert_regular_file(path: str | Path) -> None:
    path = Path(path)
    info = path.lstat()
    if not path.is_file() or path.is_symlink():
        raise FormatError(f"expected a regular, non-symlink file: {path}")
    if info.st_nlink != 1:
        raise FormatError(f"hard-linked files are forbidden: {path}")
    if not os.path.realpath(path).startswith(os.path.realpath(path.parent) + os.sep):
        raise FormatError(f"path escapes its parent: {path}")


# ---------------------------------------------------------------------------
# Raw Sorama capture I/O. A capture directory holds exactly the upstream layout:
# a channel-major little-endian int32 `sound` stream plus `container.json`.
# ---------------------------------------------------------------------------


def read_capture_metadata(container_json: str | Path) -> dict[str, Any]:
    """Extract the fields the trusted tooling needs from a Sorama container."""
    container = load_json(container_json)
    try:
        sound = container["metadata"]["files"]["sound"]
        ordering = np.asarray(sound["channelOrdering"], dtype=np.int64)
        records = sound["frontEnd"]["microphonePositions"]
        positions = np.asarray([[p["X"], p["Y"], p["Z"]] for p in records], dtype=np.float64)
        rates = [float(r) for r in sound["frontEnd"]["supportedSampleRates"]]
    except (KeyError, TypeError, ValueError) as exc:
        raise FormatError(f"invalid Sorama container metadata: {container_json}: {exc}") from exc
    count = positions.shape[0]
    if positions.shape != (MICROPHONE_COUNT, 3) or ordering.shape != (count,):
        raise FormatError(
            "expected 1024 microphone positions and ordering entries, "
            f"got {positions.shape}/{ordering.shape}"
        )
    if not np.array_equal(np.sort(ordering), np.arange(count)):
        raise FormatError("channelOrdering is not a permutation")
    if len(rates) != 1:
        raise FormatError(f"expected exactly one supported sample rate, got {rates}")
    return {
        "container": container,
        "ordering": ordering,
        "positions_by_channel_id": positions,
        # Row i of the sound stream carries channel ordering[i]; this is its position.
        "positions_by_row": positions[ordering],
        "sample_rate_hz": rates[0],
    }


def open_capture(capture_root: str | Path) -> dict[str, Any]:
    """Memory-map a raw capture directory without copying the stream."""
    capture_root = Path(capture_root)
    sound_path = capture_root / "sound"
    metadata_path = capture_root / "container.json"
    if not sound_path.is_file() or not metadata_path.is_file():
        raise FormatError(f"capture must contain sound and container.json: {capture_root}")
    metadata = read_capture_metadata(metadata_path)
    size = sound_path.stat().st_size
    samples, remainder = divmod(size, 4 * MICROPHONE_COUNT)
    if remainder or samples < 256:
        raise FormatError(f"invalid raw sound stream: {size} bytes")
    sound = np.memmap(sound_path, dtype="<i4", mode="r", shape=(MICROPHONE_COUNT, samples))
    return {**metadata, "sound": sound, "num_samples": int(samples)}


def write_capture(
    capture_root: str | Path,
    sound: np.ndarray,
    container: dict[str, Any],
    ordering: np.ndarray,
) -> None:
    """Write a capture directory with the given stream and channel ordering."""
    capture_root = Path(capture_root)
    capture_root.mkdir(parents=True, exist_ok=True)
    if sound.dtype != np.dtype("<i4") or sound.ndim != 2 or sound.shape[0] != MICROPHONE_COUNT:
        raise FormatError(f"raw sound must be little-endian int32 [1024, N], got {sound.dtype}")
    ordering = np.asarray(ordering, dtype=np.int64)
    if not np.array_equal(np.sort(ordering), np.arange(MICROPHONE_COUNT)):
        raise FormatError("channelOrdering is not a permutation")
    container = json.loads(json.dumps(container))  # deep copy
    sound_meta = container["metadata"]["files"]["sound"]
    sound_meta["channelOrdering"] = [int(v) for v in ordering]
    sound_meta["duration"] = int(round(1000.0 * sound.shape[1] / SAMPLE_RATE_HZ))
    with (capture_root / "sound").open("wb") as handle:
        handle.write(np.ascontiguousarray(sound).tobytes())
    (capture_root / "container.json").write_text(
        json.dumps(container, indent=1), encoding="utf-8"
    )


def capture_audio_float(sound: np.ndarray) -> np.ndarray:
    """Convert a raw int32 stream (or excerpt) to float32 relative pressure."""
    return (np.asarray(sound, dtype=np.float32) / np.float32(2**31)).copy()
