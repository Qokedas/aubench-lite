"""Download, inspect, and package Seeing With Sound CAM1K captures for the benchmark.

Public surface (everything the agent ever sees):
  public/capture/          raw Sorama capture: first 2.5 s of axcar1 (sound + container.json)
  public/vendor_export/    maps.npy — the trusted reference output for that capture

Private surface (never mounted into agent or candidate containers):
  private/cases/<name>/    raw excerpt or synthetic capture + reference_maps.npy
"""

from __future__ import annotations

import os
import shutil
import time
from pathlib import Path
from typing import Any

import gdown
import numpy as np
from rosbags.rosbag1 import Reader

from .constants import (
    BLOCK_LENGTH,
    FREQUENCY_BANDS,
    MICROPHONE_COUNT,
    SAMPLE_RATE_HZ,
    SOURCE_CAPTURES,
    SPEED_OF_SOUND_M_S,
    VISIBLE_SAMPLES,
)
from .formats import (
    FormatError,
    capture_audio_float,
    open_capture,
    sha256_file,
    write_capture,
    write_json,
)
from .geometry import make_far_field_grid
from .reference import beamform
from .synthetic import PlaneWave, incoherent_noise, plane_wave_audio

_INT32_SYNTH_SCALE = float(2**28)


def _download_file(file_id: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".partial")
    temporary.unlink(missing_ok=True)
    result = gdown.download(id=file_id, output=str(temporary), quiet=False)
    if not result or not temporary.is_file():
        raise RuntimeError(f"Google Drive download failed for {destination}")
    temporary.replace(destination)


def download_sources(root: str | Path, *, captures: list[str] | None = None) -> dict[str, Any]:
    root = Path(root)
    names = captures or list(SOURCE_CAPTURES)
    report: dict[str, Any] = {"root": str(root.resolve()), "captures": {}}
    for name in names:
        if name not in SOURCE_CAPTURES:
            raise ValueError(f"unknown capture {name!r}; choose from {sorted(SOURCE_CAPTURES)}")
        source = SOURCE_CAPTURES[name]
        capture_root = root / name
        sound = capture_root / "sound"
        metadata = capture_root / "container.json"
        if not sound.exists():
            _download_file(str(source["sound_id"]), sound)
        if not metadata.exists():
            _download_file(str(source["metadata_id"]), metadata)
        if sound.stat().st_size != int(source["sound_bytes"]):
            raise FormatError(
                f"{sound} is {sound.stat().st_size} bytes, expected {source['sound_bytes']}"
            )
        sound_hash = sha256_file(sound)
        metadata_hash = sha256_file(metadata)
        expected_sound_hash = source.get("sound_sha256")
        expected_metadata_hash = source.get("metadata_sha256")
        if expected_sound_hash and sound_hash != expected_sound_hash:
            raise FormatError(f"SHA-256 mismatch for {sound}")
        if expected_metadata_hash and metadata_hash != expected_metadata_hash:
            raise FormatError(f"SHA-256 mismatch for {metadata}")
        report["captures"][name] = {
            "sound": str(sound.resolve()),
            "sound_bytes": sound.stat().st_size,
            "sound_sha256": sound_hash,
            "metadata": str(metadata.resolve()),
            "metadata_sha256": metadata_hash,
        }
    return report


def inspect_rosbag(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    with Reader(path) as reader:
        connections = [
            {
                "topic": connection.topic,
                "message_type": connection.msgtype,
                "message_count": connection.msgcount,
            }
            for connection in reader.connections
        ]
        result = {
            "path": str(path.resolve()),
            "bytes": path.stat().st_size,
            "start_time_ns": reader.start_time,
            "end_time_ns": reader.end_time,
            "duration_seconds": (reader.end_time - reader.start_time) / 1e9,
            "message_count": reader.message_count,
            "connections": connections,
        }
    result["contains_cam1k_pressure"] = any(
        "sorama" in item["topic"].lower()
        or "microphone" in item["topic"].lower()
        or "audio" in item["topic"].lower()
        for item in connections
    )
    return result


def _save_case(
    root: Path,
    *,
    name: str,
    kind: str,
    sound: np.ndarray,
    container: dict[str, Any],
    ordering: np.ndarray,
    positions_by_row: np.ndarray,
    grid: np.ndarray,
) -> dict[str, Any]:
    """Write one raw private case plus its trusted reference tensor."""
    case_root = root / "cases" / name
    write_capture(case_root, sound, container, ordering)
    reference = beamform(capture_audio_float(sound), positions_by_row, grid)
    np.save(case_root / "reference_maps.npy", reference, allow_pickle=False)
    return {
        "name": name,
        "kind": kind,
        "sound_sha256": sha256_file(case_root / "sound"),
        "reference_sha256": sha256_file(case_root / "reference_maps.npy"),
        "num_samples": int(sound.shape[1]),
        "num_frames": int(reference.shape[0]),
    }


def _procedural_audio(positions_by_row: np.ndarray) -> list[tuple[str, np.ndarray]]:
    """Synthetic pressure fields, in row order matching ``positions_by_row``."""
    common = {
        "sample_rate_hz": SAMPLE_RATE_HZ,
        "num_samples": BLOCK_LENGTH,
        "speed_of_sound_m_s": SPEED_OF_SOUND_M_S,
    }
    cases: list[tuple[str, np.ndarray]] = [
        (
            "single_low_left",
            plane_wave_audio(
                positions_by_row,
                waves=[PlaneWave(315.0, -24.0, 5.0, phase_rad=0.4)],
                noise_std=0.002,
                seed=101,
                **common,
            ),
        ),
        (
            "single_mid_right",
            plane_wave_audio(
                positions_by_row,
                waves=[PlaneWave(1_000.0, 19.0, -7.0, phase_rad=1.1)],
                noise_std=0.002,
                seed=102,
                **common,
            ),
        ),
        (
            "single_high_upper",
            plane_wave_audio(
                positions_by_row,
                waves=[PlaneWave(4_000.0, -9.0, 12.0, phase_rad=2.0)],
                noise_std=0.003,
                seed=103,
                **common,
            ),
        ),
        (
            "dual_source",
            plane_wave_audio(
                positions_by_row,
                waves=[
                    PlaneWave(800.0, -20.0, -5.0, amplitude=1.0),
                    PlaneWave(2_000.0, 22.0, 8.0, amplitude=0.65, phase_rad=0.8),
                ],
                noise_std=0.002,
                seed=104,
                **common,
            ),
        ),
        (
            "broadband_coherent",
            plane_wave_audio(
                positions_by_row,
                waves=[
                    PlaneWave(center, 12.0, 3.0, amplitude=1.0 / len(FREQUENCY_BANDS), phase_rad=i)
                    for i, (_, center, _) in enumerate(FREQUENCY_BANDS)
                ],
                noise_std=0.002,
                seed=105,
                **common,
            ),
        ),
    ]
    rng = np.random.default_rng(106)
    muted = plane_wave_audio(
        positions_by_row,
        waves=[PlaneWave(1_250.0, -14.0, 9.0)],
        noise_std=0.002,
        seed=106,
        **common,
    )
    muted[rng.choice(MICROPHONE_COUNT, size=96, replace=False)] = 0
    cases.append(("muted_microphones", muted))
    gain = plane_wave_audio(
        positions_by_row,
        waves=[PlaneWave(2_500.0, 5.0, -10.0)],
        noise_std=0.002,
        seed=107,
        **common,
    )
    gain *= rng.uniform(0.65, 1.35, size=(MICROPHONE_COUNT, 1)).astype(np.float32)
    cases.append(("gain_mismatch", gain))
    cases.append(
        (
            "spatially_incoherent_noise",
            incoherent_noise(MICROPHONE_COUNT, BLOCK_LENGTH, seed=108, scale=0.1),
        )
    )
    cases.append(("zero_input", np.zeros((MICROPHONE_COUNT, BLOCK_LENGTH), dtype=np.float32)))
    return cases


def _synthetic_to_int32(audio: np.ndarray) -> np.ndarray:
    """Quantize synthetic float pressure to the device's raw int32 stream format."""
    return np.ascontiguousarray(
        np.round(np.asarray(audio, dtype=np.float64) * _INT32_SYNTH_SCALE)
    ).astype("<i4")


# Hidden real excerpts: (case name, source capture, start sample, sample count).
# Starts are deliberately not multiples of the block length, never overlap the
# visible interval [0, VISIBLE_SAMPLES) of axcar1, and lengths include trailing
# partial blocks so frame handling is exercised.
_PRIVATE_REAL_EXCERPTS: tuple[tuple[str, str, int, int], ...] = (
    ("real_axcar1_a", "axcar1", 125_003, 2 * BLOCK_LENGTH + 3_000),
    ("real_axcar1_b", "axcar1", 250_011, BLOCK_LENGTH),
    ("real_axcar1_c", "axcar1", 391_307, 3 * BLOCK_LENGTH + 1_419),
    ("real_axcar3_a", "axcar3", 47_123, BLOCK_LENGTH),
    ("real_axcar3_b", "axcar3", 171_001, 2 * BLOCK_LENGTH + 2_616),
    ("real_axcar3_c", "axcar3", 283_457, BLOCK_LENGTH + 5_000),
    ("real_axcar3_d", "axcar3", 402_113, 3 * BLOCK_LENGTH),
)


def build_benchmark(source_root: str | Path, output_root: str | Path) -> dict[str, Any]:
    source_root = Path(source_root)
    output_root = Path(output_root)
    temporary = output_root.with_name(output_root.name + f".building-{os.getpid()}")
    if temporary.exists():
        shutil.rmtree(temporary)
    public_root = temporary / "public"
    private_root = temporary / "private"
    public_root.mkdir(parents=True)
    private_root.mkdir(parents=True)

    captures = {name: open_capture(source_root / name) for name in ("axcar1", "axcar3")}
    if not np.allclose(
        captures["axcar1"]["positions_by_row"],
        captures["axcar3"]["positions_by_row"],
        atol=1e-7,
    ):
        raise FormatError("the two source captures report different microphone geometries")
    for name, capture in captures.items():
        if capture["sample_rate_hz"] != SAMPLE_RATE_HZ:
            raise FormatError(f"{name} sample rate is {capture['sample_rate_hz']}")
        if capture["num_samples"] != SAMPLE_RATE_HZ * 10:
            raise FormatError(f"{name} has unexpected length {capture['num_samples']}")

    grid = make_far_field_grid()
    axcar1 = captures["axcar1"]

    # Public: the visible raw capture and its vendor export.
    visible_sound = np.ascontiguousarray(axcar1["sound"][:, :VISIBLE_SAMPLES]).astype("<i4")
    write_capture(
        public_root / "capture", visible_sound, axcar1["container"], axcar1["ordering"]
    )
    vendor_export = beamform(capture_audio_float(visible_sound), axcar1["positions_by_row"], grid)
    export_root = public_root / "vendor_export"
    export_root.mkdir(parents=True)
    np.save(export_root / "maps.npy", vendor_export, allow_pickle=False)
    public_manifest = {
        "schema_version": "cam1k-beamforming-eval-v1",
        "created_unix": int(time.time()),
        "capture_sound_sha256": sha256_file(public_root / "capture" / "sound"),
        "vendor_export_sha256": sha256_file(export_root / "maps.npy"),
        "vendor_export_frames": int(vendor_export.shape[0]),
    }
    write_json(public_root / "manifest.json", public_manifest)

    # Private: raw real excerpts and raw synthetic captures.
    private_cases: list[dict[str, Any]] = []
    for name, capture_name, start, count in _PRIVATE_REAL_EXCERPTS:
        capture = captures[capture_name]
        if capture_name == "axcar1" and start < VISIBLE_SAMPLES:
            raise FormatError(f"{name} overlaps the visible interval")
        if start + count > capture["num_samples"]:
            raise FormatError(f"{name} exceeds the capture length")
        sound = np.ascontiguousarray(capture["sound"][:, start : start + count]).astype("<i4")
        private_cases.append(
            _save_case(
                private_root,
                name=name,
                kind="real",
                sound=sound,
                container=capture["container"],
                ordering=capture["ordering"],
                positions_by_row=capture["positions_by_row"],
                grid=grid,
            )
        )
    for name, audio in _procedural_audio(axcar1["positions_by_row"]):
        private_cases.append(
            _save_case(
                private_root,
                name=name,
                kind="procedural",
                sound=_synthetic_to_int32(audio),
                container=axcar1["container"],
                ordering=axcar1["ordering"],
                positions_by_row=axcar1["positions_by_row"],
                grid=grid,
            )
        )
    private_manifest = {
        "schema_version": "cam1k-beamforming-eval-v1",
        "created_unix": int(time.time()),
        "cases": private_cases,
        "anti_shortcut": {
            "secret_channel_permutation": True,
            "hidden_excerpts_disjoint_from_visible": True,
            "candidate_network": "none",
            "reference_maps_mounted_in_candidate": False,
        },
    }
    write_json(private_root / "manifest.json", private_manifest)

    if output_root.exists():
        shutil.rmtree(output_root)
    temporary.replace(output_root)
    return {
        "output": str(output_root.resolve()),
        "visible_samples": int(visible_sound.shape[1]),
        "vendor_export_frames": int(vendor_export.shape[0]),
        "private_real_cases": sum(item["kind"] == "real" for item in private_cases),
        "private_procedural_cases": sum(item["kind"] == "procedural" for item in private_cases),
    }
