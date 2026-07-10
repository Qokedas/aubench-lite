#!/usr/bin/env python3
"""Offline frequency-domain beamformer for Sorama CAM recordings.

The raw CAM stream is little-endian fixed-point int32, sample interleaved.  For each
8192-sample block this program forms a one-sided spectrum and evaluates a
conventional delay-and-sum beamformer in the camera viewing directions.  The
uniform array makes the steering sum a spatial DTFT; a zero-padded spatial FFT
plus fine bilinear interpolation evaluates it efficiently.
"""
from __future__ import annotations

# Avoid BLAS libraries choosing dozens of threads on modest hosts.  Set this
# before importing numpy; users can still override it in their environment.
import os
os.environ.setdefault("OPENBLAS_NUM_THREADS", "8")
os.environ.setdefault("OMP_NUM_THREADS", "8")

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

BLOCK = 8192
MAP_H, MAP_W = 24, 32
HFOV_DEG, VFOV_DEG = 70.42, 43.3
SOUND_SPEED = 343.0                 # metres/second, room-temperature air
SPATIAL_FFT = 256                   # 8x interpolation of the 32x32 array
FFT_CHUNK = 64                      # bounds temporary memory to about 35 MiB
# Vendor/manual bands: lower, centre, upper.  Centre is retained here to make
# the prescribed ordering explicit, while integration uses lower/upper.
BANDS = (
    (282., 315., 355.), (355., 400., 447.), (447., 500., 562.),
    (562., 630., 708.), (708., 800., 891.), (891., 1000., 1122.),
    (1122., 1250., 1413.), (1413., 1600., 1778.),
    (1778., 2000., 2239.), (2239., 2500., 2818.),
    (2840., 4000., 5680.),
)


def _sound_metadata(doc: dict) -> dict:
    try:
        return doc["metadata"]["files"]["sound"]
    except (KeyError, TypeError) as exc:
        raise ValueError("container.json has no metadata.files.sound object") from exc


def _read_layout(capture: Path):
    meta_path = capture / "container.json"
    sound_path = capture / "sound"
    if not meta_path.is_file() or not sound_path.is_file():
        raise ValueError("capture directory must contain regular files 'sound' and 'container.json'")
    with meta_path.open("r", encoding="utf-8") as f:
        doc = json.load(f)
    sm = _sound_metadata(doc)
    frontend = sm.get("frontEnd", {})
    pdoc = frontend.get("microphonePositions")
    if not isinstance(pdoc, list) or not pdoc:
        raise ValueError("microphone positions are missing from container.json")
    positions = np.asarray([[p["X"], p["Y"]] for p in pdoc], dtype=np.float64)

    ordering = np.asarray(sm.get("channelOrdering", []), dtype=np.int64)
    if ordering.size:
        if ordering.min() < 0 or ordering.max() >= len(positions):
            raise ValueError("invalid channelOrdering in container.json")
        # channelOrdering gives, in file order, the front-end microphone index.
        file_positions = positions[ordering]
        channels = len(ordering)
    else:
        file_positions = positions
        channels = len(positions)

    selected = sm.get("channelSelection")
    if selected and len(selected) == len(file_positions) and not all(selected):
        mask = np.asarray(selected, dtype=bool)
        file_positions = file_positions[mask]
        channels = int(mask.sum())

    rates = frontend.get("supportedSampleRates", [46875.0])
    sample_rate = float(rates[0])
    if not math.isfinite(sample_rate) or sample_rate <= 0:
        raise ValueError("invalid sample rate in container.json")

    word_bytes = np.dtype("<i4").itemsize
    byte_count = sound_path.stat().st_size
    stride = channels * word_bytes
    if channels <= 0 or byte_count < stride:
        raise ValueError("sound file is empty or channel count is invalid")
    if byte_count % stride:
        raise ValueError("sound byte count is not a whole number of multichannel int32 samples")
    samples = byte_count // stride
    return sound_path, file_positions, channels, samples, sample_rate


def _camera_directions() -> np.ndarray:
    """Return horizontal/vertical direction cosines, top-to-bottom/left-to-right."""
    # Pixel centres on a pinhole image plane.  Positive physical Y is upward,
    # hence the reversal in py so output row zero is the top of the image.
    px = ((np.arange(MAP_W, dtype=np.float64) + 0.5) - MAP_W / 2) / (MAP_W / 2)
    py = (MAP_H / 2 - (np.arange(MAP_H, dtype=np.float64) + 0.5)) / (MAP_H / 2)
    px *= math.tan(math.radians(HFOV_DEG / 2))
    py *= math.tan(math.radians(VFOV_DEG / 2))
    xx, yy = np.meshgrid(px, py)
    norm = np.sqrt(1.0 + xx * xx + yy * yy)
    return np.stack((xx / norm, yy / norm), axis=-1).reshape(-1, 2).astype(np.float32)


def _sensor_grid(file_positions: np.ndarray):
    """Map channels to ascending-Y/ascending-X cells of the uniform array."""
    # Metadata coordinates contain small float32 roundoff, so quantise only for
    # identifying grid lines.  Actual pitch comes from line medians below.
    qx = np.round(file_positions[:, 0], 5)
    qy = np.round(file_positions[:, 1], 5)
    xs = np.unique(qx)
    ys = np.unique(qy)
    if len(xs) * len(ys) != len(file_positions):
        raise ValueError("microphone positions do not form one complete rectangular grid")
    grid = np.full((len(ys), len(xs)), -1, dtype=np.int32)
    for channel, (x, y) in enumerate(zip(qx, qy)):
        ix = int(np.searchsorted(xs, x))
        iy = int(np.searchsorted(ys, y))
        if grid[iy, ix] >= 0:
            raise ValueError("duplicate microphone position in metadata")
        grid[iy, ix] = channel
    if np.any(grid < 0) or len(xs) < 2 or len(ys) < 2:
        raise ValueError("incomplete microphone grid in metadata")
    dx = float(np.median(np.diff(xs)))
    dy = float(np.median(np.diff(ys)))
    if dx <= 0 or dy <= 0 or not np.allclose(np.diff(xs), dx, atol=2e-4) \
            or not np.allclose(np.diff(ys), dy, atol=2e-4):
        raise ValueError("microphone grid is not uniformly spaced")
    return grid, dx, dy


def _interpolation_plan(sample_rate: float, dx: float, dy: float,
                        directions: np.ndarray):
    """Precompute FFT-grid coordinates for every temporal bin and map pixel."""
    freqs = np.fft.rfftfreq(BLOCK, d=1.0 / sample_rate)
    plans = []
    for lower, _centre, upper in BANDS:
        # Adjacent bands share a stated edge.  Half-open intervals prevent a bin
        # exactly on that edge from being counted twice; the final upper edge is
        # included by a tiny floating-point tolerance.
        bins = np.flatnonzero((freqs >= lower) & (freqs < upper))
        chunks = []
        for start in range(0, len(bins), FFT_CHUNK):
            kk = bins[start:start + FFT_CHUNK]
            f = freqs[kk, None].astype(np.float32)
            gx = np.mod(f * (dx * SPATIAL_FFT / SOUND_SPEED) * directions[None, :, 0],
                        SPATIAL_FFT)
            gy = np.mod(f * (dy * SPATIAL_FFT / SOUND_SPEED) * directions[None, :, 1],
                        SPATIAL_FFT)
            x0 = np.floor(gx).astype(np.int16)
            y0 = np.floor(gy).astype(np.int16)
            wx = (gx - x0).astype(np.float32)
            wy = (gy - y0).astype(np.float32)
            chunks.append((kk, x0, y0, wx, wy))
        plans.append(chunks)
    return plans


def beamform(capture: Path, output: Path) -> None:
    sound_path, positions, channels, samples, sample_rate = _read_layout(capture)
    frame_count = samples // BLOCK
    if frame_count == 0:
        raise ValueError(f"sound has only {samples} samples; at least {BLOCK} are required")

    channel_grid, dx, dy = _sensor_grid(positions)
    directions = _camera_directions()
    plans = _interpolation_plan(sample_rate, dx, dy, directions)
    window = np.hanning(BLOCK).astype(np.float32)
    # int32 samples use the CAM fixed-point Q24 scale.  Absolute calibration is
    # intentionally not asserted; this gives stable relative linear pressure.
    input_scale = np.float32(1.0 / (1 << 24))
    # Parseval/window correction and normalized DAS steering (mean of sensors).
    power_scale = np.float32(2.0 / (BLOCK * float(np.sum(window * window)) * channels**2))

    raw = np.memmap(sound_path, mode="r", dtype="<i4", shape=(samples, channels))
    result = np.empty((frame_count, len(BANDS), MAP_H, MAP_W), dtype=np.float32)
    flat_grid = channel_grid.ravel()
    pixel_count = MAP_H * MAP_W

    for frame in range(frame_count):
        begin = frame * BLOCK
        block = np.asarray(raw[begin:begin + BLOCK], dtype=np.float32)
        # Removing each ADC's constant offset avoids leakage from DC.  A Hann
        # taper keeps finite-frame leakage out of neighbouring prescribed bands.
        block -= block.mean(axis=0, keepdims=True)
        block *= input_scale
        spectrum = np.fft.rfft(block * window[:, None], axis=0).astype(np.complex64)

        for band, chunks in enumerate(plans):
            power = np.zeros(pixel_count, dtype=np.float64)
            for kk, x0, y0, wx, wy in chunks:
                # Ascending Y/X grid matches the phase convention used by fft2.
                sensor_spectrum = spectrum[kk][:, flat_grid].reshape(
                    len(kk), channel_grid.shape[0], channel_grid.shape[1])
                spatial = np.fft.fft2(sensor_spectrum,
                                      s=(SPATIAL_FFT, SPATIAL_FFT),
                                      axes=(-2, -1)).astype(np.complex64)
                row = np.arange(len(kk), dtype=np.intp)[:, None]
                xi = x0.astype(np.intp)
                yi = y0.astype(np.intp)
                x1 = (xi + 1) % SPATIAL_FFT
                y1 = (yi + 1) % SPATIAL_FFT
                # Bilinear interpolation of complex pressure, not power.
                a = spatial[row, yi, xi]
                b = spatial[row, yi, x1]
                c = spatial[row, y1, xi]
                d = spatial[row, y1, x1]
                z = ((1.0 - wy) * ((1.0 - wx) * a + wx * b)
                     + wy * ((1.0 - wx) * c + wx * d))
                power += np.sum(z.real * z.real + z.imag * z.imag,
                                axis=0, dtype=np.float64)
            result[frame, band] = (power * power_scale).reshape(MAP_H, MAP_W)

        print(f"beamformed frame {frame + 1}/{frame_count}", file=sys.stderr, flush=True)

    output.parent.mkdir(parents=True, exist_ok=True)
    # np.save preserves the required float32 dtype and exact axis ordering.
    np.save(output, result, allow_pickle=False)


def main() -> int:
    parser = argparse.ArgumentParser(description="Create CAM1K acoustic band maps")
    parser.add_argument("--capture", required=True, type=Path,
                        help="directory containing sound and container.json")
    parser.add_argument("--output", required=True, type=Path,
                        help="destination .npy file")
    args = parser.parse_args()
    try:
        beamform(args.capture, args.output)
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
        parser.error(str(exc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
