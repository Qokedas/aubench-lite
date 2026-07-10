#!/usr/bin/env python3
"""Acoustic-camera heatmap generator for a Sorama CAM1K (32x32, 1024 mic) array.

Reads a raw device capture (`sound` + `container.json`) and produces a per-frame,
per-band acoustic power map aligned to the co-mounted camera, via frequency-domain
delay-and-sum beamforming.

Usage:
    python solution.py --capture <capture_dir> --output <maps.npy>

Output: float32 array shaped [frame, band, 24, 32], plain linear power (>=0, finite).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

# ---- fixed contract parameters -------------------------------------------------
BLOCK_LENGTH = 8_192          # samples per frame
MIC_COUNT = 1_024
GRID_H, GRID_W = 24, 32       # output map rows x cols (matches camera image)
HFOV_DEG = 70.42              # camera horizontal field of view
VFOV_DEG = 43.3              # camera vertical field of view
SPEED_OF_SOUND = 343.0        # m/s

# 11 one-third-octave bands: (lower, upper) Hz taken from the vendor manual.
BANDS = [
    (282.0, 355.0), (355.0, 447.0), (447.0, 562.0), (562.0, 708.0),
    (708.0, 891.0), (891.0, 1122.0), (1122.0, 1413.0), (1413.0, 1778.0),
    (1778.0, 2239.0), (2239.0, 2818.0), (2840.0, 5680.0),
]

# Process this many frames' spectra at once (keeps peak RAM modest for long files).
FRAME_CHUNK = 24
MAX_BAND_BINS = 128   # cap bins per band (incoherent averaging is already smooth)


def load_metadata(capture: Path):
    meta = json.loads((capture / "container.json").read_text())
    sound = meta["metadata"]["files"]["sound"]
    positions = np.array(
        [[p["X"], p["Y"], p["Z"]] for p in sound["frontEnd"]["microphonePositions"]],
        dtype=np.float64,
    )
    ordering = np.array(sound["channelOrdering"], dtype=np.int64)
    # The sound stream stores channels grouped by the hardware read-out order.
    # channelOrdering[k] is the microphone-position index carried by stream channel k,
    # so the geometry aligned to the interleaved channels is positions[ordering].
    mic = positions[ordering]
    fs = float(sound["frontEnd"]["supportedSampleRates"][0])
    return mic, fs


def steering_directions():
    """Unit look-direction per output pixel (device coordinate frame).

    Standard pinhole camera: column 0 = left, column 31 = right (device +X to the
    right); row 0 = top, row 23 = bottom (device +Y up). The array looks along +Z.
    """
    th = np.tan(np.deg2rad(HFOV_DEG) / 2.0)
    tv = np.tan(np.deg2rad(VFOV_DEG) / 2.0)
    col = ((np.arange(GRID_W) + 0.5) / GRID_W * 2.0 - 1.0) * th      # left -> right
    row = (1.0 - (np.arange(GRID_H) + 0.5) / GRID_H * 2.0) * tv       # top  -> bottom
    gx, gy = np.meshgrid(col, row)
    dirs = np.stack([gx, gy, np.ones_like(gx)], axis=-1).reshape(-1, 3)
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
    return dirs  # (GRID_H*GRID_W, 3)


def compute_maps(capture: Path) -> np.ndarray:
    mic, fs = load_metadata(capture)
    sound_path = capture / "sound"
    total_samples = sound_path.stat().st_size // (4 * MIC_COUNT)
    n_frames = total_samples // BLOCK_LENGTH
    if n_frames == 0:
        raise ValueError("capture is shorter than one frame")

    dirs = steering_directions()                       # (P, 3)
    # phase term: exp(-j * 2*pi*f/c * (dir . mic)) images the source direction.
    proj = (dirs @ mic.T).astype(np.float32)           # (P, M) metres
    proj *= (2.0 * np.pi / SPEED_OF_SOUND)             # now multiply by f to get phase

    freqs = np.fft.rfftfreq(BLOCK_LENGTH, 1.0 / fs)
    window = np.hanning(BLOCK_LENGTH).astype(np.float32)

    # Frequency bins belonging to each band (computed once).
    band_bins = []
    for lo, hi in BANDS:
        idx = np.where((freqs >= lo) & (freqs < hi))[0]
        if idx.size > MAX_BAND_BINS:
            sel = np.linspace(0, idx.size - 1, MAX_BAND_BINS).round().astype(int)
            idx = idx[np.unique(sel)]
        band_bins.append(idx)

    P = dirs.shape[0]
    maps = np.zeros((n_frames, len(BANDS), P), dtype=np.float64)

    mmap = np.memmap(sound_path, dtype="<i4", mode="r",
                     shape=(total_samples, MIC_COUNT))

    for start in range(0, n_frames, FRAME_CHUNK):
        stop = min(start + FRAME_CHUNK, n_frames)
        g = stop - start
        # Build windowed spectra for this chunk of frames.
        block = np.asarray(mmap[start * BLOCK_LENGTH: stop * BLOCK_LENGTH],
                           dtype=np.float32).reshape(g, BLOCK_LENGTH, MIC_COUNT)
        block -= block.mean(axis=1, keepdims=True)
        block *= window[None, :, None]
        spec = np.fft.rfft(block, axis=1)              # (g, nfreq, M) complex
        # Reorder to (nfreq, M, g) for per-bin matmul across all chunk frames.
        spec = np.ascontiguousarray(np.transpose(spec, (1, 2, 0))).astype(np.complex64)

        for bi, bins in enumerate(band_bins):
            if bins.size == 0:
                continue
            acc = np.zeros((P, g), dtype=np.float64)
            for f in bins:
                steer = np.exp(-1j * (freqs[f] * proj)).astype(np.complex64)  # (P, M)
                beam = steer @ spec[f]                  # (P, g)
                acc += (beam.real ** 2 + beam.imag ** 2)
            maps[start:stop, bi, :] = (acc / bins.size).T

    maps = maps.reshape(n_frames, len(BANDS), GRID_H, GRID_W)
    # Relative scale is fine; normalise to a comfortable range and enforce contract.
    scale = maps.max()
    if scale > 0:
        maps = maps / scale
    maps = np.nan_to_num(maps, nan=0.0, posinf=0.0, neginf=0.0)
    maps = np.clip(maps, 0.0, None)
    return maps.astype(np.float32)


def main():
    ap = argparse.ArgumentParser(description="Sorama CAM1K acoustic heatmap generator")
    ap.add_argument("--capture", required=True, type=Path)
    ap.add_argument("--output", required=True, type=Path)
    args = ap.parse_args()

    maps = compute_maps(args.capture)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.output, maps)


if __name__ == "__main__":
    main()
