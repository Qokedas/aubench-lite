#!/usr/bin/env python3
"""Self-contained conventional DAS baseline used to verify benchmark solvability.

Reads the raw Sorama capture format directly: channel-major little-endian int32
stream plus container.json metadata (positions indexed by channel ID, with
channelOrdering mapping stream rows to channel IDs).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

BLOCK = 8_192
SPEED = 343.0
FOV_X_DEG = 70.42
FOV_Y_DEG = 43.3
HEIGHT, WIDTH = 24, 32
BANDS = [
    (282.0, 355.0), (355.0, 447.0), (447.0, 562.0), (562.0, 708.0),
    (708.0, 891.0), (891.0, 1_122.0), (1_122.0, 1_413.0), (1_413.0, 1_778.0),
    (1_778.0, 2_239.0), (2_239.0, 2_818.0), (2_840.0, 5_680.0),
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--capture", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    capture = Path(args.capture)

    container = json.loads((capture / "container.json").read_text(encoding="utf-8"))
    sound_meta = container["metadata"]["files"]["sound"]
    ordering = np.asarray(sound_meta["channelOrdering"], dtype=np.int64)
    positions = np.asarray(
        [[p["X"], p["Y"], p["Z"]] for p in sound_meta["frontEnd"]["microphonePositions"]],
        dtype=np.float64,
    )
    sample_rate = float(sound_meta["frontEnd"]["supportedSampleRates"][0])
    channels = positions.shape[0]
    mics = positions[ordering]  # row i of the stream carries channel ordering[i]

    sound_path = capture / "sound"
    samples = sound_path.stat().st_size // (4 * channels)
    raw = np.memmap(sound_path, dtype="<i4", mode="r", shape=(channels, samples))

    x_extent = np.tan(np.deg2rad(FOV_X_DEG / 2.0))
    y_extent = np.tan(np.deg2rad(FOV_Y_DEG / 2.0))
    xs = np.linspace(-x_extent, x_extent, WIDTH)
    ys = np.linspace(y_extent, -y_extent, HEIGHT)
    xx, yy = np.meshgrid(xs, ys)
    grid = np.stack([xx, yy, np.ones_like(xx)], axis=2)
    grid /= np.linalg.norm(grid, axis=2, keepdims=True)
    delays = (grid.reshape(-1, 3).astype(np.float32) @ mics.astype(np.float32).T) / np.float32(
        SPEED
    )

    frequencies = np.fft.rfftfreq(BLOCK, d=1.0 / sample_rate)
    band_bins = [np.flatnonzero((frequencies >= lo) & (frequencies <= hi)) for lo, hi in BANDS]
    window = np.hanning(BLOCK).astype(np.float32)
    frames = samples // BLOCK
    maps = np.empty((frames, len(BANDS), HEIGHT, WIDTH), dtype=np.float32)

    for frame_index in range(frames):
        start = frame_index * BLOCK
        frame = raw[:, start : start + BLOCK].astype(np.float32) / np.float32(2**31)
        frame -= frame.mean(axis=1, keepdims=True)
        spectrum = np.fft.rfft(frame * window[None, :], axis=1).astype(np.complex64)
        for band_index, bins in enumerate(band_bins):
            power = np.zeros(delays.shape[0], dtype=np.float32)
            for chunk_start in range(0, bins.size, 16):
                chunk = bins[chunk_start : chunk_start + 16]
                phase = (
                    np.float32(-2.0 * np.pi)
                    * frequencies[chunk].astype(np.float32)[:, None, None]
                    * delays[None, :, :]
                )
                steering = np.exp(1j * phase).astype(np.complex64)
                focused = np.einsum("bgm,mb->gb", steering, spectrum[:, chunk], optimize=True)
                power += (focused.real**2 + focused.imag**2).sum(axis=1)
            power /= np.float32(len(bins) * channels**2)
            maps[frame_index, band_index] = power.reshape(HEIGHT, WIDTH)
    np.maximum(maps, 0, out=maps)
    np.save(args.output, maps, allow_pickle=False)


if __name__ == "__main__":
    main()
