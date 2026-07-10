#!/usr/bin/env python3
"""Offline frequency-domain beamformer for Sorama CAM1K captures."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import numpy as np

FRAME = 8192
BANDS = ((282, 355), (355, 447), (447, 562), (562, 708),
         (708, 891), (891, 1122), (1122, 1413), (1413, 1778),
         (1778, 2239), (2239, 2818), (2840, 5680))
SOUND_SPEED = 343.0
ADC_SCALE = float(2**31)
# Isolated CAM1K channels occasionally change their DC pedestal by many ADC
# counts. Such a step is not sound and otherwise has broadband FFT leakage.
STEP_LIMIT = 2_000_000


def load_layout(capture: Path):
    with open(capture / "container.json", "r", encoding="utf-8") as f:
        doc = json.load(f)
    sm = doc["metadata"]["files"]["sound"]
    fe = sm["frontEnd"]
    fs = float(fe["supportedSampleRates"][0])
    pos = np.asarray([[p["X"], p["Y"], p.get("Z", 0.0)]
                      for p in fe["microphonePositions"]], dtype=np.float64)
    ordering = np.asarray(sm.get("channelOrdering", np.arange(len(pos))), dtype=np.int64)
    if len(ordering) != len(pos) or np.any(ordering < 0) or np.any(ordering >= len(pos)):
        raise ValueError("unsupported or invalid channelOrdering in container.json")

    # File channel i is the i-th member of channelOrdering. Map those channels
    # onto rows of descending array Y and columns of ascending array X.
    file_pos = pos[ordering]
    xs = np.unique(np.round(file_pos[:, 0], 7))
    ys = np.unique(np.round(file_pos[:, 1], 7))[::-1]
    if len(xs) != 32 or len(ys) != 32 or len(file_pos) != 1024:
        raise ValueError("this beamformer requires the CAM1K 32 x 32 microphone layout")
    flat_order = []
    for y in ys:
        row = np.where(np.isclose(file_pos[:, 1], y, atol=2e-6))[0]
        row = row[np.argsort(file_pos[row, 0])]
        if len(row) != 32:
            raise ValueError("microphone positions do not form a complete 32 x 32 grid")
        flat_order.extend(row.tolist())
    pitch_x = float(np.median(np.diff(xs)))
    pitch_y = float(np.median(np.abs(np.diff(ys))))
    return fs, np.asarray(flat_order, dtype=np.int64), pitch_x, pitch_y


def camera_directions():
    """Direction cosines at pinhole pixel centers, row-major."""
    u = (2.0 * (np.arange(32) + 0.5) / 32.0 - 1.0) * np.tan(np.deg2rad(70.42 / 2.0))
    v = (1.0 - 2.0 * (np.arange(24) + 0.5) / 24.0) * np.tan(np.deg2rad(43.3 / 2.0))
    u, v = np.meshgrid(u, v)
    n = np.sqrt(1.0 + u*u + v*v)
    return (u / n).astype(np.float32), (v / n).astype(np.float32)


def cubic_weights(t):
    """Catmull-Rom weights at offsets -1, 0, 1, 2."""
    t2 = t*t
    t3 = t2*t
    return (-0.5*t + t2 - 0.5*t3,
            1.0 - 2.5*t2 + 1.5*t3,
            0.5*t + 2.0*t2 - 1.5*t3,
            -0.5*t2 + 0.5*t3)


def sample_spatial_fft(coeff, freq, dx, dy, pitch_x, pitch_y, pad=128):
    """Evaluate each frequency's array DFT at all camera directions.

    Zero padding followed by bicubic interpolation is an efficient NUDFT for
    this regular array. The X sign accounts for the CAM1K board's X axis being
    opposite the co-mounted camera's image X axis.
    """
    spec = np.fft.fft2(coeff, s=(pad, pad), axes=(-2, -1)).astype(np.complex64)
    f = freq.astype(np.float32)[:, None, None]
    ix = (-f * np.float32(pitch_x / SOUND_SPEED) * dx[None, :, :] * pad) % pad
    iy = (-f * np.float32(pitch_y / SOUND_SPEED) * dy[None, :, :] * pad) % pad
    bx = np.floor(ix).astype(np.int32)
    by = np.floor(iy).astype(np.int32)
    wx = cubic_weights(ix - bx)
    wy = cubic_weights(iy - by)
    fi = np.arange(len(freq), dtype=np.int32)[:, None, None]
    out = np.zeros(ix.shape, dtype=np.complex64)
    offsets = (-1, 0, 1, 2)
    for j, oy in enumerate(offsets):
        yy = (by + oy) % pad
        for i, ox in enumerate(offsets):
            out += spec[fi, yy, (bx + ox) % pad] * (wy[j] * wx[i])
    return out


def beamform(capture: Path, output: Path):
    fs, grid_order, pitch_x, pitch_y = load_layout(capture)
    sound = capture / "sound"
    channels = 1024
    words = os.path.getsize(sound) // 4
    samples = words // channels
    frames = samples // FRAME
    if frames < 1:
        raise ValueError("sound file contains no complete 8192-sample frame")
    if words % channels:
        raise ValueError("sound file size is not an integer number of CAM1K samples")

    raw = np.memmap(sound, dtype="<i4", mode="r", shape=(samples, channels))
    result = np.empty((frames, len(BANDS), 24, 32), dtype=np.float32)
    win = np.hanning(FRAME).astype(np.float32)
    win_scale = np.float32(win.sum())
    freq = np.fft.rfftfreq(FRAME, d=1.0/fs)
    dx, dy = camera_directions()

    # State used to remove only isolated ADC-pedestal steps, continuously
    # across output-frame boundaries.
    previous = None
    pedestal = np.zeros(channels, dtype=np.int64)

    for frame in range(frames):
        block_i = np.asarray(raw[frame*FRAME:(frame+1)*FRAME], dtype=np.int64)
        delta = np.empty_like(block_i)
        if previous is None:
            delta[0] = 0
        else:
            delta[0] = block_i[0] - previous
        delta[1:] = block_i[1:] - block_i[:-1]
        bad = np.abs(delta) > STEP_LIMIT
        # A real impulsive wave reaches many microphones. Never classify a
        # simultaneous array-wide event as an electronics pedestal change.
        coherent = np.count_nonzero(bad, axis=1) > (channels // 10)
        bad[coherent, :] = False
        steps = np.cumsum(np.where(bad, delta, 0), axis=0, dtype=np.int64)
        corrected = block_i - pedestal[None, :] - steps
        pedestal += steps[-1]
        previous = block_i[-1].copy()
        del block_i, delta, bad, steps

        x = corrected[:, grid_order].astype(np.float32) / ADC_SCALE
        del corrected
        # Per-channel DC rejection and a Hann analysis window prevent the
        # large, harmless microphone pedestal from leaking into the bands.
        # Partial array common-mode rejection suppresses correlated front-end
        # pickup without creating a full broadside null.
        x -= x.mean(axis=0, keepdims=True)
        x -= 0.5 * x.mean(axis=1, keepdims=True)
        x *= win[:, None]
        temporal = (np.fft.rfft(x, axis=0) / win_scale).astype(np.complex64)
        del x
        temporal = temporal.reshape(len(freq), 32, 32)

        for band, (lower, upper) in enumerate(BANDS):
            # Half-open boundaries avoid counting a boundary bin twice.
            selected = (freq >= lower) & (freq < upper)
            coeff = temporal[selected]
            steered = sample_spatial_fft(coeff, freq[selected], dx, dy,
                                          pitch_x, pitch_y)
            # Mean over microphones is supplied by /1024. One-sided FFT power
            # is integrated over the manual's exact band limits.
            power = (2.0 / (channels * channels)) * np.sum(
                steered.real*steered.real + steered.imag*steered.imag, axis=0)
            result[frame, band] = np.maximum(power, 0).astype(np.float32)
        print(f"beamformed frame {frame + 1}/{frames}", flush=True)

    output.parent.mkdir(parents=True, exist_ok=True)
    np.save(output, result, allow_pickle=False)


def main():
    parser = argparse.ArgumentParser(description="Create CAM1K acoustic power maps")
    parser.add_argument("--capture", required=True, type=Path,
                        help="directory containing sound and container.json")
    parser.add_argument("--output", required=True, type=Path,
                        help="destination .npy file")
    args = parser.parse_args()
    beamform(args.capture, args.output)


if __name__ == "__main__":
    main()
