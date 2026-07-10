#!/usr/bin/env python3
"""Acoustic-camera beamformer for the Sorama CAM1K array.

Reads a raw capture (``sound`` + ``container.json``) and produces a heatmap
tensor [frame, band, 24, 32] of linear acoustic power, one viewing direction
per pixel, matching a pinhole camera view (70.42 deg wide x 43.3 deg tall).
"""
import argparse
import json
import os
import numpy as np

BLOCK = 8192
GRID_H, GRID_W = 24, 32
FOV_X_DEG = 70.42
FOV_Y_DEG = 43.3
MIC_COUNT = 1024
SPEED_OF_SOUND = 343.0

# 1/3-octave bands: (lower, center, upper) Hz. Beamform bins are chosen from
# [lower, upper); the whole band is integrated into one power value.
BANDS = [
    (282, 315, 355), (355, 400, 447), (447, 500, 562), (562, 630, 708),
    (708, 800, 891), (891, 1000, 1122), (1122, 1250, 1413),
    (1413, 1600, 1778), (1778, 2000, 2239), (2239, 2500, 2818),
    (2840, 4000, 5680),
]


def load_capture(capture_dir):
    with open(os.path.join(capture_dir, "container.json")) as fh:
        meta = json.load(fh)
    sound_meta = meta["metadata"]["files"]["sound"]
    positions = np.array(
        [[p["X"], p["Y"], p["Z"]] for p in sound_meta["frontEnd"]["microphonePositions"]],
        dtype=np.float64,
    )
    sound_path = os.path.join(capture_dir, "sound")
    n_samples = os.path.getsize(sound_path) // (4 * MIC_COUNT)
    data = np.memmap(sound_path, dtype=np.int32, mode="r",
                     shape=(n_samples, MIC_COUNT))
    # Sample rate from metadata duration (milliseconds) when available.
    duration_ms = sound_meta.get("duration")
    if duration_ms:
        fs = n_samples * 1000.0 / float(duration_ms)
    else:
        fs = 46875.0
    return data, positions, fs, n_samples


def direction_grid():
    """Unit look vectors, one per (row, col) image pixel. Row 0 top, col 0 left."""
    tx = np.tan(np.radians(FOV_X_DEG) / 2.0)
    ty = np.tan(np.radians(FOV_Y_DEG) / 2.0)
    cols = (2.0 * (np.arange(GRID_W) + 0.5) / GRID_W - 1.0) * tx      # left->right
    rows = (2.0 * (np.arange(GRID_H) + 0.5) / GRID_H - 1.0) * ty      # top->bottom
    xn, yn = np.meshgrid(cols, rows)                                  # (H, W)
    ux = xn.ravel()             # +X to the right in the image
    uy = -yn.ravel()            # image row 0 (top) -> +Y in array coords
    uz = np.ones_like(ux)       # scene is in front of the array plane
    v = np.stack([ux, uy, uz], axis=1)
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    return v                    # (H*W, 3)


def compute_maps(data, positions, fs, n_samples):
    n_frames = n_samples // BLOCK
    dirs = direction_grid()                     # (D, 3)
    delay = dirs @ positions.T                  # (D, M) = r . u
    window = np.hanning(BLOCK)[:, None]
    freqs = np.fft.rfftfreq(BLOCK, 1.0 / fs)

    # Precompute the FFT bins belonging to each band.
    band_bins = []
    for lo, _c, hi in BANDS:
        bins = np.where((freqs >= lo) & (freqs < hi))[0]
        if bins.size == 0:
            bins = np.array([int(np.argmin(np.abs(freqs - _c)))])
        band_bins.append(bins)

    norm = 1.0 / (MIC_COUNT * BLOCK) ** 2
    D = dirs.shape[0]
    maps = np.zeros((n_frames, len(BANDS), D), dtype=np.float64)

    # Work in frame batches so memory stays bounded on long recordings while
    # each steering matrix (built once per batch) is still reused across the
    # batch's frames.
    FRAME_BATCH = 24
    BIN_CHUNK = 32
    for f0 in range(0, n_frames, FRAME_BATCH):
        f1 = min(f0 + FRAME_BATCH, n_frames)
        nb = f1 - f0
        spectra = np.empty((nb, freqs.size, MIC_COUNT), dtype=np.complex64)
        for j in range(nb):
            block = np.asarray(data[(f0 + j) * BLOCK:(f0 + j + 1) * BLOCK], dtype=np.float64)
            block -= block.mean(axis=0, keepdims=True)
            spectra[j] = np.fft.rfft(block * window, axis=0).astype(np.complex64)

        for bi, bins in enumerate(band_bins):
            acc = np.zeros((nb, D), dtype=np.float64)
            for st in range(0, bins.size, BIN_CHUNK):
                ks = bins[st:st + BIN_CHUNK]
                phase = np.exp(1j * (2.0 * np.pi * freqs[ks] / SPEED_OF_SOUND)[:, None, None]
                               * delay[None, :, :]).astype(np.complex64)   # (K, D, M)
                for j in range(nb):
                    beam = np.einsum('kdm,km->kd', phase, spectra[j, ks], optimize=True)
                    acc[j] += np.sum(np.abs(beam) ** 2, axis=0)
                del phase
            maps[f0:f1, bi] = acc * (norm / bins.size)
        del spectra

    return maps.reshape(n_frames, len(BANDS), GRID_H, GRID_W).astype(np.float32)


def main():
    ap = argparse.ArgumentParser(description="Sorama CAM1K acoustic heatmaps")
    ap.add_argument("--capture", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    data, positions, fs, n_samples = load_capture(args.capture)
    maps = compute_maps(data, positions, fs, n_samples)
    maps = np.nan_to_num(maps, nan=0.0, posinf=0.0, neginf=0.0)
    maps = np.abs(maps).astype(np.float32)
    np.save(args.output, maps)


if __name__ == "__main__":
    main()
