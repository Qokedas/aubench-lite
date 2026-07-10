#!/usr/bin/env python3
"""Acoustic heatmap generator for Sorama CAM1K captures.

Usage:
    python solution.py --capture <capture_dir> --output <maps.npy>

The capture directory must contain the raw `sound` file and `container.json`
as written by the acquisition rig.

Data layout (reverse-engineered and validated on a reference capture):
  * `sound` is int32 PCM, sample-interleaved: consecutive groups of N values
    hold one sample for each of the N channels (N = microphone count, 1024).
  * Channel i of the file belongs to the microphone with id
    metadata.files.sound.channelOrdering[i]; microphone coordinates (metres,
    array plane z = 0) are metadata...frontEnd.microphonePositions, indexed
    by microphone id.  (Confirmed empirically: only this assignment makes
    inter-channel coherence decay with physical mic distance.)
  * Sample rate = sample count / duration; ~46875 Hz for this hardware.

Processing: frequency-domain delay-and-sum beamforming.
  * The recording is split into frames of 8192 consecutive samples
    (leftover samples at the end are discarded).
  * Per frame, each channel is DC-removed and FFT'd. For every FFT bin that
    falls inside a frequency band, the channel spectra are phase-aligned for
    each pixel's look direction (far-field plane-wave steering) and summed;
    squared magnitudes are accumulated into that band's map.
  * Pixels are a 24 x 32 pinhole camera grid, 70.42 deg (H) x 43.3 deg (V),
    row 0 = image top, column 0 = image left, one direction per pixel centre.

Output: float32 .npy tensor [frame, band, 24, 32]; linear relative power
(finite, non-negative, not dB).
"""
import argparse
import json
import os

import numpy as np

SPEED_OF_SOUND = 343.0   # m/s
FRAME = 8192             # samples per output frame
DEFAULT_FS = 46875.0     # Hz, CAM1K nominal rate
BANDS = [(282, 315, 355), (355, 400, 447), (447, 500, 562), (562, 630, 708),
         (708, 800, 891), (891, 1000, 1122), (1122, 1250, 1413),
         (1413, 1600, 1778), (1778, 2000, 2239), (2239, 2500, 2818),
         (2840, 4000, 5680)]
FOV_X_DEG, FOV_Y_DEG = 70.42, 43.3
ROWS, COLS = 24, 32
CHUNK_FRAMES = 16        # frames processed per pass (bounds memory use)


def load_capture(cap_dir):
    with open(os.path.join(cap_dir, 'container.json')) as f:
        meta = json.load(f)
    snd = meta['metadata']['files']['sound']
    fname = meta.get('files', {}).get('sound', 'sound')
    pos = np.array([[p['X'], p['Y'], p['Z']]
                    for p in snd['frontEnd']['microphonePositions']],
                   dtype=np.float64)
    nch = len(pos)
    order = np.asarray(snd.get('channelOrdering', np.arange(nch)), dtype=np.int64)
    chan_pos = pos[order]                      # position of each file channel
    sel = snd.get('channelSelection')
    use = np.asarray(sel, dtype=bool) if sel is not None else np.ones(nch, bool)
    raw = np.fromfile(os.path.join(cap_dir, fname), dtype=np.int32)
    nsamp = raw.size // nch
    data = raw[:nsamp * nch].reshape(nsamp, nch)
    fs = DEFAULT_FS
    dur_ms = snd.get('duration')
    if dur_ms:
        est = nsamp / (dur_ms / 1000.0)
        # snap to a clean rate when close to nominal, otherwise trust estimate
        fs = round(est / 5.0) * 5.0 if 0.9 * DEFAULT_FS < est < 1.1 * DEFAULT_FS else est
    return data, chan_pos, use, fs


def pixel_directions():
    """Unit look-direction per pixel, array coords (x right, y up, z fwd)."""
    tan_x = np.tan(np.deg2rad(FOV_X_DEG / 2.0))
    tan_y = np.tan(np.deg2rad(FOV_Y_DEG / 2.0))
    xs = ((np.arange(COLS) + 0.5) / COLS * 2.0 - 1.0) * tan_x   # left -> right
    ys = -((np.arange(ROWS) + 0.5) / ROWS * 2.0 - 1.0) * tan_y  # row 0 = top
    d = np.zeros((ROWS, COLS, 3))
    d[..., 0] = xs[None, :]
    d[..., 1] = ys[:, None]
    d[..., 2] = 1.0
    d /= np.linalg.norm(d, axis=-1, keepdims=True)
    return d.reshape(-1, 3)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--capture', required=True)
    ap.add_argument('--output', required=True)
    args = ap.parse_args()

    data, chan_pos, use, fs = load_capture(args.capture)
    data = data[:, use]
    chan_pos = chan_pos[use]
    nsamp, nch = data.shape
    n_frames = nsamp // FRAME

    freqs = np.fft.rfftfreq(FRAME, 1.0 / fs)
    df = freqs[1] - freqs[0]
    lo = min(b[0] for b in BANDS)
    hi = max(b[2] for b in BANDS)
    kidx = np.where((freqs >= lo) & (freqs < hi))[0]
    band_of = np.full(len(kidx), -1)
    for bi, (l, _, u) in enumerate(BANDS):
        band_of[(freqs[kidx] >= l) & (freqs[kidx] < u)] = bi

    dirs = pixel_directions()
    # phase per unit frequency for each (pixel, channel)
    phase = (-2.0 * np.pi / SPEED_OF_SOUND * (dirs @ chan_pos.T)).astype(np.float32)
    w_step = np.exp(1j * (df * phase))         # per-bin phase increment

    out = np.zeros((n_frames, len(BANDS), ROWS * COLS), np.float64)
    for c0 in range(0, n_frames, CHUNK_FRAMES):
        c1 = min(c0 + CHUNK_FRAMES, n_frames)
        nf = c1 - c0
        blk = data[c0 * FRAME:c1 * FRAME].reshape(nf, FRAME, nch).astype(np.float32)
        blk -= blk.mean(axis=1, keepdims=True)
        spec = np.fft.rfft(blk, axis=1)[:, kidx, :].astype(np.complex64)
        spec = np.ascontiguousarray(spec.transpose(1, 2, 0))   # (bin, ch, frame)
        w = np.exp(1j * (freqs[kidx[0]] * phase))              # steering, first bin
        for j in range(len(kidx)):
            if j:
                w *= w_step                                     # advance one bin
            b = band_of[j]
            if b >= 0:
                beam = w @ spec[j]                              # (pix, frame)
                out[c0:c1, b, :] += (np.abs(beam) ** 2).T
    out /= float(max(nch, 1)) ** 2                              # relative scale
    out = out.reshape(n_frames, len(BANDS), ROWS, COLS).astype(np.float32)
    out[~np.isfinite(out)] = 0.0
    np.clip(out, 0.0, None, out)
    np.save(args.output, np.ascontiguousarray(out))


if __name__ == '__main__':
    main()
