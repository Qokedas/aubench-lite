#!/usr/bin/env python3
"""Offline delay-and-sum acoustic-camera renderer for Sorama CAM 1K captures.

The CAM writes sample-major, signed fixed-point PCM.  Geometry and the stream's
channel order are read from container.json; no device-specific calibration file
or network service is needed.
"""
import argparse
import json
import os
from pathlib import Path
import numpy as np

FRAME = 8192
SOUND_SPEED = 343.0
BANDS = np.array([
    (282., 355.), (355., 447.), (447., 562.), (562., 708.),
    (708., 891.), (891., 1122.), (1122., 1413.), (1413., 1778.),
    (1778., 2239.), (2239., 2818.), (2840., 5680.)], dtype=np.float64)


def sound_description(container):
    """Find the sound metadata despite the small schema variations in exports."""
    meta = container.get("metadata", {}).get("files", {})
    snd = meta.get("sound", {})
    fe = snd.get("frontEnd", {})
    positions = fe.get("microphonePositions", snd.get("microphonePositions", []))
    if not positions:
        raise ValueError("container.json has no microphone positions")
    rate = fe.get("supportedSampleRates", snd.get("supportedSampleRates", [46875.]))[0]
    ordering = snd.get("channelOrdering", list(range(len(positions))))
    selected = snd.get("channelSelection", [True] * len(ordering))
    p = np.asarray([[q["X"], q["Y"]] for q in positions], dtype=np.float64)
    ordering = np.asarray(ordering, dtype=np.int64)
    selected = np.asarray(selected, dtype=bool)
    if selected.size != ordering.size:
        selected = np.ones(ordering.size, dtype=bool)
    if np.any(ordering < 0) or np.any(ordering >= len(p)):
        raise ValueError("invalid channelOrdering in container.json")
    # The ordering lists the physical microphone for each stored sound channel.
    return float(rate), p[ordering], selected


def image_directions():
    """Camera pinhole rays, x right and y up (hence row 0 is image top)."""
    tx = np.tan(np.deg2rad(70.42 * .5))
    ty = np.tan(np.deg2rad(43.3 * .5))
    x = ((np.arange(32, dtype=np.float64) + .5) / 32. * 2. - 1.) * tx
    y = (1. - (np.arange(24, dtype=np.float64) + .5) / 24. * 2.) * ty
    xx, yy = np.meshgrid(x, y)
    ray = np.stack((xx.ravel(), yy.ravel()), axis=1)
    # x and y are tangents on the z=1 pinhole plane.  Only the in-plane
    # components are needed for delays, but normalize them to unit 3-D rays.
    ray /= np.sqrt(1. + np.sum(ray * ray, axis=1))[:, None]
    return ray


def make_uniform_grid(positions, active):
    """Return channel indices in physical y,x order for a rectangular array."""
    pos = positions[active]
    # Rounding absorbs the tiny decimal representation differences in metadata.
    xs = np.unique(np.round(pos[:, 0], 5))
    ys = np.unique(np.round(pos[:, 1], 5))
    if xs.size * ys.size != pos.shape[0]:
        raise ValueError("CAM 1K geometry is not a complete rectangular grid")
    grid = np.empty((ys.size, xs.size), dtype=np.int64)
    used = np.zeros(pos.shape[0], dtype=bool)
    for i, (x, y) in enumerate(pos):
        iy = int(np.argmin(np.abs(ys - y)))
        ix = int(np.argmin(np.abs(xs - x)))
        if used[iy * xs.size + ix]:
            raise ValueError("duplicate microphone position")
        grid[iy, ix] = i
        used[iy * xs.size + ix] = True
    dx = float(np.median(np.diff(xs))) if xs.size > 1 else 0.
    dy = float(np.median(np.diff(ys))) if ys.size > 1 else 0.
    if dx <= 0. or dy <= 0.:
        raise ValueError("invalid microphone spacing")
    return grid, dx, dy


def bilinear_periodic(a, x, y):
    """Bilinear samples of a periodic spatial DFT at fractional bin positions."""
    h, w = a.shape
    x = np.mod(x, w); y = np.mod(y, h)
    x0 = np.floor(x).astype(np.intp); y0 = np.floor(y).astype(np.intp)
    x1 = (x0 + 1) % w; y1 = (y0 + 1) % h
    wx = x - x0; wy = y - y0
    return (a[y0, x0] * (1.-wx) * (1.-wy) +
            a[y0, x1] * wx * (1.-wy) +
            a[y1, x0] * (1.-wx) * wy +
            a[y1, x1] * wx * wy)


def load_pcm(path, channels, expected_samples=None):
    """Memory-map CAM fixed point samples, accepting 16 or 32 bit exports."""
    size = os.path.getsize(path)
    if expected_samples and expected_samples > 0:
        bytes_per = size / (channels * expected_samples)
        # duration is often rounded; only use it when it identifies a format.
        if abs(bytes_per - 2) < .02:
            return np.memmap(path, dtype='<i2', mode='r', shape=(size // (2*channels), channels)), 32768.
    if size % (4 * channels) == 0:
        # CAM 1K recordings normally use signed 32-bit Q8.24 fixed-point words.
        return np.memmap(path, dtype='<i4', mode='r', shape=(size // (4*channels), channels)), 16777216.
    if size % (2 * channels) == 0:
        return np.memmap(path, dtype='<i2', mode='r', shape=(size // (2*channels), channels)), 32768.
    raise ValueError("sound size is not an interleaved PCM stream")


def render(capture, output):
    capture = Path(capture)
    with open(capture / 'container.json', 'r', encoding='utf-8') as f:
        container = json.load(f)
    fs, positions, selected = sound_description(container)
    # In normal recordings all selected channels are written.  If a selection is
    # recorded as a compact stream, retain exactly those corresponding positions.
    pcm, full_scale = load_pcm(capture / 'sound', len(positions),
                               int(round(container.get('metadata', {}).get('files', {}).get('sound', {}).get('duration', 0) * fs / 1000.)))
    if not np.any(selected):
        raise ValueError("no selected microphones")
    if pcm.shape[1] == selected.size:
        active = selected
        columns = np.flatnonzero(selected)
    else:
        # Some older writers omit disabled channels from the file.
        if pcm.shape[1] != int(selected.sum()):
            raise ValueError("sound channel count does not match container metadata")
        active = selected
        columns = np.arange(pcm.shape[1])
    grid, dx, dy = make_uniform_grid(positions, active)
    nmic = int(grid.size)
    rays = image_directions()
    frames = pcm.shape[0] // FRAME
    result = np.zeros((frames, len(BANDS), 24 * 32), dtype=np.float64)

    # Zero padding gives accurate fractional spatial-frequency samples while
    # retaining the O(F log F) 2-D FFT of the regular 32 x 32 microphone grid.
    # 256 provides sub-pixel interpolation error well below the camera pixels.
    padded = 256
    bins = np.arange(1, min(FRAME // 2, int(np.floor(BANDS[-1, 1] * FRAME / fs))) + 1)
    freqs = bins.astype(np.float64) * fs / FRAME
    which = np.full(bins.size, -1, dtype=np.intp)
    for b, (low, high) in enumerate(BANDS):
        which[(freqs >= low) & (freqs <= high)] = b

    for frame in range(frames):
        samples = np.asarray(pcm[frame*FRAME:(frame+1)*FRAME, columns], dtype=np.float32)
        samples /= full_scale
        # Per-channel DC removal prevents ADC offsets from leaking into low bins.
        samples -= samples.mean(axis=0, keepdims=True)
        spectrum = np.fft.rfft(samples, axis=0)
        for j, k in enumerate(bins):
            band = which[j]
            if band < 0:
                continue
            f = freqs[j]
            spatial = np.fft.fft2(spectrum[k, grid], s=(padded, padded))
            # fft2 uses exp(-i 2pi q n); steering to a ray uses +i phase.
            qx = -f * dx * rays[:, 0] / SOUND_SPEED * padded
            qy = -f * dy * rays[:, 1] / SOUND_SPEED * padded
            steered = bilinear_periodic(spatial, qx, qy)
            # Divide by mic count for DAS and N for the temporal DFT.  Positive
            # rFFT bins represent both sides of a real spectrum, hence factor 2.
            result[frame, band] += 2. * (steered.real**2 + steered.imag**2) / (nmic*nmic*FRAME*FRAME)

    result = np.maximum(result, 0.).astype(np.float32).reshape(frames, len(BANDS), 24, 32)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.save(output, result)


def main():
    ap = argparse.ArgumentParser(description='Render CAM 1K acoustic power maps')
    ap.add_argument('--capture', required=True)
    ap.add_argument('--output', required=True)
    args = ap.parse_args()
    render(args.capture, args.output)

if __name__ == '__main__':
    main()
