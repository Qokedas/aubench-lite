# CAM1K acoustic heatmap generator

Usage:

    python solution.py --capture <capture_dir> --output <maps.npy>

`<capture_dir>` must contain the raw `sound` file and `container.json` as
written by the rig.

## What it does

Frequency-domain delay-and-sum beamforming over the 1024-microphone array:

1. **Decode** — `sound` is int32 PCM, sample-interleaved over all channels.
   File channel `i` belongs to microphone `channelOrdering[i]`, whose position
   (metres, array plane z=0) comes from `frontEnd.microphonePositions`.
   This assignment was validated empirically: it is the only one for which
   inter-channel coherence decays with physical microphone distance.
   Sample rate is derived from `duration` and the sample count (46875 Hz for
   this hardware).
2. **Framing** — one output frame per 8192 consecutive samples, leftover
   samples at the end discarded.
3. **Beamforming** — per frame, channels are DC-removed and FFT'd; for each
   FFT bin inside a band the spectra are phase-aligned (far-field plane-wave
   steering at c = 343 m/s) toward each pixel's viewing direction and summed;
   squared magnitudes accumulate into that band's map.
4. **Pixels** — 24 x 32 pinhole grid, 70.42 x 43.3 degrees, row 0 = top,
   col 0 = left, one direction per pixel centre.
5. **Bands** — the 11 vendor bands (315 ... 2500 Hz third-octaves plus the
   wide 2840-5680 Hz band); a bin belongs to band `[lower, upper)`.

Output: `.npy`, float32, `[frame, band, 24, 32]`, linear relative power
(finite, non-negative, not dB).

Runtime: ~7 s and < 2 GB RAM for a 2.5 s capture; scales linearly with
duration, memory bounded by fixed-size frame chunks.
