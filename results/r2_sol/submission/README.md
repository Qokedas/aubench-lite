# CAM1K acoustic map generator

Run from this directory with:

```bash
python solution.py --capture /path/to/capture --output /path/to/maps.npy
```

The program reads the sample rate, channel ordering, and microphone positions
from `container.json`; the `sound` stream is the device's little-endian signed
32-bit, sample-interleaved PCM. It emits `float32` linear-power maps with shape
`[complete_8192_sample_frame, 11, 24, 32]`.

## Processing

Each complete block is channel-DC-rejected, partially common-mode-rejected, and Hann-windowed, then transformed
to the frequency domain. The listed band edges select FFT bins directly. A
far-field delay-and-sum beamformer evaluates the 32 x 32 planar array in the
pinhole direction at every output pixel (70.42 by 43.3 degrees). Spatial FFT
zero-padding and bicubic interpolation make this equivalent to evaluating the
array DFT without constructing very large steering matrices. Results are
one-sided, band-integrated squared pressure, normalized by microphone count.

A streaming deglitcher removes rare, isolated CAM1K ADC pedestal changes. It
does not remove simultaneous array-wide impulses, and its state continues over
frame boundaries. Trailing incomplete samples are ignored.
