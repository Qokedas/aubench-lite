# CAM1K acoustic mapper

Run:

```bash
python solution.py --capture /path/to/capture --output /path/to/maps.npy
```

The program reads the sample rate, channel ordering, microphone coordinates, and
channel count from `container.json`. It interprets the interleaved `sound` stream
as little-endian signed 32-bit CAM fixed-point samples. Every complete 8,192-sample
block is Hann-windowed and transformed independently; trailing samples are not
used.

For each requested band, `solution.py` integrates a conventional far-field
frequency-domain delay-and-sum power map. Camera rays pass through pixel centres
of a 70.42° x 43.3° pinhole image, with top-to-bottom row and left-to-right column
ordering. The steering sum is evaluated efficiently as a finely zero-padded
spatial FFT of the uniform microphone grid. Output is a nonnegative `float32`
NumPy array in `[frame, band, 24, 32]` order. Values are linear relative power,
not decibels.
