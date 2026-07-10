# Reference solution walkthrough (HOST-SIDE ONLY — never mounted)

This document proves, constructively, that the benchmark is solvable using **only**
what the evaluated agent sees:

1. `/workspace/BRIEF.md` — the client brief,
2. `/data/capture/` — one raw recording (`sound` + `container.json`),
3. `/data/vendor_export/maps.npy` — the anchor tensor for that same recording.

Every step below names the artifact it uses and shows the measurement that forces the
decision. All quoted numbers were computed against the built benchmark with exactly
these inputs. Nothing here relies on `cam1k_eval/`, the reference source, or any
private data.

**Containment.** This file must never enter the agent or runner images or mounts.
The agent sees `/workspace` (episode dir seeded with `BRIEF.md` only) and `/data`
(`data/benchmark/public`). `docs/` appears in no Dockerfile `COPY`, no compose
volume, and no `docker run --mount`. See the leak audit at the bottom.

---

## Step 0 — Extract the contract from the brief

The brief fixes (nothing else is fixed anywhere):

- ABI: `python solution.py --capture <dir> --output <maps.npy>`
- Output: `float32 [frame, band, 24, 32]`, linear power, finite, nonnegative
- One frame per 8,192 consecutive samples, trailing partial dropped
- View: pinhole, 70.42° x 43.3°, row 0 = image top, column 0 = image left
- 11 bands with explicit lower/upper edges

## Step 1 — Read the metadata (`container.json`)

```python
sm = json.load(open(capture / "container.json"))["metadata"]["files"]["sound"]
```

Facts found there: `microphoneCount = 1024`; `supportedSampleRates = [46875.0]`;
`duration` in ms; `channelOrdering` (a permutation of 0..1023);
`frontEnd.microphonePositions` (1,024 XYZ records in metres, a planar 32 x 32 grid
spanning ±0.31 m at z = 0); `channelSelection` (all true).

## Step 2 — Decode the `sound` stream (three sub-decisions, all forced)

**Sample width.** `file_bytes / 1024 / 46875` per candidate width vs the stated
duration (2,500 ms on the provided capture):

```
int16 -> 5.0000 s   int32 -> 2.5000 s   int64 -> 1.2500 s
```

Only int32 is consistent. *(Data used: capture only.)*

**Endianness.** As `<i4`, |max| ≈ 2.9e8 — comfortable inside int32. As `>i4`, values
saturate the type (≈ 2.147e9). Little-endian. *(Capture only.)*

**Channel-major vs interleaved.** Interpret the buffer both ways and look at one
"channel"'s spectrum. Channel-major: 99.7% of energy above 100 Hz with structured
acoustic peaks. Interleaved: stride-1024 decimation aliases everything into rumble
(≈ 60% above 100 Hz, peak parked at the bottom). Channel-major, i.e. shape
`(1024, N)` in C order. *(Capture only.)*

> Trap note, verified empirically: naive lag-1 "smoothness" points the **wrong way**
> here (decimated rumble is smoother than real wideband audio). Use spectra, not
> smoothness.

## Step 3 — Map stream rows to microphone positions

Two hypotheses: row *i* is the channel `ordering[i]` (positions `pos[ordering]`), or
the inverse (`pos[argsort(ordering)]`). Two independent discriminators, both
available inside:

- **Physics (capture only).** Build any provisional beamformer (Step 5) and compare
  focus quality: correct assignment yields peak/median contrast ≈ 9.5 at 1 kHz; the
  inverse ≈ 3.5; ignoring the permutation ≈ 2.5. Only correct phase alignment
  focuses.
- **Anchor (capture + vendor export).** Per-map dB correlation of the two variants'
  output against `vendor_export/maps.npy`: **+1.000 vs +0.148**.

> Verified dead end: no pairwise statistic (broadband correlation, band-limited
> coherence, Welch MSC) distinguishes the hypotheses — a field dominated by one
> far-field source is coherent at every mic pair. The geometry lives in the phase
> structure that only focusing reads. You must do the physics.

## Step 4 — Build the viewing grid (brief only)

"Standard pinhole" means uniform sampling of the image plane, i.e. tangent-spaced:

```python
tx, ty = tan(radians(70.42 / 2)), tan(radians(43.3 / 2))
x = np.linspace(-tx, tx, 32)          # column 0 = left
y = np.linspace(ty, -ty, 24)          # row 0 = top
grid = normalize(stack([xx, yy, ones]))   # unit rays, +z toward the scene
```

The only residual conventions (pixel edges vs centers; +z sign pairing with the
steering sign) are settled in Step 6 against the anchor.

## Step 5 — Derive the beamformer (first principles)

Signal model: a far-field source from unit direction **d** reaches microphone *m* at
position **p**ₘ earlier by τₘ = (**d** · **p**ₘ)/c, c ≈ 343 m/s. Per frame of 8,192
samples: FFT each channel, and for every pixel direction and every FFT bin *f* in a
band, phase-align and sum:

```
P(pixel, band) = mean over bins f in band of  | Σₘ e^(−2πi f τₘ(pixel)) · Xₘ(f) |²
```

Delay-and-sum. Any positive global scale is fine (the brief says relative scale).

## Step 6 — Calibrate every free choice against the vendor export

Compute your maps for the provided capture and correlate per-map dB images with the
anchor. Measured outcomes:

| variant | anchor match |
|---|---|
| steering e^(−iωτ), full band, edge-spaced grid | **+1.000** |
| wrong steering sign | −0.012 |
| wrong ordering direction | +0.148 |
| y-axis flipped | +0.393 |
| x-axis mirrored | −0.082 |
| center FFT bin only per band | +0.124 |
| pixel centers instead of edges | +0.995 |
| rectangular window | +0.999 |

Every genuine error sits at ≤ +0.4; every benign convention sits at ≥ +0.99 (and the
benign ones also survive hidden grading at 0.85–1.00, verified). The anchor is a
one-experiment oracle for each fork, and there are only ~6 forks.

## Step 7 — Validate before shipping (this is where GPT-5.6 Sol failed)

Two checks, both possible entirely inside:

1. **Anchor diff.** Per-map dB correlation vs `vendor_export/maps.npy` should be
   ≈ 1.0 and peak locations should coincide (within a pixel). If your 1 kHz peak is
   rows away from the anchor's, your geometry is wrong — do not ship.
2. **Synthetic self-test.** Simulate a plane wave from a known direction through your
   own pipeline (delays from the model in Step 5) and confirm the hotspot lands on
   the predicted pixel in the predicted band. This tests your code against your own
   physics with no external data at all.

## Step 8 — Package

Vectorize the per-bin loop (chunked steering matrices; ~15 s per frame naive Python
is fine within the budget), handle arbitrary sample counts with trailing-block
dropping, read *everything* per-invocation from the given capture directory (the
colleague's recordings may differ in length and metadata detail — honor
`channelOrdering` from each file, never bake in row order), write the tensor, render
the MP4 for the visible capture, run `qokedas-check` and `qokedas-submit`.

A complete implementation assembled from exactly these decisions is
`baselines/delay_and_sum/solution.py`; it scores **1.0000** on the hidden set through
the full pipeline, including the secret per-run channel permutations.

---

## Solvability audit

| step | data used | forced by evidence? |
|---|---|---|
| contract | brief | stated |
| metadata facts | capture | stated in file |
| width / endian / layout | capture | yes (3 decisive measurements) |
| ordering direction | capture (+ anchor) | yes (9.5 vs 3.5; +1.00 vs +0.15) |
| view grid | brief (+ anchor for conventions) | yes |
| algorithm | physics | textbook derivation |
| sign / edges / band integration | capture + anchor | yes (table above) |
| window / DC | — | immaterial (≥ +0.99 match; 0.85–1.00 hidden) |
| framing | brief (+ anchor: 14 frames / 117,187 samples) | stated twice |

Every row consumes only the three provided artifacts. A model that (a) measures
instead of assumes, and (b) diffs its output against the anchor before shipping,
solves this benchmark. That is the intended skill ceiling — and the intended floor.
