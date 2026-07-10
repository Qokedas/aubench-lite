# Acoustic camera request

Hi — I run a test rig with a Sorama CAM1K acoustic array (a 32 x 32 board of 1,024
microphones) mounted right next to a video camera, both pointed at the same scene.
The vendor's desktop software shows live "acoustic heatmaps" — images telling me where
sound is coming from, per frequency band — but I need my own program so I can process
recordings in my pipeline.

## What you have

- `/data/capture/` — one raw recording from my rig, exactly as the device writes it:
  a `sound` file and a `container.json` with the device metadata.
- `/data/vendor_export/maps.npy` — the heatmap tensor my old vendor software exported
  for this same recording. Your program's output for this recording should look like
  this. It's the only export I have; I can't produce more.

## What I need

A program in `/workspace/submission/` with a `solution.py` that my pipeline invokes as:

```bash
python solution.py --capture <capture_dir> --output <maps.npy>
```

where `<capture_dir>` contains a `sound` and `container.json` like mine. It has to run
offline on a modest machine (a few minutes, a few GB of RAM, no network, only the
Python packages already installed here) because a colleague will run it on new
recordings from this rig that you and I have never seen.

Output requirements, so it drops into my pipeline:

- `.npy`, `float32`, shaped `[frame, band, 24, 32]`.
- One frame per 8,192 consecutive samples, in order, ignoring any leftover samples at
  the end of the recording.
- Each 24 x 32 map must line up with my camera image: standard pinhole view, 70.42
  degrees wide by 43.3 degrees tall, row 0 at the top of the image, column 0 at the
  left, one viewing direction per pixel.
- 11 frequency bands, in this order (lower / center / upper Hz, from the vendor
  manual):

```text
282 / 315 / 355       355 / 400 / 447       447 / 500 / 562
562 / 630 / 708       708 / 800 / 891       891 / 1000 / 1122
1122 / 1250 / 1413    1413 / 1600 / 1778    1778 / 2000 / 2239
2239 / 2500 / 2818    2840 / 4000 / 5680
```

- Values are plain linear power (relative scale is fine — my pipeline applies its own
  dB scaling for display). Finite, none negative, not in dB.

Also render me `/workspace/deliverables/preview.mp4` — the heatmaps for this recording
overlaid or tiled however you think reads best — so I can eyeball what it's doing.

## Checking and handing off

`qokedas-check /workspace/submission` runs your program on my recording and verifies
the output loads and matches the contract above. `qokedas-submit /workspace/submission`
validates the final directory (regular files only, at most 1,000 files / 100 MB).
Run it when you're done.

That's everything I know. The rest — how the device data is laid out, what the
metadata means, and how to turn microphone signals into images — is why I'm hiring you.
