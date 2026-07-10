# AUbench-lite — full results data

Four frontier models, two episodes each (skill@1 per episode), July 9, 2026. Round 1: budget not stated. Round 2: the 80-action budget stated in the prompt. Harness: one bash tool, networkless container. Grading: per-map skill above an uninformed floor vs the frozen reference; hidden set = 7 real raw excerpts + 9 procedural cases, secret channel permutation per run. Every episode scored below 0.035 against the 1.000 reference; defining a pass as a score of at least 0.5, pass@2 = 0 of 4 models.

## Round 1 (budget not stated)

| Model | Overall (0-1) | Real map | Real hotspot | Procedural | Actions | Wall (min) | Cost (USD) |
|---|---|---|---|---|---|---|---|
| Claude Fable 5 | 0.0328 | 0.0023 | 0.0124 | 0.1449 | 80 | 56.9 | $17.74 |
| GPT-5.6 Sol | 0.0275 | 0.0040 | 0.0075 | 0.1180 | 30 | 6.7 | $1.54 |
| Claude Opus 4.8 | 0.0249 | 0.0031 | 0.0035 | 0.1120 | 54 | 33.7 | $5.19 |
| GPT-5.6 Terra | 0.0249 | 0.0004 | 0.0033 | 0.1202 | 21 | 5.0 | $0.79 |

## Round 2 (80-action budget stated)

| Model | Overall (0-1) | Real map | Real hotspot | Procedural | Actions | Wall (min) | Cost (USD) |
|---|---|---|---|---|---|---|---|
| Claude Fable 5 | 0.0265 | 0.0003 | 0.0108 | 0.1206 | 36 | 45.8 | $8.85 |
| GPT-5.6 Sol | 0.0318 | 0.0123 | 0.0041 | 0.1180 | 30 | 11.6 | $2.02 |
| Claude Opus 4.8 | 0.0345 | 0.0158 | 0.0034 | 0.1217 | 34 | 19.6 | $2.12 |
| GPT-5.6 Terra | 0.0295 | 0.0063 | 0.0085 | 0.1202 | 18 | 5.7 | $0.56 |

## Combined (8 episodes)

- mean score **0.0291** · best 0.0345 · worst 0.0249 — i.e., the models scored at the level of the uninformed floor (zero output scores 0.0222; replaying the provided example scores 0.0345)
- mean actions 37.9 of 80 · mean wall 23.1 min
- mean cost $4.85 · total $38.82
- reference points: honest delay-and-sum baseline **1.0000** · zero output 0.0222 · replay-anchor cheat 0.0345 · correct DAS with hardcoded channel order 0.0390

## Token usage and pricing

| Round | Model | Uncached in | Output | Cache read | Cache write | Cost |
|---|---|---|---|---|---|---|
| r1 | Claude Fable 5 | 154 | 151,869 | 7,654,832 | 199,398 | $17.74 |
| r1 | GPT-5.6 Sol | 69,336 | 17,954 | 1,310,865 | 0 | $1.54 |
| r1 | Claude Opus 4.8 | 709 | 96,665 | 3,875,949 | 133,450 | $5.19 |
| r1 | GPT-5.6 Terra | 85,025 | 15,931 | 1,345,021 | 0 | $0.79 |
| r2 | Claude Fable 5 | 72 | 88,338 | 2,060,657 | 189,678 | $8.85 |
| r2 | GPT-5.6 Sol | 86,552 | 24,900 | 1,690,103 | 0 | $2.02 |
| r2 | Claude Opus 4.8 | 721 | 46,581 | 1,087,624 | 65,459 | $2.12 |
| r2 | GPT-5.6 Terra | 64,332 | 12,069 | 878,242 | 0 | $0.56 |

List pricing per MTok: Fable 5 $10/$50, Opus 4.8 $5/$25 (cache read 0.1x, 5-min write 1.25x); GPT-5.6 Sol $5/$30, Terra $2.50/$15 (cached input -90%). OpenAI `input_tokens` includes cached tokens; Anthropic reports them separately.

## Scoring calibration

| Submission | Overall | Interpretation |
|---|---|---|
| Honest delay-and-sum (reference class) | 1.0000 | solvability proof, built only from agent-visible artifacts |
| No DC removal variant | 1.0000 | benign convention |
| Pixel-center grid variant | 0.9025 | benign convention |
| Rectangular window variant | 0.8540 | benign convention |
| Whitened channels | ~0.46 | right algorithm, one wrong choice |
| Center-bin-only shortcut | ~0.19 | real beamforming, cut corners |
| Correct DAS, hardcoded channel order | 0.0390 | killed by secret metadata permutation |
| Replay of public anchor | 0.0345 | memorization floor |
| Zero output | 0.0222 | only the zero-input case passes |

## The shared failure

All eight submissions parsed the raw `sound` stream with a wrong layout; the device writes it channel-major. Seven of eight assumed the sample-interleaved convention common in audio software; Fable round 1 shipped a block-chunked variant. This is the shared decisive failure, not necessarily the only defect in each program: we did not verify that correcting the layout alone repairs every submission. The layout is decidable from provided data four ways: duration arithmetic pins int32/little-endian; per-channel spectra (99.7% vs 60.2% acoustic-band energy); plane-wave focusing contrast (9.5x vs 3.5x); and the provided ground-truth export (+1.000 per-map dB correlation for the correct parse vs ~0.00 for the shipped ones). All eight models used the anchor during development. Opus round 2 explicitly compared sample-major and channel-major parses against it, but that experiment simultaneously ignored `channelOrdering`, so it did not isolate layout and the model still shipped sample-major. Other runs inspected layout evidence without completing a clean anchor-controlled comparison.

## Episode identifiers

| Run | Episode | Path |
|---|---|---|
| r1 Claude Fable 5 | `20260709-200952-claude-fable-5-d40cda71` | `results/r1_fable/` |
| r1 GPT-5.6 Sol | `20260709-200858-gpt-5-6-sol-55696a42` | `results/r1_sol/` |
| r1 Claude Opus 4.8 | `20260709-200953-claude-opus-4-8-05a36ad9` | `results/r1_opus/` |
| r1 GPT-5.6 Terra | `20260709-200859-gpt-5-6-terra-a094472e` | `results/r1_terra/` |
| r2 Claude Fable 5 | `20260709-222704-claude-fable-5-809909ea` | `results/r2_fable/` |
| r2 GPT-5.6 Sol | `20260709-222706-gpt-5-6-sol-84279dd6` | `results/r2_sol/` |
| r2 Claude Opus 4.8 | `20260709-222704-claude-opus-4-8-fef81190` | `results/r2_opus/` |
| r2 GPT-5.6 Terra | `20260709-222707-gpt-5-6-terra-f13ae71c` | `results/r2_terra/` |

Wall time = episode directory creation to last transcript write (agent loop only). Each results/ directory holds report.json, transcript.jsonl, and the frozen submission as graded.
