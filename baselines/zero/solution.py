#!/usr/bin/env python3
"""Zero-output baseline: valid ABI, no signal processing. Should grade to ~0."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

BLOCK = 8_192
CHANNELS = 1_024


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--capture", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    samples = (Path(args.capture) / "sound").stat().st_size // (4 * CHANNELS)
    shape = (samples // BLOCK, 11, 24, 32)
    np.save(args.output, np.zeros(shape, dtype=np.float32), allow_pickle=False)


if __name__ == "__main__":
    main()
