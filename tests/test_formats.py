from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from cam1k_eval.constants import MICROPHONE_COUNT
from cam1k_eval.formats import (
    FormatError,
    expected_map_shape,
    open_capture,
    validate_maps,
    write_capture,
)
from cam1k_eval.geometry import make_far_field_grid


def test_expected_shape_and_output_validation() -> None:
    expected = expected_map_shape(2 * 8_192 + 100)
    assert expected == (2, 11, 24, 32)
    validate_maps(np.zeros(expected, dtype=np.float32), expected)
    with pytest.raises(FormatError, match="negative"):
        validate_maps(np.full(expected, -1, dtype=np.float32), expected)
    with pytest.raises(FormatError, match="dtype"):
        validate_maps(np.zeros(expected, dtype=np.float64), expected)


def test_grid_contains_unit_rays() -> None:
    grid = make_far_field_grid(24, 32)
    assert grid.shape == (24, 32, 3)
    np.testing.assert_allclose(np.linalg.norm(grid, axis=2), 1.0, atol=1e-12)
    assert grid[0, 0, 0] < 0  # column 0 is image-left
    assert grid[0, 0, 1] > 0  # row 0 is image-top


def _minimal_container(ordering: np.ndarray) -> dict:
    rng = np.random.default_rng(0)
    positions = rng.uniform(-0.3, 0.3, size=(MICROPHONE_COUNT, 3))
    positions[:, 2] = 0.0
    return {
        "metadata": {
            "files": {
                "sound": {
                    "duration": 0,
                    "channelOrdering": [int(v) for v in ordering],
                    "channelSelection": [True] * MICROPHONE_COUNT,
                    "frontEnd": {
                        "supportedSampleRates": [46875.0],
                        "microphonePositions": [
                            {"X": float(x), "Y": float(y), "Z": float(z)}
                            for x, y, z in positions
                        ],
                    },
                }
            }
        }
    }


def test_capture_roundtrip(tmp_path: Path) -> None:
    rng = np.random.default_rng(1)
    ordering = rng.permutation(MICROPHONE_COUNT)
    sound = rng.integers(-1000, 1000, size=(MICROPHONE_COUNT, 512), dtype=np.int32).astype("<i4")
    write_capture(tmp_path / "capture", sound, _minimal_container(ordering), ordering)
    capture = open_capture(tmp_path / "capture")
    assert capture["num_samples"] == 512
    np.testing.assert_array_equal(np.asarray(capture["sound"]), sound)
    np.testing.assert_array_equal(capture["ordering"], ordering)
    np.testing.assert_allclose(
        capture["positions_by_row"], capture["positions_by_channel_id"][ordering]
    )


def test_capture_rejects_bad_ordering(tmp_path: Path) -> None:
    ordering = np.zeros(MICROPHONE_COUNT, dtype=np.int64)
    sound = np.zeros((MICROPHONE_COUNT, 512), dtype="<i4")
    with pytest.raises(FormatError, match="permutation"):
        write_capture(tmp_path / "capture", sound, _minimal_container(ordering), ordering)
