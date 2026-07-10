from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter

from cam1k_eval.constants import BAND_CENTERS, BAND_COUNT, GRID_HEIGHT, GRID_WIDTH
from cam1k_eval.scoring import (
    aggregate_case_scores,
    build_floor_family,
    case_floors,
    raw_map_metrics,
    score_case_skill,
)

_CENTERS = np.asarray(BAND_CENTERS, dtype=np.float64)


def _structured_tensor(seed: int, frames: int = 1) -> np.ndarray:
    rng = np.random.default_rng(seed)
    base = rng.random((frames, BAND_COUNT, GRID_HEIGHT, GRID_WIDTH))
    return gaussian_filter(base, sigma=(0, 0, 2, 2)).astype(np.float32) + 1e-6


def _floors_for(reference: np.ndarray, anchor: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return case_floors(build_floor_family(anchor), reference, _CENTERS)


def test_exact_candidate_scores_one() -> None:
    anchor = _structured_tensor(1)
    reference = _structured_tensor(2)
    metrics = score_case_skill(reference, reference, _floors_for(reference, anchor))
    assert metrics["map_skill"] > 0.999
    assert metrics["hotspot_skill"] > 0.999


def test_anchor_replay_scores_zero() -> None:
    anchor = _structured_tensor(1)
    reference = _structured_tensor(2)
    replay = np.broadcast_to(anchor[0], reference.shape).copy()
    metrics = score_case_skill(replay, reference, _floors_for(reference, anchor))
    assert metrics["map_skill"] < 1e-6
    assert metrics["hotspot_skill"] < 1e-6


def test_amplified_anchor_replay_scores_zero() -> None:
    anchor = _structured_tensor(1)
    reference = _structured_tensor(2)
    for warp in (lambda t: t * 1e6, lambda t: t**0.5, lambda t: t**2.0):
        cheat = np.broadcast_to(warp(anchor[0].astype(np.float64)), reference.shape)
        metrics = score_case_skill(
            cheat.astype(np.float32), reference, _floors_for(reference, anchor)
        )
        assert metrics["map_skill"] < 0.02, warp


def test_blank_candidate_is_gated() -> None:
    reference = np.zeros((1, BAND_COUNT, GRID_HEIGHT, GRID_WIDTH), dtype=np.float32)
    reference[..., 3:6, 8:12] = 1.0
    blank = np.zeros_like(reference)
    anchor = _structured_tensor(1)
    metrics = score_case_skill(blank, reference, _floors_for(reference, anchor))
    assert metrics["map_skill"] == 0.0
    assert metrics["blank_fraction"] == 1.0


def test_both_zero_case_passes() -> None:
    reference = np.zeros((1, BAND_COUNT, GRID_HEIGHT, GRID_WIDTH), dtype=np.float32)
    anchor = _structured_tensor(1)
    metrics = score_case_skill(reference, reference, _floors_for(reference, anchor))
    assert metrics["map_skill"] == 1.0
    assert metrics["hotspot_skill"] == 1.0


def test_raw_metrics_batch_shapes() -> None:
    candidate = _structured_tensor(3)[0]
    reference = _structured_tensor(4)[0]
    metrics = raw_map_metrics(candidate, reference, _CENTERS)
    for key in ("map_score", "hotspot", "ssim", "correlation", "l1_similarity"):
        assert metrics[key].shape == (BAND_COUNT,)
        assert np.all(metrics[key] >= 0.0) and np.all(metrics[key] <= 1.0)


def test_aggregate_weights() -> None:
    cases = [
        {"kind": "real", "metrics": {"map_skill": 1.0, "hotspot_skill": 1.0}},
        {"kind": "procedural", "metrics": {"map_skill": 0.5, "hotspot_skill": 0.5}},
    ]
    aggregate = aggregate_case_scores(cases)
    assert abs(aggregate["overall"] - (0.6 * 1.0 + 0.2 * 1.0 + 0.2 * 0.5)) < 1e-12
