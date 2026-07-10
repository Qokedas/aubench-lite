"""Trusted skill-based benchmark metrics.

Every (frame, band) map is scored as skill above an "uninformed floor": the best score
any fixed, audio-independent output could achieve on that map. The floor is the max
over a family of static baselines built from the only public artifact (the vendor
export anchor) plus climatology templates the grader computes from the hidden
references themselves. Replaying, blurring, gamma-warping, amplifying, or flipping
public data scores ~0 by construction; per-map floor headroom weighting removes free
credit from maps that physics makes uninformative.

All metric primitives are vectorized over stacked maps [K, H, W] so grading a full
private set takes seconds on a laptop CPU.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.ndimage import gaussian_filter

from .constants import BAND_CENTERS, DYNAMIC_RANGE_DB, GRID_HEIGHT, GRID_WIDTH

_MIN_HEADROOM_WEIGHT = 0.05
_SSIM_C1 = 0.01**2
_SSIM_C2 = 0.03**2


# ---------------------------------------------------------------------------
# Vectorized metric primitives over stacked maps [K, H, W]
# ---------------------------------------------------------------------------


def canonicalize_batch(power: np.ndarray, dynamic_range_db: float = DYNAMIC_RANGE_DB) -> np.ndarray:
    """Per-map dB canonicalization of a [K, H, W] stack onto [0, 1]."""
    value = np.asarray(power, dtype=np.float64)
    maximum = value.max(axis=(1, 2), keepdims=True)
    tiny = np.finfo(np.float64).tiny
    safe_max = np.maximum(maximum, tiny)
    db = 10.0 * np.log10(np.maximum(value, tiny) / safe_max)
    canon = (np.clip(db, -dynamic_range_db, 0.0) + dynamic_range_db) / dynamic_range_db
    return np.where(maximum > tiny, canon, 0.0)


def _ssim_batch(candidate: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """Local SSIM per map with a small Gaussian window suitable for 24x32 maps."""
    sigma = (0.0, 1.0, 1.0)
    blur = lambda t: gaussian_filter(t, sigma=sigma, mode="reflect")  # noqa: E731
    mu_x = blur(candidate)
    mu_y = blur(reference)
    sigma_x = blur(candidate * candidate) - mu_x**2
    sigma_y = blur(reference * reference) - mu_y**2
    sigma_xy = blur(candidate * reference) - mu_x * mu_y
    numerator = (2.0 * mu_x * mu_y + _SSIM_C1) * (2.0 * sigma_xy + _SSIM_C2)
    denominator = (mu_x**2 + mu_y**2 + _SSIM_C1) * (sigma_x + sigma_y + _SSIM_C2)
    return np.clip((numerator / np.maximum(denominator, 1e-12)).mean(axis=(1, 2)), 0.0, 1.0)


def _pearson_batch(candidate: np.ndarray, reference: np.ndarray) -> np.ndarray:
    a = candidate.reshape(candidate.shape[0], -1)
    b = reference.reshape(reference.shape[0], -1)
    a = a - a.mean(axis=1, keepdims=True)
    b = b - b.mean(axis=1, keepdims=True)
    denominator = np.sqrt((a * a).sum(axis=1) * (b * b).sum(axis=1))
    correlation = np.where(
        denominator > 1e-12, (a * b).sum(axis=1) / np.maximum(denominator, 1e-12), 0.0
    )
    return np.clip(correlation, 0.0, 1.0)


def _locations_batch(maps: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Normalized (x, y) peak and energy-centroid per map. Shapes [K, 2]."""
    count, height, width = maps.shape
    flat_argmax = maps.reshape(count, -1).argmax(axis=1)
    rows, columns = np.unravel_index(flat_argmax, (height, width))
    peak = np.stack([columns / max(width - 1, 1), rows / max(height - 1, 1)], axis=1)

    yy, xx = np.meshgrid(
        np.linspace(0.0, 1.0, height), np.linspace(0.0, 1.0, width), indexing="ij"
    )
    weight = np.maximum(maps, 0.0)
    total = weight.sum(axis=(1, 2))
    safe_total = np.maximum(total, 1e-12)
    centroid_x = (weight * xx).sum(axis=(1, 2)) / safe_total
    centroid_y = (weight * yy).sum(axis=(1, 2)) / safe_total
    centroid = np.stack([centroid_x, centroid_y], axis=1)
    centroid = np.where(total[:, None] > 1e-12, centroid, 0.5)
    return peak, centroid


def _hotspot_batch(
    candidate: np.ndarray, reference: np.ndarray, centers_hz: np.ndarray
) -> np.ndarray:
    candidate_peak, candidate_centroid = _locations_batch(candidate)
    reference_peak, reference_centroid = _locations_batch(reference)
    sigma = np.clip(0.10 * np.sqrt(1_000.0 / centers_hz), 0.035, 0.20)
    peak_distance = np.linalg.norm(candidate_peak - reference_peak, axis=1)
    centroid_distance = np.linalg.norm(candidate_centroid - reference_centroid, axis=1)
    peak = np.exp(-(peak_distance**2) / (2.0 * sigma**2))
    centroid = np.exp(-(centroid_distance**2) / (2.0 * sigma**2))
    return 0.60 * peak + 0.40 * centroid


def _blank_flags(canonical: np.ndarray) -> np.ndarray:
    """A map is blank when it is (near-)constant: no spatial structure at all."""
    stds = canonical.std(axis=(1, 2))
    few_levels = np.array(
        [np.unique(np.round(m, 9)).size <= 2 for m in canonical], dtype=bool
    )
    return (stds < 1e-7) | few_levels


def raw_map_metrics(
    candidate: np.ndarray, reference: np.ndarray, centers_hz: np.ndarray
) -> dict[str, np.ndarray]:
    """Raw (pre-skill) per-map metrics for stacked [K, H, W] linear-power maps."""
    candidate_canonical = canonicalize_batch(candidate)
    reference_canonical = canonicalize_batch(reference)
    candidate_zero = candidate_canonical.max(axis=(1, 2)) == 0.0
    reference_zero = reference_canonical.max(axis=(1, 2)) == 0.0
    both_zero = candidate_zero & reference_zero
    blank = (
        _blank_flags(candidate_canonical)
        & (reference_canonical.std(axis=(1, 2)) >= 1e-5)
    )

    ssim = _ssim_batch(candidate_canonical, reference_canonical)
    correlation = _pearson_batch(candidate_canonical, reference_canonical)
    l1 = np.clip(
        1.0 - np.abs(candidate_canonical - reference_canonical).mean(axis=(1, 2)), 0.0, 1.0
    )
    map_score = 0.50 * ssim + 0.30 * correlation + 0.20 * l1
    hotspot = _hotspot_batch(candidate_canonical, reference_canonical, centers_hz)

    map_score = np.where(blank, 0.0, np.where(both_zero, 1.0, map_score))
    hotspot = np.where(blank, 0.0, np.where(both_zero, 1.0, hotspot))
    return {
        "map_score": map_score,
        "hotspot": hotspot,
        "ssim": np.where(both_zero, 1.0, np.where(blank, 0.0, ssim)),
        "correlation": np.where(both_zero, 1.0, np.where(blank, 0.0, correlation)),
        "l1_similarity": np.where(both_zero, 1.0, np.where(blank, 0.0, l1)),
        "blank": blank,
        "both_zero": both_zero,
    }


# ---------------------------------------------------------------------------
# Uninformed floor family
# ---------------------------------------------------------------------------


def _blob_prior() -> np.ndarray:
    """Audio-independent physics prior: centered blob, width scaled by wavelength."""
    yy, xx = np.mgrid[0:GRID_HEIGHT, 0:GRID_WIDTH]
    maps = np.empty((len(BAND_CENTERS), GRID_HEIGHT, GRID_WIDTH), dtype=np.float64)
    for index, center in enumerate(BAND_CENTERS):
        sigma = float(np.clip(8.0 * 1_000.0 / center, 1.5, 12.0))
        maps[index] = np.exp(
            -(((xx - GRID_WIDTH / 2) ** 2 + (yy - GRID_HEIGHT / 2) ** 2) / (2 * sigma * sigma))
        )
    return maps


def build_floor_family(
    anchor: np.ndarray, climatologies: list[np.ndarray] | None = None
) -> list[np.ndarray]:
    """Static per-band templates [11, H, W] forming the empirical uninformed floor.

    ``anchor`` is the public anchor tensor [F, 11, H, W]. ``climatologies`` are
    grader-computed per-band means over hidden references (canonical space). Group
    means are a strong empirical cap on fixed-output strategies, though not a formal
    optimum of the composite SSIM/correlation/L1 metric.
    """
    anchor = np.asarray(anchor, dtype=np.float64)
    first = anchor[0]
    mean = anchor.mean(axis=0)
    family = [
        first,
        mean,
        gaussian_filter(first, sigma=(0.0, 1.0, 1.0), mode="reflect"),
        gaussian_filter(first, sigma=(0.0, 2.5, 2.5), mode="reflect"),
        np.power(first, 0.5),
        np.power(first, 2.0),
        np.power(first, 4.0),
        first[:, :, ::-1],
        first[:, ::-1, :],
        np.roll(first, (3, 4), axis=(1, 2)),
        np.roll(first, (-3, -4), axis=(1, 2)),
        _blob_prior(),
        gaussian_filter(_blob_prior(), sigma=(0.0, 2.0, 2.0), mode="reflect"),
    ]
    for climatology in climatologies or []:
        family.append(np.asarray(climatology, dtype=np.float64) + 1e-9)
    return [np.ascontiguousarray(member) for member in family]


def case_floors(
    family: list[np.ndarray], reference: np.ndarray, centers_hz: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Max raw map/hotspot score any family member achieves on each map of a case."""
    frames = reference.shape[0]
    flat_reference = reference.reshape(-1, reference.shape[2], reference.shape[3])
    tiled_centers = np.tile(centers_hz, frames)
    floor_map = np.zeros(flat_reference.shape[0])
    floor_hotspot = np.zeros(flat_reference.shape[0])
    for member in family:
        candidate = np.broadcast_to(
            member[None], (frames, *member.shape)
        ).reshape(flat_reference.shape)
        metrics = raw_map_metrics(candidate, flat_reference, tiled_centers)
        floor_map = np.maximum(floor_map, metrics["map_score"])
        floor_hotspot = np.maximum(floor_hotspot, metrics["hotspot"])
    # A perfectly zero reference is handled by the both-zero rule, not by floors.
    reference_zero = canonicalize_batch(flat_reference).max(axis=(1, 2)) == 0.0
    floor_map = np.where(reference_zero, 0.0, np.minimum(floor_map, 0.999))
    floor_hotspot = np.where(reference_zero, 0.0, np.minimum(floor_hotspot, 0.999))
    return floor_map, floor_hotspot


# ---------------------------------------------------------------------------
# Skill scoring
# ---------------------------------------------------------------------------


def score_case_skill(
    candidate: np.ndarray,
    reference: np.ndarray,
    floors: tuple[np.ndarray, np.ndarray],
) -> dict[str, Any]:
    """Headroom-weighted skill metrics for one case tensor [F, 11, H, W]."""
    if candidate.shape != reference.shape:
        raise ValueError(f"shape mismatch: {candidate.shape} != {reference.shape}")
    centers = np.tile(np.asarray(BAND_CENTERS, dtype=np.float64), reference.shape[0])
    flat_candidate = candidate.reshape(-1, candidate.shape[2], candidate.shape[3])
    flat_reference = reference.reshape(-1, reference.shape[2], reference.shape[3])
    metrics = raw_map_metrics(flat_candidate, flat_reference, centers)
    floor_map, floor_hotspot = floors

    map_skill = np.clip(
        (metrics["map_score"] - floor_map) / np.maximum(1.0 - floor_map, 1e-6), 0.0, 1.0
    )
    hotspot_skill = np.clip(
        (metrics["hotspot"] - floor_hotspot) / np.maximum(1.0 - floor_hotspot, 1e-6),
        0.0,
        1.0,
    )
    map_skill = np.where(metrics["both_zero"], 1.0, map_skill)
    hotspot_skill = np.where(metrics["both_zero"], 1.0, hotspot_skill)

    weights = np.maximum(1.0 - floor_map, _MIN_HEADROOM_WEIGHT)
    weights = np.where(metrics["both_zero"], 1.0, weights)
    weights = weights / weights.sum()
    return {
        "map_skill": float((map_skill * weights).sum()),
        "hotspot_skill": float((hotspot_skill * weights).sum()),
        "raw_map_score": float(metrics["map_score"].mean()),
        "raw_hotspot_score": float(metrics["hotspot"].mean()),
        "ssim": float(metrics["ssim"].mean()),
        "correlation": float(metrics["correlation"].mean()),
        "l1_similarity": float(metrics["l1_similarity"].mean()),
        "blank_fraction": float(metrics["blank"].mean()),
        "mean_floor": float(floor_map.mean()),
        "frame_band_count": int(flat_reference.shape[0]),
    }


def aggregate_case_scores(cases: list[dict[str, Any]]) -> dict[str, Any]:
    real = [case for case in cases if case["kind"] == "real"]
    procedural = [case for case in cases if case["kind"] == "procedural"]
    if not real or not procedural:
        raise ValueError("grading requires real and procedural cases")
    replay = float(np.mean([case["metrics"]["map_skill"] for case in real]))
    hotspot = float(np.mean([case["metrics"]["hotspot_skill"] for case in real]))
    procedural_score = float(
        np.mean(
            [
                0.8 * case["metrics"]["map_skill"] + 0.2 * case["metrics"]["hotspot_skill"]
                for case in procedural
            ]
        )
    )
    overall = 0.60 * replay + 0.20 * hotspot + 0.20 * procedural_score
    return {
        "overall": overall,
        "percentage": 100.0 * overall,
        "replay": replay,
        "hotspot": hotspot,
        "procedural": procedural_score,
        "cases": cases,
    }
