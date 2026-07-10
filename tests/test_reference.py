from __future__ import annotations

import numpy as np

from cam1k_eval.constants import BLOCK_LENGTH, SAMPLE_RATE_HZ
from cam1k_eval.geometry import make_far_field_grid
from cam1k_eval.reference import beamform
from cam1k_eval.synthetic import PlaneWave, direction, plane_wave_audio


def _small_array() -> np.ndarray:
    x, y = np.meshgrid(np.linspace(-0.15, 0.15, 8), np.linspace(-0.15, 0.15, 8))
    return np.stack((x.ravel(), y.ravel(), np.zeros(x.size)), axis=1)


def test_plane_wave_peaks_near_source_direction() -> None:
    microphones = _small_array()
    grid = make_far_field_grid(24, 32)
    source = direction(18.0, -7.0)
    audio = plane_wave_audio(
        microphones,
        sample_rate_hz=SAMPLE_RATE_HZ,
        num_samples=BLOCK_LENGTH,
        waves=[PlaneWave(1_000.0, 18.0, -7.0)],
    )
    maps = beamform(audio, microphones, grid)
    peak = np.unravel_index(np.argmax(maps[0, 5]), grid.shape[:2])
    assert float(np.dot(grid[peak], source)) > 0.995


def test_zero_input_produces_exact_zero() -> None:
    microphones = _small_array()
    grid = make_far_field_grid(6, 8)
    audio = np.zeros((microphones.shape[0], BLOCK_LENGTH), dtype=np.float32)
    maps = beamform(audio, microphones, grid)
    assert maps.dtype == np.float32
    assert not np.any(maps)


def test_permutation_invariance() -> None:
    """Permuting rows together with their positions must not change the maps."""
    microphones = _small_array()
    grid = make_far_field_grid(6, 8)
    audio = plane_wave_audio(
        microphones,
        sample_rate_hz=SAMPLE_RATE_HZ,
        num_samples=BLOCK_LENGTH,
        waves=[PlaneWave(800.0, -10.0, 4.0)],
        noise_std=0.001,
        seed=3,
    )
    maps = beamform(audio, microphones, grid)
    permutation = np.random.default_rng(0).permutation(microphones.shape[0])
    permuted = beamform(audio[permutation], microphones[permutation], grid)
    np.testing.assert_allclose(maps, permuted, rtol=1e-4, atol=1e-12)


def test_trailing_partial_block_is_dropped() -> None:
    microphones = _small_array()
    grid = make_far_field_grid(6, 8)
    audio = np.random.default_rng(1).normal(
        0, 0.01, size=(microphones.shape[0], 2 * BLOCK_LENGTH + 999)
    )
    maps = beamform(audio, microphones, grid)
    assert maps.shape[0] == 2
