"""Deterministic private acoustic pressure-field generators."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class PlaneWave:
    frequency_hz: float
    azimuth_deg: float
    elevation_deg: float
    amplitude: float = 1.0
    phase_rad: float = 0.0


def direction(azimuth_deg: float, elevation_deg: float) -> np.ndarray:
    az = np.deg2rad(azimuth_deg)
    el = np.deg2rad(elevation_deg)
    value = np.array(
        [np.sin(az) * np.cos(el), np.sin(el), np.cos(az) * np.cos(el)],
        dtype=np.float64,
    )
    return value / np.linalg.norm(value)


def plane_wave_audio(
    microphones: np.ndarray,
    *,
    sample_rate_hz: float,
    num_samples: int,
    waves: list[PlaneWave],
    speed_of_sound_m_s: float = 343.0,
    noise_std: float = 0.0,
    seed: int = 0,
) -> np.ndarray:
    """Generate synchronized pressure for far-field sources with fractional delays."""
    t = np.arange(num_samples, dtype=np.float64) / float(sample_rate_hz)
    audio = np.zeros((microphones.shape[0], num_samples), dtype=np.float64)
    for wave in waves:
        ray = direction(wave.azimuth_deg, wave.elevation_deg)
        delays = microphones.astype(np.float64) @ ray / float(speed_of_sound_m_s)
        phase = 2.0 * np.pi * float(wave.frequency_hz) * (t[None, :] + delays[:, None]) + float(
            wave.phase_rad
        )
        audio += float(wave.amplitude) * np.sin(phase)
    if noise_std:
        rng = np.random.default_rng(seed)
        audio += rng.normal(0.0, noise_std, size=audio.shape)
    peak = float(np.max(np.abs(audio)))
    if peak > 1.0:
        audio /= peak
    return audio.astype(np.float32)


def incoherent_noise(
    microphone_count: int, num_samples: int, *, seed: int, scale: float = 0.2
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.normal(0.0, scale, size=(microphone_count, num_samples)).astype(np.float32)
