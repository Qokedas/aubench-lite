"""Trusted conventional far-field delay-and-sum reference beamformer.

This module is never mounted into agent or candidate containers. Its output defines
ground truth; the candidate-facing task pins every physically meaningful choice
(bands, block length, view geometry) while the scoring canonicalization makes the
remaining internal choices (window, DC handling, edge convention) immaterial —
measured spread across faithful variants is under 0.1% of score.
"""

from __future__ import annotations

import numpy as np

from .constants import (
    BLOCK_LENGTH,
    FREQUENCY_BANDS,
    SAMPLE_RATE_HZ,
    SPEED_OF_SOUND_M_S,
)
from .formats import validate_audio, validate_grid, validate_microphones

_BIN_CHUNK = 16


def band_bins(frame_length: int = BLOCK_LENGTH) -> list[np.ndarray]:
    frequencies = np.fft.rfftfreq(frame_length, d=1.0 / SAMPLE_RATE_HZ)
    selected = []
    for lower, _, upper in FREQUENCY_BANDS:
        bins = np.flatnonzero((frequencies >= lower) & (frequencies <= upper))
        if not bins.size:
            raise ValueError(f"frequency band [{lower}, {upper}] contains no FFT bins")
        selected.append(bins)
    return selected


def beamform(
    audio: np.ndarray,
    microphones: np.ndarray,
    grid: np.ndarray,
) -> np.ndarray:
    """Compute linear relative acoustic power maps [frame, band, height, width].

    ``audio`` is float relative pressure with row i at position ``microphones[i]``.
    Frames are non-overlapping BLOCK_LENGTH windows; the trailing partial is dropped.
    """
    audio = np.asarray(audio)
    microphones = np.asarray(microphones)
    grid = np.asarray(grid)
    validate_audio(audio)
    validate_microphones(microphones, audio.shape[0])
    validate_grid(grid)

    frame_count = audio.shape[1] // BLOCK_LENGTH
    if frame_count < 1:
        raise ValueError(f"audio has {audio.shape[1]} samples; needs >= {BLOCK_LENGTH}")
    frequencies = np.fft.rfftfreq(BLOCK_LENGTH, d=1.0 / SAMPLE_RATE_HZ)
    selected = band_bins()
    window = np.hanning(BLOCK_LENGTH).astype(np.float32)

    directions = grid.reshape(-1, 3).astype(np.float32, copy=False)
    microphone_f32 = microphones.astype(np.float32, copy=False)
    # Delay of each microphone relative to the array origin for a wave arriving from a
    # scan direction. Shape is [grid point, microphone].
    delays = (directions @ microphone_f32.T) / np.float32(SPEED_OF_SOUND_M_S)
    maps = np.empty(
        (frame_count, len(selected), grid.shape[0], grid.shape[1]), dtype=np.float32
    )
    microphone_scale = np.float32(audio.shape[0] ** 2)

    for frame_index in range(frame_count):
        start = frame_index * BLOCK_LENGTH
        frame = audio[:, start : start + BLOCK_LENGTH].astype(np.float32, copy=True)
        frame -= frame.mean(axis=1, keepdims=True)
        spectrum = np.fft.rfft(frame * window[None, :], axis=1).astype(np.complex64)
        for band_index, bins in enumerate(selected):
            power = np.zeros(directions.shape[0], dtype=np.float32)
            for chunk_start in range(0, bins.size, _BIN_CHUNK):
                chunk = bins[chunk_start : chunk_start + _BIN_CHUNK]
                # [chunk, grid, microphone] steering phases, evaluated in one shot.
                phase = (
                    np.float32(-2.0 * np.pi)
                    * frequencies[chunk].astype(np.float32)[:, None, None]
                    * delays[None, :, :]
                )
                steering = np.exp(1j * phase).astype(np.complex64)
                focused = np.einsum(
                    "bgm,mb->gb", steering, spectrum[:, chunk], optimize=True
                )
                power += (
                    (focused.real * focused.real + focused.imag * focused.imag)
                    .sum(axis=1)
                    .astype(np.float32)
                )
            power /= np.float32(len(bins)) * microphone_scale
            maps[frame_index, band_index] = power.reshape(grid.shape[:2])
    np.maximum(maps, 0, out=maps)
    return maps
