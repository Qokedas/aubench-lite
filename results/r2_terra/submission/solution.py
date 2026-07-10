#!/usr/bin/env python3
"""Offline delay-and-sum beamformer for CAM1K recordings.

The CAM1K sound stream is interleaved signed 32-bit microphone samples.  Its
container lists, for every physical capsule, the stream channel that carries
it; this is used to put the samples back at their measured array positions.
"""
import argparse
import json
import os
import sys
import numpy as np

BANDS = np.asarray([
    (282.,315.,355.), (355.,400.,447.), (447.,500.,562.),
    (562.,630.,708.), (708.,800.,891.), (891.,1000.,1122.),
    (1122.,1250.,1413.), (1413.,1600.,1778.), (1778.,2000.,2239.),
    (2239.,2500.,2818.), (2840.,4000.,5680.)], dtype=np.float64)
BLOCK = 8192
SOUND_SPEED = 343.0


def recording_info(capture):
    with open(os.path.join(capture, "container.json"), encoding="utf8") as f:
        meta = json.load(f)
    sound = meta["metadata"]["files"]["sound"]
    front = sound["frontEnd"]
    positions = np.asarray([[p["X"], p["Y"], p.get("Z", 0.)]
                            for p in front["microphonePositions"]], dtype=np.float32)
    order = np.asarray(sound.get("channelOrdering", np.arange(len(positions))), dtype=np.int64)
    rates = front.get("supportedSampleRates", [46875.0])
    rate = float(rates[0])
    return positions, order, rate


def stream_positions(positions, ordering, nchannels):
    """Return physical position for each interleaved sound-stream channel.

    CAM containers encode channelOrdering as physical-microphone -> stream
    channel.  The usual CAM1K case is a permutation of 0..1023.  The fallback
    also makes reduced captures (where ordering is already stream ordered)
    usable instead of silently choosing arbitrary microphones.
    """
    if (len(ordering) == len(positions) == nchannels and
            np.array_equal(np.sort(ordering), np.arange(nchannels))):
        return positions[np.argsort(ordering)]
    if len(ordering) == nchannels and ordering.size and ordering.min() >= 0 and ordering.max() < len(positions):
        return positions[ordering]
    if len(positions) >= nchannels:
        return positions[:nchannels]
    raise ValueError("sound stream has more channels than the microphone-position metadata")


def camera_rays():
    # Pinhole image coordinates: x increases to camera right and row zero is up.
    az = np.deg2rad(np.linspace(-70.42 / 2., 70.42 / 2., 32))
    el = np.deg2rad(np.linspace(43.3 / 2., -43.3 / 2., 24))
    xx, yy = np.meshgrid(np.tan(az), np.tan(el))
    rays = np.stack((xx, yy, np.ones_like(xx)), axis=-1).reshape(-1, 3)
    rays /= np.linalg.norm(rays, axis=1, keepdims=True)
    return rays.astype(np.float32)


def steering_tables(rays, positions, sample_rate):
    """Five quadrature samples per one-third-octave band, with DAS weights."""
    tables = []
    # Endpoint-inclusive samples approximate an integral over each specified band.
    for low, centre, high in BANDS:
        frequencies = np.array([low, (low + centre) * .5, centre,
                                (centre + high) * .5, high], dtype=np.float64)
        rows = []
        for f in frequencies:
            k = int(np.clip(np.rint(f * BLOCK / sample_rate), 1, BLOCK // 2))
            actual_f = k * sample_rate / BLOCK
            # A plane wave from ray r arrives earlier at a microphone displaced
            # toward r.  This conjugate phase aligns those arrivals.
            phase = -2j * np.pi * actual_f / SOUND_SPEED * (rays @ positions.T)
            rows.append((k, np.exp(phase).astype(np.complex64)))
        tables.append(rows)
    return tables


def make_maps(capture):
    positions, ordering, fs = recording_info(capture)
    sound_path = os.path.join(capture, "sound")
    nbytes = os.path.getsize(sound_path)
    # CAM1K stores one little-endian signed 32-bit word per microphone/sample.
    candidates = [len(ordering), len(positions)]
    nch = next((n for n in candidates if n and nbytes % (4*n) == 0), None)
    if nch is None:
        raise ValueError("sound byte count is not an interleaved int32 microphone stream")
    nsamples = nbytes // (4*nch)
    nframes = nsamples // BLOCK
    positions = stream_positions(positions, ordering, nch)
    rays = camera_rays()
    tables = steering_tables(rays, positions, fs)
    raw = np.memmap(sound_path, dtype="<i4", mode="r", shape=(nsamples, nch))
    result = np.empty((nframes, len(BANDS), 24, 32), dtype=np.float32)

    # Scale before FFT so integer recordings and floating point calculations
    # remain comfortably finite.  It is a fixed ADC conversion, not dB scaling.
    adc_scale = np.float32(1.0 / 2147483648.0)
    denom = np.float32((BLOCK * nch) ** 2)
    for frame in range(nframes):
        block = np.asarray(raw[frame*BLOCK:(frame+1)*BLOCK], dtype=np.float32)
        spectrum = np.fft.rfft(block * adc_scale, axis=0)
        for band, entries in enumerate(tables):
            image = np.zeros(24*32, dtype=np.float64)
            for k, steering in entries:
                beam = steering @ spectrum[k]
                image += (beam.real * beam.real + beam.imag * beam.imag)
            # Mean of independently steered frequency powers is linear power.
            image /= (len(entries) * denom)
            result[frame, band] = image.reshape(24, 32).astype(np.float32)
    # Protect downstream display code against unusual damaged ADC words.
    np.nan_to_num(result, copy=False, nan=0., posinf=np.finfo(np.float32).max, neginf=0.)
    np.maximum(result, 0., out=result)
    return result


def main():
    parser = argparse.ArgumentParser(description="CAM1K acoustic heatmap beamformer")
    parser.add_argument("--capture", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    maps = make_maps(args.capture)
    parent = os.path.dirname(os.path.abspath(args.output))
    if parent:
        os.makedirs(parent, exist_ok=True)
    np.save(args.output, maps)

if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print("solution.py:", exc, file=sys.stderr)
        raise
