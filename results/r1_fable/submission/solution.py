#!/usr/bin/env python3
"""Acoustic heatmap generator for Sorama CAM1K captures.

Reads a raw `sound` capture (int32, blocks of 1024 samples x 1024 channels,
channel-major within block) plus `container.json` metadata, and produces
per-frame, per-band delay-and-sum beamforming maps aligned with the camera
view (70.42 x 43.3 deg pinhole, 32 x 24 pixels).
"""
import argparse, json, os
import numpy as np

FRAME = 8192
BANDS = [(282,315,355),(355,400,447),(447,500,562),(562,630,708),(708,800,891),
         (891,1000,1122),(1122,1250,1413),(1413,1600,1778),(1778,2000,2239),
         (2239,2500,2818),(2840,4000,5680)]
SPEED = 343.0
NROW, NCOL = 24, 32
FOVX, FOVY = 70.42, 43.3
BLOCK = 1024

def load_capture(cap):
    meta = json.load(open(os.path.join(cap, 'container.json')))
    snd = meta['metadata']['files']['sound']
    pos = np.array([[p['X'], p['Y'], p['Z']] for p in snd['frontEnd']['microphonePositions']])
    order = np.array(snd['channelOrdering'])
    sel = np.array(snd.get('channelSelection', [True]*len(order)), bool)
    raw = np.fromfile(os.path.join(cap, snd.get('file', meta['files']['sound'])
                                   if isinstance(meta['files']['sound'], str) else 'sound'),
                      dtype='<i4')
    C = len(order)
    nfull = raw.size // (C * BLOCK)
    rem = (raw.size - nfull * C * BLOCK) // C
    N = nfull * BLOCK + rem
    sig = np.empty((N, C), np.float32)
    if nfull:
        sig[:nfull*BLOCK] = raw[:nfull*C*BLOCK].reshape(nfull, C, BLOCK).transpose(0, 2, 1).reshape(-1, C)
    if rem:
        sig[nfull*BLOCK:] = raw[nfull*C*BLOCK:nfull*C*BLOCK+C*rem].reshape(C, rem).T
    sig /= 2**31
    mic = pos[order]           # position of each file channel
    dur = snd.get('duration')  # ms
    fs = float(N / (dur/1000.0)) if dur else 46875.0
    fs = round(fs / 5) * 5.0   # snap (46874.8 -> 46875)
    cam = meta['metadata']['files'].get('video', {}).get('captureDevice', {}).get('position', None)
    campos = np.array([cam['X'], cam['Y'], cam['Z']]) if cam else np.zeros(3)
    dist = float(meta['metadata'].get('distance', 10.0) or 10.0)
    return sig, mic, sel, fs, campos, dist

def pixel_dirs():
    w = np.tan(np.deg2rad(FOVX/2)); h = np.tan(np.deg2rad(FOVY/2))
    xs = ((np.arange(NCOL)+0.5)/NCOL*2 - 1) * w
    ys = (1 - (np.arange(NROW)+0.5)/NROW*2) * h
    X, Y = np.meshgrid(xs, ys)
    D = np.stack([X, Y, np.ones_like(X)], -1)
    D /= np.linalg.norm(D, axis=-1, keepdims=True)
    return D.reshape(-1, 3)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--capture', required=True)
    ap.add_argument('--output', required=True)
    args = ap.parse_args()
    sig, mic, sel, fs, campos, dist = load_capture(args.capture)
    N, C = sig.shape
    nfr = N // FRAME
    dirs = pixel_dirs()
    F = campos[None, :] + dist * dirs
    delays = np.linalg.norm(F[:, None, :] - mic[None, :, :], axis=-1) / SPEED
    w = sel.astype(np.float32)
    nsel = w.sum()
    freqs = np.fft.rfftfreq(FRAME, 1.0/fs)
    out = np.zeros((nfr, len(BANDS), dirs.shape[0]), np.float64)
    # process frames in chunks to bound memory
    Xf = np.empty((nfr, len(freqs), C), np.complex64)
    for f in range(nfr):
        Xf[f] = np.fft.rfft(sig[f*FRAME:(f+1)*FRAME], axis=0)
    Xf *= w[None, None, :]
    for bi, (lo, cf, hi) in enumerate(BANDS):
        for k in np.where((freqs >= lo) & (freqs < hi))[0]:
            S = np.exp(2j*np.pi*freqs[k]*delays).astype(np.complex64)
            Y = S @ Xf[:, k, :].T
            out[:, bi, :] += np.abs(Y).T.astype(np.float64)**2
    out /= nsel**2
    out /= FRAME  # spectral normalization
    res = out.reshape(nfr, len(BANDS), NROW, NCOL).astype(np.float32)
    np.save(args.output, res)

if __name__ == '__main__':
    main()
