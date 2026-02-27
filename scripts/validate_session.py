#!/usr/bin/env python3
"""
Validate a recorded EMG session for quality.

Checks: clipping, RMS level, flat channels, 60Hz noise, missing data,
class balance, SNR.

Usage:
    python scripts/validate_session.py --file data/train/session_001.npz
    python scripts/validate_session.py --file data/train/session_001.npz --fix
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parents[1]))

from src.constants import N_CHANNELS, SAMPLE_RATE, N_CLASSES, ASL_CLASSES


CHECKS = [
    "clipping", "rms_level", "flat_channels",
    "60hz_noise", "missing_data", "class_balance", "snr",
]


def check_clipping(emg: np.ndarray, threshold: float = 0.95) -> dict:
    max_vals = np.abs(emg).max(axis=0)
    clipped = max_vals > threshold
    return {
        "pass": not clipped.any(),
        "detail": f"Clipped channels: {np.where(clipped)[0].tolist()}",
    }


def check_rms(emg: np.ndarray, min_rms: float = 0.001, max_rms: float = 0.5) -> dict:
    rms = np.sqrt(np.mean(emg ** 2, axis=0))
    bad = (rms < min_rms) | (rms > max_rms)
    return {
        "pass": not bad.any(),
        "detail": f"RMS range: [{rms.min():.4f}, {rms.max():.4f}]  Bad: {np.where(bad)[0].tolist()}",
    }


def check_flat(emg: np.ndarray, var_threshold: float = 1e-8) -> dict:
    var = np.var(emg, axis=0)
    flat = var < var_threshold
    return {
        "pass": not flat.any(),
        "detail": f"Flat channels: {np.where(flat)[0].tolist()}",
    }


def check_60hz(emg: np.ndarray, fs: float = SAMPLE_RATE, threshold_db: float = -10.0) -> dict:
    freqs = np.fft.rfftfreq(emg.shape[0], d=1.0 / fs)
    psd = np.abs(np.fft.rfft(emg, axis=0)) ** 2
    idx_60 = np.argmin(np.abs(freqs - 60))
    idx_total = slice(1, len(freqs))
    ratio_db = 10 * np.log10(psd[idx_60] / (psd[idx_total].mean(axis=0) + 1e-12) + 1e-12)
    noisy = ratio_db > -threshold_db
    return {
        "pass": not noisy.any(),
        "detail": f"60Hz ratio (dB) max: {ratio_db.max():.1f}  Noisy channels: {np.where(noisy)[0].tolist()}",
    }


def check_missing(emg: np.ndarray) -> dict:
    n_nan = np.isnan(emg).sum()
    n_inf = np.isinf(emg).sum()
    return {
        "pass": n_nan == 0 and n_inf == 0,
        "detail": f"NaN: {n_nan}  Inf: {n_inf}",
    }


def check_class_balance(labels: np.ndarray) -> dict:
    counts = np.bincount(labels, minlength=N_CLASSES)
    nonzero = counts[counts > 0]
    imbalance = nonzero.max() / (nonzero.min() + 1e-9)
    return {
        "pass": imbalance < 10.0,
        "detail": f"Classes present: {(counts > 0).sum()}  Max/min ratio: {imbalance:.1f}",
    }


def check_snr(emg: np.ndarray, noise_floor: float = 0.01) -> dict:
    signal_power = np.mean(emg ** 2, axis=0)
    snr_db = 10 * np.log10(signal_power / (noise_floor ** 2) + 1e-12)
    low = snr_db < 10.0
    return {
        "pass": not low.any(),
        "detail": f"SNR (dB) min: {snr_db.min():.1f}  Low channels: {np.where(low)[0].tolist()}",
    }


def fix_session(emg: np.ndarray, labels: np.ndarray) -> tuple:
    """Remove NaN/inf samples, clip to +-3sigma."""
    # Remove bad rows
    bad_rows = np.any(np.isnan(emg) | np.isinf(emg), axis=1)
    emg = emg[~bad_rows]
    labels = labels[~bad_rows]
    # Clip to +-3sigma
    sigma = emg.std(axis=0, keepdims=True)
    emg = np.clip(emg, -3 * sigma, 3 * sigma)
    return emg, labels


def main():
    parser = argparse.ArgumentParser(description="Validate EMG session quality")
    parser.add_argument("--file", required=True, help="Path to .npz session file")
    parser.add_argument("--fix", action="store_true", help="Fix and overwrite the file")
    args = parser.parse_args()

    data = np.load(args.file)
    emg = data["emg"]
    labels = data["labels"]
    print(f"Session: {args.file}")
    print(f"Shape: {emg.shape}  Labels: {labels.shape}  Classes: {len(np.unique(labels))}")
    print()

    check_fns = {
        "clipping":      lambda: check_clipping(emg),
        "rms_level":     lambda: check_rms(emg),
        "flat_channels": lambda: check_flat(emg),
        "60hz_noise":    lambda: check_60hz(emg),
        "missing_data":  lambda: check_missing(emg),
        "class_balance": lambda: check_class_balance(labels),
        "snr":           lambda: check_snr(emg),
    }

    n_pass = 0
    for name, fn in check_fns.items():
        result = fn()
        status = "PASS" if result["pass"] else "FAIL"
        icon = "+" if result["pass"] else "x"
        print(f"  {icon} [{status}] {name:<20} {result['detail']}")
        if result["pass"]:
            n_pass += 1

    print(f"\n{n_pass}/{len(check_fns)} checks passed")

    if args.fix:
        emg_fixed, labels_fixed = fix_session(emg, labels)
        np.savez(args.file, emg=emg_fixed, labels=labels_fixed)
        print(f"Fixed and saved: {args.file}  (removed {len(labels) - len(labels_fixed)} bad samples)")


if __name__ == "__main__":
    main()
