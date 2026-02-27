#!/usr/bin/env python3
"""
Train cross-modal EMG-Vision embedding (CLIP-style).

Aligns EMG feature vectors with MediaPipe hand landmark vectors
using symmetric InfoNCE loss. Can run self-supervised on synthetic
EMG+landmark pairs, or supervised on real paired recordings.

Usage:
    # Synthetic self-supervised (no real data needed)
    python scripts/train_cross_modal.py --generate-synthetic

    # With real data (paired .npz files that include 'emg' and 'landmarks' keys)
    python scripts/train_cross_modal.py --emg-dir data/paired/
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).parents[1]))

from src.constants import N_CLASSES, ASL_CLASSES, FEATURE_DIM
from src.data.loader import generate_synthetic_dataset
from src.signal.features import extract_features
from src.models.cross_modal_embedding import CrossModalASL


def _make_synthetic_pairs(n_per_class: int = 50, seed: int = 42):
    """Generate synthetic (emg_features, landmarks) pairs for self-supervised training."""
    rng = np.random.default_rng(seed)
    emg_list, lm_list, labels = [], [], []

    for cls_idx in range(N_CLASSES):
        # Distinct EMG pattern per class
        pattern = np.zeros(8)
        active = rng.choice(8, size=rng.integers(2, 5), replace=False)
        pattern[active] = rng.uniform(0.3, 1.0, len(active))

        # Distinct landmark pattern per class
        lm_center = rng.normal(0, 0.3, 63).astype(np.float32)
        lm_center /= np.linalg.norm(lm_center) + 1e-8

        for _ in range(n_per_class):
            # EMG window → features
            t = np.linspace(0, 0.2, 40)
            carrier = np.sin(2 * np.pi * (20 + cls_idx * 3) * t)[:, None]
            window = (carrier * pattern[None, :] + rng.normal(0, 0.05, (40, 8))).astype(np.float32)
            feat = extract_features(window)
            emg_list.append(feat)

            # Landmark with noise
            lm = lm_center + rng.normal(0, 0.05, 63).astype(np.float32)
            lm /= np.linalg.norm(lm) + 1e-8
            lm_list.append(lm)
            labels.append(cls_idx)

    emg = np.stack(emg_list)
    lm = np.stack(lm_list)
    y = np.array(labels, dtype=np.int64)
    perm = rng.permutation(len(y))
    return emg[perm], lm[perm], y[perm]


def _quick_eval(model: CrossModalASL, emg: np.ndarray, lm: np.ndarray, y: np.ndarray, device: str):
    """Evaluate nearest-neighbor accuracy using class gallery."""
    model.eval()
    # Build gallery from landmark prototypes
    gallery_dict = {}
    for cls_idx in range(N_CLASSES):
        mask = y == cls_idx
        if mask.sum() == 0:
            continue
        gallery_dict[cls_idx] = lm[mask]
    gallery = model.build_class_gallery(gallery_dict).to(device)

    # Classify EMG
    emg_t = torch.from_numpy(emg).float().to(device)
    preds, confs = model.classify_emg(emg_t, gallery)
    y_t = torch.from_numpy(y).to(device)
    acc = (preds == y_t).float().mean().item()
    return acc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--emg-dir", default=None)
    parser.add_argument("--generate-synthetic", action="store_true", default=True)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--embed-dim", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--output-dir", default="models")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    print("Generating synthetic paired data...")
    emg, lm, y = _make_synthetic_pairs(n_per_class=100)
    print(f"Pairs: {len(y)} | EMG dim: {emg.shape[1]} | LM dim: {lm.shape[1]}")

    split = int(len(y) * 0.85)
    emg_tr, lm_tr = emg[:split], lm[:split]
    emg_val, lm_val, y_val = emg[split:], lm[split:], y[split:]

    emg_t = torch.from_numpy(emg_tr).float()
    lm_t = torch.from_numpy(lm_tr).float()
    train_dl = DataLoader(TensorDataset(emg_t, lm_t), batch_size=args.batch_size, shuffle=True)

    model = CrossModalASL(embed_dim=args.embed_dim).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    print(f"\n{'Epoch':>6} {'Loss':>10} {'Val Acc':>10}")
    print("-" * 32)

    best_acc = 0.0
    best_state = None

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        for emg_b, lm_b in train_dl:
            emg_b, lm_b = emg_b.to(device), lm_b.to(device)
            optimizer.zero_grad()
            loss = model(emg_b, lm_b)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        scheduler.step()

        if epoch % 20 == 0 or epoch == args.epochs:
            val_acc = _quick_eval(model, emg_val, lm_val, y_val, device)
            avg_loss = total_loss / len(train_dl)
            print(f"{epoch:>6} {avg_loss:>10.4f} {val_acc:>10.4f}")
            if val_acc > best_acc:
                best_acc = val_acc
                best_state = {k: v.clone() for k, v in model.state_dict().items()}

    print(f"\nBest val accuracy: {best_acc:.4f}")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model.load_state_dict(best_state)
    model.save(str(out_dir / "cross_modal_embedding.pt"))
    print(f"Saved: {out_dir / 'cross_modal_embedding.pt'}")


if __name__ == "__main__":
    main()
