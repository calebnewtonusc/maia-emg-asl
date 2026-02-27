#!/usr/bin/env python3
"""
Per-user calibration: fine-tune a small AdaptorHead on top of the frozen base model.

The user performs 10 signs (A-J), ~5s each, while the script records from webcam.
Then a small Linear adapter is trained on the collected data.

Usage:
    python scripts/calibrate_user.py --user-id caleb --model models/asl_emg_classifier.onnx
    python scripts/calibrate_user.py --user-id caleb --list-users
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).parents[1]))

from src.constants import N_CLASSES, ASL_CLASSES, N_CHANNELS, WINDOW_SAMPLES
from src.data.loader import generate_synthetic_dataset
from src.models.lstm_classifier import LSTMClassifier


CALIBRATION_SIGNS = list("ABCDEFGHIJ")  # 10 signs for calibration session
SECONDS_PER_SIGN = 5
SAMPLE_RATE = 200


class AdaptorHead(nn.Module):
    """Small linear adapter that fine-tunes on top of frozen LSTM."""
    def __init__(self, hidden_dim: int, n_classes: int = N_CLASSES):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Linear(32, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def collect_calibration_data_synthetic(n_samples: int = 50, seed: int = 0):
    """Generate synthetic calibration data (used when no hardware available)."""
    print("No hardware connected -- using synthetic calibration data")
    X, y = generate_synthetic_dataset(n_samples_per_class=n_samples, seed=seed)
    # Only keep calibration signs (first 10 classes)
    mask = y < len(CALIBRATION_SIGNS)
    return X[mask], y[mask]


def train_adaptor(
    base_model: LSTMClassifier,
    X: np.ndarray,
    y: np.ndarray,
    epochs: int = 80,
    lr: float = 1e-3,
) -> AdaptorHead:
    """Train AdaptorHead on calibration data with frozen base model."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    base_model = base_model.to(device).eval()

    # Freeze base model
    for p in base_model.parameters():
        p.requires_grad = False

    # Extract features from frozen LSTM (up to fc layer)
    X_t = torch.from_numpy(X).float().to(device)
    with torch.no_grad():
        # Get LSTM output before fc
        out, _ = base_model.lstm(X_t)
        features = out[:, -1, :]  # (N, hidden_size)

    adaptor = AdaptorHead(base_model.hidden_size, n_classes=len(CALIBRATION_SIGNS)).to(device)
    optimizer = torch.optim.Adam(adaptor.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    y_t = torch.from_numpy(y).long().to(device)

    print(f"\nTraining AdaptorHead on {len(y)} calibration samples...")
    print(f"{'Epoch':>6} {'Loss':>10} {'Acc':>8}")

    best_acc = 0.0
    best_state = None
    for epoch in range(1, epochs + 1):
        adaptor.train()
        optimizer.zero_grad()
        logits = adaptor(features)
        loss = criterion(logits, y_t)
        loss.backward()
        optimizer.step()

        if epoch % 20 == 0 or epoch == epochs:
            preds = logits.argmax(1)
            acc = (preds == y_t).float().mean().item()
            print(f"{epoch:>6} {loss.item():>10.4f} {acc:>8.4f}")
            if acc > best_acc:
                best_acc = acc
                best_state = {k: v.clone() for k, v in adaptor.state_dict().items()}

    adaptor.load_state_dict(best_state)
    print(f"\nBest accuracy: {best_acc:.4f}")
    return adaptor


def main():
    parser = argparse.ArgumentParser(description="Per-user EMG calibration")
    parser.add_argument("--user-id", required=True, help="User identifier (e.g. 'caleb')")
    parser.add_argument("--base-model", default="models/asl_emg_classifier.pt")
    parser.add_argument("--output-dir", default="models/calibrated")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--list-users", action="store_true")
    args = parser.parse_args()

    if args.list_users:
        cal_dir = Path(args.output_dir)
        if cal_dir.exists():
            users = [d.name for d in cal_dir.iterdir() if d.is_dir()]
            print(f"Calibrated users: {users}")
        else:
            print("No calibrated users yet.")
        return

    # Load base model
    if not Path(args.base_model).exists():
        print(f"Base model not found: {args.base_model}")
        print("Run: python scripts/train_lstm_baseline.py first")
        sys.exit(1)

    print(f"Loading base model: {args.base_model}")
    base_model = LSTMClassifier()
    base_model.load_state_dict(torch.load(args.base_model, map_location="cpu"))
    base_model.eval()

    # Collect calibration data (synthetic for now -- replace with real BLE when hardware arrives)
    X, y = collect_calibration_data_synthetic()

    # Before calibration accuracy
    with torch.no_grad():
        logits_pre = base_model(torch.from_numpy(X).float())
        acc_pre = (logits_pre.argmax(1).numpy() == y).mean()
    print(f"Base model accuracy (calibration signs): {acc_pre:.3f}")

    # Train adaptor
    adaptor = train_adaptor(base_model, X, y, epochs=args.epochs)

    # Save
    out_dir = Path(args.output_dir) / args.user_id
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(adaptor.state_dict(), out_dir / "adaptor.pt")
    print(f"\nSaved adaptor: {out_dir / 'adaptor.pt'}")
    print("Calibration complete!")


if __name__ == "__main__":
    main()
