#!/usr/bin/env python3
"""Train Conformer baseline on synthetic data."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).parents[1]))

from src.constants import N_CHANNELS, WINDOW_SAMPLES, N_CLASSES, BATCH_SIZE, LEARNING_RATE
from src.data.loader import generate_synthetic_dataset
from src.models.conformer_classifier import ConformerClassifier


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--n-layers", type=int, default=4)
    parser.add_argument("--output-dir", default="models")
    args = parser.parse_args()

    print("Generating synthetic dataset...")
    X, y = generate_synthetic_dataset(n_samples_per_class=200, seed=42)
    split = int(len(y) * 0.8)
    X_tr, y_tr = X[:split], y[:split]
    X_val, y_val = X[split:], y[split:]

    X_tr_t = torch.from_numpy(X_tr)
    y_tr_t = torch.from_numpy(y_tr)

    train_loader = DataLoader(TensorDataset(X_tr_t, y_tr_t), batch_size=BATCH_SIZE, shuffle=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = ConformerClassifier(
        n_channels=N_CHANNELS, n_classes=N_CLASSES,
        d_model=args.d_model, n_layers=args.n_layers,
    ).to(device)
    print(f"Conformer params: {sum(p.numel() for p in model.parameters()):,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5)
    criterion = nn.CrossEntropyLoss()

    best_loss = float("inf")
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(train_loader)
        scheduler.step(avg_loss)
        if epoch % 20 == 0:
            print(f"Epoch {epoch:4d} | Loss: {avg_loss:.4f}")
        if avg_loss < best_loss:
            best_loss = avg_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(best_state, out_dir / "conformer_classifier.pt")

    # Export ONNX
    model.load_state_dict(best_state)
    model.eval()
    dummy = torch.zeros(1, WINDOW_SAMPLES, N_CHANNELS)
    onnx_path = out_dir / "conformer_classifier.onnx"
    torch.onnx.export(
        model, dummy, str(onnx_path),
        input_names=["input"], output_names=["output"],
        dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
        opset_version=17,
    )
    print(f"Saved: {onnx_path}")


if __name__ == "__main__":
    main()
