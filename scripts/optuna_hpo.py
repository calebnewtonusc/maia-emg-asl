#!/usr/bin/env python3
"""
Optuna hyperparameter optimization for LSTM and CNN-LSTM models.

Usage:
    python scripts/optuna_hpo.py --model lstm --n-trials 50
    python scripts/optuna_hpo.py --model cnn_lstm --n-trials 30 --train-best
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).parents[1]))

from src.constants import N_CHANNELS, WINDOW_SAMPLES, N_CLASSES
from src.data.loader import generate_synthetic_dataset
from src.models.lstm_classifier import LSTMClassifier
from src.models.cnn_lstm_classifier import CNNLSTMClassifier


def _make_loaders(seed: int = 42):
    X, y = generate_synthetic_dataset(n_samples_per_class=150, seed=seed)
    split = int(len(y) * 0.8)
    X_tr, y_tr = torch.from_numpy(X[:split]), torch.from_numpy(y[:split])
    X_val, y_val = torch.from_numpy(X[split:]), torch.from_numpy(y[split:])
    train_dl = DataLoader(TensorDataset(X_tr, y_tr), batch_size=128, shuffle=True)
    val_dl = DataLoader(TensorDataset(X_val, y_val), batch_size=256)
    return train_dl, val_dl, X_val, y_val


def _train_eval(model, train_dl, val_dl, lr: float, epochs: int = 30, device: str = "cpu") -> float:
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    for _ in range(epochs):
        model.train()
        for xb, yb in train_dl:
            optimizer.zero_grad()
            criterion(model(xb.to(device)), yb.to(device)).backward()
            optimizer.step()
    model.eval()
    correct = 0
    with torch.no_grad():
        for xb, yb in val_dl:
            correct += (model(xb.to(device)).argmax(1) == yb.to(device)).sum().item()
    return correct / (len(val_dl.dataset))


def objective_lstm(trial, train_dl, val_dl, device):
    hidden = trial.suggest_categorical("hidden_size", [64, 128, 256])
    layers = trial.suggest_int("num_layers", 1, 3)
    dropout = trial.suggest_float("dropout", 0.1, 0.5)
    lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
    model = LSTMClassifier(hidden_size=hidden, num_layers=layers, dropout=dropout)
    return _train_eval(model, train_dl, val_dl, lr, device=device)


def objective_cnn_lstm(trial, train_dl, val_dl, device):
    f1 = trial.suggest_categorical("f1", [16, 32, 64])
    f2 = trial.suggest_categorical("f2", [32, 64, 128])
    hidden = trial.suggest_categorical("hidden", [64, 128])
    dropout = trial.suggest_float("dropout", 0.1, 0.4)
    lr = trial.suggest_float("lr", 1e-4, 5e-3, log=True)
    model = CNNLSTMClassifier(cnn_filters=[f1, f2], lstm_hidden=hidden, dropout=dropout)
    return _train_eval(model, train_dl, val_dl, lr, device=device)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["lstm", "cnn_lstm"], default="lstm")
    parser.add_argument("--n-trials", type=int, default=30)
    parser.add_argument("--train-best", action="store_true")
    parser.add_argument("--output-dir", default="models")
    args = parser.parse_args()

    try:
        import optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)
    except ImportError:
        print("ERROR: optuna not installed. Run: pip install optuna")
        sys.exit(1)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    train_dl, val_dl, X_val, y_val = _make_loaders()

    study = optuna.create_study(direction="maximize")
    if args.model == "lstm":
        study.optimize(lambda t: objective_lstm(t, train_dl, val_dl, device), n_trials=args.n_trials)
    else:
        study.optimize(lambda t: objective_cnn_lstm(t, train_dl, val_dl, device), n_trials=args.n_trials)

    best = study.best_trial
    print(f"\nBest val accuracy: {best.value:.4f}")
    print(f"Best params: {best.params}")

    # Save best params
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    hp_path = out_dir / f"best_hparams_{args.model}.json"
    with open(hp_path, "w") as f:
        json.dump({"model": args.model, "val_acc": best.value, "params": best.params}, f, indent=2)
    print(f"Saved: {hp_path}")


if __name__ == "__main__":
    main()
