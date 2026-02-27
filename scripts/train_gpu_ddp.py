#!/usr/bin/env python3
"""
Distributed multi-GPU training via PyTorch DDP.

Usage (single machine, 4 GPUs):
    torchrun --nproc_per_node=4 scripts/train_gpu_ddp.py --model lstm
    torchrun --nproc_per_node=4 scripts/train_gpu_ddp.py --model conformer --config configs/conformer_gpu.yaml

Fallback (single GPU / CPU):
    python scripts/train_gpu_ddp.py --model lstm
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, TensorDataset, DistributedSampler

sys.path.insert(0, str(Path(__file__).parents[1]))

from src.constants import (
    N_CHANNELS, WINDOW_SAMPLES, N_CLASSES,
    BATCH_SIZE, LEARNING_RATE, MAX_EPOCHS, EARLY_STOP_PATIENCE,
)
from src.data.loader import generate_synthetic_dataset
from src.models.lstm_classifier import LSTMClassifier
from src.models.conformer_classifier import ConformerClassifier


def setup_ddp():
    """Initialize DDP process group if RANK is set, else single-process."""
    rank = int(os.environ.get("RANK", -1))
    if rank < 0:
        return 0, 0, 1, "cuda" if torch.cuda.is_available() else "cpu"
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    dist.init_process_group("nccl")
    device = f"cuda:{local_rank}"
    torch.cuda.set_device(device)
    return rank, local_rank, world_size, device


def cleanup_ddp():
    if dist.is_initialized():
        dist.destroy_process_group()


def train_epoch(model, loader, optimizer, criterion, device, scaler=None):
    model.train()
    total_loss = 0.0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        if scaler:
            with torch.autocast(device_type="cuda"):
                loss = criterion(model(xb), yb)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss = criterion(model(xb), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


def evaluate(model, loader, device):
    model.eval()
    correct = 0
    with torch.no_grad():
        for xb, yb in loader:
            preds = model(xb.to(device)).argmax(1)
            correct += (preds == yb.to(device)).sum().item()
    return correct / len(loader.dataset)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["lstm", "conformer"], default="lstm")
    parser.add_argument("--config", default=None)
    parser.add_argument("--data-dir", default=None, help="Directory with train/val .npz files")
    parser.add_argument("--output-dir", default="models")
    parser.add_argument("--epochs", type=int, default=MAX_EPOCHS)
    args = parser.parse_args()

    rank, local_rank, world_size, device = setup_ddp()
    is_main = rank == 0

    if is_main:
        print(f"=== MAIA DDP Training: {args.model} | Rank {rank}/{world_size} | Device: {device} ===")

    # Load config if provided
    cfg = {}
    if args.config and Path(args.config).exists():
        import yaml
        with open(args.config) as f:
            cfg = yaml.safe_load(f)

    # Data
    if is_main:
        print("Loading data (synthetic)...")
    X, y = generate_synthetic_dataset(n_samples_per_class=300, seed=42)
    split = int(len(y) * 0.85)
    X_tr, y_tr = X[:split], y[:split]
    X_val, y_val = X[split:], y[split:]
    X_tr_t = torch.from_numpy(X_tr)
    y_tr_t = torch.from_numpy(y_tr)

    train_ds = TensorDataset(X_tr_t, y_tr_t)
    sampler = DistributedSampler(train_ds, num_replicas=world_size, rank=rank) if world_size > 1 else None
    batch_size = cfg.get("training", {}).get("batch_size", BATCH_SIZE)
    train_dl = DataLoader(train_ds, batch_size=batch_size, sampler=sampler, shuffle=(sampler is None))
    val_dl = DataLoader(TensorDataset(torch.from_numpy(X_val), torch.from_numpy(y_val)), batch_size=512)

    # Model
    if args.model == "lstm":
        model = LSTMClassifier()
    else:
        d_model = cfg.get("d_model", 128)
        n_layers = cfg.get("n_layers", 4)
        model = ConformerClassifier(d_model=d_model, n_layers=n_layers)

    model = model.to(device)
    if world_size > 1:
        model = DDP(model, device_ids=[local_rank])

    if is_main:
        base = model.module if hasattr(model, "module") else model
        print(f"Params: {sum(p.numel() for p in base.parameters()):,}")

    lr = cfg.get("training", {}).get("learning_rate", LEARNING_RATE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = nn.CrossEntropyLoss()
    scaler = torch.cuda.amp.GradScaler() if device.startswith("cuda") else None

    best_acc = 0.0
    best_state = None
    patience_ctr = 0

    for epoch in range(1, args.epochs + 1):
        if sampler:
            sampler.set_epoch(epoch)
        loss = train_epoch(model, train_dl, optimizer, criterion, device, scaler)
        scheduler.step()

        if epoch % 20 == 0 or epoch == args.epochs:
            acc = evaluate(model, val_dl, device)
            if is_main:
                print(f"Epoch {epoch:4d} | Loss {loss:.4f} | Val Acc {acc:.4f}")
            if acc > best_acc:
                best_acc = acc
                base = model.module if hasattr(model, "module") else model
                best_state = {k: v.clone() for k, v in base.state_dict().items()}
                patience_ctr = 0
            else:
                patience_ctr += 1
                if patience_ctr >= EARLY_STOP_PATIENCE:
                    if is_main:
                        print(f"Early stop at epoch {epoch}")
                    break

    if is_main and best_state:
        out_dir = Path(args.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        model_name = "asl_emg_classifier" if args.model == "lstm" else "conformer_classifier"
        pt_path = out_dir / f"{model_name}.pt"
        torch.save(best_state, pt_path)
        print(f"Saved: {pt_path} | Best val acc: {best_acc:.4f}")

        # Export ONNX
        if args.model == "lstm":
            m = LSTMClassifier()
        else:
            m = ConformerClassifier(
                d_model=cfg.get("d_model", 128),
                n_layers=cfg.get("n_layers", 4),
            )
        m.load_state_dict(best_state)
        m.eval()
        dummy = torch.zeros(1, WINDOW_SAMPLES, N_CHANNELS)
        onnx_path = out_dir / f"{model_name}.onnx"
        torch.onnx.export(
            m, dummy, str(onnx_path),
            input_names=["input"], output_names=["output"],
            dynamic_axes={"input": {0: "batch"}},
            opset_version=17,
        )
        print(f"ONNX: {onnx_path}")

    cleanup_ddp()


if __name__ == "__main__":
    main()
