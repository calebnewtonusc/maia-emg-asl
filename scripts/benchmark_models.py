#!/usr/bin/env python3
"""Benchmark LSTM, CNN-LSTM, and Conformer inference latency."""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parents[1]))

from src.constants import N_CHANNELS, WINDOW_SAMPLES, N_CLASSES
from src.models.lstm_classifier import LSTMClassifier
from src.models.cnn_lstm_classifier import CNNLSTMClassifier
from src.models.conformer_classifier import ConformerClassifier


def bench(name: str, model: torch.nn.Module, n_runs: int = 200):
    model.eval()
    dummy = torch.zeros(1, WINDOW_SAMPLES, N_CHANNELS)
    # Warmup
    for _ in range(10):
        with torch.no_grad():
            model(dummy)
    times = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        with torch.no_grad():
            model(dummy)
        times.append((time.perf_counter() - t0) * 1000)
    arr = np.array(times)
    print(f"  {name:<24} mean={arr.mean():6.2f}ms  p50={np.median(arr):6.2f}ms  p95={np.percentile(arr,95):6.2f}ms")


def bench_onnx(name: str, onnx_path: str, n_runs: int = 200):
    try:
        import onnxruntime as ort
    except ImportError:
        print(f"  {name:<24} [onnxruntime not installed]")
        return
    if not Path(onnx_path).exists():
        print(f"  {name:<24} [file not found: {onnx_path}]")
        return
    sess = ort.InferenceSession(onnx_path)
    dummy = np.zeros((1, WINDOW_SAMPLES, N_CHANNELS), dtype=np.float32)
    for _ in range(10):
        sess.run(None, {"input": dummy})
    times = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        sess.run(None, {"input": dummy})
        times.append((time.perf_counter() - t0) * 1000)
    arr = np.array(times)
    print(f"  {name:<24} mean={arr.mean():6.2f}ms  p50={np.median(arr):6.2f}ms  p95={np.percentile(arr,95):6.2f}ms")


def main():
    print("=== MAIA Model Latency Benchmark ===\n")
    print("PyTorch (CPU):")
    bench("LSTM", LSTMClassifier())
    bench("CNN-LSTM", CNNLSTMClassifier())
    bench("Conformer", ConformerClassifier())
    print("\nONNX Runtime (CPU):")
    bench_onnx("LSTM (ONNX)", "models/asl_emg_classifier.onnx")
    bench_onnx("Conformer (ONNX)", "models/conformer_classifier.onnx")


if __name__ == "__main__":
    main()
