#!/usr/bin/env python3
"""
Export trained model to Apple CoreML (.mlpackage) for on-device iOS inference.

Targets the Apple Neural Engine (iOS 16+, .mlpackage format).

Usage:
    python scripts/export_coreml.py --model models/asl_emg_classifier.pt
    python scripts/export_coreml.py --model models/asl_emg_classifier.pt --quantize
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parents[1]))

from src.constants import N_CHANNELS, WINDOW_SAMPLES, N_CLASSES, ASL_CLASSES
from src.models.lstm_classifier import LSTMClassifier


def export_coreml(model_path: str, output_dir: str = "models", quantize: bool = False):
    try:
        import coremltools as ct
    except ImportError:
        print("ERROR: coremltools not installed. Run: pip install coremltools")
        print("Note: coremltools requires macOS and Python 3.8-3.11")
        sys.exit(1)

    print(f"Loading model: {model_path}")
    model = LSTMClassifier()
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()

    # Trace model
    dummy = torch.zeros(1, WINDOW_SAMPLES, N_CHANNELS)
    traced = torch.jit.trace(model, dummy)

    # Convert to CoreML
    print("Converting to CoreML...")
    mlmodel = ct.convert(
        traced,
        inputs=[ct.TensorType(
            name="input",
            shape=(1, WINDOW_SAMPLES, N_CHANNELS),
            dtype=float,
        )],
        outputs=[ct.TensorType(name="output")],
        minimum_deployment_target=ct.target.iOS16,
        compute_units=ct.ComputeUnit.ALL,  # use ANE + GPU + CPU
    )

    # Add metadata
    mlmodel.author = "MAIA Biotech"
    mlmodel.short_description = "Real-time ASL letter recognition from 8-channel sEMG"
    mlmodel.version = "0.1.0"
    mlmodel.input_description["input"] = "sEMG window: (1, 40, 8) float32, 200ms @ 200Hz"
    mlmodel.output_description["output"] = f"Logits for {N_CLASSES} ASL classes (A-Z)"

    # Optional linear quantization (reduces size ~4x)
    if quantize:
        from coremltools.optimize.coreml import linear_quantize_weights, OptimizationConfig
        config = OptimizationConfig()
        mlmodel = linear_quantize_weights(mlmodel, config)

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = "_quantized" if quantize else ""

    # .mlpackage (iOS 16+, uses Neural Engine)
    mlpackage_path = out_dir / f"asl_emg_classifier{suffix}.mlpackage"
    mlmodel.save(str(mlpackage_path))
    print(f"Saved: {mlpackage_path}")

    # Also save .mlmodel for iOS 15 compatibility
    mlmodel_path = out_dir / f"asl_emg_classifier{suffix}.mlmodel"
    mlmodel.save(str(mlmodel_path))
    print(f"Saved: {mlmodel_path}")

    size_mb = sum(f.stat().st_size for f in mlpackage_path.rglob("*") if f.is_file()) / 1e6
    print(f"CoreML package size: {size_mb:.1f} MB")


def main():
    parser = argparse.ArgumentParser(description="Export model to Apple CoreML")
    parser.add_argument("--model", default="models/asl_emg_classifier.pt")
    parser.add_argument("--output-dir", default="models")
    parser.add_argument("--quantize", action="store_true", help="Apply linear quantization")
    args = parser.parse_args()
    export_coreml(args.model, args.output_dir, args.quantize)


if __name__ == "__main__":
    main()
