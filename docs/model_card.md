# Model Card: MAIA ASL-EMG Classifier

## Overview

| Field | Value |
|-------|-------|
| Model name | `asl_emg_classifier` |
| Task | Multi-class ASL letter recognition (A-Z) |
| Modality | Surface EMG (sEMG) |
| Architecture | LSTM (bidirectional-capable) |
| Parameters | ~246K (LSTM) / ~1.5M (Conformer) |
| Input | (1, 40, 8) float32 -- 200ms window, 8 channels, 200Hz |
| Output | (1, 26) logits -- 26 ASL letters |
| Format | ONNX opset 17, CoreML .mlpackage |
| Hardware target | iPhone (Neural Engine), Railway (CPU inference) |

## Signal Processing

1. Bandpass filter: 20-450 Hz (4th-order Butterworth)
2. Notch filter: 60 Hz (Q=30)
3. Full-wave rectification
4. Sliding window: 200ms / 50% overlap

## Performance

| Training Data | Val Accuracy | Notes |
|---------------|-------------|-------|
| Synthetic (2,600 samples) | ~70-85% | Baseline, no real data |
| Synthetic + augmentation | ~80-90% | Medium augmentation pipeline |
| Real EMG (when collected) | TBD | Target: >95% |
| WLASL cross-modal pre-train | TBD | With vision teacher bootstrap |

## Inference Latency (CPU)

| Runtime | Mean | P95 |
|---------|------|-----|
| PyTorch CPU | ~8ms | ~12ms |
| ONNX Runtime CPU | ~4ms | ~6ms |
| CoreML (Apple Neural Engine) | ~2ms | ~3ms |

## Limitations

- Synthetic training data -- accuracy will improve dramatically once real sEMG recordings are collected
- Trained on A-Z only -- does not cover ASL words or phrases
- Electrode placement sensitivity -- requires consistent wrist placement
- Not validated on users with different muscle mass, hand sizes, or arm lengths

## Changelog

| Version | Date | Notes |
|---------|------|-------|
| 0.1.0 | 2026-02 | Initial baseline, synthetic data |

## Citation

```bibtex
@software{maia_emg_asl,
  author = {Newton, Caleb and MAIA Biotech},
  title = {MAIA EMG-ASL: Real-time Sign Language Recognition from sEMG},
  year = {2026},
  url = {https://github.com/calebnewtonusc/maia-emg-asl}
}
```
