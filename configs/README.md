# Training Configurations

YAML configs for different training scenarios.

| File | Description | Hardware |
|------|-------------|----------|
| `lstm_default.yaml` | LSTM baseline | CPU/single GPU |
| `conformer_gpu.yaml` | Conformer multi-GPU | IYA Nvidia Lab (4+ GPUs) |

## Signal Constants Reference

| Constant | Value | Meaning |
|----------|-------|---------|
| SAMPLE_RATE | 200 Hz | Neural band sample rate |
| N_CHANNELS | 8 | sEMG channels |
| WINDOW_SAMPLES | 40 | 200ms window @ 200Hz |
| HOP_SAMPLES | 20 | 100ms hop (50% overlap) |
| FEATURE_DIM | 80 | 8 channels x 10 features |
| N_CLASSES | 26 | ASL letters A-Z |

## Usage

```python
import yaml
with open("configs/lstm_default.yaml") as f:
    cfg = yaml.safe_load(f)
```
