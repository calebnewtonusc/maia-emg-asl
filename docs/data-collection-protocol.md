# EMG-ASL Data Collection Protocol

This protocol defines the procedure for collecting sEMG + label paired data
once the MAIA Neural Band hardware arrives.

## Session Setup

### Equipment
- MAIA Neural Band (8-channel sEMG, 200Hz, BLE 5.0)
- iPhone with MAIA app (BLE connection)
- Webcam or phone camera (for vision teacher labeling)
- Quiet room, no strenuous activity 30 min prior

### Electrode Placement

```
Dorsal view -- right forearm:
  +------------------------------+
  |  Ch1  Ch2  Ch3  Ch4          |  <- Extensor group
  |  Ch5  Ch6  Ch7  Ch8          |  <- Flexor group
  +------------------------------+
  Wrist (distal) <-  -> Elbow (proximal)
  ~5cm from wrist crease
```

## Collection Procedure

1. **Rest baseline** (30s): Arm relaxed, record 30s of baseline noise
2. **Calibration** (10 signs x 5s): Perform A-J with 3s rest between signs
3. **Full alphabet** (26 signs x 10 reps x 3s): Each letter 10 times, 2s rest
4. **Rest** (30s): Final baseline

**Total per session: ~20 minutes**

## Labeling

Option 1 (Vision teacher, recommended):
```bash
python scripts/auto_label_session.py \
    --video session_001_video.mp4 \
    --emg data/train/session_001.npz
```

Option 2 (Manual): Add `labels` array to .npz with sample-level class indices.

## Quality Checks

Run after each session:
```bash
python scripts/validate_session.py --file data/train/session_001.npz
# Fix any issues automatically:
python scripts/validate_session.py --file data/train/session_001.npz --fix
```

## File Format

Sessions saved as `.npz` with keys:
- `emg`: `(N_samples, 8)` float32, normalized to +/-1.0 range
- `labels`: `(N_samples,)` int32, class index 0-25 (A-Z), -1 for unlabeled

## IRB Note

For formal data collection studies, document:
- Informed consent
- Age, handedness, prior ASL experience
- Exclusion: active skin lesions on forearm, pacemaker users

This protocol follows guidelines for minimal-risk physiological recording per 45 CFR 46.
