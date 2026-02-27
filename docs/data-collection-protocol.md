# EMG-ASL Data Collection Protocol

This document is the authoritative guide for recording sEMG data with the MAIA Neural Band once hardware arrives. Follow it exactly every session to ensure cross-session consistency and model accuracy.

---

## Overview

| Parameter | Value |
|-----------|-------|
| Hardware | MAIA Neural Band (8-channel sEMG, 200Hz, BLE 5.0) |
| Session duration | ~20 minutes |
| Signs per session | 26 ASL letters × 10 repetitions = 260 trials |
| Output format | `.npz` — shape `(N_samples, 8)` EMG + `(N_samples,)` labels |
| Labeling method | Vision teacher (recommended) or manual |
| Target accuracy | > 95% on real data (vs. ~80% on synthetic) |

---

## Equipment Checklist

Before each session, confirm:

- [ ] MAIA Neural Band — charged (LED solid green when connected via USB-C)
- [ ] iPhone with MAIA app — Bluetooth enabled
- [ ] Second phone or webcam — for video recording (vision teacher labeling)
- [ ] Isopropyl alcohol wipes + paper towels — electrode prep
- [ ] Quiet room — no vibrating sources, avoid fluorescent lighting (60Hz interference)
- [ ] No strenuous forearm activity in the 30 minutes before recording

---

## Electrode Placement

Consistent placement is the single biggest factor in cross-session accuracy. Always photograph the band position before removing it.

### Anatomical landmarks

```
DORSAL view — right forearm:

         Elbow
           ▲
           │
    ┌──────────────────┐  ← 5 cm proximal to wrist crease
    │ Ch1  Ch2  Ch3  Ch4│  Extensor digitorum group
    │ Ch5  Ch6  Ch7  Ch8│  Flexor digitorum group
    └──────────────────┘
           │
           ▼
         Wrist crease
```

### Step-by-step

1. **Clean skin** — wipe the forearm with isopropyl alcohol from wrist to mid-forearm. Let dry for 30 seconds. Do not touch the cleaned area.
2. **Locate the wrist crease** — the horizontal fold that appears when you flex the wrist.
3. **Position the band** — center the band **5 cm proximal** (toward the elbow) from the wrist crease. The band should sit over the muscle belly, not over the tendons.
4. **Secure the strap** — snug enough that the electrodes maintain contact through forearm rotation, but loose enough to fit two fingers underneath. You should not see blanching (white skin) around the edges.
5. **Confirm contact** — run a quick signal check (see Signal Quality below). All 8 channels should show clean noise floor (< 5µV RMS at rest).
6. **Photograph** — take a photo of the band position from both dorsal and lateral views before each session. Store as `data/sessions/session_NNN_placement.jpg`.

### Left-handed participants

Mirror the placement: 5 cm proximal to the wrist crease on the left forearm. Note handedness in session metadata.

---

## Pre-Session Signal Quality Check

```bash
python scripts/validate_session.py --live --device MAIA-Band
```

This connects via BLE and streams live signal. Check:

| Metric | Acceptable | Action if failing |
|--------|-----------|-------------------|
| Resting RMS per channel | < 5µV | Re-clean skin, re-center band |
| 60Hz component | < 10% of total power | Move away from monitors/power strips |
| Dead channels (flat) | 0 | Re-seat band, check electrode contact |
| Signal symmetry (Ch1-4 vs Ch5-8) | Within 3× | Rotate band slightly |

Do not proceed if any channel is flat or shows > 20µV resting noise.

---

## Session Procedure

Run `python scripts/calibrate_user.py --user-id <id>` before the first session for a new user. This sets the per-user normalization baseline (takes ~2 minutes).

### Phase 1 — Resting baseline (30 seconds)

- Arm on the table, palm up, fingers relaxed
- Do not move, speak, or clench during this phase
- Saved as `baseline_start` in the `.npz` metadata

### Phase 2 — Warm-up calibration (5 minutes)

Perform letters **A through J** only, 3 repetitions each, 3-second hold, 2-second rest between.

Cues from the MAIA app: a letter is displayed on screen. Hold the static sign for 3 seconds when prompted, then relax fully.

### Phase 3 — Full alphabet recording (12 minutes)

26 letters × 10 repetitions each:
- **Hold duration:** 2 seconds per rep
- **Rest between reps:** 1.5 seconds
- **Rest between letters:** 5 seconds
- **Block break:** 1 minute after every 6 letters (A–F, G–L, M–R, S–Z)

**Sign cue timing:**

```
─────── 0s ── CUE appears ── 0.5s ── participant forms sign ── 2.5s ── CUE off ── 4s ── next ──
```

The 0.5-second pre-cue window is discarded during labeling to avoid movement artifact.

### Phase 4 — Resting baseline (30 seconds)

Same as Phase 1. Used to check for electrode drift.

**Total session time: ~20 minutes**

---

## Recording with the MAIA App

1. Open MAIA app → **Settings** → confirm server URL and BLE device name (`MAIA-Band`)
2. Tap **ASL Live** → **Start Recording Session**
3. Enter session ID (e.g., `caleb_001_session_003`) and participant metadata
4. Start second-device camera for video (needed for vision teacher labeling)
5. Follow on-screen cues
6. When session ends, tap **Export** → files save to `data/train/` via iCloud or AirDrop

---

## Auto-Labeling with Vision Teacher

The vision teacher pipeline uses MediaPipe Hands on the synchronized video to generate per-sample labels automatically. This is strongly preferred over manual labeling for accuracy and speed.

### Requirements

- Synchronized video recording (same device clock as EMG timestamps)
- Adequate lighting — MediaPipe needs clear hand visibility
- Stable camera angle (dorsal or frontal view of the signing hand)

### Run

```bash
python scripts/auto_label_session.py \
    --video data/sessions/caleb_001_session_003.mp4 \
    --emg   data/train/caleb_001_session_003.npz \
    --user-id caleb_001 \
    --output data/train/caleb_001_session_003_labeled.npz
```

The script:
1. Extracts MediaPipe Hands landmarks at 30fps
2. Resamples to 200Hz to match EMG timestamps
3. Classifies each frame using the cross-modal embedding
4. Writes per-sample labels into the `.npz` file
5. Reports labeling confidence — rejects samples below 0.7 confidence (marked `-1`)

### Manual labeling fallback

If video is unavailable, create a label array manually:

```python
import numpy as np
data = np.load("session.npz")
# labels: integer 0-25 (A=0, B=1, ... Z=25), -1 for unlabeled/rest
labels = np.full(len(data["emg"]), -1, dtype=np.int32)
# Fill in your manual annotations...
np.savez("session_labeled.npz", emg=data["emg"], labels=labels, **{k: data[k] for k in data if k not in ("emg", "labels")})
```

---

## Quality Validation

Run after every session before adding data to the training set:

```bash
python scripts/validate_session.py --file data/train/session_labeled.npz
```

### Pass thresholds

| Check | Threshold | Description |
|-------|-----------|-------------|
| Label coverage | > 80% | Fraction of samples with valid label (≥ 0) |
| Per-class count | ≥ 5 reps | Each of 26 letters must have ≥ 5 labeled reps |
| Channel SNR | > 20 dB | Signal-to-noise per channel |
| Baseline drift | < 10µV shift | Between Phase 1 and Phase 4 baselines |
| Motion artifact | < 5% samples | Detected via high-frequency burst filter |

### Auto-fix mode

```bash
python scripts/validate_session.py --file data/train/session_labeled.npz --fix
```

Applies: drift correction, notch filter re-application, artifact interpolation. Does not fix missing labels.

---

## File Format

All session files are `.npz` archives:

```python
np.savez(
    "session.npz",
    emg=...,         # (N_samples, 8)  float32  — normalized to [-1.0, 1.0]
    labels=...,      # (N_samples,)    int32    — class 0–25 (A–Z), -1 = unlabeled/rest
    sample_rate=200, # scalar          int      — always 200
    user_id=...,     # string                   — e.g. "caleb_001"
    session_id=...,  # string                   — e.g. "caleb_001_session_003"
    date=...,        # string                   — ISO 8601 e.g. "2026-03-15"
    handedness=...,  # string                   — "right" or "left"
    placement_cm=5,  # scalar          float    — distance from wrist crease in cm
)
```

---

## Multi-Session Best Practices

| Practice | Why |
|----------|-----|
| Same time of day | Muscle fatigue + hydration affect impedance |
| Same electrode position | Use placement photos to reproduce exactly |
| Minimum 3 sessions before training | Increases generalization across electrode shift |
| Calibrate each session | Per-session normalization compensates for day-to-day variance |
| Record at least 1 "degraded" session | Intentional slight misplacement builds robustness |
| 24-hour rest between sessions | Avoids fatigue-related signal drift |

### Recommended collection roadmap

```
Session 1  →  Calibration + learn the protocol
Session 2  →  First full 260-trial dataset
Session 3  →  Same position — confirms repeatability
Session 4  →  Slightly different band position (1 cm shift) — robustness
Session 5+ →  Add new participants for cross-user generalization
```

---

## Participant Metadata

Record for each participant (store in `data/participants/<user_id>/metadata.json`):

```json
{
  "user_id": "caleb_001",
  "age": 21,
  "sex": "M",
  "handedness": "right",
  "forearm_circumference_cm": 28,
  "asl_experience": "none",
  "sessions": ["caleb_001_session_001", "caleb_001_session_002"],
  "exclusions": []
}
```

---

## IRB / Ethics

For formal data collection involving other participants:

- Obtain written informed consent before any recording
- Disclose: wrist-worn electrode band, physiological data collected, data stored on research servers
- Exclusion criteria: active skin lesions or rash on forearm, implanted cardiac devices (pacemaker/ICD), known latex allergy (electrode adhesive)
- Participants may withdraw at any time and request data deletion
- Data stored on Cloudflare R2 with access restricted to project contributors

This protocol is consistent with guidelines for minimal-risk physiological recording under **45 CFR 46.104(d)(2)** (benign behavioral intervention). IRB review recommended before collecting data from participants other than yourself.
