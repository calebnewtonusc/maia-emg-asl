# Hardware Setup — MAIA Neural Band

Setup guide for the MAIA Neural Band: unboxing, firmware, BLE pairing, electrode placement, and per-session calibration.

---

## What's in the Box

| Component | Qty | Notes |
|-----------|-----|-------|
| MAIA Neural Band | 1 | 8-channel sEMG · 200Hz · BLE 5.0 |
| USB-C charging/flash cable | 1 | Used for charging and firmware updates |
| Isopropyl alcohol wipes | 10 | Skin prep before electrode contact |
| Replacement electrode gel pads | 2 sets | If dry electrodes lose conductivity |

---

## BLE Specification

```
Device name:    MAIA-Band
Service UUID:   12345678-1234-5678-1234-56789abcdef1
Notify UUID:    12345678-1234-5678-1234-56789abcdef0

Packet format:  16 bytes per packet, 200 packets/sec
  Bytes  0-1:   Channel 1  (int16 big-endian · range ±32767 = ±1.0V)
  Bytes  2-3:   Channel 2
  Bytes  4-5:   Channel 3
  Bytes  6-7:   Channel 4
  Bytes  8-9:   Channel 5
  Bytes 10-11:  Channel 6
  Bytes 12-13:  Channel 7
  Bytes 14-15:  Channel 8

Voltage range:  ±1.0V (maps to ±32767 int16)
Sample rate:    200 Hz
Window size:    40 samples = 200ms (used by model)
Hop size:       20 samples = 100ms (50% overlap)
```

### Parsing a packet (Python)

```python
import struct

def parse_packet(data: bytes) -> list[float]:
    """16-byte BLE packet → 8 float32 voltages in [-1.0, 1.0]"""
    channels = struct.unpack(">8h", data)   # big-endian int16 × 8
    return [v / 32767.0 for v in channels]
```

### Parsing a packet (TypeScript / React Native)

```typescript
function parsePacket(data: ArrayBuffer): number[] {
  const view = new DataView(data);
  return Array.from({ length: 8 }, (_, i) =>
    view.getInt16(i * 2, false) / 32767.0  // false = big-endian
  );
}
```

---

## First-Time Setup

### 1. Charge the band

Connect USB-C. LED behavior:
- Solid red → charging
- Solid green → fully charged
- Blinking red → battery critically low (< 5%)

Charge fully before first use (~90 minutes from empty).

### 2. Flash latest firmware

Firmware files are provided separately by MAIA Biotech.

```bash
# Power off: hold button 3 seconds until LED goes out
# Connect USB-C while holding MODE button (band enters DFU mode)
# LED blinks blue rapidly in DFU mode

./scripts/flash_firmware.sh --firmware firmware/maia_band_v1.x.x.bin
# LED blinks green 3× when complete — band reboots automatically
```

### 3. Power on

Short press the button (1 second). LED blinks green twice = booting. Solid green = advertising BLE and ready to pair.

### 4. Pair with iPhone

1. Open MAIA app → **Settings** → **BLE Device Name** → `MAIA-Band`
2. Tap **Connect** — the app scans and connects automatically
3. Connected: LED pulses green slowly (1s on, 1s off)
4. App shows live signal waveforms in **ASL Live** → **Debug Mode**

---

## Electrode Placement

See [data-collection-protocol.md](data-collection-protocol.md) for the full placement protocol. Summary:

1. **Clean skin** with isopropyl wipe, dry 30 seconds
2. **Position band** 5 cm proximal to the wrist crease, centered over the muscle belly
3. **Secure strap** — snug but two fingers should fit underneath
4. **Verify signal** — all 8 channels active with < 5µV RMS at rest

```
DORSAL view — right forearm:

         Elbow  ▲
                │
  ┌─────────────────────────────┐
  │  Ch1   Ch2   Ch3   Ch4      │  ← Extensor digitorum (dorsal)
  │  Ch5   Ch6   Ch7   Ch8      │  ← Flexor digitorum (volar)
  └─────────────────────────────┘
       5 cm from wrist crease
                │
         Wrist  ▼
```

### Channel anatomy reference

| Channel | Muscle group | Primary motion |
|---------|-------------|----------------|
| Ch1–Ch2 | Extensor digitorum communis | Finger extension |
| Ch3–Ch4 | Extensor carpi ulnaris | Wrist extension / ulnar dev. |
| Ch5–Ch6 | Flexor digitorum superficialis | Finger flexion |
| Ch7–Ch8 | Flexor carpi radialis | Wrist flexion / radial dev. |

---

## Per-Session Calibration

Run before every recording session (takes ~2 minutes):

```bash
python scripts/calibrate_user.py --user-id <id>
```

What it does:
1. Records 30s resting baseline (arm relaxed)
2. Computes per-channel mean + standard deviation
3. Saves normalization parameters to `data/calibrations/<user_id>_latest.json`
4. Validates signal quality — fails loudly if any channel is dead or saturated

The calibration file is loaded automatically by the MAIA app and inference server when the user ID is set.

### Recalibrate when

- Band was removed and replaced (even to the same position)
- More than 24 hours since last session
- Signal quality check fails
- Changing which arm is wearing the band

---

## LED Reference

| LED state | Meaning |
|-----------|---------|
| Solid red | Charging |
| Solid green | Fully charged (while plugged in) |
| Blinks green 2× on boot | Starting up |
| Slow pulse green (1s/1s) | BLE connected, streaming |
| Fast blink green | BLE advertising, waiting for connection |
| Blinks blue rapid | DFU (firmware flash) mode |
| Solid red (unplugged) | Battery < 10% — charge soon |
| Blinks red (unplugged) | Battery < 5% — shutting down |
| No light | Off |

---

## Troubleshooting

| Symptom | Likely cause | Fix |
|---------|-------------|-----|
| Band won't power on | Dead battery | Charge via USB-C (solid red = charging) |
| No BLE advertisement | Band is off | Short press button; check battery LED |
| App can't find band | BLE off on iPhone | Enable Bluetooth in iOS Settings |
| App connects then immediately disconnects | BLE interference | Move away from 2.4GHz Wi-Fi sources, other BLE devices |
| One or more flat channels | Poor electrode contact | Re-seat band; re-clean skin; check electrode pad condition |
| All channels saturated (±1.0V) | Loose connection or firmware issue | Re-seat band; reflash firmware |
| 60Hz noise on all channels | Power line interference | Move away from CRT monitors, fluorescent lights, power strips |
| 60Hz noise on individual channels | Broken electrode | Check pad condition; replace if degraded |
| Signal OK at rest but distorted during sign | Excessive strap tension | Loosen strap — over-tightening restricts blood flow and changes signal |
| `validate_session` fails SNR check | Low-quality session | Re-run calibration; ensure quiet room and still posture |
| Firmware flash fails | Band not in DFU mode | Re-enter DFU: hold MODE while connecting USB-C |

### Electrode pad replacement

Dry Ag/AgCl pads last approximately 20–30 sessions. Replace when:
- Resting noise exceeds 5µV on any channel after cleaning
- Visible wear, cracks, or delamination on the pad surface

To replace: peel off old pad (gentle pull at edge), clean the metal contact with isopropyl, press new pad firmly for 10 seconds.

---

## Storage & Care

- Store in the included case or a dry cloth pouch — avoid extreme temperatures
- Do not submerge in water — the band is splash-resistant, not waterproof
- Charge before long storage (> 2 weeks) to prevent battery degradation
- Keep electrode pads covered when not in use to prevent drying
- Clean the band housing with a damp cloth — do not use solvents
