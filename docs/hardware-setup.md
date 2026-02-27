# Hardware Setup -- MAIA Neural Band

## Parts

| Component | Spec | Notes |
|-----------|------|-------|
| MAIA Neural Band | 8-ch sEMG, 200Hz, BLE 5.0 | Ships with charging cable |
| Dry electrodes | Ag/AgCl, 10mm dia | Pre-applied on band |
| iPhone | iOS 16+, BLE 5.0 | For MAIA app |
| USB-C cable | For firmware flash | Included |

## BLE Connection

```
Service UUID:  12345678-1234-5678-1234-56789abcdef1
Notify UUID:   12345678-1234-5678-1234-56789abcdef0

Packet format: 16 bytes
  Bytes 0-1:   Channel 1 (int16 big-endian, range +-32767 = +-1V)
  Bytes 2-3:   Channel 2
  ...
  Bytes 14-15: Channel 8
  Rate: 200 packets/sec (200 Hz)
```

## Firmware Flash

1. Power off the band (hold button 3s)
2. Connect USB-C while holding MODE button
3. Run: `./scripts/flash_firmware.sh` *(firmware file provided separately)*
4. LED blinks green 3x when complete

## Electrode Placement

1. Clean forearm with isopropyl alcohol wipe, let dry 30s
2. Center the band ~5cm proximal to the wrist crease
3. Secure strap -- snug but not restricting circulation
4. Run BLE scan: `python scripts/test_websocket.py --url ws://localhost:8000/ws/emg`

## Troubleshooting

| Symptom | Likely cause | Fix |
|---------|-------------|-----|
| No BLE advertisement | Band off or discharged | Hold button 2s to power on; charge if LED red |
| Flat channel(s) | Poor electrode contact | Re-center band, clean skin |
| 60Hz noise | Power line interference | Move away from monitors; check grounding |
| High latency | BLE congestion | Disable other BLE devices |
| validate_session FAIL | Signal issues | Run `--fix` flag |
| App disconnects | BLE range | Move iPhone within 3m |
