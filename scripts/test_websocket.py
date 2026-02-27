#!/usr/bin/env python3
"""
WebSocket test client for MAIA inference server.

Usage:
    python scripts/test_websocket.py
    python scripts/test_websocket.py --url wss://your-app.railway.app/ws/emg --api-key yourkey
    python scripts/test_websocket.py --stress --n-windows 100
"""
from __future__ import annotations

import argparse
import asyncio
import struct
import sys
import time
import numpy as np

from src.constants import N_CHANNELS, WINDOW_SAMPLES


def make_binary_frame(window: np.ndarray) -> bytes:
    """Encode (40, 8) float32 window -> 640-byte big-endian int16."""
    clipped = np.clip(window, -1.0, 1.0)
    int16_data = (clipped * 32767).astype(np.int16)
    return struct.pack(f">{WINDOW_SAMPLES * N_CHANNELS}h", *int16_data.flatten())


async def run_test(url: str, api_key: str, n_windows: int = 5, stress: bool = False):
    try:
        import websockets
    except ImportError:
        print("ERROR: websockets not installed. Run: pip install websockets")
        sys.exit(1)

    headers = {"X-API-Key": api_key}
    rng = np.random.default_rng(42)

    print(f"Connecting to {url}")
    try:
        async with websockets.connect(url, additional_headers=headers, open_timeout=10) as ws:
            print(f"Connected! Sending {n_windows} windows...\n")
            latencies = []
            for i in range(n_windows):
                # Synthetic window
                window = rng.normal(0, 0.1, (WINDOW_SAMPLES, N_CHANNELS)).astype(np.float32)
                frame = make_binary_frame(window)
                t0 = time.perf_counter()
                await ws.send(frame)
                response = await asyncio.wait_for(ws.recv(), timeout=5.0)
                latency = (time.perf_counter() - t0) * 1000
                latencies.append(latency)
                import json
                result = json.loads(response)
                if not stress or i % 20 == 0:
                    print(f"  [{i+1:3d}] class={result.get('class','?'):>4}  "
                          f"conf={result.get('confidence',0):.3f}  "
                          f"latency={latency:.1f}ms")
            print(f"\nLatency: mean={np.mean(latencies):.1f}ms  "
                  f"p95={np.percentile(latencies, 95):.1f}ms  "
                  f"p99={np.percentile(latencies, 99):.1f}ms")
    except Exception as e:
        print(f"ERROR: {e}")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="ws://localhost:8000/ws/emg")
    parser.add_argument("--api-key", default="dev-key-change-me")
    parser.add_argument("--n-windows", type=int, default=10)
    parser.add_argument("--stress", action="store_true")
    args = parser.parse_args()
    if args.stress:
        args.n_windows = 500
    asyncio.run(run_test(args.url, args.api_key, args.n_windows, args.stress))


if __name__ == "__main__":
    main()
