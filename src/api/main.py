"""MAIA EMG-ASL Inference Server — FastAPI + WebSocket."""
from __future__ import annotations

import asyncio
import logging
import os
import struct
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

import numpy as np
import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from src.api.auth import APIKeyMiddleware
from src.constants import (
    N_CHANNELS, WINDOW_SAMPLES, SAMPLE_RATE,
    N_CLASSES, ASL_CLASSES, CONFIDENCE_THRESHOLD,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Global model state
# ---------------------------------------------------------------------------
_ort_session: Optional[object] = None
_model_path: str = ""


def _load_onnx_model(path: str):
    global _ort_session, _model_path
    try:
        import onnxruntime as ort
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = int(os.environ.get("ORT_NUM_THREADS", "2"))
        _ort_session = ort.InferenceSession(path, sess_options=opts)
        _model_path = path
        logger.info(f"ONNX model loaded from {path}")
    except Exception as e:
        logger.error(f"Failed to load ONNX model: {e}")
        _ort_session = None


async def _download_model_from_r2():
    """Download the latest model from Cloudflare R2 at startup."""
    r2_url = os.environ.get("R2_MODEL_URL")
    if not r2_url:
        return

    dest = Path(os.environ.get("ONNX_MODEL_PATH", "models/asl_emg_classifier.onnx"))
    if dest.exists():
        logger.info(f"Model already exists at {dest}, skipping download")
        return

    dest.parent.mkdir(parents=True, exist_ok=True)
    logger.info(f"Downloading model from R2: {r2_url}")
    try:
        import httpx
        async with httpx.AsyncClient(timeout=120) as client:
            async with client.stream("GET", r2_url) as response:
                response.raise_for_status()
                with open(dest, "wb") as f:
                    async for chunk in response.aiter_bytes(chunk_size=65536):
                        f.write(chunk)
        logger.info(f"Model downloaded to {dest}")
    except Exception as e:
        logger.error(f"Failed to download model from R2: {e}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup: download model from R2 (if configured), load ONNX."""
    await _download_model_from_r2()
    model_path = os.environ.get("ONNX_MODEL_PATH", "models/asl_emg_classifier.onnx")
    if os.path.exists(model_path):
        _load_onnx_model(model_path)
    else:
        logger.warning(f"No ONNX model found at {model_path}. Set R2_MODEL_URL or ONNX_MODEL_PATH.")
    yield
    logger.info("MAIA server shutting down")


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------
app = FastAPI(
    title="MAIA EMG-ASL Inference API",
    version="0.1.0",
    description="Real-time ASL recognition from surface EMG signals",
    lifespan=lifespan,
)

app.add_middleware(APIKeyMiddleware)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# REST endpoints
# ---------------------------------------------------------------------------
@app.get("/health")
async def health():
    return {"status": "ok", "model_loaded": _ort_session is not None}


@app.get("/info")
async def info():
    return {
        "n_classes": N_CLASSES,
        "classes": ASL_CLASSES,
        "sample_rate": SAMPLE_RATE,
        "window_samples": WINDOW_SAMPLES,
        "n_channels": N_CHANNELS,
        "confidence_threshold": CONFIDENCE_THRESHOLD,
        "model_path": _model_path,
    }


@app.post("/predict")
async def predict_rest(body: dict):
    """
    REST endpoint for single window prediction.
    Body: {"window": [[ch0, ch1, ..., ch7], ...]}  (40 samples × 8 channels)
    """
    if _ort_session is None:
        raise HTTPException(503, "Model not loaded")
    window = np.array(body["window"], dtype=np.float32)
    if window.shape != (WINDOW_SAMPLES, N_CHANNELS):
        raise HTTPException(400, f"Expected window shape ({WINDOW_SAMPLES}, {N_CHANNELS}), got {window.shape}")
    ort_input = window.reshape(1, WINDOW_SAMPLES, N_CHANNELS)
    outputs = _ort_session.run(None, {"input": ort_input})
    logits = outputs[0][0]
    probs = _softmax(logits)
    pred_idx = int(np.argmax(probs))
    return {
        "class": ASL_CLASSES[pred_idx],
        "confidence": float(probs[pred_idx]),
        "probabilities": {ASL_CLASSES[i]: float(p) for i, p in enumerate(probs)},
    }


# ---------------------------------------------------------------------------
# WebSocket streaming endpoint
# ---------------------------------------------------------------------------
@app.websocket("/ws/emg")
async def websocket_emg(ws: WebSocket):
    """
    Binary WebSocket for real-time EMG inference.

    Client sends:  640 bytes  (40 samples × 8 channels × int16 big-endian)
    Server sends:  JSON  {"class": "A", "confidence": 0.92, "latency_ms": 12}
    """
    await ws.accept()
    logger.info("WebSocket client connected")
    try:
        while True:
            data = await ws.receive_bytes()
            t0 = time.perf_counter()

            if len(data) != WINDOW_SAMPLES * N_CHANNELS * 2:
                await ws.send_json({"error": f"Expected {WINDOW_SAMPLES * N_CHANNELS * 2} bytes, got {len(data)}"})
                continue

            # Decode big-endian int16
            n_samples = WINDOW_SAMPLES * N_CHANNELS
            raw = struct.unpack(f">{n_samples}h", data)
            window = (np.array(raw, dtype=np.float32) / 32768.0).reshape(WINDOW_SAMPLES, N_CHANNELS)

            if _ort_session is None:
                await ws.send_json({"error": "Model not loaded"})
                continue

            ort_input = window.reshape(1, WINDOW_SAMPLES, N_CHANNELS)
            outputs = _ort_session.run(None, {"input": ort_input})
            probs = _softmax(outputs[0][0])
            pred_idx = int(np.argmax(probs))
            confidence = float(probs[pred_idx])
            latency_ms = (time.perf_counter() - t0) * 1000

            response = {
                "class": ASL_CLASSES[pred_idx] if confidence >= CONFIDENCE_THRESHOLD else None,
                "confidence": confidence,
                "latency_ms": round(latency_ms, 2),
            }
            await ws.send_json(response)

    except WebSocketDisconnect:
        logger.info("WebSocket client disconnected")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        await ws.close(code=1011)


def _softmax(x: np.ndarray) -> np.ndarray:
    ex = np.exp(x - x.max())
    return ex / ex.sum()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("src.api.main:app", host="0.0.0.0", port=port, reload=False)
