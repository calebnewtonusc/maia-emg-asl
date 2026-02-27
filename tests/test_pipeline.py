"""
MAIA test suite -- signal processing, models, augmentation.
Run: pytest tests/ -v
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

from src.constants import (
    N_CHANNELS, WINDOW_SAMPLES, N_CLASSES, FEATURE_DIM,
    SAMPLE_RATE, ASL_CLASSES,
)
from src.signal.filters import bandpass_filter, notch_filter, full_wave_rectify, preprocess
from src.signal.features import extract_features, window_signal
from src.data.loader import generate_synthetic_dataset, create_windows
from src.models.lstm_classifier import LSTMClassifier
from src.models.cnn_lstm_classifier import CNNLSTMClassifier
from src.models.conformer_classifier import ConformerClassifier


# =========================================================================
# Constants
# =========================================================================
class TestConstants:
    def test_feature_dim(self):
        assert FEATURE_DIM == N_CHANNELS * 10

    def test_classes(self):
        assert len(ASL_CLASSES) == N_CLASSES == 26
        assert ASL_CLASSES[0] == "A"
        assert ASL_CLASSES[-1] == "Z"


# =========================================================================
# Filters
# =========================================================================
class TestFilters:
    def test_bandpass_shape(self):
        sig = np.random.randn(200, N_CHANNELS).astype(np.float32)
        out = bandpass_filter(sig, fs=SAMPLE_RATE)
        assert out.shape == sig.shape

    def test_notch_shape(self):
        sig = np.random.randn(200, N_CHANNELS).astype(np.float32)
        out = notch_filter(sig, fs=SAMPLE_RATE)
        assert out.shape == sig.shape

    def test_rectify_nonnegative(self):
        sig = np.random.randn(100, N_CHANNELS)
        out = full_wave_rectify(sig)
        assert (out >= 0).all()

    def test_preprocess_pipeline(self):
        sig = np.random.randn(400, N_CHANNELS).astype(np.float32)
        out = preprocess(sig, fs=SAMPLE_RATE)
        assert out.shape == sig.shape
        assert (out >= 0).all()  # rectified


# =========================================================================
# Feature extraction
# =========================================================================
class TestFeatureExtraction:
    def test_feature_dim(self):
        window = np.random.randn(WINDOW_SAMPLES, N_CHANNELS).astype(np.float32)
        feat = extract_features(window)
        assert feat.shape == (FEATURE_DIM,)

    def test_window_signal(self):
        sig = np.random.randn(200, N_CHANNELS).astype(np.float32)
        feats = window_signal(sig, window_samples=WINDOW_SAMPLES, hop_samples=20)
        assert feats.ndim == 2
        assert feats.shape[1] == FEATURE_DIM


# =========================================================================
# Data loader
# =========================================================================
class TestDataLoader:
    def test_synthetic_shapes(self):
        X, y = generate_synthetic_dataset(n_samples_per_class=10, seed=0)
        assert X.shape[1:] == (WINDOW_SAMPLES, N_CHANNELS)
        assert len(X) == len(y)
        assert y.max() < N_CLASSES

    def test_create_windows(self):
        rng = np.random.default_rng(0)
        emg = rng.random((300, N_CHANNELS)).astype(np.float32)
        labels = rng.integers(0, N_CLASSES, 300).astype(np.int32)
        X, y = create_windows(emg, labels)
        assert X.ndim == 3
        assert X.shape[1:] == (WINDOW_SAMPLES, N_CHANNELS)


# =========================================================================
# LSTM Classifier
# =========================================================================
class TestLSTMClassifier:
    def test_forward_shape(self, tiny_lstm, synthetic_windows):
        X, _ = synthetic_windows
        out = tiny_lstm(X[:4])
        assert out.shape == (4, N_CLASSES)

    def test_predict_proba_sums_to_one(self, tiny_lstm, synthetic_windows):
        X, _ = synthetic_windows
        probs = tiny_lstm.predict_proba(X[:4])
        assert torch.allclose(probs.sum(dim=-1), torch.ones(4), atol=1e-5)

    def test_different_batch_sizes(self, tiny_lstm):
        for bs in [1, 8, 32]:
            x = torch.zeros(bs, WINDOW_SAMPLES, N_CHANNELS)
            out = tiny_lstm(x)
            assert out.shape == (bs, N_CLASSES)


# =========================================================================
# CNN-LSTM Classifier
# =========================================================================
class TestCNNLSTMClassifier:
    def test_forward_shape(self, synthetic_windows):
        model = CNNLSTMClassifier()
        X, _ = synthetic_windows
        out = model(X[:4])
        assert out.shape == (4, N_CLASSES)

    def test_batch_size_1(self):
        model = CNNLSTMClassifier()
        x = torch.zeros(1, WINDOW_SAMPLES, N_CHANNELS)
        out = model(x)
        assert out.shape == (1, N_CLASSES)


# =========================================================================
# Conformer Classifier
# =========================================================================
class TestConformerClassifier:
    def test_forward_shape(self, synthetic_windows):
        model = ConformerClassifier(d_model=32, n_layers=1, n_heads=2)
        X, _ = synthetic_windows
        out = model(X[:4])
        assert out.shape == (4, N_CLASSES)

    def test_parameter_count(self):
        model = ConformerClassifier()
        n_params = sum(p.numel() for p in model.parameters())
        # Should be reasonable (< 5M for small config)
        assert n_params > 10_000


# =========================================================================
# Augmentation
# =========================================================================
class TestAugmentation:
    def test_gaussian_noise(self):
        from src.data.augmentation import gaussian_noise
        rng = np.random.default_rng(0)
        w = np.ones((WINDOW_SAMPLES, N_CHANNELS), dtype=np.float32)
        out = gaussian_noise(w, rng)
        assert out.shape == w.shape
        assert not np.allclose(out, w)

    def test_channel_dropout(self):
        from src.data.augmentation import channel_dropout
        rng = np.random.default_rng(0)
        w = np.ones((WINDOW_SAMPLES, N_CHANNELS), dtype=np.float32)
        out = channel_dropout(w, rng, max_drop=4)
        assert out.shape == w.shape

    def test_pipeline_output_shape(self):
        from src.data.augmentation import get_default_pipeline
        pipe = get_default_pipeline("medium", seed=0)
        w = np.random.randn(WINDOW_SAMPLES, N_CHANNELS).astype(np.float32)
        out = pipe(w)
        assert out.shape == w.shape

    def test_mixup(self):
        from src.data.augmentation import mixup_emg
        rng = np.random.default_rng(0)
        X = np.random.randn(16, WINDOW_SAMPLES, N_CHANNELS).astype(np.float32)
        y = np.arange(16, dtype=np.int64) % N_CLASSES
        X_mixed, ya, yb, lam = mixup_emg(X, y, rng=rng)
        assert X_mixed.shape == X.shape
        assert 0.0 <= lam <= 1.0
