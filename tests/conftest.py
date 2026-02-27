"""Shared pytest fixtures for MAIA test suite."""
import numpy as np
import pytest
import torch

from src.constants import N_CHANNELS, WINDOW_SAMPLES, N_CLASSES
from src.data.loader import generate_synthetic_dataset


@pytest.fixture(scope="session")
def synthetic_session():
    """(X, y) arrays for the full test session."""
    X, y = generate_synthetic_dataset(n_samples_per_class=20, seed=0)
    return X, y


@pytest.fixture(scope="session")
def synthetic_windows(synthetic_session):
    """Torch tensors (X_t, y_t) for model tests."""
    X, y = synthetic_session
    return torch.from_numpy(X), torch.from_numpy(y)


@pytest.fixture(scope="function")
def tiny_lstm():
    """Small LSTM for fast unit tests."""
    from src.models.lstm_classifier import LSTMClassifier
    return LSTMClassifier(hidden_size=32, num_layers=1, dropout=0.0)
