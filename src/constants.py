"""MAIA EMG-ASL signal and model constants."""

# --- Signal acquisition ---
SAMPLE_RATE = 200          # Hz
N_CHANNELS = 8             # sEMG channels
WINDOW_SAMPLES = 40        # 200ms @ 200Hz
HOP_SAMPLES = 20           # 50% overlap
WINDOW_MS = 200            # ms
HOP_MS = 100               # ms

# --- Feature extraction ---
N_FEATURES_PER_CHANNEL = 10
FEATURE_DIM = N_CHANNELS * N_FEATURES_PER_CHANNEL  # 80

# --- Classes ---
ASL_CLASSES = [chr(i) for i in range(ord('A'), ord('Z') + 1)]  # A-Z
N_CLASSES = len(ASL_CLASSES)  # 26

# --- BLE packet format ---
BLE_PACKET_BYTES = 16   # 8 channels × int16
BLE_CHAR_UUID = "12345678-1234-5678-1234-56789abcdef0"
BLE_SERVICE_UUID = "12345678-1234-5678-1234-56789abcdef1"

# --- Model defaults ---
LSTM_HIDDEN = 128
LSTM_LAYERS = 2
LSTM_DROPOUT = 0.3
CNN_FILTERS = [32, 64]
CONFORMER_D_MODEL = 128
CONFORMER_N_HEADS = 4
CONFORMER_N_LAYERS = 4

# --- Training ---
BATCH_SIZE = 256
LEARNING_RATE = 1e-3
MAX_EPOCHS = 200
EARLY_STOP_PATIENCE = 15

# --- Inference ---
CONFIDENCE_THRESHOLD = 0.75
DEBOUNCE_MS = 300
