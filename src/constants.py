"""MAIA EMG-ASL signal and model constants.

Hardware target: Meta Neural Band (16 bipolar channels, 2kHz)
Research datasets: facebookresearch/generic-neuromotor-interface,
                   facebookresearch/emg2pose,
                   facebookresearch/emg2qwerty
"""

# --- Signal acquisition (Meta Neural Band spec) ---
SAMPLE_RATE    = 2000         # Hz  (2kHz, same as Meta research hardware)
N_CHANNELS     = 16           # bipolar sEMG channels
WINDOW_SAMPLES = 400          # 200ms @ 2kHz
HOP_SAMPLES    = 200          # 50% overlap → 100ms hop
WINDOW_MS      = 200          # ms
HOP_MS         = 100          # ms

# --- Feature extraction ---
N_FEATURES_PER_CHANNEL = 10
FEATURE_DIM = N_CHANNELS * N_FEATURES_PER_CHANNEL  # 160

# --- Classes ---
ASL_CLASSES = [chr(i) for i in range(ord('A'), ord('Z') + 1)]  # A-Z
N_CLASSES = len(ASL_CLASSES)  # 26

# --- BLE packet format (Meta Neural Band) ---
BLE_PACKET_BYTES  = 32   # 16 channels × int16 = 32 bytes per packet
# Note: Meta Neural Band BLE UUIDs — update once hardware arrives
BLE_CHAR_UUID     = "12345678-1234-5678-1234-56789abcdef0"
BLE_SERVICE_UUID  = "12345678-1234-5678-1234-56789abcdef1"

# --- WebSocket frame ---
WS_FRAME_BYTES = WINDOW_SAMPLES * N_CHANNELS * 2  # 12,800 bytes

# --- Model defaults ---
LSTM_HIDDEN       = 256       # larger hidden for 16ch input
LSTM_LAYERS       = 2
LSTM_DROPOUT      = 0.3
CNN_FILTERS       = [64, 128]
CONFORMER_D_MODEL = 256
CONFORMER_N_HEADS = 8
CONFORMER_N_LAYERS = 6

# --- Training ---
BATCH_SIZE          = 256
LEARNING_RATE       = 1e-3
MAX_EPOCHS          = 200
EARLY_STOP_PATIENCE = 15

# --- Inference ---
CONFIDENCE_THRESHOLD = 0.75
DEBOUNCE_MS          = 300

# --- Meta research dataset specs (for loaders) ---
META_SAMPLE_RATE    = 2000    # Hz — matches SAMPLE_RATE (no resampling needed)
META_N_CHANNELS     = 16      # matches N_CHANNELS
META_HIGHPASS_HZ    = 40.0    # Meta pre-applied high-pass filter cutoff
META_TASKS          = ("discrete_gestures", "handwriting", "wrist")
NINAPRO_SAMPLE_RATE = 2000    # Hz (DB5+)
