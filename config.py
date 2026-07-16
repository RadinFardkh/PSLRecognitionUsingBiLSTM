import os

# ========== Paths ==========
DATA_DIR = "data"  # folder with class subfolders
PROCESSED_DIR = "processed"  # where .npy files go
MODEL_DIR = "models"    # where models go bruh
LOG_DIR = "logs"

# ========== Preprocessing ==========
USE_HOLISTIC = True  # True = MediaPipe Holistic (pose+hands), False = Hands only
MAX_FRAMES = None  # keep all frames, or set to a fixed number (e.g., 60)
SEQUENCE_LENGTH = 60  # for padded training, fixed length (pads/truncates)
FEATURE_TYPE = "both"  # "angles", "coords", or "both"

# Normalisation for coordinates (if used)
COORD_NORM = "wrist"  # "wrist" (hand translation) or "shoulders" (body)
# For angles, no extra normalisation needed

START_INDEX = 61      # First sample to process (1-based)
END_INDEX = None     # Last sample to process (inclusive). None = until the end

# ========== Training ==========
MODEL_TYPE = "bilstm"  # "bilstm" or "transformer" / "bilstm" is the recommended
BATCH_SIZE = 32
EPOCHS = 60     # 60 for small database, 100 for bigger
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
DROPOUT = 0.3
REDUCE_LR_FACTOR = 0.1
REDUCE_LR_PATIENCE = 7
EARLY_STOPPING_PATIENCE = 15    # stop after x epochs if no loss

# BiLSTM specific
LSTM_UNITS = 128    # no need for 64 unless you have worm
LSTM_DROPOUT = DROPOUT  # the default dropout

# Transformer specific
D_MODEL = 64
NUM_HEADS = 4
NUM_LAYERS = 2
TRANSFORMER_DROPOUT = 0.2

# ========== TFLite Export ==========
QUANTIZE = True
TFLITE_MODEL_NAME = "psl_model.tflite"

# ========== Misc ==========
SEED = 42   # random, 42 is just popular
NUM_CLASSES = 33  # TODO: should change after any increase in database
