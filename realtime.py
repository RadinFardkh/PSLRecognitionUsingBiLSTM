import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
import json
import os
from collections import deque
import config

# -------------- Load TFLite model & class map --------------

interpreter = tf.lite.Interpreter(
    model_path=os.path.join(config.MODEL_DIR, config.TFLITE_MODEL_NAME)
)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

with open(os.path.join(config.PROCESSED_DIR, "class_map.json")) as f:
    class_map = json.load(f)
idx_to_class = {v: k for k, v in class_map.items()}


# ----------------- Causal One-Euro filter -----------------

# This time it's frame-by-frame which realllly helps the flow of prediction
class LowPassFilter:
    def __init__(self, alpha):
        self.alpha = alpha
        self.y_prev = None

    def filter(self, x):
        if self.y_prev is None:
            self.y_prev = x
            return x
        y = self.alpha * x + (1 - self.alpha) * self.y_prev
        self.y_prev = y
        return y


class CausalOneEuroFilter:
    def __init__(self, min_cutoff=1.0, beta=0.0, d_cutoff=1.0):
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
        self.x_prev = None
        self.dx_prev = None
        self.lp_x = LowPassFilter(self._alpha(min_cutoff))
        self.lp_dx = LowPassFilter(self._alpha(d_cutoff))
        self.first_time = True

    def _alpha(self, cutoff, dt=1.0):
        tau = 1.0 / (2 * np.pi * cutoff)
        return 1.0 / (1.0 + tau / dt)

    def filter(self, x, dt=1.0):
        if self.first_time:
            self.x_prev = x
            self.dx_prev = 0.0
            self.first_time = False
            return x
        dx = (x - self.x_prev) / dt
        dx_hat = self.lp_dx.filter(dx)
        cutoff = self.min_cutoff + self.beta * abs(dx_hat)
        self.lp_x = LowPassFilter(self._alpha(cutoff, dt))
        x_hat = self.lp_x.filter(x)
        self.x_prev = x_hat
        return x_hat


# Create a bank of filters – one per hand landmark coordinate
def create_filter_bank():
    bank = []
    for hand in range(2):  # left, right
        hand_bank = []
        for lm in range(21):
            lm_bank = [CausalOneEuroFilter(min_cutoff=1.0, beta=0.05) for _ in range(3)]
            hand_bank.append(lm_bank)
        bank.append(hand_bank)
    return bank


filter_bank = create_filter_bank()


def filter_landmarks(left_raw, right_raw):
    """Apply causal filter to current frame's landmarks."""
    left_out = np.zeros_like(left_raw)
    right_out = np.zeros_like(right_raw)
    for lm in range(21):
        for c in range(3):
            left_out[lm, c] = filter_bank[0][lm][c].filter(left_raw[lm, c])
            right_out[lm, c] = filter_bank[1][lm][c].filter(right_raw[lm, c])
    return left_out, right_out


# -------------- Feature extraction (must match config.FEATURE_TYPE) --------------
HAND_TRIPLETS = [
    (0, 1, 2), (1, 2, 3), (2, 3, 4),
    (0, 5, 6), (5, 6, 7), (6, 7, 8),
    (0, 9, 10), (9, 10, 11), (10, 11, 12),
    (0, 13, 14), (13, 14, 15), (14, 15, 16),
    (0, 17, 18), (17, 18, 19), (18, 19, 20),
    (5, 0, 9), (9, 0, 13), (13, 0, 17), (1, 0, 5), (17, 0, 5)
]


def compute_joint_angles(hand_coords):
    angles = np.zeros(20, dtype=np.float32)
    for i, (a, b, c) in enumerate(HAND_TRIPLETS):
        v1 = hand_coords[a] - hand_coords[b]
        v2 = hand_coords[c] - hand_coords[b]
        dot = np.dot(v1, v2)
        norm = np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8
        angles[i] = np.arccos(np.clip(dot / norm, -1.0, 1.0))
    return angles


# NORMALIZINGGGGG
def normalise_hand_coords(hand_coords):
    wrist = hand_coords[0]
    centered = hand_coords - wrist
    dists = np.linalg.norm(centered, axis=1)
    scale = np.max(dists) + 1e-8
    return (centered / scale).flatten()


def extract_features_from_landmarks(left_lm, right_lm):
    """
    Returns a 1D feature vector according to config.FEATURE_TYPE.
    """
    left_angles = np.zeros(20, dtype=np.float32)
    right_angles = np.zeros(20, dtype=np.float32)
    left_coords = np.zeros(63, dtype=np.float32)
    right_coords = np.zeros(63, dtype=np.float32)

    if np.any(left_lm):
        left_angles = compute_joint_angles(left_lm)
        left_coords = normalise_hand_coords(left_lm)
    if np.any(right_lm):
        right_angles = compute_joint_angles(right_lm)
        right_coords = normalise_hand_coords(right_lm)

    if config.FEATURE_TYPE == "angles":
        return np.concatenate([left_angles, right_angles])
    elif config.FEATURE_TYPE == "coords":
        return np.concatenate([left_coords, right_coords])
    else:  # "both"
        return np.concatenate([left_angles, right_angles, left_coords, right_coords])


# --------------------- Sliding window & majority voting ---------------------
sequence_buffer = deque(maxlen=config.SEQUENCE_LENGTH)
prediction_window = deque(maxlen=15)
current_prediction = None

# ------------------------ MediaPipe setup ------------------------
mp_holistic = mp.solutions.holistic
detector = mp_holistic.Holistic(
    # If True, it would treat every frame like a completely un-related photo to other
    # frames, better for image detection
    static_image_mode=False,
    model_complexity=1,  # 0 would be faster but dumber so why not 1?
    min_detection_confidence=0.55,  # Just changed them to .55 instead of .50
    min_tracking_confidence=0.55
)

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Cannot open camera")
    exit()

print("Real‑time sign recognition started. Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = detector.process(rgb)

    # Extract raw landmarks (zero if missing)
    left_raw = np.zeros((21, 3), dtype=np.float32)
    right_raw = np.zeros((21, 3), dtype=np.float32)
    if results.left_hand_landmarks:
        left_raw = np.array([[lm.x, lm.y, lm.z] for lm in results.left_hand_landmarks.landmark])
    if results.right_hand_landmarks:
        right_raw = np.array([[lm.x, lm.y, lm.z] for lm in results.right_hand_landmarks.landmark])

    # Apply causal filtering
    left_filt, right_filt = filter_landmarks(left_raw, right_raw)

    # Compute feature vector
    feat = extract_features_from_landmarks(left_filt, right_filt)
    sequence_buffer.append(feat)

    # Run inference every 3 frames (save compute)
    if len(sequence_buffer) == config.SEQUENCE_LENGTH and (len(sequence_buffer) % 3 == 0):
        seq = np.array(sequence_buffer, dtype=np.float32)  # (T, D)
        seq = np.expand_dims(seq, axis=0)  # (1, T, D)
        interpreter.set_tensor(input_details[0]['index'], seq)
        interpreter.invoke()
        output = interpreter.get_tensor(output_details[0]['index'])[0]
        pred_class = np.argmax(output)
        prediction_window.append(pred_class)

        # Majority voting (≥70% to change)
        if len(prediction_window) == prediction_window.maxlen:
            counts = np.bincount(prediction_window)
            dominant = np.argmax(counts)
            if counts[dominant] / len(prediction_window) >= 0.7:
                current_prediction = dominant
            else:
                current_prediction = None

    # Display result
    if current_prediction is not None:
        sign_name = idx_to_class[current_prediction]
        cv2.putText(frame, f"Sign: {sign_name}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    else:
        cv2.putText(frame, "No sign", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    cv2.imshow('Real-time PSL Recognition', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
detector.close()
print("Demo finished.")
