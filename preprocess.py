import cv2
import mediapipe as mp
import numpy as np
import os
import json
from pathlib import Path
from tqdm import tqdm
import config

# ---------- Determine expected feature dimension at import time ----------
# We know the structure based on config.FEATURE_TYPE
if config.FEATURE_TYPE == "angles":
    EXPECTED_FEATURE_DIM = 40  # 20 left + 20 right
elif config.FEATURE_TYPE == "coords":
    EXPECTED_FEATURE_DIM = 126  # 63 left + 63 right
else:  # "both"
    EXPECTED_FEATURE_DIM = 166  # 40 + 126

print(f"Expected feature dimension per frame: {EXPECTED_FEATURE_DIM}")


# ---------- One-Euro Filter ----------
# This part just filters jittering in detection
# important but don't know the details that much
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


class OneEuroFilter:
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


# Applies the jittering filter to the landmarks (smoothens)
def smooth_landmarks(landmarks_3d, min_cutoff=1.0, beta=0.05):
    T, N, _ = landmarks_3d.shape
    smoothed = np.zeros_like(landmarks_3d)
    for n in range(N):
        for coord in range(3):
            filt = OneEuroFilter(min_cutoff=min_cutoff, beta=beta, d_cutoff=1.0)
            for t in range(T):
                smoothed[t, n, coord] = filt.filter(landmarks_3d[t, n, coord])
    return smoothed


# ---------- MediaPipe setup ----------
if config.USE_HOLISTIC:
    # Will detect hands and pose, super better use this one
    detector = mp.solutions.holistic.Holistic(
        static_image_mode=False,
        model_complexity=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )
else:
    # Will only detect hands
    detector = mp.solutions.hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        model_complexity=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )

# ---------- Angle computation ----------
HAND_TRIPLETS = [
    # Thumb
    (0, 1, 2),
    (1, 2, 3),
    (2, 3, 4),
    # Index finger
    (0, 5, 6),
    (5, 6, 7),
    (6, 7, 8),
    # Middle finger
    (0, 9, 10),
    (9, 10, 11),
    (10, 11, 12),
    # Ring finger
    (0, 13, 14),
    (13, 14, 15),
    (14, 15, 16),
    # Pinky
    (0, 17, 18),
    (17, 18, 19),
    (18, 19, 20),
    # Knuckle abduction angles (angles BETWEEN fingers at MCP)
    # Probably for normalization based on wrist
    (5, 0, 9),  # index_mcp → wrist → middle_mcp
    (9, 0, 13),  # middle_mcp → wrist → ring_mcp
    (13, 0, 17),  # ring_mcp → wrist → pinky_mcp
    (1, 0, 5),  # thumb_cmc → wrist → index_mcp
    (17, 0, 5),  # pinky_mcp → wrist → index_mcp (full span)
]


# Computes and returns the finalized angles
def compute_joint_angles(hand_coords):
    angles = np.zeros(len(HAND_TRIPLETS), dtype=np.float32)
    for i, (a, b, c) in enumerate(HAND_TRIPLETS):
        v1 = hand_coords[a] - hand_coords[b]
        v2 = hand_coords[c] - hand_coords[b]
        dot = np.dot(v1, v2)
        norm = np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8
        angles[i] = np.arccos(np.clip(dot / norm, -1.0, 1.0))
    return angles


# Normalizes hand coordination
def normalise_hand_coords(hand_coords):
    wrist = hand_coords[0]
    centered = hand_coords - wrist
    dists = np.linalg.norm(centered, axis=1)
    scale = np.max(dists) + 1e-8
    return centered / scale


# Returns the finalized and complete result of extraction
def extract_features_from_landmarks(left_lm, right_lm):
    """
    ALWAYS returns an array of exactly EXPECTED_FEATURE_DIM length.
    """
    # Start with zeros
    result = np.zeros(EXPECTED_FEATURE_DIM, dtype=np.float32)

    if config.FEATURE_TYPE == "angles":
        if np.any(left_lm):
            result[:20] = compute_joint_angles(left_lm)
        if np.any(right_lm):
            result[20:40] = compute_joint_angles(right_lm)

    elif config.FEATURE_TYPE == "coords":
        if np.any(left_lm):
            result[:63] = normalise_hand_coords(left_lm).flatten()
        if np.any(right_lm):
            result[63:126] = normalise_hand_coords(right_lm).flatten()

    else:  # "both"
        if np.any(left_lm):
            result[:20] = compute_joint_angles(left_lm)
            result[40:103] = normalise_hand_coords(left_lm).flatten()
        if np.any(right_lm):
            result[20:40] = compute_joint_angles(right_lm)
            result[103:166] = normalise_hand_coords(right_lm).flatten()

    return result


# ---------- Video processing ----------
# Doesn't skip frames for any reason, reads every frame

# Processes the video sample and returns the features.
def process_video(video_path):
    cap = cv2.VideoCapture(video_path)
    raw_left, raw_right = [], []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        # Converts to RGB for compatibility
        rgb_video = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = detector.process(rgb_video)

        # The main process, then appends to the empty lists
        lh = np.zeros((21, 3), dtype=np.float32)
        rh = np.zeros((21, 3), dtype=np.float32)
        if config.USE_HOLISTIC:
            if results.left_hand_landmarks:
                lh = np.array([[lm.x, lm.y, lm.z] for lm in results.left_hand_landmarks.landmark])
            if results.right_hand_landmarks:
                rh = np.array([[lm.x, lm.y, lm.z] for lm in results.right_hand_landmarks.landmark])
        else:
            if results.multi_hand_landmarks:
                for idx, hand_lm in enumerate(results.multi_hand_landmarks):
                    label = results.multi_handedness[idx].classification[0].label
                    lm = np.array([[l.x, l.y, l.z] for l in hand_lm.landmark])
                    if label == "Left":
                        lh = lm
                    else:
                        rh = lm
        raw_left.append(lh)
        raw_right.append(rh)
    cap.release()

    if len(raw_left) == 0:
        return None

    left_arr = np.array(raw_left)  # (T, 21, 3)
    right_arr = np.array(raw_right)  # (T, 21, 3)

    # Apply jittering filter
    if len(left_arr) >= 2:
        left_arr = smooth_landmarks(left_arr)
    if len(right_arr) >= 2:
        right_arr = smooth_landmarks(right_arr)

    # Build features — every frame guaranteed same dimension (coords or angles or both)
    features = np.zeros((len(left_arr), EXPECTED_FEATURE_DIM), dtype=np.float32)
    for t in range(len(left_arr)):
        features[t] = extract_features_from_landmarks(left_arr[t], right_arr[t])

    # Optional truncation (well better to record 1.1 secs - 2 secs)
    if config.MAX_FRAMES:
        features = features[:config.MAX_FRAMES]

    return features


# ---------- Main ----------
def main():
    # Makes the 'processed' folder and subfolders
    Path(config.PROCESSED_DIR).mkdir(parents=True, exist_ok=True)
    # Finds every class in the data folder
    class_dirs = sorted([d for d in os.listdir(config.DATA_DIR) if os.path.isdir(os.path.join(config.DATA_DIR, d))])
    # Labels every class
    class_map = {cls: idx for idx, cls in enumerate(class_dirs)}
    # Dumps it into class_map.json
    with open(os.path.join(config.PROCESSED_DIR, "class_map.json"), "w") as f:
        json.dump(class_map, f, indent=2)

    # Processes every video and dumps everything in 'processed' subfolders
    all_samples = []
    for cls in class_dirs:
        cls_path = os.path.join(config.DATA_DIR, cls)
        video_files = sorted(Path(cls_path).glob("*.mp4"))

        # Select only a range of videos
        start = max(config.START_INDEX - 1, 0)  # Convert to 0-based indexing
        end = config.END_INDEX  # None means until the end

        if end is not None:
            video_files = video_files[start:end]
        else:
            video_files = video_files[start:]
        print(
            f"Processing class '{cls}' "
            f"(videos {config.START_INDEX}"
            f"{'' if config.END_INDEX is None else f'-{config.END_INDEX}'})"
        )
        for vf in tqdm(video_files):
            frames = process_video(str(vf))
            if frames is None or frames.shape[0] < 2:
                tqdm.write(f"  Skipping {vf.name} (too few frames)")
                continue
            # Saves them as .npy
            out_name = f"{cls}_{vf.stem}.npy"
            np.save(os.path.join(config.PROCESSED_DIR, out_name), frames)
            all_samples.append({
                "file": out_name,
                "class": cls,
                "label": class_map[cls]
            })
    # A .json file which has the label, class and file name of every sample.
    with open(os.path.join(config.PROCESSED_DIR, "samples.json"), "w") as f:
        json.dump(all_samples, f, indent=2)
    print(f"Preprocessing done. Total samples: {len(all_samples)}")


if __name__ == "__main__":
    main()
