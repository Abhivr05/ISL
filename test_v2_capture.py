import os

DEBUG_DIR = "debug_v2_sequences"
os.makedirs(DEBUG_DIR, exist_ok=True)

sequence_counter = 0
import cv2
import numpy as np
import torch
from collections import deque

from models.tcn_model import TCNModel
from utils.mediapipe_utils import initialize_holistic
from utils.landmark_utils import extract_landmarks
from preprocessing.normalization import normalize_landmarks


# ============================================================
# SETTINGS
# ============================================================

MODEL_PATH = "trained_models/isl_tcn_19class.pth"

SEQUENCE_LENGTH = 30
FEATURE_DIMENSION = 225

# Movement detection
MOTION_THRESHOLD = 0.015
MOTION_BURST_REQUIRED = 3
MOTION_HISTORY_SIZE = 7

# Keep a few frames immediately before movement is detected
PRE_MOTION_FRAMES = 5

# Ignore very low-confidence predictions
CONFIDENCE_THRESHOLD = 0.50


# ============================================================
# LOAD MODEL
# ============================================================

print("Loading model...")

checkpoint = torch.load(
    MODEL_PATH,
    map_location="cpu"
)

class_names = checkpoint["classes"]
input_size = checkpoint["input_size"]
num_classes = len(class_names)

model = TCNModel(
    input_size=input_size,
    num_classes=num_classes
)

model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

print("Model loaded.")
print("Classes:")
print(class_names)


# ============================================================
# MEDIAPIPE
# ============================================================

holistic = initialize_holistic()


# ============================================================
# CAMERA
# ============================================================

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("ERROR: Could not open webcam.")
    exit()

print("\nCamera started.")
print("Press Q to quit.")
print("Make a sign only after the system is ready.\n")


# ============================================================
# STATE
# ============================================================

IDLE = 0
CAPTURING = 1

state = IDLE

pre_motion_buffer = deque(maxlen=PRE_MOTION_FRAMES)
motion_history = deque(maxlen=MOTION_HISTORY_SIZE)

capture_sequence = []

previous_features = None

motion_bursts = 0


# ============================================================
# PREDICTION
# ============================================================

def predict_sequence(sequence):

    sequence = np.asarray(
        sequence,
        dtype=np.float32
    )

    if sequence.shape != (SEQUENCE_LENGTH, FEATURE_DIMENSION):
        print(
            "Invalid sequence shape:",
            sequence.shape
        )
        return None, 0.0

    tensor = torch.tensor(
        sequence,
        dtype=torch.float32
    ).unsqueeze(0)

    with torch.no_grad():

        output = model(tensor)

        probabilities = torch.softmax(
            output,
            dim=1
        )

        confidence, index = torch.max(
            probabilities,
            dim=1
        )

    predicted_word = class_names[index.item()]
    confidence = confidence.item()

    return predicted_word, confidence


# ============================================================
# MAIN LOOP
# ============================================================

while True:

    ret, frame = cap.read()

    if not ret:
        print("Could not read frame.")
        break

    frame = cv2.flip(frame, 1)
    # --------------------------------------------------------
    # Convert BGR → RGB
    # --------------------------------------------------------

    rgb_frame = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )

    # --------------------------------------------------------
    # MediaPipe Holistic
    # --------------------------------------------------------

    results = holistic.process(rgb_frame)

    # --------------------------------------------------------
    # Extract 225 features
    # --------------------------------------------------------

    if results is None:
        cv2.imshow(
            "V2 Sign Recognition",
            frame
        )

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

        continue

    landmarks = extract_landmarks(results)

    if landmarks is None:
        cv2.imshow(
            "V2 Sign Recognition",
            frame
        )

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

        continue

    # --------------------------------------------------------
    # Normalize
    # --------------------------------------------------------

    features = normalize_landmarks(
        landmarks
    )

    features = np.asarray(
        features,
        dtype=np.float32
    )

    # Safety check
    if features.shape != (FEATURE_DIMENSION,):
        print(
            "Unexpected feature shape:",
            features.shape
        )
        continue

    # ========================================================
    # MOTION CALCULATION
    # ========================================================

    if previous_features is None:

        motion = 0.0

    else:

        motion = np.mean(
            np.abs(
                features - previous_features
            )
        )

    previous_features = features.copy()

    # Store recent frames
    pre_motion_buffer.append(
        features.copy()
    )

    # Store motion state
    is_moving = motion > MOTION_THRESHOLD

    motion_history.append(
        is_moving
    )

    # ========================================================
    # IDLE STATE
    # ========================================================

    if state == IDLE:

        # Count movement bursts
        if is_moving:

            motion_bursts += 1

        # Reset if there has been a long quiet period
        if len(motion_history) >= MOTION_HISTORY_SIZE:

            recent_movement = sum(
                motion_history
            )

            if recent_movement == 0:

                motion_bursts = 0

        # ----------------------------------------------------
        # Movement detected enough times
        # ----------------------------------------------------

        if motion_bursts >= MOTION_BURST_REQUIRED:

            print(
                "\nMovement detected!"
            )

            print(
                "Starting 30-frame capture..."
            )

            # Start with the few frames immediately before
            # movement detection.
            capture_sequence = list(
                pre_motion_buffer
            )

            # Make sure we don't already exceed 30
            capture_sequence = capture_sequence[
                -PRE_MOTION_FRAMES:
            ]

            state = CAPTURING

            motion_bursts = 0

            motion_history.clear()

    # ========================================================
    # CAPTURING STATE
    # ========================================================

    elif state == CAPTURING:

        capture_sequence.append(
            features.copy()
        )

        # ----------------------------------------------------
        # Exactly 30 frames
        # ----------------------------------------------------

        if len(capture_sequence) >= SEQUENCE_LENGTH:

            capture_sequence = capture_sequence[
                :SEQUENCE_LENGTH
            ]
            sequence_counter += 1

            debug_path = os.path.join(
                DEBUG_DIR,
                f"v2_sequence_{sequence_counter:03d}.npy"
            )

            np.save(
                debug_path,
                np.asarray(capture_sequence, dtype=np.float32)
            )

            print(f"Saved: {debug_path}")

            print(
                "30 frames captured."
            )

            # ------------------------------------------------
            # Predict
            # ------------------------------------------------

            word, confidence = predict_sequence(
                capture_sequence
            )

            if word is not None:

                if confidence >= CONFIDENCE_THRESHOLD:

                    print(
                        f"Prediction: {word}"
                    )

                    print(
                        f"Confidence: {confidence:.3f}"
                    )

                else:

                    print(
                        f"Prediction: {word}"
                    )

                    print(
                        f"Confidence: {confidence:.3f}"
                        f" (LOW)"
                    )

            print(
                "Waiting for next movement..."
            )

            # ------------------------------------------------
            # Reset
            # ------------------------------------------------

            capture_sequence = []

            pre_motion_buffer.clear()

            motion_history.clear()

            motion_bursts = 0

            state = IDLE

    # ========================================================
    # DISPLAY
    # ========================================================

    if state == IDLE:

        status = "READY - Make a sign"

    else:

        status = (
            f"CAPTURING "
            f"{len(capture_sequence)}/{SEQUENCE_LENGTH}"
        )

    cv2.putText(
        frame,
        status,
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2
    )

    cv2.putText(
        frame,
        f"Motion: {motion:.4f}",
        (20, 75),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2
    )

    cv2.imshow(
        "V2 Sign Recognition",
        frame
    )

    # --------------------------------------------------------
    # Quit
    # --------------------------------------------------------

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


# ============================================================
# CLEANUP
# ============================================================

cap.release()

holistic.close()

cv2.destroyAllWindows()

print("\nCamera stopped.")
