import os
import cv2
import torch
import numpy as np

from collections import deque

from utils.mediapipe_utils import initialize_holistic
from utils.landmark_utils import extract_landmarks
from preprocessing.normalization import normalize_landmarks
from models.tcn_model import TCNModel


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_PATH = "trained_models/isl_tcn.pth"

SEQUENCE_LENGTH = 30
FEATURE_DIMENSION = 225

MOTION_THRESHOLD = 0.015
MOTION_BURST_REQUIRED = 3

# Keep a few frames from immediately before movement starts
PRE_MOTION_FRAMES = 5

DEBUG_DIR = "debug_8_signs"

os.makedirs(DEBUG_DIR, exist_ok=True)


# ============================================================
# DEVICE
# ============================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Using device:", device)


# ============================================================
# LOAD MODEL
# ============================================================

checkpoint = torch.load(
    MODEL_PATH,
    map_location=device
)

class_names = checkpoint["class_names"]

print()
print("Classes in model:")

for i, name in enumerate(class_names):
    print(f"{i}: {name}")


model = TCNModel(
    input_size=checkpoint["input_size"],
    num_classes=checkpoint["num_classes"]
)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

model.to(device)
model.eval()


# ============================================================
# MEDIAPIPE
# ============================================================

holistic = initialize_holistic()


# ============================================================
# WEBCAM
# ============================================================

cap = cv2.VideoCapture(0)

if not cap.isOpened():

    print("ERROR: Could not open webcam.")

    holistic.close()

    raise SystemExit


# ============================================================
# STATE
# ============================================================

state = "IDLE"

# Small buffer containing frames immediately before movement
pre_motion_buffer = deque(
    maxlen=PRE_MOTION_FRAMES
)

# Sequence being captured
capture_buffer = []

motion_history = deque(
    maxlen=7
)

previous_features = None

movement_count = 0

sequence_number = 0


# ============================================================
# MAIN LOOP
# ============================================================

print()
print("=" * 60)
print("REALTIME 8-SIGN DIAGNOSTIC")
print("=" * 60)
print()
print("Testing:")
print("GOOD  HELLO  HELP  I")
print("SORRY  THANK_YOU  WATER  YOU")
print()
print("Controls:")
print("  Q -> Quit")
print("  S -> Save current sequence")
print()
print("Perform ONE sign at a time.")
print()
print("The system captures 30 frames starting")
print("when movement is detected.")
print()
print("=" * 60)


try:

    while True:

        ret, frame = cap.read()

        if not ret:

            print("Could not read webcam frame.")

            break


        # ----------------------------------------------------
        # Mirror webcam
        # ----------------------------------------------------

        frame = cv2.flip(
            frame,
            1
        )


        # ----------------------------------------------------
        # MediaPipe
        # ----------------------------------------------------

        rgb = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        results = holistic.process(
            rgb
        )


        # ----------------------------------------------------
        # Extract landmarks
        # ----------------------------------------------------

        features = extract_landmarks(
            results
        )

        features = np.asarray(
            features,
            dtype=np.float32
        )

        if features.shape != (
            FEATURE_DIMENSION,
        ):

            continue


        # ----------------------------------------------------
        # Normalize
        # ----------------------------------------------------

        normalized = normalize_landmarks(
            features
        )

        normalized = np.asarray(
            normalized,
            dtype=np.float32
        )


        # ----------------------------------------------------
        # Motion calculation
        # ----------------------------------------------------

        motion = 0.0

        if previous_features is not None:

            motion = float(
                np.mean(
                    np.abs(
                        normalized -
                        previous_features
                    )
                )
            )

        previous_features = normalized.copy()


        # ----------------------------------------------------
        # Motion history
        # ----------------------------------------------------

        motion_history.append(
            motion
        )

        recent_movement = sum(
            value > MOTION_THRESHOLD
            for value in motion_history
        )

        is_moving = (
            recent_movement >=
            MOTION_BURST_REQUIRED
        )


        # ====================================================
        # IDLE
        # ====================================================

        if state == "IDLE":

            # Keep a few frames before movement
            pre_motion_buffer.append(
                normalized.copy()
            )


            if is_moving:

                movement_count += 1

            else:

                movement_count = 0


            # ------------------------------------------------
            # Movement detected
            # ------------------------------------------------

            if movement_count >= MOTION_BURST_REQUIRED:

                print()
                print(
                    "Movement detected -> "
                    "capturing 30 frames"
                )

                state = "CAPTURING"


                # Start sequence with the few frames
                # immediately before movement
                capture_buffer = list(
                    pre_motion_buffer
                )


                # The pre-motion buffer may contain
                # fewer than 5 frames at startup
                if len(capture_buffer) > (
                    SEQUENCE_LENGTH - 1
                ):

                    capture_buffer = (
                        capture_buffer[
                            -(SEQUENCE_LENGTH - 1):
                        ]
                    )


                # Add current frame
                capture_buffer.append(
                    normalized.copy()
                )


                movement_count = 0


        # ====================================================
        # CAPTURING
        # ====================================================

        elif state == "CAPTURING":

            capture_buffer.append(
                normalized.copy()
            )


            # ------------------------------------------------
            # Exactly 30 frames collected
            # ------------------------------------------------

            if len(capture_buffer) >= SEQUENCE_LENGTH:

                sequence = np.array(
                    capture_buffer[
                        :SEQUENCE_LENGTH
                    ],
                    dtype=np.float32
                )


                # =================================================
                # PREDICTION
                # =================================================

                input_tensor = torch.tensor(
                    sequence,
                    dtype=torch.float32
                ).unsqueeze(0).to(device)


                with torch.no_grad():

                    output = model(
                        input_tensor
                    )

                    probabilities = torch.softmax(
                        output,
                        dim=1
                    )

                    confidence, prediction = (
                        torch.max(
                            probabilities,
                            dim=1
                        )
                    )


                predicted_class = (
                    class_names[
                        prediction.item()
                    ]
                )

                confidence_value = (
                    confidence.item()
                )


                # =================================================
                # SAVE
                # =================================================

                sequence_number += 1

                filename = (
                    f"sequence_{sequence_number:03d}.npy"
                )

                filepath = os.path.join(
                    DEBUG_DIR,
                    filename
                )

                np.save(
                    filepath,
                    sequence
                )


                # =================================================
                # RESULT
                # =================================================

                print()
                print("=" * 60)

                print(
                    f"Prediction : "
                    f"{predicted_class}"
                )

                print(
                    f"Confidence : "
                    f"{confidence_value:.4f}"
                )

                print(
                    f"Sequence   : "
                    f"{filepath}"
                )

                print(
                    f"Shape      : "
                    f"{sequence.shape}"
                )

                print("=" * 60)


                # =================================================
                # TOP 3
                # =================================================

                top_k = min(
                    3,
                    len(class_names)
                )

                top_probs, top_indices = (
                    torch.topk(
                        probabilities[0],
                        top_k
                    )
                )


                print(
                    "Top predictions:"
                )

                for prob, idx in zip(
                    top_probs,
                    top_indices
                ):

                    print(
                        f"  "
                        f"{class_names[idx.item()]:12s}"
                        f" "
                        f"{prob.item():.4f}"
                    )


                # =================================================
                # RESET
                # =================================================

                state = "IDLE"

                capture_buffer = []

                pre_motion_buffer.clear()

                motion_history.clear()

                movement_count = 0


        # ====================================================
        # KEYBOARD
        # ====================================================

        key = cv2.waitKey(1) & 0xFF


        if key == ord("q"):

            break


        # Manual save
        if key == ord("s"):

            if len(capture_buffer) == SEQUENCE_LENGTH:

                sequence_number += 1

                filename = (
                    f"manual_{sequence_number:03d}.npy"
                )

                filepath = os.path.join(
                    DEBUG_DIR,
                    filename
                )

                np.save(
                    filepath,
                    np.array(
                        capture_buffer,
                        dtype=np.float32
                    )
                )

                print(
                    f"Saved manual sequence: "
                    f"{filepath}"
                )


        # ====================================================
        # DISPLAY
        # ====================================================

        cv2.putText(
            frame,
            f"STATE: {state}",
            (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 255),
            2
        )

        cv2.putText(
            frame,
            f"MOTION: {motion:.4f}",
            (20, 70),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2
        )

        cv2.putText(
            frame,
            f"CAPTURE: "
            f"{len(capture_buffer)}/{SEQUENCE_LENGTH}",
            (20, 105),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2
        )

        cv2.putText(
            frame,
            "Q = QUIT",
            (20, frame.shape[0] - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2
        )

        cv2.imshow(
            "8-Sign Realtime Diagnostic",
            frame
        )


finally:

    cap.release()

    cv2.destroyAllWindows()

    holistic.close()

    print()
    print("Diagnostic closed.")