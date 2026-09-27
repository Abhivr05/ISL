import os
import cv2
import numpy as np
from collections import deque

from utils.mediapipe_utils import initialize_holistic
from utils.landmark_utils import extract_landmarks
from preprocessing.normalization import normalize_landmarks


# ============================================================
# CONFIGURATION
# ============================================================

SEQUENCE_LENGTH = 30
FEATURE_DIMENSION = 225

MOTION_THRESHOLD = 0.015
MOTION_BURST_REQUIRED = 3
MOTION_HISTORY_SIZE = 7

SIGN_END_FRAMES = 10

OUTPUT_DIR = "debug_labeled_sequences"

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# LABELS
# ============================================================

sequence_counter = {
    "YES": 0,
    "NO": 0
}

current_label = None


# ============================================================
# STATE
# ============================================================

# IMPORTANT:
# The program starts in IDLE.
state = "IDLE"

frame_buffer = deque(maxlen=SEQUENCE_LENGTH)

motion_history = deque(maxlen=MOTION_HISTORY_SIZE)

previous_features = None

movement_count = 0
still_count = 0


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


print()
print("=" * 60)
print("LABELED LIVE SEQUENCE CAPTURE")
print("=" * 60)
print()
print("Controls:")
print("  Y -> Select YES")
print("  N -> Select NO")
print("  Q -> Quit")
print()
print("After selecting a label:")
print("  Perform the sign.")
print("  The system will automatically detect the movement.")
print("  After the movement stops, the latest 30 frames are saved.")
print()
print("Output folder:")
print(f"  {OUTPUT_DIR}")
print()
print("=" * 60)
print()


# ============================================================
# MAIN LOOP
# ============================================================

try:

    while True:

        ret, frame = cap.read()

        if not ret:
            print("ERROR: Could not read frame from webcam.")
            break

        # Mirror webcam
        frame = cv2.flip(frame, 1)

        # ----------------------------------------------------
        # MEDIAPIPE
        # ----------------------------------------------------

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        results = holistic.process(rgb_frame)

        # ----------------------------------------------------
        # LANDMARK EXTRACTION
        # ----------------------------------------------------

        features = extract_landmarks(results)

        # Make sure we have exactly 225 features
        features = np.asarray(features, dtype=np.float32)

        if features.shape != (FEATURE_DIMENSION,):
            print(
                f"WARNING: Unexpected feature shape: {features.shape}. "
                f"Expected ({FEATURE_DIMENSION},)"
            )

            cv2.imshow("Labeled Live Capture", frame)

            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                break

            continue

        # ----------------------------------------------------
        # NORMALIZATION
        # ----------------------------------------------------

        normalized_features = normalize_landmarks(features)

        normalized_features = np.asarray(
            normalized_features,
            dtype=np.float32
        )

        # ----------------------------------------------------
        # ROLLING 30-FRAME BUFFER
        # ----------------------------------------------------

        frame_buffer.append(normalized_features)

        # ----------------------------------------------------
        # MOTION DETECTION
        # ----------------------------------------------------

        motion_value = 0.0

        if previous_features is not None:

            motion_value = float(
                np.mean(
                    np.abs(
                        normalized_features - previous_features
                    )
                )
            )

        previous_features = normalized_features.copy()

        motion_history.append(motion_value)

        # Determine whether there is movement
        recent_movement = sum(
            value > MOTION_THRESHOLD
            for value in motion_history
        )

        is_moving = recent_movement >= MOTION_BURST_REQUIRED

        # ----------------------------------------------------
        # KEYBOARD INPUT
        # ----------------------------------------------------

        key = cv2.waitKey(1) & 0xFF

        # Quit
        if key == ord("q"):
            break

        # Select YES
        elif key == ord("y"):

            current_label = "YES"

            # Reset state for a fresh capture
            state = "IDLE"
            still_count = 0
            movement_count = 0
            frame_buffer.clear()

            print()
            print("Selected label: YES")
            print("Perform the YES sign...")

        # Select NO
        elif key == ord("n"):

            current_label = "NO"

            # Reset state for a fresh capture
            state = "IDLE"
            still_count = 0
            movement_count = 0
            frame_buffer.clear()

            print()
            print("Selected label: NO")
            print("Perform the NO sign...")

        # ----------------------------------------------------
        # STATE MACHINE
        # ----------------------------------------------------

        # ====================================================
        # IDLE
        # ====================================================

        if state == "IDLE":

            # Do NOT start capturing if no label was selected.
            if current_label is not None:

                if is_moving:

                    movement_count += 1

                else:

                    movement_count = 0

                # Start SIGNING when movement is detected
                if movement_count >= MOTION_BURST_REQUIRED:

                    state = "SIGNING"

                    still_count = 0

                    print(
                        f"\nMovement detected -> "
                        f"capturing {current_label}"
                    )

        # ====================================================
        # SIGNING
        # ====================================================

        elif state == "SIGNING":

            # Keep collecting the latest 30 frames
            if is_moving:

                still_count = 0

            else:

                still_count += 1

            # ------------------------------------------------
            # Sign has ended
            # ------------------------------------------------

            if still_count >= SIGN_END_FRAMES:

                # Need at least 30 frames
                if len(frame_buffer) == SEQUENCE_LENGTH:

                    sequence = np.array(
                        frame_buffer,
                        dtype=np.float32
                    )

                    # Verify shape
                    if sequence.shape != (
                        SEQUENCE_LENGTH,
                        FEATURE_DIMENSION
                    ):

                        print(
                            "ERROR: Invalid sequence shape:",
                            sequence.shape
                        )

                    else:

                        # ------------------------------------
                        # SAVE SEQUENCE
                        # ------------------------------------

                        sequence_counter[current_label] += 1

                        count = sequence_counter[current_label]

                        filename = (
                            f"{current_label}_{count:03d}.npy"
                        )

                        filepath = os.path.join(
                            OUTPUT_DIR,
                            filename
                        )

                        np.save(filepath, sequence)

                        print()
                        print("-" * 60)
                        print("SEQUENCE SAVED")
                        print(f"Label : {current_label}")
                        print(f"File  : {filepath}")
                        print(f"Shape : {sequence.shape}")
                        print("-" * 60)

                        # ------------------------------------
                        # RESET
                        # ------------------------------------

                        state = "IDLE"

                        movement_count = 0
                        still_count = 0

                        frame_buffer.clear()

                        print(
                            "Capture complete. "
                            "Press Y or N for the next sequence."
                        )

                else:

                    print(
                        f"Not enough frames: "
                        f"{len(frame_buffer)}/{SEQUENCE_LENGTH}"
                    )

                    state = "IDLE"

                    movement_count = 0
                    still_count = 0

        # ----------------------------------------------------
        # DISPLAY
        # ----------------------------------------------------

        # Current label
        if current_label is None:
            label_text = "NO LABEL SELECTED"
        else:
            label_text = f"LABEL: {current_label}"

        # State
        state_text = f"STATE: {state}"

        # Motion
        motion_text = f"MOTION: {motion_value:.4f}"

        # Buffer
        buffer_text = (
            f"BUFFER: {len(frame_buffer)}/{SEQUENCE_LENGTH}"
        )

        # Draw information
        cv2.putText(
            frame,
            label_text,
            (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2
        )

        cv2.putText(
            frame,
            state_text,
            (20, 70),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 255),
            2
        )

        cv2.putText(
            frame,
            motion_text,
            (20, 105),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2
        )

        cv2.putText(
            frame,
            buffer_text,
            (20, 140),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2
        )

        cv2.putText(
            frame,
            "Y=YES  N=NO  Q=QUIT",
            (20, frame.shape[0] - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2
        )

        cv2.imshow(
            "Labeled Live Capture",
            frame
        )


finally:

    cap.release()

    cv2.destroyAllWindows()

    holistic.close()

    print()
    print("Capture program closed.")