
import os
import cv2
import numpy as np
from collections import deque

from utils.mediapipe_utils import initialize_holistic
from utils.landmark_utils import extract_landmarks
from preprocessing.normalization import normalize_landmarks


# ============================================================
# SETTINGS
# ============================================================

SAVE_DIR = "webcam_dataset"

SEQUENCE_LENGTH = 30
FEATURE_DIMENSION = 225

# Movement detection
MOTION_THRESHOLD = 0.015
MOTION_BURST_REQUIRED = 3
MOTION_HISTORY_SIZE = 7

# After detecting movement, collect this many frames
PRE_MOTION_FRAMES = 5

# Number of samples to collect before changing sign
TARGET_SAMPLES = 30

# Frames that must be quiet before another sign can start
IDLE_FRAMES_REQUIRED = 15


# ============================================================
# VOCABULARY
# ============================================================

CLASSES = [
    "GOOD",
    "HELLO",
    "HELP",
    "I",
    "NO",
    "SORRY",
    "THANK_YOU",
    "WATER",
    "YES",
    "YOU",

    "HOUSE",
    "SCHOOL",
    "OFFICE",
    "HOSPITAL",
    "MARKET",
    "BOOK",
    "BAG",
    "MONEY",
    "MEDICINE",
    "TODAY",
]


# ============================================================
# CREATE DIRECTORIES
# ============================================================

for class_name in CLASSES:
    os.makedirs(
        os.path.join(SAVE_DIR, class_name),
        exist_ok=True
    )


# ============================================================
# SELECT CLASS
# ============================================================

print("\n" + "=" * 60)
print("20-CLASS WEBCAM DATA COLLECTOR")
print("=" * 60)

for i, class_name in enumerate(CLASSES, start=1):
    existing = len([
        f for f in os.listdir(
            os.path.join(SAVE_DIR, class_name)
        )
        if f.endswith(".npy")
    ])

    print(
        f"{i:2d}. {class_name:12s} "
        f"(existing: {existing})"
    )

print("\nEnter the number of the sign you want to collect.")
print("You can press N later to choose another sign.")
print("Press Q in the camera window to quit.")


while True:

    choice = input("\nSelect class number: ").strip()

    try:
        class_index = int(choice) - 1

        if 0 <= class_index < len(CLASSES):
            break

    except ValueError:
        pass

    print("Invalid selection. Enter a number from 1 to 20.")


current_class = CLASSES[class_index]

print(f"\nSelected: {current_class}")


# ============================================================
# HELPER
# ============================================================

def get_sample_count(class_name):

    folder = os.path.join(
        SAVE_DIR,
        class_name
    )

    return len([
        f for f in os.listdir(folder)
        if f.endswith(".npy")
    ])


def save_sequence(class_name, sequence):

    folder = os.path.join(
        SAVE_DIR,
        class_name
    )

    existing = get_sample_count(class_name)

    filename = f"{class_name}_{existing + 1:03d}.npy"

    path = os.path.join(
        folder,
        filename
    )

    sequence = np.asarray(
        sequence,
        dtype=np.float32
    )

    np.save(path, sequence)

    return path


# ============================================================
# MEDIAPIPE
# ============================================================

print("\nStarting MediaPipe...")

holistic = initialize_holistic()


# ============================================================
# CAMERA
# ============================================================

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("ERROR: Could not open webcam.")
    holistic.close()
    exit()


# ============================================================
# STATE
# ============================================================

IDLE = 0
CAPTURING = 1
COOLDOWN = 2

state = IDLE

pre_motion_buffer = deque(
    maxlen=PRE_MOTION_FRAMES
)

motion_history = deque(
    maxlen=MOTION_HISTORY_SIZE
)

capture_sequence = []

previous_features = None

motion_bursts = 0

idle_frames = 0

sample_count = get_sample_count(current_class)


print("\n" + "=" * 60)
print(f"READY FOR: {current_class}")
print(f"Existing samples: {sample_count}")
print(f"Target: {TARGET_SAMPLES}")
print("=" * 60)

print("\nInstructions:")
print("1. Wait until READY appears.")
print("2. Make the sign naturally.")
print("3. Hold/complete the sign normally.")
print("4. The sequence will be saved automatically.")
print("5. Repeat.")
print("6. Press N to change sign.")
print("7. Press Q to quit.\n")


# ============================================================
# MAIN LOOP
# ============================================================

while True:

    ret, frame = cap.read()

    if not ret:
        print("Could not read frame.")
        break

    # --------------------------------------------------------
    # Mirror camera display
    # --------------------------------------------------------

    frame = cv2.flip(frame, 1)

    # --------------------------------------------------------
    # RGB
    # --------------------------------------------------------

    rgb_frame = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )

    # --------------------------------------------------------
    # MediaPipe
    # --------------------------------------------------------

    results = holistic.process(
        rgb_frame
    )

    if results is None:
        cv2.imshow(
            "20-Class Data Collector",
            frame
        )

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break

        continue

    # --------------------------------------------------------
    # Extract 225 features
    # --------------------------------------------------------

    landmarks = extract_landmarks(
        results
    )

    if landmarks is None:
        cv2.imshow(
            "20-Class Data Collector",
            frame
        )

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
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

    if features.shape != (
        FEATURE_DIMENSION,
    ):
        continue

    # ========================================================
    # MOTION
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

    is_moving = (
        motion > MOTION_THRESHOLD
    )

    # Keep recent frames
    pre_motion_buffer.append(
        features.copy()
    )

    # ========================================================
    # IDLE
    # ========================================================

    if state == IDLE:

        if is_moving:

            motion_bursts += 1
            idle_frames = 0

        else:

            idle_frames += 1

        # Reset burst count after enough quiet frames
        if idle_frames >= IDLE_FRAMES_REQUIRED:

            motion_bursts = 0

        # ----------------------------------------------------
        # Start capture
        # ----------------------------------------------------

        if motion_bursts >= MOTION_BURST_REQUIRED:

            print(
                f"\nMovement detected for "
                f"{current_class}!"
            )

            capture_sequence = list(
                pre_motion_buffer
            )

            capture_sequence = (
                capture_sequence[
                    -PRE_MOTION_FRAMES:
                ]
            )

            state = CAPTURING

            motion_bursts = 0
            idle_frames = 0

    # ========================================================
    # CAPTURING
    # ========================================================

    elif state == CAPTURING:

        capture_sequence.append(
            features.copy()
        )

        if len(capture_sequence) >= SEQUENCE_LENGTH:

            capture_sequence = (
                capture_sequence[
                    :SEQUENCE_LENGTH
                ]
            )

            # ----------------------------------------------
            # Safety check
            # ----------------------------------------------

            sequence = np.asarray(
                capture_sequence,
                dtype=np.float32
            )

            if sequence.shape == (
                SEQUENCE_LENGTH,
                FEATURE_DIMENSION
            ):

                path = save_sequence(
                    current_class,
                    sequence
                )

                sample_count = (
                    get_sample_count(
                        current_class
                    )
                )

                print(
                    f"Saved: {path}"
                )

                print(
                    f"{current_class}: "
                    f"{sample_count}/{TARGET_SAMPLES}"
                )

            # ----------------------------------------------
            # Reset
            # ----------------------------------------------

            capture_sequence = []

            pre_motion_buffer.clear()

            motion_history.clear()

            motion_bursts = 0

            idle_frames = 0

            state = COOLDOWN

    # ========================================================
    # COOLDOWN
    # ========================================================

    elif state == COOLDOWN:

        # Wait until the person stops moving before
        # allowing another capture.

        if is_moving:

            idle_frames = 0

        else:

            idle_frames += 1

        if idle_frames >= IDLE_FRAMES_REQUIRED:

            state = IDLE

            motion_bursts = 0

            pre_motion_buffer.clear()

            print(
                "\nREADY - Make the next sign."
            )

    # ========================================================
    # DISPLAY
    # ========================================================

    if state == IDLE:

        status = "READY"

    elif state == CAPTURING:

        status = (
            f"CAPTURING "
            f"{len(capture_sequence)}/{SEQUENCE_LENGTH}"
        )

    else:

        status = "WAITING"

    cv2.putText(
        frame,
        f"Sign: {current_class}",
        (20, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (0, 255, 0),
        2
    )

    cv2.putText(
        frame,
        f"Samples: {sample_count}/{TARGET_SAMPLES}",
        (20, 70),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2
    )

    cv2.putText(
        frame,
        status,
        (20, 105),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (0, 255, 255),
        2
    )

    cv2.putText(
        frame,
        f"Motion: {motion:.4f}",
        (20, 140),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        2
    )

    cv2.imshow(
        "20-Class Data Collector",
        frame
    )

    # ========================================================
    # KEYBOARD
    # ========================================================

    key = cv2.waitKey(1) & 0xFF

    # Quit
    if key == ord("q"):
        break

    # Change sign
    if key == ord("n"):

        print("\nStopping current class...")

        print(
            f"Collected {sample_count} "
            f"samples for {current_class}."
        )

        print("\nAvailable classes:")

        for i, class_name in enumerate(
            CLASSES,
            start=1
        ):
            count = get_sample_count(
                class_name
            )

            print(
                f"{i:2d}. {class_name:12s} "
                f"({count} samples)"
            )

        while True:

            choice = input(
                "\nSelect next class number: "
            ).strip()

            try:

                new_index = int(choice) - 1

                if 0 <= new_index < len(CLASSES):
                    break

            except ValueError:
                pass

            print(
                "Invalid selection."
            )

        current_class = CLASSES[
            new_index
        ]

        sample_count = get_sample_count(
            current_class
        )

        capture_sequence = []

        pre_motion_buffer.clear()

        motion_history.clear()

        motion_bursts = 0

        idle_frames = 0

        previous_features = None

        state = IDLE

        print("\n" + "=" * 60)
        print(
            f"READY FOR: {current_class}"
        )
        print(
            f"Existing samples: {sample_count}"
        )
        print(
            f"Target: {TARGET_SAMPLES}"
        )
        print("=" * 60)


# ============================================================
# CLEANUP
# ============================================================

cap.release()

holistic.close()

cv2.destroyAllWindows()

print("\nData collection stopped.")
print(f"Dataset location: {SAVE_DIR}")


