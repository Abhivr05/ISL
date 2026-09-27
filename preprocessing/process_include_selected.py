import os
import cv2
import numpy as np
from pathlib import Path
from collections import deque

from utils.mediapipe_utils import initialize_holistic
from utils.landmark_utils import extract_landmarks
from preprocessing.normalization import normalize_landmarks


# ============================================================
# CONFIGURATION
# ============================================================

INCLUDE_ROOT = Path(r"E:\dataset for abijith")

OUTPUT_ROOT = Path("include_selected_dataset")

SEQUENCE_LENGTH = 30
FEATURE_DIMENSION = 225

# Movement threshold used to locate the signing portion
MOTION_THRESHOLD = 0.015

# Classes selected for the first INCLUDE expansion
SELECTED_CLASSES = [
    "HOUSE",
    "SCHOOL",
    "OFFICE",
    "HOSPITAL",
    "MARKET",

    "BOOK",
    "TELEPHONE",
    "BAG",
    "KEY",

    "TODAY",
    "TOMORROW",
    "YESTERDAY",
    "MORNING",
    "EVENING",

    "MONEY",
    "MEDICINE",
    "DEATH",
    "PEACE",
    "SPORT",
    "TECHNOLOGY",
]


# ============================================================
# FIND INCLUDE CLASS FOLDER
# ============================================================

def find_class_folder(class_name):
    """
    Search the INCLUDE directory recursively for:

        <number>. <class_name>

    Example:
        24. Window
        28. Store/Shop
    """

    target = class_name.lower()

    for folder in INCLUDE_ROOT.rglob("*"):
        if not folder.is_dir():
            continue

        folder_name = folder.name.strip()

        # Remove the numeric prefix, e.g.:
        # "24. Window" -> "Window"
        if "." in folder_name:
            name_without_number = folder_name.split(".", 1)[1].strip()
        else:
            name_without_number = folder_name

        if name_without_number.lower() == target:
            return folder

    return None


# ============================================================
# EXTRACT ONE VIDEO
# ============================================================

def extract_video_features(video_path, holistic):
    """
    Extract normalized 225-dimensional landmark features
    from every frame of a video.
    """

    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        print(f"    ERROR: Could not open {video_path}")
        return []

    features = []

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        results = holistic.process(frame_rgb)

        landmarks = extract_landmarks(results)

        if landmarks is None:
            continue

        # Make sure we have exactly 225 features
        if len(landmarks) != FEATURE_DIMENSION:
            print(
                f"    WARNING: Expected {FEATURE_DIMENSION} features, "
                f"got {len(landmarks)}"
            )
            continue

        normalized = normalize_landmarks(landmarks)

        normalized = np.asarray(normalized, dtype=np.float32)

        if normalized.shape != (FEATURE_DIMENSION,):
            print(
                f"    WARNING: Invalid shape {normalized.shape}"
            )
            continue

        features.append(normalized)

    cap.release()

    return features


# ============================================================
# FIND BEST 30-FRAME WINDOW
# ============================================================

def select_best_sequence(features):
    """
    Select the 30-frame window containing the most movement.

    This avoids taking a sequence consisting mostly of
    stationary/resting frames.
    """

    if len(features) < SEQUENCE_LENGTH:
        return None

    features = np.asarray(features, dtype=np.float32)

    # --------------------------------------------------------
    # Calculate frame-to-frame movement
    # --------------------------------------------------------

    movement = np.mean(
        np.abs(features[1:] - features[:-1]),
        axis=1
    )

    # --------------------------------------------------------
    # If video is longer than 30 frames, search all windows
    # --------------------------------------------------------

    best_start = 0
    best_score = -1

    for start in range(len(features) - SEQUENCE_LENGTH + 1):

        end = start + SEQUENCE_LENGTH

        # Movement values correspond to transitions between
        # frames, so use the transitions inside this window.
        window_movement = movement[start:end - 1]

        # Give higher score to windows containing movement.
        active_movement = window_movement[
            window_movement > MOTION_THRESHOLD
        ]

        if len(active_movement) > 0:
            score = np.sum(active_movement)
        else:
            score = np.sum(window_movement)

        if score > best_score:
            best_score = score
            best_start = start

    sequence = features[
        best_start:best_start + SEQUENCE_LENGTH
    ]

    if sequence.shape != (SEQUENCE_LENGTH, FEATURE_DIMENSION):
        return None

    return sequence


# ============================================================
# PROCESS ONE CLASS
# ============================================================

def process_class(class_name, holistic):

    class_folder = find_class_folder(class_name)

    if class_folder is None:
        print(f"\n{class_name}: FOLDER NOT FOUND")
        return 0, 0

    videos = sorted(
        list(class_folder.glob("*.MP4")) +
        list(class_folder.glob("*.mp4")) +
        list(class_folder.glob("*.MOV")) +
        list(class_folder.glob("*.mov"))
    )

    output_folder = OUTPUT_ROOT / class_name
    output_folder.mkdir(parents=True, exist_ok=True)

    print(
        f"\n{class_name}: "
        f"{len(videos)} videos"
    )

    valid = 0
    invalid = 0

    for index, video_path in enumerate(videos, start=1):

        print(
            f"  [{index}/{len(videos)}] "
            f"{video_path.name}"
        )

        try:
            features = extract_video_features(
                video_path,
                holistic
            )

            if len(features) < SEQUENCE_LENGTH:
                print(
                    f"    SKIPPED: only "
                    f"{len(features)} valid frames"
                )
                invalid += 1
                continue

            sequence = select_best_sequence(features)

            if sequence is None:
                print("    SKIPPED: could not create sequence")
                invalid += 1
                continue

            output_name = (
                f"{class_name.lower()}_{index:03d}.npy"
            )

            output_path = output_folder / output_name

            np.save(output_path, sequence)

            print(
                f"    saved {sequence.shape}"
            )

            valid += 1

        except Exception as e:
            print(f"    ERROR: {e}")
            invalid += 1

    return valid, invalid


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 60)
    print("INCLUDE SELECTED DATASET PROCESSING")
    print("=" * 60)

    print(f"\nSource:")
    print(INCLUDE_ROOT)

    print(f"\nOutput:")
    print(OUTPUT_ROOT.resolve())

    print(f"\nSelected classes: {len(SELECTED_CLASSES)}")

    # Create output directory
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    total_valid = 0
    total_invalid = 0

    # --------------------------------------------------------
    # MediaPipe Holistic
    # --------------------------------------------------------

    holistic = initialize_holistic()

    try:

        for class_name in SELECTED_CLASSES:

            valid, invalid = process_class(
                class_name,
                holistic
            )

            total_valid += valid
            total_invalid += invalid

    finally:

        holistic.close()

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    print("\n" + "=" * 60)
    print("PROCESSING COMPLETE")
    print("=" * 60)

    print(f"Valid sequences   : {total_valid}")
    print(f"Invalid/skipped   : {total_invalid}")
    print(f"Total             : {total_valid + total_invalid}")

    print("\nOutput:")
    print(OUTPUT_ROOT.resolve())

    print("\nEach valid file should have shape:")
    print("(30, 225)")


if __name__ == "__main__":
    main()