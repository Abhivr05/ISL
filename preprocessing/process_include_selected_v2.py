import cv2
import numpy as np
from pathlib import Path

from utils.mediapipe_utils import initialize_holistic
from utils.landmark_utils import extract_landmarks
from preprocessing.normalization import normalize_landmarks


# ============================================================
# CONFIGURATION
# ============================================================

INCLUDE_ROOT = Path(r"E:\dataset for abijith")

OUTPUT_ROOT = Path("include_selected_dataset_v2")

SEQUENCE_LENGTH = 30
FEATURE_DIMENSION = 225

MOTION_THRESHOLD = 0.015

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
# FIND CLASS FOLDER
# ============================================================

def find_class_folder(class_name):

    target = class_name.lower()

    for folder in INCLUDE_ROOT.rglob("*"):

        if not folder.is_dir():
            continue

        name = folder.name.strip()

        if "." in name:
            name = name.split(".", 1)[1].strip()

        if name.lower() == target:
            return folder

    return None


# ============================================================
# EXTRACT VIDEO FEATURES
# ============================================================

def extract_video_features(video_path, holistic):

    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        print(f"    ERROR opening {video_path}")
        return []

    features = []

    while True:

        ret, frame = cap.read()

        if not ret:
            break

        frame_rgb = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        results = holistic.process(frame_rgb)

        landmarks = extract_landmarks(results)

        if landmarks is None:
            continue

        if len(landmarks) != FEATURE_DIMENSION:
            continue

        normalized = normalize_landmarks(landmarks)

        normalized = np.asarray(
            normalized,
            dtype=np.float32
        )

        if normalized.shape != (FEATURE_DIMENSION,):
            continue

        features.append(normalized)

    cap.release()

    return features


# ============================================================
# FIND BEST WINDOW
# ============================================================

def find_best_windows(features):

    if len(features) < SEQUENCE_LENGTH:
        return []

    features = np.asarray(
        features,
        dtype=np.float32
    )

    movement = np.mean(
        np.abs(features[1:] - features[:-1]),
        axis=1
    )

    # --------------------------------------------------------
    # We want four reasonably separated windows.
    # --------------------------------------------------------

    possible_windows = []

    for start in range(
        0,
        len(features) - SEQUENCE_LENGTH + 1
    ):

        end = start + SEQUENCE_LENGTH

        window_movement = movement[
            start:end - 1
        ]

        active = window_movement[
            window_movement > MOTION_THRESHOLD
        ]

        if len(active) > 0:
            score = np.sum(active)
        else:
            score = np.sum(window_movement)

        possible_windows.append(
            (score, start)
        )

    # Highest movement first
    possible_windows.sort(
        reverse=True,
        key=lambda x: x[0]
    )

    selected = []

    # Don't allow heavily overlapping windows.
    MIN_DISTANCE = SEQUENCE_LENGTH

    for score, start in possible_windows:

        if all(
            abs(start - existing_start)
            >= MIN_DISTANCE
            for existing_start in selected
        ):

            selected.append(start)

        if len(selected) == 4:
            break

    # If the video is too short for 4 separate windows,
    # use the best available windows.
    if len(selected) == 0:
        selected = [0]

    selected.sort()

    sequences = []

    for start in selected:

        sequence = features[
            start:start + SEQUENCE_LENGTH
        ]

        if sequence.shape == (
            SEQUENCE_LENGTH,
            FEATURE_DIMENSION
        ):
            sequences.append(sequence)

    return sequences


# ============================================================
# PROCESS CLASS
# ============================================================

def process_class(class_name, holistic):

    class_folder = find_class_folder(class_name)

    if class_folder is None:

        print(
            f"\n{class_name}: FOLDER NOT FOUND"
        )

        return 0

    videos = sorted(
        [
            p for p in class_folder.iterdir()
            if p.is_file()
            and p.suffix.lower() in
            [".mp4", ".mov"]
        ]
    )

    output_folder = (
        OUTPUT_ROOT / class_name
    )

    output_folder.mkdir(
        parents=True,
        exist_ok=True
    )

    print(
        f"\n{class_name}: "
        f"{len(videos)} videos"
    )

    total_saved = 0

    for video_index, video_path in enumerate(
        videos,
        start=1
    ):

        print(
            f"  [{video_index}/{len(videos)}] "
            f"{video_path.name}"
        )

        features = extract_video_features(
            video_path,
            holistic
        )

        if len(features) < SEQUENCE_LENGTH:

            print(
                f"    SKIPPED: "
                f"{len(features)} frames"
            )

            continue

        sequences = find_best_windows(
            features
        )

        video_id = video_path.stem

        for sequence_index, sequence in enumerate(
            sequences,
            start=1
        ):

            filename = (
                f"{class_name.lower()}_"
                f"{video_id}_"
                f"{sequence_index:02d}.npy"
            )

            output_path = (
                output_folder / filename
            )

            np.save(
                output_path,
                sequence
            )

            total_saved += 1

        print(
            f"    saved {len(sequences)} sequences"
        )

    return total_saved


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 60)
    print("INCLUDE DATASET PROCESSING - VIDEO ID VERSION")
    print("=" * 60)

    print(f"\nSource:")
    print(INCLUDE_ROOT)

    print("\nOutput:")
    print(OUTPUT_ROOT.resolve())

    print(
        f"\nClasses: "
        f"{len(SELECTED_CLASSES)}"
    )

    OUTPUT_ROOT.mkdir(
        parents=True,
        exist_ok=True
    )

    holistic = initialize_holistic()

    total = 0

    try:

        for class_name in SELECTED_CLASSES:

            total += process_class(
                class_name,
                holistic
            )

    finally:

        holistic.close()

    print("\n" + "=" * 60)
    print("PROCESSING COMPLETE")
    print("=" * 60)

    print(
        f"Total sequences: {total}"
    )

    print(
        "\nExample filename:"
    )

    print(
        "house_MVI_8750_01.npy"
    )

    print(
        "\nEach file should have shape:"
    )

    print("(30, 225)")


if __name__ == "__main__":
    main()