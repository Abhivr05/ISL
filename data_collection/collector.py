import json
import os
import re

import cv2
import numpy as np

from utils.mediapipe_utils import initialize_holistic
from utils.landmark_utils import extract_landmarks
from preprocessing.normalization import normalize_landmarks

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

VOCABULARY_PATH = os.path.join(PROJECT_ROOT, "config", "vocabulary.json")
DATASET_PATH = os.path.join(PROJECT_ROOT, "raw_dataset")

SEQUENCE_LENGTH = 30
FEATURE_DIMENSION = 225


def load_vocabulary():
    """Load gesture vocabulary."""
    with open(VOCABULARY_PATH, "r", encoding="utf-8") as file:
        return json.load(file)


def get_next_sample_number(gesture_directory):
    """Return the next unused sample number without overwriting existing files."""
    if not os.path.exists(gesture_directory):
        return 1

    max_number = 0

    for file_name in os.listdir(gesture_directory):
        if not file_name.endswith(".npy"):
            continue

        match = re.fullmatch(r"sample_(\d+)\.npy", file_name)

        if match:
            max_number = max(max_number, int(match.group(1)))

    return max_number + 1


def save_sequence(gesture_name, sequence):
    """Validate and save a gesture sequence."""
    if sequence.shape != (SEQUENCE_LENGTH, FEATURE_DIMENSION):
        raise ValueError(
            f"Invalid sequence shape: {sequence.shape}. "
            f"Expected: ({SEQUENCE_LENGTH}, {FEATURE_DIMENSION})"
        )

    gesture_directory = os.path.join(DATASET_PATH, gesture_name)
    os.makedirs(gesture_directory, exist_ok=True)

    sample_number = get_next_sample_number(gesture_directory)
    file_name = f"sample_{sample_number:03d}.npy"
    file_path = os.path.join(gesture_directory, file_name)

    np.save(file_path, sequence)

    return file_path


def collect_gesture(gesture_name, holistic):
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("ERROR: Could not open webcam.")
        return

    sequence = []
    recording = False

    gesture_directory = os.path.join(DATASET_PATH, gesture_name)

    existing_count = len([
        file for file in os.listdir(gesture_directory)
        if file.endswith(".npy")
    ]) if os.path.exists(gesture_directory) else 0

    print()
    print("=" * 50)
    print(f"GESTURE: {gesture_name}")
    print("=" * 50)
    print(f"Existing samples: {existing_count}")
    print(f"Next sample: {get_next_sample_number(gesture_directory)}")
    print("Press SPACE inside the webcam window to record.")
    print("Press Q inside the webcam window to quit.")
    print("=" * 50)

    while cap.isOpened():
        success, frame = cap.read()

        if not success:
            continue

        frame = cv2.flip(frame, 1)
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = holistic.process(image)

        feature_vector = extract_landmarks(results)
        normalized_features = normalize_landmarks(feature_vector)

        if recording:
            sequence.append(normalized_features)
            frame_number = len(sequence)

            cv2.putText(
                frame,
                f"RECORDING {frame_number}/{SEQUENCE_LENGTH}",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2
            )

            if len(sequence) == SEQUENCE_LENGTH:
                sequence_array = np.array(sequence, dtype=np.float32)

                file_path = save_sequence(
                    gesture_name,
                    sequence_array
                )

                print()
                print("Sample saved successfully!")
                print("File:", file_path)
                print("Shape:", sequence_array.shape)

                recording = False
                sequence = []

        else:
            cv2.putText(
                frame,
                f"GESTURE: {gesture_name}",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2
            )
            cv2.putText(
                frame,
                "SPACE = Record",
                (20, 75),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2
            )
            cv2.putText(
                frame,
                "Q = Quit",
                (20, 110),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2
            )

        cv2.imshow("ISL Data Collection", frame)

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            print("Exiting...")
            break

        elif key == 32:
            if not recording:
                recording = True
                sequence = []

                print()
                print("Recording started...")
                print(f"Collecting {SEQUENCE_LENGTH} frames...")

    cap.release()
    cv2.destroyAllWindows()


def main():
    vocabulary = load_vocabulary()

    print()
    print("=" * 50)
    print("ISL DATA COLLECTION")
    print("=" * 50)

    print("\nAvailable gestures:\n")

    for class_id, gesture_name in vocabulary.items():
        print(f"{class_id}: {gesture_name}")

    print()
    class_id = input("Enter class ID: ").strip()

    if class_id not in vocabulary:
        print("Invalid class ID.")
        return

    gesture_name = vocabulary[class_id]

    holistic = initialize_holistic()

    try:
        collect_gesture(
            gesture_name,
            holistic
        )
    finally:
        holistic.close()


if __name__ == "__main__":
    main()
