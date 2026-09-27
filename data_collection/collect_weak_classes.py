import cv2
import os
import numpy as np
from collections import deque

from utils.mediapipe_utils import initialize_holistic
from utils.landmark_utils import extract_landmarks
from preprocessing.normalization import normalize_landmarks


# Classes we want to improve
CLASSES = ["HELP", "WATER", "YOU"]

SEQUENCE_LENGTH = 30
SAMPLES_PER_CLASS = 40

OUTPUT_DIR = "raw_dataset_weak"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def collect_class(class_name):
    class_dir = os.path.join(OUTPUT_DIR, class_name)
    os.makedirs(class_dir, exist_ok=True)

    existing = len([
        f for f in os.listdir(class_dir)
        if f.endswith(".npy")
    ])

    print("\n" + "=" * 50)
    print(f"Collecting: {class_name}")
    print(f"Existing samples: {existing}")
    print(f"Target samples: {SAMPLES_PER_CLASS}")
    print("Press SPACE to start recording a sample.")
    print("Press Q to quit.")
    print("=" * 50)

    cap = cv2.VideoCapture(0)
    holistic = initialize_holistic()

    sample_count = existing

    while sample_count < SAMPLES_PER_CLASS:

        frames = []

        while True:
            ret, frame = cap.read()

            if not ret:
                continue

            frame = cv2.flip(frame, 1)

            display = frame.copy()

            cv2.putText(
                display,
                f"{class_name}  |  Sample {sample_count + 1}/{SAMPLES_PER_CLASS}",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2
            )

            cv2.putText(
                display,
                "Press SPACE to record",
                (20, 80),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2
            )

            cv2.imshow("ISL Data Collection", display)

            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                holistic.close()
                cap.release()
                cv2.destroyAllWindows()
                return False

            if key == 32:  # SPACE
                break

        print(f"\nRecording {class_name} sample {sample_count + 1}...")

        while len(frames) < SEQUENCE_LENGTH:

            ret, frame = cap.read()

            if not ret:
                continue

            frame = cv2.flip(frame, 1)

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = holistic.process(rgb)

            landmarks = extract_landmarks(results)
            normalized = normalize_landmarks(landmarks)

            frames.append(normalized)

            display = frame.copy()

            cv2.putText(
                display,
                f"Recording {class_name}: {len(frames)}/{SEQUENCE_LENGTH}",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2
            )

            cv2.imshow("ISL Data Collection", display)

            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                holistic.close()
                cap.release()
                cv2.destroyAllWindows()
                return False

        sequence = np.array(frames, dtype=np.float32)

        filename = os.path.join(
            class_dir,
            f"{class_name.lower()}_{sample_count + 1:03d}.npy"
        )

        np.save(filename, sequence)

        sample_count += 1

        print(f"Saved: {filename}")

        # Small pause between samples
        cv2.waitKey(500)

    holistic.close()
    cap.release()
    cv2.destroyAllWindows()

    return True


for class_name in CLASSES:

    completed = collect_class(class_name)

    if not completed:
        print("\nCollection stopped.")
        break

print("\nData collection finished.")