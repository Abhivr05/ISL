import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import cv2
import numpy as np

from utils.mediapipe_utils import initialize_holistic
from utils.landmark_utils import extract_landmarks
from preprocessing.normalization import normalize_landmarks


SOURCE_DIR = r"E:\dataset for abijith\Pronouns_1of2\Pronouns\41. you"
OUTPUT_DIR = "include_processed/YOU"

SEQUENCE_LENGTH = 30

os.makedirs(OUTPUT_DIR, exist_ok=True)


def process_video(video_path, output_path, holistic):
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"Could not open: {video_path}")
        return False

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if total_frames < SEQUENCE_LENGTH:
        print(f"Too short: {video_path} ({total_frames} frames)")
        cap.release()
        return False

    # Select 30 evenly spaced frames across the whole video
    frame_indices = np.linspace(
        0,
        total_frames - 1,
        SEQUENCE_LENGTH,
        dtype=int
    )

    sequence = []

    current_frame = 0
    target_index = 0

    while target_index < SEQUENCE_LENGTH:

        ret, frame = cap.read()

        if not ret:
            break

        if current_frame == frame_indices[target_index]:

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = holistic.process(rgb)

            landmarks = extract_landmarks(results)
            normalized = normalize_landmarks(landmarks)

            sequence.append(normalized)

            target_index += 1

        current_frame += 1

    cap.release()

    if len(sequence) != SEQUENCE_LENGTH:
        print(f"Invalid sequence: {video_path}")
        return False

    sequence = np.asarray(sequence, dtype=np.float32)

    if sequence.shape != (30, 225):
        print(f"Wrong shape: {sequence.shape}")
        return False

    np.save(output_path, sequence)

    return True


def main():

    video_files = [
        f for f in os.listdir(SOURCE_DIR)
        if f.lower().endswith((".mov", ".mp4", ".avi"))
    ]

    video_files.sort()

    print(f"Found {len(video_files)} videos.")

    holistic = initialize_holistic()

    processed = 0
    failed = 0

    try:
        for i, filename in enumerate(video_files, start=1):

            video_path = os.path.join(SOURCE_DIR, filename)

            output_name = f"you_{i:03d}.npy"
            output_path = os.path.join(OUTPUT_DIR, output_name)

            print(f"[{i}/{len(video_files)}] {filename}")

            success = process_video(
                video_path,
                output_path,
                holistic
            )

            if success:
                processed += 1
                print(f"  Saved: {output_path}")
            else:
                failed += 1

    finally:
        holistic.close()

    print("\n" + "=" * 40)
    print("Processing complete")
    print(f"Processed: {processed}")
    print(f"Failed:    {failed}")
    print("=" * 40)


if __name__ == "__main__":
    main()