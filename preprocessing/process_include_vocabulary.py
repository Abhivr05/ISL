import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import cv2
import numpy as np

from utils.mediapipe_utils import initialize_holistic
from utils.landmark_utils import extract_landmarks
from preprocessing.normalization import normalize_landmarks


SOURCE_DIR =SOURCE_DIRS = {
    "HOME": r"E:\dataset for abijith\Places_1of4\Places\19. House",
    "SCHOOL": r"E:\dataset for abijith\Places_2of4\Places\24. School",
    "HOSPITAL": r"E:\dataset for abijith\Places_3of4\Places\30. Hospital",
    "MONEY": r"E:\dataset for abijith\Society_1of3\Society\4. Money",
    "MEDICINE": r"E:\dataset for abijith\Society_1of3\Society\3. Medicine",
    "TODAY": r"E:\dataset for abijith\Days_and_Time_1of3\Days_and_Time\73. Today",
    "TOMORROW": r"E:\dataset for abijith\Days_and_Time_2of3\Days_and_Time\74. Tomorrow",
    "YESTERDAY": r"E:\dataset for abijith\Days_and_Time_2of3\Days_and_Time\75. Yesterday",
}
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

    total_processed = 0
    total_failed = 0

    for class_name, source_dir in SOURCE_DIRS.items():

        output_dir = os.path.join(
            "include_processed",
            class_name
        )

        os.makedirs(output_dir, exist_ok=True)

        video_files = [
            f for f in os.listdir(source_dir)
            if f.lower().endswith((".mov", ".mp4", ".avi"))
        ]

        video_files.sort()

        print("\n" + "=" * 50)
        print(f"Class: {class_name}")
        print(f"Source: {source_dir}")
        print(f"Videos found: {len(video_files)}")
        print("=" * 50)

        holistic = initialize_holistic()

        processed = 0
        failed = 0

        try:

            for i, filename in enumerate(video_files, start=1):

                video_path = os.path.join(
                    source_dir,
                    filename
                )

                output_name = f"{class_name.lower()}_{i:03d}.npy"

                output_path = os.path.join(
                    output_dir,
                    output_name
                )

                print(
                    f"[{i}/{len(video_files)}] {filename}"
                )

                success = process_video(
                    video_path,
                    output_path,
                    holistic
                )

                if success:
                    processed += 1
                    total_processed += 1
                    print(f"  Saved: {output_path}")
                else:
                    failed += 1
                    total_failed += 1

        finally:
            holistic.close()

        print(
            f"{class_name}: "
            f"Processed={processed}, Failed={failed}"
        )

    print("\n" + "=" * 50)
    print("ALL PROCESSING COMPLETE")
    print(f"Total processed: {total_processed}")
    print(f"Total failed:    {total_failed}")
    print("=" * 50)

if __name__ == "__main__":
    main()