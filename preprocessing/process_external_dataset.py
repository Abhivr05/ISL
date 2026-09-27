import cv2
import numpy as np
from pathlib import Path
import sys

# ---------------------------------------------------------
# Project path setup
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

sys.path.append(str(PROJECT_ROOT))

from utils.mediapipe_utils import initialize_holistic
from utils.landmark_utils import extract_landmarks
from preprocessing.normalization import normalize_landmarks


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

DATASET_DIR = PROJECT_ROOT / "external_dataset" / "isl_40words"
OUTPUT_DIR = PROJECT_ROOT / "processed_dataset"

SEQUENCE_LENGTH = 30
FEATURE_DIMENSION = 225


# Public dataset label -> project label
LABEL_MAPPING = {
    "hello": "HELLO",
    "help": "HELP",
    "me": "I",
    "no": "NO",
    "sorry": "SORRY",
    "thank_you": "THANK_YOU",
    "water": "WATER",
    "yes": "YES",
    "you": "YOU",
}


# ---------------------------------------------------------
# Extract one 30-frame sequence from a video
# ---------------------------------------------------------

def process_video(video_path):
    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        print(f"ERROR: Could not open {video_path}")
        return None

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if total_frames < SEQUENCE_LENGTH:
        cap.release()
        print(f"SKIP: Not enough frames - {video_path}")
        return None

    # 30 evenly distributed frame indices
    frame_indices = np.linspace(
        0,
        total_frames - 1,
        SEQUENCE_LENGTH
    ).astype(int)

    sequence = []

    holistic = initialize_holistic()

    current_frame = 0
    target_index = 0

    while target_index < SEQUENCE_LENGTH:

        ret, frame = cap.read()

        if not ret:
            break

        if current_frame == frame_indices[target_index]:

            # Convert BGR -> RGB for MediaPipe
            rgb_frame = cv2.cvtColor(
                frame,
                cv2.COLOR_BGR2RGB
            )

            results = holistic.process(rgb_frame)

            # Extract exactly 225 features
            landmarks = extract_landmarks(results)

            # Normalize using the same preprocessing
            normalized = normalize_landmarks(landmarks)

            if normalized.shape != (FEATURE_DIMENSION,):
                print(
                    f"ERROR: Unexpected feature shape "
                    f"{normalized.shape} in {video_path}"
                )

                holistic.close()
                cap.release()
                return None

            sequence.append(normalized)

            target_index += 1

        current_frame += 1

    holistic.close()
    cap.release()

    # Make sure all 30 frames were successfully processed
    if len(sequence) != SEQUENCE_LENGTH:
        print(
            f"SKIP: Could not extract 30 frames - {video_path}"
        )
        return None

    sequence = np.array(sequence, dtype=np.float32)

    if sequence.shape != (SEQUENCE_LENGTH, FEATURE_DIMENSION):
        print(
            f"ERROR: Final shape is {sequence.shape}, "
            f"expected {(SEQUENCE_LENGTH, FEATURE_DIMENSION)}"
        )
        return None

    return sequence


# ---------------------------------------------------------
# Main processing
# ---------------------------------------------------------

def main():

    if not DATASET_DIR.exists():
        print(f"Dataset directory not found:")
        print(DATASET_DIR)
        return

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    video_files = sorted(DATASET_DIR.rglob("*.mp4"))

    print(f"Found {len(video_files)} videos.")
    print()

    processed_count = 0
    skipped_count = 0

    for video_path in video_files:

        public_label = video_path.parent.name.lower()

        if public_label not in LABEL_MAPPING:
            print(f"SKIP: Unknown label - {public_label}")
            skipped_count += 1
            continue

        project_label = LABEL_MAPPING[public_label]

        print(
            f"Processing: {public_label:10} -> "
            f"{project_label:10} | {video_path.name}"
        )

        sequence = process_video(video_path)

        if sequence is None:
            skipped_count += 1
            continue

        # Create class directory
        class_dir = OUTPUT_DIR / project_label
        class_dir.mkdir(parents=True, exist_ok=True)

        # Give each sample a unique filename
        existing_files = list(class_dir.glob("*.npy"))
        sample_number = len(existing_files)

        output_path = class_dir / f"external_{sample_number:03d}.npy"

        np.save(output_path, sequence)

        print(
            f"  Saved: {output_path} "
            f"shape={sequence.shape}"
        )

        processed_count += 1

    print()
    print("=" * 60)
    print("PROCESSING COMPLETE")
    print("=" * 60)
    print(f"Videos found:     {len(video_files)}")
    print(f"Processed:        {processed_count}")
    print(f"Skipped:          {skipped_count}")
    print(f"Output directory: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()