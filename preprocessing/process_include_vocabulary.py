import os
import sys

# Allow imports from project root
sys.path.append(
    os.path.dirname(
        os.path.dirname(
            os.path.abspath(__file__)
        )
    )
)

import cv2
import numpy as np

from utils.mediapipe_utils import initialize_holistic
from utils.landmark_utils import extract_landmarks
from preprocessing.normalization import normalize_landmarks


# ============================================================
# INCLUDE DATASET SOURCE DIRECTORIES
# ============================================================

SOURCE_DIRS = {
    "HAPPY": r"E:\dataset for abijith\Adjectives_1of8\Adjectives\3. happy",

    "SICK": r"E:\dataset for abijith\Adjectives_8of8\Adjectives\98. sick",

    "HEALTHY": r"E:\dataset for abijith\Adjectives_8of8\Adjectives\99. healthy",

    "FRIEND": r"E:\dataset for abijith\People_5of5\People\81. Friend",
}


# ============================================================
# CONFIGURATION
# ============================================================

SEQUENCE_LENGTH = 30

OUTPUT_ROOT = "include_processed"


# ============================================================
# PROCESS ONE VIDEO
# ============================================================

def process_video(video_path, output_path, holistic):

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"Could not open: {video_path}")
        return False

    total_frames = int(
        cap.get(cv2.CAP_PROP_FRAME_COUNT)
    )

    # Make sure video contains enough frames
    if total_frames < SEQUENCE_LENGTH:

        print(
            f"Too short: {video_path} "
            f"({total_frames} frames)"
        )

        cap.release()

        return False

    # --------------------------------------------------------
    # Select 30 evenly spaced frames
    # --------------------------------------------------------

    frame_indices = np.linspace(
        0,
        total_frames - 1,
        SEQUENCE_LENGTH,
        dtype=int
    )

    sequence = []

    current_frame = 0
    target_index = 0

    # --------------------------------------------------------
    # Read video
    # --------------------------------------------------------

    while target_index < SEQUENCE_LENGTH:

        ret, frame = cap.read()

        if not ret:
            break

        if current_frame == frame_indices[target_index]:

            # Convert BGR → RGB
            rgb = cv2.cvtColor(
                frame,
                cv2.COLOR_BGR2RGB
            )

            # MediaPipe Holistic
            results = holistic.process(rgb)

            # Extract 225 landmarks
            landmarks = extract_landmarks(results)

            # Apply existing normalization
            normalized = normalize_landmarks(
                landmarks
            )

            sequence.append(normalized)

            target_index += 1

        current_frame += 1

    cap.release()

    # --------------------------------------------------------
    # Validate sequence
    # --------------------------------------------------------

    if len(sequence) != SEQUENCE_LENGTH:

        print(
            f"Invalid sequence: {video_path}"
        )

        return False

    # Convert to numpy array
    sequence = np.asarray(
        sequence,
        dtype=np.float32
    )

    # --------------------------------------------------------
    # Validate expected shape
    # --------------------------------------------------------

    if sequence.shape != (30, 225):

        print(
            f"Wrong shape: {sequence.shape}"
        )

        return False

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    np.save(
        output_path,
        sequence
    )

    return True


# ============================================================
# MAIN
# ============================================================

def main():

    total_processed = 0
    total_failed = 0

    print("=" * 60)
    print("INCLUDE VOCABULARY PROCESSING")
    print("=" * 60)

    # --------------------------------------------------------
    # Process each class
    # --------------------------------------------------------

    for class_name, source_dir in SOURCE_DIRS.items():

        # Output:
        # include_processed/HAPPY
        # include_processed/SICK
        # etc.

        output_dir = os.path.join(
            OUTPUT_ROOT,
            class_name
        )

        os.makedirs(
            output_dir,
            exist_ok=True
        )

        # ----------------------------------------------------
        # Check source directory
        # ----------------------------------------------------

        if not os.path.exists(source_dir):

            print(
                f"\nERROR: Source directory does not exist:"
            )

            print(source_dir)

            total_failed += 1

            continue

        # ----------------------------------------------------
        # Find videos
        # ----------------------------------------------------

        video_files = [
            f
            for f in os.listdir(source_dir)
            if f.lower().endswith(
                (".mov", ".mp4", ".avi")
            )
        ]

        video_files.sort()

        print("\n" + "=" * 60)

        print(
            f"Class: {class_name}"
        )

        print(
            f"Source: {source_dir}"
        )

        print(
            f"Videos found: {len(video_files)}"
        )

        print(
            f"Output: {output_dir}"
        )

        print("=" * 60)

        # ----------------------------------------------------
        # Initialize MediaPipe
        # ----------------------------------------------------

        holistic = initialize_holistic()

        processed = 0
        failed = 0

        try:

            # ------------------------------------------------
            # Process every video
            # ------------------------------------------------

            for i, filename in enumerate(
                video_files,
                start=1
            ):

                video_path = os.path.join(
                    source_dir,
                    filename
                )

                output_name = (
                    f"{class_name.lower()}_{i:03d}.npy"
                )

                output_path = os.path.join(
                    output_dir,
                    output_name
                )

                print(
                    f"[{i}/{len(video_files)}] "
                    f"{filename}"
                )

                success = process_video(
                    video_path,
                    output_path,
                    holistic
                )

                if success:

                    processed += 1
                    total_processed += 1

                    print(
                        f"  Saved: {output_path}"
                    )

                else:

                    failed += 1
                    total_failed += 1

        finally:

            holistic.close()

        print(
            f"\n{class_name}: "
            f"Processed={processed}, "
            f"Failed={failed}"
        )

    # ========================================================
    # FINAL SUMMARY
    # ========================================================

    print("\n" + "=" * 60)
    print("ALL PROCESSING COMPLETE")
    print("=" * 60)

    print(
        f"Total processed: {total_processed}"
    )

    print(
        f"Total failed:    {total_failed}"
    )

    print("=" * 60)


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()