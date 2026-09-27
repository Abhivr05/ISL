import os
import glob
import numpy as np


SEQUENCE_DIR = "debug_8_signs"


def frame_motion(sequence):
    """
    Mean absolute feature change between consecutive frames.
    """
    return np.mean(
        np.abs(sequence[1:] - sequence[:-1]),
        axis=1
    )


def analyze_sequence(file_path):
    sequence = np.load(file_path)

    print("\n" + "=" * 70)
    print(os.path.basename(file_path))
    print("=" * 70)

    print("Shape:", sequence.shape)

    # Basic statistics
    print("Min:", np.min(sequence))
    print("Max:", np.max(sequence))
    print("Mean:", np.mean(sequence))
    print("Std:", np.std(sequence))

    # How many values are zero?
    zero_ratio = np.mean(sequence == 0) * 100
    print(f"Zero values: {zero_ratio:.2f}%")

    # Motion between frames
    motion = frame_motion(sequence)

    print("\nMotion statistics:")
    print("  Mean:", np.mean(motion))
    print("  Max :", np.max(motion))
    print("  Min :", np.min(motion))

    # Split sequence into 3 parts
    first = motion[:10]
    middle = motion[10:20]
    last = motion[20:]

    print("\nMotion distribution:")
    print(f"  First 10 frames : mean={np.mean(first):.5f}, active={np.sum(first > 0.015)}/10")
    print(f"  Middle 10 frames: mean={np.mean(middle):.5f}, active={np.sum(middle > 0.015)}/10")
    print(f"  Last 9 frames   : mean={np.mean(last):.5f}, active={np.sum(last > 0.015)}/9")

    # Highest-motion frames
    top_indices = np.argsort(motion)[-5:][::-1]

    print("\nHighest-motion frames:")
    for idx in top_indices:
        print(
            f"  Frame {idx:02d} -> motion={motion[idx]:.5f}"
        )

    # Split features:
    # 0:99   = pose
    # 99:162 = left hand
    # 162:225 = right hand
    pose = sequence[:, :99]
    left_hand = sequence[:, 99:162]
    right_hand = sequence[:, 162:225]

    print("\nFeature activity:")
    print(f"  Pose mean abs       : {np.mean(np.abs(pose)):.5f}")
    print(f"  Left hand mean abs  : {np.mean(np.abs(left_hand)):.5f}")
    print(f"  Right hand mean abs : {np.mean(np.abs(right_hand)):.5f}")


# --------------------------------------------------
# Find sequences
# --------------------------------------------------

files = sorted(
    glob.glob(
        os.path.join(
            SEQUENCE_DIR,
            "sequence_*.npy"
        )
    )
)

if not files:
    print("No sequence files found.")
    print("Expected folder:", os.path.abspath(SEQUENCE_DIR))
    exit()


print("Found", len(files), "sequences.")

for file_path in files:
    analyze_sequence(file_path)


print("\n" + "=" * 70)
print("ANALYSIS COMPLETE")
print("=" * 70)