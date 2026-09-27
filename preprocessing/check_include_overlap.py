from pathlib import Path
import numpy as np


DATASET_ROOT = Path("include_selected_dataset")

SEQUENCES_PER_VIDEO = 4


def sequence_distance(a, b):
    """
    Mean absolute difference between two sequences.
    Lower = more similar.
    """
    return float(np.mean(np.abs(a - b)))


def main():

    print("=" * 60)
    print("INCLUDE SEQUENCE OVERLAP CHECK")
    print("=" * 60)

    all_distances = []

    for class_dir in sorted(DATASET_ROOT.iterdir()):

        if not class_dir.is_dir():
            continue

        files = sorted(class_dir.glob("*.npy"))

        print(f"\n{class_dir.name}")

        # The processor creates 4 sequences per source video.
        # Group consecutive files into groups of 4.
        for start in range(0, len(files), SEQUENCES_PER_VIDEO):

            group = files[start:start + SEQUENCES_PER_VIDEO]

            if len(group) < SEQUENCES_PER_VIDEO:
                continue

            sequences = [
                np.load(file).astype(np.float32)
                for file in group
            ]

            distances = []

            for i in range(len(sequences)):
                for j in range(i + 1, len(sequences)):
                    distance = sequence_distance(
                        sequences[i],
                        sequences[j]
                    )
                    distances.append(distance)
                    all_distances.append(distance)

            average_distance = np.mean(distances)
            minimum_distance = np.min(distances)

            print(
                f"  {group[0].stem} - {group[-1].stem} | "
                f"avg distance: {average_distance:.6f} | "
                f"min: {minimum_distance:.6f}"
            )

    print("\n" + "=" * 60)
    print("OVERALL")
    print("=" * 60)

    if all_distances:

        print(
            f"Number of sequence pairs : "
            f"{len(all_distances)}"
        )

        print(
            f"Average distance         : "
            f"{np.mean(all_distances):.6f}"
        )

        print(
            f"Minimum distance         : "
            f"{np.min(all_distances):.6f}"
        )

        print(
            f"Maximum distance         : "
            f"{np.max(all_distances):.6f}"
        )


if __name__ == "__main__":
    main()