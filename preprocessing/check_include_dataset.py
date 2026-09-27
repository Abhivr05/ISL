from pathlib import Path
import numpy as np


DATASET_ROOT = Path("include_selected_dataset")

EXPECTED_SHAPE = (30, 225)


def main():

    print("=" * 60)
    print("INCLUDE DATASET QUALITY CHECK")
    print("=" * 60)

    total = 0
    valid = 0
    invalid = 0

    class_counts = {}

    for class_dir in sorted(DATASET_ROOT.iterdir()):

        if not class_dir.is_dir():
            continue

        files = sorted(class_dir.glob("*.npy"))

        class_counts[class_dir.name] = len(files)

        for file_path in files:

            total += 1

            try:
                data = np.load(file_path)

                if data.shape != EXPECTED_SHAPE:
                    print(
                        f"BAD SHAPE: {file_path} -> {data.shape}"
                    )
                    invalid += 1
                    continue

                if not np.isfinite(data).all():
                    print(
                        f"BAD VALUES: {file_path}"
                    )
                    invalid += 1
                    continue

                valid += 1

            except Exception as e:
                print(
                    f"ERROR: {file_path} -> {e}"
                )
                invalid += 1

    print("\nClass counts:")
    print("-" * 40)

    for class_name, count in class_counts.items():
        print(f"{class_name:15s} : {count}")

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    print(f"Total files       : {total}")
    print(f"Valid files       : {valid}")
    print(f"Invalid files     : {invalid}")

    if total > 0:
        print(
            f"Success rate      : "
            f"{valid / total * 100:.2f}%"
        )


if __name__ == "__main__":
    main()