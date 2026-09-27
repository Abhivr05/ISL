import numpy as np
from pathlib import Path

DATASET_DIR = Path("processed_dataset")

EXPECTED_SHAPE = (30, 225)

total = 0
valid = 0
invalid = 0

print("Checking processed dataset...\n")

for class_dir in sorted(DATASET_DIR.iterdir()):

    if not class_dir.is_dir():
        continue

    class_total = 0
    class_valid = 0

    for file_path in sorted(class_dir.glob("*.npy")):

        total += 1
        class_total += 1

        try:
            data = np.load(file_path)

            if data.shape == EXPECTED_SHAPE:
                valid += 1
                class_valid += 1
            else:
                invalid += 1
                print(
                    f"INVALID: {file_path} "
                    f"shape={data.shape}"
                )

        except Exception as e:
            invalid += 1
            print(f"ERROR: {file_path} -> {e}")

    print(
        f"{class_dir.name:12} | "
        f"{class_valid}/{class_total} valid"
    )

print("\n" + "=" * 50)
print("CHECK COMPLETE")
print("=" * 50)
print(f"Total files:  {total}")
print(f"Valid files:  {valid}")
print(f"Invalid:      {invalid}")