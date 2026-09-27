import os
import numpy as np


DATASET_DIR = "raw_dataset"
INCLUDE_DIR = "include_processed"

CLASSES = [
    "GOOD",
    "HELLO",
    "HELP",
    "I",
    "NO",
    "SORRY",
    "THANK_YOU",
    "WATER",
    "YES",
    "YOU",
    "HOME",
    "SCHOOL",
    "HOSPITAL",
    "MONEY",
    "MEDICINE",
    "TODAY",
    "TOMORROW",
    "YESTERDAY",
]


def check_folder(folder_path, class_name):
    if not os.path.exists(folder_path):
        return 0, 0

    count = 0
    invalid = 0

    for filename in os.listdir(folder_path):

        if not filename.endswith(".npy"):
            continue

        filepath = os.path.join(folder_path, filename)

        try:
            data = np.load(filepath)

            if data.shape != (30, 225):
                invalid += 1
                print(
                    f"  INVALID: {class_name}/{filename} "
                    f"-> {data.shape}"
                )
            else:
                count += 1

        except Exception as e:
            invalid += 1
            print(
                f"  ERROR: {class_name}/{filename} -> {e}"
            )

    return count, invalid


print("=" * 60)
print("19-CLASS DATASET CHECK")
print("=" * 60)

total_samples = 0
total_invalid = 0

for class_name in CLASSES:

    raw_folder = os.path.join(DATASET_DIR, class_name)
    include_folder = os.path.join(INCLUDE_DIR, class_name)

    raw_count, raw_invalid = check_folder(
        raw_folder,
        class_name
    )

    include_count, include_invalid = check_folder(
        include_folder,
        class_name
    )

    total = raw_count + include_count
    invalid = raw_invalid + include_invalid

    total_samples += total
    total_invalid += invalid

    print(
        f"{class_name:12s} "
        f"raw={raw_count:3d}  "
        f"INCLUDE={include_count:3d}  "
        f"total={total:3d}"
    )

print("=" * 60)
print(f"TOTAL VALID SAMPLES : {total_samples}")
print(f"TOTAL INVALID       : {total_invalid}")
print("=" * 60)

print("\nExpected sequence shape: (30, 225)")