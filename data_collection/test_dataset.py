import os
import numpy as np

DATASET_PATH = "raw_dataset"

EXPECTED_SHAPE = (30, 225)

total_samples = 0
all_valid = True

print("\n===== DATASET VALIDATION =====\n")

for class_name in sorted(os.listdir(DATASET_PATH)):

    class_path = os.path.join(DATASET_PATH, class_name)

    if not os.path.isdir(class_path):
        continue

    samples = [
        file for file in os.listdir(class_path)
        if file.endswith(".npy")
    ]

    print(f"{class_name}: {len(samples)} samples")

    for sample_file in sorted(samples):

        file_path = os.path.join(class_path, sample_file)

        try:
            data = np.load(file_path)

            if data.shape != EXPECTED_SHAPE:
                print(
                    f"  ERROR: {sample_file} "
                    f"has shape {data.shape}"
                )
                all_valid = False

            if data.dtype != np.float32:
                print(
                    f"  ERROR: {sample_file} "
                    f"has dtype {data.dtype}"
                )
                all_valid = False

            if not np.isfinite(data).all():
                print(
                    f"  ERROR: {sample_file} "
                    f"contains NaN or Inf"
                )
                all_valid = False

            total_samples += 1

        except Exception as e:
            print(f"  ERROR: {sample_file} -> {e}")
            all_valid = False

print("\n==============================")
print(f"Total samples: {total_samples}")

if all_valid:
    print("Dataset validation: PASSED")
else:
    print("Dataset validation: FAILED")

print("==============================")