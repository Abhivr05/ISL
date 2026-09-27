import os
import numpy as np


# --------------------------------------------------
# Dataset paths
# --------------------------------------------------

LOCAL_DATASET_PATH = "raw_dataset"
EXTERNAL_DATASET_PATH = "processed_dataset"

EXPECTED_SHAPE = (30, 225)

# Keep this order consistent with the existing model
CLASS_NAMES = [
    "GOOD",
    "HELLO",
    "HELP",
    "I",
    "NO",
    "SORRY",
    "THANK_YOU",
    "WATER",
    "YES",
    "YOU"
]


def load_samples_from_class(dataset_path, class_name):
    """
    Load all valid .npy samples for one class.
    """

    class_path = os.path.join(
        dataset_path,
        class_name
    )

    if not os.path.isdir(class_path):
        return []

    samples = []

    sample_files = sorted(
        [
            file
            for file in os.listdir(class_path)
            if file.endswith(".npy")
        ]
    )

    for sample_file in sample_files:

        file_path = os.path.join(
            class_path,
            sample_file
        )

        sequence = np.load(file_path)

        if sequence.shape != EXPECTED_SHAPE:

            print(
                f"Skipping {file_path}: "
                f"expected {EXPECTED_SHAPE}, "
                f"got {sequence.shape}"
            )

            continue

        samples.append(sequence)

    return samples


def load_dataset(
    local_dataset_path=LOCAL_DATASET_PATH,
    external_dataset_path=EXTERNAL_DATASET_PATH
):
    """
    Load both the original local dataset and the
    processed external dataset.

    Returns:
        X: NumPy array of shape (N, 30, 225)
        y: NumPy array of shape (N,)
        class_names: List of class names
    """

    X = []
    y = []

    print("Classes used:")
    for index, class_name in enumerate(CLASS_NAMES):
        print(f"{index}: {class_name}")

    print("\nLoading samples...\n")

    # --------------------------------------------------
    # Load each class
    # --------------------------------------------------

    for label, class_name in enumerate(CLASS_NAMES):

        local_samples = load_samples_from_class(
            local_dataset_path,
            class_name
        )

        external_samples = load_samples_from_class(
            external_dataset_path,
            class_name
        )

        total_samples = (
            len(local_samples)
            + len(external_samples)
        )

        # Add local samples
        for sequence in local_samples:
            X.append(sequence)
            y.append(label)

        # Add external samples
        for sequence in external_samples:
            X.append(sequence)
            y.append(label)

        print(
            f"{class_name:12} | "
            f"Local: {len(local_samples):3} | "
            f"External: {len(external_samples):3} | "
            f"Total: {total_samples:3}"
        )

    # --------------------------------------------------
    # Convert to NumPy arrays
    # --------------------------------------------------

    X = np.array(
        X,
        dtype=np.float32
    )

    y = np.array(
        y,
        dtype=np.int64
    )

    print("\n===== DATASET LOADED =====")
    print(f"X shape: {X.shape}")
    print(f"y shape: {y.shape}")
    print(f"Number of classes: {len(CLASS_NAMES)}")
    print("==========================")

    return X, y, CLASS_NAMES


if __name__ == "__main__":
    load_dataset()