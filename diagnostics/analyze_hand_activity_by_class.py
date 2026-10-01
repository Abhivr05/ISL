import os
import glob
import numpy as np


PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

REAL_DATASET_DIR = os.path.join(
    PROJECT_ROOT,
    "raw_dataset"
)

RANDOM_DIR = os.path.join(
    PROJECT_ROOT,
    "diagnostics",
    "idle_sequences"
)

SEQUENCE_LENGTH = 30
FEATURE_DIMENSION = 225


def hand_features(sequence):

    points = sequence.reshape(
        SEQUENCE_LENGTH,
        75,
        3
    )

    left = points[:, 33:54, :]
    right = points[:, 54:75, :]

    left_diff = np.diff(
        left,
        axis=0
    )

    right_diff = np.diff(
        right,
        axis=0
    )

    left_movement = np.linalg.norm(
        left_diff,
        axis=2
    ).mean(axis=1)

    right_movement = np.linalg.norm(
        right_diff,
        axis=2
    ).mean(axis=1
    )

    threshold = 0.01

    left_active = np.mean(
        left_movement > threshold
    )

    right_active = np.mean(
        right_movement > threshold
    )

    left_mean = left_movement.mean()
    right_mean = right_movement.mean()

    left_max = left_movement.max()
    right_max = right_movement.max()

    return {
        "left_active": left_active,
        "right_active": right_active,
        "left_mean": left_mean,
        "right_mean": right_mean,
        "left_max": left_max,
        "right_max": right_max
    }


def load_real_data():

    data = {}

    class_dirs = [
        path
        for path in glob.glob(
            os.path.join(
                REAL_DATASET_DIR,
                "*"
            )
        )
        if os.path.isdir(path)
    ]

    for class_dir in sorted(class_dirs):

        class_name = os.path.basename(
            class_dir
        )

        data[class_name] = []

        files = glob.glob(
            os.path.join(
                class_dir,
                "*.npy"
            )
        )

        for path in files:

            try:

                sequence = np.asarray(
                    np.load(path),
                    dtype=np.float32
                )

                if sequence.shape != (
                    SEQUENCE_LENGTH,
                    FEATURE_DIMENSION
                ):
                    continue

                data[class_name].append(
                    hand_features(sequence)
                )

            except Exception:
                pass

    return data


def load_random_data():

    results = []

    files = glob.glob(
        os.path.join(
            RANDOM_DIR,
            "*.npy"
        )
    )

    for path in files:

        try:

            sequence = np.asarray(
                np.load(path),
                dtype=np.float32
            )

            if sequence.shape != (
                SEQUENCE_LENGTH,
                FEATURE_DIMENSION
            ):
                continue

            results.append(
                (
                    os.path.basename(path),
                    hand_features(sequence)
                )
            )

        except Exception:
            pass

    return results


def average(values):

    if not values:
        return 0.0

    return float(
        np.mean(values)
    )


def main():

    print("=" * 120)
    print(
        "HAND ACTIVITY BY SIGN CLASS"
    )
    print("=" * 120)

    real_data = load_real_data()
    random_data = load_random_data()

    print()
    print(
        f"Real classes loaded : {len(real_data)}"
    )

    print(
        f"Random sequences    : {len(random_data)}"
    )

    print()
    print("-" * 120)

    print(
        f"{'Class':<15}"
        f"{'Samples':>8}"
        f"{'LH Active':>12}"
        f"{'RH Active':>12}"
        f"{'LH Mean':>12}"
        f"{'RH Mean':>12}"
        f"{'LH Max':>12}"
        f"{'RH Max':>12}"
    )

    print("-" * 120)

    for class_name in sorted(real_data):

        rows = real_data[class_name]

        if not rows:
            continue

        print(
            f"{class_name:<15}"
            f"{len(rows):>8}"
            f"{average([x['left_active'] for x in rows]):>12.3f}"
            f"{average([x['right_active'] for x in rows]):>12.3f}"
            f"{average([x['left_mean'] for x in rows]):>12.4f}"
            f"{average([x['right_mean'] for x in rows]):>12.4f}"
            f"{average([x['left_max'] for x in rows]):>12.4f}"
            f"{average([x['right_max'] for x in rows]):>12.4f}"
        )

    print("-" * 120)

    print()
    print("=" * 120)
    print("RANDOM SEQUENCES")
    print("=" * 120)

    print()

    print(
        f"{'Sequence':<28}"
        f"{'LH Active':>12}"
        f"{'RH Active':>12}"
        f"{'LH Mean':>12}"
        f"{'RH Mean':>12}"
    )

    print("-" * 80)

    for name, row in random_data:

        print(
            f"{name:<28}"
            f"{row['left_active']:>12.3f}"
            f"{row['right_active']:>12.3f}"
            f"{row['left_mean']:>12.4f}"
            f"{row['right_mean']:>12.4f}"
        )

    print()
    print("=" * 120)
    print("ANALYSIS COMPLETE")
    print("=" * 120)


if __name__ == "__main__":
    main()