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


# ---------------------------------------------------------
# FEATURE EXTRACTION
# ---------------------------------------------------------

def extract_features(sequence):

    # (30, 225) -> (30, 75, 3)
    points = sequence.reshape(
        SEQUENCE_LENGTH,
        75,
        3
    )

    groups = {
        "pose": points[:, 0:33, :],
        "left": points[:, 33:54, :],
        "right": points[:, 54:75, :]
    }

    features = {}

    all_frame_movement = []

    for name, group in groups.items():

        # Frame-to-frame landmark displacement
        diffs = np.diff(
            group,
            axis=0
        )

        movement = np.linalg.norm(
            diffs,
            axis=2
        )

        frame_movement = movement.mean(
            axis=1
        )

        # Basic movement
        features[f"{name}_mean"] = float(
            frame_movement.mean()
        )

        features[f"{name}_std"] = float(
            frame_movement.std()
        )

        features[f"{name}_max"] = float(
            frame_movement.max()
        )

        features[f"{name}_median"] = float(
            np.median(frame_movement)
        )

        # Active frames
        active_threshold = 0.01

        features[f"{name}_active_ratio"] = float(
            np.mean(
                frame_movement > active_threshold
            )
        )

        # Movement acceleration
        if len(frame_movement) > 1:

            acceleration = np.diff(
                frame_movement
            )

            features[f"{name}_accel_mean"] = float(
                np.mean(
                    np.abs(acceleration)
                )
            )

            features[f"{name}_accel_std"] = float(
                np.std(acceleration)
            )

        else:

            features[f"{name}_accel_mean"] = 0.0
            features[f"{name}_accel_std"] = 0.0

        all_frame_movement.append(
            frame_movement
        )

    # -----------------------------------------------------
    # Combined body movement
    # -----------------------------------------------------

    combined = np.mean(
        np.stack(all_frame_movement),
        axis=0
    )

    features["combined_mean"] = float(
        combined.mean()
    )

    features["combined_std"] = float(
        combined.std()
    )

    features["combined_max"] = float(
        combined.max()
    )

    features["combined_active_ratio"] = float(
        np.mean(combined > 0.01)
    )

    # -----------------------------------------------------
    # Movement concentration
    # -----------------------------------------------------

    total_movement = np.sum(combined)

    if total_movement > 0:

        normalized = (
            combined / total_movement
        )

        features["movement_concentration"] = float(
            np.max(normalized)
        )

    else:

        features["movement_concentration"] = 0.0

    # -----------------------------------------------------
    # Direction changes
    # -----------------------------------------------------

    pose_center = points[:, 0:33, :].mean(
        axis=1
    )

    trajectory = np.diff(
        pose_center,
        axis=0
    )

    if len(trajectory) > 2:

        direction_changes = 0

        for i in range(
            1,
            len(trajectory)
        ):

            previous = trajectory[i - 1]
            current = trajectory[i]

            dot = np.dot(
                previous,
                current
            )

            if dot < 0:
                direction_changes += 1

        features["pose_direction_changes"] = (
            direction_changes
            / (len(trajectory) - 1)
        )

    else:

        features["pose_direction_changes"] = 0.0

    # -----------------------------------------------------
    # Start/end movement
    # -----------------------------------------------------

    first_frames = combined[:5]
    last_frames = combined[-5:]

    features["start_movement"] = float(
        first_frames.mean()
    )

    features["end_movement"] = float(
        last_frames.mean()
    )

    # -----------------------------------------------------
    # Movement profile
    # -----------------------------------------------------

    first_half = combined[:15]
    second_half = combined[15:]

    features["first_half_mean"] = float(
        first_half.mean()
    )

    features["second_half_mean"] = float(
        second_half.mean()
    )

    features["half_difference"] = abs(
        features["first_half_mean"]
        -
        features["second_half_mean"]
    )

    return features


# ---------------------------------------------------------
# LOAD RANDOM DATA
# ---------------------------------------------------------

def load_random_sequences():

    files = sorted(
        glob.glob(
            os.path.join(
                RANDOM_DIR,
                "*.npy"
            )
        )
    )

    sequences = []

    for path in files:

        sequence = np.asarray(
            np.load(path),
            dtype=np.float32
        )

        if sequence.shape == (
            SEQUENCE_LENGTH,
            FEATURE_DIMENSION
        ):

            sequences.append(
                (
                    os.path.basename(path),
                    sequence
                )
            )

    return sequences


# ---------------------------------------------------------
# LOAD REAL SIGN DATA
# ---------------------------------------------------------

def load_real_sequences():

    sequences = []

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

                sequences.append(
                    (
                        class_name,
                        os.path.basename(path),
                        sequence
                    )
                )

            except Exception:
                continue

    return sequences


# ---------------------------------------------------------
# STATISTICS
# ---------------------------------------------------------

def summarize(
    name,
    feature_rows
):

    print()
    print("=" * 100)
    print(name)
    print("=" * 100)

    if not feature_rows:
        print("No data.")
        return

    keys = feature_rows[0].keys()

    print(
        f"{'Feature':<32}"
        f"{'Mean':>12}"
        f"{'Std':>12}"
        f"{'Min':>12}"
        f"{'Max':>12}"
    )

    print("-" * 100)

    for key in keys:

        values = np.array([
            row[key]
            for row in feature_rows
        ])

        print(
            f"{key:<32}"
            f"{values.mean():>12.5f}"
            f"{values.std():>12.5f}"
            f"{values.min():>12.5f}"
            f"{values.max():>12.5f}"
        )


# ---------------------------------------------------------
# MAIN
# ---------------------------------------------------------

def main():

    print("=" * 100)
    print(
        "REAL SIGN vs RANDOM MOVEMENT "
        "TEMPORAL ANALYSIS"
    )
    print("=" * 100)

    print()
    print("Real dataset:")
    print(REAL_DATASET_DIR)

    print()
    print("Random sequences:")
    print(RANDOM_DIR)

    # -----------------------------------------------------
    # Random
    # -----------------------------------------------------

    random_sequences = (
        load_random_sequences()
    )

    print()
    print(
        f"Random sequences loaded: "
        f"{len(random_sequences)}"
    )

    random_features = []

    for name, sequence in random_sequences:

        features = extract_features(
            sequence
        )

        random_features.append(
            features
        )

    # -----------------------------------------------------
    # Real signs
    # -----------------------------------------------------

    real_sequences = (
        load_real_sequences()
    )

    print(
        f"Real sign sequences loaded: "
        f"{len(real_sequences)}"
    )

    real_features = []

    for class_name, filename, sequence in (
        real_sequences
    ):

        features = extract_features(
            sequence
        )

        real_features.append(
            features
        )

    # -----------------------------------------------------
    # Summary
    # -----------------------------------------------------

    summarize(
        "REAL SIGN SEQUENCES",
        real_features
    )

    summarize(
        "RANDOM / NON-SIGN SEQUENCES",
        random_features
    )

    # -----------------------------------------------------
    # Separation analysis
    # -----------------------------------------------------

    print()
    print("=" * 100)
    print(
        "FEATURE SEPARATION"
    )
    print("=" * 100)

    if not real_features or not random_features:
        print(
            "Not enough data for comparison."
        )
        return

    keys = real_features[0].keys()

    separation = []

    for key in keys:

        real_values = np.array([
            row[key]
            for row in real_features
        ])

        random_values = np.array([
            row[key]
            for row in random_features
        ])

        real_mean = real_values.mean()
        random_mean = random_values.mean()

        pooled_std = np.sqrt(
            (
                real_values.var()
                +
                random_values.var()
            ) / 2
        )

        if pooled_std > 1e-8:

            effect = abs(
                real_mean
                -
                random_mean
            ) / pooled_std

        else:

            effect = 0.0

        separation.append(
            (
                effect,
                key,
                real_mean,
                random_mean
            )
        )

    separation.sort(
        reverse=True
    )

    print()
    print(
        f"{'Feature':<32}"
        f"{'Separation':>14}"
        f"{'Real Mean':>14}"
        f"{'Random Mean':>14}"
    )

    print("-" * 100)

    for effect, key, real_mean, random_mean in (
        separation
    ):

        print(
            f"{key:<32}"
            f"{effect:>14.3f}"
            f"{real_mean:>14.5f}"
            f"{random_mean:>14.5f}"
        )

    # -----------------------------------------------------
    # Top candidate features
    # -----------------------------------------------------

    print()
    print("=" * 100)
    print(
        "TOP CANDIDATE FEATURES"
    )
    print("=" * 100)

    for i, (
        effect,
        key,
        real_mean,
        random_mean
    ) in enumerate(
        separation[:10],
        start=1
    ):

        print(
            f"{i:2}. "
            f"{key:<30} "
            f"separation={effect:.3f} "
            f"real={real_mean:.5f} "
            f"random={random_mean:.5f}"
        )

    print()
    print("=" * 100)
    print(
        "ANALYSIS COMPLETE"
    )
    print("=" * 100)


if __name__ == "__main__":
    main()