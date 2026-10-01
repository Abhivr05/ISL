import os
import glob
import numpy as np
import torch

from models.tcn_model import TCNModel


# =========================================================
# PATHS
# =========================================================

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

MODEL_PATH = os.path.join(
    PROJECT_ROOT,
    "trained_models",
    "isl_tcn_19class_v3.pth"
)


# =========================================================
# MODEL
# =========================================================

SEQUENCE_LENGTH = 30
FEATURE_DIMENSION = 225

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
    "YOU",
    "HOME",
    "SCHOOL",
    "HOSPITAL",
    "MONEY",
    "MEDICINE",
    "TODAY",
    "TOMORROW",
    "YESTERDAY"
]


# =========================================================
# FEATURE EXTRACTION
# =========================================================

FEATURE_NAMES = [
    "pose_mean",
    "pose_std",
    "pose_median",
    "pose_max",

    "left_mean",
    "left_std",
    "left_median",
    "left_max",
    "left_active",

    "right_mean",
    "right_std",
    "right_median",
    "right_max",
    "right_active",

    "combined_mean",
    "combined_std",
    "combined_max",

    "movement_concentration",
    "pose_direction_changes",

    "start_movement",
    "end_movement",

    "first_half_mean",
    "second_half_mean",
    "half_difference"
]


def extract_features(sequence):

    points = sequence.reshape(
        SEQUENCE_LENGTH,
        75,
        3
    )

    pose = points[:, 0:33, :]
    left = points[:, 33:54, :]
    right = points[:, 54:75, :]

    # -----------------------------------------------------
    # Movement
    # -----------------------------------------------------

    pose_diff = np.diff(
        pose,
        axis=0
    )

    left_diff = np.diff(
        left,
        axis=0
    )

    right_diff = np.diff(
        right,
        axis=0
    )

    pose_movement = np.linalg.norm(
        pose_diff,
        axis=2
    ).mean(axis=1)

    left_movement = np.linalg.norm(
        left_diff,
        axis=2
    ).mean(axis=1)

    right_movement = np.linalg.norm(
        right_diff,
        axis=2
    ).mean(axis=1
    )

    combined = np.mean(
        np.stack([
            pose_movement,
            left_movement,
            right_movement
        ]),
        axis=0
    )

    # -----------------------------------------------------
    # Direction changes
    # -----------------------------------------------------

    pose_center = pose.mean(
        axis=1
    )

    trajectory = np.diff(
        pose_center,
        axis=0
    )

    direction_changes = 0

    if len(trajectory) > 1:

        for i in range(
            1,
            len(trajectory)
        ):

            if np.dot(
                trajectory[i - 1],
                trajectory[i]
            ) < 0:

                direction_changes += 1

    if len(trajectory) > 1:

        direction_ratio = (
            direction_changes
            /
            (len(trajectory) - 1)
        )

    else:

        direction_ratio = 0.0

    # -----------------------------------------------------
    # Movement concentration
    # -----------------------------------------------------

    total = combined.sum()

    if total > 0:

        movement_concentration = (
            combined.max()
            /
            total
        )

    else:

        movement_concentration = 0.0

    # -----------------------------------------------------
    # Feature vector
    # -----------------------------------------------------

    return np.array([
        pose_movement.mean(),
        pose_movement.std(),
        np.median(pose_movement),
        pose_movement.max(),

        left_movement.mean(),
        left_movement.std(),
        np.median(left_movement),
        left_movement.max(),
        np.mean(left_movement > 0.01),

        right_movement.mean(),
        right_movement.std(),
        np.median(right_movement),
        right_movement.max(),
        np.mean(right_movement > 0.01),

        combined.mean(),
        combined.std(),
        combined.max(),

        movement_concentration,
        direction_ratio,

        combined[:5].mean(),
        combined[-5:].mean(),

        combined[:15].mean(),
        combined[15:].mean(),
        abs(
            combined[:15].mean()
            -
            combined[15:].mean()
        )
    ], dtype=np.float32)


# =========================================================
# LOAD REAL DATA
# =========================================================

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
                    extract_features(
                        sequence
                    )
                )

            except Exception:
                pass

    return data


# =========================================================
# LOAD RANDOM DATA
# =========================================================

def load_random_data():

    data = []

    files = sorted(
        glob.glob(
            os.path.join(
                RANDOM_DIR,
                "*.npy"
            )
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

            data.append(
                (
                    os.path.basename(path),
                    sequence
                )
            )

        except Exception:
            pass

    return data


# =========================================================
# MODEL
# =========================================================

def load_model():

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    checkpoint = torch.load(
        MODEL_PATH,
        map_location=device
    )

    model = TCNModel(
        input_size=FEATURE_DIMENSION,
        num_classes=len(CLASS_NAMES)
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.to(device)
    model.eval()

    return model, device


def predict(
    model,
    device,
    sequence
):

    tensor = torch.tensor(
        sequence,
        dtype=torch.float32
    ).unsqueeze(0).to(device)

    with torch.no_grad():

        output = model(
            tensor
        )

        probabilities = torch.softmax(
            output,
            dim=1
        )[0]

    values, indices = torch.topk(
        probabilities,
        k=2
    )

    prediction = CLASS_NAMES[
        int(indices[0])
    ]

    confidence = float(
        values[0]
    )

    runner_up = CLASS_NAMES[
        int(indices[1])
    ]

    runner_confidence = float(
        values[1]
    )

    margin = (
        confidence
        -
        runner_confidence
    )

    return (
        prediction,
        confidence,
        runner_up,
        runner_confidence,
        margin
    )


# =========================================================
# CLASS PROFILE
# =========================================================

def build_profiles(
    real_data
):

    profiles = {}

    for class_name, rows in real_data.items():

        if not rows:
            continue

        matrix = np.stack(
            rows
        )

        profiles[class_name] = {
            "mean": matrix.mean(
                axis=0
            ),
            "std": matrix.std(
                axis=0
            )
        }

    return profiles


# =========================================================
# DISTANCE
# =========================================================

def profile_distance(
    vector,
    profile
):

    mean = profile["mean"]
    std = profile["std"]

    # Prevent zero-variance features
    std = np.maximum(
        std,
        0.01
    )

    z = np.abs(
        vector - mean
    ) / std

    return float(
        np.mean(z)
    )


# =========================================================
# REAL SAMPLE BASELINE
# =========================================================

def calculate_real_baselines(
    real_data,
    profiles
):

    baselines = {}

    for class_name, rows in real_data.items():

        if class_name not in profiles:
            continue

        distances = []

        for vector in rows:

            distances.append(
                profile_distance(
                    vector,
                    profiles[class_name]
                )
            )

        if distances:

            baselines[class_name] = {
                "mean": float(
                    np.mean(distances)
                ),
                "median": float(
                    np.median(distances)
                ),
                "max": float(
                    np.max(distances)
                ),
                "p95": float(
                    np.percentile(
                        distances,
                        95
                    )
                )
            }

    return baselines


# =========================================================
# MAIN
# =========================================================

def main():

    print("=" * 120)
    print(
        "V3 CLASS PROFILE vs RANDOM SEQUENCE TEST"
    )
    print("=" * 120)

    print()
    print(
        "Real dataset:",
        REAL_DATASET_DIR
    )

    print(
        "Random dataset:",
        RANDOM_DIR
    )

    print(
        "Model:",
        MODEL_PATH
    )

    # -----------------------------------------------------
    # Load data
    # -----------------------------------------------------

    real_data = load_real_data()

    random_data = load_random_data()

    print()
    print(
        "Real classes loaded:",
        len(real_data)
    )

    print(
        "Random sequences loaded:",
        len(random_data)
    )

    # -----------------------------------------------------
    # Profiles
    # -----------------------------------------------------

    profiles = build_profiles(
        real_data
    )

    baselines = calculate_real_baselines(
        real_data,
        profiles
    )

    print()
    print("=" * 120)
    print("REAL CLASS PROFILE BASELINES")
    print("=" * 120)

    print()
    print(
        f"{'Class':<15}"
        f"{'Samples':>9}"
        f"{'Mean Dist':>12}"
        f"{'Median':>12}"
        f"{'95%':>12}"
        f"{'Max':>12}"
    )

    print("-" * 75)

    for class_name in sorted(
        baselines
    ):

        b = baselines[
            class_name
        ]

        print(
            f"{class_name:<15}"
            f"{len(real_data[class_name]):>9}"
            f"{b['mean']:>12.3f}"
            f"{b['median']:>12.3f}"
            f"{b['p95']:>12.3f}"
            f"{b['max']:>12.3f}"
        )

    # -----------------------------------------------------
    # Load model
    # -----------------------------------------------------

    model, device = load_model()

    print()
    print(
        "Model device:",
        device
    )

    # -----------------------------------------------------
    # Random tests
    # -----------------------------------------------------

    print()
    print("=" * 120)
    print("RANDOM SEQUENCE RESULTS")
    print("=" * 120)

    print()

    print(
        f"{'Sequence':<26}"
        f"{'TCN':<15}"
        f"{'Conf':>8}"
        f"{'Margin':>9}"
        f"{'Profile':>12}"
        f"{'Real P95':>10}"
        f"{'Ratio':>10}"
    )

    print("-" * 100)

    for filename, sequence in random_data:

        prediction, confidence, runner_up, runner_confidence, margin = (
            predict(
                model,
                device,
                sequence
            )
        )

        vector = extract_features(
            sequence
        )

        # ---------------------------------------------
        # If class exists in raw dataset
        # ---------------------------------------------

        if prediction in profiles:

            distance = profile_distance(
                vector,
                profiles[prediction]
            )

            p95 = baselines[
                prediction
            ]["p95"]

            if p95 > 0:

                ratio = (
                    distance
                    /
                    p95
                )

            else:

                ratio = 0.0

            profile_text = (
                f"{distance:.3f}"
            )

            p95_text = (
                f"{p95:.3f}"
            )

            ratio_text = (
                f"{ratio:.2f}x"
            )

        else:

            profile_text = "N/A"
            p95_text = "N/A"
            ratio_text = "N/A"

        print(
            f"{filename:<26}"
            f"{prediction:<15}"
            f"{confidence * 100:>7.2f}%"
            f"{margin * 100:>8.2f}%"
            f"{profile_text:>12}"
            f"{p95_text:>10}"
            f"{ratio_text:>10}"
        )

    print()
    print("=" * 120)
    print(
        "INTERPRETATION"
    )
    print("=" * 120)

    print()
    print(
        "Profile ratio = random sequence distance / "
        "95th percentile distance of real samples "
        "from that class."
    )

    print()
    print(
        "Ratio <= 1.0 means the random sequence is "
        "within the observed 95% profile range."
    )

    print(
        "Ratio > 1.0 means it is farther from the "
        "class profile than 95% of real samples."
    )

    print()
    print(
        "This is diagnostic only. "
        "No rejection threshold is being applied."
    )

    print()
    print("=" * 120)
    print(
        "TEST COMPLETE"
    )
    print("=" * 120)


if __name__ == "__main__":
    main()