
import os
import numpy as np


# ============================================================
# SETTINGS
# ============================================================

RAW_DATASET = "raw_dataset"
LIVE_DATASET = "debug_v2_sequences"

SEQUENCE_LENGTH = 30
FEATURE_DIMENSION = 225


# ============================================================
# LOAD .NPY FILES
# ============================================================

def load_sequences(folder):

    sequences = []

    if not os.path.exists(folder):
        return sequences

    for root, _, files in os.walk(folder):

        for file in files:

            if not file.endswith(".npy"):
                continue

            path = os.path.join(root, file)

            try:

                data = np.load(path)

                if data.shape == (
                    SEQUENCE_LENGTH,
                    FEATURE_DIMENSION
                ):
                    sequences.append(
                        (path, data.astype(np.float32))
                    )

            except Exception as e:

                print(
                    f"Could not read {path}: {e}"
                )

    return sequences


# ============================================================
# STATISTICS
# ============================================================

def sequence_statistics(sequence):

    # Average movement between consecutive frames
    motion = np.mean(
        np.abs(
            sequence[1:] - sequence[:-1]
        )
    )

    # Amount of variation throughout sequence
    variation = np.std(sequence)

    # Average absolute landmark value
    magnitude = np.mean(
        np.abs(sequence)
    )

    return motion, variation, magnitude


# ============================================================
# LOAD DATA
# ============================================================

print("\nLoading existing training data...")

training = load_sequences(
    RAW_DATASET
)

print(
    f"Training sequences found: {len(training)}"
)

print("\nLoading live V2 sequences...")

live = load_sequences(
    LIVE_DATASET
)

print(
    f"Live V2 sequences found: {len(live)}"
)


# ============================================================
# TRAINING STATISTICS BY CLASS
# ============================================================

print("\n" + "=" * 90)
print("TRAINING DATA STATISTICS")
print("=" * 90)

training_by_class = {}

for path, sequence in training:

    class_name = os.path.basename(
        os.path.dirname(path)
    )

    training_by_class.setdefault(
        class_name,
        []
    ).append(sequence)


for class_name in sorted(training_by_class):

    values = [
        sequence_statistics(seq)
        for seq in training_by_class[class_name]
    ]

    motions = [v[0] for v in values]
    variations = [v[1] for v in values]
    magnitudes = [v[2] for v in values]

    print(
        f"{class_name:12s} "
        f"n={len(values):3d} | "
        f"motion={np.mean(motions):.5f} | "
        f"std={np.mean(variations):.5f} | "
        f"magnitude={np.mean(magnitudes):.5f}"
    )


# ============================================================
# LIVE STATISTICS
# ============================================================

print("\n" + "=" * 90)
print("LIVE V2 SEQUENCES")
print("=" * 90)

live_values = []

for path, sequence in live:

    motion, variation, magnitude = (
        sequence_statistics(sequence)
    )

    live_values.append(
        (motion, variation, magnitude)
    )

    print(
        f"{os.path.basename(path):25s} "
        f"motion={motion:.5f} | "
        f"std={variation:.5f} | "
        f"magnitude={magnitude:.5f}"
    )


# ============================================================
# OVERALL COMPARISON
# ============================================================

if live_values and training:

    training_values = [
        sequence_statistics(seq)
        for _, seq in training
    ]

    print("\n" + "=" * 90)
    print("OVERALL COMPARISON")
    print("=" * 90)

    labels = [
        "Motion",
        "Variation",
        "Magnitude"
    ]

    for i, label in enumerate(labels):

        train_mean = np.mean([
            x[i]
            for x in training_values
        ])

        train_std = np.std([
            x[i]
            for x in training_values
        ])

        live_mean = np.mean([
            x[i]
            for x in live_values
        ])

        print(
            f"{label:12s} | "
            f"TRAIN mean={train_mean:.5f}, "
            f"std={train_std:.5f} | "
            f"LIVE mean={live_mean:.5f}"
        )


# ============================================================
# HELLO COMPARISON
# ============================================================

hello_training = training_by_class.get(
    "HELLO",
    []
)

if hello_training and live:

    print("\n" + "=" * 90)
    print("HELLO vs LIVE COMPARISON")
    print("=" * 90)

    hello_values = [
        sequence_statistics(seq)
        for seq in hello_training
    ]

    for i, label in enumerate(labels):

        train_mean = np.mean([
            x[i]
            for x in hello_values
        ])

        live_mean = np.mean([
            x[i]
            for x in live_values
        ])

        print(
            f"{label:12s} | "
            f"HELLO training={train_mean:.5f} | "
            f"LIVE={live_mean:.5f}"
        )


# ============================================================
# SHAPE CHECK
# ============================================================

print("\n" + "=" * 90)
print("SHAPE CHECK")
print("=" * 90)

bad_training = 0
bad_live = 0

for path, sequence in training:

    if sequence.shape != (
        SEQUENCE_LENGTH,
        FEATURE_DIMENSION
    ):
        bad_training += 1

for path, sequence in live:

    if sequence.shape != (
        SEQUENCE_LENGTH,
        FEATURE_DIMENSION
    ):
        bad_live += 1

print(
    f"Training bad shapes: {bad_training}"
)

print(
    f"Live bad shapes:     {bad_live}"
)

print("\nDiagnostic complete.")
