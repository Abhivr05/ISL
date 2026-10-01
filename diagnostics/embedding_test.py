import os
import sys
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn

# ============================================================
# PROJECT PATH
# ============================================================

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from models.tcn_model import TCNModel


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_PATH = os.path.join(
    PROJECT_ROOT,
    "trained_models",
    "isl_tcn_19class_v3.pth"
)

REAL_DATASET = os.path.join(
    PROJECT_ROOT,
    "raw_dataset"
)

RANDOM_DATASET = os.path.join(
    PROJECT_ROOT,
    "diagnostics",
    "idle_sequences"
)

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
    "YESTERDAY",
]


# ============================================================
# DEVICE
# ============================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("=" * 70)
print("TCN INTERNAL EMBEDDING DIAGNOSTIC")
print("=" * 70)
print("Device:", device)
print("Model:", MODEL_PATH)
print()


# ============================================================
# LOAD MODEL
# ============================================================

model = TCNModel(
    input_size=FEATURE_DIMENSION,
    num_classes=len(CLASS_NAMES)
)

checkpoint = torch.load(
    MODEL_PATH,
    map_location=device
)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

model.to(device)
model.eval()

print("Model loaded successfully.")
print()


# ============================================================
# FIND FINAL LINEAR CLASSIFIER
# ============================================================

linear_layers = [
    module
    for module in model.modules()
    if isinstance(module, nn.Linear)
]

if not linear_layers:
    raise RuntimeError(
        "Could not find a Linear classifier in TCNModel."
    )

final_classifier = linear_layers[-1]

print(
    "Final classifier:",
    final_classifier
)

print(
    "Embedding dimension:",
    final_classifier.in_features
)

print()


# ============================================================
# CAPTURE INPUT TO FINAL CLASSIFIER
# ============================================================

captured_embedding = None


def capture_embedding(module, inputs):
    global captured_embedding

    captured_embedding = (
        inputs[0]
        .detach()
        .cpu()
        .numpy()
    )


hook = final_classifier.register_forward_pre_hook(
    capture_embedding
)


# ============================================================
# PREDICTION + EMBEDDING
# ============================================================

def predict_with_embedding(sequence):

    global captured_embedding

    captured_embedding = None

    sequence = np.asarray(
        sequence,
        dtype=np.float32
    )

    if sequence.shape != (
        SEQUENCE_LENGTH,
        FEATURE_DIMENSION
    ):
        raise ValueError(
            f"Invalid shape {sequence.shape}; "
            f"expected {(SEQUENCE_LENGTH, FEATURE_DIMENSION)}"
        )

    tensor = torch.from_numpy(
        sequence
    ).unsqueeze(0).to(device)

    with torch.no_grad():

        logits = model(tensor)

        probabilities = torch.softmax(
            logits,
            dim=1
        )

    probs = probabilities[0].cpu().numpy()

    prediction = int(
        np.argmax(probs)
    )

    confidence = float(
        probs[prediction]
    )

    sorted_probs = np.sort(probs)

    margin = float(
        sorted_probs[-1] -
        sorted_probs[-2]
    )

    embedding = np.asarray(
        captured_embedding[0],
        dtype=np.float32
    )

    return (
        prediction,
        confidence,
        margin,
        embedding
    )


# ============================================================
# LOAD REAL DATA
# ============================================================

real_samples = []

for class_index, class_name in enumerate(CLASS_NAMES):

    class_dir = os.path.join(
        REAL_DATASET,
        class_name
    )

    if not os.path.isdir(class_dir):
        continue

    for filename in sorted(
        os.listdir(class_dir)
    ):

        if not filename.endswith(".npy"):
            continue

        path = os.path.join(
            class_dir,
            filename
        )

        try:
            sequence = np.load(path)

            if sequence.shape != (
                SEQUENCE_LENGTH,
                FEATURE_DIMENSION
            ):
                continue

            real_samples.append(
                (
                    class_name,
                    sequence
                )
            )

        except Exception as exc:
            print(
                "Skipping:",
                path,
                "|",
                exc
            )


print(
    "Real samples loaded:",
    len(real_samples)
)


# ============================================================
# EXTRACT REAL EMBEDDINGS
# ============================================================

class_embeddings = defaultdict(list)

real_results = []

print(
    "\nExtracting real embeddings..."
)

for class_name, sequence in real_samples:

    prediction, confidence, margin, embedding = (
        predict_with_embedding(sequence)
    )

    class_embeddings[class_name].append(
        embedding
    )

    real_results.append(
        {
            "class": class_name,
            "prediction": CLASS_NAMES[prediction],
            "confidence": confidence,
            "margin": margin,
            "embedding": embedding,
        }
    )


# ============================================================
# CLASS CENTROIDS
# ============================================================

class_centroids = {}

for class_name, embeddings in class_embeddings.items():

    class_centroids[class_name] = np.mean(
        np.stack(embeddings),
        axis=0
    )


# ============================================================
# REAL SAMPLE DISTANCES
# ============================================================

class_distances = defaultdict(list)

for result in real_results:

    class_name = result["class"]

    centroid = class_centroids[
        class_name
    ]

    distance = float(
        np.linalg.norm(
            result["embedding"] -
            centroid
        )
    )

    class_distances[class_name].append(
        distance
    )


# ============================================================
# CLASS DISTANCE STATISTICS
# ============================================================

print()
print("=" * 70)
print("REAL CLASS EMBEDDING DISTRIBUTIONS")
print("=" * 70)

class_stats = {}

for class_name in CLASS_NAMES:

    distances = class_distances.get(
        class_name,
        []
    )

    if not distances:
        continue

    distances = np.asarray(
        distances
    )

    stats = {
        "mean": float(np.mean(distances)),
        "median": float(np.median(distances)),
        "p95": float(np.percentile(distances, 95)),
        "max": float(np.max(distances)),
    }

    class_stats[class_name] = stats

    print(
        f"{class_name:12s} "
        f"mean={stats['mean']:.4f} "
        f"median={stats['median']:.4f} "
        f"p95={stats['p95']:.4f} "
        f"max={stats['max']:.4f}"
    )


# ============================================================
# RANDOM / BACKGROUND DATA
# ============================================================

random_samples = []

if os.path.isdir(RANDOM_DATASET):

    for filename in sorted(
        os.listdir(RANDOM_DATASET)
    ):

        if not filename.endswith(".npy"):
            continue

        path = os.path.join(
            RANDOM_DATASET,
            filename
        )

        try:

            sequence = np.load(path)

            if sequence.shape != (
                SEQUENCE_LENGTH,
                FEATURE_DIMENSION
            ):
                continue

            random_samples.append(
                (
                    filename,
                    sequence
                )
            )

        except Exception as exc:

            print(
                "Skipping random sequence:",
                filename,
                exc
            )


print()
print(
    "Random/background samples:",
    len(random_samples)
)


# ============================================================
# RANDOM EMBEDDING ANALYSIS
# ============================================================

print()
print("=" * 70)
print("RANDOM / BACKGROUND EMBEDDING ANALYSIS")
print("=" * 70)

random_distances = []

for filename, sequence in random_samples:

    prediction, confidence, margin, embedding = (
        predict_with_embedding(sequence)
    )

    predicted_class = CLASS_NAMES[
        prediction
    ]

    centroid = class_centroids.get(
        predicted_class
    )

    if centroid is not None:

        distance = float(
            np.linalg.norm(
                embedding -
                centroid
            )
        )

        p95 = class_stats[
            predicted_class
        ]["p95"]

        ratio = (
            distance / p95
            if p95 > 0
            else float("inf")
        )

        random_distances.append(
            distance
        )

    else:

        distance = float("nan")
        ratio = float("nan")

    print(
        f"{filename:28s} "
        f"pred={predicted_class:12s} "
        f"conf={confidence * 100:6.2f}% "
        f"margin={margin * 100:6.2f}% "
        f"distance={distance:8.4f} "
        f"ratio={ratio:6.2f}x"
    )


# ============================================================
# NEAREST CLASS ANALYSIS
# ============================================================

print()
print("=" * 70)
print("RANDOM → NEAREST REAL CLASS")
print("=" * 70)

for filename, sequence in random_samples:

    prediction, confidence, margin, embedding = (
        predict_with_embedding(sequence)
    )

    distances = []

    for class_name, centroid in class_centroids.items():

        distance = float(
            np.linalg.norm(
                embedding -
                centroid
            )
        )

        distances.append(
            (
                distance,
                class_name
            )
        )

    distances.sort()

    nearest_distance, nearest_class = (
        distances[0]
    )

    second_distance, second_class = (
        distances[1]
    )

    separation = (
        second_distance -
        nearest_distance
    )

    print(
        f"{filename:28s} "
        f"nearest={nearest_class:12s} "
        f"distance={nearest_distance:8.4f} "
        f"2nd={second_class:12s} "
        f"separation={separation:8.4f}"
    )


# ============================================================
# CLEANUP
# ============================================================

hook.remove()

print()
print("=" * 70)
print("DIAGNOSTIC COMPLETE")
print("=" * 70)