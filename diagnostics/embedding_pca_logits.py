import os
import sys

import numpy as np
import torch
import torch.nn as nn

try:
    from sklearn.decomposition import PCA
except ImportError:
    print("ERROR: scikit-learn is required.")
    print("Install with: pip install scikit-learn")
    raise


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

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


# ============================================================
# CONFIG
# ============================================================

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

print("=" * 80)
print("TCN EMBEDDING PCA + LOGIT DIAGNOSTIC")
print("=" * 80)
print("Device:", device)
print()


# ============================================================
# LOAD MODEL
# ============================================================

from models.tcn_model import TCNModel


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


# ============================================================
# FIND FINAL CLASSIFIER
# ============================================================

linear_layers = [
    layer
    for layer in model.modules()
    if isinstance(layer, nn.Linear)
]

if not linear_layers:
    raise RuntimeError(
        "Could not find final Linear layer."
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
# HOOK EMBEDDING
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
# PREDICT
# ============================================================

def predict(sequence):

    global captured_embedding

    captured_embedding = None

    sequence = np.asarray(
        sequence,
        dtype=np.float32
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

    logits_np = logits[0].cpu().numpy()
    probs_np = probabilities[0].cpu().numpy()

    prediction = int(
        np.argmax(probs_np)
    )

    sorted_indices = np.argsort(
        probs_np
    )[::-1]

    top1 = sorted_indices[0]
    top2 = sorted_indices[1]

    confidence = float(
        probs_np[top1]
    )

    margin = float(
        probs_np[top1] -
        probs_np[top2]
    )

    embedding = np.asarray(
        captured_embedding[0],
        dtype=np.float32
    )

    return {
        "prediction": prediction,
        "confidence": confidence,
        "margin": margin,
        "logits": logits_np,
        "embedding": embedding,
    }


# ============================================================
# LOAD REAL SAMPLES
# ============================================================

real_samples = []

for class_name in CLASS_NAMES:

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

        sequence = np.load(path)

        if sequence.shape != (
            SEQUENCE_LENGTH,
            FEATURE_DIMENSION
        ):
            continue

        real_samples.append(
            (
                class_name,
                filename,
                sequence
            )
        )


print(
    "Real samples:",
    len(real_samples)
)


# ============================================================
# EXTRACT REAL DATA
# ============================================================

real_embeddings = []
real_logits = []
real_labels = []
real_predictions = []

for class_name, filename, sequence in real_samples:

    result = predict(sequence)

    real_embeddings.append(
        result["embedding"]
    )

    real_logits.append(
        result["logits"]
    )

    real_labels.append(
        class_name
    )

    real_predictions.append(
        result
    )


real_embeddings = np.asarray(
    real_embeddings
)

real_logits = np.asarray(
    real_logits
)


# ============================================================
# PCA
# ============================================================

print()
print("=" * 80)
print("PCA")
print("=" * 80)

pca = PCA(
    n_components=2,
    random_state=42
)

real_pca = pca.fit_transform(
    real_embeddings
)

print(
    "PC1 variance:",
    f"{pca.explained_variance_ratio_[0] * 100:.2f}%"
)

print(
    "PC2 variance:",
    f"{pca.explained_variance_ratio_[1] * 100:.2f}%"
)

print(
    "Combined:",
    f"{sum(pca.explained_variance_ratio_) * 100:.2f}%"
)


# ============================================================
# REAL CLASS CENTROIDS IN PCA
# ============================================================

class_pca_centroids = {}

for class_name in CLASS_NAMES:

    indices = [
        i
        for i, label in enumerate(real_labels)
        if label == class_name
    ]

    if not indices:
        continue

    class_pca_centroids[class_name] = np.mean(
        real_pca[indices],
        axis=0
    )


# ============================================================
# REAL CLASS SPREAD
# ============================================================

print()
print("=" * 80)
print("REAL CLASS PCA SPREAD")
print("=" * 80)

class_pca_spreads = {}

for class_name, centroid in class_pca_centroids.items():

    indices = [
        i
        for i, label in enumerate(real_labels)
        if label == class_name
    ]

    points = real_pca[indices]

    distances = np.linalg.norm(
        points - centroid,
        axis=1
    )

    class_pca_spreads[class_name] = {
        "mean": float(np.mean(distances)),
        "p95": float(np.percentile(distances, 95)),
        "max": float(np.max(distances)),
    }

    print(
        f"{class_name:12s} "
        f"mean={np.mean(distances):8.3f} "
        f"p95={np.percentile(distances, 95):8.3f} "
        f"max={np.max(distances):8.3f}"
    )


# ============================================================
# RANDOM DATA
# ============================================================

random_samples = []

for filename in sorted(
    os.listdir(RANDOM_DATASET)
):

    if not filename.endswith(".npy"):
        continue

    path = os.path.join(
        RANDOM_DATASET,
        filename
    )

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


print()
print(
    "Random samples:",
    len(random_samples)
)


# ============================================================
# RANDOM RESULTS
# ============================================================

random_embeddings = []
random_logits = []
random_results = []

for filename, sequence in random_samples:

    result = predict(sequence)

    random_embeddings.append(
        result["embedding"]
    )

    random_logits.append(
        result["logits"]
    )

    random_results.append(
        (
            filename,
            result
        )
    )


random_embeddings = np.asarray(
    random_embeddings
)

random_logits = np.asarray(
    random_logits
)


# ============================================================
# PROJECT RANDOM INTO SAME PCA SPACE
# ============================================================

random_pca = pca.transform(
    random_embeddings
)


# ============================================================
# RANDOM PCA DISTANCE TO PREDICTED CLASS
# ============================================================

print()
print("=" * 80)
print("RANDOM PCA CLASS DISTANCE")
print("=" * 80)

for i, (filename, result) in enumerate(
    random_results
):

    predicted_class = CLASS_NAMES[
        result["prediction"]
    ]

    random_point = random_pca[i]

    centroid = class_pca_centroids.get(
        predicted_class
    )

    if centroid is None:
        continue

    distance = float(
        np.linalg.norm(
            random_point -
            centroid
        )
    )

    stats = class_pca_spreads[
        predicted_class
    ]

    ratio = (
        distance / stats["p95"]
        if stats["p95"] > 0
        else float("inf")
    )

    print(
        f"{filename:28s} "
        f"{predicted_class:12s} "
        f"conf={result['confidence'] * 100:6.2f}% "
        f"distance={distance:8.3f} "
        f"p95_ratio={ratio:6.2f}x"
    )


# ============================================================
# LOGIT STATISTICS
# ============================================================

print()
print("=" * 80)
print("LOGIT MAGNITUDE")
print("=" * 80)

real_abs_logits = np.abs(
    real_logits
)

random_abs_logits = np.abs(
    random_logits
)

print(
    "Real mean |logit|:",
    f"{np.mean(real_abs_logits):.4f}"
)

print(
    "Random mean |logit|:",
    f"{np.mean(random_abs_logits):.4f}"
)

print(
    "Real max |logit|:",
    f"{np.max(real_abs_logits):.4f}"
)

print(
    "Random max |logit|:",
    f"{np.max(random_abs_logits):.4f}"
)


# ============================================================
# TOP LOGIT + LOGIT GAP
# ============================================================

def logit_statistics(logits):

    sorted_logits = np.sort(
        logits
    )[::-1]

    top1 = float(
        sorted_logits[0]
    )

    top2 = float(
        sorted_logits[1]
    )

    return (
        top1,
        top2,
        top1 - top2
    )


real_top1 = []
real_gaps = []

for logits in real_logits:

    top1, top2, gap = (
        logit_statistics(logits)
    )

    real_top1.append(top1)
    real_gaps.append(gap)


random_top1 = []
random_gaps = []

for logits in random_logits:

    top1, top2, gap = (
        logit_statistics(logits)
    )

    random_top1.append(top1)
    random_gaps.append(gap)


print()
print("=" * 80)
print("LOGIT SEPARATION")
print("=" * 80)

print(
    "Real top logit mean:",
    f"{np.mean(real_top1):.4f}"
)

print(
    "Random top logit mean:",
    f"{np.mean(random_top1):.4f}"
)

print(
    "Real top-logit gap mean:",
    f"{np.mean(real_gaps):.4f}"
)

print(
    "Random top-logit gap mean:",
    f"{np.mean(random_gaps):.4f}"
)


# ============================================================
# RANDOM DETAILED LOGITS
# ============================================================

print()
print("=" * 80)
print("RANDOM LOGIT DETAILS")
print("=" * 80)

for filename, result in random_results:

    logits = result["logits"]

    order = np.argsort(
        logits
    )[::-1]

    top = order[:3]

    print()
    print(filename)

    for rank, index in enumerate(
        top,
        start=1
    ):

        print(
            f"  {rank}. "
            f"{CLASS_NAMES[index]:12s} "
            f"logit={logits[index]:9.4f}"
        )

    print(
        f"  confidence={result['confidence'] * 100:.2f}%"
    )

    print(
        f"  probability margin={result['margin'] * 100:.2f}%"
    )


# ============================================================
# GOOD-SPECIFIC ANALYSIS
# ============================================================

print()
print("=" * 80)
print("GOOD-SPECIFIC ANALYSIS")
print("=" * 80)

good_index = CLASS_NAMES.index("GOOD")

real_good_indices = [
    i
    for i, label in enumerate(real_labels)
    if label == "GOOD"
]

real_good_logits = real_logits[
    real_good_indices,
    good_index
]

random_good_logits = random_logits[
    :,
    good_index
]

print(
    "Real GOOD logit:"
)

print(
    f"  mean={np.mean(real_good_logits):.4f}"
)

print(
    f"  median={np.median(real_good_logits):.4f}"
)

print(
    f"  p95={np.percentile(real_good_logits, 95):.4f}"
)

print(
    f"  max={np.max(real_good_logits):.4f}"
)

print()
print(
    "Random GOOD logit:"
)

print(
    f"  mean={np.mean(random_good_logits):.4f}"
)

print(
    f"  median={np.median(random_good_logits):.4f}"
)

print(
    f"  p95={np.percentile(random_good_logits, 95):.4f}"
)

print(
    f"  max={np.max(random_good_logits):.4f}"
)


# ============================================================
# SAVE PCA DATA
# ============================================================

output_path = os.path.join(
    PROJECT_ROOT,
    "diagnostics",
    "embedding_pca_results.npz"
)

np.savez(
    output_path,
    real_pca=real_pca,
    random_pca=random_pca,
    real_labels=np.asarray(
        real_labels
    ),
    random_names=np.asarray(
        [name for name, _ in random_samples]
    ),
)

print()
print(
    "PCA data saved:",
    output_path
)


# ============================================================
# CLEANUP
# ============================================================

hook.remove()

print()
print("=" * 80)
print("DIAGNOSTIC COMPLETE")
print("=" * 80)