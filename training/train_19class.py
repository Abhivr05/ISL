import sys
import os

sys.path.append(
    os.path.dirname(
        os.path.dirname(
            os.path.abspath(__file__)
        )
    )
)

import random
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader

from models.tcn_model import TCNModel


# ============================================================
# Configuration
# ============================================================

RAW_DIR = "raw_dataset"
INCLUDE_DIR = "include_processed"

# Save as a NEW model
MODEL_PATH = "trained_models/isl_tcn_19class_v4.pth"

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

TARGET_PER_CLASS = 40

SEQUENCE_LENGTH = 30
INPUT_SIZE = 225

BATCH_SIZE = 16
EPOCHS = 60
LEARNING_RATE = 0.0005

TRAIN_RATIO = 0.8

# Reproducibility
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)


# ============================================================
# Augmentation
# ============================================================

def augment_sequence(sequence):
    """
    Apply small realistic variations to a sequence.

    Input:
        (30, 225)

    Output:
        (30, 225)
    """

    augmented = sequence.copy()

    # --------------------------------------------------------
    # Small landmark noise
    # --------------------------------------------------------

    noise = np.random.normal(
        0,
        0.008,
        augmented.shape
    ).astype(np.float32)

    augmented += noise

    # --------------------------------------------------------
    # Small global scale variation
    # --------------------------------------------------------

    scale = np.random.uniform(
        0.97,
        1.03
    )

    augmented *= scale

    # --------------------------------------------------------
    # Small temporal shift
    # --------------------------------------------------------

    shift = random.choice([
        -1,
        0,
        1
    ])

    if shift == 1:

        augmented = np.concatenate(
            [
                augmented[1:],
                augmented[-1:]
            ],
            axis=0
        )

    elif shift == -1:

        augmented = np.concatenate(
            [
                augmented[:1],
                augmented[:-1]
            ],
            axis=0
        )

    return augmented.astype(np.float32)


# ============================================================
# Load normal class data
# ============================================================

def load_class_data(class_name):

    samples = []

    # --------------------------------------------------------
    # Original webcam data
    # --------------------------------------------------------

    raw_folder = os.path.join(
        RAW_DIR,
        class_name
    )

    if os.path.exists(raw_folder):

        for filename in os.listdir(raw_folder):

            if not filename.endswith(".npy"):
                continue

            filepath = os.path.join(
                raw_folder,
                filename
            )

            data = np.load(filepath)

            if data.shape == (
                SEQUENCE_LENGTH,
                INPUT_SIZE
            ):

                samples.append(
                    data.astype(np.float32)
                )

    # --------------------------------------------------------
    # INCLUDE processed data
    # --------------------------------------------------------

    include_folder = os.path.join(
        INCLUDE_DIR,
        class_name
    )

    if os.path.exists(include_folder):

        for filename in os.listdir(include_folder):

            if not filename.endswith(".npy"):
                continue

            filepath = os.path.join(
                include_folder,
                filename
            )

            data = np.load(filepath)

            if data.shape == (
                SEQUENCE_LENGTH,
                INPUT_SIZE
            ):

                samples.append(
                    data.astype(np.float32)
                )

    return samples


# ============================================================
# Special YOU dataset
# ============================================================

def load_you_data():

    samples = []

    raw_folder = os.path.join(
        RAW_DIR,
        "YOU"
    )

    print()
    print("Special YOU selection:")
    print("  Original webcam : sample_001 - sample_038")
    print("  New webcam      : sample_039 - sample_050")
    print("  INCLUDE YOU     : excluded")
    print()

    # --------------------------------------------------------
    # Original webcam YOU samples
    # sample_001 -> sample_038
    # --------------------------------------------------------

    for i in range(1, 39):

        filename = f"sample_{i:03d}.npy"

        filepath = os.path.join(
            raw_folder,
            filename
        )

        if not os.path.exists(filepath):
            print(
                f"WARNING: Missing {filename}"
            )
            continue

        data = np.load(filepath)

        if data.shape == (
            SEQUENCE_LENGTH,
            INPUT_SIZE
        ):

            samples.append(
                data.astype(np.float32)
            )

    # --------------------------------------------------------
    # New webcam YOU samples
    # sample_039 -> sample_050
    # --------------------------------------------------------

    for i in range(39, 51):

        filename = f"sample_{i:03d}.npy"

        filepath = os.path.join(
            raw_folder,
            filename
        )

        if not os.path.exists(filepath):
            print(
                f"WARNING: Missing {filename}"
            )
            continue

        data = np.load(filepath)

        if data.shape == (
            SEQUENCE_LENGTH,
            INPUT_SIZE
        ):

            samples.append(
                data.astype(np.float32)
            )

    return samples


# ============================================================
# Build balanced dataset
# ============================================================

all_samples = []
all_labels = []

print("=" * 60)
print("BUILDING 19-CLASS DATASET - V3")
print("=" * 60)

for label, class_name in enumerate(CLASSES):

    # --------------------------------------------------------
    # YOU uses controlled webcam-only selection
    # --------------------------------------------------------

    if class_name == "YOU":

        samples = load_you_data()

    else:

        samples = load_class_data(
            class_name
        )

    original_count = len(samples)

    if original_count == 0:

        raise RuntimeError(
            f"No data found for class: {class_name}"
        )

    # --------------------------------------------------------
    # Shuffle the real samples
    # --------------------------------------------------------

    random.shuffle(samples)

    # --------------------------------------------------------
    # Keep ALL real samples
    #
    # Only classes below TARGET_PER_CLASS
    # will receive augmentation.
    # --------------------------------------------------------

    while len(samples) < TARGET_PER_CLASS:

        source = random.choice(
            samples
        )

        augmented = augment_sequence(
            source
        )

        samples.append(
            augmented
        )

    # --------------------------------------------------------
    # Print class information
    # --------------------------------------------------------

    augmented_count = (
        len(samples) - original_count
    )

    print(
        f"{class_name:12s} "
        f"original={original_count:3d} "
        f"augmented={augmented_count:3d} "
        f"final={len(samples):3d}"
    )

    # --------------------------------------------------------
    # Add to complete dataset
    # --------------------------------------------------------

    all_samples.extend(
        samples
    )

    all_labels.extend(
        [label] * len(samples)
    )


# ============================================================
# Convert to NumPy arrays
# ============================================================

X = np.array(
    all_samples,
    dtype=np.float32
)

y = np.array(
    all_labels,
    dtype=np.int64
)

print("=" * 60)

print(
    f"FINAL DATASET SHAPE: {X.shape}"
)

print(
    f"LABEL SHAPE:         {y.shape}"
)

print("=" * 60)


# ============================================================
# Shuffle dataset
# ============================================================

indices = np.arange(
    len(X)
)

np.random.shuffle(
    indices
)

X = X[indices]
y = y[indices]


# ============================================================
# Train / validation split
# ============================================================

split_index = int(
    len(X) * TRAIN_RATIO
)

X_train = X[:split_index]
y_train = y[:split_index]

X_val = X[split_index:]
y_val = y[split_index:]

print(
    f"Training samples:   {len(X_train)}"
)

print(
    f"Validation samples: {len(X_val)}"
)


# ============================================================
# PyTorch datasets
# ============================================================

train_dataset = TensorDataset(
    torch.tensor(X_train),
    torch.tensor(y_train)
)

val_dataset = TensorDataset(
    torch.tensor(X_val),
    torch.tensor(y_val)
)


# ============================================================
# DataLoaders
# ============================================================

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)


# ============================================================
# Model
# ============================================================

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

print(
    f"Device: {device}"
)

model = TCNModel(
    input_size=INPUT_SIZE,
    num_classes=len(CLASSES)
)

model = model.to(
    device
)


# ============================================================
# Training setup
# ============================================================

criterion = torch.nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
)


# ============================================================
# Training
# ============================================================

best_val_accuracy = 0.0

os.makedirs(
    os.path.dirname(MODEL_PATH),
    exist_ok=True
)


for epoch in range(EPOCHS):

    # ========================================================
    # Training
    # ========================================================

    model.train()

    train_correct = 0
    train_total = 0
    train_loss_total = 0.0

    for batch_X, batch_y in train_loader:

        batch_X = batch_X.to(
            device
        )

        batch_y = batch_y.to(
            device
        )

        optimizer.zero_grad()

        outputs = model(
            batch_X
        )

        loss = criterion(
            outputs,
            batch_y
        )

        loss.backward()

        optimizer.step()

        # ----------------------------------------------------
        # Loss
        # ----------------------------------------------------

        train_loss_total += (
            loss.item()
            * batch_X.size(0)
        )

        # ----------------------------------------------------
        # Accuracy
        # ----------------------------------------------------

        predictions = outputs.argmax(
            dim=1
        )

        train_correct += (
            predictions == batch_y
        ).sum().item()

        train_total += (
            batch_y.size(0)
        )

    train_loss = (
        train_loss_total
        / train_total
    )

    train_accuracy = (
        train_correct
        / train_total
    ) * 100


    # ========================================================
    # Validation
    # ========================================================

    model.eval()

    val_correct = 0
    val_total = 0
    val_loss_total = 0.0

    with torch.no_grad():

        for batch_X, batch_y in val_loader:

            batch_X = batch_X.to(
                device
            )

            batch_y = batch_y.to(
                device
            )

            outputs = model(
                batch_X
            )

            loss = criterion(
                outputs,
                batch_y
            )

            val_loss_total += (
                loss.item()
                * batch_X.size(0)
            )

            predictions = outputs.argmax(
                dim=1
            )

            val_correct += (
                predictions == batch_y
            ).sum().item()

            val_total += (
                batch_y.size(0)
            )

    val_loss = (
        val_loss_total
        / val_total
    )

    val_accuracy = (
        val_correct
        / val_total
    ) * 100


    # ========================================================
    # Print epoch result
    # ========================================================

    print(
        f"Epoch {epoch + 1:02d}/{EPOCHS} | "
        f"Train Loss: {train_loss:.4f} | "
        f"Train Acc: {train_accuracy:.2f}% | "
        f"Val Loss: {val_loss:.4f} | "
        f"Val Acc: {val_accuracy:.2f}%"
    )


    # ========================================================
    # Save best model
    # ========================================================

    if val_accuracy > best_val_accuracy:

        best_val_accuracy = val_accuracy

        torch.save(
            {
                "classes": CLASSES,
                "input_size": INPUT_SIZE,
                "sequence_length": SEQUENCE_LENGTH,
                "model_state_dict": model.state_dict(),
                "best_val_accuracy": best_val_accuracy,
            },
            MODEL_PATH
        )

        print(
            f"  -> Best model saved "
            f"({best_val_accuracy:.2f}%)"
        )


# ============================================================
# Finished
# ============================================================

print("=" * 60)
print("TRAINING COMPLETE")
print("=" * 60)

print(
    f"Best validation accuracy: "
    f"{best_val_accuracy:.2f}%"
)

print(
    f"Model saved to: "
    f"{MODEL_PATH}"
)

print("=" * 60)