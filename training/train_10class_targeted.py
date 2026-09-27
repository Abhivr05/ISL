import os
import sys
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split

# Add project root to Python path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)

from models.tcn_model import TCNModel


# ============================================================
# Configuration
# ============================================================

RAW_DATASET = os.path.join(PROJECT_ROOT, "raw_dataset")
INCLUDE_YOU = os.path.join(PROJECT_ROOT, "include_processed", "YOU")

OUTPUT_MODEL = os.path.join(
    PROJECT_ROOT,
    "trained_models",
    "isl_tcn_targeted.pth"
)

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
    "YOU"
]

SEQUENCE_LENGTH = 30
FEATURES = 225

TARGET_PER_CLASS = 50

BATCH_SIZE = 16
EPOCHS = 60
LEARNING_RATE = 0.0005

SEED = 42


# ============================================================
# Reproducibility
# ============================================================

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


# ============================================================
# Augmentation
# ============================================================

def augment_sequence(sequence):
    """
    Small controlled augmentation.
    Keeps the original sign structure intact.
    """

    augmented = sequence.copy()

    # Small landmark noise
    noise = np.random.normal(
        0,
        0.008,
        augmented.shape
    ).astype(np.float32)

    augmented += noise

    # Small global scale change
    scale = np.random.uniform(0.97, 1.03)
    augmented *= scale

    # Small temporal shift
    shift = random.choice([-1, 0, 1])

    if shift == 1:
        augmented = np.concatenate(
            [augmented[:1], augmented[:-1]],
            axis=0
        )

    elif shift == -1:
        augmented = np.concatenate(
            [augmented[1:], augmented[-1:]],
            axis=0
        )

    return augmented.astype(np.float32)


# ============================================================
# Load raw dataset
# ============================================================

def load_class_data(class_name):

    class_dir = os.path.join(
        RAW_DATASET,
        class_name
    )

    sequences = []

    files = sorted([
        f for f in os.listdir(class_dir)
        if f.endswith(".npy")
    ])

    for filename in files:

        path = os.path.join(
            class_dir,
            filename
        )

        sequence = np.load(path)

        if sequence.shape == (30, 225):
            sequences.append(sequence.astype(np.float32))
        else:
            print(
                f"Skipping invalid file: {path} "
                f"shape={sequence.shape}"
            )

    return sequences


# ============================================================
# Main dataset preparation
# ============================================================

all_sequences = []
all_labels = []


for class_id, class_name in enumerate(CLASSES):

    sequences = load_class_data(class_name)

    print(
        f"{class_name:10s}: "
        f"{len(sequences)} original samples"
    )

    # --------------------------------------------------------
    # Add INCLUDE YOU samples
    # --------------------------------------------------------

    if class_name == "YOU":

        include_files = sorted([
            f for f in os.listdir(INCLUDE_YOU)
            if f.endswith(".npy")
        ])

        for filename in include_files:

            path = os.path.join(
                INCLUDE_YOU,
                filename
            )

            sequence = np.load(path)

            if sequence.shape == (30, 225):
                sequences.append(
                    sequence.astype(np.float32)
                )

        print(
            f"{class_name:10s}: "
            f"{len(sequences)} total after INCLUDE"
        )

    # --------------------------------------------------------
    # Keep at least TARGET_PER_CLASS samples
    # --------------------------------------------------------

    if len(sequences) > TARGET_PER_CLASS:

        # Keep all original data rather than throwing away
        # useful samples.
        selected = sequences

    else:

        selected = sequences.copy()

        while len(selected) < TARGET_PER_CLASS:

            base = random.choice(sequences)

            selected.append(
                augment_sequence(base)
            )

    print(
        f"{class_name:10s}: "
        f"{len(selected)} samples used"
    )

    for sequence in selected:

        all_sequences.append(sequence)
        all_labels.append(class_id)


# ============================================================
# Convert to arrays
# ============================================================

X = np.asarray(
    all_sequences,
    dtype=np.float32
)

y = np.asarray(
    all_labels,
    dtype=np.int64
)

print("\nFinal dataset:")
print("X shape:", X.shape)
print("y shape:", y.shape)


# ============================================================
# Train / validation split
# ============================================================

X_train, X_val, y_train, y_val = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=SEED,
    stratify=y
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
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("\nDevice:", device)

model = TCNModel(
    input_size=FEATURES,
    num_classes=len(CLASSES)
)

model = model.to(device)


# ============================================================
# Training
# ============================================================

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
)


best_val_accuracy = 0.0


for epoch in range(EPOCHS):

    # --------------------------------------------------------
    # Training
    # --------------------------------------------------------

    model.train()

    correct = 0
    total = 0
    train_loss = 0.0

    for inputs, labels in train_loader:

        inputs = inputs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(inputs)

        loss = criterion(
            outputs,
            labels
        )

        loss.backward()
        optimizer.step()

        train_loss += loss.item()

        predictions = torch.argmax(
            outputs,
            dim=1
        )

        correct += (
            predictions == labels
        ).sum().item()

        total += labels.size(0)

    train_accuracy = (
        100.0 * correct / total
    )

    # --------------------------------------------------------
    # Validation
    # --------------------------------------------------------

    model.eval()

    correct = 0
    total = 0

    with torch.no_grad():

        for inputs, labels in val_loader:

            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)

            predictions = torch.argmax(
                outputs,
                dim=1
            )

            correct += (
                predictions == labels
            ).sum().item()

            total += labels.size(0)

    val_accuracy = (
        100.0 * correct / total
    )

    print(
        f"Epoch {epoch + 1:02d}/{EPOCHS} | "
        f"Train: {train_accuracy:.2f}% | "
        f"Val: {val_accuracy:.2f}%"
    )

    # --------------------------------------------------------
    # Save best model
    # --------------------------------------------------------

    if val_accuracy > best_val_accuracy:

        best_val_accuracy = val_accuracy

        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "classes": CLASSES,
                "input_size": FEATURES,
                "sequence_length": SEQUENCE_LENGTH
            },
            OUTPUT_MODEL
        )

        print(
            f"  -> Best model saved "
            f"({best_val_accuracy:.2f}%)"
        )


print("\n" + "=" * 50)
print("TARGETED TRAINING COMPLETE")
print(f"Best validation accuracy: {best_val_accuracy:.2f}%")
print(f"Model saved: {OUTPUT_MODEL}")
print("=" * 50)