
import os
import sys
import random
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

# Add project root to Python path
PROJECT_ROOT = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

sys.path.insert(0, PROJECT_ROOT)

from models.tcn_model import TCNModel


# ============================================================
# SETTINGS
# ============================================================

DATASET_DIR = "raw_dataset"

AUGMENTED_DIR = "augmented_dataset_10class_v2"

MODEL_PATH = "trained_models/isl_tcn_10class_v2.pth"

SEQUENCE_LENGTH = 30
FEATURE_DIMENSION = 225

BATCH_SIZE = 16
EPOCHS = 60
LEARNING_RATE = 0.0005

SEED = 42

# Target number of training samples per class
TARGET_PER_CLASS = 60


# ============================================================
# REPRODUCIBILITY
# ============================================================

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


# ============================================================
# CLASSES
# ============================================================

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
]

CLASS_TO_INDEX = {
    name: i
    for i, name in enumerate(CLASS_NAMES)
}


# ============================================================
# AUGMENTATION
# ============================================================

def augment_sequence(sequence):

    augmented = sequence.copy()

    # --------------------------------------------------------
    # 1. Small landmark noise
    # --------------------------------------------------------

    noise = np.random.normal(
        loc=0.0,
        scale=0.008,
        size=augmented.shape
    ).astype(np.float32)

    augmented += noise

    # --------------------------------------------------------
    # 2. Small global scale variation
    # --------------------------------------------------------

    scale = np.random.uniform(
        0.97,
        1.03
    )

    augmented *= scale

    # --------------------------------------------------------
    # 3. Small temporal shift
    # --------------------------------------------------------

    shift = random.choice(
        [-1, 0, 0, 0, 1]
    )

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
# LOAD ORIGINAL DATA
# ============================================================

def load_original_dataset():

    data = []
    labels = []

    print("\nLoading original dataset...")

    for class_name in CLASS_NAMES:

        folder = os.path.join(
            DATASET_DIR,
            class_name
        )

        if not os.path.exists(folder):

            print(
                f"WARNING: Missing folder: {folder}"
            )

            continue

        files = sorted(
            f for f in os.listdir(folder)
            if f.endswith(".npy")
        )

        class_index = CLASS_TO_INDEX[
            class_name
        ]

        for file in files:

            path = os.path.join(
                folder,
                file
            )

            sequence = np.load(path)

            if sequence.shape != (
                SEQUENCE_LENGTH,
                FEATURE_DIMENSION
            ):

                print(
                    f"Skipping bad shape: {path} "
                    f"{sequence.shape}"
                )

                continue

            data.append(
                sequence.astype(np.float32)
            )

            labels.append(
                class_index
            )

        print(
            f"{class_name:12s}: {len(files)} samples"
        )

    return (
        np.array(data, dtype=np.float32),
        np.array(labels, dtype=np.int64)
    )


# ============================================================
# CREATE BALANCED AUGMENTED DATASET
# ============================================================

def create_augmented_dataset(
    original_data,
    original_labels
):

    print(
        "\nCreating balanced augmented dataset..."
    )

    os.makedirs(
        AUGMENTED_DIR,
        exist_ok=True
    )

    for class_name in CLASS_NAMES:

        os.makedirs(
            os.path.join(
                AUGMENTED_DIR,
                class_name
            ),
            exist_ok=True
        )

    for class_index, class_name in enumerate(
        CLASS_NAMES
    ):

        class_data = original_data[
            original_labels == class_index
        ]

        folder = os.path.join(
            AUGMENTED_DIR,
            class_name
        )

        # ----------------------------------------------------
        # Save original samples
        # ----------------------------------------------------

        for i, sequence in enumerate(
            class_data
        ):

            path = os.path.join(
                folder,
                f"{class_name}_original_{i:03d}.npy"
            )

            np.save(
                path,
                sequence
            )

        current_count = len(
            class_data
        )

        # ----------------------------------------------------
        # Create augmented samples
        # ----------------------------------------------------

        augmentation_count = max(
            0,
            TARGET_PER_CLASS - current_count
        )

        for i in range(
            augmentation_count
        ):

            source_index = random.randrange(
                current_count
            )

            source_sequence = (
                class_data[source_index]
            )

            augmented = augment_sequence(
                source_sequence
            )

            path = os.path.join(
                folder,
                f"{class_name}_aug_{i:03d}.npy"
            )

            np.save(
                path,
                augmented
            )

        print(
            f"{class_name:12s}: "
            f"{current_count} original + "
            f"{augmentation_count} augmented = "
            f"{TARGET_PER_CLASS}"
        )


# ============================================================
# DATASET CLASS
# ============================================================

class SignDataset(Dataset):

    def __init__(
        self,
        data,
        labels
    ):

        self.data = torch.tensor(
            data,
            dtype=torch.float32
        )

        self.labels = torch.tensor(
            labels,
            dtype=torch.long
        )

    def __len__(self):

        return len(self.data)

    def __getitem__(
        self,
        index
    ):

        return (
            self.data[index],
            self.labels[index]
        )


# ============================================================
# LOAD AUGMENTED DATASET
# ============================================================

def load_augmented_dataset():

    data = []
    labels = []

    print(
        "\nLoading augmented dataset..."
    )

    for class_index, class_name in enumerate(
        CLASS_NAMES
    ):

        folder = os.path.join(
            AUGMENTED_DIR,
            class_name
        )

        files = sorted(
            f for f in os.listdir(folder)
            if f.endswith(".npy")
        )

        for file in files:

            path = os.path.join(
                folder,
                file
            )

            sequence = np.load(
                path
            )

            if sequence.shape != (
                SEQUENCE_LENGTH,
                FEATURE_DIMENSION
            ):
                continue

            data.append(
                sequence.astype(np.float32)
            )

            labels.append(
                class_index
            )

    return (
        np.array(data, dtype=np.float32),
        np.array(labels, dtype=np.int64)
    )


# ============================================================
# TRAINING
# ============================================================

def main():

    print("=" * 70)
    print("10-CLASS TCN V2 TRAINING")
    print("=" * 70)

    # --------------------------------------------------------
    # Load original
    # --------------------------------------------------------

    original_data, original_labels = (
        load_original_dataset()
    )

    print(
        f"\nOriginal samples: "
        f"{len(original_data)}"
    )

    # --------------------------------------------------------
    # Create augmentation
    # --------------------------------------------------------

    create_augmented_dataset(
        original_data,
        original_labels
    )

    # --------------------------------------------------------
    # Load augmented dataset
    # --------------------------------------------------------

    data, labels = (
        load_augmented_dataset()
    )

    print(
        f"\nTotal samples: {len(data)}"
    )

    # --------------------------------------------------------
    # Split
    # --------------------------------------------------------

    train_data, val_data, train_labels, val_labels = (
        train_test_split(
            data,
            labels,
            test_size=0.20,
            random_state=SEED,
            stratify=labels
        )
    )

    print(
        f"Training samples: {len(train_data)}"
    )

    print(
        f"Validation samples: {len(val_data)}"
    )

    # --------------------------------------------------------
    # Datasets
    # --------------------------------------------------------

    train_dataset = SignDataset(
        train_data,
        train_labels
    )

    val_dataset = SignDataset(
        val_data,
        val_labels
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

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    model = TCNModel(
        input_size=FEATURE_DIMENSION,
        num_classes=len(CLASS_NAMES)
    )

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    model = model.to(device)

    print(
        f"\nDevice: {device}"
    )

    # --------------------------------------------------------
    # Loss / optimizer
    # --------------------------------------------------------

    criterion = torch.nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE
    )

    # --------------------------------------------------------
    # Best model tracking
    # --------------------------------------------------------

    best_accuracy = 0.0

    os.makedirs(
        os.path.dirname(MODEL_PATH),
        exist_ok=True
    )

    # ========================================================
    # EPOCH LOOP
    # ========================================================

    for epoch in range(
        EPOCHS
    ):

        model.train()

        correct = 0
        total = 0

        running_loss = 0.0

        for sequences, targets in train_loader:

            sequences = sequences.to(
                device
            )

            targets = targets.to(
                device
            )

            optimizer.zero_grad()

            outputs = model(
                sequences
            )

            loss = criterion(
                outputs,
                targets
            )

            loss.backward()

            optimizer.step()

            running_loss += (
                loss.item()
                * sequences.size(0)
            )

            predictions = torch.argmax(
                outputs,
                dim=1
            )

            correct += (
                predictions == targets
            ).sum().item()

            total += targets.size(0)

        train_accuracy = (
            correct / total
        ) * 100

        # ----------------------------------------------------
        # Validation
        # ----------------------------------------------------

        model.eval()

        val_correct = 0
        val_total = 0

        class_correct = np.zeros(
            len(CLASS_NAMES),
            dtype=int
        )

        class_total = np.zeros(
            len(CLASS_NAMES),
            dtype=int
        )

        with torch.no_grad():

            for sequences, targets in val_loader:

                sequences = sequences.to(
                    device
                )

                targets = targets.to(
                    device
                )

                outputs = model(
                    sequences
                )

                predictions = torch.argmax(
                    outputs,
                    dim=1
                )

                val_correct += (
                    predictions == targets
                ).sum().item()

                val_total += (
                    targets.size(0)
                )

                for target, prediction in zip(
                    targets.cpu().numpy(),
                    predictions.cpu().numpy()
                ):

                    class_total[target] += 1

                    if target == prediction:

                        class_correct[target] += 1

        val_accuracy = (
            val_correct / val_total
        ) * 100

        avg_loss = (
            running_loss / total
        )

        print(
            f"Epoch {epoch + 1:02d}/{EPOCHS} | "
            f"Loss {avg_loss:.4f} | "
            f"Train {train_accuracy:.2f}% | "
            f"Val {val_accuracy:.2f}%"
        )

        # ----------------------------------------------------
        # Save best model
        # ----------------------------------------------------

        if val_accuracy > best_accuracy:

            best_accuracy = val_accuracy

            torch.save(
                {
                    "model_state_dict":
                        model.state_dict(),

                    "class_names":
                        CLASS_NAMES,

                    "input_size":
                        FEATURE_DIMENSION,

                    "sequence_length":
                        SEQUENCE_LENGTH,

                    "num_classes":
                        len(CLASS_NAMES),

                    "best_validation_accuracy":
                        best_accuracy,
                },
                MODEL_PATH
            )

            print(
                f"  → Best model saved "
                f"({best_accuracy:.2f}%)"
            )

    # ========================================================
    # FINAL EVALUATION
    # ========================================================

    print("\n" + "=" * 70)
    print("FINAL RESULT")
    print("=" * 70)

    print(
        f"Best validation accuracy: "
        f"{best_accuracy:.2f}%"
    )

    # --------------------------------------------------------
    # Load best checkpoint
    # --------------------------------------------------------

    checkpoint = torch.load(
        MODEL_PATH,
        map_location=device
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.eval()

    # --------------------------------------------------------
    # Per-class accuracy
    # --------------------------------------------------------

    class_correct = np.zeros(
        len(CLASS_NAMES),
        dtype=int
    )

    class_total = np.zeros(
        len(CLASS_NAMES),
        dtype=int
    )

    with torch.no_grad():

        for sequences, targets in val_loader:

            sequences = sequences.to(
                device
            )

            outputs = model(
                sequences
            )

            predictions = torch.argmax(
                outputs,
                dim=1
            )

            for target, prediction in zip(
                targets.numpy(),
                predictions.cpu().numpy()
            ):

                class_total[target] += 1

                if target == prediction:

                    class_correct[target] += 1

    print("\nPer-class validation accuracy:")

    for i, class_name in enumerate(
        CLASS_NAMES
    ):

        if class_total[i] > 0:

            accuracy = (
                class_correct[i]
                / class_total[i]
            ) * 100

        else:

            accuracy = 0.0

        print(
            f"{class_name:12s}: "
            f"{accuracy:.2f}% "
            f"({class_correct[i]}/{class_total[i]})"
        )

    print(
        f"\nModel saved to:"
        f"\n{MODEL_PATH}"
    )


if __name__ == "__main__":
    main()

