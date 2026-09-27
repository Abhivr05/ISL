import random
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from models.tcn_model import TCNModel


# ============================================================
# CONFIGURATION
# ============================================================

RAW_DATASET = Path("raw_dataset")
INCLUDE_DATASET = Path("include_selected_dataset_v2")

OUTPUT_MODEL = Path(
    "trained_models/isl_tcn_include_experiment.pth"
)

SEQUENCE_LENGTH = 30
FEATURE_DIMENSION = 225

BATCH_SIZE = 16
EPOCHS = 50
LEARNING_RATE = 0.0005

RANDOM_STATE = 42

VALIDATION_RATIO = 0.20

ORIGINAL_CLASSES = [
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

INCLUDE_CLASSES = [
    "HOUSE",
    "SCHOOL",
    "OFFICE",
    "HOSPITAL",
    "MARKET",
    "BOOK",
    "TELEPHONE",
    "BAG",
    "KEY",
    "TODAY",
    "TOMORROW",
    "YESTERDAY",
    "MORNING",
    "EVENING",
    "MONEY",
    "MEDICINE",
    "DEATH",
    "PEACE",
    "SPORT",
    "TECHNOLOGY",
]

CLASS_NAMES = ORIGINAL_CLASSES + INCLUDE_CLASSES

CLASS_TO_ID = {
    name: index
    for index, name in enumerate(CLASS_NAMES)
}


# ============================================================
# REPRODUCIBILITY
# ============================================================

random.seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)
torch.manual_seed(RANDOM_STATE)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(RANDOM_STATE)


# ============================================================
# DATASET
# ============================================================

class SequenceDataset(Dataset):

    def __init__(self, samples):

        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):

        path, label = self.samples[index]

        data = np.load(path).astype(np.float32)

        tensor = torch.tensor(
            data,
            dtype=torch.float32
        )

        label = torch.tensor(
            label,
            dtype=torch.long
        )

        return tensor, label


# ============================================================
# LOAD EXISTING DATASET
# ============================================================

def load_original_dataset():

    samples = []

    for class_name in ORIGINAL_CLASSES:

        class_dir = RAW_DATASET / class_name

        if not class_dir.exists():
            print(
                f"WARNING: Missing class folder: "
                f"{class_dir}"
            )
            continue

        files = sorted(
            class_dir.glob("*.npy")
        )

        label = CLASS_TO_ID[class_name]

        for file_path in files:
            samples.append(
                (file_path, label)
            )

    return samples


# ============================================================
# LOAD INCLUDE DATASET
# ============================================================

def load_include_dataset():

    samples = []

    for class_name in INCLUDE_CLASSES:

        class_dir = (
            INCLUDE_DATASET / class_name
        )

        if not class_dir.exists():
            print(
                f"WARNING: Missing INCLUDE folder: "
                f"{class_dir}"
            )
            continue

        files = sorted(
            class_dir.glob("*.npy")
        )

        label = CLASS_TO_ID[class_name]

        for file_path in files:
            samples.append(
                (file_path, label)
            )

    return samples


# ============================================================
# EXTRACT INCLUDE VIDEO ID
# ============================================================

def get_video_id(path):

    # Example:
    #
    # house_MVI_8750_01.npy
    #
    # -> house_MVI_8750

    name = path.stem

    if name.endswith("_01"):
        name = name[:-3]
    elif name.endswith("_02"):
        name = name[:-3]
    elif name.endswith("_03"):
        name = name[:-3]
    elif name.endswith("_04"):
        name = name[:-3]

    return name


# ============================================================
# VIDEO-AWARE INCLUDE SPLIT
# ============================================================

def split_include_dataset(samples):

    by_class = defaultdict(list)

    for path, label in samples:
        by_class[label].append(path)

    train_samples = []
    val_samples = []

    rng = random.Random(RANDOM_STATE)

    for label, files in by_class.items():

        # Group sequences by original video
        video_groups = defaultdict(list)

        for file_path in files:

            video_id = get_video_id(
                file_path
            )

            video_groups[video_id].append(
                file_path
            )

        video_ids = list(
            video_groups.keys()
        )

        rng.shuffle(video_ids)

        if len(video_ids) < 2:

            # Safety fallback
            split_index = max(
                1,
                int(
                    len(video_ids)
                    * (1 - VALIDATION_RATIO)
                )
            )

        else:

            split_index = int(
                len(video_ids)
                * (1 - VALIDATION_RATIO)
            )

            split_index = min(
                max(split_index, 1),
                len(video_ids) - 1
            )

        train_videos = video_ids[
            :split_index
        ]

        val_videos = video_ids[
            split_index:
        ]

        for video_id in train_videos:

            for path in video_groups[
                video_id
            ]:
                train_samples.append(
                    (path, label)
                )

        for video_id in val_videos:

            for path in video_groups[
                video_id
            ]:
                val_samples.append(
                    (path, label)
                )

    return train_samples, val_samples


# ============================================================
# ORIGINAL DATASET SPLIT
# ============================================================

def split_original_dataset(samples):

    by_class = defaultdict(list)

    for path, label in samples:
        by_class[label].append(path)

    train_samples = []
    val_samples = []

    rng = random.Random(RANDOM_STATE)

    for label, files in by_class.items():

        files = list(files)

        rng.shuffle(files)

        split_index = int(
            len(files)
            * (1 - VALIDATION_RATIO)
        )

        split_index = min(
            max(split_index, 1),
            len(files) - 1
        )

        train_files = files[
            :split_index
        ]

        val_files = files[
            split_index:
        ]

        for path in train_files:
            train_samples.append(
                (path, label)
            )

        for path in val_files:
            val_samples.append(
                (path, label)
            )

    return train_samples, val_samples


# ============================================================
# EVALUATION
# ============================================================

def evaluate(
    model,
    loader,
    device
):

    model.eval()

    correct = 0
    total = 0

    class_correct = defaultdict(int)
    class_total = defaultdict(int)

    with torch.no_grad():

        for x, y in loader:

            x = x.to(device)
            y = y.to(device)

            outputs = model(x)

            predictions = torch.argmax(
                outputs,
                dim=1
            )

            correct += (
                predictions == y
            ).sum().item()

            total += y.size(0)

            for true_label, prediction in zip(
                y.cpu().numpy(),
                predictions.cpu().numpy()
            ):

                class_total[int(true_label)] += 1

                if true_label == prediction:
                    class_correct[int(true_label)] += 1

    accuracy = (
        100 * correct / total
        if total > 0
        else 0
    )

    class_accuracy = {}

    for class_id in sorted(class_total):

        class_accuracy[
            CLASS_NAMES[class_id]
        ] = (
            100
            * class_correct[class_id]
            / class_total[class_id]
        )

    return accuracy, class_accuracy


# ============================================================
# TRAINING
# ============================================================

def main():

    print("=" * 60)
    print("INCLUDE TRAINING EXPERIMENT")
    print("=" * 60)

    print(
        f"\nClasses: {len(CLASS_NAMES)}"
    )

    print(
        f"Original classes: "
        f"{len(ORIGINAL_CLASSES)}"
    )

    print(
        f"INCLUDE classes: "
        f"{len(INCLUDE_CLASSES)}"
    )

    # --------------------------------------------------------
    # Load datasets
    # --------------------------------------------------------

    original_samples = (
        load_original_dataset()
    )

    include_samples = (
        load_include_dataset()
    )

    print(
        f"\nOriginal samples: "
        f"{len(original_samples)}"
    )

    print(
        f"INCLUDE samples: "
        f"{len(include_samples)}"
    )

    print(
        f"Total samples: "
        f"{len(original_samples) + len(include_samples)}"
    )

    # --------------------------------------------------------
    # Split separately
    # --------------------------------------------------------

    (
        original_train,
        original_val
    ) = split_original_dataset(
        original_samples
    )

    (
        include_train,
        include_val
    ) = split_include_dataset(
        include_samples
    )

    train_samples = (
        original_train +
        include_train
    )

    val_samples = (
        original_val +
        include_val
    )

    random.shuffle(train_samples)
    random.shuffle(val_samples)

    print(
        f"\nTraining samples: "
        f"{len(train_samples)}"
    )

    print(
        f"Validation samples: "
        f"{len(val_samples)}"
    )

    print(
        "\nOriginal split:"
    )

    print(
        f"  Train: {len(original_train)}"
    )

    print(
        f"  Val  : {len(original_val)}"
    )

    print(
        "\nINCLUDE split:"
    )

    print(
        f"  Train: {len(include_train)}"
    )

    print(
        f"  Val  : {len(include_val)}"
    )

    # --------------------------------------------------------
    # DataLoaders
    # --------------------------------------------------------

    train_dataset = SequenceDataset(
        train_samples
    )

    val_dataset = SequenceDataset(
        val_samples
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
    # Device
    # --------------------------------------------------------

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(
        f"\nDevice: {device}"
    )

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    model = TCNModel(
        input_size=FEATURE_DIMENSION,
        num_classes=len(CLASS_NAMES)
    )

    model = model.to(device)

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE
    )

    # --------------------------------------------------------
    # Training
    # --------------------------------------------------------

    best_val_accuracy = 0.0

    best_state = None

    for epoch in range(1, EPOCHS + 1):

        model.train()

        running_loss = 0.0

        for x, y in train_loader:

            x = x.to(device)
            y = y.to(device)

            optimizer.zero_grad()

            outputs = model(x)

            loss = criterion(
                outputs,
                y
            )

            loss.backward()

            optimizer.step()

            running_loss += (
                loss.item()
                * x.size(0)
            )

        train_loss = (
            running_loss
            / len(train_dataset)
        )

        val_accuracy, _ = evaluate(
            model,
            val_loader,
            device
        )

        print(
            f"Epoch {epoch:02d}/{EPOCHS} | "
            f"Loss: {train_loss:.4f} | "
            f"Val Accuracy: "
            f"{val_accuracy:.2f}%"
        )

        if val_accuracy > best_val_accuracy:

            best_val_accuracy = (
                val_accuracy
            )

            best_state = {
                key: value.cpu().clone()
                for key, value
                in model.state_dict().items()
            }

    # --------------------------------------------------------
    # Restore best model
    # --------------------------------------------------------

    if best_state is not None:

        model.load_state_dict(
            best_state
        )

    # --------------------------------------------------------
    # Final evaluation
    # --------------------------------------------------------

    overall_accuracy, class_accuracy = (
        evaluate(
            model,
            val_loader,
            device
        )
    )

    print("\n" + "=" * 60)
    print("FINAL VALIDATION RESULTS")
    print("=" * 60)

    print(
        f"\nOverall accuracy: "
        f"{overall_accuracy:.2f}%"
    )

    print("\nOriginal classes:")

    original_values = []

    for class_name in ORIGINAL_CLASSES:

        if class_name in class_accuracy:

            accuracy = class_accuracy[
                class_name
            ]

            original_values.append(
                accuracy
            )

            print(
                f"  {class_name:12s}: "
                f"{accuracy:.2f}%"
            )

    if original_values:

        print(
            f"\nOriginal-class average: "
            f"{np.mean(original_values):.2f}%"
        )

    print("\nINCLUDE classes:")

    include_values = []

    for class_name in INCLUDE_CLASSES:

        if class_name in class_accuracy:

            accuracy = class_accuracy[
                class_name
            ]

            include_values.append(
                accuracy
            )

            print(
                f"  {class_name:12s}: "
                f"{accuracy:.2f}%"
            )

    if include_values:

        print(
            f"\nINCLUDE-class average: "
            f"{np.mean(include_values):.2f}%"
        )

    # --------------------------------------------------------
    # Save experiment model
    # --------------------------------------------------------

    OUTPUT_MODEL.parent.mkdir(
        parents=True,
        exist_ok=True
    )

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
                best_val_accuracy,
        },
        OUTPUT_MODEL
    )

    print(
        f"\nModel saved to:"
    )

    print(
        OUTPUT_MODEL.resolve()
    )

    print("\nExisting model was NOT modified.")


if __name__ == "__main__":
    main()