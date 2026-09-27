import os
import numpy as np
import torch

from models.tcn_model import TCNModel


OLD_MODEL = "trained_models/isl_tcn.pth"
NEW_MODEL = "trained_models/isl_tcn_include_experiment.pth"

SEQUENCE_DIRS = [
    "debug_labeled_sequences",
    "debug_sequences",
]


def load_model(path):
    checkpoint = torch.load(path, map_location="cpu")

    class_names = checkpoint["class_names"]
    input_size = checkpoint["input_size"]
    num_classes = checkpoint["num_classes"]

    model = TCNModel(
        input_size=input_size,
        num_classes=num_classes
    )

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    return model, class_names


def predict(model, class_names, sequence):
    sequence = np.asarray(sequence, dtype=np.float32)

    if sequence.shape != (30, 225):
        raise ValueError(
            f"Expected (30, 225), got {sequence.shape}"
        )

    tensor = torch.tensor(sequence).unsqueeze(0)

    with torch.no_grad():
        output = model(tensor)
        probabilities = torch.softmax(output, dim=1)

    confidence, index = torch.max(probabilities, dim=1)

    return (
        class_names[index.item()],
        confidence.item()
    )


def get_true_label(filename, folder):
    # Labeled sequences are stored with names such as:
    # YES_001.npy
    # NO_002.npy

    name = os.path.splitext(filename)[0]

    if name.startswith("YES_"):
        return "YES"

    if name.startswith("NO_"):
        return "NO"

    # No reliable ground truth for the other debug sequences
    return None


def test_directory(directory, old_model, old_classes, new_model, new_classes):
    if not os.path.exists(directory):
        print(f"\nDirectory not found: {directory}")
        return

    files = sorted(
        f for f in os.listdir(directory)
        if f.endswith(".npy")
    )

    print("\n" + "=" * 90)
    print(f"TESTING: {directory}")
    print("=" * 90)

    old_correct = 0
    new_correct = 0
    labeled_count = 0

    for filename in files:
        path = os.path.join(directory, filename)

        try:
            sequence = np.load(path)

            old_prediction, old_confidence = predict(
                old_model,
                old_classes,
                sequence
            )

            new_prediction, new_confidence = predict(
                new_model,
                new_classes,
                sequence
            )

            true_label = get_true_label(filename, directory)

            if true_label is not None:
                labeled_count += 1

                old_ok = old_prediction == true_label
                new_ok = new_prediction == true_label

                if old_ok:
                    old_correct += 1

                if new_ok:
                    new_correct += 1

                old_mark = "✓" if old_ok else "✗"
                new_mark = "✓" if new_ok else "✗"

                print(
                    f"{filename:25s} "
                    f"TRUE={true_label:8s} | "
                    f"OLD={old_prediction:12s} "
                    f"{old_confidence:.3f} {old_mark} | "
                    f"NEW={new_prediction:12s} "
                    f"{new_confidence:.3f} {new_mark}"
                )

            else:
                agreement = (
                    "SAME"
                    if old_prediction == new_prediction
                    else "DIFF"
                )

                print(
                    f"{filename:25s} "
                    f"OLD={old_prediction:12s} "
                    f"{old_confidence:.3f} | "
                    f"NEW={new_prediction:12s} "
                    f"{new_confidence:.3f} | "
                    f"{agreement}"
                )

        except Exception as e:
            print(f"{filename}: ERROR - {e}")

    if labeled_count > 0:
        print("\nLabeled accuracy:")
        print(
            f"Old model: {old_correct}/{labeled_count} "
            f"= {old_correct / labeled_count * 100:.2f}%"
        )
        print(
            f"New model: {new_correct}/{labeled_count} "
            f"= {new_correct / labeled_count * 100:.2f}%"
        )


def main():

    print("Loading old model...")
    old_model, old_classes = load_model(OLD_MODEL)

    print("Old classes:")
    print(old_classes)

    print("\nLoading new INCLUDE model...")
    new_model, new_classes = load_model(NEW_MODEL)

    print("New classes:")
    print(new_classes)

    print("\nModels loaded successfully.")

    for directory in SEQUENCE_DIRS:
        test_directory(
            directory,
            old_model,
            old_classes,
            new_model,
            new_classes
        )


if __name__ == "__main__":
    main()