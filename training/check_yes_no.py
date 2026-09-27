import sys
import os

# Allow importing project modules
sys.path.append(
    os.path.dirname(
        os.path.dirname(
            os.path.abspath(__file__)
        )
    )
)

import numpy as np
import torch

from models.tcn_model import TCNModel


# ============================================================
# Configuration
# ============================================================

MODEL_PATH = "trained_models/isl_tcn_19class_v3.pth"

RAW_DIR = "raw_dataset"

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

SEQUENCE_LENGTH = 30
INPUT_SIZE = 225

# ============================================================
# Load model
# ============================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("=" * 60)
print("YES / NO DIAGNOSTIC")
print("=" * 60)

print(f"Model : {MODEL_PATH}")
print(f"Device: {device}")
print()


checkpoint = torch.load(
    MODEL_PATH,
    map_location=device
)

model = TCNModel(
    input_size=checkpoint["input_size"],
    num_classes=len(checkpoint["classes"])
)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

model = model.to(device)
model.eval()

print(
    f"Model validation accuracy: "
    f"{checkpoint.get('best_val_accuracy', 'N/A'):.2f}%"
)

print()


# ============================================================
# Find class indexes
# ============================================================

NO_INDEX = CLASSES.index("NO")
YES_INDEX = CLASSES.index("YES")

print(f"NO  class index: {NO_INDEX}")
print(f"YES class index: {YES_INDEX}")
print()


# ============================================================
# Evaluate one class
# ============================================================

def evaluate_class(class_name):

    folder = os.path.join(
        RAW_DIR,
        class_name
    )

    if not os.path.exists(folder):

        print(
            f"ERROR: Folder not found: {folder}"
        )

        return

    files = sorted(
        [
            f
            for f in os.listdir(folder)
            if f.endswith(".npy")
        ]
    )

    print("=" * 60)
    print(f"ACTUAL CLASS: {class_name}")
    print("=" * 60)

    if len(files) == 0:

        print("No .npy files found.")
        return

    correct = 0
    total = 0

    predictions = {}

    confidence_sum = 0.0

    for filename in files:

        filepath = os.path.join(
            folder,
            filename
        )

        data = np.load(
            filepath
        ).astype(np.float32)

        if data.shape != (
            SEQUENCE_LENGTH,
            INPUT_SIZE
        ):

            print(
                f"Skipping {filename}: "
                f"shape={data.shape}"
            )

            continue

        # ----------------------------------------------------
        # Convert to tensor
        # Shape:
        # (30, 225)
        # ->
        # (1, 30, 225)
        # ----------------------------------------------------

        input_tensor = torch.tensor(
            data,
            dtype=torch.float32
        ).unsqueeze(0).to(device)

        # ----------------------------------------------------
        # Prediction
        # ----------------------------------------------------

        with torch.no_grad():

            outputs = model(
                input_tensor
            )

            probabilities = torch.softmax(
                outputs,
                dim=1
            )

        confidence, prediction = torch.max(
            probabilities,
            dim=1
        )

        prediction_index = (
            prediction.item()
        )

        confidence_value = (
            confidence.item()
        )

        predicted_class = (
            CLASSES[prediction_index]
        )

        confidence_sum += confidence_value

        total += 1

        if predicted_class == class_name:

            correct += 1

        if predicted_class not in predictions:

            predictions[predicted_class] = 0

        predictions[predicted_class] += 1

        # ----------------------------------------------------
        # Print YES / NO related predictions
        # ----------------------------------------------------

        print(
            f"{filename:16s} "
            f"-> {predicted_class:12s} "
            f"confidence={confidence_value:.3f}"
        )

    # ========================================================
    # Summary
    # ========================================================

    accuracy = (
        correct / total * 100
        if total > 0
        else 0
    )

    average_confidence = (
        confidence_sum / total
        if total > 0
        else 0
    )

    print()
    print(
        f"Correct: {correct}/{total}"
    )

    print(
        f"Accuracy: {accuracy:.2f}%"
    )

    print(
        f"Average confidence: "
        f"{average_confidence:.3f}"
    )

    print()
    print("Prediction distribution:")

    for prediction_class, count in sorted(
        predictions.items(),
        key=lambda x: x[1],
        reverse=True
    ):

        percentage = (
            count / total * 100
        )

        print(
            f"  {prediction_class:12s}: "
            f"{count:3d} "
            f"({percentage:5.1f}%)"
        )

    print()


# ============================================================
# Run diagnostic
# ============================================================

evaluate_class("NO")
evaluate_class("YES")


# ============================================================
# Final comparison
# ============================================================

print("=" * 60)
print("DIAGNOSTIC COMPLETE")
print("=" * 60)

print()
print("Interpretation:")
print()
print("If NO samples are mostly predicted as YES:")
print("  -> The model has a NO/YES class confusion.")
print()
print("If NO and YES are correctly classified here")
print("but fail during webcam testing:")
print("  -> The problem is more likely live capture,")
print("     lighting, landmark quality, or realtime logic.")
print()