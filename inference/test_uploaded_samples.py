import os
import numpy as np
import torch

from models.tcn_model import TCNModel


# --------------------------------------------------
# Configuration
# --------------------------------------------------

MODEL_PATH = "trained_models/isl_tcn.pth"

SAMPLE_FILES = [
    ("I", "sample_006.npy"),
    ("I", "sample_007.npy"),
    ("YOU", "sample_009.npy"),
    ("YOU", "sample_010.npy"),
    ("YOU", "sample_011.npy"),
]


# --------------------------------------------------
# Device
# --------------------------------------------------

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print(f"Using device: {device}")


# --------------------------------------------------
# Load checkpoint
# --------------------------------------------------

checkpoint = torch.load(
    MODEL_PATH,
    map_location=device
)

class_names = checkpoint["class_names"]

print("\nClasses:")
for index, name in enumerate(class_names):
    print(f"{index}: {name}")


# --------------------------------------------------
# Create model
# --------------------------------------------------

model = TCNModel(
    input_size=checkpoint["input_size"],
    num_classes=checkpoint["num_classes"]
)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

model = model.to(device)
model.eval()


# --------------------------------------------------
# Test samples
# --------------------------------------------------

print("\n===== SAMPLE PREDICTIONS =====")

for actual_class, filename in SAMPLE_FILES:

    if not os.path.exists(filename):
        print(f"\nFile not found: {filename}")
        continue

    sequence = np.load(filename)

    print(f"\nActual class: {actual_class}")
    print(f"File: {filename}")
    print(f"Shape: {sequence.shape}")

    if sequence.shape != (30, 225):
        print("ERROR: Invalid shape")
        continue

    # NumPy → PyTorch
    input_tensor = torch.tensor(
        sequence,
        dtype=torch.float32
    )

    # Add batch dimension
    input_tensor = input_tensor.unsqueeze(0)

    input_tensor = input_tensor.to(device)

    # --------------------------------------------------
    # Prediction
    # --------------------------------------------------

    with torch.no_grad():

        outputs = model(input_tensor)

        probabilities = torch.softmax(
            outputs,
            dim=1
        )

        confidence, predicted_index = torch.max(
            probabilities,
            dim=1
        )

    predicted_index = predicted_index.item()
    confidence = confidence.item()

    predicted_class = class_names[predicted_index]

    print(
        f"Predicted: {predicted_class}"
    )

    print(
        f"Confidence: {confidence * 100:.2f}%"
    )

    # Show top 3 predictions
    top_probabilities, top_indices = torch.topk(
        probabilities[0],
        k=3
    )

    print("Top 3:")

    for probability, index in zip(
        top_probabilities,
        top_indices
    ):

        print(
            f"  {class_names[index.item()]}: "
            f"{probability.item() * 100:.2f}%"
        )


print("\n===== TEST COMPLETE =====")