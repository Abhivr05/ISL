import os
import glob
import numpy as np
import torch

from models.tcn_model import TCNModel


# --------------------------------------------------
# Paths
# --------------------------------------------------

MODEL_PATH = "trained_models/isl_tcn.pth"
SEQUENCE_DIR = "debug_8_signs"


# --------------------------------------------------
# Device
# --------------------------------------------------

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Device:", device)
print()


# --------------------------------------------------
# Load checkpoint
# --------------------------------------------------

checkpoint = torch.load(MODEL_PATH, map_location=device)

class_names = checkpoint["class_names"]

model = TCNModel(
    input_size=checkpoint["input_size"],
    num_classes=checkpoint["num_classes"]
)

model.load_state_dict(checkpoint["model_state_dict"])
model.to(device)
model.eval()


print("Model loaded")
print("Classes:", class_names)
print("Input size:", checkpoint["input_size"])
print("Sequence length:", checkpoint["sequence_length"])
print()


# --------------------------------------------------
# Find saved sequences
# --------------------------------------------------

sequence_files = sorted(
    glob.glob(os.path.join(SEQUENCE_DIR, "sequence_*.npy"))
)

if not sequence_files:
    print("No saved sequences found.")
    print("Expected folder:", SEQUENCE_DIR)
    exit()


# --------------------------------------------------
# Predict each saved sequence
# --------------------------------------------------

print("=" * 70)
print("OFFLINE PREDICTION OF SAVED LIVE SEQUENCES")
print("=" * 70)

with torch.no_grad():

    for file_path in sequence_files:

        sequence = np.load(file_path)

        print()
        print(os.path.basename(file_path))
        print("Shape:", sequence.shape)

        # Convert to tensor
        x = torch.tensor(
            sequence,
            dtype=torch.float32
        ).unsqueeze(0).to(device)

        # Prediction
        output = model(x)

        probabilities = torch.softmax(output, dim=1)

        confidence, predicted_index = torch.max(
            probabilities,
            dim=1
        )

        predicted_index = predicted_index.item()
        confidence = confidence.item()

        predicted_class = class_names[predicted_index]

        # Top 3 predictions
        top_probs, top_indices = torch.topk(
            probabilities,
            min(3, len(class_names)),
            dim=1
        )

        print(
            f"Prediction: {predicted_class:<12} "
            f"Confidence: {confidence:.4f}"
        )

        print("Top 3:")

        for prob, idx in zip(
            top_probs[0],
            top_indices[0]
        ):
            print(
                f"  {class_names[idx.item()]:<12} "
                f"{prob.item():.4f}"
            )

print()
print("=" * 70)
print("DONE")
print("=" * 70)