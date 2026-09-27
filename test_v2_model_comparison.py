
import os
import numpy as np
import torch

from models.tcn_model import TCNModel


OLD_MODEL = "trained_models/isl_tcn.pth"
NEW_MODEL = "trained_models/isl_tcn_10class_v2.pth"

SEQUENCE_DIR = "debug_v2_sequences"


def load_model(path):
    checkpoint = torch.load(
        path,
        map_location="cpu"
    )

    model = TCNModel(
        input_size=checkpoint["input_size"],
        num_classes=checkpoint["num_classes"]
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.eval()

    return model, checkpoint["class_names"]


def predict(model, classes, sequence):

    sequence = np.asarray(
        sequence,
        dtype=np.float32
    )

    tensor = torch.tensor(
        sequence,
        dtype=torch.float32
    ).unsqueeze(0)

    with torch.no_grad():

        output = model(tensor)

        probabilities = torch.softmax(
            output,
            dim=1
        )

        confidence, index = torch.max(
            probabilities,
            dim=1
        )

    return (
        classes[index.item()],
        confidence.item()
    )


# ============================================================
# LOAD MODELS
# ============================================================

print("\nLoading old model...")

old_model, old_classes = load_model(
    OLD_MODEL
)

print("Old model loaded.")


print("\nLoading new 10-class V2 model...")

new_model, new_classes = load_model(
    NEW_MODEL
)

print("New model loaded.")


# ============================================================
# TEST SAVED LIVE SEQUENCES
# ============================================================

print("\n" + "=" * 90)
print("SAVED V2 LIVE SEQUENCE COMPARISON")
print("=" * 90)

files = sorted(
    f
    for f in os.listdir(SEQUENCE_DIR)
    if f.endswith(".npy")
)

if not files:

    print(
        f"No .npy files found in {SEQUENCE_DIR}"
    )

    exit()


for filename in files:

    path = os.path.join(
        SEQUENCE_DIR,
        filename
    )

    sequence = np.load(path)

    if sequence.shape != (30, 225):

        print(
            f"{filename:25s} "
            f"INVALID SHAPE: {sequence.shape}"
        )

        continue

    old_word, old_conf = predict(
        old_model,
        old_classes,
        sequence
    )

    new_word, new_conf = predict(
        new_model,
        new_classes,
        sequence
    )

    print(
        f"{filename:25s} "
        f"OLD: {old_word:12s} {old_conf:.3f} | "
        f"NEW: {new_word:12s} {new_conf:.3f}"
    )


print("\n" + "=" * 90)
print("TEST COMPLETE")
print("=" * 90)

print(
    "\nThe new model is saved separately and "
    "the original model was not modified."
)
