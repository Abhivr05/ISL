import numpy as np
import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report

from dataset.dataset_loader import load_dataset
from models.tcn_model import TCNModel


# --------------------------------------------------
# Configuration
# --------------------------------------------------

MODEL_PATH = "trained_models/isl_tcn.pth"

VALIDATION_SIZE = 0.2
RANDOM_STATE = 42


# --------------------------------------------------
# Device
# --------------------------------------------------

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print(f"Using device: {device}")


# --------------------------------------------------
# Load dataset
# --------------------------------------------------

X, y, class_names = load_dataset()

print("\nDataset loaded.")
print(f"X shape: {X.shape}")
print(f"y shape: {y.shape}")


# --------------------------------------------------
# Same train/validation split as training
# --------------------------------------------------

_, X_val, _, y_val = train_test_split(
    X,
    y,
    test_size=VALIDATION_SIZE,
    random_state=RANDOM_STATE,
    stratify=y
)

print("\n===== VALIDATION SET =====")
print(f"Validation samples: {len(X_val)}")
print("==========================")


# --------------------------------------------------
# Load model
# --------------------------------------------------

checkpoint = torch.load(
    MODEL_PATH,
    map_location=device
)

model = TCNModel(
    input_size=225,
    num_classes=len(class_names)
)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

model = model.to(device)
model.eval()


# --------------------------------------------------
# Convert validation data to tensor
# --------------------------------------------------

X_val_tensor = torch.tensor(
    X_val,
    dtype=torch.float32
).to(device)


# --------------------------------------------------
# Prediction
# --------------------------------------------------

with torch.no_grad():

    outputs = model(X_val_tensor)

    predictions = torch.argmax(
        outputs,
        dim=1
    ).cpu().numpy()


# --------------------------------------------------
# Overall accuracy
# --------------------------------------------------

accuracy = (
    np.sum(predictions == y_val)
    / len(y_val)
) * 100

print(
    f"\nValidation accuracy: {accuracy:.2f}%"
)


# --------------------------------------------------
# Classification report
# --------------------------------------------------

print("\n===== PER-CLASS RESULTS =====")

print(
    classification_report(
        y_val,
        predictions,
        labels=np.arange(len(class_names)),
        target_names=class_names,
        zero_division=0
    )
)


# --------------------------------------------------
# Confusion matrix
# --------------------------------------------------

cm = confusion_matrix(
    y_val,
    predictions,
    labels=np.arange(len(class_names))
)

print("\n===== CONFUSION MATRIX =====")

print(
    "Rows = Actual"
)
print(
    "Columns = Predicted\n"
)

print(
    "             "
    + " ".join(f"{name:>10}" for name in class_names)
)

for i, row in enumerate(cm):

    print(
        f"{class_names[i]:>10} "
        + " ".join(f"{value:10d}" for value in row)
    )
    