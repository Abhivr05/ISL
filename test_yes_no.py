import numpy as np
import torch
from collections import Counter
from sklearn.model_selection import train_test_split

from models.tcn_model import TCNModel
from dataset.dataset_loader import load_dataset


# --------------------------------------------------
# Configuration
# --------------------------------------------------

MODEL_PATH = "trained_models/isl_tcn.pth"
RANDOM_STATE = 42

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
    "YOU"
]


# --------------------------------------------------
# Load dataset
# --------------------------------------------------

print("=" * 60)
print("YES / NO OFFLINE DIAGNOSTIC")
print("=" * 60)

X, y, class_names = load_dataset()

print(f"\nDataset shape: {X.shape}")
print(f"Labels shape : {y.shape}")
print(f"Classes      : {class_names}")


# --------------------------------------------------
# Same train/validation split used during training
# --------------------------------------------------

X_train, X_val, y_train, y_val = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=RANDOM_STATE,
    stratify=y
)

print(f"\nValidation samples: {len(X_val)}")


# --------------------------------------------------
# Load model
# --------------------------------------------------

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = TCNModel(
    input_size=225,
    num_classes=len(class_names)
)

checkpoint = torch.load(
    MODEL_PATH,
    map_location=device
)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

model.to(device)
model.eval()

print(f"Device: {device}")
print("Model loaded successfully.")


# --------------------------------------------------
# Run predictions
# --------------------------------------------------

X_tensor = torch.tensor(
    X_val,
    dtype=torch.float32
).to(device)

with torch.no_grad():
    outputs = model(X_tensor)
    probabilities = torch.softmax(outputs, dim=1)
    predictions = torch.argmax(probabilities, dim=1)

predictions = predictions.cpu().numpy()
probabilities = probabilities.cpu().numpy()


# --------------------------------------------------
# Find class IDs
# --------------------------------------------------

name_to_id = {
    name: index
    for index, name in enumerate(class_names)
}

NO_ID = name_to_id["NO"]
YES_ID = name_to_id["YES"]


# --------------------------------------------------
# Analyze YES
# --------------------------------------------------

print("\n" + "=" * 60)
print("ACTUAL YES → MODEL PREDICTIONS")
print("=" * 60)

yes_indices = np.where(y_val == YES_ID)[0]

yes_predictions = predictions[yes_indices]

yes_counter = Counter(yes_predictions)

for class_id, count in yes_counter.most_common():

    class_name = class_names[class_id]

    print(
        f"{class_name:12s}: "
        f"{count:2d} / {len(yes_indices)}"
    )


# --------------------------------------------------
# Analyze NO
# --------------------------------------------------

print("\n" + "=" * 60)
print("ACTUAL NO → MODEL PREDICTIONS")
print("=" * 60)

no_indices = np.where(y_val == NO_ID)[0]

no_predictions = predictions[no_indices]

no_counter = Counter(no_predictions)

for class_id, count in no_counter.most_common():

    class_name = class_names[class_id]

    print(
        f"{class_name:12s}: "
        f"{count:2d} / {len(no_indices)}"
    )


# --------------------------------------------------
# Detailed YES samples
# --------------------------------------------------

print("\n" + "=" * 60)
print("INDIVIDUAL YES SAMPLES")
print("=" * 60)

for index in yes_indices:

    actual = class_names[y_val[index]]
    predicted = class_names[predictions[index]]
    confidence = probabilities[index][predictions[index]]

    print(
        f"Actual: {actual:5s} | "
        f"Predicted: {predicted:12s} | "
        f"Confidence: {confidence:.2f}"
    )


# --------------------------------------------------
# Detailed NO samples
# --------------------------------------------------

print("\n" + "=" * 60)
print("INDIVIDUAL NO SAMPLES")
print("=" * 60)

for index in no_indices:

    actual = class_names[y_val[index]]
    predicted = class_names[predictions[index]]
    confidence = probabilities[index][predictions[index]]

    print(
        f"Actual: {actual:5s} | "
        f"Predicted: {predicted:12s} | "
        f"Confidence: {confidence:.2f}"
    )


# --------------------------------------------------
# Accuracy for YES and NO
# --------------------------------------------------

yes_correct = np.sum(y_val[yes_indices] == predictions[yes_indices])
no_correct = np.sum(y_val[no_indices] == predictions[no_indices])

yes_accuracy = yes_correct / len(yes_indices)
no_accuracy = no_correct / len(no_indices)

print("\n" + "=" * 60)
print("RESULT")
print("=" * 60)

print(
    f"YES accuracy: "
    f"{yes_accuracy * 100:.2f}% "
    f"({yes_correct}/{len(yes_indices)})"
)

print(
    f"NO accuracy : "
    f"{no_accuracy * 100:.2f}% "
    f"({no_correct}/{len(no_indices)})"
)

print("=" * 60)