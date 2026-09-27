import random
import os

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

from dataset.dataset_loader import load_dataset
from models.tcn_model import TCNModel


# --------------------------------------------------
# Configuration
# --------------------------------------------------

BATCH_SIZE = 16
EPOCHS = 30
LEARNING_RATE = 0.0005

VALIDATION_SIZE = 0.2
RANDOM_STATE = 42

MODEL_PATH = "trained_models/isl_tcn.pth"

# --------------------------------------------------
# Reproducibility
# --------------------------------------------------

random.seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)
torch.manual_seed(RANDOM_STATE)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(RANDOM_STATE)

# --------------------------------------------------
# Device
# --------------------------------------------------

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print(f"Using device: {device}")


# --------------------------------------------------
# Load Dataset
# --------------------------------------------------

X, y, class_names = load_dataset()

print("\nDataset ready for training.")


# --------------------------------------------------
# Train / Validation Split
# --------------------------------------------------

X_train, X_val, y_train, y_val = train_test_split(
    X,
    y,
    test_size=VALIDATION_SIZE,
    random_state=RANDOM_STATE,
    stratify=y
)

print("\n===== DATA SPLIT =====")
print(f"Training samples:   {len(X_train)}")
print(f"Validation samples: {len(X_val)}")
print("======================")


# --------------------------------------------------
# Convert NumPy → PyTorch tensors
# --------------------------------------------------

X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.long)

X_val = torch.tensor(X_val, dtype=torch.float32)
y_val = torch.tensor(y_val, dtype=torch.long)


# --------------------------------------------------
# Create DataLoaders
# --------------------------------------------------

train_dataset = TensorDataset(
    X_train,
    y_train
)

val_dataset = TensorDataset(
    X_val,
    y_val
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


# --------------------------------------------------
# Create Model
# --------------------------------------------------

model = TCNModel(
    input_size=225,
    num_classes=len(class_names)
)

model = model.to(device)


# --------------------------------------------------
# Loss and Optimizer
# --------------------------------------------------

criterion = nn.CrossEntropyLoss()

optimizer = optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
)


# --------------------------------------------------
# Best Model Tracking
# --------------------------------------------------

best_validation_accuracy = 0.0

print("\n===== TRAINING STARTED =====")


# --------------------------------------------------
# Training Loop
# --------------------------------------------------

for epoch in range(EPOCHS):

    model.train()

    running_loss = 0.0
    correct = 0
    total = 0

    for inputs, labels in train_loader:

        inputs = inputs.to(device)
        labels = labels.to(device)

        # Clear previous gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(inputs)

        # Calculate loss
        loss = criterion(outputs, labels)

        # Backpropagation
        loss.backward()

        # Update model
        optimizer.step()

        running_loss += loss.item()

        # Calculate training accuracy
        _, predicted = torch.max(outputs, 1)

        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    train_loss = running_loss / len(train_loader)
    train_accuracy = 100 * correct / total


    # --------------------------------------------------
    # Validation
    # --------------------------------------------------

    model.eval()

    val_correct = 0
    val_total = 0
    val_loss = 0.0

    with torch.no_grad():

        for inputs, labels in val_loader:

            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)

            loss = criterion(outputs, labels)

            val_loss += loss.item()

            _, predicted = torch.max(outputs, 1)

            val_total += labels.size(0)

            val_correct += (
                predicted == labels
            ).sum().item()

    validation_loss = val_loss / len(val_loader)

    validation_accuracy = (
        100 * val_correct / val_total
    )


    # --------------------------------------------------
    # Print Epoch Results
    # --------------------------------------------------

    print(
        f"Epoch [{epoch + 1:02d}/{EPOCHS}] "
        f"Loss: {train_loss:.4f} "
        f"Train Acc: {train_accuracy:.2f}% "
        f"Val Loss: {validation_loss:.4f} "
        f"Val Acc: {validation_accuracy:.2f}%"
    )


    # --------------------------------------------------
    # Save Best Model
    # --------------------------------------------------

    if validation_accuracy > best_validation_accuracy:

        best_validation_accuracy = validation_accuracy

        os.makedirs(
            os.path.dirname(MODEL_PATH),
            exist_ok=True
        )

        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "class_names": class_names,
                "input_size": 225,
                "sequence_length": 30,
                "num_classes": len(class_names)
            },
            MODEL_PATH
        )

        print(
            f"  -> Best model saved "
            f"(Val Acc: {validation_accuracy:.2f}%)"
        )


# --------------------------------------------------
# Training Complete
# --------------------------------------------------

print("\n===== TRAINING COMPLETE =====")

print(
    f"Best validation accuracy: "
    f"{best_validation_accuracy:.2f}%"
)

print(
    f"Best model saved to: {MODEL_PATH}"
)

