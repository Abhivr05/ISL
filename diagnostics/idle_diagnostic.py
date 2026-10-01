import os
import sys
import time
import cv2
import numpy as np
import torch

# ============================================================
# PROJECT PATH
# ============================================================

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


# ============================================================
# PROJECT IMPORTS
# ============================================================

from utils.mediapipe_utils import initialize_holistic
from utils.landmark_utils import extract_landmarks
from preprocessing.normalization import normalize_landmarks
from models.tcn_model import TCNModel


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_PATH = os.path.join(
    PROJECT_ROOT,
    "trained_models",
    "isl_tcn_19class_v3.pth"
)

SEQUENCE_LENGTH = 30
FEATURE_DIMENSION = 225

class_names = [
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


# ============================================================
# DEVICE
# ============================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# LOAD MODEL
# ============================================================

print("=" * 65)
print("ISL IDLE / FALSE-DETECTION DIAGNOSTIC")
print("=" * 65)

print("Device:", device)
print("Model:", MODEL_PATH)
print()

checkpoint = torch.load(
    MODEL_PATH,
    map_location=device
)

model = TCNModel(
    input_size=FEATURE_DIMENSION,
    num_classes=len(class_names)
)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

model.to(device)
model.eval()

print("Model loaded successfully.")
print()


# ============================================================
# PREDICTION
# ============================================================

def predict_sequence(sequence):

    sequence_array = np.asarray(
        sequence,
        dtype=np.float32
    )

    if sequence_array.shape != (
        SEQUENCE_LENGTH,
        FEATURE_DIMENSION
    ):
        raise ValueError(
            f"Invalid sequence shape: {sequence_array.shape}"
        )

    tensor = torch.tensor(
        sequence_array,
        dtype=torch.float32
    ).unsqueeze(0)

    tensor = tensor.to(device)

    with torch.no_grad():

        output = model(tensor)

        probabilities = torch.softmax(
            output,
            dim=1
        )[0]

    top_values, top_indices = torch.topk(
        probabilities,
        k=3
    )

    results = []

    for value, index in zip(
        top_values.cpu().numpy(),
        top_indices.cpu().numpy()
    ):

        results.append(
            (
                class_names[int(index)],
                float(value)
            )
        )

    return results


# ============================================================
# MOTION
# ============================================================

def calculate_motion(sequence):

    sequence = np.asarray(
        sequence,
        dtype=np.float32
    )

    differences = np.abs(
        sequence[1:] - sequence[:-1]
    )

    frame_motion = np.mean(
        differences,
        axis=1
    )

    return {
        "mean": float(np.mean(frame_motion)),
        "max": float(np.max(frame_motion)),
        "min": float(np.min(frame_motion)),
        "std": float(np.std(frame_motion)),
    }


# ============================================================
# CAMERA
# ============================================================

holistic = initialize_holistic()

cap = cv2.VideoCapture(0)

if not cap.isOpened():

    print("ERROR: Could not open webcam.")
    sys.exit(1)


# ============================================================
# INSTRUCTIONS
# ============================================================

print("=" * 65)
print("TEST INSTRUCTIONS")
print("=" * 65)

print()
print("We will capture 10 normal/non-signing sequences.")
print()
print("For each test:")
print("  1. Stay in front of the camera.")
print("  2. DO NOT perform an ISL sign.")
print("  3. You may behave naturally:")
print("       - sit normally")
print("       - move slightly")
print("       - look around")
print("       - move your head")
print("       - adjust your posture")
print("  4. Press SPACE to capture a 30-frame window.")
print()
print("Press Q to quit.")
print("=" * 65)
print()


# ============================================================
# OUTPUT DIRECTORY
# ============================================================

output_dir = os.path.join(
    PROJECT_ROOT,
    "diagnostics",
    "idle_sequences"
)

os.makedirs(
    output_dir,
    exist_ok=True
)


# ============================================================
# TEST LOOP
# ============================================================

test_number = 0

while test_number < 10:

    ret, frame = cap.read()

    if not ret:
        print("ERROR: Could not read webcam frame.")
        break

    display = frame.copy()

    cv2.putText(
        display,
        f"Idle Diagnostic: {test_number + 1}/10",
        (20, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2
    )

    cv2.putText(
        display,
        "Press SPACE to capture | Q to quit",
        (20, 70),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2
    )

    cv2.imshow(
        "ISL Idle Diagnostic",
        display
    )

    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break

    if key != 32:
        continue

    print()
    print(
        f"Capturing idle sequence "
        f"{test_number + 1}/10..."
    )

    sequence = []

    for _ in range(SEQUENCE_LENGTH):

        ret, frame = cap.read()

        if not ret:
            break

        rgb = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        results = holistic.process(rgb)

        raw_features = extract_landmarks(
            results
        )

        normalized_features = normalize_landmarks(
            raw_features
        )

        sequence.append(
            normalized_features
        )

        cv2.imshow(
            "ISL Idle Diagnostic",
            frame
        )

        cv2.waitKey(1)

    if len(sequence) != SEQUENCE_LENGTH:

        print("Capture failed.")
        continue

    sequence = np.asarray(
        sequence,
        dtype=np.float32
    )

    # --------------------------------------------------------
    # PREDICTION
    # --------------------------------------------------------

    top3 = predict_sequence(
        sequence
    )

    motion = calculate_motion(
        sequence
    )

    # --------------------------------------------------------
    # SAVE SEQUENCE
    # --------------------------------------------------------

    timestamp = int(time.time())

    save_path = os.path.join(
        output_dir,
        f"idle_{timestamp}.npy"
    )

    np.save(
        save_path,
        sequence
    )

    # --------------------------------------------------------
    # DISPLAY RESULT
    # --------------------------------------------------------

    print("-" * 65)

    print(
        f"Test {test_number + 1}"
    )

    print(
        f"Top prediction : "
        f"{top3[0][0]} "
        f"({top3[0][1] * 100:.2f}%)"
    )

    print(
        f"Runner-up     : "
        f"{top3[1][0]} "
        f"({top3[1][1] * 100:.2f}%)"
    )

    print(
        f"Third          : "
        f"{top3[2][0]} "
        f"({top3[2][1] * 100:.2f}%)"
    )

    print()

    print(
        f"Motion mean    : "
        f"{motion['mean']:.5f}"
    )

    print(
        f"Motion max     : "
        f"{motion['max']:.5f}"
    )

    print(
        f"Motion std     : "
        f"{motion['std']:.5f}"
    )

    print(
        f"Saved          : "
        f"{save_path}"
    )

    test_number += 1

    print()
    print(
        "Return to a normal position."
    )
    print(
        "Press SPACE for the next test."
    )


# ============================================================
# CLEANUP
# ============================================================

cap.release()
holistic.close()

cv2.destroyAllWindows()

print()
print("=" * 65)
print("DIAGNOSTIC COMPLETE")
print("=" * 65)
print(
    f"Captured {test_number} idle sequences."
)
print(
    f"Saved to: {output_dir}"
)
print()