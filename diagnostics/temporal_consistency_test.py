import cv2
import numpy as np
import torch
import time

from utils.mediapipe_utils import initialize_holistic
from utils.landmark_utils import extract_landmarks
from preprocessing.normalization import normalize_landmarks
from models.tcn_model import TCNModel


# ============================================================
# CONFIG
# ============================================================

MODEL_PATH = "trained_models/isl_tcn_19class_v3.pth"

SEQUENCE_LENGTH = 30
FEATURE_DIM = 225

TOTAL_FRAMES = 120
WINDOW_STEP = 5

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

device = torch.device("cpu")


# ============================================================
# MODEL
# ============================================================

checkpoint = torch.load(
    MODEL_PATH,
    map_location=device,
    weights_only=False
)

if isinstance(checkpoint, dict):
    state_dict = checkpoint.get(
        "model_state_dict",
        checkpoint
    )
else:
    state_dict = checkpoint


model = TCNModel(
    input_size=FEATURE_DIM,
    num_classes=len(CLASS_NAMES)
)

model.load_state_dict(state_dict)
model.to(device)
model.eval()


# ============================================================
# PREDICTION
# ============================================================

@torch.no_grad()
def predict(sequence):

    x = torch.tensor(
        sequence,
        dtype=torch.float32
    ).unsqueeze(0)

    logits = model(x)

    if isinstance(logits, (tuple, list)):
        logits = logits[0]

    logits = logits.squeeze(0).cpu().numpy()

    # Softmax
    exp_logits = np.exp(
        logits - np.max(logits)
    )

    probs = exp_logits / exp_logits.sum()

    order = np.argsort(logits)[::-1]

    top = order[0]
    second = order[1]

    return {
        "class": CLASS_NAMES[top],
        "confidence": float(probs[top]),
        "logit": float(logits[top]),
        "margin": float(
            probs[top] - probs[second]
        ),
        "second": CLASS_NAMES[second],
    }


# ============================================================
# CAMERA
# ============================================================

holistic = initialize_holistic()

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    raise RuntimeError("Could not open webcam")


print()
print("=" * 80)
print("TEMPORAL CONSISTENCY TEST")
print("=" * 80)

print()
print("Controls:")
print("  SPACE = start capture")
print("  R     = reset")
print("  Q     = quit")
print()
print(f"Capture length : {TOTAL_FRAMES} frames")
print(f"Window         : {SEQUENCE_LENGTH} frames")
print(f"Step           : {WINDOW_STEP} frames")
print()


# ============================================================
# CAPTURE
# ============================================================

frames = []
capturing = False

while True:

    ret, frame = cap.read()

    if not ret:
        break

    frame = cv2.flip(frame, 1)

    rgb = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )

    results = holistic.process(rgb)

    if results.pose_landmarks:

        features = extract_landmarks(results)

        if features is not None:

            normalized = normalize_landmarks(
                features
            )

            if normalized is not None:
                frames.append(normalized)

                if len(frames) > TOTAL_FRAMES:
                    frames.pop(0)

    # Display
    display = frame.copy()

    cv2.putText(
        display,
        f"Frames: {len(frames)}/{TOTAL_FRAMES}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2
    )

    if capturing:

        cv2.putText(
            display,
            "CAPTURING...",
            (20, 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 255),
            2
        )

    else:

        cv2.putText(
            display,
            "Press SPACE to capture",
            (20, 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2
        )

    cv2.imshow(
        "Temporal Consistency Test",
        display
    )

    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break

    if key == ord("r"):
        frames = []
        capturing = False
        print("\nReset.")

    if key == 32 and not capturing:

        frames = []
        capturing = True

        print()
        print("CAPTURING 120 FRAMES...")
        print("Perform ONE movement/sign continuously.")
        print()

    if capturing and len(frames) >= TOTAL_FRAMES:
        break


cap.release()
holistic.close()
cv2.destroyAllWindows()


# ============================================================
# ANALYZE OVERLAPPING WINDOWS
# ============================================================

if len(frames) < TOTAL_FRAMES:

    print(
        f"Only captured {len(frames)} frames."
    )

    raise SystemExit


frames = np.array(frames)

results = []

for start in range(
    0,
    TOTAL_FRAMES - SEQUENCE_LENGTH + 1,
    WINDOW_STEP
):

    end = start + SEQUENCE_LENGTH

    sequence = frames[start:end]

    result = predict(sequence)

    result["start"] = start
    result["end"] = end

    results.append(result)


# ============================================================
# PRINT RESULTS
# ============================================================

print()
print("=" * 100)
print("WINDOW PREDICTIONS")
print("=" * 100)

print(
    f"{'WINDOW':10s} "
    f"{'PREDICTION':14s} "
    f"{'CONF':>9s} "
    f"{'LOGIT':>10s} "
    f"{'MARGIN':>9s} "
    f"{'RUNNER-UP':14s}"
)

print("-" * 100)

for r in results:

    print(
        f"{r['start']:03d}-{r['end']:03d}    "
        f"{r['class']:14s} "
        f"{r['confidence'] * 100:8.2f}% "
        f"{r['logit']:10.3f} "
        f"{r['margin'] * 100:8.2f}% "
        f"{r['second']:14s}"
    )


# ============================================================
# CONSISTENCY SUMMARY
# ============================================================

predictions = [
    r["class"]
    for r in results
]

confidences = [
    r["confidence"]
    for r in results
]

logits = [
    r["logit"]
    for r in results
]


unique, counts = np.unique(
    predictions,
    return_counts=True
)

order = np.argsort(counts)[::-1]

print()
print("=" * 80)
print("CONSISTENCY SUMMARY")
print("=" * 80)

for i in order:

    print(
        f"{unique[i]:14s} "
        f"{counts[i]:2d}/{len(predictions)} "
        f"({counts[i] / len(predictions) * 100:.1f}%)"
    )


dominant_idx = order[0]
dominant_class = unique[dominant_idx]
dominant_count = counts[dominant_idx]

print()
print(
    f"DOMINANT CLASS : {dominant_class}"
)

print(
    f"CONSISTENCY    : "
    f"{dominant_count}/{len(predictions)} "
    f"({dominant_count / len(predictions) * 100:.1f}%)"
)

print(
    f"AVG CONFIDENCE : "
    f"{np.mean(confidences) * 100:.2f}%"
)

print(
    f"AVG LOGIT      : "
    f"{np.mean(logits):.3f}"
)

print()
print("=" * 80)