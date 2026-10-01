# diagnostics/class_logit_calibration.py

import os
import glob
import numpy as np
import torch

from models.tcn_model import TCNModel


# ============================================================
# CONFIG
# ============================================================

MODEL_PATH = "trained_models/isl_tcn_19class_v3.pth"
REAL_DATASET = "raw_dataset"
RANDOM_DATASET = "diagnostics/idle_sequences"

SEQUENCE_LENGTH = 30
FEATURE_DIM = 225

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


# ============================================================
# MODEL
# ============================================================

device = torch.device("cpu")

checkpoint = torch.load(
    MODEL_PATH,
    map_location=device,
    weights_only=False
)

# Handle either checkpoint format
if isinstance(checkpoint, dict):
    state_dict = checkpoint.get("model_state_dict", checkpoint)
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
# FIND CLASS FROM FILE PATH
# ============================================================

def get_class_from_path(path):
    parts = os.path.normpath(path).split(os.sep)

    for cls in CLASS_NAMES:
        if cls in parts:
            return cls

    return None


# ============================================================
# EXTRACT RAW LOGITS
# ============================================================

@torch.no_grad()
def get_logits(sequence):

    x = torch.tensor(
        sequence,
        dtype=torch.float32
    ).unsqueeze(0).to(device)

    logits = model(x)

    # If model returns tuple/list
    if isinstance(logits, (tuple, list)):
        logits = logits[0]

    return logits.squeeze(0).cpu().numpy()


# ============================================================
# LOAD REAL DATA
# ============================================================

real_samples = []

for path in glob.glob(
    os.path.join(REAL_DATASET, "**", "*.npy"),
    recursive=True
):

    cls = get_class_from_path(path)

    if cls is None:
        continue

    try:
        sequence = np.load(path)

        if sequence.shape != (SEQUENCE_LENGTH, FEATURE_DIM):
            continue

        logits = get_logits(sequence)

        predicted_idx = int(np.argmax(logits))
        predicted_class = CLASS_NAMES[predicted_idx]

        real_samples.append({
            "path": path,
            "class": cls,
            "logit": float(logits[CLASS_NAMES.index(cls)]),
            "predicted": predicted_class,
            "correct": predicted_class == cls,
        })

    except Exception as e:
        print(f"Skipping {path}: {e}")


print()
print("=" * 80)
print("REAL SIGN LOGIT DISTRIBUTIONS")
print("=" * 80)

# ============================================================
# CALCULATE CLASS DISTRIBUTIONS
# ============================================================

calibration = {}

for cls in CLASS_NAMES:

    values = [
        x["logit"]
        for x in real_samples
        if x["class"] == cls
    ]

    if not values:
        continue

    values = np.array(values)

    p5 = np.percentile(values, 5)
    p10 = np.percentile(values, 10)
    p25 = np.percentile(values, 25)
    median = np.percentile(values, 50)
    p75 = np.percentile(values, 75)
    p95 = np.percentile(values, 95)

    calibration[cls] = {
        "p5": p5,
        "p10": p10,
        "p25": p25,
        "median": median,
        "p75": p75,
        "p95": p95,
    }

    correct_count = sum(
        x["correct"]
        for x in real_samples
        if x["class"] == cls
    )

    total = len(values)

    print(
        f"{cls:12s} "
        f"N={total:3d} "
        f"correct={correct_count:3d}/{total:<3d} "
        f"P5={p5:8.3f} "
        f"P10={p10:8.3f} "
        f"MED={median:8.3f} "
        f"P95={p95:8.3f}"
    )


# ============================================================
# RANDOM / BACKGROUND DATA
# ============================================================

random_files = sorted(
    glob.glob(
        os.path.join(RANDOM_DATASET, "*.npy")
    )
)

print()
print("=" * 100)
print("RANDOM / BACKGROUND LOGIT ANALYSIS")
print("=" * 100)

print(
    f"{'FILE':12s} "
    f"{'PRED':12s} "
    f"{'CONF':>8s} "
    f"{'LOGIT':>10s} "
    f"{'P5':>10s} "
    f"{'P10':>10s} "
    f"{'MED':>10s} "
    f"{'P5?':>6s} "
    f"{'P10?':>7s}"
)

print("-" * 100)


random_results = []

for path in random_files:

    try:
        sequence = np.load(path)

        if sequence.shape != (SEQUENCE_LENGTH, FEATURE_DIM):
            continue

        logits = get_logits(sequence)

        predicted_idx = int(np.argmax(logits))
        predicted_class = CLASS_NAMES[predicted_idx]

        # Softmax confidence
        exp_logits = np.exp(
            logits - np.max(logits)
        )
        probs = exp_logits / exp_logits.sum()

        confidence = float(probs[predicted_idx])

        predicted_logit = float(
            logits[predicted_idx]
        )

        if predicted_class in calibration:

            stats = calibration[predicted_class]

            p5 = stats["p5"]
            p10 = stats["p10"]
            median = stats["median"]

            below_p5 = predicted_logit < p5
            below_p10 = predicted_logit < p10
            below_median = predicted_logit < median

            p5_text = "YES" if below_p5 else "NO"
            p10_text = "YES" if below_p10 else "NO"

        else:
            p5 = p10 = median = np.nan
            below_p5 = below_p10 = below_median = False
            p5_text = "N/A"
            p10_text = "N/A"

        filename = os.path.basename(path)

        print(
            f"{filename:12s} "
            f"{predicted_class:12s} "
            f"{confidence*100:7.2f}% "
            f"{predicted_logit:10.3f} "
            f"{p5:10.3f} "
            f"{p10:10.3f} "
            f"{median:10.3f} "
            f"{p5_text:>6s} "
            f"{p10_text:>7s}"
        )

        random_results.append({
            "file": filename,
            "class": predicted_class,
            "confidence": confidence,
            "logit": predicted_logit,
            "below_p5": below_p5,
            "below_p10": below_p10,
            "below_median": below_median,
        })

    except Exception as e:
        print(f"Skipping {path}: {e}")


# ============================================================
# SUMMARY
# ============================================================

print()
print("=" * 80)
print("BACKGROUND REJECTION SUMMARY")
print("=" * 80)

total_random = len(random_results)

for level, key in [
    ("P5", "below_p5"),
    ("P10", "below_p10"),
    ("MEDIAN", "below_median"),
]:

    rejected = sum(
        x[key]
        for x in random_results
    )

    percentage = (
        rejected / total_random * 100
        if total_random
        else 0
    )

    print(
        f"{level:8s}: "
        f"{rejected}/{total_random} "
        f"background sequences below threshold "
        f"({percentage:.1f}%)"
    )


# ============================================================
# IMPORTANT CLASSES
# ============================================================

print()
print("=" * 80)
print("FOCUS CLASSES")
print("=" * 80)

for cls in ["GOOD", "WATER", "YES", "NO"]:

    if cls not in calibration:
        print(f"{cls}: no real calibration data")
        continue

    s = calibration[cls]

    print(
        f"{cls:8s} -> "
        f"P5={s['p5']:.3f}, "
        f"P10={s['p10']:.3f}, "
        f"MEDIAN={s['median']:.3f}, "
        f"P95={s['p95']:.3f}"
    )

print()
print("Calibration complete.")