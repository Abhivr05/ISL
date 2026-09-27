from pathlib import Path
import numpy as np

ROOT = Path(".")
RAW = ROOT / "raw_dataset"
INCLUDE = ROOT / "include_processed"

CLASSES = [
    "YES", "NO", "TODAY", "YESTERDAY",
    "MONEY", "MEDICINE", "HOSPITAL"
]

def count_npy(folder):
    if not folder.exists():
        return 0, []
    files = sorted(folder.rglob("*.npy"))
    valid = []
    bad = []
    for f in files:
        try:
            arr = np.load(f, allow_pickle=False)
            if arr.shape == (30, 225):
                valid.append(f)
            else:
                bad.append((f, arr.shape))
        except Exception as e:
            bad.append((f, str(e)))
    return len(valid), bad

print("\n=== PROBLEM CLASS DATASET CHECK ===\n")
print(f"{'CLASS':<12} {'RAW':>6} {'INCLUDE':>8} {'TOTAL':>7}")
print("-" * 38)

for cls in CLASSES:
    raw_count, raw_bad = count_npy(RAW / cls)
    inc_count, inc_bad = count_npy(INCLUDE / cls)
    print(f"{cls:<12} {raw_count:>6} {inc_count:>8} {raw_count + inc_count:>7}")

    if raw_bad:
        print(f"  RAW invalid files: {len(raw_bad)}")
    if inc_bad:
        print(f"  INCLUDE invalid files: {len(inc_bad)}")

print("\nExpected valid shape: (30, 225)")
print("This script only reads data; it does not modify anything.")
