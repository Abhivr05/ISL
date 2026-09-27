import cv2
from pathlib import Path


DATASET_DIR = Path("external_dataset/isl_40words")


for video_path in sorted(DATASET_DIR.rglob("*.mp4")):

    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        print(f"ERROR: Could not open {video_path}")
        continue

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    duration = frame_count / fps if fps > 0 else 0

    cap.release()

    print(
        f"{video_path.parent.name:10} | "
        f"{frame_count:4} frames | "
        f"{fps:5.1f} FPS | "
        f"{width}x{height} | "
        f"{duration:.2f}s"
    )