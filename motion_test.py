import cv2
import numpy as np
from collections import deque

from utils.mediapipe_utils import initialize_holistic
from utils.landmark_utils import extract_landmarks
from preprocessing.normalization import normalize_landmarks


# --------------------------------------------------
# Configuration
# --------------------------------------------------

MOTION_THRESHOLD = 0.025

motion_history = deque(maxlen=7)
previous_landmarks = None


# --------------------------------------------------
# Calculate motion
# --------------------------------------------------

def calculate_motion(current_landmarks, previous_landmarks):
    if previous_landmarks is None:
        return 0.0

    difference = np.abs(current_landmarks - previous_landmarks)

    # Mean movement across all 225 normalized features
    motion = np.mean(difference)

    return float(motion)


# --------------------------------------------------
# Main
# --------------------------------------------------

def main():

    global previous_landmarks

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("ERROR: Could not open webcam.")
        return

    holistic = initialize_holistic()

    print("\n" + "=" * 60)
    print("MOTION TEST")
    print("=" * 60)

    print("\nFollow this sequence:")
    print("1. Stay completely STILL for about 5 seconds")
    print("2. Sign HELLO for about 3 seconds")
    print("3. Stay STILL for about 3 seconds")
    print("4. Sign GOOD for about 3 seconds")
    print("5. Stay STILL for about 3 seconds")
    print("6. Sign another word continuously")
    print("\nPress Q to quit.")
    print("=" * 60)

    frame_count = 0

    while True:

        ret, frame = cap.read()

        if not ret:
            print("ERROR: Could not read webcam frame.")
            break

        frame = cv2.flip(frame, 1)

        # ------------------------------------------
        # MediaPipe Holistic
        # ------------------------------------------

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        results = holistic.process(rgb_frame)

        # ------------------------------------------
        # Extract 225 features
        # ------------------------------------------

        landmarks = extract_landmarks(results)

        # ------------------------------------------
        # Normalize landmarks
        # ------------------------------------------

        normalized_landmarks = normalize_landmarks(landmarks)

        # ------------------------------------------
        # Calculate motion
        # ------------------------------------------

        motion = calculate_motion(
            normalized_landmarks,
            previous_landmarks
        )

        previous_landmarks = normalized_landmarks.copy()

        motion_history.append(motion)

        # ------------------------------------------
        # Calculate recent average
        # ------------------------------------------

        if len(motion_history) > 0:
            average_motion = np.mean(motion_history)
        else:
            average_motion = 0.0

        # ------------------------------------------
        # Determine activity using current threshold
        # ------------------------------------------

        if motion >= MOTION_THRESHOLD:
            activity = "MOVING"
        else:
            activity = "STILL"

        # ------------------------------------------
        # Display information
        # ------------------------------------------

        cv2.putText(
            frame,
            f"Motion: {motion:.5f}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2
        )

        cv2.putText(
            frame,
            f"Avg Motion: {average_motion:.5f}",
            (20, 75),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 255),
            2
        )

        cv2.putText(
            frame,
            f"Threshold: {MOTION_THRESHOLD:.5f}",
            (20, 110),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2
        )

        cv2.putText(
            frame,
            activity,
            (20, 155),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 0, 255) if activity == "STILL" else (0, 255, 0),
            3
        )

        cv2.putText(
            frame,
            "Press Q to quit",
            (20, 195),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            1
        )

        cv2.imshow("ISL Motion Test", frame)

        # ------------------------------------------
        # Print motion periodically
        # ------------------------------------------

        frame_count += 1

        if frame_count % 30 == 0:
            print(
                f"Motion: {motion:.5f} | "
                f"Average: {average_motion:.5f} | "
                f"Activity: {activity}"
            )

        # ------------------------------------------
        # Quit
        # ------------------------------------------

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break

    cap.release()
    holistic.close()
    cv2.destroyAllWindows()

    print("\nMotion test finished.")


if __name__ == "__main__":
    main()