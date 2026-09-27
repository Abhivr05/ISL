import cv2
import mediapipe as mp
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.mediapipe_utils import initialize_holistic
from utils.landmark_utils import extract_landmarks


cap = cv2.VideoCapture(0)

holistic = initialize_holistic()


while cap.isOpened():

    success, frame = cap.read()

    if not success:
        continue

    frame = cv2.flip(frame, 1)

    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = holistic.process(image)

    feature_vector = extract_landmarks(results)

    print(len(feature_vector))

    cv2.imshow("Landmark Test", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


cap.release()
cv2.destroyAllWindows()