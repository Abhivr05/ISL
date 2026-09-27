import mediapipe as mp


# Initialize MediaPipe modules
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils


# Create Holistic model
def initialize_holistic():

    holistic = mp_holistic.Holistic(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )

    return holistic