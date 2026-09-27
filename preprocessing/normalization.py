import numpy as np


POSE_LANDMARKS = 33
HAND_LANDMARKS = 21
COORDINATES = 3

LEFT_SHOULDER = 11
RIGHT_SHOULDER = 12

EPSILON = 1e-6


def normalize_landmarks(feature_vector):
    """
    Normalize a 225-dimensional landmark feature vector.

    The landmark coordinates are centered around the midpoint
    between the left and right shoulders and scaled using the
    shoulder-to-shoulder distance.

    Missing hand landmarks represented by zeros are preserved.
    """

    feature_vector = np.asarray(
        feature_vector,
        dtype=np.float32
    )

    if feature_vector.shape != (225,):
        raise ValueError(
            f"Expected feature vector of shape (225,), "
            f"got {feature_vector.shape}"
        )

    # Convert flattened vector back to landmarks
    landmarks = feature_vector.reshape(75, 3).copy()

    # Pose landmarks
    pose = landmarks[:POSE_LANDMARKS]

    # Check whether both shoulders are available
    left_shoulder = pose[LEFT_SHOULDER]
    right_shoulder = pose[RIGHT_SHOULDER]

    shoulders_available = (
        not np.allclose(left_shoulder, 0.0)
        and not np.allclose(right_shoulder, 0.0)
    )

    if not shoulders_available:
        return feature_vector.copy()

    # Reference point: midpoint between shoulders
    center = (left_shoulder + right_shoulder) / 2.0

    # Scale: distance between shoulders
    scale = np.linalg.norm(
        left_shoulder - right_shoulder
    )

    if scale < EPSILON:
        scale = 1.0

    # Apply normalization only to detected landmarks
    detected = np.any(
        np.abs(landmarks) > EPSILON,
        axis=1
    )

    landmarks[detected] = (
        landmarks[detected] - center
    ) / scale

    return landmarks.flatten().astype(np.float32)