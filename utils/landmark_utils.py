import numpy as np


POSE_LANDMARKS = 33
HAND_LANDMARKS = 21
COORDINATES = 3

FEATURE_DIMENSION = (
    POSE_LANDMARKS * COORDINATES
    + HAND_LANDMARKS * COORDINATES
    + HAND_LANDMARKS * COORDINATES
)


def extract_pose(results):
    """Extract 33 pose landmarks as x, y, z coordinates."""

    if results.pose_landmarks:
        pose = np.array(
            [
                [landmark.x, landmark.y, landmark.z]
                for landmark in results.pose_landmarks.landmark
            ],
            dtype=np.float32
        ).flatten()
    else:
        pose = np.zeros(POSE_LANDMARKS * COORDINATES, dtype=np.float32)

    return pose


def extract_left_hand(results):
    """Extract 21 left-hand landmarks as x, y, z coordinates."""

    if results.left_hand_landmarks:
        left_hand = np.array(
            [
                [landmark.x, landmark.y, landmark.z]
                for landmark in results.left_hand_landmarks.landmark
            ],
            dtype=np.float32
        ).flatten()
    else:
        left_hand = np.zeros(HAND_LANDMARKS * COORDINATES, dtype=np.float32)

    return left_hand


def extract_right_hand(results):
    """Extract 21 right-hand landmarks as x, y, z coordinates."""

    if results.right_hand_landmarks:
        right_hand = np.array(
            [
                [landmark.x, landmark.y, landmark.z]
                for landmark in results.right_hand_landmarks.landmark
            ],
            dtype=np.float32
        ).flatten()
    else:
        right_hand = np.zeros(HAND_LANDMARKS * COORDINATES, dtype=np.float32)

    return right_hand


def extract_landmarks(results):
    """
    Extract pose, left-hand and right-hand landmarks.

    Returns:
        np.ndarray: 225-dimensional feature vector.
    """

    pose = extract_pose(results)
    left_hand = extract_left_hand(results)
    right_hand = extract_right_hand(results)

    feature_vector = np.concatenate(
        [pose, left_hand, right_hand]
    ).astype(np.float32)

    return feature_vector