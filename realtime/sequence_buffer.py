from collections import deque

import numpy as np


class SequenceBuffer:
    """
    Maintains a fixed-length sequence of frame-level features.
    """

    def __init__(self, sequence_length=30, feature_dimension=225):
        self.sequence_length = sequence_length
        self.feature_dimension = feature_dimension

        self.buffer = deque(
            maxlen=sequence_length
        )

    def add(self, feature_vector):
        """
        Add one frame-level feature vector.
        """

        feature_vector = np.asarray(
            feature_vector,
            dtype=np.float32
        )

        if feature_vector.shape != (
            self.feature_dimension,
        ):
            raise ValueError(
                f"Expected feature vector of shape "
                f"({self.feature_dimension},), "
                f"got {feature_vector.shape}"
            )

        self.buffer.append(feature_vector)

    def is_ready(self):
        """Return True when the buffer contains enough frames."""

        return len(self.buffer) == self.sequence_length

    def get_sequence(self):
        """
        Return the current sequence.

        Shape:
            (30, 225)
        """

        if not self.is_ready():
            return None

        return np.stack(self.buffer, axis=0)

    def clear(self):
        """Clear the sequence buffer."""

        self.buffer.clear()

    def __len__(self):
        return len(self.buffer)