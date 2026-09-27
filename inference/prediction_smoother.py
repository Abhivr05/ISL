from collections import deque, Counter


class PredictionSmoother:
    """
    Smooth consecutive model predictions using
    a fixed-size history window.
    """

    def __init__(self, window_size=7, confidence_threshold=0.60):

        self.window_size = window_size
        self.confidence_threshold = confidence_threshold

        self.prediction_history = deque(
            maxlen=window_size
        )

    def update(self, prediction, confidence):
        """
        Add a new prediction and return the smoothed result.

        Args:
            prediction: Predicted class name.
            confidence: Model confidence between 0 and 1.

        Returns:
            Smoothed prediction or None.
        """

        # Ignore low-confidence predictions
        if confidence < self.confidence_threshold:
            return None

        self.prediction_history.append(
            prediction
        )

        if not self.prediction_history:
            return None

        # Count recent predictions
        counts = Counter(
            self.prediction_history
        )

        smoothed_prediction, count = (
            counts.most_common(1)[0]
        )

        return smoothed_prediction

    def reset(self):
        """Clear prediction history."""

        self.prediction_history.clear()


if __name__ == "__main__":

    smoother = PredictionSmoother(
        window_size=5,
        confidence_threshold=0.60
    )

    test_predictions = [
        ("HELLO", 0.90),
        ("HELLO", 0.88),
        ("YOU", 0.65),
        ("HELLO", 0.91),
        ("HELLO", 0.87),
    ]

    print("===== SMOOTHER TEST =====")

    for prediction, confidence in test_predictions:

        result = smoother.update(
            prediction,
            confidence
        )

        print(
            f"Input: {prediction:10s} "
            f"Confidence: {confidence:.2f} "
            f"→ Smoothed: {result}"
        )

    print("=========================")