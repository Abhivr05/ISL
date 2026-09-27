import numpy as np

from preprocessing.normalization import normalize_landmarks


def main():
    features = np.random.rand(225).astype(np.float32)

    normalized = normalize_landmarks(features)

    print("Original shape:", features.shape)
    print("Normalized shape:", normalized.shape)
    print("Original dimension:", len(features))
    print("Normalized dimension:", len(normalized))


if __name__ == "__main__":
    main()