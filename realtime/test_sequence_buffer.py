import numpy as np

from realtime.sequence_buffer import SequenceBuffer


def main():

    buffer = SequenceBuffer(
        sequence_length=30,
        feature_dimension=225
    )

    for frame_number in range(30):

        feature_vector = np.random.rand(
            225
        ).astype(np.float32)

        buffer.add(feature_vector)

        print(
            f"Frame {frame_number + 1}: "
            f"{len(buffer)}/30"
        )

    sequence = buffer.get_sequence()

    print("\nSequence ready:", buffer.is_ready())
    print("Sequence shape:", sequence.shape)


if __name__ == "__main__":
    main()