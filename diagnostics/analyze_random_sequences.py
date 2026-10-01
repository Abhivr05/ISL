import os
import glob
import numpy as np
import torch

from models.tcn_model import TCNModel


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

MODEL_PATH = os.path.join(
    PROJECT_ROOT,
    "trained_models",
    "isl_tcn_19class_v3.pth"
)

SEQUENCE_DIR = os.path.join(
    PROJECT_ROOT,
    "diagnostics",
    "idle_sequences"
)

SEQUENCE_LENGTH = 30
FEATURE_DIMENSION = 225


# ============================================================
# EXACT V3 CLASS ORDER
# ============================================================

CLASS_NAMES = [
    "GOOD",
    "HELLO",
    "HELP",
    "I",
    "NO",
    "SORRY",
    "THANK_YOU",
    "WATER",
    "YES",
    "YOU",
    "HOME",
    "SCHOOL",
    "HOSPITAL",
    "MONEY",
    "MEDICINE",
    "TODAY",
    "TOMORROW",
    "YESTERDAY"
]


# ============================================================
# MOVEMENT STATISTICS
# ============================================================

def movement_stats(sequence):

    # 225 features
    #
    # Pose       = 99  features
    # Left hand  = 63  features
    # Right hand = 63  features
    #
    # Total = 225
    #
    # 225 / 3 = 75 landmarks

    points = sequence.reshape(
        30,
        75,
        3
    )

    groups = {

        "pose":
            points[:, 0:33, :],

        "left":
            points[:, 33:54, :],

        "right":
            points[:, 54:75, :]
    }

    results = {}

    for name, group in groups.items():

        # Difference between consecutive frames
        diffs = np.diff(
            group,
            axis=0
        )

        # Euclidean movement
        movement = np.linalg.norm(
            diffs,
            axis=2
        )

        # Average movement per frame
        frame_movement = movement.mean(
            axis=1
        )

        results[name] = float(
            frame_movement.mean()
        )

    return results


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 120)
    print(
        "V3 TCN - RANDOM / NON-SIGN PREDICTION TEST"
    )
    print("=" * 120)

    # --------------------------------------------------------
    # Device
    # --------------------------------------------------------

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print()
    print("Model :")
    print(MODEL_PATH)

    print()
    print("Input directory :")
    print(SEQUENCE_DIR)

    print()
    print("Device :", device)


    # --------------------------------------------------------
    # Load checkpoint
    # --------------------------------------------------------

    checkpoint = torch.load(
        MODEL_PATH,
        map_location=device
    )


    # --------------------------------------------------------
    # Create model
    # --------------------------------------------------------

    model = TCNModel(
        input_size=FEATURE_DIMENSION,
        num_classes=len(CLASS_NAMES)
    )


    # --------------------------------------------------------
    # Load trained weights
    # --------------------------------------------------------

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.to(device)

    model.eval()

    print()
    print("Model loaded successfully.")


    # --------------------------------------------------------
    # Find random sequences
    # --------------------------------------------------------

    files = sorted(
        glob.glob(
            os.path.join(
                SEQUENCE_DIR,
                "*.npy"
            )
        )
    )


    if not files:

        print()
        print(
            "ERROR: No .npy files found."
        )

        return


    print()
    print(
        f"Found {len(files)} random sequences."
    )


    # ========================================================
    # RESULT TABLE
    # ========================================================

    print()
    print("-" * 120)

    print(
        f"{'Sequence':<26}"
        f"{'Prediction':<15}"
        f"{'Conf.':>9}"
        f"{'Runner-up':<15}"
        f"{'Conf.':>9}"
        f"{'Margin':>9}"
        f"{'Pose':>10}"
        f"{'LH':>10}"
        f"{'RH':>10}"
    )

    print("-" * 120)


    results = []


    # ========================================================
    # PROCESS EACH RANDOM SEQUENCE
    # ========================================================

    for file_path in files:

        name = os.path.basename(
            file_path
        )

        try:

            # ------------------------------------------------
            # Load sequence
            # ------------------------------------------------

            sequence = np.asarray(
                np.load(file_path),
                dtype=np.float32
            )


            # ------------------------------------------------
            # Validate shape
            # ------------------------------------------------

            if sequence.shape != (
                SEQUENCE_LENGTH,
                FEATURE_DIMENSION
            ):

                print(
                    f"{name:<26}"
                    f"INVALID SHAPE "
                    f"{sequence.shape}"
                )

                continue


            # ------------------------------------------------
            # Convert to tensor
            # ------------------------------------------------

            tensor = torch.tensor(
                sequence,
                dtype=torch.float32
            ).unsqueeze(0)

            tensor = tensor.to(device)


            # ------------------------------------------------
            # TCN prediction
            # ------------------------------------------------

            with torch.no_grad():

                output = model(
                    tensor
                )

                probabilities = torch.softmax(
                    output,
                    dim=1
                )[0]


            # ------------------------------------------------
            # Get top 2 predictions
            # ------------------------------------------------

            values, indices = torch.topk(
                probabilities,
                k=2
            )


            top_confidence = float(
                values[0].item()
            )

            runner_confidence = float(
                values[1].item()
            )


            prediction_index = int(
                indices[0].item()
            )

            runner_index = int(
                indices[1].item()
            )


            prediction = CLASS_NAMES[
                prediction_index
            ]

            runner_up = CLASS_NAMES[
                runner_index
            ]


            # ------------------------------------------------
            # Confidence margin
            # ------------------------------------------------

            margin = (
                top_confidence
                - runner_confidence
            )


            # ------------------------------------------------
            # Movement statistics
            # ------------------------------------------------

            motion = movement_stats(
                sequence
            )


            # ------------------------------------------------
            # Print result
            # ------------------------------------------------

            print(
                f"{name:<26}"
                f"{prediction:<15}"
                f"{top_confidence * 100:>8.2f}%"
                f"{runner_up:<15}"
                f"{runner_confidence * 100:>8.2f}%"
                f"{margin * 100:>8.2f}%"
                f"{motion['pose']:>10.5f}"
                f"{motion['left']:>10.5f}"
                f"{motion['right']:>10.5f}"
            )


            results.append({

                "prediction":
                    prediction,

                "confidence":
                    top_confidence,

                "runner_up":
                    runner_up,

                "runner_confidence":
                    runner_confidence,

                "margin":
                    margin
            })


        except Exception as e:

            print(
                f"{name:<26}"
                f"ERROR: {e}"
            )


    # ========================================================
    # SUMMARY
    # ========================================================

    if not results:

        print()
        print(
            "No valid sequences were processed."
        )

        return


    print()
    print("=" * 120)
    print("SUMMARY")
    print("=" * 120)


    # --------------------------------------------------------
    # Confidence
    # --------------------------------------------------------

    confidences = [
        result["confidence"]
        for result in results
    ]


    margins = [
        result["margin"]
        for result in results
    ]


    print()

    print(
        f"Average confidence : "
        f"{np.mean(confidences) * 100:.2f}%"
    )

    print(
        f"Minimum confidence : "
        f"{np.min(confidences) * 100:.2f}%"
    )

    print(
        f"Maximum confidence : "
        f"{np.max(confidences) * 100:.2f}%"
    )


    # --------------------------------------------------------
    # Margin
    # --------------------------------------------------------

    print()

    print(
        f"Average margin     : "
        f"{np.mean(margins) * 100:.2f}%"
    )

    print(
        f"Minimum margin     : "
        f"{np.min(margins) * 100:.2f}%"
    )


    # --------------------------------------------------------
    # Prediction counts
    # --------------------------------------------------------

    print()
    print(
        "Prediction counts:"
    )


    counts = {}


    for result in results:

        prediction = result[
            "prediction"
        ]

        counts[prediction] = (
            counts.get(
                prediction,
                0
            )
            + 1
        )


    for label, count in sorted(
        counts.items(),
        key=lambda x: (
            -x[1],
            x[0]
        )
    ):

        print(
            f"  {label:<15}"
            f"{count}"
        )


    # --------------------------------------------------------
    # Finish
    # --------------------------------------------------------

    print()
    print("=" * 120)
    print(
        "RANDOM PREDICTION TEST COMPLETE"
    )
    print("=" * 120)


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()