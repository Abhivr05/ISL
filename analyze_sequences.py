import os
import glob
import numpy as np


# ============================================================
# CONFIGURATION
# ============================================================

SEQUENCE_DIR = "debug_8_signs"

SEQUENCE_LENGTH = 30

MOTION_THRESHOLD = 0.015


# ============================================================
# FIND SEQUENCES
# ============================================================

files = sorted(
    glob.glob(
        os.path.join(
            SEQUENCE_DIR,
            "sequence_*.npy"
        )
    )
)


if not files:

    print("No sequence files found.")

    print()
    print(
        f"Expected files inside: {SEQUENCE_DIR}"
    )

    raise SystemExit


print()
print("=" * 70)
print("30-FRAME SEQUENCE ANALYSIS")
print("=" * 70)
print()

print(
    f"Found {len(files)} saved sequences."
)

print()


# ============================================================
# ANALYZE EACH SEQUENCE
# ============================================================

for filepath in files:

    sequence = np.load(filepath)

    filename = os.path.basename(filepath)

    print("=" * 70)

    print(f"FILE: {filename}")

    print(f"SHAPE: {sequence.shape}")

    # --------------------------------------------------------
    # Check shape
    # --------------------------------------------------------

    if sequence.shape != (30, 225):

        print(
            "WARNING: Unexpected shape!"
        )

        print()

        continue


    # --------------------------------------------------------
    # Calculate frame-to-frame motion
    # --------------------------------------------------------

    motion_values = []

    for i in range(1, SEQUENCE_LENGTH):

        previous_frame = sequence[i - 1]

        current_frame = sequence[i]

        motion = np.mean(
            np.abs(
                current_frame -
                previous_frame
            )
        )

        motion_values.append(
            float(motion)
        )


    motion_values = np.array(
        motion_values
    )


    # --------------------------------------------------------
    # Basic statistics
    # --------------------------------------------------------

    maximum_motion = np.max(
        motion_values
    )

    average_motion = np.mean(
        motion_values
    )

    active_frames = np.sum(
        motion_values > MOTION_THRESHOLD
    )


    # --------------------------------------------------------
    # Find strongest movement
    # --------------------------------------------------------

    peak_index = np.argmax(
        motion_values
    )

    # motion_values[0] represents
    # movement from frame 0 -> frame 1
    peak_frame = peak_index + 1


    # --------------------------------------------------------
    # Divide sequence into 3 sections
    # --------------------------------------------------------

    first_section = motion_values[
        0:10
    ]

    middle_section = motion_values[
        10:20
    ]

    last_section = motion_values[
        20:29
    ]


    first_active = np.sum(
        first_section >
        MOTION_THRESHOLD
    )

    middle_active = np.sum(
        middle_section >
        MOTION_THRESHOLD
    )

    last_active = np.sum(
        last_section >
        MOTION_THRESHOLD
    )


    # --------------------------------------------------------
    # Print results
    # --------------------------------------------------------

    print()

    print(
        f"Average motion : "
        f"{average_motion:.5f}"
    )

    print(
        f"Maximum motion : "
        f"{maximum_motion:.5f}"
    )

    print(
        f"Peak movement  : "
        f"between frames "
        f"{peak_frame - 1} -> {peak_frame}"
    )

    print(
        f"Active frames  : "
        f"{active_frames}/29"
    )

    print()

    print("Movement distribution:")

    print(
        f"  Frames  1-10 : "
        f"{first_active}"
    )

    print(
        f"  Frames 11-20 : "
        f"{middle_active}"
    )

    print(
        f"  Frames 21-29 : "
        f"{last_active}"
    )


    # --------------------------------------------------------
    # Print every frame's motion
    # --------------------------------------------------------

    print()

    print("Frame-to-frame motion:")

    for i, value in enumerate(
        motion_values
    ):

        marker = ""

        if value > MOTION_THRESHOLD:
            marker = "  <-- ACTIVE"

        print(
            f"  {i:02d} -> {i+1:02d} : "
            f"{value:.5f}"
            f"{marker}"
        )


    # --------------------------------------------------------
    # Interpretation
    # --------------------------------------------------------

    print()

    if active_frames <= 3:

        print(
            "WARNING: Very little movement "
            "inside this sequence."
        )

    elif last_active > (
        first_active +
        middle_active
    ):

        print(
            "NOTE: Movement is concentrated "
            "toward the END of the sequence."
        )

    elif first_active > (
        middle_active +
        last_active
    ):

        print(
            "NOTE: Movement is concentrated "
            "toward the BEGINNING of the sequence."
        )

    else:

        print(
            "Movement appears reasonably "
            "distributed across the sequence."
        )

    print()


# ============================================================
# DONE
# ============================================================

print("=" * 70)
print("ANALYSIS COMPLETE")
print("=" * 70)