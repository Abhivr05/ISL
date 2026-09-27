
import cv2
import numpy as np
import torch
from collections import deque

from utils.mediapipe_utils import initialize_holistic
from utils.landmark_utils import extract_landmarks
from preprocessing.normalization import normalize_landmarks

from models.tcn_model import TCNModel
from inference.prediction_smoother import PredictionSmoother

from sentence.word_buffer import WordBuffer
from sentence.sentence_generator import SentenceGenerator


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_PATH = "trained_models/isl_tcn.pth"

SEQUENCE_LENGTH = 30
FEATURE_DIMENSION = 225

# Prediction smoothing
SMOOTHING_WINDOW = 7
CONFIDENCE_THRESHOLD = 0.50

# Motion detection
MOTION_THRESHOLD = 0.015
MOTION_BURST_REQUIRED = 3

# Frames kept before movement starts
PRE_MOTION_FRAMES = 5


# ============================================================
# DEVICE
# ============================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("=" * 60)
print("REAL-TIME INDIAN SIGN LANGUAGE RECOGNITION")
print("=" * 60)
print("Device:", device)


# ============================================================
# LOAD MODEL
# ============================================================

checkpoint = torch.load(
    MODEL_PATH,
    map_location=device
)

class_names = checkpoint["class_names"]

model = TCNModel(
    input_size=checkpoint["input_size"],
    num_classes=checkpoint["num_classes"]
)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

model.to(device)
model.eval()

print("Model loaded successfully")
print("Classes:", class_names)
print("Input size:", checkpoint["input_size"])
print("Sequence length:", checkpoint["sequence_length"])
print()


# ============================================================
# INITIALIZE MEDIAPIPE
# ============================================================

holistic = initialize_holistic()


# ============================================================
# INITIALIZE PREDICTION SMOOTHER
# ============================================================

smoother = PredictionSmoother(
    window_size=SMOOTHING_WINDOW,
    confidence_threshold=CONFIDENCE_THRESHOLD
)


# ============================================================
# INITIALIZE WORD BUFFER AND SENTENCE GENERATOR
# ============================================================

word_buffer = WordBuffer()

sentence_generator = SentenceGenerator()


# ============================================================
# WEBCAM
# ============================================================

cap = cv2.VideoCapture(0)

if not cap.isOpened():

    print("ERROR: Could not open webcam.")

    holistic.close()

    raise SystemExit


# ============================================================
# RUNTIME VARIABLES
# ============================================================

# Stores the current 30-frame sequence
sequence_buffer = deque(
    maxlen=SEQUENCE_LENGTH
)

# Stores a few frames before movement
pre_motion_buffer = deque(
    maxlen=PRE_MOTION_FRAMES
)

previous_features = None

state = "IDLE"

movement_count = 0

current_prediction = "Waiting..."

current_confidence = 0.0

last_stable_word = ""

# Text shown on screen
buffered_words_text = ""

generated_sentence = ""


# ============================================================
# MOTION CALCULATION
# ============================================================

def calculate_motion(
    current_features,
    previous_features
):
    """
    Calculate the average absolute landmark movement
    between two consecutive frames.
    """

    if previous_features is None:
        return 0.0

    return float(
        np.mean(
            np.abs(
                current_features -
                previous_features
            )
        )
    )


# ============================================================
# MODEL PREDICTION
# ============================================================

def predict_sequence(sequence):
    """
    Predict a 30-frame sequence using the TCN.
    """

    sequence = np.asarray(
        sequence,
        dtype=np.float32
    )

    # Make sure the sequence is correct
    if sequence.shape != (
        SEQUENCE_LENGTH,
        FEATURE_DIMENSION
    ):

        print(
            "Invalid sequence shape:",
            sequence.shape
        )

        return None, 0.0


    # Add batch dimension
    #
    # (30, 225)
    #      ↓
    # (1, 30, 225)

    input_tensor = torch.tensor(
        sequence,
        dtype=torch.float32
    ).unsqueeze(0).to(device)


    with torch.no_grad():

        output = model(
            input_tensor
        )

        probabilities = torch.softmax(
            output,
            dim=1
        )

        confidence, prediction = torch.max(
            probabilities,
            dim=1
        )


    predicted_index = prediction.item()

    confidence = confidence.item()

    predicted_class = class_names[
        predicted_index
    ]

    return predicted_class, confidence


# ============================================================
# UPDATE WORD BUFFER DISPLAY
# ============================================================

def update_text():

    global buffered_words_text
    global generated_sentence

    # Get words from the existing WordBuffer
    words = word_buffer.get_words()

    if words:

        buffered_words_text = " ".join(words)

    else:

        buffered_words_text = "No words yet"


    # Generate sentence from buffered words

    if words:

        generated_sentence = (
            sentence_generator.generate(
                words
            )
        )

    else:

        generated_sentence = ""


# ============================================================
# MAIN LOOP
# ============================================================

while True:

    # ========================================================
    # READ FRAME
    # ========================================================

    ret, frame = cap.read()

    if not ret:

        print(
            "ERROR: Could not read webcam frame."
        )

        break


    # ========================================================
    # FLIP CAMERA
    # ========================================================
    #
    # Webcam image is normally mirrored.
    #
    # We flip it before MediaPipe so that:
    #
    # corrected camera
    #       ↓
    # MediaPipe
    #       ↓
    # model
    #
    # all use the same orientation.
    #
    # ========================================================

    frame = cv2.flip(
        frame,
        1
    )


    # ========================================================
    # RGB CONVERSION
    # ========================================================

    rgb_frame = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )


    # ========================================================
    # MEDIAPIPE HOLISTIC
    # ========================================================

    results = holistic.process(
        rgb_frame
    )


    # ========================================================
    # EXTRACT 225 FEATURES
    # ========================================================

    features = extract_landmarks(
        results
    )


    # ========================================================
    # NORMALIZE LANDMARKS
    # ========================================================

    normalized_features = normalize_landmarks(
        features
    )

    normalized_features = np.asarray(
        normalized_features,
        dtype=np.float32
    )


    # ========================================================
    # CALCULATE MOTION
    # ========================================================

    motion = calculate_motion(
        normalized_features,
        previous_features
    )

    previous_features = (
        normalized_features.copy()
    )

    is_moving = (
        motion > MOTION_THRESHOLD
    )


    # ========================================================
    # IDLE STATE
    # ========================================================

    if state == "IDLE":

        # Keep recent frames before movement
        pre_motion_buffer.append(
            normalized_features.copy()
        )


        # Count consecutive movement frames

        if is_moving:

            movement_count += 1

        else:

            movement_count = 0


        # ----------------------------------------------------
        # Start capturing
        # ----------------------------------------------------

        if movement_count >= MOTION_BURST_REQUIRED:

            state = "CAPTURING"

            sequence_buffer.clear()


            # Add frames immediately before movement

            for previous_frame in pre_motion_buffer:

                sequence_buffer.append(
                    previous_frame
                )


            # Add current frame

            sequence_buffer.append(
                normalized_features.copy()
            )

            movement_count = 0


    # ========================================================
    # CAPTURING STATE
    # ========================================================

    elif state == "CAPTURING":

        # Add current frame

        sequence_buffer.append(
            normalized_features.copy()
        )


        # ----------------------------------------------------
        # Display capture progress
        # ----------------------------------------------------

        # ----------------------------------------------------
        # When 30 frames are available
        # ----------------------------------------------------

        if len(sequence_buffer) >= SEQUENCE_LENGTH:

            sequence = np.array(
                sequence_buffer,
                dtype=np.float32
            )

            sequence = sequence[
                :SEQUENCE_LENGTH
            ]


            # =================================================
            # TCN PREDICTION
            # =================================================

            prediction, confidence = (
                predict_sequence(
                    sequence
                )
            )


            if prediction is not None:

                current_prediction = prediction

                current_confidence = confidence


                # =================================================
                # TEMPORAL SMOOTHING
                # =================================================

                stable_prediction = (
                    smoother.update(
                        prediction,
                        confidence
                    )
                )


                # =================================================
                # ACCEPT STABLE WORD
                # =================================================

                if stable_prediction is not None:

                    last_stable_word = (
                        stable_prediction
                    )


                    # Add to WordBuffer

                    word_buffer.add_word(
                        stable_prediction
                    )


                    # Update words + sentence

                    update_text()


                    # ------------------------------------------------
                    # IMPORTANT:
                    # Reset smoother after accepting a word.
                    #
                    # This prevents predictions from the previous
                    # sign affecting the next sign.
                    # ------------------------------------------------

                    smoother = PredictionSmoother(
                        window_size=SMOOTHING_WINDOW,
                        confidence_threshold=CONFIDENCE_THRESHOLD
                    )


            # =================================================
            # RESET SEQUENCE CAPTURE
            # =================================================

            sequence_buffer.clear()

            pre_motion_buffer.clear()

            movement_count = 0

            state = "IDLE"


    # ========================================================
    # DISPLAY PANEL
    # ========================================================

    # --------------------------------------------------------
    # State
    # --------------------------------------------------------

    cv2.putText(
        frame,
        f"State: {state}",
        (20, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (0, 255, 0),
        2
    )


    # --------------------------------------------------------
    # Prediction
    # --------------------------------------------------------

    cv2.putText(
        frame,
        f"Prediction: {current_prediction}",
        (20, 70),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (255, 255, 255),
        2
    )


    # --------------------------------------------------------
    # Confidence
    # --------------------------------------------------------

    cv2.putText(
        frame,
        f"Confidence: {current_confidence:.2f}",
        (20, 105),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2
    )


    # --------------------------------------------------------
    # Motion
    # --------------------------------------------------------

    cv2.putText(
        frame,
        f"Motion: {motion:.3f}",
        (20, 140),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.60,
        (255, 255, 255),
        2
    )


    # --------------------------------------------------------
    # Capture progress
    # --------------------------------------------------------

    if state == "CAPTURING":

        progress = len(
            sequence_buffer
        )

    else:

        progress = 0


    cv2.putText(
        frame,
        f"Frames: {progress}/{SEQUENCE_LENGTH}",
        (20, 175),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.60,
        (255, 255, 255),
        2
    )


    # ========================================================
    # WORD BUFFER
    # ========================================================

    cv2.putText(
        frame,
        "Buffered Words:",
        (20, 220),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (0, 255, 255),
        2
    )


    # Display words on next line

    cv2.putText(
        frame,
        buffered_words_text,
        (20, 255),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (0, 255, 255),
        2
    )


    # ========================================================
    # GENERATED SENTENCE
    # ========================================================

    cv2.putText(
        frame,
        "Sentence:",
        (20, 300),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 200, 0),
        2
    )


    # OpenCV text can become too long,
    # so display the generated sentence below.

    cv2.putText(
        frame,
        generated_sentence,
        (20, 335),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.70,
        (255, 200, 0),
        2
    )


    # ========================================================
    # CONTROLS
    # ========================================================

    cv2.putText(
        frame,
        "Q: Quit    C: Clear",
        (20, 375),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (200, 200, 200),
        1
    )


    # ========================================================
    # DISPLAY
    # ========================================================

    cv2.imshow(
        "Real-Time ISL Recognition",
        frame
    )


    # ========================================================
    # KEYBOARD
    # ========================================================

    key = cv2.waitKey(1) & 0xFF


    # --------------------------------------------------------
    # Quit
    # --------------------------------------------------------

    if key == ord("q"):

        break


    # --------------------------------------------------------
    # Clear words and sentence
    # --------------------------------------------------------

    elif key == ord("c"):

        word_buffer.clear()

        # Reset smoother too

        smoother = PredictionSmoother(
            window_size=SMOOTHING_WINDOW,
            confidence_threshold=CONFIDENCE_THRESHOLD
        )

        buffered_words_text = "No words yet"

        generated_sentence = ""

        last_stable_word = ""

        current_prediction = "Waiting..."

        current_confidence = 0.0

        sequence_buffer.clear()

        pre_motion_buffer.clear()

        movement_count = 0

        state = "IDLE"


# ============================================================
# CLEANUP
# ============================================================

cap.release()

holistic.close()

cv2.destroyAllWindows()

print()
print("Real-time recognition stopped.")
