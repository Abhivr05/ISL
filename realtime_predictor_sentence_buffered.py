import cv2
import numpy as np
import torch
from collections import deque
import time

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

MODEL_PATH = "trained_models/isl_tcn_19class.pth"

SEQUENCE_LENGTH = 30
FEATURE_DIMENSION = 225

# ------------------------------------------------------------
# Prediction smoothing
# ------------------------------------------------------------

SMOOTHING_WINDOW = 7
CONFIDENCE_THRESHOLD = 0.50

# ------------------------------------------------------------
# Motion detection
# ------------------------------------------------------------

MOTION_THRESHOLD = 0.020

# Keep this small so the next sign can start quickly.
MOTION_BURST_REQUIRED = 2

# Frames kept before movement starts.
PRE_MOTION_FRAMES = 5

# ------------------------------------------------------------
# Sentence
# ------------------------------------------------------------

# If no new word is recognized for this amount of time,
# generate the sentence.
SENTENCE_GENERATE_PAUSE = 1.5
SENTENCE_COMPLETE_PAUSE = 3.0


# ============================================================
# DEVICE
# ============================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("=" * 60)
print("REAL-TIME INDIAN SIGN LANGUAGE RECOGNITION")
print("19 CLASS + SENTENCE GENERATION")
print("=" * 60)
print("Device:", device)


# ============================================================
# CLASS NAMES
# ============================================================

class_names = [
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
    "YESTERDAY",
]


# ============================================================
# LOAD MODEL
# ============================================================

checkpoint = torch.load(
    MODEL_PATH,
    map_location=device
)

model = TCNModel(
    input_size=FEATURE_DIMENSION,
    num_classes=len(class_names)
)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

model.to(device)
model.eval()

print("Model loaded successfully")
print("Classes:", class_names)
print("Input size:", FEATURE_DIMENSION)
print("Sequence length:", SEQUENCE_LENGTH)
print()


# ============================================================
# INITIALIZE MEDIAPIPE
# ============================================================

holistic = initialize_holistic()


# ============================================================
# INITIALIZE SMOOTHER
# ============================================================

smoother = PredictionSmoother(
    window_size=SMOOTHING_WINDOW,
    confidence_threshold=CONFIDENCE_THRESHOLD
)


# ============================================================
# WORD BUFFER + SENTENCE GENERATOR
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

sequence_buffer = deque(
    maxlen=SEQUENCE_LENGTH
)

pre_motion_buffer = deque(
    maxlen=PRE_MOTION_FRAMES
)

previous_features = None

state = "IDLE"

movement_count = 0

current_prediction = "Waiting..."

current_confidence = 0.0

last_stable_word = ""

last_word_time = None

generated_sentence = ""

sentence_generated = False

sentence_completed = False


# ============================================================
# MOTION CALCULATION
# ============================================================

def calculate_motion(
    current_features,
    previous_features
):

    if previous_features is None:

        return 0.0

    return float(
        np.mean(
            np.abs(
                current_features
                -
                previous_features
            )
        )
    )


# ============================================================
# MODEL PREDICTION
# ============================================================

def predict_sequence(sequence):

    sequence = np.asarray(
        sequence,
        dtype=np.float32
    )

    if sequence.shape != (
        SEQUENCE_LENGTH,
        FEATURE_DIMENSION
    ):

        print(
            "Invalid sequence shape:",
            sequence.shape
        )

        return None, 0.0

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

    predicted_index = (
        prediction.item()
    )

    confidence = (
        confidence.item()
    )

    predicted_class = (
        class_names[
            predicted_index
        ]
    )

    return (
        predicted_class,
        confidence
    )


# ============================================================
# RESET SMOOTHER
# ============================================================

def reset_smoother():

    global smoother

    smoother = PredictionSmoother(
        window_size=SMOOTHING_WINDOW,
        confidence_threshold=CONFIDENCE_THRESHOLD
    )


# ============================================================
# UPDATE WORD BUFFER
# ============================================================

def accept_word(word):

    global last_stable_word
    global last_word_time
    global generated_sentence
    global sentence_generated
    global sentence_completed

    # --------------------------------------------------------
    # WordBuffer handles consecutive duplicates.
    # --------------------------------------------------------

    added = word_buffer.add_word(
        word
    )

    if not added:

        return False

    last_stable_word = word

    last_word_time = time.time()

    # A new word is being added to the current sentence.
    # Keep all words in the buffer while the sentence is built.
    generated_sentence = ""
    sentence_generated = False
    sentence_completed = False

    print()
    print("-" * 60)
    print(
        "WORD RECOGNIZED:",
        word
    )
    print(
        "WORDS:",
        word_buffer.get_words()
    )
    print("-" * 60)

    return True


# ============================================================
# UPDATE / COMPLETE SENTENCE
# ============================================================

def check_sentence_completion():

    global generated_sentence
    global sentence_generated
    global sentence_completed
    global last_word_time

    words = word_buffer.get_words()

    if not words:
        return

    if last_word_time is None:
        return

    elapsed = (
        time.time()
        -
        last_word_time
    )

    # Generate/update the sentence after a short pause.
    # IMPORTANT: do NOT clear the WordBuffer here.
    if (
        not sentence_generated
        and
        elapsed >= SENTENCE_GENERATE_PAUSE
    ):

        generated_sentence = (
            sentence_generator.generate(
                words
            )
        )

        sentence_generated = True

        print()
        print("-" * 60)
        print("SENTENCE UPDATED")
        print("Words:", words)
        print("Sentence:", generated_sentence)
        print("-" * 60)

    # Only clear the words after a longer pause.
    if (
        sentence_generated
        and
        elapsed >= SENTENCE_COMPLETE_PAUSE
    ):

        sentence_completed = True

        print()
        print("=" * 60)
        print("SENTENCE COMPLETED")
        print("Words:", words)
        print("Sentence:", generated_sentence)
        print("=" * 60)
        print()

        word_buffer.clear()

        last_word_time = None
        sentence_generated = False


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
    # MIRROR CAMERA
    # ========================================================

    frame = cv2.flip(
        frame,
        1
    )


    # ========================================================
    # MEDIAPIPE
    # ========================================================

    rgb_frame = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )

    results = holistic.process(
        rgb_frame
    )


    # ========================================================
    # LANDMARK EXTRACTION
    # ========================================================

    features = extract_landmarks(
        results
    )


    # ========================================================
    # NORMALIZATION
    # ========================================================

    normalized_features = (
        normalize_landmarks(
            features
        )
    )

    normalized_features = np.asarray(
        normalized_features,
        dtype=np.float32
    )


    # ========================================================
    # MOTION
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
    # IDLE
    # ========================================================

    if state == "IDLE":

        # ----------------------------------------------------
        # Always keep a few frames before movement.
        # ----------------------------------------------------

        pre_motion_buffer.append(
            normalized_features.copy()
        )


        if is_moving:

            movement_count += 1

        else:

            movement_count = 0


        # ----------------------------------------------------
        # Start gesture immediately after movement.
        # ----------------------------------------------------

        if (
            movement_count
            >=
            MOTION_BURST_REQUIRED
        ):

            state = "CAPTURING"

            sequence_buffer.clear()

            # Add pre-motion frames.
            for previous_frame in (
                pre_motion_buffer
            ):

                sequence_buffer.append(
                    previous_frame
                )

            # Add current frame.
            sequence_buffer.append(
                normalized_features.copy()
            )

            movement_count = 0

            # Reset prediction smoother for the
            # new gesture.
            reset_smoother()


    # ========================================================
    # CAPTURING
    # ========================================================

    elif state == "CAPTURING":

        sequence_buffer.append(
            normalized_features.copy()
        )


        # ----------------------------------------------------
        # 30 frames available
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
            # PREDICTION
            # =================================================

            prediction, confidence = (
                predict_sequence(
                    sequence
                )
            )


            if prediction is not None:

                current_prediction = (
                    prediction
                )

                current_confidence = (
                    confidence
                )


                # =============================================
                # TEMPORAL SMOOTHING
                # =============================================

                stable_prediction = (
                    smoother.update(
                        prediction,
                        confidence
                    )
                )


                if stable_prediction is not None:

                    accepted = accept_word(
                        stable_prediction
                    )

                    if accepted:

                        # Update sentence timer.
                        last_word_time = (
                            time.time()
                        )


            # =================================================
            # RESET FOR NEXT GESTURE
            # =================================================

            sequence_buffer.clear()

            pre_motion_buffer.clear()

            movement_count = 0

            state = "IDLE"

            reset_smoother()


    # ========================================================
    # SENTENCE COMPLETION
    # ========================================================

    check_sentence_completion()


    # ========================================================
    # DRAW LANDMARKS
    # ========================================================

    try:

        import mediapipe as mp

        mp_drawing = (
            mp.solutions.drawing_utils
        )

        mp_holistic = (
            mp.solutions.holistic
        )

        mp_drawing.draw_landmarks(
            frame,
            results.pose_landmarks,
            mp_holistic.POSE_CONNECTIONS
        )

        mp_drawing.draw_landmarks(
            frame,
            results.left_hand_landmarks,
            mp_holistic.HAND_CONNECTIONS
        )

        mp_drawing.draw_landmarks(
            frame,
            results.right_hand_landmarks,
            mp_holistic.HAND_CONNECTIONS
        )

    except Exception:

        pass


    # ========================================================
    # DISPLAY - STATE
    # ========================================================

    cv2.putText(
        frame,
        f"State: {state}",
        (20, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (0, 255, 0),
        2
    )


    # ========================================================
    # DISPLAY - PREDICTION
    # ========================================================

    cv2.putText(
        frame,
        f"Prediction: {current_prediction}",
        (20, 70),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.70,
        (255, 255, 255),
        2
    )


    # ========================================================
    # DISPLAY - CONFIDENCE
    # ========================================================

    cv2.putText(
        frame,
        f"Confidence: {current_confidence:.2f}",
        (20, 105),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.60,
        (255, 255, 255),
        2
    )


    # ========================================================
    # DISPLAY - MOTION
    # ========================================================

    cv2.putText(
        frame,
        f"Motion: {motion:.3f}",
        (20, 140),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.60,
        (255, 255, 255),
        2
    )


    # ========================================================
    # DISPLAY - FRAMES
    # ========================================================

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
    # DISPLAY - WORDS
    # ========================================================

    words = word_buffer.get_words()

    cv2.putText(
        frame,
        "Words:",
        (20, 220),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (0, 255, 255),
        2
    )


    # OpenCV cannot automatically wrap text.
    # Display up to the most recent words.
    words_text = " ".join(words)

    if len(words_text) > 65:

        words_text = (
            "..." +
            words_text[-62:]
        )


    cv2.putText(
        frame,
        words_text,
        (20, 255),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.60,
        (0, 255, 255),
        2
    )


    # ========================================================
    # DISPLAY - SENTENCE
    # ========================================================

    cv2.putText(
        frame,
        "Sentence:",
        (20, 305),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 200, 0),
        2
    )


    sentence_text = generated_sentence

    if len(sentence_text) > 65:

        sentence_text = (
            sentence_text[:62] +
            "..."
        )


    cv2.putText(
        frame,
        sentence_text,
        (20, 340),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 200, 0),
        2
    )


    # ========================================================
    # CONTROLS
    # ========================================================

    cv2.putText(
        frame,
        "Q: Quit    C: Clear",
        (20, 380),
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
    # Clear
    # --------------------------------------------------------

    elif key == ord("c"):

        word_buffer.clear()

        reset_smoother()

        sequence_buffer.clear()

        pre_motion_buffer.clear()

        movement_count = 0

        state = "IDLE"

        current_prediction = "Waiting..."

        current_confidence = 0.0

        last_stable_word = ""

        last_word_time = None

        generated_sentence = ""

        sentence_generated = False

        sentence_completed = False

        print(
            "\nCleared."
        )


# ============================================================
# CLEANUP
# ============================================================

cap.release()

holistic.close()

cv2.destroyAllWindows()

print()
print("Recognition stopped.")

