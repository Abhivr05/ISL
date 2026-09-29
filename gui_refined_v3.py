"""

=============================================================================

REAL-TIME INDIAN SIGN LANGUAGE RECOGNITION & BASIC SENTENCE GENERATION

Framework: PyQt5 Desktop UI + OpenCV + MediaPipe Holistic + PyTorch TCN (V3)

Checkpoint: trained_models/isl_tcn_19class_v3.pth (18 Active Classes)

=============================================================================

"""



import sys

import os

import time


from collections import deque

import cv2

import numpy as np

import torch



from PyQt5.QtWidgets import (

    QApplication, QMainWindow, QWidget, QLabel, QPushButton,

    QVBoxLayout, QHBoxLayout, QGridLayout, QFrame, QSlider,

    QProgressBar, QCheckBox, QSizePolicy, QToolTip

)

from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer

from PyQt5.QtGui import QImage, QPixmap, QFont



# Core project pipeline modules

from utils.mediapipe_utils import initialize_holistic

from utils.landmark_utils import extract_landmarks

from preprocessing.normalization import normalize_landmarks

from models.tcn_model import TCNModel

from inference.prediction_smoother import PredictionSmoother

from sentence.word_buffer import WordBuffer

from sentence.sentence_generator import SentenceGenerator





# ============================================================

# VOCABULARY REFERENCE DICTIONARY (18 Classes)

# ============================================================

SIGN_DESCRIPTIONS = {

    "HELLO": "Open hand raised near forehead/temple waving outward.",

    "GOOD": "Thumbs up gesture with dominant hand held forward.",

    "HELP": "Closed fist placed on open palm, lifting slightly upwards.",

    "I": "Index finger pointing directly to the center of your chest.",

    "NO": "Index and middle finger tapping thumb together firmly.",

    "YES": "Closed fist nodding up and down like a head nod gesture.",

    "SORRY": "Fist rubbing gently in a small circle over the chest.",

    "THANK_YOU": "Flat open hand touching chin/lips and moving forward.",

    "WATER": "W-handshape (3 middle fingers extended) tapping near chin/lips.",

    "YOU": "Index finger pointing directly towards the other person.",

    "HOME": "Fingertips of both hands touching together like a roof.",

    "SCHOOL": "Flat hands clapping horizontally (dominant over non-dominant).",

    "HOSPITAL": "Cross gesture drawn on the upper arm with index/middle finger.",

    "MONEY": "Thumb rubbing fingertips repeatedly (counting currency).",

    "MEDICINE": "Middle finger rubbing circular on open palm or touching tongue.",

    "TODAY": "Both open hands bouncing downward slightly in front of body.",

    "TOMORROW": "Thumb brushing forward along the cheek towards the front.",

    "YESTERDAY": "Thumb pointing backward over the shoulder from the cheek."

}





# ============================================================

# TTS WORKER THREAD (Windows Speech API via PowerShell)

# ============================================================

class ISLInferenceWorker(QThread):

    frame_ready = pyqtSignal(QImage)
    status_updated = pyqtSignal(dict)

    def __init__(self, model_path="trained_models/isl_tcn_19class_v3.pth"):
        super().__init__()

        self.model_path = model_path
        self.running = True
        self.paused = False
        self.show_landmarks = True
        self.show_guide_box = True
        self.flip_horizontal = True

        # Keep the proven recognition pipeline.
        self.sequence_length = 30
        self.feature_dimension = 225

        self.smoothing_window = 7
        self.confidence_threshold = 0.50
        self.strong_confidence_threshold = 0.70
        self.strong_margin_threshold = 0.30
        self.min_confidence_margin = 0.15

        # Slightly lower start threshold so subtle signs are not skipped.
        # Motion is still required, but hand presence is also checked.
        self.motion_threshold = 0.015
        self.motion_burst_required = 2

        # Gesture segmentation: do not assume every sign occupies all 30
        # camera frames. The model still receives exactly 30 frames, but a
        # shorter completed gesture is temporally resampled to 30 frames.
        self.min_gesture_frames = 12
        self.release_motion_frames = 4
        self.max_gesture_frames = 45

        # Reject clearly unreliable MediaPipe pose frames before they enter
        # the sequence. Do NOT require a hand here: YES/NO can be expressed
        # mainly through pose/head movement.
        self.min_shoulder_visibility = 0.50

        # The previously working recognizer used 0 here.
        # This prevents idle frames from contaminating the gesture.
        self.pre_motion_frames = 0

        self.sentence_generate_pause = 1.5
        self.sentence_complete_pause = 3.0

        self.class_names = [
            "GOOD", "HELLO", "HELP", "I", "NO", "SORRY",
            "THANK_YOU", "WATER", "YES", "YOU",
            "HOME", "SCHOOL", "HOSPITAL", "MONEY", "MEDICINE",
            "TODAY", "TOMORROW", "YESTERDAY"
        ]

        self.word_buffer = WordBuffer()
        self.sentence_generator = SentenceGenerator()
        self.smoother = PredictionSmoother(
            window_size=self.smoothing_window,
            confidence_threshold=self.confidence_threshold
        )

        self.last_word_time = None
        self.generated_sentence = ""
        self.sentence_generated = False

    def reset_smoother(self):
        self.smoother = PredictionSmoother(
            window_size=self.smoothing_window,
            confidence_threshold=self.confidence_threshold
        )

    def landmark_quality_ok(self, results, normalized_features):
        """Return True when the current MediaPipe frame is usable for inference.

        Pose landmarks and both shoulder landmarks must be reliable enough
        for the existing normalization. Hand visibility is deliberately NOT
        required because some active classes, especially YES/NO, do not rely
        on hand landmarks.
        """
        try:
            if not results.pose_landmarks:
                return False

            pose = results.pose_landmarks.landmark
            if len(pose) < 13:
                return False

            left_shoulder = pose[11]
            right_shoulder = pose[12]

            if (
                left_shoulder.visibility < self.min_shoulder_visibility
                or right_shoulder.visibility < self.min_shoulder_visibility
            ):
                return False

            features = np.asarray(normalized_features, dtype=np.float32)
            if features.shape != (self.feature_dimension,):
                return False

            if not np.all(np.isfinite(features)):
                return False

            return True

        except Exception:
            return False

    @staticmethod
    def calculate_motion(current_features, previous_features):
        if previous_features is None:
            return 0.0

        return float(
            np.mean(np.abs(current_features - previous_features))
        )

    @staticmethod
    def calculate_sequence_motion(sequence):
        sequence = np.asarray(sequence, dtype=np.float32)

        if len(sequence) < 2:
            return 0.0

        return float(np.mean(np.abs(np.diff(sequence, axis=0))))

    def predict_sequence(self, model, device, sequence):
        sequence = np.asarray(sequence, dtype=np.float32)

        if sequence.shape != (
            self.sequence_length,
            self.feature_dimension
        ):
            return None, 0.0, 0.0, 0.0, "-"

        input_tensor = (
            torch.tensor(sequence, dtype=torch.float32)
            .unsqueeze(0)
            .to(device)
        )

        with torch.no_grad():
            output = model(input_tensor)
            probabilities = torch.softmax(output, dim=1)

            top_probabilities, top_indices = torch.topk(
                probabilities, k=2, dim=1
            )

        predicted_index = top_indices[0, 0].item()
        confidence = top_probabilities[0, 0].item()
        second_confidence = top_probabilities[0, 1].item()

        predicted_class = self.class_names[predicted_index]
        second_class = self.class_names[top_indices[0, 1].item()]

        return (
            predicted_class,
            confidence,
            second_confidence,
            confidence - second_confidence,
            second_class
        )

    def accept_word(self, word):
        added = self.word_buffer.add_word(word)

        if not added:
            return False

        self.last_word_time = time.time()
        self.generated_sentence = self.sentence_generator.generate(
            self.word_buffer.get_words()
        )
        self.sentence_generated = True

        return True

    @staticmethod
    def temporal_resample(sequence, target_length=30):
        """Resample a completed gesture to the model's fixed frame length.

        The TCN was trained on 30-frame sequences. A live sign may finish in
        fewer than 30 frames, so instead of padding the rest with idle frames,
        interpolate the complete gesture trajectory to exactly 30 frames.
        """
        sequence = np.asarray(sequence, dtype=np.float32)

        if sequence.ndim != 2 or len(sequence) == 0:
            return None

        if len(sequence) == target_length:
            return sequence.copy()

        if len(sequence) == 1:
            return np.repeat(sequence, target_length, axis=0)

        old_x = np.linspace(0.0, 1.0, len(sequence))
        new_x = np.linspace(0.0, 1.0, target_length)
        result = np.empty((target_length, sequence.shape[1]), dtype=np.float32)

        for feature_idx in range(sequence.shape[1]):
            result[:, feature_idx] = np.interp(
                new_x, old_x, sequence[:, feature_idx]
            )

        return result

    def run(self):
        device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        try:
            checkpoint = torch.load(
                self.model_path,
                map_location=device
            )

            model = TCNModel(
                input_size=self.feature_dimension,
                num_classes=len(self.class_names)
            )

            model.load_state_dict(checkpoint["model_state_dict"])
            model.to(device)
            model.eval()

        except Exception as exc:
            self.status_updated.emit({
                "state": "ERROR",
                "prediction": "Model Error",
                "confidence": 0.0,
                "second_prediction": "-",
                "second_confidence": 0.0,
                "margin": 0.0,
                "motion": 0.0,
                "buffer_len": 0,
                "words": [],
                "sentence": str(exc),
                "fps": 0.0
            })
            return

        holistic = initialize_holistic()

        import mediapipe as mp
        mp_drawing = mp.solutions.drawing_utils
        mp_holistic = mp.solutions.holistic

        landmark_spec = mp_drawing.DrawingSpec(
            color=(0, 240, 255), thickness=2, circle_radius=2
        )
        connection_spec = mp_drawing.DrawingSpec(
            color=(56, 189, 248), thickness=2, circle_radius=1
        )

        cap = cv2.VideoCapture(0)

        if not cap.isOpened():
            holistic.close()
            return

        # Candidate gesture can be shorter or slightly longer than 30 raw
        # frames. It is converted to exactly 30 frames only when classified.
        sequence_buffer = deque(maxlen=self.max_gesture_frames)
        pre_motion_buffer = deque(maxlen=5)

        previous_features = None
        state = "IDLE"
        movement_count = 0
        release_motion_count = 0
        last_accepted_prediction = None
        release_ready = True

        current_prediction = "Waiting..."
        current_confidence = 0.0
        second_prediction = "-"
        second_confidence = 0.0
        confidence_margin = 0.0

        fps_tracker = deque(maxlen=15)
        prev_time = time.time()

        while self.running:
            if self.paused:
                time.sleep(0.05)
                continue

            ret, frame = cap.read()

            if not ret:
                time.sleep(0.01)
                continue

            curr_time = time.time()
            fps = 1.0 / max(curr_time - prev_time, 1e-5)
            prev_time = curr_time
            fps_tracker.append(fps)
            avg_fps = sum(fps_tracker) / len(fps_tracker)

            if self.flip_horizontal:
                frame = cv2.flip(frame, 1)

            h, w, _ = frame.shape

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = holistic.process(rgb_frame)

            if self.show_guide_box:
                box_x1, box_y1 = int(w * 0.18), int(h * 0.12)
                box_x2, box_y2 = int(w * 0.82), int(h * 0.92)

                cv2.rectangle(
                    frame,
                    (box_x1, box_y1),
                    (box_x2, box_y2),
                    (70, 90, 120),
                    1,
                    cv2.LINE_AA
                )

                cv2.putText(
                    frame,
                    "SIGNING ZONE",
                    (box_x1 + 10, box_y1 + 22),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (148, 163, 184),
                    1,
                    cv2.LINE_AA
                )

            if self.show_landmarks:
                try:
                    if results.pose_landmarks:
                        mp_drawing.draw_landmarks(
                            frame,
                            results.pose_landmarks,
                            mp_holistic.POSE_CONNECTIONS,
                            landmark_spec,
                            connection_spec
                        )

                    if results.left_hand_landmarks:
                        mp_drawing.draw_landmarks(
                            frame,
                            results.left_hand_landmarks,
                            mp_holistic.HAND_CONNECTIONS,
                            landmark_spec,
                            connection_spec
                        )

                    if results.right_hand_landmarks:
                        mp_drawing.draw_landmarks(
                            frame,
                            results.right_hand_landmarks,
                            mp_holistic.HAND_CONNECTIONS,
                            landmark_spec,
                            connection_spec
                        )
                except Exception:
                    pass

            features = extract_landmarks(results)
            normalized_features = np.asarray(
                normalize_landmarks(features),
                dtype=np.float32
            )

            frame_quality_ok = self.landmark_quality_ok(
                results,
                normalized_features
            )

            motion = self.calculate_motion(
                normalized_features,
                previous_features
            )
            previous_features = normalized_features.copy()

            # Motion is only allowed to start a gesture when MediaPipe is
            # actually tracking the pose and at least one hand. This avoids
            # opening capture windows because of noisy body/landmark motion.
            is_moving = (
                frame_quality_ok
                and motion > self.motion_threshold
            )

            if self.pre_motion_frames > 0 and frame_quality_ok:
                pre_motion_buffer.append(
                    normalized_features.copy()
                )

            # ========================================================
            # IDLE / GESTURE CAPTURE
            # ========================================================
            if state == "IDLE":

                # Keep a small history so the first frames of a fast gesture
                # are not lost while the motion gate is arming.
                pre_motion_buffer.append(normalized_features.copy())

                if is_moving:
                    movement_count += 1
                else:
                    movement_count = 0

                if movement_count >= self.motion_burst_required:
                    state = "CAPTURING"
                    sequence_buffer.clear()
                    release_motion_count = 0
                    movement_count = 0

                    # Include the short lead-in, then continue collecting.
                    for previous_frame in pre_motion_buffer:
                        sequence_buffer.append(previous_frame)

                    if not sequence_buffer or not np.array_equal(
                        sequence_buffer[-1], normalized_features
                    ):
                        sequence_buffer.append(normalized_features.copy())

                    self.reset_smoother()

            elif state == "CAPTURING":

                if frame_quality_ok:
                    sequence_buffer.append(normalized_features.copy())

                # Detect the end of the current gesture. We deliberately do
                # this BEFORE waiting for 30 frames. If the user completes a
                # sign in 15-20 frames, those are the frames we want.
                if frame_quality_ok and motion < self.motion_threshold:
                    release_motion_count += 1
                elif is_moving:
                    release_motion_count = 0

                gesture_finished = (
                    len(sequence_buffer) >= self.min_gesture_frames
                    and release_motion_count >= self.release_motion_frames
                )

                # Safety fallback: very long gestures are classified at 45
                # frames rather than waiting indefinitely.
                force_classify = (
                    len(sequence_buffer) >= self.max_gesture_frames
                )

                # Normal case: sign has finished before the 30-frame limit.
                # Fallback: a long sign reaches the maximum capture length.
                if gesture_finished or force_classify:
                    raw_sequence = np.array(
                        sequence_buffer, dtype=np.float32
                    )

                    sequence = self.temporal_resample(
                        raw_sequence, self.sequence_length
                    )

                    if sequence is not None:
                        (
                            prediction,
                            confidence,
                            second_confidence,
                            confidence_margin,
                            second_class
                        ) = self.predict_sequence(
                            model, device, sequence
                        )

                        if prediction is not None:
                            sequence_motion = self.calculate_sequence_motion(
                                sequence
                            )

                            current_prediction = prediction
                            current_confidence = confidence
                            second_prediction = second_class

                            strong_prediction = (
                                confidence >= self.strong_confidence_threshold
                                and confidence_margin >= self.strong_margin_threshold
                            )

                            normal_prediction = (
                                confidence >= self.confidence_threshold
                                and confidence_margin >= self.min_confidence_margin
                                and sequence_motion >= self.motion_threshold
                            )

                            if strong_prediction or normal_prediction:
                                can_accept = (
                                    prediction != last_accepted_prediction
                                    or release_ready
                                )

                                if can_accept:
                                    accepted = self.accept_word(prediction)
                                    if accepted:
                                        last_accepted_prediction = prediction
                                        release_ready = False

                    # Finished gesture: return to IDLE. The 5-frame history
                    # remains available to catch the beginning of the next
                    # sign, so continuous signing does not require a long
                    # idle position between words.
                    sequence_buffer.clear()
                    state = "IDLE"
                    movement_count = 0
                    release_motion_count = 0
                    release_ready = True
                    self.reset_smoother()

                    # A new moving gesture is allowed to re-arm immediately.
                    # Same-word repetition becomes available after this
                    # completed low-motion gesture.

            # ========================================================
            # SENTENCE
            # ========================================================
            words = self.word_buffer.get_words()

            if words and self.last_word_time is not None:

                elapsed = time.time() - self.last_word_time

                if (
                    not self.sentence_generated
                    and elapsed >= self.sentence_generate_pause
                ):
                    self.generated_sentence = (
                        self.sentence_generator.generate(words)
                    )
                    self.sentence_generated = True

                if (
                    self.sentence_generated
                    and elapsed >= self.sentence_complete_pause
                ):
                    self.word_buffer.clear()
                    self.last_word_time = None
                    self.sentence_generated = False

            rgb_display = cv2.cvtColor(
                frame,
                cv2.COLOR_BGR2RGB
            )

            q_img = QImage(
                rgb_display.data,
                w,
                h,
                3 * w,
                QImage.Format_RGB888
            )

            self.frame_ready.emit(q_img.copy())

            self.status_updated.emit({
                "state": state,
                "prediction": current_prediction,
                "confidence": current_confidence,
                "second_prediction": second_prediction,
                "second_confidence": second_confidence,
                "margin": confidence_margin,
                "motion": motion,
                "buffer_len": len(sequence_buffer),
                "words": words,
                "sentence": self.generated_sentence,
                "fps": avg_fps
            })

        cap.release()
        holistic.close()

    def stop(self):
        self.running = False
        self.wait()


class ISLMainWindow(QMainWindow):

    def __init__(self):

        super().__init__()

        self.setWindowTitle("Indian Sign Language Recognition & Sentence Generator")

        self.setMinimumSize(1260, 800)




        # Initialize Worker Thread first so UI can read default thresholds

        self.worker = ISLInferenceWorker()

        self.worker.frame_ready.connect(self.update_video_frame)

        self.worker.status_updated.connect(self.update_status_dashboard)




        self.init_styles()

        self.init_ui()



        # Start background inference loop

        self.worker.start()



    def init_styles(self):

        self.setStyleSheet("""

            QMainWindow {

                background-color: #0b0f19;

            }

            QWidget {

                color: #e2e8f0;

                font-family: 'Segoe UI', -apple-system, BlinkMacSystemFont, Roboto, sans-serif;

            }

            QFrame.card {

                background-color: #131b2e;

                border: 1px solid #1e293b;

                border-radius: 12px;

                padding: 12px;

            }

            QLabel.section-title {

                color: #94a3b8;

                font-size: 11px;

                font-weight: 700;

                letter-spacing: 1px;

                text-transform: uppercase;

                margin-bottom: 4px;

            }

            QLabel.highlight-title {

                color: #38bdf8;

                font-size: 26px;

                font-weight: 800;

            }

            QPushButton.primary-btn {

                background-color: #0284c7;

                color: #ffffff;

                font-weight: 600;

                border: none;

                border-radius: 8px;

                padding: 8px 16px;

                font-size: 13px;

            }

            QPushButton.primary-btn:hover {

                background-color: #0369a1;

            }

            QPushButton.secondary-btn {

                background-color: #1e293b;

                color: #cbd5e1;

                border: 1px solid #334155;

                border-radius: 8px;

                padding: 6px 14px;

                font-size: 12px;

                font-weight: 600;

            }

            QPushButton.secondary-btn:hover {

                background-color: #334155;

                color: #ffffff;

            }

            QPushButton.chip-btn {

                background-color: #0f172a;

                color: #38bdf8;

                border: 1px solid #0284c7;

                border-radius: 14px;

                padding: 4px 12px;

                font-weight: 600;

                font-size: 12px;

            }

            QPushButton.chip-btn:hover {

                background-color: #1e293b;

                color: #7dd3fc;

            }

            QProgressBar {

                border: none;

                background-color: #1e293b;

                border-radius: 4px;

                height: 8px;

                text-align: right;

            }

            QProgressBar::chunk {

                background-color: #38bdf8;

                border-radius: 4px;

            }

            QSlider::groove:horizontal {

                height: 4px;

                background: #1e293b;

                border-radius: 2px;

            }

            QSlider::sub-page:horizontal {

                background: #0284c7;

                border-radius: 2px;

            }

            QSlider::handle:horizontal {

                background: #38bdf8;

                border: 2px solid #0284c7;

                width: 14px;

                margin-top: -5px;

                margin-bottom: -5px;

                border-radius: 7px;

            }

            QCheckBox {

                color: #cbd5e1;

                font-size: 12px;

            }

            QCheckBox::indicator {

                width: 16px;

                height: 16px;

                border-radius: 4px;

                border: 1px solid #475569;

                background: #0f172a;

            }

            QCheckBox::indicator:checked {

                background-color: #0284c7;

                border-color: #38bdf8;

            }

            QToolTip {

                background-color: #0f172a;

                color: #f8fafc;

                border: 1px solid #334155;

                padding: 6px;

                border-radius: 6px;

                font-size: 12px;

            }

        """)



    def init_ui(self):

        central_widget = QWidget()

        self.setCentralWidget(central_widget)

        main_layout = QVBoxLayout(central_widget)

        main_layout.setContentsMargins(18, 14, 18, 14)

        main_layout.setSpacing(14)



        # ----------------------------------------------------

        # TOP HEADER

        # ----------------------------------------------------

        header_layout = QHBoxLayout()

        header_title_layout = QVBoxLayout()



        title_label = QLabel("REAL-TIME INDIAN SIGN LANGUAGE RECOGNITION")

        title_label.setFont(QFont("Segoe UI", 16, QFont.Bold))

        title_label.setStyleSheet("color: #f8fafc; font-weight: 800; letter-spacing: 0.5px;")



        subtitle_label = QLabel("Dilated TCN Deep Learning Model • MediaPipe 225 Landmarks • Basic Sentence Formation")

        subtitle_label.setStyleSheet("color: #64748b; font-size: 12px;")



        header_title_layout.addWidget(title_label)

        header_title_layout.addWidget(subtitle_label)

        header_layout.addLayout(header_title_layout)



        header_layout.addStretch()



        # Header Badges

        self.fps_badge = QLabel("FPS: --")

        self.fps_badge.setStyleSheet("background-color: #1e293b; color: #38bdf8; padding: 5px 12px; border-radius: 8px; font-weight: 700; font-size: 12px;")

        header_layout.addWidget(self.fps_badge)



        model_badge = QLabel("V3 Checkpoint (18 Classes)")

        model_badge.setStyleSheet("background-color: #064e3b; color: #34d399; padding: 5px 12px; border-radius: 8px; font-weight: 700; font-size: 12px;")

        header_layout.addWidget(model_badge)



        self.live_status_badge = QLabel("● LIVE CAMERA")

        self.live_status_badge.setStyleSheet("background-color: #1e293b; color: #10b981; padding: 5px 12px; border-radius: 8px; font-weight: 700; font-size: 12px;")

        header_layout.addWidget(self.live_status_badge)



        main_layout.addLayout(header_layout)



        # ----------------------------------------------------

        # BODY: TWO-COLUMN WORKSPACE

        # ----------------------------------------------------

        content_layout = QHBoxLayout()

        content_layout.setSpacing(16)



        # LEFT COLUMN: Video Viewport & Camera Tools

        left_column = QVBoxLayout()

        left_column.setSpacing(12)



        # Camera Viewport Card

        camera_card = QFrame()

        camera_card.setObjectName("cameraCard")

        camera_card.setProperty("class", "card")

        camera_card.setStyleSheet("background-color: #0f172a; border: 1px solid #1e293b; border-radius: 12px;")

        cam_layout = QVBoxLayout(camera_card)

        cam_layout.setContentsMargins(10, 10, 10, 10)



        self.video_viewport = QLabel()

        self.video_viewport.setMinimumSize(640, 480)

        self.video_viewport.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        self.video_viewport.setAlignment(Qt.AlignCenter)

        self.video_viewport.setStyleSheet("background-color: #020617; border-radius: 8px;")

        cam_layout.addWidget(self.video_viewport)



        # Controls under camera

        cam_tools_layout = QHBoxLayout()

        self.landmarks_checkbox = QCheckBox("MediaPipe Landmarks")

        self.landmarks_checkbox.setChecked(True)

        self.landmarks_checkbox.toggled.connect(self.on_toggle_landmarks)

        cam_tools_layout.addWidget(self.landmarks_checkbox)



        self.guide_checkbox = QCheckBox("Signing Guide Zone")

        self.guide_checkbox.setChecked(True)

        self.guide_checkbox.toggled.connect(self.on_toggle_guide)

        cam_tools_layout.addWidget(self.guide_checkbox)



        cam_tools_layout.addStretch()



        self.pause_cam_btn = QPushButton("Pause Camera")

        self.pause_cam_btn.setProperty("class", "secondary-btn")

        self.pause_cam_btn.clicked.connect(self.on_toggle_pause_camera)

        cam_tools_layout.addWidget(self.pause_cam_btn)



        cam_layout.addLayout(cam_tools_layout)

        left_column.addWidget(camera_card, stretch=4)



        content_layout.addLayout(left_column, stretch=6)



        # RIGHT COLUMN: Recognition, Buffer & Sentence Dashboard

        right_column = QVBoxLayout()

        right_column.setSpacing(14)



        # CARD 1: Live Recognition & Confidence

        recog_card = QFrame()

        recog_card.setProperty("class", "card")

        recog_layout = QVBoxLayout(recog_card)

        recog_layout.setSpacing(8)



        recog_head = QHBoxLayout()

        r_title = QLabel("REAL-TIME PREDICTION")

        r_title.setProperty("class", "section-title")

        recog_head.addWidget(r_title)



        recog_head.addStretch()



        self.state_badge = QLabel("IDLE")

        self.state_badge.setStyleSheet("background-color: #334155; color: #cbd5e1; padding: 2px 8px; border-radius: 6px; font-size: 11px; font-weight: 700;")

        recog_head.addWidget(self.state_badge)

        recog_layout.addLayout(recog_head)



        pred_row = QHBoxLayout()

        self.prediction_label = QLabel("Waiting...")

        self.prediction_label.setProperty("class", "highlight-title")

        pred_row.addWidget(self.prediction_label)

        pred_row.addStretch()



        self.confidence_label = QLabel("0.0%")

        self.confidence_label.setStyleSheet("color: #38bdf8; font-size: 24px; font-weight: 800;")

        pred_row.addWidget(self.confidence_label)

        recog_layout.addLayout(pred_row)



        self.confidence_bar = QProgressBar()

        self.confidence_bar.setRange(0, 100)

        self.confidence_bar.setValue(0)

        recog_layout.addWidget(self.confidence_bar)



        # Meta metrics

        meta_layout = QHBoxLayout()

        self.second_label = QLabel("Runner-up: -")

        self.second_label.setStyleSheet("color: #94a3b8; font-size: 11px;")

        meta_layout.addWidget(self.second_label)



        self.margin_label = QLabel("Margin: +0.00")

        self.margin_label.setStyleSheet("color: #94a3b8; font-size: 11px;")

        meta_layout.addWidget(self.margin_label)



        meta_layout.addStretch()



        self.buffer_prog_label = QLabel("Motion: 0.000")

        self.buffer_prog_label.setStyleSheet("color: #94a3b8; font-size: 11px;")

        meta_layout.addWidget(self.buffer_prog_label)

        recog_layout.addLayout(meta_layout)



        right_column.addWidget(recog_card)



        # CARD 2: Active Word Buffer

        buf_card = QFrame()

        buf_card.setProperty("class", "card")

        buf_layout = QVBoxLayout(buf_card)

        buf_layout.setSpacing(10)



        buf_header = QHBoxLayout()

        b_title = QLabel("RECOGNIZED WORD BUFFER")

        b_title.setProperty("class", "section-title")

        buf_header.addWidget(b_title)

        buf_header.addStretch()



        self.backspace_btn = QPushButton("⌫ Backspace")

        self.backspace_btn.setProperty("class", "secondary-btn")

        self.backspace_btn.clicked.connect(self.on_backspace)

        buf_header.addWidget(self.backspace_btn)



        self.clear_btn = QPushButton("↺ Clear")

        self.clear_btn.setProperty("class", "secondary-btn")

        self.clear_btn.clicked.connect(self.on_clear_buffer)

        buf_header.addWidget(self.clear_btn)

        buf_layout.addLayout(buf_header)



        self.words_display = QLabel("No signs recorded yet. Perform a sign in the camera zone.")

        self.words_display.setWordWrap(True)

        self.words_display.setStyleSheet("""

            background-color: #0b0f19;

            border: 1px dashed #334155;

            border-radius: 8px;

            padding: 12px;

            font-size: 14px;

            font-weight: 600;

            color: #38bdf8;

            min-height: 48px;

        """)

        buf_layout.addWidget(self.words_display)

        right_column.addWidget(buf_card)



        # CARD 3: Translated English Sentence & Audio

        sentence_card = QFrame()

        sentence_card.setProperty("class", "card")

        sentence_layout = QVBoxLayout(sentence_card)

        sentence_layout.setSpacing(10)



        s_header = QHBoxLayout()

        s_title = QLabel("ENGLISH SENTENCE TRANSLATION")

        s_title.setProperty("class", "section-title")

        s_header.addWidget(s_title)

        s_header.addStretch()



        self.sentence_box = QLabel("Sentence will appear after the signing pause...")

        self.sentence_box.setWordWrap(True)

        self.sentence_box.setStyleSheet("""

            background-color: #0b0f19;

            border: 1px solid #1e293b;

            border-left: 4px solid #10b981;

            border-radius: 8px;

            padding: 14px;

            font-size: 17px;

            font-weight: 700;

            color: #f8fafc;

            min-height: 52px;

        """)

        sentence_layout.addWidget(self.sentence_box)
        # Actions
        actions_layout = QHBoxLayout()

        self.copy_btn = QPushButton("📋 Copy Text")
        self.copy_btn.setProperty("class", "secondary-btn")
        self.copy_btn.clicked.connect(self.on_copy_sentence)
        actions_layout.addWidget(self.copy_btn)

        actions_layout.addStretch()
        sentence_layout.addLayout(actions_layout)


        right_column.addWidget(sentence_card)



        right_column.addStretch()

        content_layout.addLayout(right_column, stretch=4)

        main_layout.addLayout(content_layout)



    # --------------------------------------------------------

    # SLOTS & SIGNAL HANDLERS

    # --------------------------------------------------------

    def update_video_frame(self, q_img):

        pixmap = QPixmap.fromImage(q_img)

        scaled_pixmap = pixmap.scaled(

            self.video_viewport.size(),

            Qt.KeepAspectRatio,

            Qt.SmoothTransformation

        )

        self.video_viewport.setPixmap(scaled_pixmap)



    def update_status_dashboard(self, status):

        self.fps_badge.setText(f"FPS: {status['fps']:.1f}")



        state = status["state"]

        if state == "CAPTURING":

            self.state_badge.setText(f"CAPTURING ({status['buffer_len']}/30)")

            self.state_badge.setStyleSheet("background-color: #854d0e; color: #fef08a; padding: 2px 8px; border-radius: 6px; font-weight: 700; font-size: 11px;")

        elif state == "IDLE":

            self.state_badge.setText("IDLE / LISTENING")

            self.state_badge.setStyleSheet("background-color: #1e293b; color: #94a3b8; padding: 2px 8px; border-radius: 6px; font-weight: 700; font-size: 11px;")



        pred = status["prediction"]

        conf = status["confidence"]

        self.prediction_label.setText(pred)



        conf_pct = int(conf * 100)

        self.confidence_label.setText(f"{conf_pct}%")

        self.confidence_bar.setValue(conf_pct)



        # Dynamic Bar Color

        if conf >= 0.70:

            self.confidence_bar.setStyleSheet("QProgressBar::chunk { background-color: #10b981; }")

            self.prediction_label.setStyleSheet("color: #10b981; font-size: 26px; font-weight: 800;")

        elif conf >= 0.50:

            self.confidence_bar.setStyleSheet("QProgressBar::chunk { background-color: #38bdf8; }")

            self.prediction_label.setStyleSheet("color: #38bdf8; font-size: 26px; font-weight: 800;")

        else:

            self.confidence_bar.setStyleSheet("QProgressBar::chunk { background-color: #f59e0b; }")

            self.prediction_label.setStyleSheet("color: #f59e0b; font-size: 26px; font-weight: 800;")



        self.second_label.setText(f"Runner-up: {status['second_prediction']} ({int(status['second_confidence'] * 100)}%)")

        self.margin_label.setText(f"Margin: +{status['margin']:.2f}")

        self.buffer_prog_label.setText(f"Motion: {status['motion']:.3f}")



        # Update Word Buffer display

        words = status["words"]

        if words:

            chips = "  ".join([f"[{w}]" for w in words])

            self.words_display.setText(chips)

        else:

            self.words_display.setText("No signs recorded yet. Perform a sign in the camera zone.")



        # Update Sentence

        sentence = status["sentence"]

        if sentence:

            self.sentence_box.setText(f'"{sentence}"')

        elif not words:

            self.sentence_box.setText("Sentence will appear after the signing pause...")



    def on_toggle_landmarks(self, checked):

        self.worker.show_landmarks = checked



    def on_toggle_guide(self, checked):

        self.worker.show_guide_box = checked



    def on_toggle_pause_camera(self):

        self.worker.paused = not self.worker.paused

        if self.worker.paused:

            self.pause_cam_btn.setText("Resume Camera")

            self.live_status_badge.setText("⏸ CAMERA PAUSED")

            self.live_status_badge.setStyleSheet("background-color: #1e293b; color: #f59e0b; padding: 5px 12px; border-radius: 8px; font-weight: 700; font-size: 12px;")

        else:

            self.pause_cam_btn.setText("Pause Camera")

            self.live_status_badge.setText("● LIVE CAMERA")

            self.live_status_badge.setStyleSheet("background-color: #1e293b; color: #10b981; padding: 5px 12px; border-radius: 8px; font-weight: 700; font-size: 12px;")



    def on_backspace(self):

        self.worker.word_buffer.remove_last_word()



    def on_clear_buffer(self):

        self.worker.word_buffer.clear()

        self.sentence_box.setText("Sentence will appear after the signing pause...")



    def on_copy_sentence(self):

        text = self.sentence_box.text().replace('"', '').strip()

        if text and not text.startswith("Sentence will be"):

            clipboard = QApplication.clipboard()

            clipboard.setText(text)

            self.copy_btn.setText("✓ Copied!")

            QTimer.singleShot(1500, lambda: self.copy_btn.setText("📋 Copy Text"))



    def closeEvent(self, event):

        self.worker.stop()

        event.accept()





# ============================================================

# ENTRY POINT

# ============================================================

def main():

    app = QApplication(sys.argv)

    window = ISLMainWindow()

    window.show()

    sys.exit(app.exec_())





if __name__ == "__main__":

    main()