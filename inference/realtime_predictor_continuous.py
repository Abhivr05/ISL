import sys

from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont
from PyQt5.QtWidgets import (
    QApplication,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QHBoxLayout,
    QWidget,
)


class ISLWindow(QWidget):

    def __init__(self):
        super().__init__()

        self.setWindowTitle(
            "Real-Time Indian Sign Language Recognition"
        )

        self.setMinimumSize(1100, 700)

        self.setup_ui()


    def setup_ui(self):

        self.setStyleSheet("""
            QWidget {
                background-color: #101820;
                color: white;
            }

            QLabel {
                color: white;
            }

            QPushButton {
                background-color: #263238;
                color: white;
                border: 1px solid #455A64;
                padding: 10px 20px;
                border-radius: 6px;
                font-size: 14px;
            }

            QPushButton:hover {
                background-color: #37474F;
            }
        """)

        # ====================================================
        # TITLE
        # ====================================================

        title = QLabel(
            "REAL-TIME INDIAN SIGN LANGUAGE RECOGNITION"
        )

        title.setAlignment(
            Qt.AlignCenter
        )

        title.setFont(
            QFont("Arial", 20, QFont.Bold)
        )


        # ====================================================
        # CAMERA PLACEHOLDER
        # ====================================================

        camera = QLabel(
            "WEBCAM\n\nCamera feed will appear here"
        )

        camera.setAlignment(
            Qt.AlignCenter
        )

        camera.setFont(
            QFont("Arial", 18)
        )

        camera.setStyleSheet("""
            background-color: #000000;
            border: 2px solid #455A64;
            border-radius: 8px;
        """)


        # ====================================================
        # RIGHT PANEL
        # ====================================================

        panel = QVBoxLayout()


        status_title = QLabel("STATUS")

        status_title.setFont(
            QFont("Arial", 11, QFont.Bold)
        )

        status = QLabel(
            "● Listening..."
        )

        status.setFont(
            QFont("Arial", 16, QFont.Bold)
        )


        # ----------------------------------------------------
        # Detected word
        # ----------------------------------------------------

        detected_title = QLabel(
            "DETECTED WORD"
        )

        detected_title.setFont(
            QFont("Arial", 11, QFont.Bold)
        )

        detected_word = QLabel(
            "Waiting..."
        )

        detected_word.setAlignment(
            Qt.AlignCenter
        )

        detected_word.setFont(
            QFont("Arial", 30, QFont.Bold)
        )


        confidence = QLabel(
            "Confidence: --"
        )

        confidence.setAlignment(
            Qt.AlignCenter
        )


        # ----------------------------------------------------
        # Recognized words
        # ----------------------------------------------------

        words_title = QLabel(
            "RECOGNIZED WORDS"
        )

        words_title.setFont(
            QFont("Arial", 11, QFont.Bold)
        )

        words = QLabel(
            "No words yet"
        )

        words.setWordWrap(
            True
        )

        words.setFont(
            QFont("Arial", 16)
        )


        # ----------------------------------------------------
        # Sentence
        # ----------------------------------------------------

        sentence_title = QLabel(
            "GENERATED SENTENCE"
        )

        sentence_title.setFont(
            QFont("Arial", 11, QFont.Bold)
        )

        sentence = QLabel(
            "Start signing..."
        )

        sentence.setWordWrap(
            True
        )

        sentence.setFont(
            QFont("Arial", 18, QFont.Bold)
        )


        # Add everything

        panel.addWidget(status_title)
        panel.addWidget(status)

        panel.addSpacing(20)

        panel.addWidget(detected_title)
        panel.addWidget(detected_word)
        panel.addWidget(confidence)

        panel.addSpacing(20)

        panel.addWidget(words_title)
        panel.addWidget(words)

        panel.addSpacing(20)

        panel.addWidget(sentence_title)
        panel.addWidget(sentence)

        panel.addStretch()


        # ====================================================
        # BUTTONS
        # ====================================================

        clear_button = QPushButton(
            "Clear Sentence"
        )

        clear_button.clicked.connect(
            lambda: (
                words.setText("No words yet"),
                sentence.setText("Start signing...")
            )
        )


        exit_button = QPushButton(
            "Exit"
        )

        exit_button.clicked.connect(
            self.close
        )


        buttons = QHBoxLayout()

        buttons.addWidget(
            clear_button
        )

        buttons.addWidget(
            exit_button
        )


        panel.addLayout(
            buttons
        )


        # ====================================================
        # MAIN LAYOUT
        # ====================================================

        content = QHBoxLayout()

        content.addWidget(
            camera,
            3
        )

        right_panel = QWidget()

        right_panel.setLayout(
            panel
        )

        content.addWidget(
            right_panel,
            1
        )


        main_layout = QVBoxLayout(
            self
        )

        main_layout.addWidget(
            title
        )

        main_layout.addLayout(
            content
        )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    app = QApplication(
        sys.argv
    )

    window = ISLWindow()

    window.show()

    sys.exit(
        app.exec_()
    )
    