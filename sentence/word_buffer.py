from collections import deque


class WordBuffer:
    """
    Stores recognized words while preventing
    repeated predictions from being added.

    Example:

        YOU
        YOU
        YOU
        WANT
        WANT
        WATER

    becomes:

        ["YOU", "WANT", "WATER"]
    """

    def __init__(self, max_words=20):
        """
        Parameters
        ----------
        max_words : int
            Maximum number of words stored in the buffer.
        """

        self.max_words = max_words

        self.words = deque(
            maxlen=max_words
        )

        self.last_word = None


    # --------------------------------------------------
    # Add Word
    # --------------------------------------------------

    def add_word(self, word):
        """
        Add a recognized word to the buffer.

        Consecutive duplicate predictions
        are ignored.

        Returns
        -------
        bool
            True if the word was added.
            False if it was ignored.
        """

        if word is None:
            return False

        word = str(word).strip().upper()

        if not word:
            return False

        # Ignore consecutive duplicate words
        if word == self.last_word:
            return False

        self.words.append(word)

        self.last_word = word

        return True


    # --------------------------------------------------
    # Get Words
    # --------------------------------------------------

    def get_words(self):
        """
        Return the current words as a list.
        """

        return list(self.words)


    # --------------------------------------------------
    # Get Sentence
    # --------------------------------------------------

    def get_text(self):
        """
        Return the current words as a space-separated string.
        """

        return " ".join(self.words)


    # --------------------------------------------------
    # Clear Buffer
    # --------------------------------------------------

    def clear(self):
        """
        Clear all stored words.
        """

        self.words.clear()

        self.last_word = None


    # --------------------------------------------------
    # Remove Last Word
    # --------------------------------------------------

    def remove_last_word(self):
        """
        Remove the most recently added word.

        Returns
        -------
        str or None
            Removed word, or None if buffer is empty.
        """

        if not self.words:
            return None

        removed_word = self.words.pop()

        # Update last_word
        if self.words:
            self.last_word = self.words[-1]
        else:
            self.last_word = None

        return removed_word


# --------------------------------------------------
# Test
# --------------------------------------------------

if __name__ == "__main__":

    print("===== WORD BUFFER TEST =====")

    buffer = WordBuffer()

    test_predictions = [
        "YOU",
        "YOU",
        "YOU",
        "WANT",
        "WANT",
        "WATER",
        "WATER",
        "GOOD",
    ]

    print("\nInput predictions:")

    for prediction in test_predictions:

        added = buffer.add_word(prediction)

        print(
            f"{prediction:10s} "
            f"-> {'Added' if added else 'Ignored'}"
        )


    print("\nStored words:")
    print(buffer.get_words())

    print("\nText:")
    print(buffer.get_text())


    print("\nRemoving last word:")

    removed = buffer.remove_last_word()

    print(f"Removed: {removed}")
    print(f"Current text: {buffer.get_text()}")


    print("\nClearing buffer:")

    buffer.clear()

    print(f"Current text: '{buffer.get_text()}'")

    print("\n===== TEST COMPLETE =====")

