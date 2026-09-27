class SentenceGenerator:
    """
    Rule-based sentence generator for recognized ISL glosses.

    Converts a sequence of recognized words into
    a simple English sentence.
    """

    def __init__(self):
        # Simple vocabulary mappings
        self.word_map = {
            "HELLO": "hello",
            "GOODBYE": "goodbye",
            "THANK_YOU": "thank you",
            "PLEASE": "please",
            "SORRY": "sorry",
            "YES": "yes",
            "NO": "no",
            "HELP": "help",
            "WELCOME": "welcome",
            "UNDERSTAND": "understand",

            "I": "I",
            "YOU": "you",
            "HE": "he",
            "SHE": "she",
            "WE": "we",

            "FRIEND": "friend",
            "FAMILY": "family",
            "NAME": "name",

            "WHAT": "what",
            "WHO": "who",
            "WHERE": "where",
            "WHEN": "when",
            "WHY": "why",
            "HOW": "how",

            "WANT": "want",
            "NEED": "need",
            "GO": "go",
            "COME": "come",
            "EAT": "eat",
            "DRINK": "drink",
            "GIVE": "give",
            "TAKE": "take",
            "LIKE": "like",
            "KNOW": "know",

            "WATER": "water",
            "FOOD": "food",
            "HOME": "home",
            "SCHOOL": "school",
            "COLLEGE": "college",
            "WORK": "work",
            "HOSPITAL": "hospital",
            "DOCTOR": "doctor",
            "TOILET": "toilet",
            "PHONE": "phone",

            "TODAY": "today",
            "TOMORROW": "tomorrow",
            "NOW": "now",
            "LATER": "later",

            "GOOD": "good",
            "BAD": "bad",
        }


    def generate(self, words):
        """
        Generate an English sentence from recognized words.

        Parameters
        ----------
        words : list
            List of recognized ISL glosses.

        Returns
        -------
        str
            Generated English sentence.
        """

        if not words:
            return ""

        # Normalize input
        words = [
            str(word).strip().upper()
            for word in words
            if word is not None and str(word).strip()
        ]

        if not words:
            return ""

        # --------------------------------------------------
        # Special rules
        # --------------------------------------------------

        # I + WANT + X
        if len(words) >= 3:
            if words[0] == "I" and words[1] == "WANT":
                object_word = self._translate(words[2])

                return f"I want {object_word}."

        # I + NEED + X
        if len(words) >= 3:
            if words[0] == "I" and words[1] == "NEED":
                object_word = self._translate(words[2])

                return f"I need {object_word}."

        # I + LIKE + X
        if len(words) >= 3:
            if words[0] == "I" and words[1] == "LIKE":
                object_word = self._translate(words[2])

                return f"I like {object_word}."

        # YOU + WANT + X
        if len(words) >= 3:
            if words[0] == "YOU" and words[1] == "WANT":
                object_word = self._translate(words[2])

                return f"You want {object_word}."

        # YOU + GOOD
        if words == ["YOU", "GOOD"]:
            return "You are good."

        # YOU + BAD
        if words == ["YOU", "BAD"]:
            return "You are bad."

        # I + GOOD
        if words == ["I", "GOOD"]:
            return "I am good."

        # I + SORRY
        if words == ["I", "SORRY"]:
            return "I am sorry."

        # I + BAD
        if words == ["I", "BAD"]:
            return "I am bad."

        # YOU + HELP
        if words == ["YOU", "HELP"]:
            return "You need help."

        # I + HELP
        if words == ["I", "HELP"]:
            return "I need help."

        # THANK_YOU + YOU
        if words == ["THANK_YOU", "YOU"]:
            return "Thank you."

        # --------------------------------------------------
        # Question rules
        # --------------------------------------------------

        if words == ["YOU", "WHAT"]:
            return "What do you want?"

        if words == ["YOU", "WHERE"]:
            return "Where are you?"

        if words == ["YOU", "HOW"]:
            return "How are you?"

        if words == ["YOU", "WHY"]:
            return "Why?"

        # --------------------------------------------------
        # Single-word output
        # --------------------------------------------------

        if len(words) == 1:
            sentence = self._translate(words[0])

            return self._capitalize(sentence) + "."

        # --------------------------------------------------
        # Generic fallback
        # --------------------------------------------------

        translated_words = [
            self._translate(word)
            for word in words
        ]

        sentence = " ".join(translated_words)

        return self._capitalize(sentence) + "."


    def _translate(self, word):
        """
        Convert an ISL gloss into its English representation.
        """

        return self.word_map.get(
            word,
            word.lower().replace("_", " ")
        )


    @staticmethod
    def _capitalize(text):
        """
        Capitalize the first character.
        """

        if not text:
            return text

        return text[0].upper() + text[1:]


# --------------------------------------------------
# Test
# --------------------------------------------------

if __name__ == "__main__":

    print("===== SENTENCE GENERATOR TEST =====")

    generator = SentenceGenerator()

    test_cases = [
        ["HELLO"],
        ["THANK_YOU"],
        ["I", "WANT", "WATER"],
        ["I", "NEED", "HELP"],
        ["I", "LIKE", "WATER"],
        ["YOU", "GOOD"],
        ["I", "GOOD"],
        ["YOU", "HELP"],
        ["YOU", "WHAT"],
        ["YOU", "WHERE"],
        ["YOU", "HOW"],
        ["GOOD", "WATER"],
    ]

    for words in test_cases:

        sentence = generator.generate(words)

        print(
            f"{words} -> {sentence}"
        )

    print("\n===== TEST COMPLETE =====")