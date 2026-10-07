"""Feedback (transition) function for Wordle.

Pattern encoding:
- one base-3 digit per position
- position 0 is the least significant digit
- 0 = grey (absent)
- 1 = yellow (present, wrong position)
- 2 = green (correct position)

There are 3^5 = 243 possible feedback patterns.
ALL_GREEN = 242 (2 + 2*3 + 2*9 + 2*27 + 2*81).
"""

from __future__ import annotations

import numpy as np


WORD_LENGTH = 5
ALPHABET_SIZE = 26

# There are three possible states for each of the five letter positions.
N_PATTERNS = 3 ** WORD_LENGTH
ALL_GREEN = N_PATTERNS - 1

# Powers of three let each position store one feedback state in one integer.
_POW3 = np.array([3**i for i in range(WORD_LENGTH)], dtype=np.int32)


def _validate_word(word: str, name: str) -> None:
    """Validate a Wordle word before calculating feedback."""
    # Check the type first so the later string operations are safe.
    if not isinstance(word, str):
        raise TypeError(f"{name} must be a string")

    # Wordle inputs must contain exactly five letters.
    if len(word) != WORD_LENGTH:
        raise ValueError(
            f"{name} must contain exactly {WORD_LENGTH} characters; "
            f"got {len(word)}"
        )

    # Restrict inputs to the ASCII alphabet used by the letter encoding.
    if not word.isascii() or not word.isalpha():
        raise ValueError(f"{name} must contain ASCII alphabetic characters only")

    # Keeping words lowercase makes letter IDs and comparisons unambiguous.
    if word != word.lower():
        raise ValueError(f"{name} must be lowercase")


def feedback(guess: str, answer: str) -> int:
    """Return the Wordle feedback pattern for a guess/answer pair.

    This scalar implementation is the reference implementation used for
    correctness testing. It uses the standard two-pass duplicate-letter rule:
    greens are assigned first, then remaining answer letters are available
    for yellow matches.
    """
    _validate_word(guess, "guess")
    _validate_word(answer, "answer")

    result = [0] * WORD_LENGTH
    remaining: dict[str, int] = {}

    # Reserve only non-green answer letters for the second pass. This prevents
    # duplicate guess letters from receiving more matches than the answer has.
    for g, a in zip(guess, answer):
        if g != a:
            remaining[a] = remaining.get(a, 0) + 1

    # Pass 1: exact matches (green).
    for i, (g, a) in enumerate(zip(guess, answer)):
        if g == a:
            result[i] = 2

    # Pass 2: remaining matches (yellow).
    for i, g in enumerate(guess):
        if result[i] == 0 and remaining.get(g, 0) > 0:
            result[i] = 1
            remaining[g] -= 1

    # Convert the five ternary states into the compact integer representation.
    return sum(d * (3**i) for i, d in enumerate(result))


def pattern_to_str(pattern: int) -> str:
    """Convert an encoded pattern to Wordle notation, e.g. 242 -> 'GGGGG'."""
    if not isinstance(pattern, (int, np.integer)):
        raise TypeError("pattern must be an integer")

    if not 0 <= int(pattern) < N_PATTERNS:
        raise ValueError(f"pattern must be between 0 and {N_PATTERNS - 1}")

    pattern = int(pattern)
    # Read one base-3 digit per position and map 0/1/2 to grey/yellow/green.
    return "".join(
        "-YG"[(pattern // (3**i)) % 3] for i in range(WORD_LENGTH)
    )


def encode(words) -> np.ndarray:
    """Encode lowercase five-letter words as integer letter IDs (a-z -> 0-25)."""
    words = list(words)
    encoded = np.empty((len(words), WORD_LENGTH), dtype=np.int8)

    for row, word in enumerate(words):
        _validate_word(word, f"words[{row}]")
        # Store each letter as a compact zero-based ID: a -> 0, ..., z -> 25.
        encoded[row] = [ord(letter) - ord("a") for letter in word]

    return encoded
