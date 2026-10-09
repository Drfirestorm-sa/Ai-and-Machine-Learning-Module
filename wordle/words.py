"""Wordle dataset loading, cleaning, and configuration.

Datasets:
    A: Original answers; only original answers may be guessed.
    B: Expanded answers and expanded guesses using the existing allowed list.
    C: Original answers with the existing expanded guess vocabulary.
    D: Original answers with original answers plus Cambridge A2/B1 guesses.
    E: Original answers with original answers plus Cambridge/Oxford guesses.

Datasets D and E keep the original answer distribution unchanged. They
change only the legal guess vocabulary, allowing a controlled comparison of
different guess sets.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal, Sequence

WORD_LENGTH = 5
DATA_DIR = Path(__file__).resolve().parent.parent / "data"

POSSIBLE_WORDS_FILE = "possible_words.txt"
ALLOWED_WORDS_FILE = "allowed_words.txt"
CAMBRIDGE_WORDS_FILE = "cambridge_5letter.txt"
COMBINED_WORDS_FILE = "combined_5letter.txt"

DatasetName = Literal["A", "B", "C", "D", "E"]


def clean_words(raw_lines: Sequence[str]) -> list[str]:
    """Keep unique, lowercase, five-letter ASCII alphabetic words."""
    seen: set[str] = set()
    cleaned: list[str] = []

    for line in raw_lines:
        if not isinstance(line, str):
            raise TypeError("raw_lines must contain strings")

        word = line.strip().lower()

        if (
            len(word) == WORD_LENGTH
            and word.isascii()
            and word.isalpha()
            and word not in seen
        ):
            seen.add(word)
            cleaned.append(word)

    return cleaned


def _read(filename: str) -> list[str]:
    """Read one vocabulary file from the data directory."""
    path = DATA_DIR / filename

    if not path.exists():
        raise FileNotFoundError(f"Vocabulary file not found: {path}")

    if not path.is_file():
        raise OSError(f"Vocabulary path is not a file: {path}")

    return path.read_text(encoding="utf-8").splitlines()


def _load_raw_vocabularies() -> tuple[list[str], list[str]]:
    """Load and validate the original answer and allowed-guess lists."""
    answers = clean_words(_read(POSSIBLE_WORDS_FILE))
    allowed = clean_words(_read(ALLOWED_WORDS_FILE))

    if not answers:
        raise ValueError(
            "possible_words.txt produced no valid five-letter words"
        )

    if not allowed:
        raise ValueError(
            "allowed_words.txt produced no valid five-letter words"
        )

    return answers, allowed


def _expanded_guess_vocabulary(
    answers: list[str],
    allowed: list[str],
) -> list[str]:
    """Combine answers and allowed guesses, preserving order."""
    return list(dict.fromkeys([*answers, *allowed]))


def _load_vocabulary_file(filename: str) -> list[str]:
    """Read and validate an additional cleaned vocabulary list."""
    words = clean_words(_read(filename))

    if not words:
        raise ValueError(
            f"{filename} produced no valid five-letter words"
        )

    return words


def load_dataset(name: DatasetName) -> tuple[list[str], list[str]]:
    """Return possible answers and legal guesses for a dataset.

    Dataset choices:
        A: Original answer list for both answers and guesses.
        B: Expanded list for both answers and guesses.
        C: Original answers with the existing expanded guess list.
        D: Original answers with Cambridge vocabulary added to guesses.
        E: Original answers with Cambridge/Oxford vocabulary added to guesses.

    The original possible-answer list is not expanded in D or E. Every
    original answer remains a legal guess in every configuration.
    """
    answers, allowed = _load_raw_vocabularies()
    expanded_guesses = _expanded_guess_vocabulary(answers, allowed)

    if name == "A":
        return answers, answers.copy()

    if name == "B":
        return expanded_guesses.copy(), expanded_guesses

    if name == "C":
        return answers, expanded_guesses

    if name == "D":
        cambridge = _load_vocabulary_file(CAMBRIDGE_WORDS_FILE)
        guesses = list(dict.fromkeys([*answers, *cambridge]))
        return answers, guesses

    if name == "E":
        combined = _load_vocabulary_file(COMBINED_WORDS_FILE)
        guesses = list(dict.fromkeys([*answers, *combined]))
        return answers, guesses

    raise ValueError(
        f"Unknown dataset {name!r}; expected 'A', 'B', 'C', 'D', or 'E'"
    )
