"""Section 8 - Dataset / Environment: loading and cleaning vocabularies.

The dataset module separates two concepts that are important for the
experiments:

* answer vocabulary: words that may actually be selected as hidden targets
* guess vocabulary: words that the agent is allowed to play

Datasets:
    A: 2,309 answers; only those answers may be guessed.
    B: expanded answers + expanded guesses using the union of the two lists.
    C: 2,309 possible answers with the full expanded guess vocabulary.

Dataset C is especially useful scientifically because it isolates the effect
of giving the search agent a larger ACTION SPACE without also changing the
underlying target distribution.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal, Sequence

WORD_LENGTH = 5
DATA_DIR = Path(__file__).resolve().parent.parent / "data"

POSSIBLE_WORDS_FILE = "possible_words.txt"
ALLOWED_WORDS_FILE = "allowed_words.txt"
DatasetName = Literal["A", "B", "C"]


def clean_words(raw_lines: Sequence[str]) -> list[str]:
    """Clean and deduplicate raw vocabulary entries.

    Cleaning rules:
    - Strip surrounding whitespace.
    - Normalise words to lowercase.
    - Keep exactly five-letter ASCII alphabetic words.
    - Remove duplicates while preserving first-seen order.
    - Keep repeated-letter words such as ``geese``.
    """
    seen: set[str] = set()
    cleaned: list[str] = []

    for line in raw_lines:
        if not isinstance(line, str):
            raise TypeError("raw_lines must contain strings")

        word = line.strip().lower()

        # Apply every rule before adding the word so invalid entries never
        # affect the preserved order or duplicate tracking.
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
    """Read one vocabulary file and return its lines."""
    path = DATA_DIR / filename

    # Fail early with a specific message instead of exposing a later read
    # error that does not explain which vocabulary is missing.
    if not path.exists():
        raise FileNotFoundError(f"Vocabulary file not found: {path}")

    if not path.is_file():
        raise OSError(f"Vocabulary path is not a file: {path}")

    return path.read_text(encoding="utf-8").splitlines()


def _load_raw_vocabularies() -> tuple[list[str], list[str]]:
    """Load and clean the answer and allowed-guess vocabularies."""
    answers = clean_words(_read(POSSIBLE_WORDS_FILE))
    allowed = clean_words(_read(ALLOWED_WORDS_FILE))

    if not answers:
        raise ValueError("possible_words.txt produced no valid five-letter words")

    if not allowed:
        raise ValueError("allowed_words.txt produced no valid five-letter words")

    return answers, allowed


def _expanded_guess_vocabulary(
    answers: list[str],
    allowed: list[str],
) -> list[str]:
    """Return the union of answers and allowed guesses, preserving order."""
    # Answers come first so the combined list remains deterministic and
    # retains the original answer-list priority.
    return list(dict.fromkeys([*answers, *allowed]))


def load_dataset(name: DatasetName) -> tuple[list[str], list[str]]:
    """Load one of the supported Wordle dataset configurations.

    Dataset choices:
        ``A``: Original answer list for both answers and guesses.
        ``B``: Combined answer/guess list for both answers and guesses.
        ``C``: Original answer list with the combined guess list.

    Args:
        name: The dataset configuration to load.

    Returns:
        A pair containing ``(answers, guesses)``.

    Raises:
        ValueError: If ``name`` is not ``A``, ``B``, or ``C``.
    """
    answers, allowed = _load_raw_vocabularies()
    expanded_guesses = _expanded_guess_vocabulary(answers, allowed)

    if name == "A":
        # The restricted configuration keeps the action space small.
        return answers, answers.copy()
    if name == "B":
        # Both targets and playable guesses use the expanded vocabulary.
        return expanded_guesses.copy(), expanded_guesses
    if name == "C":
        # Keep the original target distribution while expanding only guesses.
        return answers, expanded_guesses

    raise ValueError(f"Unknown dataset {name!r}; expected 'A', 'B', or 'C'")