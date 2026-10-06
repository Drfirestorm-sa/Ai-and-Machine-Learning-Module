"""Section 8 - Dataset / Environment: loading and cleaning the vocabularies."""
from __future__ import annotations
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def clean_words(raw_lines):
    """Section 8.2 data cleaning.

    * remove empty / invalid entries
    * normalise case and whitespace
    * keep exactly five-letter alphabetic (a-z) words
    * remove duplicates (order preserved)
    * repeated-letter words (e.g. 'geese') are deliberately KEPT
    """
    seen, out = set(), []
    for line in raw_lines:
        w = line.strip().lower()
        if len(w) == 5 and w.isascii() and w.isalpha() and w not in seen:
            seen.add(w)
            out.append(w)
    return out


def _read(name):
    return (DATA_DIR / name).read_text(encoding="utf-8").splitlines()


def load_dataset(name: str = "A"):
    """Return (answers, guesses) for an experimental dataset.

    Dataset A (standard):  answers = guesses = the 2,309 official answer words.
    Dataset B (expanded):  answers = guesses = the full ~12.9k accepted-guess list.

    `guesses` always starts with `answers` so guess index i == answer index i
    for i < len(answers); this lets the agent treat candidates as guesses.
    """
    answers_raw = clean_words(_read("possible_words.txt"))
    allowed_raw = clean_words(_read("allowed_words.txt"))
    if name.upper() == "A":
        answers = answers_raw
        guesses = list(answers)
    elif name.upper() == "B":
        union = list(dict.fromkeys(answers_raw + allowed_raw))
        answers = union
        guesses = union
    else:
        raise ValueError("dataset must be 'A' or 'B'")
    return answers, guesses
