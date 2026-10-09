from __future__ import annotations

import pathlib
import sys

# Add the repository root so pytest can import the local ``wordle`` package
# when tests are run directly from the repository.
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import random

import numpy as np
import pytest

from wordle.agents import (
    AlphabeticalAgent,
    Engine,
    RandomCandidateAgent,
    SearchAgent,
    entropy_score,
    expected_size_score,
    minimax_score,
)
from wordle.feedback import (
    ALL_GREEN,
    N_PATTERNS,
    feedback,
    pattern_matrix,
    pattern_to_str,
)
from wordle.game import GameState, WordleGame
from wordle.words import clean_words, load_dataset


# ---------------------------------------------------------------------------
# Feedback / transition tests


def test_known_feedback():
    """Check standard and duplicate-letter Wordle cases."""
    # An exact match is the simplest baseline for both output formats.
    assert pattern_to_str(
        feedback("crane", "crane")
    ) == "GGGGG"

    assert feedback("crane", "crane") == ALL_GREEN

    # These cases exercise repeated letters when only some occurrences match.
    # Only one 'e' occurs in "abide", so the second unmatched e is grey.
    assert pattern_to_str(
        feedback("geese", "abide")
    ) == "----G"

    assert pattern_to_str(
        feedback("speed", "abide")
    ) == "--Y-Y"

    assert pattern_to_str(
        feedback("llama", "hello")
    ) == "YY---"

    assert pattern_to_str(
        feedback("allee", "eagle")
    ) == "YY-YG"


@pytest.mark.parametrize(
    "guess, answer",
    [
        ("hello", "world"),
        ("crate", "slate"),
        ("geese", "abide"),
        ("mamma", "eagle"),
    ],
)
def test_vectorised_matches_scalar(guess, answer):
    """One-cell vectorised feedback must match the scalar implementation."""
    matrix = pattern_matrix([guess], [answer])
    assert matrix.shape == (1, 1)
    assert matrix[0, 0] == feedback(guess, answer)


def test_vectorised_matches_scalar_random_and_duplicate_cases():
    """Compare the full vectorised matrix against scalar feedback."""
    answers, _ = load_dataset("A")
    # A fixed seed keeps the randomly sampled test data reproducible.
    rng = random.Random(1)

    # Include hand-picked duplicate-letter cases alongside normal dataset words.
    guesses = (
        [rng.choice(answers) for _ in range(80)]
        + ["geese", "llama", "eerie", "mamma", "sassy"]
    )
    test_answers = (
        [rng.choice(answers) for _ in range(120)]
        + ["abide", "hello", "eagle", "alley"]
    )

    matrix = pattern_matrix(guesses, test_answers)


# ---------------------------------------------------------------------------
# Vocabulary dataset D/E integration tests


def test_datasets_d_and_e_preserve_original_answers():
    """D and E change legal guesses without changing possible answers."""
    original_answers, _ = load_dataset("A")
    answers_d, guesses_d = load_dataset("D")
    answers_e, guesses_e = load_dataset("E")

    assert answers_d == original_answers
    assert answers_e == original_answers

    # Every possible target must remain legal to guess.
    assert set(answers_d).issubset(guesses_d)
    assert set(answers_e).issubset(guesses_e)


def test_dataset_e_guess_set_contains_dataset_d_guess_set():
    """Dataset E's combined vocabulary should include all Dataset D guesses."""
    _, guesses_d = load_dataset("D")
    _, guesses_e = load_dataset("E")

    assert set(guesses_d).issubset(set(guesses_e))


def test_datasets_d_and_e_have_unique_valid_words():
    """Vocabulary integration produces unique five-letter ASCII words."""
    for dataset in ("D", "E"):
        answers, guesses = load_dataset(dataset)

        assert len(answers) == len(set(answers))
        assert len(guesses) == len(set(guesses))
        assert all(
            len(word) == 5 and word.isascii() and word.isalpha()
            for word in answers
        )
        assert all(
            len(word) == 5 and word.isascii() and word.isalpha()
            for word in guesses
        )

