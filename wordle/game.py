"""Sections 2-3: Wordle environment and state representation.

The environment is responsible for:
- storing the visible game state
- applying a guess to the fixed target
- recording the resulting feedback
- maintaining simple letter knowledge
- exposing the remaining guess budget

The search agent is responsible for candidate filtering.  When the agent
updates its candidate set, it should synchronize ``state.candidate_words``
using ``set_candidate_words()``.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .feedback import ALL_GREEN, feedback, pattern_to_str


WORD_LENGTH = 5
# A blank row keeps the board shape consistent before a guess is made.
EMPTY_WORD = " " * WORD_LENGTH
# Feedback uses grey, yellow, and green markers for absent, misplaced, and
# correctly placed letters respectively.
VALID_LETTER_MARKS = {"-", "Y", "G"}


def _validate_target(target: str) -> None:
    """Validate the fixed Wordle target when the environment is created."""
    if not isinstance(target, str):
        raise TypeError("target must be a string")

    if len(target) != WORD_LENGTH:
        raise ValueError(
            f"target must contain exactly {WORD_LENGTH} characters; "
            f"got {len(target)}"
        )

    if not target.isascii() or not target.isalpha():
        raise ValueError("target must contain ASCII alphabetic characters only")

    if target != target.lower():
        raise ValueError("target must be lowercase")


@dataclass
class GameState:
    """Complete observable state used by the Wordle agent/environment.

    ``candidate_words`` is the agent's current belief about which answers
    remain possible.  The environment cannot calculate this set by itself
    because it does not own the answer vocabulary, so the agent must
    synchronize it after each observation.
    """

    max_guesses: int = 6
    previous_guesses: list[str] = field(default_factory=list)
    feedback: list[int] = field(default_factory=list)
    known_letters: set[str] = field(default_factory=set)
    excluded_letters: set[str] = field(default_factory=set)
    candidate_words: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        # Validate the guess limit and keep guesses aligned with their feedback.
        if not isinstance(self.max_guesses, int):
            raise TypeError("max_guesses must be an integer")

        if self.max_guesses <= 0:
            raise ValueError("max_guesses must be greater than 0")

        if len(self.previous_guesses) != len(self.feedback):
            raise ValueError(
                "previous_guesses and feedback must contain the same number "
                "of entries"
            )

    @property
    def guesses_remaining(self) -> int:
        """Number of guesses still available."""
        # Do not expose a negative budget if a completed state is inspected.
        return max(0, self.max_guesses - len(self.previous_guesses))

    @property
    def board(self) -> list[tuple[str, str]]:
        """Return the board as ``(guess, feedback_string)`` rows.

        Unused rows are padded with blank five-character strings.
        """
        rows = [
            # Convert the stored numeric pattern into symbols for display.
            (guess, pattern_to_str(pattern))
            for guess, pattern in zip(self.previous_guesses, self.feedback)
        ]

        # Pad the board with blank rows until it reaches the guess limit.
        return rows + [
            (EMPTY_WORD, EMPTY_WORD)
            for _ in range(self.guesses_remaining)
        ]

    def set_candidate_words(self, candidates) -> None:
        """Update the agent's current candidate set.

        The environment does not own the answer vocabulary, so it cannot
        calculate this set by itself.  The agent must synchronize it after
        each observation.
        """
        # Candidate filtering belongs to the agent, but the state stores the
        # latest filtered list so other parts of the program can inspect it.
        if not isinstance(candidates, list):
            raise TypeError("candidates must be a list")

        if not all(isinstance(word, str) for word in candidates):
            raise TypeError("candidates must be a list of strings")

        self.candidate_words = candidates