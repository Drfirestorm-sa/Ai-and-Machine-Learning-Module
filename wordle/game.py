"""Sections 2-3: the Wordle environment and the state representation."""
from __future__ import annotations
from dataclasses import dataclass, field
from .feedback import feedback, pattern_to_str, ALL_GREEN


@dataclass
class GameState:
    """State = {board, previous_guesses, feedback, known_letters,
    excluded_letters, guesses_remaining, candidate_words}  (report Sec. 3.1)."""
    max_guesses: int = 6
    previous_guesses: list = field(default_factory=list)
    feedback: list = field(default_factory=list)          # int patterns
    known_letters: set = field(default_factory=set)       # letters seen green/yellow
    excluded_letters: set = field(default_factory=set)    # letters proven absent
    candidate_words: list = field(default_factory=list)   # filled in by the agent

    @property
    def guesses_remaining(self) -> int:
        return self.max_guesses - len(self.previous_guesses)

    @property
    def board(self) -> list:
        """6x5 board: list of (word, 'G/Y/-' string) rows, padded with blanks."""
        rows = [(g, pattern_to_str(p)) for g, p in zip(self.previous_guesses, self.feedback)]
        return rows + [("     ", "     ")] * self.guesses_remaining

    def update(self, guess: str, pattern: int) -> None:
        self.previous_guesses.append(guess)
        self.feedback.append(pattern)
        marks = pattern_to_str(pattern)
        for ch, m in zip(guess, marks):
            if m in "GY":
                self.known_letters.add(ch)
        for ch, m in zip(guess, marks):          # grey only excludes if never green/yellow
            if m == "-" and ch not in self.known_letters:
                self.excluded_letters.add(ch)


class WordleGame:
    """Environment: deterministic, single-agent, sequential, static, discrete."""

    def __init__(self, target: str, max_guesses: int = 6):
        self.target = target
        self.state = GameState(max_guesses=max_guesses)

    @property
    def solved(self) -> bool:
        return bool(self.state.feedback) and self.state.feedback[-1] == ALL_GREEN

    @property
    def over(self) -> bool:
        return self.solved or self.state.guesses_remaining <= 0

    def step(self, guess: str) -> int:
        """Apply an action (a guess); return the feedback pattern."""
        if self.over:
            raise RuntimeError("game is over")
        p = feedback(guess, self.target)
        self.state.update(guess, p)
        return p
