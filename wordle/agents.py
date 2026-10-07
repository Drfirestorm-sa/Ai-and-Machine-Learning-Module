"""Sections 5-6: search-based Wordle agents.

Search formulation
------------------
* Node   : a belief state = the set of candidate answers still consistent
           with all feedback observed so far.
* Action : a legal guess from the guess vocabulary.
* Edge   : an action can lead to one of up to 243 feedback-pattern
           successor states.
* Goal   : receive the all-green feedback pattern.

``SearchAgent`` performs greedy one-step heuristic search: at the current
belief state, every available guess is scored using an information/partition
heuristic and the highest-scoring action is selected.  After the real
feedback is observed, candidate filtering produces the next belief state.

This is intentionally described as greedy heuristic search rather than
classical best-first search: there is no OPEN/CLOSED frontier, no explicit
search tree, and no multi-node expansion queue.

The two baselines use the SAME candidate filtering but do not score actions
with an information heuristic. This isolates the value of heuristic action
selection.
"""

from __future__ import annotations

import numpy as np

from .feedback import N_PATTERNS, pattern_matrix


CHUNK_ELEMS = 4_000_000  # Bound large temporary arrays when scoring guesses.


class Engine:
    """Shared precomputation for a Wordle vocabulary pair.

    ``answers`` defines the hidden-target space.
    ``guesses`` defines the legal action/guess space.

    ``P[g, a]`` stores the encoded feedback pattern produced by guess ``g``
    against answer ``a``.
    """

    def __init__(self, answers, guesses):
        self.answers = list(answers)
        self.guesses = list(guesses)

        if not self.answers:
            raise ValueError("answers must not be empty")

        if not self.guesses:
            raise ValueError("guesses must not be empty")

        if len(set(self.answers)) != len(self.answers):
            raise ValueError("answers must contain unique words")

        if len(set(self.guesses)) != len(self.guesses):
            raise ValueError("guesses must contain unique words")

        self.guess_index = {word: i for i, word in enumerate(self.guesses)}

        missing_answers = [
            word for word in self.answers
            if word not in self.guess_index
        ]
        if missing_answers:
            raise ValueError(
                "Every possible answer must also be a legal guess; "
                f"missing {len(missing_answers)} answer(s)"
            )

        # Explicit mapping removes the old hidden assumption that
        # answer index i must always equal guess index i.
        self.answer_guess_ids = np.array(
            [self.guess_index[word] for word in self.answers],
            dtype=np.int32,
        )

        self.P = pattern_matrix(
            self.guesses,
            self.answers,
        )

    def partition_counts(
        self,
        cand: np.ndarray,
        guess_ids: np.ndarray,
    ) -> np.ndarray:
        """Count candidate answers in each feedback partition for each guess.

        Returns:
            Integer array with shape ``(len(guess_ids), 243)``.
            Row ``g`` contains the number of candidate answers that would
            produce each possible feedback pattern if guess ``g`` were played.
        """
        cand = np.asarray(cand, dtype=np.intp)
        guess_ids = np.asarray(guess_ids, dtype=np.intp)
        if cand.ndim != 1 or guess_ids.ndim != 1:
            raise ValueError("cand and guess_ids must be one-dimensional")
        if np.any(cand < 0) or np.any(cand >= len(self.answers)):
            raise IndexError("candidate index out of range")
        if np.any(guess_ids < 0) or np.any(guess_ids >= len(self.guesses)):
            raise IndexError("guess index out of range")

        # Each row is a histogram of feedback patterns for one possible guess.
        # Indexing P by cand keeps the calculation limited to live answers.
        counts = np.zeros((len(guess_ids), N_PATTERNS), dtype=np.int32)
        for row, guess_id in enumerate(guess_ids):
            counts[row] = np.bincount(
                self.P[guess_id, cand],
                minlength=N_PATTERNS,
            )
        return counts


def entropy_score(counts: np.ndarray, n_candidates: int) -> np.ndarray:
    """Return information gain, in bits, for each partition row."""
    if n_candidates <= 0:
        raise ValueError("n_candidates must be positive")
    counts = np.asarray(counts, dtype=np.float64)
    probabilities = counts / n_candidates
    with np.errstate(divide="ignore", invalid="ignore"):
        # Empty partitions contribute zero rather than log2(0).
        return -np.sum(
            np.where(probabilities > 0, probabilities * np.log2(probabilities), 0),
            axis=1,
        )


def expected_size_score(counts: np.ndarray, n_candidates: int) -> np.ndarray:
    """Return the expected number of candidates remaining after a guess."""
    if n_candidates <= 0:
        raise ValueError("n_candidates must be positive")
    counts = np.asarray(counts, dtype=np.float64)
    return np.sum(counts * counts, axis=1) / n_candidates


def minimax_score(counts: np.ndarray) -> np.ndarray:
    """Return the size of the largest feedback partition for each guess."""
    return np.max(np.asarray(counts), axis=1)


class SearchAgent:
    """Greedy one-step search agent using an Engine's feedback table."""

    def __init__(self, engine: Engine, heuristic: str = "entropy"):
        if heuristic not in {"entropy", "expected_size", "minimax"}:
            raise ValueError(
                "heuristic must be 'entropy', 'expected_size', or 'minimax'"
            )
        self.engine = engine
        self.heuristic = heuristic
        self.evals = 0
        self.reset()

    def reset(self) -> None:
        self.cand = np.arange(len(self.engine.answers), dtype=np.intp)
        self.previous_guesses: list[int] = []
        self._first_guess = None

    def candidate_words(self) -> list[str]:
        return [self.engine.answers[i] for i in self.cand]

    def choose(self) -> str:
        if len(self.cand) == 0:
            raise RuntimeError("no candidate answers remain")
        if len(self.cand) == 1:
            # Once one answer remains, guessing it is always optimal.
            guess_id = int(self.engine.answer_guess_ids[self.cand[0]])
        else:
            # Do not spend another turn repeating a guess already observed.
            available = np.array(
                [i for i in range(len(self.engine.guesses))
                 if i not in self.previous_guesses],
                dtype=np.intp,
            )
            counts = self.engine.partition_counts(self.cand, available)
            self.evals += len(available)
            if self.heuristic == "entropy":
                scores = entropy_score(counts, len(self.cand))
                # Higher entropy means a more informative partition.
                guess_id = int(available[int(np.argmax(scores))])
            elif self.heuristic == "expected_size":
                scores = expected_size_score(counts, len(self.cand))
                # Lower expected remainder means faster candidate reduction.
                guess_id = int(available[int(np.argmin(scores))])
            else:
                scores = minimax_score(counts)
                # Minimax limits the largest group left by any feedback.
                guess_id = int(available[int(np.argmin(scores))])
        self.previous_guesses.append(guess_id)
        if self._first_guess is None:
            self._first_guess = self.engine.guesses[guess_id]
        return self.engine.guesses[guess_id]

    def observe(self, guess: str, pattern: int) -> None:
        if guess not in self.engine.guess_index:
            raise ValueError(f"unknown guess: {guess!r}")
        if not isinstance(pattern, (int, np.integer)) or not 0 <= int(pattern) < N_PATTERNS:
            raise ValueError(f"pattern must be between 0 and {N_PATTERNS - 1}")
        guess_id = self.engine.guess_index[guess]
        # Keep only answers whose precomputed feedback matches the observation.
        self.cand = self.cand[self.engine.P[guess_id, self.cand] == int(pattern)]


class RandomCandidateAgent(SearchAgent):
    """Constraint-filtering baseline that chooses a random candidate."""

    def __init__(self, engine: Engine, seed: int = 42):
        super().__init__(engine, heuristic="entropy")
        self.rng = np.random.default_rng(seed)

    def choose(self) -> str:
        if len(self.cand) == 0:
            raise RuntimeError("no candidate answers remain")
        # This baseline deliberately samples only answers still consistent
        # with the feedback; it does not use partition scores.
        guess_id = int(self.engine.answer_guess_ids[
            self.cand[self.rng.integers(len(self.cand))]
        ])
        self.previous_guesses.append(guess_id)
        if self._first_guess is None:
            self._first_guess = self.engine.guesses[guess_id]
        return self.engine.guesses[guess_id]


class AlphabeticalAgent(SearchAgent):
    """Constraint-filtering baseline that chooses the first candidate."""

    def choose(self) -> str:
        if len(self.cand) == 0:
            raise RuntimeError("no candidate answers remain")
        # Candidate order follows the answer vocabulary, giving a
        # deterministic dictionary-order baseline.
        guess_id = int(self.engine.answer_guess_ids[self.cand[0]])
        self.previous_guesses.append(guess_id)
        if self._first_guess is None:
            self._first_guess = self.engine.guesses[guess_id]
        return self.engine.guesses[guess_id]