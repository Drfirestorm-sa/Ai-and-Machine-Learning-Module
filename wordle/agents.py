"""Sections 5-6: search-based Wordle agents.

Search formulation
------------------
* Node  : a belief state = the set of candidate answers still consistent
          with all feedback so far.
* Action: guess a word from the guess vocabulary.
* Edge  : an action leads to one of up to 243 successor nodes, one per
          feedback pattern (the partition of the candidates by feedback).
* Goal  : the all-green pattern.

`SearchAgent` is an informed (heuristic) best-first search: every legal action
at the current node is generated, scored with a heuristic h(g), and the best
one is expanded (played).  Candidate reduction (constraint filtering) prunes the
successor node using the observed feedback.  The two baselines use the SAME
constraint filtering but no heuristic, isolating the value of the heuristic.
"""
from __future__ import annotations
import numpy as np
from .feedback import pattern_matrix, N_PATTERNS, ALL_GREEN

CHUNK_ELEMS = 4_000_000   # bound temporary memory when scoring many guesses


class Engine:
    """Shared precomputation: vocabularies + (guess x answer) feedback table."""

    def __init__(self, answers, guesses):
        assert list(guesses[:len(answers)]) == list(answers)
        self.answers, self.guesses = list(answers), list(guesses)
        self.P = pattern_matrix(self.guesses, self.answers)      # (G, A) uint8
        self.guess_index = {w: i for i, w in enumerate(self.guesses)}

    def partition_counts(self, cand: np.ndarray, guess_ids: np.ndarray) -> np.ndarray:
        """For each guess, how many candidates fall in each feedback pattern -> (g,243)."""
        n = len(cand)
        step = max(1, CHUNK_ELEMS // max(n, 1))
        out = np.empty((len(guess_ids), N_PATTERNS), dtype=np.int32)
        for s in range(0, len(guess_ids), step):
            ids = guess_ids[s:s + step]
            sub = self.P[np.ix_(ids, cand)].astype(np.int32)
            sub += (np.arange(len(ids), dtype=np.int32) * N_PATTERNS)[:, None]
            out[s:s + step] = np.bincount(sub.ravel(), minlength=len(ids) * N_PATTERNS
                                          ).reshape(len(ids), N_PATTERNS)
        return out


# ----------------------------------------------------------------- heuristics
def entropy_score(counts: np.ndarray, n: int) -> np.ndarray:
    """H(g) = -sum p_i log2 p_i over feedback patterns (higher is better)."""
    p = counts / n
    with np.errstate(divide="ignore", invalid="ignore"):
        terms = np.where(counts > 0, p * np.log2(p), 0.0)
    return -terms.sum(axis=1)


def expected_size_score(counts, n):
    """Negative expected number of candidates left, sum c^2/n (higher is better)."""
    return -(counts.astype(np.float64) ** 2).sum(axis=1) / n


def minimax_score(counts, n):
    """Negative size of the worst-case partition (higher is better)."""
    return -counts.max(axis=1).astype(np.float64)


HEURISTICS = {"entropy": entropy_score,
              "expected_size": expected_size_score,
              "minimax": minimax_score}


# --------------------------------------------------------------------- agents
class BaseAgent:
    name = "base"

    def __init__(self, engine: Engine):
        self.e = engine
        self.reset()

    def reset(self):
        self.cand = np.arange(len(self.e.answers))   # candidate answer indices
        self.evals = 0                               # guess-evaluations (h computations)

    def observe(self, guess: str, pattern: int):
        """Candidate reduction: keep only answers consistent with the feedback."""
        gi = self.e.guess_index[guess]
        self.cand = self.cand[self.e.P[gi, self.cand] == pattern]

    def candidate_words(self):
        return [self.e.answers[i] for i in self.cand]

    def choose(self) -> str:
        raise NotImplementedError


class SearchAgent(BaseAgent):
    """Informed best-first search driven by a heuristic over guesses."""

    def __init__(self, engine, heuristic="entropy", first_guess=None):
        self.heuristic = heuristic
        self.score_fn = HEURISTICS[heuristic]
        self.name = f"search-{heuristic}"
        self._first_guess = first_guess        # cached root decision (same every game)
        super().__init__(engine)

    def choose(self) -> str:
        n = len(self.cand)
        if n == 1:
            return self.e.answers[self.cand[0]]
        if n == 2:                              # 50% win now, guaranteed next turn
            return self.e.answers[self.cand[0]]
        root = len(self.cand) == len(self.e.answers)
        if root and self._first_guess is not None:
            return self._first_guess
        ids = np.arange(len(self.e.guesses))
        counts = self.e.partition_counts(self.cand, ids)
        self.evals += len(ids)
        score = self.score_fn(counts, n)
        # tie-break: prefer a guess that could itself be the answer
        is_cand = np.zeros(len(ids), dtype=bool)
        is_cand[self.cand] = True                # guess id == answer id for answers
        score = score + 1e-9 * is_cand
        best = int(np.argmax(score))
        word = self.e.guesses[best]
        if root:
            self._first_guess = word
        return word


class RandomCandidateAgent(BaseAgent):
    """Baseline: constraint filtering only; guess a random consistent word."""
    name = "random-candidate"

    def __init__(self, engine, seed=0):
        self.rng = np.random.default_rng(seed)
        super().__init__(engine)

    def choose(self) -> str:
        return self.e.answers[int(self.rng.choice(self.cand))]


class AlphabeticalAgent(BaseAgent):
    """Baseline: depth-first style - first consistent word in dictionary order."""
    name = "dfs-alphabetical"

    def choose(self) -> str:
        return self.e.answers[int(self.cand[0])]
