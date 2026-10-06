"""Feedback (transition) function for Wordle.

Pattern encoding: one base-3 digit per position, position 0 is the least
significant digit.  0 = grey (absent), 1 = yellow (present, wrong place),
2 = green (correct).  There are 3^5 = 243 possible patterns; ALL_GREEN = 242.
"""
from __future__ import annotations
import numpy as np

N_PATTERNS = 243
ALL_GREEN = 242
_POW3 = np.array([1, 3, 9, 27, 81], dtype=np.int32)


def feedback(guess: str, answer: str) -> int:
    """Reference (scalar) implementation with correct duplicate-letter handling."""
    res = [0] * 5
    remaining = {}
    for g, a in zip(guess, answer):
        if g != a:
            remaining[a] = remaining.get(a, 0) + 1
    for i, (g, a) in enumerate(zip(guess, answer)):
        if g == a:
            res[i] = 2
    for i, g in enumerate(guess):
        if res[i] == 0 and remaining.get(g, 0) > 0:
            res[i] = 1
            remaining[g] -= 1
    return sum(d * 3 ** i for i, d in enumerate(res))


def pattern_to_str(p: int) -> str:
    """e.g. 242 -> 'GGGGG', using G/Y/- for green/yellow/grey."""
    return "".join("-YG"[(p // 3 ** i) % 3] for i in range(5))


def encode(words) -> np.ndarray:
    return np.array([[ord(c) - 97 for c in w] for w in words], dtype=np.int8)


def pattern_matrix(guesses, answers, block: int = 128) -> np.ndarray:
    """Vectorised feedback for every (guess, answer) pair -> uint8 (G, A).

    Uses the rule: a non-green guess letter L at position i is yellow iff the
    number of earlier non-green guess positions holding L is smaller than the
    number of L's in the answer that were NOT matched green.
    """
    G, A = encode(guesses), encode(answers)
    nA = len(answers)
    cnt = np.zeros((nA, 26), dtype=np.int8)
    for i in range(5):
        np.add.at(cnt, (np.arange(nA), A[:, i]), 1)
    out = np.empty((len(guesses), nA), dtype=np.uint8)
    for s in range(0, len(guesses), block):
        g = G[s:s + block]                                  # (b,5)
        green = g[:, None, :] == A[None, :, :]              # (b,nA,5)
        same = g[:, :, None] == g[:, None, :]               # (b,5,5) same[b,k,i]
        pat = np.zeros((len(g), nA), dtype=np.int32)
        for i in range(5):
            matched = np.zeros((len(g), nA), dtype=np.int8)       # greens with letter g_i
            earlier = np.zeros((len(g), nA), dtype=np.int8)       # earlier non-green same letter
            for j in range(5):
                sj = same[:, j, i][:, None]
                matched += (green[:, :, j] & sj)
                if j < i:
                    earlier += ((~green[:, :, j]) & sj)
            rem = cnt[:, g[:, i]].T - matched                     # (b,nA)
            yellow = (~green[:, :, i]) & (earlier < rem)
            pat += _POW3[i] * (2 * green[:, :, i] + yellow)
        out[s:s + block] = pat
    return out
