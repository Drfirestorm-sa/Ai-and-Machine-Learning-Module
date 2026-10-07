"""
Evaluate Wordle agents on different datasets.

Usage:
    python -u evaluate.py --dataset A --games 100
    python -u evaluate.py --dataset A
    python -u evaluate.py --dataset B --games 100
    python -u evaluate.py --dataset C --games 100
"""

from __future__ import annotations

import argparse
import ctypes
import json
import os
import platform
import statistics
import sys
import time
from pathlib import Path

import numpy as np

from wordle.agents import (
    AlphabeticalAgent,
    Engine,
    RandomCandidateAgent,
    SearchAgent,
)
from wordle.words import load_dataset


RESULTS_DIR = Path("results")

DEFAULT_AGENTS = [
    "search-entropy",
    "search-expected_size",
    "search-minimax",
    "random-candidate",
    "alphabetical-candidate",
]


def process_rss_mb() -> float | None:
    try:
        if os.name == "nt":

            class PROCESS_MEMORY_COUNTERS(ctypes.Structure):
                _fields_ = [
                    ("cb", ctypes.c_ulong),
                    ("PageFaultCount", ctypes.c_ulong),
                    ("PeakWorkingSetSize", ctypes.c_size_t),
                    ("WorkingSetSize", ctypes.c_size_t),
                    ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                    ("PagefileUsage", ctypes.c_size_t),
                    ("PeakPagefileUsage", ctypes.c_size_t),
                ]

            counters = PROCESS_MEMORY_COUNTERS()
            counters.cb = ctypes.sizeof(PROCESS_MEMORY_COUNTERS)

            process = ctypes.windll.kernel32.GetCurrentProcess()

            result = ctypes.windll.psapi.GetProcessMemoryInfo(
                process,
                ctypes.byref(counters),
                counters.cb,
            )

            if result:
                return counters.WorkingSetSize / (1024 * 1024)

            return None

        import resource

        usage = resource.getrusage(resource.RUSAGE_SELF)

        if sys.platform == "darwin":
            return usage.ru_maxrss / (1024 * 1024)

        return usage.ru_maxrss / 1024

    except Exception:
        return None


def make_agent(name: str, engine: Engine):

    if name == "search-entropy":
        return SearchAgent(
            engine,
            heuristic="entropy",
        )

    if name == "search-expected_size":
        return SearchAgent(
            engine,
            heuristic="expected_size",
        )

    if name == "search-minimax":
        return SearchAgent(
            engine,
            heuristic="minimax",
        )

    if name == "random-candidate":
        return RandomCandidateAgent(engine)

    if name in {
        "alphabetical-candidate",
        "dfs-alphabetical",
    }:
        return AlphabeticalAgent(engine)

    raise ValueError(f"Unknown agent: {name}")


def play(
    agent,
    target: str,
    max_guesses: int = 6,
) -> tuple[bool, int]:

    agent.reset()

    target_id = agent.engine.answers.index(target)

    for _ in range(max_guesses):

        guess = agent.choose()

        guess_id = agent.engine.guess_index[guess]

        pattern = agent.engine.P[
            guess_id,
            target_id,
        ]

        agent.observe(
            guess,
            int(pattern),
        )

        if guess == target:
            return (
                True,
                len(agent.previous_guesses),
            )

    return False, max_guesses


def evaluate(
    agent_name: str,
    engine: Engine,
    targets: list[str],
    max_guesses: int = 6,
) -> dict:

    agent = make_agent(
        agent_name,
        engine,
    )

    attempts_all: list[int] = []
    attempts_successful: list[int] = []

    runtimes: list[float] = []
    eval_counts: list[int] = []

    errors: list[dict] = []

    wins = 0
    failures = 0

    rss_before = process_rss_mb()

    total_start = time.perf_counter()

    for game_number, target in enumerate(
        targets,
        start=1,
    ):

        eval_before = getattr(
            agent,
            "evals",
            0,
        )

        start = time.perf_counter()

        try:
            won, attempts = play(
                agent,
                target,
                max_guesses=max_guesses,
            )

        except Exception as exc:
            won = False
            attempts = max_guesses

            errors.append(
                {
                    "target": target,
                    "error": repr(exc),
                }
            )

        elapsed = (
            time.perf_counter()
            - start
        )

        eval_after = getattr(
            agent,
            "evals",
            0,
        )

        game_evals = (
            eval_after
            - eval_before
        )

        runtimes.append(elapsed)
        attempts_all.append(attempts)
        eval_counts.append(game_evals)

        if won:
            wins += 1
            attempts_successful.append(attempts)
        else:
            failures += 1

        if (
            game_number % 100 == 0
            or game_number == len(targets)
        ):
            print(
                f"  {agent_name}: "
                f"{game_number}/{len(targets)} games",
                flush=True,
            )

    total_elapsed = (
        time.perf_counter()
        - total_start
    )

    rss_after = process_rss_mb()

    win_rate = (
        wins / len(targets)
        if targets
        else 0.0
    )

    avg_all = (
        statistics.mean(attempts_all)
        if attempts_all
        else None
    )

    avg_successful = (
        statistics.mean(attempts_successful)
        if attempts_successful
        else None
    )

    median_successful = (
        statistics.median(attempts_successful)
        if attempts_successful
        else None
    )

    max_successful = (
        max(attempts_successful)
        if attempts_successful
        else None
    )

    avg_ms_per_game = (
        statistics.mean(runtimes) * 1000
        if runtimes
        else None
    )

    first_game_ms = (
        runtimes[0] * 1000
        if runtimes
        else None
    )

    subsequent_game_ms = (
        statistics.mean(runtimes[1:]) * 1000
        if len(runtimes) > 1
        else None
    )

    total_evals = sum(eval_counts)

    avg_evals_per_game = (
        total_evals / len(targets)
        if targets
        else None
    )

    distribution: dict[str, int] = {}

    for attempts in attempts_all:
        key = str(attempts)
        distribution[key] = (
            distribution.get(key, 0) + 1
        )

    return {
        "agent": agent_name,
        "games": len(targets),
        "wins": wins,
        "failures": failures,
        "win_rate": win_rate,
        "avg_guesses_all_games": avg_all,
        "avg_guesses_successful_games": avg_successful,
        "median_guesses_successful_games": median_successful,
        "max_guesses_successful_games": max_successful,
        "guess_distribution": distribution,
        "avg_ms_per_game": avg_ms_per_game,
        "first_game_ms": first_game_ms,
        "subsequent_game_ms": subsequent_game_ms,
        "total_runtime_seconds": total_elapsed,
        "total_guess_evaluations": total_evals,
        "avg_guess_evaluations_per_game": avg_evals_per_game,
        "rss_before_mb": rss_before,
        "rss_after_mb": rss_after,
        "errors": errors,
    }


def choose_targets(
    answers: list[str],
    games: int | None,
    seed: int,
) -> list[str]:

    if games is None or games >= len(answers):
        return list(answers)

    rng = np.random.default_rng(seed)

    indices = rng.choice(
        len(answers),
        size=games,
        replace=False,
    )

    return [
        answers[int(index)]
        for index in indices
    ]


def main():

    parser = argparse.ArgumentParser(
        description="Evaluate Wordle agents."
    )

    parser.add_argument(
        "--dataset",
        choices=["A", "B", "C"],
        default="A",
    )

    parser.add_argument(
        "--games",
        type=int,
        default=None,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    parser.add_argument(
        "--agents",
        nargs="+",
        default=DEFAULT_AGENTS,
    )

    parser.add_argument(
        "--max-guesses",
        type=int,
        default=6,
    )

    parser.add_argument(
        "--out",
        type=str,
        default=None,
    )

    args = parser.parse_args()

    print(
        f"Loading Dataset {args.dataset}...",
        flush=True,
    )

    answers, guesses = load_dataset(
        args.dataset
    )

    print(
        f"Loaded {len(answers):,} answers and "
        f"{len(guesses):,} guesses.",
        flush=True,
    )

    print(
        "Building feedback matrix...",
        flush=True,
    )

    matrix_start = time.perf_counter()

    engine = Engine(
        answers=answers,
        guesses=guesses,
    )

    matrix_time = (
        time.perf_counter()
        - matrix_start
    )

    matrix_mb = (
        engine.P.nbytes
        / (1024 * 1024)
    )

    print(
        f"Feedback matrix ready in "
        f"{matrix_time:.2f}s "
        f"({matrix_mb:.2f} MB).",
        flush=True,
    )

    print(
        f"Python: {platform.python_version()}",
        flush=True,
    )

    print(
        f"NumPy: {np.__version__}",
        flush=True,
    )

    print(
        f"Platform: {platform.platform()}",
        flush=True,
    )

    targets = choose_targets(
        answers=answers,
        games=args.games,
        seed=args.seed,
    )

    print(
        f"Evaluating {len(targets):,} games...",
        flush=True,
    )

    results = []

    for agent_name in args.agents:

        print(
            f"\nRunning {agent_name}...",
            flush=True,
        )

        result = evaluate(
            agent_name=agent_name,
            engine=engine,
            targets=targets,
            max_guesses=args.max_guesses,
        )

        results.append(result)

        print(
            f"{agent_name}: "
            f"win={result['win_rate']:.4f} "
            f"avg_all={result['avg_guesses_all_games']:.3f} "
            f"avg_success={result['avg_guesses_successful_games']} "
            f"fail={result['failures']} "
            f"ms/game={result['avg_ms_per_game']:.2f}",
            flush=True,
        )

        if result["errors"]:
            print(
                f"  Errors: {len(result['errors'])}",
                flush=True,
            )

            print(
                f"  First error: {result['errors'][0]}",
                flush=True,
            )

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    if args.out:
        output_file = Path(args.out)
    else:
        output_file = (
            RESULTS_DIR
            / f"results_dataset{args.dataset}.json"
        )

    output = {
        "dataset": args.dataset,
        "n_answers": len(answers),
        "n_guesses": len(guesses),
        "evaluation_games": len(targets),
        "seed": args.seed,
        "max_guesses": args.max_guesses,
        "feedback_table_build_s": matrix_time,
        "feedback_table_MB": matrix_mb,
        "python": platform.python_version(),
        "numpy": np.__version__,
        "platform": platform.platform(),
        "processor": platform.processor(),
        "results": results,
    }

    with output_file.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            output,
            file,
            indent=2,
        )

    print(
        f"\nSaved: {output_file}",
        flush=True,
    )


if __name__ == "__main__":
    main()