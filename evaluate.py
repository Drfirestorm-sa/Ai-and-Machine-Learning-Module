"""Sections 10-12: controlled evaluation of the agents.

Usage:
    python evaluate.py --dataset A                       # all 2,309 answers
    python evaluate.py --dataset B --games 500 --seed 42 # sampled games
    python evaluate.py --dataset A --agents search-entropy random-candidate
"""
from __future__ import annotations
import argparse, json, platform, resource, statistics, sys, time
from collections import Counter
import numpy as np
from wordle.words import load_dataset
from wordle.game import WordleGame
from wordle.agents import Engine, SearchAgent, RandomCandidateAgent, AlphabeticalAgent


def make_agent(name, eng, seed):
    if name.startswith("search-"):
        return SearchAgent(eng, heuristic=name.split("-", 1)[1])
    if name == "random-candidate":
        return RandomCandidateAgent(eng, seed=seed)
    if name == "dfs-alphabetical":
        return AlphabeticalAgent(eng)
    raise ValueError(name)


def play(agent, target, max_guesses):
    agent.reset()
    game = WordleGame(target, max_guesses)
    evals0 = agent.evals
    while not game.over:
        g = agent.choose()
        agent.observe(g, game.step(g))
    return game.solved, len(game.state.previous_guesses), agent.evals - evals0


def evaluate(agent_name, eng, targets, max_guesses, seed):
    agent = make_agent(agent_name, eng, seed)
    # warm-up: computes (and caches) the root decision once; reported separately
    t0 = time.perf_counter()
    agent.reset(); agent.choose()
    precompute = time.perf_counter() - t0
    solved, guesses, evals, times = [], [], [], []
    for t in targets:
        t0 = time.perf_counter()
        ok, n, ev = play(agent, t, max_guesses)
        times.append(time.perf_counter() - t0)
        solved.append(ok); guesses.append(n); evals.append(ev)
    won = [n for ok, n in zip(solved, guesses) if ok]
    return {
        "agent": agent_name,
        "games": len(targets),
        "win_rate": sum(solved) / len(solved),
        "avg_guesses_solved": statistics.mean(won) if won else None,
        "median_guesses_solved": statistics.median(won) if won else None,
        "max_guesses_solved": max(won) if won else None,
        "failed_games": len(solved) - sum(solved),
        "distribution": dict(sorted(Counter(n if ok else "fail"
                                            for ok, n in zip(solved, guesses)).items(), key=str)),
        "total_runtime_s": sum(times),
        "mean_time_per_game_ms": 1000 * statistics.mean(times),
        "max_time_per_game_ms": 1000 * max(times),
        "mean_guess_evaluations_per_game": statistics.mean(evals),
        "first_guess": getattr(agent, "_first_guess", None),
        "root_precompute_s": precompute,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="A", choices=["A", "B"])
    ap.add_argument("--games", type=int, default=0, help="0 = every answer")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-guesses", type=int, default=6)
    ap.add_argument("--agents", nargs="+",
                    default=["search-entropy", "search-expected_size", "search-minimax",
                             "random-candidate", "dfs-alphabetical"])
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    answers, guesses = load_dataset(a.dataset)
    t0 = time.perf_counter()
    eng = Engine(answers, guesses)
    table_s = time.perf_counter() - t0
    rng = np.random.default_rng(a.seed)
    targets = answers if a.games in (0, len(answers)) else \
        [answers[i] for i in rng.choice(len(answers), a.games, replace=False)]

    env = {"python": platform.python_version(), "numpy": np.__version__,
           "platform": platform.platform(), "processor": platform.processor() or platform.machine()}
    meta = {"dataset": a.dataset, "n_answers": len(answers), "n_guesses": len(guesses),
            "evaluation_games": len(targets), "seed": a.seed, "max_guesses": a.max_guesses,
            "feedback_table_build_s": table_s,
            "feedback_table_MB": eng.P.nbytes / 1e6, "environment": env}
    print(json.dumps(meta, indent=2))

    results = []
    for name in a.agents:
        r = evaluate(name, eng, targets, a.max_guesses, a.seed)
        r["peak_rss_MB_after"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
        results.append(r)
        print(f"{name:22s} win={r['win_rate']:.4f} avg={r['avg_guesses_solved']:.3f} "
              f"med={r['median_guesses_solved']} max={r['max_guesses_solved']} "
              f"fail={r['failed_games']} ms/game={r['mean_time_per_game_ms']:.2f} "
              f"first={r['first_guess']}")
    out = a.out or f"results/results_dataset{a.dataset}.json"
    with open(out, "w") as f:
        json.dump({"meta": meta, "results": results}, f, indent=2)
    print("saved", out)


if __name__ == "__main__":
    main()
