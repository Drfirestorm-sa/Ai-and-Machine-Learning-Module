"""Watch the search agent solve a word and print its reasoning at each step.

    python demo.py crane
    python demo.py            # random target
"""
import sys, random
from wordle.words import load_dataset
from wordle.game import WordleGame
from wordle.agents import Engine, SearchAgent, entropy_score
from wordle.feedback import pattern_to_str
import numpy as np

answers, guesses = load_dataset("A")
target = sys.argv[1] if len(sys.argv) > 1 else random.choice(answers)
assert target in answers, "target must be in the answer list"
eng = Engine(answers, guesses)
agent, game = SearchAgent(eng), WordleGame(target)
print(f"Target: {target}\n")
while not game.over:
    n_before = len(agent.cand)
    g = agent.choose()
    h = entropy_score(eng.partition_counts(agent.cand, np.array([eng.guess_index[g]])), n_before)[0]
    p = game.step(g)
    agent.observe(g, p)
    print(f"guess {len(game.state.previous_guesses)}: {g}  {pattern_to_str(p)}   "
          f"H={h:.2f} bits   candidates {n_before} -> {len(agent.cand)}")
print("\nSolved!" if game.solved else "\nFailed.")
print("known letters:", sorted(game.state.known_letters),
      "| excluded:", sorted(game.state.excluded_letters))
