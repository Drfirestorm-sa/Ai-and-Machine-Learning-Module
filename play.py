"""Use the search agent to help you play a real Wordle game.

    python play.py            # standard 2,309-word list (fast)
    python play.py --dataset B   # expanded 12,953-word list (slower, covers more words)

Each turn the program suggests a guess. Type that word into Wordle, then enter
the colours you got back as 5 letters:
    g = green   y = yellow   x (or - or . or b) = grey/black
e.g.  gxyxx
If you played a different word from the suggestion, type it first.
"""
import argparse
import numpy as np
from wordle.words import load_dataset
from wordle.agents import Engine, SearchAgent
from wordle.feedback import ALL_GREEN


def parse_feedback(text):
    """'gxyxx' -> base-3 pattern int, or None if invalid."""
    text = text.strip().lower()
    if len(text) != 5:
        return None
    digit = {"g": 2, "y": 1, "x": 0, "-": 0, ".": 0, "b": 0}
    if any(ch not in digit for ch in text):
        return None
    return sum(digit[ch] * 3 ** i for i, ch in enumerate(text))


def run(eng, input_fn=input, print_fn=print, max_turns=6):
    agent = SearchAgent(eng)
    for turn in range(1, max_turns + 1):
        suggestion = agent.choose()
        n = len(agent.cand)
        print_fn(f"\nGuess {turn}: try  {suggestion.upper()}   ({n} possible words left)")
        # which word did the player actually enter?
        while True:
            word = input_fn(f"Word you played [{suggestion}]: ").strip().lower() or suggestion
            if word in eng.guess_index:
                break
            print_fn(f"'{word}' isn't in my word list - play another word or press Enter for {suggestion}.")
        # what feedback did Wordle give?
        while True:
            pattern = parse_feedback(input_fn("Feedback (g/y/x, e.g. gxyxx): "))
            if pattern is not None:
                break
            print_fn("Enter exactly 5 letters using g, y and x only.")
        if pattern == ALL_GREEN:
            print_fn(f"\nSolved in {turn} guesses!")
            return
        agent.observe(word, pattern)
        if len(agent.cand) == 0:
            print_fn("\nNo words match that feedback. Either a colour was entered wrongly, "
                     "or the answer isn't in my list (try --dataset B).")
            return
        if len(agent.cand) <= 10:
            print_fn("Remaining possibilities: " + ", ".join(agent.candidate_words()))
    print_fn("\nOut of guesses.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="A", choices=["A", "B"])
    a = ap.parse_args()
    print("Loading...")
    answers, guesses = load_dataset(a.dataset)
    run(Engine(answers, guesses))