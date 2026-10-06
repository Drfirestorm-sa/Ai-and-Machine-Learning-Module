import random, sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
import numpy as np
from wordle.feedback import feedback, pattern_matrix, pattern_to_str, ALL_GREEN
from wordle.words import clean_words, load_dataset
from wordle.game import WordleGame
from wordle.agents import Engine, SearchAgent


def test_known_feedback():
    assert pattern_to_str(feedback("crane", "crane")) == "GGGGG"
    assert feedback("crane", "crane") == ALL_GREEN
    # duplicate letters: only one 'e' in answer 'abide' -> second guess 'e' is grey
    assert pattern_to_str(feedback("geese", "abide")) == "----G"
    assert pattern_to_str(feedback("speed", "abide")) == "--Y-Y"
    assert pattern_to_str(feedback("llama", "hello")) == "YY---"
    assert pattern_to_str(feedback("allee", "eagle")) == "YY-YG"


def test_vectorised_matches_scalar():
    answers, _ = load_dataset("A")
    rnd = random.Random(1)
    gs = [rnd.choice(answers) for _ in range(150)] + ["geese", "llama", "eerie", "mamma", "sassy"]
    ans = [rnd.choice(answers) for _ in range(300)] + ["abide", "hello", "eagle", "alley"]
    P = pattern_matrix(gs, ans)
    for i, g in enumerate(gs):
        for j, a in enumerate(ans):
            assert P[i, j] == feedback(g, a), (g, a)


def test_cleaning():
    out = clean_words([" CRANE ", "crane", "", "ab", "abcdef", "cr4ne", "geese", "Geese"])
    assert out == ["crane", "geese"]


def test_agent_solves_and_is_consistent():
    answers, guesses = load_dataset("A")
    eng = Engine(answers, guesses)
    for target in ["crane", "geese", "mamma", "fuzzy", "aback"]:
        game, agent = WordleGame(target), SearchAgent(eng)
        while not game.over:
            g = agent.choose()
            agent.observe(g, game.step(g))
            assert target in agent.candidate_words()      # never prunes the truth
        assert game.solved
