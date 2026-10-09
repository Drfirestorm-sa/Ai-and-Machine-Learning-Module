# CS21002 Wordle AI - Search-Based Solver

## Setup
    pip install numpy matplotlib pytest
    python -m pytest -q tests          # correctness tests
    python demo.py crane               # watch the agent solve a word
    python evaluate.py --dataset A     # full evaluation (2,309 games)
    python evaluate.py --dataset B --games 300 --seed 42
    python evaluate.py --dataset C --games 300 --seed 42
    python evaluate.py --dataset D --games 300 --seed 42
    python evaluate.py --dataset E --games 300 --seed 42
    python make_figures.py             # guess-distribution charts

## Code -> report section map
| File | Report section |
| --- | --- |
| wordle/game.py (GameState, WordleGame) | 2 Agent/Environment, 3 State, actions, transition, goal |
| wordle/feedback.py | 3.3 State transition (feedback function, 243 patterns) |
| wordle/agents.py: Engine, SearchAgent | 5 Search-based AI, 6 Search strategy |
| wordle/agents.py: entropy_score, expected_size_score, minimax_score | 6.4 Heuristics / information gain |
| wordle/agents.py: observe() | 6.3 Candidate reduction |
| wordle/agents.py: RandomCandidateAgent, AlphabeticalAgent | 6.1 uninformed baselines |
| wordle/words.py | 8 Dataset, 8.2 Data cleaning |
| evaluate.py | 10 Experimental setup, 11 Evaluation, 12 Results |

## Datasets
* A (standard): 2,309 possible answers, 2,309 legal guesses
* B (expanded): 12,953 possible answers, 12,953 legal guesses
* C (expanded-guess search): 2,309 possible answers, 12,953 legal guesses; only the legal guess space is expanded
* D (Cambridge guesses): original 2,309 possible answers; guesses include the original answers and cleaned five-letter Cambridge A2/B1 vocabulary
* E (Cambridge + Oxford guesses): original 2,309 possible answers; guesses include the original answers and cleaned five-letter Cambridge/Oxford vocabulary

Word lists: 3b1b/videos (_2022/wordle/data) on GitHub - cite in references.

## Agents
* search-entropy        - greedy one-step search using expected information gain
* search-expected_size  - greedy one-step search minimising expected candidates left
* search-minimax        - greedy one-step search minimising the worst-case partition
* random-candidate      - baseline: constraint filtering only, random consistent guess
* alphabetical-candidate - baseline: first consistent word in dictionary order

The search methods score candidate guesses using the feedback partition structure:
each guess can produce one of 243 possible feedback patterns. After feedback,
the candidate answer set is filtered before the next greedy choice.

## Evaluation results
`evaluate.py` supports Datasets A–E, evaluates all agents under the same
conditions, and records win rate, average guesses, guess distribution, runtime,
and guess-evaluation counts. Datasets D and E keep the original answer list
unchanged and vary the legal guess vocabulary. The default maximum is 6 guesses.

The new D/E results should be generated before drawing performance conclusions;
no D/E performance results are reported yet.

Dataset A has been fully evaluated over all 2,309 possible answers:

| Agent | Win rate | Average guesses |
| --- | ---: | ---: |
| search-entropy | 100% | 3.672 |
| search-expected_size | 100% | 3.705 |
| search-minimax | 100% | 3.803 |
| random-candidate | 98.57% | 4.049 |
| alphabetical-candidate | 97.57% | 4.316 |

Dataset C has also been fully evaluated over all 2,309 possible answers:

| Agent | Win rate | Average guesses |
| --- | ---: | ---: |
| search-entropy | 100% | 3.636 |
| search-expected_size | 100% | 3.653 |
| search-minimax | 100% | 3.770 |
| random-candidate | 98.57% | 4.049 |
| alphabetical-candidate | 97.57% | 4.316 |

Compared with Dataset A, Dataset C produced the following average guesses:

| Agent | Dataset A | Dataset C |
| --- | ---: | ---: |
| entropy | 3.672 | 3.636 |
| expected_size | 3.705 | 3.653 |
| minimax | 3.803 | 3.770 |
| random | 4.049 | 4.049 |
| alphabetical | 4.316 | 4.316 |

All three search heuristics achieved a 100% win rate on both Datasets A and C.
The expanded legal guess space in C slightly reduced the average guesses for
all three search heuristics, while the baseline results were unchanged. Dataset
C has a much larger search/action space, so the search methods require
substantially more computation: entropy took 846.55 ms/game, expected-size
took 314.66 ms/game, and minimax took 270.15 ms/game. This makes Dataset C a
trade-off between solution quality and computational cost rather than
universally better.
