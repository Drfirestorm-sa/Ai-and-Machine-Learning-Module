# CS21002 Wordle AI - Search-Based Solver

## Setup
    pip install numpy matplotlib pytest
    python -m pytest -q tests          # correctness tests
    python demo.py crane               # watch the agent solve a word
    python evaluate.py --dataset A     # full evaluation (2,309 games)
    python evaluate.py --dataset B --games 300 --seed 42
    python make_figures.py             # guess-distribution charts

## Code -> report section map
| File | Report section |
| --- | --- |
| wordle/game.py (GameState, WordleGame) | 2 Agent/Environment, 3 State, actions, transition, goal |
| wordle/feedback.py | 3.3 State transition (feedback function, 243 patterns) |
| wordle/agents.py: Engine, SearchAgent | 5 Search-based AI, 6 Search strategy |
| wordle/agents.py: entropy_score | 6.4 Heuristic / information gain |
| wordle/agents.py: observe() | 6.3 Candidate reduction |
| wordle/agents.py: Random/Alphabetical agents | 6.1 uninformed baselines |
| wordle/words.py | 8 Dataset, 8.2 Data cleaning |
| evaluate.py | 10 Experimental setup, 11 Evaluation, 12 Results |

## Datasets
* A (standard): 2,309 answers, guesses = answers
* B (expanded): 12,953-word union list, guesses = answers

Word lists: 3b1b/videos (_2022/wordle/data) on GitHub - cite in references.

## Agents
* search-entropy        - informed best-first search, h = expected information gain (main method)
* search-expected_size  - alternative heuristic: minimise expected candidates left
* search-minimax        - alternative heuristic: minimise worst-case partition
* random-candidate      - baseline: constraint filtering only, random consistent guess
* dfs-alphabetical      - baseline: first consistent word in dictionary order

The ML agent should plug in by implementing `choose()` / `observe()` like `BaseAgent`
and reusing `evaluate.py` so both methods are tested under identical conditions.
