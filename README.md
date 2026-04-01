# Online MCCFR Research

Implementing and benchmarking online Monte Carlo Counterfactual Regret Minimization
algorithms using [OpenSpiel](https://github.com/google-deepmind/open_spiel).

**Algorithms:**
- **OS-MCCFR** — OpenSpiel's built-in offline Outcome Sampling MCCFR (baseline)
- **OOS** — Online Outcome Sampling (Lisý, Lanctot, Bowling 2015)
- **ISMCTS** — Information Set Monte Carlo Tree Search (UCT variant)
- **ISGT** — Information Set Graph Targeting (novel) — proximity-weighted targeting
  over the information set graph

**Benchmark games:** Leduc Poker, Liar's Dice, Goofspiel (imperfect-info variant)

## Directory Layout

```
├── run.py                  Unified CLI entry point
├── play_game.py            Random playout runner
├── requirements.txt        Python dependencies
│
├── src/                    Algorithm and utility code
│   ├── games.py            Game loading helpers and config (wraps pyspiel)
│   ├── metrics.py          Exploitability and convergence measurement
│   ├── display.py          Output formatting and display-flag parsing (-s, -r, -w)
│   ├── outcome_sampling.py Offline OS-MCCFR baseline wrapper
│   ├── oos.py              Online Outcome Sampling (OOSBot, Algorithm 1)
│   ├── ismcts.py           ISMCTS wrapper with custom resamplers
│   ├── isgt.py             Information Set Graph Targeting (novel)
│   └── online.py           Head-to-head match runner for Bot agents
│
├── experiments/            Benchmark experiment harnesses
│   ├── configs.py          Algorithm registry, default params, budget constants
│   ├── root_convergence.py Exploitability vs iterations from the game root
│   └── aggregate_exploit.py Aggregate exploitability via multi-match method
│
├── eval/                   Evaluation and visualization
│   ├── aggregate.py        Multi-match aggregate exploitability computation
│   ├── plots.py            Matplotlib plotting (convergence curves, budget sweeps)
│   └── compare.py          CLI for generating comparison plots from saved results
│
├── tests/                  Pytest tests
│   ├── test_smoke.py       OpenSpiel loads, random playouts, exploitability works
│   ├── test_outcome_sampling.py  Offline OS-MCCFR convergence
│   ├── test_oos.py         OOS convergence
│   ├── test_online.py      Head-to-head match tests
│   └── test_isgt.py        ISGT convergence (placeholder)
│
├── knowledge/              Papers and reference material
└── results/                Experiment outputs (JSON)
    ├── root_convergence/   Per-game and combined root convergence results
    ├── aggregate_exploit/  Per-game and combined aggregate exploitability results
    └── plots/              Generated comparison plots (PNG)
```

## Setup

```bash
python -m venv env
source env/bin/activate
pip install -r requirements.txt
```

## Running Tests

```bash
pytest tests/
```

## CLI Reference

All commands go through `run.py`:

```
python run.py <command> [args...]
```

Available commands: `play`, `outcome_sampling`, `online`, `root_convergence`, `aggregate_exploit`, `plot`

### Play — Random Playouts

Run a random playout of a game to see its structure:

```bash
python run.py play                    # defaults to leduc_poker
python run.py play liars_dice
python run.py play goofspiel
python run.py play all                # play all three games
```

### Outcome Sampling — Offline OS-MCCFR Baseline

Run offline Outcome Sampling MCCFR and measure convergence:

```bash
python run.py outcome_sampling                             # leduc_poker, 10K iterations
python run.py outcome_sampling leduc_poker 50000
python run.py outcome_sampling leduc_poker 10000 -s        # + strategy table
python run.py outcome_sampling leduc_poker 10000 -s -r     # + strategy + regret tables
python run.py outcome_sampling leduc_poker 10000 -w        # + visit weights
```

Display flags:
- `-s` — Print average strategy at every information set
- `-r` — Print cumulative regrets at every information set
- `-w` — Print visit weight accumulators per information set

### Online — Head-to-Head Matches

Run bot-vs-bot matches and print win rates:

```bash
python run.py online leduc_poker oos ismcts 100             # 100 games, OOS vs ISMCTS
python run.py online leduc_poker oos oos 500 --sims 1000    # OOS self-play, 1000 sims/move
python run.py online liars_dice ismcts random 200           # ISMCTS vs random
```

Arguments:
- Positional: `<game> <bot0> <bot1> [num_games]` (default: `leduc_poker oos ismcts 100`)
- `--sims N` — Simulations per move (default: 1000)
- `--delta D` — OOS targeting parameter (default: 0.9)

Bot names: `oos`, `ismcts`, `random`

### Root Convergence Experiment

Measure exploitability vs iterations from the game root (reproduces paper Figures 3a,d,h):

```bash
python run.py root_convergence                                          # all games, all algos
python run.py root_convergence leduc_poker                              # single game
python run.py root_convergence leduc_poker --algos oos                  # OOS only
python run.py root_convergence --checkpoints 100,1000,10000,50000       # custom checkpoints
python run.py root_convergence --seed 123                               # custom seed
```

Arguments:
- Positional: `[game ...]` — games to run (default: all three)
- `--algos oos,ismcts` — comma-separated algorithm list
- `--checkpoints 100,500,...` — iteration thresholds (default: `100, 500, 1000, 2000, 5000, 10000, 20000, 50000`)
- `--seed N` — random seed (default: 42)

Results saved to `results/root_convergence/`.

### Aggregate Exploitability Experiment

Sweep sims-per-move budgets and compute aggregate exploitability via the multi-match method (reproduces paper Figures 3b,e,i):

```bash
python run.py aggregate_exploit                                         # all games, all algos
python run.py aggregate_exploit leduc_poker                             # single game
python run.py aggregate_exploit --algos oos --budgets 100,500,1000      # OOS only, custom budgets
python run.py aggregate_exploit --matches 200                           # fewer matches (faster)
python run.py aggregate_exploit --seed 123
```

Arguments:
- Positional: `[game ...]` — games to run (default: all three)
- `--algos oos,ismcts` — comma-separated algorithm list
- `--budgets 100,500,...` — sims-per-move budgets (default: `100, 250, 500`)
- `--matches N` — matches per evaluation (default: 500)
- `--seed N` — random seed (default: 42)

Results saved to `results/aggregate_exploit/`.

### Plot — Generate Comparison Plots

Generate plots from saved experiment results:

```bash
python run.py plot                                          # all experiment types
python run.py plot root_convergence                         # root convergence only
python run.py plot aggregate_exploit                        # aggregate exploitability only
python run.py plot --results-dir results                    # custom results directory
python run.py plot --output-dir results/plots               # custom output directory
```

Plots are saved as PNG files to `results/plots/` (by default). Generates both multi-game comparison grids and individual per-game plots.

## Algorithm Details

### OOS (Online Outcome Sampling)

Implementation of Algorithm 1 from Lisý, Lanctot, Bowling (2015). Builds game trees incrementally during online play, using IST (Information Set Targeting) to focus exploration on the current game situation.

Default parameters: `δ=0.9` (targeting), `ε=0.6` (exploration), `γ=0.01` (regret floor).

### ISMCTS (Information Set MCTS)

Wraps OpenSpiel's ISMCTS with UCT selection and custom resamplers for games that require determinization (Liar's Dice, Goofspiel). Uses `RandomRolloutEvaluator` for leaf node evaluation.

## Experiment Design

Both experiment types follow Section 4.2 of the OOS paper:

- **Root convergence**: Run each algorithm from an empty game state, accumulating iterations. Measure exploitability of the learned strategy at log-spaced checkpoints. Tests how well each algorithm converges when given unbounded computation.

- **Aggregate exploitability**: For each sims-per-move budget, play many matches against a random opponent. Accumulate per-information-set strategy frequencies across matches, then compute exploitability of the combined strategy. Tests real-time decision quality under fixed computation budgets.
