"""Experiment: Retro (uniform decay) vs OOS — aggregate exploitability.

Main comparison table for the paper.  Follows Lisý et al. 2015, §4.4.2:
fixed 500 matches vs uniform random opponent, sweep simulations per move.

Uses LevelUniform for Retro — the parameter-free default that won 3/4
games in the decay ablation (001) and tied in the 4th.

Both algorithms use ε=0.6 (paper default for aggregate exploitability).

Usage:
    python -m experiments.retro_uniform_vs_oos
"""

from eval.config import EvalConfig
from eval.aggregate_exploitability import run
from src.isgt import LevelUniform


config = EvalConfig(
    games=["kuhn_poker", "leduc_poker", "goofspiel", "liars_dice"],
    algorithm=["oos", "retro"],
    delta=0.9,
    epsilon=0.6,
    gamma=0.01,
    decay_fn=LevelUniform(),
    sims_per_move=[50, 100, 200, 500, 1000, 2000],
    num_matches=500,
    output_dir="results/retro_uniform_vs_oos",
    workers=4,
    seed=42,
)

if __name__ == "__main__":
    run(config)
