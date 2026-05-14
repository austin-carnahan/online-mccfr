"""Experiment: Retrospective Sampling vs OOS — aggregate exploitability.

Matches the paper's experiment design (Lisý et al. 2015, §4.4.2):
fixed 500 matches vs random, sweep simulations per move.

Compares OOS (δ=0.9, paper default) against RetroBot with two decay
functions on Kuhn and Leduc poker.

    LevelExponential(0.5) — smooth multi-level decay (the default)
    LevelStep(0.01)       — aggressive level-0 targeting (≈ OOS-like)

Usage:
    python -m experiments.retro_vs_oos
"""

from eval.config import EvalConfig
from eval.aggregate_exploitability import run
from src.isgt import LevelExponential, LevelStep


config = EvalConfig(
    games=["kuhn_poker", "leduc_poker"],
    algorithm=["oos", "retro"],
    delta=0.9,                  # OOS paper default (ignored by retro)
    epsilon=0.6,
    gamma=0.01,
    decay_fn=[LevelExponential(0.5), LevelStep(0.01)],
    sims_per_move=[50, 100, 200, 500, 1000, 2000],
    num_matches=500,
    output_dir="results/retro_vs_oos",
    workers=4,
    seed=42,
)

if __name__ == "__main__":
    run(config)
