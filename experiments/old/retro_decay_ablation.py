"""Experiment: Decay function ablation for Retrospective Sampling.

Matches the paper's experiment design (Lisý et al. 2015, §4.4.2):
fixed 500 matches vs random, sweep simulations per move.

Sweeps 6 decay functions from aggressive targeting to uniform,
measuring aggregate exploitability on 4 games.

    LevelStep(0.01)       — ~100% at k=0 (OOS-like)
    LevelExponential(0.3) — aggressive decay
    LevelExponential(0.5) — moderate (current default)
    LevelExponential(0.7) — mild decay
    LevelExponential(0.9) — near-uniform
    LevelUniform()        — flat (no targeting preference)

Usage:
    python -m experiments.retro_decay_ablation
"""

from eval.config import EvalConfig
from eval.aggregate_exploitability import run
from src.isgt import LevelExponential, LevelStep, LevelUniform


config = EvalConfig(
    games=["kuhn_poker", "leduc_poker", "goofspiel", "liars_dice"],
    algorithm="retro",
    epsilon=0.6,
    gamma=0.01,
    decay_fn=[
        LevelStep(0.01),
        LevelExponential(0.3),
        LevelExponential(0.5),
        LevelExponential(0.7),
        LevelExponential(0.9),
        LevelUniform(),
    ],
    sims_per_move=[50, 100, 200, 500, 1000],
    num_matches=500,
    output_dir="results/retro_decay_ablation",
    workers=4,
    seed=42,
)

if __name__ == "__main__":
    run(config)
