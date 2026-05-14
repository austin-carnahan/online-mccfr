"""Experiment: Epsilon ablation for Retro (uniform decay).

Tests whether Retro's D+1 divergence levels change the optimal
exploration parameter ε compared to OOS's default of 0.6.

Hypothesis: Retro's intermediate levels provide structured exploration,
potentially preferring lower ε than OOS (less random noise needed when
you have better-targeted trajectories).

Follows Lisý et al. 2015, §4.4.2: fixed 500 matches vs uniform random
opponent, sweep simulations per move.

Usage:
    python -m experiments.retro_epsilon_ablation
"""

from eval.config import EvalConfig
from eval.aggregate_exploitability import run
from src.isgt import LevelUniform


config = EvalConfig(
    games=["kuhn_poker", "leduc_poker", "goofspiel", "liars_dice"],
    algorithm="retro",
    epsilon=[0.1, 0.2, 0.4, 0.6, 0.8],
    gamma=0.01,
    decay_fn=LevelUniform(),
    sims_per_move=[50, 100, 200, 500, 1000, 2000],
    num_matches=500,
    output_dir="results/retro_epsilon_ablation",
    workers=4,
    seed=42,
)

if __name__ == "__main__":
    run(config)
