"""OOS vs ISGT depth/decay ablation — Match-count sensitivity.

Expands the depth-0 matchcount experiment with:
  - ISGT depth 0, 1, 2
  - Two decay functions: LevelUniform, LevelExponential(0.7)
  - OOS baseline

Usage:
    python -m experiments.oos_vs_isgt_depth_ablation
    python -m experiments.oos_vs_isgt_depth_ablation --quick
"""

import sys

from eval.config import EvalConfig
from eval.aggregate_exploitability import run
from src.isgt import LevelUniform, LevelExponential


FULL_CONFIG = EvalConfig(
    games=["kuhn_poker", "leduc_poker", "goofspiel", "liars_dice"],
    algorithm=["oos", "isgt"],
    delta=0.9,
    epsilon=0.6,
    gamma=0.01,
    bias_mode="full",
    max_iig_depth=[0, 1, 2],
    decay_fn=[LevelUniform(), LevelExponential(0.7)],
    sims_per_move=[200, 600],
    num_matches=[10, 30, 100],
    workers=4,
    output_dir="results/oos_vs_isgt_depth_ablation",
)

QUICK_CONFIG = EvalConfig(
    games=["kuhn_poker", "leduc_poker", "goofspiel", "liars_dice"],
    algorithm=["oos", "isgt"],
    delta=0.9,
    epsilon=0.6,
    gamma=0.01,
    bias_mode="full",
    max_iig_depth=[0, 1, 2],
    decay_fn=[LevelUniform(), LevelExponential(0.7)],
    sims_per_move=[200],
    num_matches=[10, 30],
    workers=4,
    output_dir="results/oos_vs_isgt_depth_ablation_quick",
)


if __name__ == "__main__":
    config = QUICK_CONFIG if "--quick" in sys.argv else FULL_CONFIG
    run(config)


