"""Experiment: High-budget convergence test — Retro vs OOS vs decay variants.

Phase 1A of the experiment roadmap: extend the simulation budget to
paper-equivalent ranges (~0.5s–5s wall time) to verify:

1. Retro (uniform) continues to improve and doesn't plateau
2. OOS continues to improve (slower than Retro)
3. Step decay plateaus or diverges at high budgets (the ISMCTS-UCT analog)
4. Exp(0.5) — moderate concentration — tracks uniform or bends toward step?

Paper-matched parameters (Lisý et al. §4.4.3):
    OOS: δ=0.5, ε=0.4 (paper's γ), γ=0.01
    Retro: ε=0.4, γ=0.01 (δ ignored — retro uses level weights)

The concentration spectrum: uniform (flat) → exp(0.5) (moderate) → step
(extreme). If uniform and exp(0.5) both converge while step plateaus,
that maps the safe boundary for dynamic decay.

Budget calibration (from benchmark):
    kuhn:  18,061 sims/s → s=50000 ≈ 2.8s, s=100000 ≈ 5.5s
    leduc: 10,547 sims/s → s=50000 ≈ 4.7s, s=100000 ≈ 9.5s
    goof:   9,846 sims/s → s=50000 ≈ 5.1s, s=100000 ≈ 10.2s
    liars: 14,475 sims/s → s=50000 ≈ 3.5s, s=100000 ≈ 6.9s

100 matches — sufficient for trend detection at high sims.
Includes s=500,1000 for continuity with prior experiments (s≤2000).
Total jobs: 4 games × 4 configs × 8 sims = 128 jobs.
Estimated runtime: ~2-4 hours with 4 workers.

Usage:
    python -m experiments.retro_highbudget_convergence
"""

from eval.config import EvalConfig
from eval.aggregate_exploitability import run
from src.isgt import LevelUniform, LevelStep, LevelExponential


config = EvalConfig(
    games=["kuhn_poker", "leduc_poker", "goofspiel", "liars_dice"],
    algorithm=["oos", "retro"],
    delta=0.5,
    epsilon=0.4,
    gamma=0.01,
    decay_fn=[LevelUniform(), LevelExponential(0.5), LevelStep(0.01)],
    sims_per_move=[500, 1000, 2000, 5000, 10000, 20000, 50000, 100000],
    num_matches=100,
    output_dir="results/retro_highbudget_convergence",
    workers=4,
    seed=42,
)

if __name__ == "__main__":
    run(config)
