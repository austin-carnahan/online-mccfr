# Root Convergence — Iteration-Based

**Date**: 2025-01  
**Goal**: Replicate paper Section 4.4.1 root convergence experiment — verify that MCCFR (ISGT untargeted) converges while ISMCTS diverges, using iteration count on the X axis.

## Experiment Setup

- **Algorithms**:
  - **ISGT** (untargeted, uniform weights) — equivalent to vanilla MCCFR
  - **ISMCTS** — OpenSpiel's IS-MCTS wrapper (uct_c=2.0, max_simulations=100)
  - OOS was initially included but removed mid-experiment (produces identical results to untargeted ISGT)
- **Games**: Kuhn Poker, Leduc Poker, Liar's Dice, Goofspiel
- **Iterations**: 200,000 max
- **Checkpoints**: 100, 500, 1K, 2K, 5K, 10K, 20K, 50K, 100K, 200K
- **Metric**: Full-game exploitability (NashConv) at each checkpoint
- **Seed**: 42

## Key Findings

- **ISGT converges** on all four games as expected (it is vanilla MCCFR).
- **ISMCTS diverges** on Leduc Poker and Goofspiel — exploitability does not decrease with more iterations, matching the paper's prediction.
- On Liar's Dice, ISMCTS maintains a slight edge at higher iteration counts, likely due to ISMCTS running many more game simulations per "iteration" (100 rollouts per move decision).
- Kuhn Poker is too small to meaningfully distinguish the algorithms.

## Outcome

Experiment confirmed the qualitative result from the paper but revealed that **iteration count is not a fair comparison metric** — ISMCTS does far less work per iteration than a full MCCFR tree traversal. This led to refactoring the experiment to use **wall-clock time** on the X axis (matching the paper's actual presentation in Section 4.4.1).

## Files

- `all_results.json` — per-game, per-algorithm exploitability at each checkpoint
- `root_convergence.png` — convergence plots (one subplot per game)
