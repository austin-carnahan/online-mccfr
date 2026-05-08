# TODOs

## Algorithm Design & Implementation

- [ ] Draft ISGT algorithm pseudocode
  - Define the sampling weight function `w(d)` where `d` = IIG distance
  - Specify how IIG levels are computed incrementally during tree building
  - Decide on chance-node forcing strategy (binary like OOS, or graduated)
  - Write up as Algorithm 2 (parallel to OOS Algorithm 1)
- [ ] Implement ISGT in `src/isgt.py`
  - IIG construction from OpenSpiel game tree
  - BFS level computation from active infoset
  - Proximity-weighted targeting in the sampling loop
  - ISGTBot class compatible with `src/online.py` match runner
- [ ] Add ISGT to experiment configs (`experiments/configs.py`)
- [ ] Write `tests/test_isgt.py` — convergence + correctness

## Ablations: Sampling Decay Functions

- [ ] Implement and compare decay functions for terminal history weights at IIG distances:
  - Exponential: `w(d) = α^d`
  - Polynomial: `w(d) = 1 / (1 + d)^p`
  - Linear cutoff: `w(d) = max(0, 1 - d/D)`
  - Step function (OOS-like): `w(d) = 1 if d == 0 else ε`
- [ ] Sweep decay parameters on Leduc Poker (fast iteration)
- [ ] Measure exploitability convergence curves per decay variant
- [ ] Select best-performing decay for full benchmarks

## Benchmarking

- [ ] Run root convergence experiments: ISGT vs OOS vs ISMCTS
  - Leduc Poker
  - Liar's Dice
  - Goofspiel
- [ ] Run aggregate exploitability experiments: ISGT vs OOS vs ISMCTS
- [ ] Generate comparison plots

## Variance Analysis

- [ ] Estimate or compute per-terminal-history variance contribution at each IIG level
  - Measure regret update magnitude vs IIG distance
  - Compare variance of updates at Level 0–1 vs Level 2+ vs Level 3+
- [ ] Validate that proximity weighting reduces variance empirically
  - Compare update variance per iteration: ISGT vs OOS
- [ ] Produce figures showing variance reduction vs IIG distance

## Visualizations (done)

- [x] Kuhn Poker game tree with infoset overlay
- [x] ITG visualization (P1=Q slice)
- [x] IIG visualization with level coloring (P1=Q slice)
- [x] IIG detailed with terminal history paths

## Infrastructure

- [ ] Wire up `eval/compare.py` CLI for generating comparison plots
- [ ] Verify plot CLI command works end-to-end
