# Root Convergence — Time-Based (ISGT vs ISMCTS)

**Date**: 2026-04  
**Goal**: Replicate paper Section 4.4.1 root convergence experiment with wall-clock time on the X axis (matching the paper's presentation). Verify that MCCFR converges while ISMCTS diverges.

## Experiment Setup

- **Algorithms**:
  - **ISGT** (untargeted, uniform weights, ε=0.6, γ=0.01) — equivalent to vanilla MCCFR
  - **ISMCTS** — OpenSpiel IS-MCTS (uct_c=2.0, max_simulations=1 per loop iteration)
- **Games**: Leduc Poker, Liar's Dice (6,1,1), Goofspiel
- **Time budget**: 600 seconds per algorithm per game
- **Time checkpoints**: 1, 2, 5, 10, 20, 30, 60, 90, 120, 180, 240, 300, 420, 600s
- **Metric**: Full-game exploitability (NashConv) at each checkpoint
- **Seed**: 42

## Key Findings

### Leduc Poker ✅ (matches paper)
- **ISMCTS**: Starts high (1.56 at 1s), converges to ~0.29 by 240s, then diverges slightly to 0.305 at 600s. 9.6M iterations.
- **ISGT**: Starts higher (2.51 at 1s) but converges steadily to 0.148 at 600s. 1.7M iterations.
- ISMCTS stalls/diverges while ISGT continues to converge — consistent with the paper's result.

### Goofspiel ✅ (matches paper)
- **ISMCTS**: Quickly reaches 0.068 at ~90s, then clearly diverges to 0.170 at 600s. 8.5M iterations.
- **ISGT**: Converges steadily to 0.012 at 600s. 7.0M iterations.
- Clear ISMCTS divergence — consistent with the paper.

### Liar's Dice ❌ (does NOT match paper)
- **ISMCTS**: Converges steadily from 0.666 to 0.178 at 600s. 6.8M iterations (~11.4K iter/s).
- **ISGT**: Extremely slow — only 30K iterations in 600s (~50 iter/s). Exploitability stuck at 0.601.
- **Paper expects**: OOS/MCCFR should converge while ISMCTS diverges on liar's dice.
- **What we see**: ISMCTS converges steadily, ISGT barely runs.

## Liar's Dice Anomaly — Hypotheses

1. **IIG precomputation overhead**: ISGT precomputes the Information Set Game Tree (IIG) for the full game. Liar's dice may have a much larger IIG than Goofspiel or Leduc, causing massive overhead in the `ISGTBot.__init__()` phase and per-iteration weight lookups.

2. **Exploitability computation cost**: NashConv for liar's dice may be expensive (~4s per call). Since we compute it at 14 checkpoints, this could consume significant wall-clock time. But this affects both algorithms equally, so it doesn't explain why ISGT gets only 50 iter/s while ISMCTS gets 11.4K iter/s.

3. **Per-iteration cost of MCCFR vs ISMCTS**: Each MCCFR iteration (ISGT's `_walk()`) traverses root-to-terminal with full policy lookups. Each ISMCTS simulation is lightweight by comparison. The game tree depth/branching of liar's dice may make MCCFR iterations much heavier than ISMCTS simulations.

4. **IIG-specific vs shared MCCFR issue**: The anomaly could be caused by ISGT's IIG machinery specifically, or it could be inherent to MCCFR on liar's dice. Testing plain OOS (which is MCCFR without IIG) would isolate this.

## Next Steps

- **Diagnostic experiment**: Run OOS vs ISMCTS on liar's dice only (600s) to isolate whether the slowness is ISGT-specific (IIG overhead) or inherent to MCCFR-style algorithms on this game.
- If OOS is also slow → MCCFR per-iteration cost on liar's dice is the issue, not IIG.
- If OOS is fast → IIG precomputation/weight lookups are the bottleneck.

## Files

- `all_results.json` — per-game, per-algorithm exploitability at each time checkpoint
- `root_convergence.png` — convergence plots (one subplot per game)
