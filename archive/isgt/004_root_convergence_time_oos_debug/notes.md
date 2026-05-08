# Root Convergence — OOS vs ISMCTS Debug (Liar's Dice)

**Date**: 2026-04  
**Goal**: Isolate whether ISGT's poor liar's dice performance (from experiment 003) is caused by IIG-specific overhead or is inherent to MCCFR-style algorithms. Test plain OOS (MCCFR without IIG machinery) against ISMCTS on liar's dice only.

## Experiment Setup

- **Algorithms**:
  - **OOS** (δ=0.0 untargeted, ε=0.6, γ=0.01) — vanilla MCCFR, no IIG overhead
  - **ISMCTS** — OpenSpiel IS-MCTS (uct_c=2.0, max_simulations=1 per loop iteration)
- **Game**: Liar's Dice (6,1,1) only
- **Time budget**: 600 seconds per algorithm
- **Time checkpoints**: 1, 2, 5, 10, 20, 30, 60, 90, 120, 180, 240, 300, 420, 600s
- **Metric**: Full-game exploitability (NashConv)
- **Seed**: 42

## Key Findings

### OOS beats ISMCTS on liar's dice ✅
- At 60s: OOS 0.288 vs ISMCTS 0.275 (ISMCTS slightly ahead early)
- At 120s: OOS 0.206 vs ISMCTS 0.227 (OOS overtakes)
- At 600s: OOS **0.088** vs ISMCTS **0.177** (OOS is 2x better)
- OOS throughput: ~12.8K iter/s. ISMCTS: ~11K iter/s. Comparable.

### Conclusion 1: IIG overhead is the ISGT bottleneck
- ISGT on liar's dice (experiment 003): ~50 iter/s, exploitability 0.601 at 600s
- OOS on liar's dice (this experiment): ~12.8K iter/s, exploitability 0.088 at 600s
- Same MCCFR core algorithm, ~250x throughput difference → IIG precomputation and weight lookups are the bottleneck
- This explains why ISGT still wins on Leduc and Goofspiel: their IIG graphs are much smaller, so the overhead is negligible

### Conclusion 2: Separate per-iteration issue on liar's dice
- Reviewing iteration-based results from experiment 002, ISGT slightly underperforms ISMCTS on liar's dice even per-iteration (0.409 vs 0.322 at 200K iterations), while winning on all other games
- This is a distinct issue from the IIG overhead — ε-greedy MCCFR is less sample-efficient than UCT on liar's dice's game structure
- However, this per-iteration gap is mild: OOS overcomes it with sustained convergence over time, reaching 0.088 vs ISMCTS's 0.177 at 600s
- ISMCTS converges faster early but plateaus; MCCFR converges slower per-iteration but doesn't plateau

### Note on exploitability computation cost
- Both algorithms show near-zero iteration progress between the 1s and ~15s checkpoints (~4s per NashConv call on liar's dice)
- This is a measurement artifact, not an algorithm issue — affects both equally

## Files

- `all_results.json` — OOS and ISMCTS exploitability at each time checkpoint for liar's dice
