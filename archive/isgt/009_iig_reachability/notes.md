# IIG Reachability Test — Upstream BFS Terminal Coverage

**Date**: 2026-04-06  
**Archive**: 009_iig_reachability  
**Runtime**: <1 second (no matches, pure graph analysis)  
**Predecessor**: 008_isgt_delta_ablation_full (Q1)

## Motivation

Archive 008 Finding 6 hypothesized that shaped decay functions underperform ConstantWeight because the upstream BFS from a given active infoset may not reach all terminals. If some terminals are only reachable through infosets *not* in the upstream neighborhood, shaped decays assign them floor-level weight while constant gives them full weight.

This experiment tests that hypothesis directly: for each game and each infoset I₀, compute the fraction of total terminals reachable through any infoset in the upstream BFS from I₀.

## Method

For each infoset I₀:
1. Run upstream BFS from I₀ → get set of upstream-reachable infosets (including I₀)
2. Union all z-sets of those infosets → set of reachable terminal IDs
3. Coverage = |reachable terminals| / |total terminals|

No matches or MCCFR iterations needed — pure IIG graph analysis.

**Script**: `experiments/iig_reachability.py`

## Results

### Cross-Game Summary

| Game | Infosets | Terminals | Min Coverage | Avg Coverage | Full (100%) |
|------|---------|-----------|-------------|-------------|-------------|
| kuhn_poker | 12 | 30 | 33.3% | 66.7% | 3/12 |
| leduc_poker | 936 | 5,520 | 16.7% | 99.3% | 918/936 |
| liars_dice | 24,576 | 147,420 | 16.7% | 100.0% | 24,570/24,576 |
| goofspiel | 162 | 576 | 100.0% | 100.0% | 162/162 |

### Per-Game Detail

**Kuhn Poker**: 9 of 12 infosets have incomplete coverage. P0's root infosets (P0|0, P0|1, P0|2) have only 33.3% coverage — each only reaches the 1/3 of terminals consistent with its own card deal. P1's infosets reach 66.7% (their deal + one upstream P0 infoset). Only the 3 deepest infosets (P1 after seeing a bet and knowing the opponent's prior action) reach 100%.

**Leduc Poker**: 18 of 936 infosets have incomplete coverage (all 16.7%). These are exactly P0's 6 round-1 opening infosets (one per private card dealt) × 3 possible hands. Each only sees the 1/6 of terminals consistent with its own deal. The remaining 918 infosets have full coverage through their upstream chains.

**Liar's Dice**: 6 of 24,576 infosets have incomplete coverage (all 16.7%). These are P0's 6 root infosets (P0|1 through P0|6, one per dice roll). Each reaches only the 1/6 of terminals from its own roll. All 24,570 other infosets have 100% coverage.

**Goofspiel**: 100% coverage everywhere. Every infoset reaches all 576 terminals through its upstream BFS. This is because the turn-based simultaneous wrapper interleaves both players' actions, so every infoset has upstream connections spanning the full tree.

### Pattern: The 1/K Root Gap

The incomplete infosets are always **P0's root infosets** — the first player's opening decision after the private chance deal. With K possible private outcomes:
- Coverage = 1/K (only terminals consistent with the dealt private info)
- Upstream BFS size = 1 (only themselves, no predecessors)

For Kuhn (K=3): 33.3%. For Leduc and Liar's Dice (K=6): 16.7%.

## Analysis

### Finding 1: The reachability gap is real but narrow

Only root infosets are affected — 99%+ of infosets have full coverage. This means the reachability gap cannot be the primary explanation for ConstantWeight's broad dominance over shaped decays in archives 007–008. The effect is concentrated at a handful of infosets.

### Finding 2: The gap aligns with δ's impact

The 16.7% coverage at root infosets on Leduc and Liar's Dice corresponds exactly to the private deal structure. The δ parameter fixes this: untargeted iterations (probability 1−δ) sample chance nodes from natural probabilities, providing the missing 83.3% of terminal coverage at these infosets.

This explains why δ<1.0 specifically helps Leduc and Liar's Dice (the two games with incomplete root coverage) and has no effect on Goofspiel (100% coverage everywhere). δ isn't just generic exploration — it fills a specific structural gap that the upstream BFS cannot reach.

### Finding 3: Goofspiel's 100% coverage explains its δ-invariance

Goofspiel's turn-based simultaneous structure means every infoset connects to the full tree. The targeted and untargeted scenarios provide identical terminal coverage. This is why chance+constant on Goofspiel produces identical results across all δ values to 14+ decimal places.

### Finding 4: Constant's strength has a different explanation

Since the reachability gap only affects a few root infosets, ConstantWeight's dominance over shaped decays must come from something else. Archive 008 Finding 4 established the core structural framework: **mode** responds to tree depth, while **decay** responds to tree width. Shaped decay needs width — many distant infosets to deprioritize — to provide value over constant. Only Liar's Dice has enough width (36 private outcomes × wide bidding space, 24,576 infosets) for proximity-based targeting to differentiate, yielding a 1.3% exp_0.5 edge. The other games lack width: Kuhn and Goofspiel have too few IIG levels, and Leduc's IIG chain is linear (depth without width).

This is orthogonal to the mode axis, where Leduc's *depth* enables full mode's chain coherence (archive 008 Finding 2). The two structural properties — depth and width — interact with different ISGT parameters independently.

### Finding 5: Root infosets matter disproportionately

Although only 6/24,576 infosets in Liar's Dice have incomplete coverage, these root infosets are visited on *every single game*. The active player faces their root infoset at the very first decision of every match. Regret computation at these infosets requires counterfactual reasoning across all possible opponent holdings — exactly the terminals that the upstream BFS can't reach.

So while the *count* of affected infosets is small, the *frequency* of visiting them is 100%. This amplifies the importance of the δ mechanism at these specific infosets.

## Connection to δ Implementation

The dual sample-reach tracking (s1/s2) in the ISGT implementation correctly handles this:
- **Targeted iteration (prob δ)**: z* constrains chance sampling. At root infosets, z* only selects terminals consistent with the current deal → 16.7% coverage. `s1` tracks this targeted reach.
- **Untargeted iteration (prob 1−δ)**: Chance nodes sample from natural probabilities. All deals are possible → 100% coverage. `s2` tracks this untargeted reach.
- **Terminal**: `l = δ·s1 + (1−δ)·s2` combines both, correctly importance-weighting the contribution of both scenarios.

## Files

- `iig_reachability_output.txt` — Full terminal output from the diagnostic script

## Resolved Questions

**Archive 008 Q1 (IIG reachability and terminal coverage)**: ✅ **Answered.** Upstream BFS does NOT cover all terminals at root infosets. Coverage = 1/K where K = number of private chance outcomes. This affects only the first player's opening infosets, but they are visited every game. The δ parameter directly fills this gap. However, this narrow gap does not explain ConstantWeight's broad dominance — that comes from insufficient tree *width* in the benchmark games (see Finding 4). Decay needs distant, irrelevant regions to deprioritize; only Liar's Dice is wide enough to provide that.
