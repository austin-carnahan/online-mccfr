# Experiment 001: Depth-Delta Aggregate Exploitability

**Date**: 2026-05
**Goal**: Compare depth-delta (DD) with OOS on aggregate exploitability across games and simulation budgets.

## Setup

- **Algorithms**: OOS (δ=0.5), DD with three schedules: linear, constant(δ=0.5), exp(α=0.5)
- **Games**: kuhn_poker, leduc_poker, goofspiel, liars_dice
- **Budgets**: 500, 1000, 2000, 5000, 10000, 20000, 50000, 100000 sims/move
- **Matches**: 100 per config (against random opponent for aggregate method)
- **Parameters**: ε=0.4, γ=0.01

## Key Findings

### 1. DD's distribution shaping beats OOS at low-to-mid budgets

In goofspiel, leduc, and liar's dice, DD outperforms OOS across most of the budget range. The advantage is largest at low budgets (s≤5000) — roughly 10-17% less exploitable — and narrows as budget increases. In goofspiel and liar's dice, DD still leads or ties OOS even at s=100000.

### 2. Kuhn poker: convergence quality problem (not coverage)

DD flatlines at ~0.042 exploitability from s=5000 onward while OOS continues dropping to 0.007 at s=100000. Initially this looked like a coverage problem, but **Table 2 shows 100% infoset coverage at all budgets for both DD and OOS** — all 12 infosets are reached and updated.

This means the plateau is a **convergence quality** issue: every infoset gets regret updates, but the importance-weighted updates under targeting don't drive the strategy to NE with the same precision as untargeted OOS. Possible causes:

- **Importance weight distortion**: DD's targeting skews which terminals contribute to regret estimates. In a small game requiring precise mixing ratios (e.g., bluff with J exactly 1/3), this bias prevents exact convergence.
- **Average strategy accumulation bias**: The avg strategy is weighted by `opp_reach / l`. Targeting distorts `l` (sample reach), so even if regrets converge, the reported average policy may be off.
- **Asymptotically correct but slow**: DD may still converge to NE eventually, but at a rate much slower than OOS in games small enough for OOS to solve outright.

Notably, in liar's dice DD covers only 57% of infosets at s=100000 yet *beats* OOS (which covers 64%). The mechanism that helps in large games (focused updates on match-relevant states) is exactly what hurts in small games (distorted updates at states trivially reachable anyway).

**Variance profile hypothesis**: DD's per-node blended denominator `l` is on average *better* (larger, more stable) than OOS's denominator — on-path nodes accumulate `δ + (1-δ)·σ_ε(target) ≥ δ`, keeping `l` large and weights small. However, off-path divergence events contribute a small `(1-δ)·σ_ε(a)` factor to `l`, producing rare but large `1/l` weights. DD thus has **lower average variance but higher tail variance** compared to OOS. In large games the tail is harmless (many paths per infoset dilute outliers). In Kuhn, the NE requires exact mixing ratios (e.g., bluff J at 1/3) and there are only 12 infosets — a few high-weight off-path samples dominate the average strategy and prevent precise convergence. OOS avoids this via its binary coin: 50% of iterations are fully untargeted, giving every infoset many moderate-weight updates that average out cleanly regardless of game size.

### 3. Leduc crossover at high budget

OOS just barely overtakes DD at s=100000 (0.402 vs 0.421). This is the only non-trivial game showing a crossover within the tested range. It suggests that for medium-sized games, OOS's uniform exploration eventually catches up — but it takes far more budget than in Kuhn.

### 4. Depth-based schedules show no meaningful signal

The three schedules (linear, constant, exponential) track nearly identically across all games and budgets. Differences are within noise. This strongly suggests that **depth is not the right axis for modulating δ**. The per-node grading based on position in the match history simply doesn't produce different behavior from a flat schedule.

## Interpretation

The core mechanism driving DD's advantage is **distribution shaping** — biasing sampling toward match-relevant terminals — not the depth-dependent δ gradient. A flat δ=0.5 does just as well as a schedule that varies δ from root to leaf. This makes sense: what matters is *whether* you target, not *how aggressively you target at each depth*.

This points toward **adaptive approaches** as the next direction: instead of grading by depth (a static geometric property), modulate targeting based on a dynamic signal — e.g., regret magnitude, value uncertainty, or visit counts at each infoset. The schedule machinery is in place; it just needs a better input signal than depth.

## Next Steps

- Diagnose *why* depth doesn't differentiate: is it because all depths contribute equally to variance, or because the depth range is too small in these games?
- Test adaptive schedules (regret-proportional, value-gap, running-budget)
- Audit DD's average strategy accumulation for correctness under targeting — verify `opp_reach / l` weighting is unbiased
- Determine whether the Kuhn plateau is an inherent property of targeted sampling in small games or a fixable implementation issue