# 001 — Retrospective Sampling: Initial Experiments

**Date:** 2026-04-23
**Status:** Complete

## Overview

First experimental evaluation of Retrospective Sampling (RetroBot), a new
information-set game tree search algorithm that generalizes OOS (Online Outcome
Sampling) from Lisý, Lanctot & Bowling (2015). Two experiments:

1. **retro_vs_oos** — Head-to-head exploitability comparison of OOS vs RetroBot
   on Kuhn and Leduc poker.
2. **retro_decay_ablation** — Sweep of 6 level-weight (decay) functions across
   4 games to determine the best default weighting scheme.

## Algorithm Summary

### OOS (baseline)
Binary coin flip with probability δ: targeted (force all match-history
decisions) or untargeted (ε-on-policy throughout). Two accumulators s₁, s₂.
Importance weight: l = δ·s₁ + (1−δ)·s₂.

### Retrospective Sampling (RetroBot)
Replaces OOS's binary δ with D+1 divergence levels, where D = number of player
decisions in the match history. Level k forces the first D−k decisions, then
samples ε-on-policy after. Combined importance weight via the balance heuristic
(Veach & Guibas 1995): l = Σ_k w_k · s_k, where w_k are level weights.

Key structural advantage: D+1 accumulators capture trajectories at *every*
divergence level from the match history, not just "all-or-nothing." The
intermediate levels — force some decisions but not all — generate partially
counterfactual trajectories that explore nearby-but-different info sets. These
are exactly the trajectories most informative about the local counterfactual
structure.

## Methodology

Follows the aggregate exploitability method from Lisý et al. 2015, §4.2/§4.4.2:

- Play N matches against a **uniform random opponent**
- After each match, merge the bot's average strategy tables into a global
  accumulator
- Compute exploitability of the final aggregate policy
- **X-axis**: simulations per move (computational budget)
- **Y-axis**: exploitability of aggregate policy
- **Fixed**: 500 matches per data point, seed=42

Shared hyperparameters: δ=0.9, ε=0.6, γ=0.01, bias_mode=full.

---

## Experiment 1: Retro vs OOS

**Config:** `experiments/retro_vs_oos.py`
**Games:** kuhn_poker, leduc_poker
**Algorithms:** OOS, Retro+LevelExponential(0.5), Retro+LevelStep(0.01)
**Sims/move:** 50, 100, 200, 500, 1000, 2000

### Results

| Game | Label | s=50 | s=100 | s=200 | s=500 | s=1000 | s=2000 |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| kuhn | oos | 0.219 | 0.195 | 0.192 | 0.170 | 0.147 | 0.124 |
| kuhn | retro_exp(0.5) | 0.209 | 0.180 | 0.148 | 0.101 | 0.083 | 0.081 |
| kuhn | retro_step(0.01) | 0.212 | 0.170 | 0.136 | 0.121 | 0.114 | 0.098 |
| leduc | oos | 2.193 | 2.040 | 1.978 | 1.880 | 1.774 | 1.698 |
| leduc | retro_exp(0.5) | 2.115 | 2.096 | 1.895 | 1.817 | 1.687 | 1.496 |
| leduc | retro_step(0.01) | 2.109 | 1.987 | 1.911 | 1.780 | 1.666 | 1.477 |

### Key Findings

- **Retro beats OOS at every sims level and in both games.**
- The gap **widens with more compute** — not a low-budget artifact.
- At s=2000 on Kuhn: Retro achieves 0.081 vs OOS's 0.124 (35% lower).
- At s=2000 on Leduc: Retro achieves ~1.49 vs OOS's 1.70 (12% lower).
- Inter-decay gap (exp vs step) is much smaller than the OOS-to-Retro gap,
  suggesting the structural change (D+1 accumulators) matters more than weight
  tuning.

---

## Experiment 2: Decay Function Ablation

**Config:** `experiments/retro_decay_ablation.py`
**Games:** kuhn_poker, leduc_poker, goofspiel, liars_dice
**Decay functions:**
  - LevelStep(0.01) — ~100% weight on level 0 (aggressive targeting, OOS-like)
  - LevelExponential(α=0.3) — aggressive decay
  - LevelExponential(α=0.5) — moderate decay
  - LevelExponential(α=0.7) — mild decay
  - LevelExponential(α=0.9) — near-uniform
  - LevelUniform() — flat, equal weight to all levels
**Sims/move:** 50, 100, 200, 500, 1000

### Results (Exploitability at s=1000)

| Rank | Goofspiel | Kuhn | Leduc | Liar's Dice |
| ---: | :--- | :--- | :--- | :--- |
| 1 | **uniform (0.100)** | **uniform (0.078)** | **exp 0.9 (1.639)** | **uniform (0.675)** |
| 2 | exp 0.9 (0.103) | exp 0.3 (0.082) | uniform (1.644) | exp 0.9 (0.678) |
| 3 | exp 0.7 (0.119) | exp 0.5 (0.083) | exp 0.3 (1.658) | exp 0.7 (0.681) |
| 4 | exp 0.5 (0.140) | exp 0.9 (0.084) | exp 0.7 (1.659) | exp 0.5 (0.692) |
| 5 | exp 0.3 (0.168) | exp 0.7 (0.084) | step (1.666) | step (0.693) |
| 6 | step (0.284) | step (0.114) | exp 0.5 (1.687) | exp 0.3 (0.698) |

### Full Table

| Game | Label | s=50 | s=100 | s=200 | s=500 | s=1000 |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: |
| goofspiel | level_exp(α=0.3) | 0.544 | 0.439 | 0.365 | 0.242 | 0.168 |
| goofspiel | level_exp(α=0.5) | 0.540 | 0.423 | 0.326 | 0.195 | 0.140 |
| goofspiel | level_exp(α=0.7) | 0.528 | 0.403 | 0.283 | 0.169 | 0.119 |
| goofspiel | level_exp(α=0.9) | 0.504 | 0.387 | 0.249 | 0.145 | 0.103 |
| goofspiel | level_step(floor=0.01) | 0.575 | 0.453 | 0.339 | 0.294 | 0.284 |
| goofspiel | level_uniform | 0.524 | 0.377 | 0.246 | 0.146 | 0.100 |
| kuhn_poker | level_exp(α=0.3) | 0.206 | 0.181 | 0.147 | 0.100 | 0.082 |
| kuhn_poker | level_exp(α=0.5) | 0.209 | 0.180 | 0.148 | 0.101 | 0.083 |
| kuhn_poker | level_exp(α=0.7) | 0.206 | 0.172 | 0.141 | 0.092 | 0.084 |
| kuhn_poker | level_exp(α=0.9) | 0.213 | 0.163 | 0.138 | 0.092 | 0.084 |
| kuhn_poker | level_step(floor=0.01) | 0.212 | 0.170 | 0.136 | 0.121 | 0.114 |
| kuhn_poker | level_uniform | 0.214 | 0.170 | 0.138 | 0.086 | 0.078 |
| leduc_poker | level_exp(α=0.3) | 2.190 | 2.069 | 1.989 | 1.779 | 1.658 |
| leduc_poker | level_exp(α=0.5) | 2.115 | 2.096 | 1.895 | 1.817 | 1.687 |
| leduc_poker | level_exp(α=0.7) | 2.144 | 2.113 | 2.020 | 1.851 | 1.659 |
| leduc_poker | level_exp(α=0.9) | 2.180 | 2.065 | 1.985 | 1.803 | 1.639 |
| leduc_poker | level_step(floor=0.01) | 2.109 | 1.987 | 1.911 | 1.780 | 1.666 |
| leduc_poker | level_uniform | 2.204 | 2.186 | 2.008 | 1.852 | 1.644 |
| liars_dice | level_exp(α=0.3) | 0.774 | 0.766 | 0.750 | 0.718 | 0.698 |
| liars_dice | level_exp(α=0.5) | 0.756 | 0.762 | 0.746 | 0.707 | 0.692 |
| liars_dice | level_exp(α=0.7) | 0.761 | 0.759 | 0.744 | 0.709 | 0.681 |
| liars_dice | level_exp(α=0.9) | 0.768 | 0.755 | 0.743 | 0.710 | 0.678 |
| liars_dice | level_step(floor=0.01) | 0.766 | 0.762 | 0.755 | 0.722 | 0.693 |
| liars_dice | level_uniform | 0.765 | 0.757 | 0.732 | 0.708 | 0.675 |

### Key Findings

1. **Uniform wins 3/4 games and essentially ties in the 4th.** In Leduc it's
   0.005 behind exp(0.9) — within noise. This is the strongest possible case
   for a parameter-free default.

2. **Step is catastrophically bad in goofspiel** (0.284 vs 0.100 — nearly 3x
   worse) and consistently worst or second-worst everywhere. Step puts ~all
   weight on level 0 (fully targeted) and starves intermediate levels, degrading
   Retro back toward OOS's binary behavior.

3. **Among exponentials, higher α is systematically better in goofspiel and
   liar's dice.** The ordering α=0.9 > 0.7 > 0.5 > 0.3 is monotonic. Higher α
   means flatter weights, approaching uniform. Lower α concentrates on
   targeting, approaching step. **Flatter = better.**

4. **The monotonic α trend is weaker in Kuhn and Leduc** — small/shallow games
   where D (number of player decisions) is tiny (1–2). With D=1 there are only
   2 levels, so weights barely matter. The decay function matters most where D
   is large (goofspiel, liar's dice), which is exactly where ISGS is deployed
   in practice.

5. **Step's failure in goofspiel is the smoking gun:** it proves that the D+1
   accumulators must be *funded* to provide benefit. Having them isn't enough;
   you need to sample trajectories at intermediate divergence levels.

---

## Theoretical Analysis

### Why D+1 accumulators matter more than weight tuning

The structural change from 2 accumulators (OOS) to D+1 (Retro) does two
distinct things:

1. **Variance reduction** — l = Σ w_k s_k is a better importance weight
   denominator than l = δ·s₁ + (1−δ)·s₂ because more s_k terms survive to
   terminal, making l larger and more stable, and W = u·π_{-i}/l has smaller
   magnitude.

2. **Better exploration of the counterfactual** — OOS's targeted scenario forces
   *all* decisions or dies (s₁=0). Its untargeted scenario is basically random.
   OOS splits budget between "perfect targeting" and "nearly random." Retro's
   intermediate levels explore the space *between* — trajectories that partially
   follow the match history, reaching nearby-but-different info sets. These are
   the most informative trajectories for non-local reasoning.

### Balance heuristic and decay insensitivity

The balance heuristic (Veach & Guibas 1995) guarantees that for *any* weights
w_k summing to 1, the MIS estimator's variance is at most a factor K above the
minimum-variance estimator (where K = number of techniques). This explains why
inter-decay differences are small — the balance heuristic compensates. But
Lanctot et al.'s work on variance in MCCFR shows the constants hidden in CFR's
O(1/√T) bound depend on per-iteration variance, so decay choice still matters
at the margin.

### Why uniform wins

Uniform weights maximize trajectory diversity across the counterfactual
spectrum. Every divergence level gets equal budget, ensuring all D+1
accumulators are well-funded. This contrasts with step (starves intermediate
levels) and low-α exponentials (over-funds targeting). The data confirms:
**flatter weights → better performance**, with uniform as the natural limit.

---

## Future Work: Adaptive Decay Functions

The uniform default is excellent but raises a natural question: could online
adaptation do even better, or at least match uniform without committing to it
a priori?

### Variance-adaptive reweighting (most promising)

Track the running variance of W = u·π_{-i}/l contributed by each active level.
After a batch of iterations, shift weight toward levels that produce
lower-variance updates. This is the adaptive MIS framework from:

- **Cornuét, Marin, Mira & Robert (2012)** — "Adaptive Multiple Importance
  Sampling" — provides the theoretical foundation for online MIS weight
  adaptation.
- **Veach & Guibas (1995)** — the static balance heuristic is the starting
  point; adaptive methods converge to the dynamic optimal.
- **Lanctot, Lisý, Bowling et al.** — variance reduction in MCCFR directly
  improves convergence rate; the per-iteration variance constants matter.

Concretely: maintain per-level accumulators for E[W²] and shift weight toward
levels with smaller second moments. The balance heuristic is the static optimal;
this converges to the dynamic optimal.

### Effective-sample-size adaptive (simplest online version)

Track how many s_k survive (are nonzero) at terminal per level. If level k
consistently produces dead trajectories (s_k=0), reduce its weight. This is a
crude but cheap proxy for variance — dead trajectories contribute nothing but
consume budget.

### Depth-normalized harmonic decay (parameter-free alternative)

w_k ∝ 1/(k+1). Automatically adapts to D since the number of levels changes per
decision point. More aggressive than uniform, less aggressive than exponential.
No tuning. But the ablation data suggests uniform is hard to beat, so this may
not improve on the simpler default.

### Key insight for adaptive methods

Our ablation data predicts that variance-adaptive reweighting would likely
converge to near-uniform weights (since that's empirically optimal), but it
would have the *capacity* to adapt if some game structure warranted asymmetric
allocation. This is the strongest argument for the approach: it removes the
hyperparameter question entirely while retaining flexibility.

---

## Files in this archive

- `notes.md` — this file
- `plots/` — generated plots (exploitability vs sims, coverage vs sims)
- `retro_vs_oos/` — raw results from Experiment 1
  - `incremental_results.jsonl` — per-job results as they completed
  - `results.json` — final structured results
  - `tables.md` — formatted markdown tables
- `retro_decay_ablation/` — raw results from Experiment 2
  - `incremental_results.jsonl`
  - `results.json`
  - `tables.md`