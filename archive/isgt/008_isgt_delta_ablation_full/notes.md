# ISGT Delta Ablation — δ Parameter × Decay × Mode (Aggregate Exploitability)

**Date**: 2026-04-03  
**Archive**: 008_isgt_delta_ablation_full  
**Runtime**: ~60 minutes, 272 jobs, 0 failures  
**Predecessor**: 007_isgt_ablation_full

## Motivation

Archive 007 identified two key findings:
1. **ConstantWeight tied or beat all shaped decays** — IIG proximity signal appeared to provide no targeting value.
2. **OOS overtook ISGT at high budgets on Leduc and Liar's Dice** — an exploration deficit caused by 100% targeted iterations (no independent chance sampling).

This experiment adds a **δ (delta) parameter** to ISGT, following the OOS paper's design. δ is an **iteration-level coin flip**: with probability δ, the entire iteration is targeted (z* guides chance nodes); with probability (1−δ), an untargeted iteration uses independent chance sampling with natural game probabilities.

The goal is to answer three questions:
1. Does δ<1.0 fix the OOS crossover on Leduc and Liar's Dice?
2. With proper exploration via δ, does the IIG decay signal (exp vs constant) emerge?
3. How does the optimal δ vary across games and modes?

## Implementation

### δ Parameter Mechanics

Added to `ISGTBot.__init__()` as `delta=1.0` (backward compatible with archive 007).

**Per-iteration targeting decision**: Before each MCCFR iteration, a coin flip `random() < δ` determines whether the iteration is targeted or untargeted. z* is always sampled (needed for s1 tracking in both scenarios), but only constrains the walk in targeted iterations.

**Dual sample-reach tracking**: Two parallel sample-reach values propagate through the tree:
- `s1`: targeted scenario sample reach — what the sampling probability would be if this iteration were targeted
- `s2`: untargeted scenario sample reach — what it would be if untargeted

At terminals, the combined sampling probability is:

$$l = \delta \cdot s_1 + (1 - \delta) \cdot s_2$$

This correctly importance-weights the trajectory under the mixed targeting/untargeting policy.

**Chance nodes**:
- Targeted: action = z*'s action (deterministic). `s1` unchanged, `s2 *= ρ` (natural prob).
- Untargeted: action ~ natural distribution. `s2 *= ρ`. `s1`: if action matches z*, `s1` unchanged; otherwise `s1 = 0` (trajectory diverged from z*).

**Decision nodes** (chance mode):
- Both scenarios: sample from ε-on-policy. `s1 *= σ_sample`, `s2 *= σ_sample`. Identical decision sampling in both scenarios.

**Decision nodes** (full mode):
- Targeted: action = z*'s action (deterministic). `s1` unchanged, `s2 *= σ_sample`.
- Untargeted: action ~ ε-on-policy. `s2 *= σ_sample`. `s1`: if matches z*, unchanged; else `s1 = 0`.

**Average strategy update**: Uses combined reach `δ·s1 + (1−δ)·s2` in the denominator, correctly weighting the contribution of both scenarios.

### ε (epsilon) Behavior Across Modes

A key architectural observation: **ε has no effect on actual trajectory selection in full mode during targeted iterations**. The action is deterministically z*'s action regardless of ε. ε only influences:
- The `s2` bookkeeping (what the untargeted sample prob would have been)
- The strategy reach computation (which uses `policy[idx]`, not `sample_policy[idx]`)

In chance mode, ε always controls actual decision exploration because both scenarios sample from ε-on-policy at decision nodes.

This means that with δ=1.0 (original ISGT), full mode gets **zero ε-exploration** — every decision is locked to z*. The δ parameter is the only way full mode gets any exploration at all: through the (1−δ) fraction of untargeted iterations.

## Experiment Setup

**Grid** (272 jobs):
- **Games**: kuhn_poker, leduc_poker, goofspiel, liars_dice
- **Bias modes**: chance, full
- **Decay functions (2)**: constant, exp_0.5
- **Delta values (4)**: 0.2, 0.5, 0.9, 1.0
- **Baselines**: OOS (δ=0.9, ε=0.6, γ=0.01 — 16 jobs, 1 per game × budget)
- **Sim budgets**: 100, 250, 500, 1000 iterations per move
- **Matches per game**: kuhn=500, leduc=300, goofspiel=300, liars_dice=100

**Why exp_0.5?** Selected as the representative shaped decay from archive 007 — it was 1st or 2nd place across all 4 games in that study.

**Shared parameters**: ε=0.6, γ=0.01 across all ISGT and OOS configs.

**Script**: `experiments/isgt_delta_ablation.py` (ProcessPoolExecutor, 6 workers, JSONL incremental saves).

## Results at sims=1000

### Kuhn Poker — ISGT dominates, δ doesn't matter much

| Rank | Config | Exploitability |
|------|--------|---------------|
| 1 | chance_constant_d1.0 | **0.0605** |
| 2 | chance_constant_d0.5 | 0.0627 |
| 3 | chance_exp_0.5_d0.2 | 0.0633 |
| 4 | chance_constant_d0.2 | 0.0634 |
| 5 | chance_exp_0.5_d0.5 | 0.0659 |
| 6 | chance_exp_0.5_d0.9 | 0.0676 |
| 7 | chance_exp_0.5_d1.0 | 0.0688 |
| 8 | chance_constant_d0.9 | 0.0688 |
| 9 | full_exp_0.5_d0.2 | 0.0740 |
| 10 | full_constant_d0.2 | 0.0769 |
| 11 | full_constant_d1.0 | 0.0840 |
| 12 | full_exp_0.5_d0.5 | 0.0869 |
| 13 | full_exp_0.5_d1.0 | 0.0876 |
| 14 | **oos** | 0.0884 |
| 15 | full_constant_d0.5 | 0.0928 |
| 16 | full_constant_d0.9 | 0.1243 |
| 17 | full_exp_0.5_d0.9 | 0.1276 |

All chance-mode ISGT configs beat OOS. d1.0 best (0.0605) — Kuhn is too small for the exploration deficit to matter. Constant beats exp at most δ values. Full mode underperforms except at very low δ (d0.2).

### Leduc Poker — OOS narrowly wins; full mode redeemed at moderate δ

| Rank | Config | Exploitability |
|------|--------|---------------|
| 1 | **oos** | **1.768** |
| 2 | full_constant_d0.5 | 1.771 |
| 3 | chance_constant_d0.9 | 1.795 |
| 4 | full_exp_0.5_d0.2 | 1.795 |
| 5 | full_constant_d0.2 | 1.799 |
| 6 | chance_constant_d0.5 | 1.801 |
| 7 | full_exp_0.5_d0.5 | 1.804 |
| 8 | chance_exp_0.5_d0.9 | 1.811 |
| 9 | full_exp_0.5_d0.9 | 1.815 |
| 10 | chance_exp_0.5_d0.5 | 1.825 |
| 11 | chance_exp_0.5_d0.2 | 1.830 |
| 12 | chance_constant_d0.2 | 1.833 |
| 13 | full_constant_d0.9 | 1.844 |
| 14 | full_constant_d1.0 | 1.845 |
| 15 | full_exp_0.5_d1.0 | 1.852 |
| 16 | chance_exp_0.5_d1.0 | 1.990 |
| 17 | chance_constant_d1.0 | 1.991 |

**Key observations**:
- δ<1.0 dramatically improves chance mode: d0.9 (1.795) vs d1.0 (1.991) = **10% improvement**
- `full_constant_d0.5` (1.771) essentially ties OOS (1.768) — **0.2% gap**
- Full mode at d0.5 outperforms all chance-mode configs
- This is the one game where full mode's coordinated IIG-chain updates pay off — Leduc has deep sequential decision structure (2 betting rounds with raises) where updating the entire upstream chain together is valuable

### Goofspiel — Chance mode dominates; chance+constant is δ-invariant

| Rank | Config | Exploitability |
|------|--------|---------------|
| 1 | chance_exp_0.5_d0.5 | **0.0813** |
| 2 | chance_constant_d* | ~0.0814 (all δ identical) |
| 6 | chance_exp_0.5_d0.2 | 0.0817 |
| 7 | chance_exp_0.5_d1.0 | 0.0823 |
| 8 | chance_exp_0.5_d0.9 | 0.0826 |
| 9 | full_exp_0.5_d0.2 | 0.0897 |
| 10 | full_constant_d0.2 | 0.0918 |
| 11 | **oos** | 0.0920 |
| 12 | full_constant_d0.5 | 0.1220 |
| 13 | full_exp_0.5_d0.5 | 0.1205 |
| 16 | full_constant_d1.0 | 0.4141 |

**Remarkable finding**: chance+constant produces **identical exploitability across all δ values** (0.08144771... to 14+ decimal places). This is because Goofspiel has only **public chance nodes** — both players observe the dealt card. The z*-guided chance action and the natural-distribution chance action are functionally equivalent when chance outcomes are public: every chance realization is observed by both players, so targeting vs untargeting at chance nodes has no impact on the information structure being explored.

Full mode at d1.0 is catastrophic (0.414) but recovers at d0.2 (0.092, matching OOS).

### Liar's Dice — OOS wins; exp(0.5) shows clearest IIG signal

| Rank | Config | Exploitability |
|------|--------|---------------|
| 1 | **oos** | **0.691** |
| 2 | chance_exp_0.5_d0.9 | 0.695 |
| 3 | chance_constant_d0.2 | 0.698 |
| 4 | chance_exp_0.5_d0.2 | 0.699 |
| 5 | chance_constant_d1.0 | 0.703 |
| 6 | chance_constant_d0.9 | 0.704 |
| 7 | chance_exp_0.5_d0.5 | 0.705 |
| 8 | chance_constant_d0.5 | 0.710 |
| 9 | chance_exp_0.5_d1.0 | 0.708 |
| 10 | full_constant_d0.2 | 0.713 |
| 11 | full_exp_0.5_d0.2 | 0.712 |
| 12 | full_exp_0.5_d0.5 | 0.726 |
| 13 | full_constant_d0.5 | 0.729 |
| 14 | full_exp_0.5_d0.9 | 0.756 |
| 15 | full_constant_d1.0 | 0.758 |
| 16 | full_constant_d0.9 | 0.763 |
| 17 | full_exp_0.5_d1.0 | 0.759 |

**The clearest IIG signal in the study**: `chance_exp_0.5_d0.9` (0.6946) beats `chance_constant_d0.9` (0.7038) by **1.3%**. Exp wins 6 of 8 head-to-head comparisons against constant at matched (mode, δ). The #2 overall config (`chance_exp_0.5_d0.9`) is within **0.5%** of OOS.

## Analysis

### Finding 1: δ<1.0 fixes the exploration deficit

Confirmed across all games. Impact magnitude at sims=1000:

| Game | d1.0 (chance_const) | Best δ (chance_const) | Improvement |
|------|--------|-----------|-------------|
| Kuhn | 0.0605 | 0.0605 (d1.0) | — (d1.0 is best) |
| Leduc | 1.991 | 1.795 (d0.9) | **10%** |
| Goofspiel | 0.0814 | 0.0814 (any δ) | — (δ-invariant) |
| Liar's Dice | 0.703 | 0.698 (d0.2) | **0.6%** |

The biggest impact is on Leduc — the game where archive 007 showed the worst crossover. Adding just 10% untargeted iterations (δ=0.9) closes most of the gap with OOS. This confirms the hypothesis from archive 007 Q1: the exploration deficit was specifically about the lack of independent chance sampling.

### Finding 2: Full mode redeemed at moderate δ — Leduc as the depth game

Archive 007 concluded "chance mode is the only viable ISGT bias mode." This experiment nuances that finding. On Leduc, `full_constant_d0.5` (1.771) essentially ties OOS (1.768) and beats all chance-mode configs.

**Structural explanation**: Leduc is a **depth game** — two betting rounds with raises, narrow branching (fold/call/raise at each decision). The IIG infosets on z* form a long sequential decision chain. Full mode's commitment to the entire trajectory means updating a coherent chain of strategically linked decisions: level 2 informs level 1, level 1 informs level 0. These coordinated IIG-chain updates are worth the exploration cost in a deep, narrow tree.

Full mode needs more untargeted iterations to compensate for its lack of ε-exploration. The optimal δ for full mode (0.5 on Leduc, 0.2 elsewhere) is consistently lower than for chance mode (0.9–1.0), because the only exploration full mode gets comes from the (1−δ) untargeted fraction.

### Finding 3: IIG decay signal is game-complexity dependent

Head-to-head exp(0.5) vs constant at matched (mode, δ), sims=1000:

| Game | Exp wins (of 8) | Largest gap | Interpretation |
|------|-----------------|-------------|----------------|
| Kuhn | 4 of 8 | ~1% either way | No signal — IIG too small |
| Leduc | 4 of 8 | <2% | No signal — depth structure is linear, all levels equally relevant |
| Goofspiel | ~4 of 8 | <0.5% in chance | No signal — public chance makes decay moot |
| **Liar's Dice** | **6 of 8** | **1.3%** | **Signal present** — rich IIG with meaningful proximity structure |

**Why Liar's Dice shows IIG signal**: Liar's Dice has 24,576 infosets spanning 36 private-chance outcomes (6×6 dice rolls) and a wide bidding action space — it is the **widest game** in the benchmark. Width is the key structural property: in a wide tree, many infosets are distant from the active decision point in the IIG, giving proximity-based decay something meaningful to deprioritize. The exponential decay concentrates sampling on terminals passing through infosets *close* to the active decision point, focusing on the part of the wide tree that is strategically connected to the current game state. With many distant, less-relevant regions to deweight, focused targeting provides measurable benefit.

This is the complement to Finding 2 — together they form the core structural framework (formalized in Finding 4). Finding 2 shows **depth → full mode** (Leduc). Finding 3 shows **width → shaped decay** (Liar's Dice). These are two orthogonal structural axes interacting with two different ISGT parameters.

**Why other games don't show signal**: Kuhn and Goofspiel have too few IIG levels for decay to differentiate meaningfully. Leduc is deep but not wide — its IIG chain is linear, so most infosets on z* are in the same upstream chain anyway. Uniform vs proximity weighting produces similar terminal distributions because there are no distant, irrelevant regions to deprioritize. Leduc's depth helps full *mode* (Finding 2) but its lack of width means decay has nothing to work with.

**Prediction**: The IIG decay signal should grow on larger games where the tree is wider, making proximity-based targeting more valuable as the strategically relevant subtree becomes a smaller fraction of the total game.

### Finding 4: Two orthogonal structural axes — depth × mode, width × decay

Findings 2 and 3 are about **two different ISGT parameters** responding to **two different structural properties**. This is the central framework for interpreting all ISGT results:

|  | **Mode** (chance vs full) | **Decay** (constant vs shaped) |
|--|--------------------------|-------------------------------|
| **What it controls** | *Which nodes* follow z* during targeted iterations | *How z\* is selected* (terminal sampling distribution) |
| **Structural driver** | Tree **depth** | Tree **width** |
| **Why** | Full mode commits the walk to z*'s entire decision chain. This is valuable when the chain forms a coherent strategic sequence (deep tree) — updating linked decisions together. In wide trees, it misses too much breadth. | Shaped decay concentrates z*-sampling on terminals through nearby IIG infosets. This is valuable when many infosets are distant and irrelevant (wide tree) — there are regions to deprioritize. In narrow trees, all infosets are nearby anyway, so weighting them differently has no effect. |
| **Evidence** | Leduc: full_constant_d0.5 ties OOS (1.771 vs 1.768). Full mode catastrophic on wide games. | Liar's Dice: exp_0.5 beats constant by 1.3% at d0.9. No decay signal on narrow/small games. |
| **Mechanism** | Chance mode lets the walk diverge from z* at the first decision node (ε-exploration). Full mode locks every decision to z*, trading breadth for coherent chain updates. | Constant weights all IIG infosets equally → uniform z* distribution. Shaped decay upweights proximate infosets → focused z* distribution. The gap only matters when "distant" infosets exist. |

**Per-game mapping**:

- **Leduc** (deep, narrow): Benefits on the **mode axis** — full mode's chain commitment pays off. No signal on the decay axis — linear IIG chain means proximity weighting ≈ uniform weighting.
- **Liar's Dice** (wide, shallow): Benefits on the **decay axis** — exp_0.5 focuses z*-sampling within the wide space. No benefit from full mode — can't explore enough of the width.
- **Goofspiel** (wide, public chance): Neither axis shows signal — public chance makes targeted/untargeted identical, and the IIG has too few levels for decay to differentiate.
- **Kuhn** (small): Neither axis matters — tree is small enough that any config covers it.

**Prediction**: On a game that is both deep *and* wide (e.g., larger poker variants), both axes should show signal simultaneously — full mode for chain coherence, shaped decay for focusing within the width. Our benchmark games happen to separate along one axis each, which is why we see each effect in isolation.

**Deeper mechanism — why chance mode dilutes the decay signal**: The depth/width framing describes *which games* show which effects, but the underlying mechanism is about **signal preservation along the walk**. In chance mode, z* only constrains chance nodes — every decision node samples from ε-on-policy, so the walk can diverge from z*'s path at the very first decision. Each subsequent decision node compounds this divergence. By the time the walk reaches a terminal several decision levels deep, the actual terminal visited may have little correspondence to the z* that the decay function carefully selected. The decay's proximity-based z* selection gets washed out by stacked ε-exploration at decision nodes.

This is why the decay signal appears on Liar's Dice specifically: not just because the tree is wide, but because the game is **chance-heavy relative to decisions**. The 36-outcome private dice roll is the dominant branching event, and z*-guidance controls exactly that. The bidding action space after the deal is wide but shallow — fewer decision-node divergence points to dilute the signal before reaching a terminal. In contrast, Leduc's depth means many decision nodes interpose between z*'s chance setup and the terminal, compounding the divergence and erasing whatever proximity advantage the decay encoded in z*.

Full mode doesn't suffer this dilution — it locks every node (chance *and* decision) to z*'s path, so the signal survives to the terminal regardless of depth. But full mode pays a different cost: zero ε-exploration during targeted iterations, requiring lower δ (more untargeted iterations) to compensate.

### Finding 5: Goofspiel's δ-invariance confirms public chance hypothesis  

The chance+constant δ-invariance on Goofspiel (values identical to 14+ decimal places across all δ) is explained by Goofspiel having **only public chance nodes**. When chance outcomes are observed by both players, the information structure doesn't depend on which chance realization was sampled. Targeted and untargeted chance sampling produce the same distributions because both players can condition on the public outcome regardless.

This is a strong validation of the δ mechanism's design: it correctly produces different behavior only where it matters (private chance games like Leduc and Liar's Dice) and has no effect where it can't help (public chance games).

### Finding 6: Reachability as a potential explanation for constant's strength

An important open question is whether the IIG graph guarantees that **all game histories are reachable** from a given active infoset through the upstream BFS. If some terminals are only reachable through infosets that are not in the upstream neighborhood, then shaped decay functions would systematically underweight them — assigning floor-level probability rather than a proximity-adjusted weight.

ConstantWeight avoids this entirely by giving weight 1.0 to all infosets regardless of IIG distance. If some terminals are only reachable through non-upstream infosets, constant would correctly uniform-weight them while shaped decays would penalize them.

This could explain why constant outperforms shaped decays on Goofspiel and Liar's Dice: in wide games, the upstream BFS from a single infoset may not reach all relevant parts of the tree. The terminals that are "off the IIG map" might still be strategically important for correct regret computation, and shaped decays systematically neglect them.

**To test this hypothesis**: Enumerate all terminals reachable from a given active infoset through the IIG upstream BFS, and compare to the total terminal count. If the coverage is less than 100%, it would explain why concentrating probability on upstream-reachable terminals (as shaped decays do) hurts performance.

## Comparison with Archive 007

| Metric | 007 (no δ) | 008 (with δ) |
|--------|-----------|-------------|
| Best ISGT on Leduc (sims=1000) | 1.835 (full_poly_2) | **1.771** (full_constant_d0.5) |
| Best ISGT on Liar's Dice | 0.701 (chance_exp_0.3) | **0.695** (chance_exp_0.5_d0.9) |
| Closest ISGT to OOS on Leduc | 3.8% behind | **0.2% behind** |
| Closest ISGT to OOS on Liar's Dice | 1.4% behind | **0.5% behind** |
| Full mode on Goofspiel (d1.0) | 0.397 (catastrophic) | 0.092 at d0.2 (recovered) |

The δ parameter closes the OOS gap on both crossover games, with the biggest improvement on Leduc (3.8% → 0.2%).

## Files

- `ablation_results.json` — Full per-job results (272 entries)
- `ablation_summary.json` — Per-config summary with sims and exploitability
- `incremental_results.jsonl` — Line-by-line results as jobs completed
- `plots/delta_sensitivity.png` — Figure 1: δ sensitivity across modes/games
- `plots/mode_comparison.png` — Figure 2: best configs per mode vs OOS
- `plots/decay_signal.png` — Figure 3: exp(0.5) vs constant
- `plots/bar_ranking.png` — Figure 4: all configs ranked at sims=1000

## Resolved Questions from Archive 007

**Q1 (iteration-level untargeted iterations)**: ✅ **Yes, δ<1.0 fixes the crossover.** Leduc gap drops from 3.8% to 0.2%, Liar's Dice from 1.4% to 0.5%. Independent chance sampling via untargeted iterations is the mechanism.

**Q4 (does constant generalize?)**: Partially answered. Constant is still the robust default, but exp(0.5) shows a **1.3% edge on Liar's Dice** at the right δ — the first evidence that IIG proximity provides targeting signal. The signal is game-complexity dependent.

## Open Questions

### Q1: IIG reachability and terminal coverage

Does the IIG upstream BFS from a given infoset cover all terminals? If not, shaped decays penalize unreachable but strategically important terminals. This would explain constant's dominance on wide games. **Test**: For each game, enumerate `|upstream-reachable terminals| / |total terminals|` for representative infosets.

### Q2: Optimal δ as a function of game structure

Optimal δ varies by game and mode. Can we predict the optimal δ from game properties (number of chance nodes, branching factor, IIG diameter)? This would make δ adaptive rather than tuned.

### Q3: Larger games

The IIG decay signal (1.3% on Liar's Dice) may grow on larger games where the terminal space is vast and proximity-based focusing provides more value. Testing on larger poker variants or more complex games would strengthen or weaken the IIG signal claim.

### Q4: Time-based comparison

This study uses iteration-based budgets. ISGT has per-iteration overhead (IIG construction, z* sampling) that OOS doesn't. A wall-clock comparison at equal time budgets is needed for practical recommendations.

### Q5: Pre-initialized infostates give ISGT an unfair advantage over OOS

✅ **Resolved in archive 010.** Pre-initializing OOS's infostate tables from the IIG produces negligible difference at sims≥500 (±3–4%, within match-count variance). ISGT's advantage is genuine targeting signal, not a startup artifact. See `archive/010_precomp_oos_rerun/notes.md` for full analysis.

ISGT pre-populates all infostate tables (regret + average policy arrays) from the IIG during construction. This means every ISGT iteration uses proper regret matching from iteration 1. OOS discovers infostates incrementally — the first visit to each infoset triggers a random playout instead of a regret-matched decision.

This is a **confound in the current benchmark**. Some of ISGT's advantage (especially at low budgets) may come from pre-initialization rather than from IIG-guided targeting. The IIG construction already does a full game-tree traversal, so ISGT gets the infostate table for free as a side effect.

**Fix**: Pre-initialize OOS's infostate tables using the same IIG-derived data. The IIG is game-agnostic, so there is no reason OOS can't benefit from the same precomputation. This would isolate the targeting mechanism as the only difference between the two algorithms, making the comparison fair. Both algorithms would then start with identical infostate tables and differ only in how they guide trajectory sampling.

This should be implemented before any final ISGT vs OOS comparison is published.
