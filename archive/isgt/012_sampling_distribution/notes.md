# 012 — Sampling Distribution Analysis

**Date**: 2026-04-06  
**Archive**: 012_sampling_distribution  
**Predecessor**: 011_validation  
**Status**: ✅ Complete — led directly to two-stage sampling refactor

## Motivation

Archives 007–008 established that ISGT is competitive with OOS on some games (Kuhn, Goofspiel) and achieves near-parity on others (Leduc within 0.01%, Liar's Dice within 0.1%), but does not clearly beat it overall. Within ISGT, a persistent puzzle remained: ConstantWeight (uniform z* selection with no IIG proximity signal) tied or beat all shaped decay functions. The shaped decays were supposed to concentrate sampling on terminals passing through nearby IIG infosets, but this proximity signal showed no consistent benefit.

Archive 011 confirmed the implementation is correct. This experiment asks **why** the decay functions aren't helping: by directly measuring where z* samples land across IIG distance levels, we can see what the decay functions are actually doing to the sampling distribution.

**What we found was that the decay functions were not doing what we thought they were doing.** The per-terminal weighting scheme interacts with the game tree's structure in ways that distort the intended level-allocation semantics of the decay function, explaining ConstantWeight's dominance and motivating a fundamental redesign of the z* selection mechanism.

## Method

A dedicated analysis harness (`experiments/sampling_analysis_012.py`) runs small batches of ISGT iterations and records two observables per iteration:

1. **z* IIG distance** — the IIG distance of the sampled target terminal (reflects the theoretical sampling distribution induced by the decay function)
2. **Reached terminal IIG distance** — the IIG distance of the terminal the walk actually reaches (reflects the effective targeting after δ, mode, and ε interact)

Each observable is histogrammed by IIG distance level (0, 1, 2, ... and -1 for floor/non-upstream terminals).

## Parameter Sweeps

| Sweep | Varying | Fixed | Question |
|-------|---------|-------|----------|
| Decay | constant, exp(0.5), exp(0.1), step(0.01) | δ=0.9, chance | Does decay concentrate z* at lower levels? |
| Delta | 0.0, 0.2, 0.5, 0.9, 1.0 | exp(0.5), chance | Does δ dilute the targeting signal? |
| Mode | chance, full | exp(0.5), δ=0.9 | Does full mode over-concentrate? |
| Epsilon | 0.0, 0.2, 0.6, 1.0 | exp(0.5), δ=0.9, chance | Does exploration broaden the distribution? |

## Histograms Produced

For each configuration, three histograms:
1. **z* targeted-only** — pure IIG + decay effect
2. **Reached targeted-only** — effective targeting
3. **Reached all-samples** — full algorithm behavior

## Files

- `experiments/sampling_analysis_012.py` — Analysis harness

## Results

**Run**: 500 sims/config, seed=42, 4 games × 4 sweeps = 16 experiments.  
**Active infoset**: Deepest infoset per game (max upstream BFS depth).

### Game-Level IIG Structure

| Game | Active Infoset | BFS Depth | Upstream Terminals | Total Terminals | Levels Present |
|------|---------------|:---------:|-------------------:|----------------:|:---------------|
| Kuhn Poker   | P0\|0pb | 2  | 30     | 30      | 0, 1, 2 |
| Leduc Poker  | P1\|..Round2.. | 7  | 5,520  | 5,520   | 0–7 |
| Goofspiel    | P1\|...depth5 | 5  | 576    | 576     | 0, 2, 4 (odd levels empty) |
| Liar's Dice  | P0\|1 1-1..2-6 | 12 | 147,420 | 147,420 | 0–12 |

All four games have 100% upstream coverage from the deepest infoset — every terminal passes through at least one upstream infoset. The IIGs have rich multi-level structure: Kuhn has 3 levels, Leduc has 8, Goofspiel has 3 active levels (alternating player structure skips odd levels), and Liar's Dice has 13.

### Sweep 1 — Decay Function Comparison

Fixed: δ=0.9, chance mode, ε=0.6.

**Kuhn Poker** (3 levels):

| Decay       | L0  | L1  | L2  |
|-------------|:---:|:---:|:---:|
| constant    | 15% | 27% | 57% |
| exp(0.5)    | 33% | 32% | 35% |
| exp(0.1)    | 81% | 15% | 4%  |
| step(0.01)  | 95% | 1%  | 4%  |

**Leduc Poker** (8 levels):

| Decay       | L0  | L1  | L2  | L3  | L4  | L5  | L6  | L7  |
|-------------|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| constant    | 0%  | 1%  | 1%  | 1%  | 14% | 21% | 20% | 42% |
| exp(0.5)    | 5%  | 16% | 6%  | 7%  | 26% | 23% | 8%  | 9%  |
| exp(0.1)    | 56% | 42% | 2%  | 0%  | 0%  | —   | —   | —   |
| step(0.01)  | 13% | 1%  | 1%  | 2%  | 10% | 21% | 16% | 36% |

**Liar's Dice** (13 levels):

| Decay       | L0  | L1  | L2  | L3–L9 | L10 | L11 | L12 |
|-------------|:---:|:---:|:---:|:-----:|:---:|:---:|:---:|
| constant    | 0%  | 0%  | 0%  | 9%    | 10% | 27% | 53% |
| exp(0.5)    | 2%  | 11% | 8%  | 52%   | 6%  | 7%  | 7%  |
| exp(0.1)    | 43% | 46% | 8%  | 2%    | 0%  | 1%  | 1%  |
| step(0.01)  | 0%  | 0%  | 0%  | 12%   | 11% | 27% | 50% |

**Goofspiel** (levels 0, 2, 4):

| Decay       | L0  | L2  | L4  |
|-------------|:---:|:---:|:---:|
| constant    | 0%  | 6%  | 94% |
| exp(0.5)    | 9%  | 21% | 70% |
| exp(0.1)    | 92% | 7%  | 2%  |
| step(0.01)  | 44% | 3%  | 53% |

**Finding 1 — The decay function does shape z* across levels — but not as intended.** With deep active infosets, the IIG reveals genuine multi-level structure. Steeper decays (exp(0.1)) concentrate >80% of z* at L0–L1, while constant weight is heavily bottom-loaded — 42% at L7 for Leduc, 53% at L12 for Liar's Dice. The decay parameter does affect the distribution, but the relationship between intended level allocation and actual level allocation is distorted by terminal-count imbalance (see Core Discovery below).

**Finding 2 — Constant weight is anti-targeted.** Constant assigns uniform weight to all infosets, but distant levels contain exponentially more terminals. After normalization across all terminals, the result is a sampling distribution dominated by the farthest upstream levels — the opposite of what proximity-based targeting intends. On Liar's Dice, 53% of z* lands at L12 (farthest from active infoset) and effectively 0% at L0.

**Finding 3 — step(0.01) does NOT concentrate at L0 in deep games.** On Kuhn (depth 2), step puts 95% at L0. But on Leduc (depth 7) it only puts 13% at L0 and 36% at L7 — similar to constant. On Liar's Dice (depth 12), step is nearly identical to constant (50% at L12). The floor weight of 0.01 is not low enough relative to the large number of terminals at distant levels. The many terminals at deep levels overwhelm the per-terminal weight advantage of L0.

**Finding 4 — exp(0.5) provides moderate concentration.** It spreads samples more evenly across levels than constant but less aggressively than exp(0.1). On Liar's Dice, it achieves a roughly uniform spread across all 13 levels (2–11% per level), which may be why archive 008 found a small edge for exp(0.5) on Liar's Dice.

### Sweep 2 — δ (Targeting Probability)

Fixed: exp(0.5), chance mode, ε=0.6. Showing Liar's Dice (13 levels, most informative).

| δ   | tgt_rate | L0–L3 | L4–L8 | L9–L12 |
|-----|:--------:|:-----:|:-----:|:------:|
| 0.0 | 0%       | —     | —      | —     |
| 0.2 | 19%      | 37%   | 30%    | 20%   |
| 0.5 | 50%      | 35%   | 28%    | 19%   |
| 0.9 | 89%      | 31%   | 31%    | 24%   |
| 1.0 | 100%     | 33%   | 31%    | 23%   |

**Finding 5**: δ controls targeting frequency linearly (δ ≈ targeting rate), as expected. The z* distribution across IIG levels is roughly invariant to δ — δ determines *how often* targeting fires, not *where* within the IIG the z* lands. Confirmed across all games. This is a secondary finding; δ, mode, and ε interact with the *walk*, not with z* selection, and are not affected by the per-terminal vs level-allocation distinction that is the primary finding of this archive.

### Sweep 3 — Bias Mode (chance vs full)

Fixed: exp(0.5), δ=0.9, ε=0.6. Showing Liar's Dice.

| Mode    | L0–L3 | L4–L8 | L9–L12 |
|---------|:-----:|:-----:|:------:|
| chance  | 31%   | 31%   | 24%    |
| full    | 33%   | 30%   | 21%    |

**Finding 6**: Mode has minimal effect on the z* distribution. Full mode shows a slight shift toward lower levels (+2 pp in L0–L3) across all games. The effect is modest — mode primarily affects which decision nodes are constrained during the walk, not the z* selection.

### Sweep 4 — ε (Exploration)

Fixed: exp(0.5), δ=0.9, chance mode. Showing Liar's Dice.

| ε   | L0–L3 | L4–L8 | L9–L12 |
|-----|:-----:|:-----:|:------:|
| 0.0 | 32%   | 33%   | 20%    |
| 0.2 | 29%   | 30%   | 25%    |
| 0.6 | 31%   | 31%   | 24%    |
| 1.0 | 32%   | 34%   | 19%    |

**Finding 7**: Epsilon has no meaningful effect on z* distribution. All values produce level distributions within ±3 pp. Expected: ε controls OOS walk exploration, not z* selection.

## Structural Analysis — Terminal Count Per Level

The histogram data above shows sampling fractions per level, but the raw terminal counts per level reveal *why* the distributions look the way they do.

### Terminal Distribution Is Exponentially Skewed

**Liar's Dice** (depth 12, 147,420 terminals, 6 infosets per level):

| Level | Infosets | Terminals | % of Game |
|:-----:|:--------:|---------:|----------:|
| 0     | 1        | 6        | 0.004%    |
| 1     | 6        | 66       | 0.04%     |
| 2     | 6        | 72       | 0.05%     |
| 3     | 6        | 144      | 0.1%      |
| 4     | 6        | 288      | 0.2%      |
| 5     | 6        | 576      | 0.4%      |
| 6     | 6        | 1,152    | 0.8%      |
| 7     | 6        | 2,304    | 1.6%      |
| 8     | 6        | 4,608    | 3.1%      |
| 9     | 6        | 9,216    | 6.3%      |
| 10    | 6        | 18,432   | 12.5%     |
| 11    | 6        | 36,864   | 25.0%     |
| 12    | 6        | 73,692   | 50.0%     |

Terminals **double with each level**. L0 has 6 terminals; L12 has 73,692 — a 12,000× ratio. The infoset count per level is perfectly uniform (6), but the terminal count grows exponentially because each upstream infoset connects to an exponentially larger portion of the game tree.

**Leduc Poker** (depth 7, 5,520 terminals):

| Level | Infosets | Terminals | % of Game |
|:-----:|:--------:|---------:|----------:|
| 0     | 1        | 8        | 0.1%      |
| 1     | 4        | 56       | 1.0%      |
| 2     | 5        | 36       | 0.7%      |
| 3     | 5        | 80       | 1.4%      |
| 4     | 5        | 745      | 13.5%     |
| 5     | 6        | 1,295    | 23.5%     |
| 6     | 6        | 1,080    | 19.6%     |
| 7     | 6        | 2,220    | 40.2%     |

Same pattern: L0 has 8 terminals, L7 has 2,220 (278× ratio).

**Goofspiel** (depth 5, 576 terminals):

| Level | Infosets | Terminals | % of Game |
|:-----:|:--------:|---------:|----------:|
| 0     | 1        | 4        | 0.7%      |
| 2     | 1        | 32       | 5.6%      |
| 4     | 1        | 540      | 93.8%     |

Odd levels are empty (alternating player structure). L4 holds 93.8% of all terminals.

**Kuhn Poker** (depth 2, 30 terminals):

| Level | Infosets | Terminals | % of Game |
|:-----:|:--------:|---------:|----------:|
| 0     | 1        | 4        | 13.3%     |
| 1     | 2        | 8        | 26.7%     |
| 2     | 3        | 18       | 60.0%     |

### Implications

This exponential skew explains the key findings:

1. **Why constant is "anti-targeted"**: Constant assigns equal weight to all infosets, but distant levels have exponentially more terminals. The uniform-per-infoset weight translates to a terminal distribution dominated by the farthest levels.

2. **Why step(0.01) breaks at depth**: Step gives weight 1.0 to L0 and 0.01 to all other levels. At L0, 1 infoset × 6 terminals × weight 1.0 = effective mass 6. At L12, 6 infosets × 73,692 terminals × weight 0.01 = effective mass ~4,422. The floor wins by ~700×. The floor weight would need to be ~$10^{-5}$ to overcome this.

3. **Why exp(0.5) works**: $0.5^{12} = 0.000244$. The weight at L12 is 4,096× smaller than L0. This approximately cancels the 12,000× terminal count advantage, producing roughly balanced sampling across levels.

4. **Why exp(0.1) over-concentrates**: $0.1^{12} = 10^{-12}$. The weight at L12 is $10^{12}$× smaller than L0 — far more than needed to overcome the terminal count ratio. This starves distant levels completely.

5. **Per-terminal concentration at L0**: Even when exp(0.5) puts only 2% of samples at L0, that's 2% spread across 6 terminals vs 7% across 73,692 terminals at L12. Each L0 terminal gets sampled ~1,000× more often per terminal than a L12 terminal. The per-level percentages hide enormous per-terminal concentration at L0.

## The Core Discovery: Per-Terminal Weighting vs Level Allocation

This is the most important finding of archive 012, and it reframes the interpretation of all prior decay function results.

### How z* Selection Worked (Pre-Refactor)

The ISGT algorithm selects a target terminal history z* from an IIG-weighted distribution. The intended semantics were:

> "Terminals passing through nearby IIG infosets should be sampled more frequently."

The implementation computed z*'s probability as follows:

1. **Assign each infoset a weight** via the decay function: $w(\text{iid}) = f(d(\text{iid}))$ where $d$ is the upstream BFS distance from the active infoset. E.g., $\text{exp}(0.5)$ assigns $w = 0.5^d$.

2. **Assign each terminal a weight**: $w(z) = \max_{\text{iid} \in \text{path}(z)} w(\text{iid})$ — the maximum IIG weight among infosets on $z$'s path.

3. **Normalize over all terminals**: $P(z) = w(z) / \sum_{z'} w(z')$.

This is a **flat distribution over terminals**. The normalization denominator sums over every terminal in the game — 147,420 terms for Liar's Dice. The decay function's per-infoset weights are translated into per-terminal weights, then compete in a single normalizing sum.

### Why This Doesn't Do What It Looks Like

The mental model when designing the decay functions was level-based:

> "exp(0.5) gives L0 weight 1.0, L1 weight 0.5, L2 weight 0.25, ... so L0 should get the most samples."

But this reasoning conflates *infoset-level weighting* with *terminal-level allocation*. The crucial variable it ignores is **how many terminals are at each level**.

In Liar's Dice (from a deep active infoset):

| Level | Infosets | Terminals | Per-infoset weight (exp 0.5) | Total mass (terminals × weight) |
|:-----:|:--------:|----------:|:----------------------------:|--------------------------------:|
| 0     | 1        | 6         | 1.0                          | 6.0                            |
| 1     | 6        | 66        | 0.5                          | 33.0                           |
| 2     | 6        | 72        | 0.25                         | 18.0                           |
| ...   |          |           |                              |                                 |
| 10    | 6        | 18,432    | 0.001                        | 18.4                           |
| 11    | 6        | 36,864    | 0.0005                       | 18.4                           |
| 12    | 6        | 73,692    | 0.00024                      | 17.9                           |

After normalization, each level's share of z* = (total mass at level) / (sum of all masses). The terminal count **doubles at each level** while exp(0.5) halves the weight — these roughly cancel, producing approximately equal allocation per level. This is what the histogram data confirmed: exp(0.5) spreads 2–11% across all 13 levels.

But this approximate balance at exp(0.5) is a **coincidence of matching growth rates**, not a designed property. A different game with 3× terminal growth per level would make exp(0.5) anti-targeted (distant levels dominate). A game with 1.2× growth would make it over-concentrate at L0.

For ConstantWeight ($w = 1.0$ everywhere), the mass at each level is simply the terminal count — and since terminals grow exponentially, the distribution is dominated by the farthest level:

| ConstantWeight on Liar's Dice | Mass | Share |
|------|------:|------:|
| L0   | 6     | 0.004% |
| L12  | 73,692 | 50.0% |

**ConstantWeight is not "uniform across levels" — it is anti-targeted, sending 53% of z* to the farthest level from the active decision.** This is the opposite of what proximity-based targeting is supposed to achieve.

### What the Histograms Revealed

The decay comparison histograms show this mechanism playing out across all four games:

**ConstantWeight** produces a sampling distribution that mirrors the terminal count distribution — heavily bottom-loaded. On Liar's Dice: 53% at L12, 0% at L0. On Leduc: 42% at L7, 0% at L0. On Goofspiel: 94% at L4 (farthest active level), 0% at L0.

**exp(0.5)** happens to approximately cancel the terminal-count growth on Liar's Dice ($0.5^{12} \approx 0.0002$ vs 12,000× growth ratio), producing the most balanced distribution. But on Goofspiel (where L4 has 135× more terminals than L0), exp(0.5) still puts 70% at L4 — the cancellation fails because the growth rate is different.

**exp(0.1)** over-corrects massively. $0.1^{12} = 10^{-12}$ overwhelms any terminal-count growth, sending 80–92% of z* to L0 across all games. Archive 008 showed this hurts performance — it starves distant infosets of necessary updates.

**step(0.01)** reveals the mechanism most clearly. On Kuhn (depth 2, mild terminal growth): 95% at L0 — the step function does what it's supposed to. On Liar's Dice (depth 12): step is **indistinguishable from constant** (50% at L12). The floor weight of 0.01, applied to 73,692 terminals at L12, produces mass of ~737 — dwarfing the weight-1.0 × 6 terminals at L0. The floor weight would need to be ~$10^{-5}$ to even compete.

### What This Means for the Decay Functions

The decay functions as originally parameterized **do not have game-independent semantics**. What $\text{exp}(\alpha)$ actually means depends on the game's terminal-count growth rate:

| Game's growth rate | exp(0.5) behavior |
|-------------------|-------------------|
| 2× per level (Liar's Dice) | ≈ balanced across levels (accidental) |
| 3× per level (hypothetical) | anti-targeted, distant-dominated |
| 1.5× per level (hypothetical) | over-concentrated at L0 |

This is why ConstantWeight dominated the ablation results: it's the only function with a **stable, game-independent meaning** (uniform over terminals). The shaped decays were trying to express level preferences through a mechanism that translates those preferences differently depending on the game's structure. The result was unpredictable, game-dependent behavior — sometimes helping (exp(0.5) on Liar's Dice, +1.3%), sometimes neutral (Kuhn, Leduc), sometimes harmful (exp(0.1) everywhere).

### Resolution: Two-Stage Level-First Sampling

This analysis directly motivated a refactoring of `_get_target_dist` to **two-stage level-first sampling**:

1. **Stage 1 — Pick a level** from the decay-weighted distribution: $P(\ell) = w(\ell) / \sum_{\ell'} w(\ell')$
2. **Stage 2 — Pick a terminal** uniformly within that level: $P(z | \ell) = 1 / |T_\ell|$

Combined: $P(z) = \frac{w(\text{level}(z))}{\sum_{\ell} w(\ell)} \times \frac{1}{|T_{\text{level}(z)}|}$

Each terminal is assigned to its **closest upstream level** (minimum BFS distance among its on-path infosets). Terminals with no upstream infosets go to a floor group.

This decouples the decay function from the terminal-count imbalance. Under two-stage sampling:
- **ConstantWeight** gives 50% to L0 and 50% to floor (or equal shares across all levels) — **level-balanced**, not anti-targeted.
- **exp(0.5)** gives 66.7% to L0, 33.3% to floor on a 2-level game — the decay function directly controls level allocation regardless of how many terminals each level contains.
- **exp(0.5) means the same thing on every game**: "halve the level probability with each BFS distance." The per-level terminal count no longer distorts the signal.

The two-stage refactor was implemented in `src/isgt.py` (`_get_target_dist`) and passes all 22 unit tests and all 11 validation checks from archive 011.

**Variance tradeoff**: Two-stage sampling concentrates more probability mass on fewer terminals (L0 has 6 terminals on Liar's Dice), which increases per-terminal importance weight variance at distant levels. This is an explicit, tunable cost — the decay parameter α directly controls how much variance is traded for proximity concentration. Whether this tradeoff is net positive is an empirical question for the next ablation (archive 013).

## Plots

16 PNG files produced in `results/plots/sampling_012/`:

| Game | Decay | Delta | Mode | Epsilon |
|------|-------|-------|------|---------|
| Kuhn     | `kuhn_poker_decay.png`    | `kuhn_poker_delta.png`    | `kuhn_poker_mode.png`    | `kuhn_poker_epsilon.png`    |
| Leduc    | `leduc_poker_decay.png`   | `leduc_poker_delta.png`   | `leduc_poker_mode.png`   | `leduc_poker_epsilon.png`   |
| Goofspiel| `goofspiel_decay.png`     | `goofspiel_delta.png`     | `goofspiel_mode.png`     | `goofspiel_epsilon.png`     |
| Liar's Dice | `liars_dice_decay.png` | `liars_dice_delta.png`    | `liars_dice_mode.png`    | `liars_dice_epsilon.png`    |

## Files

- `experiments/sampling_analysis_012.py` — Analysis harness
- `experiments/plot_sampling_012.py` — Plot generation
- `results/sampling_012/*.json` — Raw data (4 files)
- `results/plots/sampling_012/*.png` — Plots (16 files)
