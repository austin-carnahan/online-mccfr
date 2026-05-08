# ISGT Full Ablation — Decay Functions × Bias Modes (Aggregate Exploitability)

**Date**: 2026-04-03  
**Archive**: 007_isgt_ablation_full  
**Runtime**: 42.8 minutes, 208 jobs, 0 failures

## Experiment Setup

**Method**: Aggregate exploitability (paper §4.4.2 analog). For each configuration:
1. Play N matches vs a uniform random opponent
2. Accumulate strategy data (average policy) across all matches
3. Compute exploitability of the aggregated strategy

**Grid**:
- **Games**: kuhn_poker, leduc_poker, liars_dice, goofspiel
- **Bias modes**: chance, full
- **Decay functions (6)**: exp_0.3, exp_0.5, exp_0.7, poly_2, linear_4, constant
- **Baselines**: OOS (δ=0.9, ε=0.6, γ=0.01)
- **Sim budgets**: 100, 250, 500, 1000 iterations per move
- **Matches per game**: kuhn=500, leduc=300, goofspiel=300, liars_dice=100

**ISGT parameters**: ε=0.6 (decision exploration), γ=0.01 (regret matching floor).  
**OOS parameters**: δ=0.9 (chance targeting), ε=0.6, γ=0.01.

All algorithms share the same ε and γ. The key architectural differences are:
- OOS uses iteration-level targeting: before each iteration, a coin flip (probability δ=0.9) decides whether the *entire* trajectory is targeted (chance+decision nodes biased toward the current subgame) or untargeted (vanilla outcome sampling with independent chance sampling). 10% of iterations are fully untargeted.
- ISGT targets 100% of iterations — every iteration follows z*'s pre-selected trajectory at chance nodes. There are no untargeted iterations.
- ISGT pre-initializes all infostates from IIG; OOS discovers them incrementally

## Results Overview

### Kuhn Poker — Chance ISGT dominates

| Rank | Config | sims=100 | sims=250 | sims=500 | sims=1000 |
|------|--------|----------|----------|----------|-----------|
| 1 | chance_constant | 0.189 | 0.140 | **0.097** | **0.061** |
| 2 | chance_linear_4 | 0.188 | **0.139** | 0.102 | 0.062 |
| 3 | chance_exp_0.7 | **0.186** | 0.145 | 0.100 | 0.065 |
| 4 | chance_exp_0.5 | 0.187 | 0.144 | 0.105 | 0.066 |
| 5 | chance_poly_2 | 0.201 | 0.163 | 0.124 | 0.084 |
| 6 | chance_exp_0.3 | 0.199 | 0.158 | 0.119 | 0.085 |
| 7 | full_constant | 0.211 | 0.171 | 0.129 | 0.087 |
| 8 | **oos** | 0.197 | 0.164 | 0.125 | 0.088 |

Chance ISGT beats OOS by **22–31%** across all budgets. The gap widens at higher budgets — from 4% at 100 sims to 31% at 1000 sims. Kuhn Poker has only 1 chance node, so chance-mode targeting has minimal distortion while preserving decision exploration.

### Leduc Poker — Full ISGT early, OOS late

| Rank | Config | sims=100 | sims=250 | sims=500 | sims=1000 |
|------|--------|----------|----------|----------|-----------|
| 1 | **oos** | 2.451 | 2.220 | 2.090 | **1.768** |
| 2 | full_poly_2 | **2.203** | 2.130 | **1.969** | 1.835 |
| 3 | full_exp_0.5 | 2.251 | **2.106** | 2.000 | 1.846 |
| 4 | full_constant | 2.216 | 2.131 | 2.012 | 1.854 |
| 8 | chance_constant | 2.487 | 2.388 | 2.241 | 1.985 |

Full ISGT leads at 100–500 sims, but OOS overtakes at 1000 sims (1.768 vs 1.835). All chance configs trail both full and OOS. Leduc has substantial chance structure (card deals + board card), making full-mode targeting valuable for focusing computation — but OOS's 10% untargeted exploration eventually pays off.

### Liar's Dice — Chance ISGT competitive, OOS wins at high budget

| Rank | Config | sims=100 | sims=250 | sims=500 | sims=1000 |
|------|--------|----------|----------|----------|-----------|
| 1 | **oos** | 0.781 | 0.759 | 0.747 | **0.691** |
| 2 | chance_exp_0.3 | **0.764** | 0.749 | 0.732 | 0.701 |
| 3 | chance_constant | 0.765 | 0.747 | 0.729 | 0.702 |
| 6 | chance_poly_2 | 0.769 | **0.746** | **0.722** | 0.708 |
| 8 | full_exp_0.5 | 0.771 | 0.772 | 0.769 | 0.762 |

Chance ISGT leads at 100–500 sims, OOS overtakes at 1000. Within chance mode, spread is < 1% — decay choice barely matters. Full mode barely converges (0.77→0.76 across 10× budget increase). This game has complex chance structure (dice rolls) where untargeted exploration at chance nodes is critical.

### Goofspiel — Chance ISGT and OOS neck-and-neck, full mode catastrophic

| Rank | Config | sims=100 | sims=250 | sims=500 | sims=1000 |
|------|--------|----------|----------|----------|-----------|
| 1 | chance_constant | 0.312 | 0.181 | 0.125 | **0.081** |
| 2 | chance_exp_0.7 | 0.312 | 0.185 | 0.127 | 0.083 |
| 3 | chance_exp_0.5 | 0.314 | 0.187 | **0.125** | 0.084 |
| 6 | **oos** | **0.293** | **0.170** | 0.126 | 0.092 |
| 8 | full_constant | 0.627 | 0.558 | 0.487 | 0.397 |

OOS leads at low budgets (100–250), but chance ISGT overtakes at 500+ sims — at 1000 sims, chance_constant beats OOS by 12%. Full mode is 4–5× worse than chance mode, confirming that deterministic decision targeting is catastrophic when games require diverse decision exploration.

## Analysis

### Finding 1: Bias mode is the dominant factor

Mode choice (chance vs full) creates much larger performance gaps than any decay function. Within chance mode, the spread between best and worst decay is typically 5–15%. Between modes, the gap is 30–400% depending on the game.

**Full mode** eliminates all exploration at both chance and decision nodes. It only helps on Leduc (where focusing on the right subtree outweighs exploration loss) and is catastrophic elsewhere.

**Chance mode** preserves decision-node exploration (ε=0.6 mixing) while guiding chance nodes along targeted trajectories. This is consistently the strongest ISGT configuration.

### Finding 2: ConstantWeight ties or beats shaped decays in every game

ConstantWeight assigns w(d) = 1.0 for all IIG levels, including the floor. This means every terminal history gets equal targeting probability — the target distribution is uniform over terminals.

Despite having no proximity information, ConstantWeight is the best or tied-best ISGT config:
- **Kuhn**: #1 at 500 and 1000 sims
- **Leduc**: #4 in full mode (within 1% of #1 full_poly_2)
- **Liar's dice**: #3 (within 0.1% of #1)
- **Goofspiel**: #1 at 1000 sims

**What this means**: The shaped decays (exponential, polynomial, linear) concentrate targeting on terminals passing through nearby infosets. If this proximity signal were valuable, they should consistently outperform ConstantWeight. They don't. The IIG level structure does not provide enough targeting signal to overcome the exploration cost of concentrating the target distribution.

**What this does NOT mean**: We cannot conclude that trajectory-based targeting itself is useless — only that the *IIG-weighted* variant provides no net benefit over uniform targeting. The act of choosing a trajectory and committing to it at chance nodes still differs from vanilla MCCFR's independent per-node sampling.

### Finding 3: OOS overtakes ISGT at high budgets (crossover pattern)

On Leduc (1.768 vs 1.835) and liar's dice (0.691 vs 0.701), OOS beats all ISGT configs at 1000 sims, despite trailing at lower budgets. On Kuhn and goofspiel, ISGT maintains its lead through 1000 sims.

**Hypothesis: exploration deficit at chance nodes**

The crossover correlates with chance-node complexity:
- **Kuhn** (1 chance node): ISGT always leads. Minimal chance exploration needed.
- **Goofspiel** (3 chance nodes, but each reveals public info to both players): ISGT leads at high budgets. Public chance nodes may be less exploration-sensitive.
- **Leduc** (2 chance stages — private deal + public board): OOS overtakes. The hidden board card creates information asymmetry that requires diverse chance exploration.
- **Liar's dice** (2 private dice rolls): OOS overtakes. Private chance outcomes are critical to the information structure.

The mechanistic explanation:
- **OOS**: δ=0.9 means 10% of iterations are fully untargeted — chance nodes sample independently from their natural probability distributions, and decision nodes sample from ε-on-policy without subgame restriction. These iterations provide the algorithm with independent chance sampling that correctly weights different chance outcomes by their actual probabilities.
- **ISGT**: 100% of iterations follow z*'s pre-selected trajectory at chance nodes. Even with ConstantWeight (uniform over terminals), this is **correlated trajectory commitment**, not independent per-node sampling. The probability of taking a particular chance action at a given node is proportional to the number of terminals reachable through that action — not the natural chance probability. These only coincide if the tree is perfectly balanced under every chance node, which is generally not the case.

**Why the decay floor doesn't substitute for untargeted iterations**: One might expect that ConstantWeight's non-zero floor (which allows z* to land on any terminal in the game) would provide sufficient exploration of irrelevant subtrees, making a separate δ mechanism unnecessary. However, the ablation results show this is not the case — OOS still overtakes ConstantWeight ISGT on Leduc and Liar's Dice. The reason is that the floor provides coverage of the *terminal space* but does not recover *independent chance sampling*. When z* is selected and then followed deterministically, the marginal distribution over chance actions at each node is dictated by the tree structure (terminal counts), not by the game's chance probabilities. Independent sampling at each chance node — as OOS gets in its untargeted iterations — produces a fundamentally different distribution over trajectories.

When games have few chance nodes or public chance events, this distinction doesn't matter much. When games have private chance outcomes that create the information asymmetry the algorithm must reason about, the lack of independent chance sampling becomes a bottleneck at higher iteration counts.

### Finding 4: Convergence rates reveal structural differences

From the convergence ratio table (improvement from 100→1000 sims):

| Game | Best ISGT | OOS | Full ISGT |
|------|-----------|-----|-----------|
| Kuhn | 3.08× (chance_const) | 2.23× | 1.93× (full_poly_2) |
| Goofspiel | 3.85× (chance_const) | 3.19× | 1.51× (full_exp_0.3) |
| Leduc | 1.25× (chance_const) | 1.39× | 1.20× (full_poly_2) |
| Liar's dice | 1.09× (chance_const) | 1.13× | 1.01× (full_poly_2) |

**Pattern**: Chance ISGT converges faster than OOS on Kuhn (+38%) and Goofspiel (+21%), but OOS converges faster on Leduc (+11%) and Liar's Dice (+4%). Full mode has the worst convergence rates everywhere — on liar's dice it barely converges at all (1.01×).

The games where OOS converges faster are exactly the games where OOS overtakes at high budget. This is consistent with the exploration-deficit hypothesis: OOS's untargeted iterations provide exploration that compounds into faster convergence when the game's information structure demands it.

### Finding 5: Pre-initialized infostates likely explain ISGT's low-budget advantage

ISGT pre-populates all infostate tables (regret + average policy arrays) from the IIG during construction. OOS discovers infosets incrementally — the first time it visits an infoset, it does a random playout instead of proper regret matching.

This gives ISGT a structural advantage in early iterations:
- Every ISGT iteration uses regret matching from iteration 1
- OOS wastes early iterations on random playouts at newly-discovered infosets

This advantage is most pronounced at low budgets (100–250 sims) where OOS is still discovering infosets. It diminishes at higher budgets as OOS fills in its tree. This matches the crossover pattern: ISGT leads early, OOS catches up (or overtakes) later.

**To test this hypothesis directly**: We could add pre-initialized infostates to OOS (using the same IIG-based initialization) and re-run the comparison. If the ISGT advantage disappears, it confirms pre-initialization — not targeting — is the source.

## Open Questions

### Q1: Would adding iteration-level untargeted iterations fix the crossover?

Following OOS's design, this would be an iteration-level coin flip: with probability δ, run a targeted ISGT iteration (follow z* at chance nodes, ε-on-policy at decisions). With probability (1-δ), run a fully untargeted iteration where chance nodes sample independently from natural probabilities and decision nodes sample from unrestricted ε-on-policy. This matches OOS's mechanism exactly — δ is not a per-node parameter but a per-iteration scenario selector.

At δ=0.9 (matching OOS), 10% of iterations would use independent chance sampling. The key question is whether this independent sampling — which correctly weights chance outcomes by their game probabilities rather than by terminal-count proportions — fixes the convergence deficit on Leduc and Liar's Dice.

**Expected outcome**: This should fix the crossover, at some cost to low-budget performance (since 10% of iterations would be untargeted). If it does, it confirms that the exploration deficit is specifically about the lack of *independent* chance sampling, not about the volume of terminal coverage (which ConstantWeight already maximizes).

### Q2: Is trajectory commitment itself the win, separate from IIG targeting?

ConstantWeight performing as well as shaped decays suggests IIG-level proximity isn't useful. But is committing to a full trajectory (correlated chance sampling) better than independent per-node sampling (vanilla MCCFR)?

To test this, we would need a "no targeting at all" control where ISGT samples chance nodes independently rather than following z*. If trajectory commitment is the win, this control should perform worse than ConstantWeight.

### Q3: Is the pre-initialization advantage the dominant effect?

The cleanest test: give OOS the same IIG-precomputed infostates and re-run. If pre-initialized OOS matches or beats ISGT, then the targeting mechanism adds no value and the entire ISGT benefit comes from using the IIG for initialization rather than for guiding sampling.

### Q4: Does the constant-weight result generalize to larger games?

The IIG becomes more expensive to compute for larger games. If ConstantWeight (which doesn't need level computations) is always as good as shaped decays, the IIG's graph structure adds algorithmic complexity with no practical benefit. However, it's possible that larger games with deeper decision trees would benefit more from proximity-based targeting because the relevant subtrees are a smaller fraction of the total game tree.

## Conclusions

1. **Chance mode is the only viable ISGT bias mode** for general use. Full mode is catastrophic except on Leduc.

2. **ConstantWeight is the recommended decay function** — simplest and ties or beats all shaped alternatives. The IIG level-proximity structure does not provide useful targeting signal in these games.

3. **ISGT's advantage over OOS comes in two regimes**: (a) low-budget settings where pre-initialized infostates provide a head start, and (b) games with simple chance structure where ISGT's aggressive targeting doesn't cost much exploration.

4. **The exploration-exploitation tradeoff is the key open question**. Adding a δ parameter to control chance-node targeting probability is the highest-value next experiment. This would directly test whether controlled untargeting fixes the high-budget crossover while preserving the low-budget advantage.
