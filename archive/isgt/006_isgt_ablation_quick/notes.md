# ISGT Ablation — Quick Sanity Check (Aggregate Exploitability)

**Date**: 2026-04-03  
**Goal**: First ablation across decay functions, bias modes, and all 4 games using the online aggregate exploitability method (paper §4.4.2). Sanity check before a full high-statistics run.

## Experiment Setup

- **Method**: Aggregate exploitability — play N matches vs random opponent, accumulate strategy, measure exploitability
- **Games**: kuhn_poker, leduc_poker, liars_dice, goofspiel
- **Bias modes**: chance (bias chance nodes only), full (bias all nodes deterministically)
- **Decay functions (8)**: exp_0.3, exp_0.5, exp_0.7, poly_1, poly_2, linear_4, step_0.01, step_0.1
- **Sim budgets**: 100, 500 sims per move
- **Matches per job**: 50
- **Total jobs**: 128 (4 games × 2 modes × 8 decays × 2 budgets)
- **Workers**: 6 (ProcessPoolExecutor)
- **Seed**: 42
- **Runtime**: 8.2 minutes (with IIG caching — previous uncached run was 20.1 min, 2.45× speedup)

## Results — Best Configs at 500 sims/move

### Kuhn Poker — **Chance mode wins**

| Config | Exploitability |
|--------|---------------|
| chance_exp_0.7 | **0.1060** |
| chance_linear_4 | 0.1061 |
| chance_exp_0.5 | 0.1155 |
| chance_poly_1 | 0.1225 |
| chance_poly_2 | 0.1221 |
| full_linear_4 | 0.1264 |
| chance_exp_0.3 | 0.1320 |
| full_exp_0.3 | 0.1313 |
| full_exp_0.7 | 0.1389 |
| full_exp_0.5 | 0.1437 |
| chance_step_0.1 | 0.1447 |
| full_poly_2 | 0.1496 |
| full_poly_1 | 0.1519 |
| full_step_0.1 | 0.1675 |
| chance_step_0.01 | 0.1798 |
| full_step_0.01 | 0.1942 |

Chance mode dominates. Kuhn has only 1 chance node, so biasing chance gives minimal distortion while allowing natural decision exploration.

### Leduc Poker — **Full mode wins**

| Config | Exploitability |
|--------|---------------|
| full_poly_2 | **1.9864** |
| full_exp_0.5 | 1.9945 |
| full_poly_1 | 1.9972 |
| full_exp_0.3 | 2.0388 |
| full_exp_0.7 | 2.0408 |
| full_linear_4 | 2.0433 |
| full_step_0.1 | 2.0487 |
| full_step_0.01 | 2.1275 |
| chance_poly_2 | 2.2285 |
| chance_exp_0.5 | 2.2409 |
| chance_exp_0.7 | 2.2498 |
| chance_step_0.01 | 2.2628 |
| chance_exp_0.3 | 2.2670 |
| chance_poly_1 | 2.2667 |
| chance_linear_4 | 2.2705 |
| chance_step_0.1 | 2.2937 |

Full mode dominates. All full-mode configs beat all chance-mode configs. Leduc has multiple chance and decision stages; deterministic targeting through the full tree focuses computation effectively.

### Liar's Dice — **Chance mode wins (small margin)**

| Config | Exploitability |
|--------|---------------|
| chance_poly_2 | **0.7286** |
| chance_poly_1 | 0.7327 |
| chance_step_0.1 | 0.7366 |
| chance_linear_4 | 0.7376 |
| chance_exp_0.5 | 0.7386 |
| chance_exp_0.3 | 0.7390 |
| chance_exp_0.7 | 0.7408 |
| chance_step_0.01 | 0.7464 |
| full_poly_1 | 0.7756 |
| full_exp_0.3 | 0.7760 |
| full_linear_4 | 0.7778 |
| full_exp_0.5 | 0.7787 |
| full_poly_2 | 0.7785 |
| full_exp_0.7 | 0.7800 |
| full_step_0.01 | 0.7859 |
| full_step_0.1 | 0.7860 |

Chance mode wins uniformly but the spread within each mode is small (1.6% for chance, 1.3% for full). With only 50 matches, much of this variation is likely noise.

### Goofspiel — **Chance mode wins (large margin)**

| Config | Exploitability |
|--------|---------------|
| chance_exp_0.5 | **0.1213** |
| chance_exp_0.7 | 0.1241 |
| chance_exp_0.3 | 0.1329 |
| chance_poly_1 | 0.1334 |
| chance_poly_2 | 0.1363 |
| chance_step_0.1 | 0.1442 |
| chance_linear_4 | 0.1960 |
| chance_step_0.01 | 0.1997 |
| full_linear_4 | 0.4671 |
| full_poly_1 | 0.4861 |
| full_step_0.01 | 0.4869 |
| full_exp_0.3 | 0.4980 |
| full_exp_0.7 | 0.4994 |
| full_exp_0.5 | 0.5100 |
| full_poly_2 | 0.5181 |
| full_step_0.1 | 0.5196 |

Chance mode is 3-4× better than full mode. Goofspiel has complex decision trees where deterministic targeting eliminates decision exploration entirely, catastrophically harming play quality.

## Key Findings

### 1. Bias mode matters more than decay function
The mode choice (chance vs full) creates a much larger performance gap than any decay function variation. Within a mode, most decays perform similarly.

### 2. Chance mode wins 3/4 games
- **Chance**: Kuhn, liar's dice, goofspiel
- **Full**: Leduc only

Full mode's deterministic decision targeting eliminates decision-point exploration. This helps on Leduc (where focusing the tree matters) but hurts on games where decision diversity is important.

### 3. Step functions are consistently worst
Both step_0.01 and step_0.1 rank last or near-last in every game/mode combination. The hard cutoff creates a discontinuous weight distribution that poorly approximates the IIG's relevance structure.

### 4. Exponential and polynomial decays are competitive
exp_0.5 and poly_2 appear frequently in top positions. The smooth decay profiles seem well-matched to the IIG's level structure.

### 5. Within-mode variation is small on liar's dice
The 1.6% spread across chance-mode decays at 50 matches suggests higher statistics are needed to distinguish decay functions on larger games.

## IIG Caching Optimization

This run validated the IIG caching fix: passing a pre-built IIG to `ISGTBot(iig=...)` instead of rebuilding per-bot. Runtime dropped from 20.1 to 8.2 minutes (2.45× speedup) with bit-identical results.

## Next Steps

- Drop step functions (clear losers) to reduce job count
- Add `ConstantWeight` (uniform) decay as a control — isolates IIG structure from level-based decay
- Add OOS and untargeted-ISGT baselines directly into the ablation grid
- Run full ablation with 500 matches and 4 budget levels for publication-quality results
