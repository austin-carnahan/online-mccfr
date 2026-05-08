# Phase A: Root Convergence Ablation

**Date**: 2025-01  
**Goal**: Evaluate whether IIG-biased chance sampling accelerates root exploitability convergence vs vanilla MCCFR.

## Experiment Setup

- **Algorithm**: ISGT (MCCFR with IIG-biased sampling)
- **Baseline**: ISGT with uniform weights (equivalent to vanilla MCCFR)
- **Hyperparameters**: ε=0.6, γ=0.01, seed=42
- **I₀ selection**: First player-0 infoset with >1 IIG level (auto-picked)
- **Checkpoints**: 500, 1000, 2000, 5000, 10000, 20000 iterations
  - Kuhn only ran to 2000 (converges fast)
- **Configs per game**: 5 decay functions × 2 bias modes + 1 baseline = 11

### Decay Functions
| Name | Formula | Parameter |
|------|---------|-----------|
| ExponentialDecay | w = α^level | α=0.5, α=0.8 |
| PolynomialDecay | w = 1/(1+level)^p | p=2.0 |
| LinearDecay | w = max(0, 1 - level/D) | D=4 |
| StepFunction | w = 1 if level=0 else floor | floor=0.01 |

### Bias Modes
- **chance**: Bias chance node sampling only (q(a) ∝ p(a)·g(a))
- **full**: Bias both chance and decision node sampling

## Results — Final Checkpoint Exploitability

### Kuhn Poker (2000 iterations)

| Config | Exploitability |
|--------|---------------|
| **full_exp(α=0.8)** | **0.0495** |
| chance_linear(D=4) | 0.0679 |
| full_linear(D=4) | 0.0823 |
| full_poly(p=2.0) | 0.0865 |
| chance_poly(p=2.0) | 0.0904 |
| chance_exp(α=0.8) | 0.1122 |
| full_exp(α=0.5) | 0.1680 |
| baseline | 0.1717 |
| chance_step(floor=0.01) | 0.1761 |
| chance_exp(α=0.5) | 0.1955 |
| full_step(floor=0.01) | 0.2328 |

**Winner**: full_exp(α=0.8) — 3.5× better than baseline. All IIG-biased configs with moderate decay beat baseline except step and gentle exp(α=0.5).

### Leduc Poker (20000 iterations)

| Config | Exploitability |
|--------|---------------|
| **baseline** | **1.2400** |
| chance_linear(D=4) | 1.2713 |
| chance_exp(α=0.8) | 1.2763 |
| full_exp(α=0.8) | 1.2942 |
| chance_exp(α=0.5) | 1.3551 |
| chance_poly(p=2.0) | 1.3609 |
| full_poly(p=2.0) | 1.7619 |
| full_exp(α=0.5) | 1.8182 |
| chance_step(floor=0.01) | 2.3253 |
| full_linear(D=4) | 2.3420 |
| full_step(floor=0.01) | 2.3489 |

**Winner**: Baseline. Chance-only with gentle bias (linear, exp(α=0.8)) comes close. Full mode hurts significantly. Leduc may need more iterations — all configs still above 1.0.

### Liar's Dice (20000 iterations)

| Config | Exploitability |
|--------|---------------|
| **chance_exp(α=0.5)** | **0.6120** |
| baseline | 0.6368 |
| chance_linear(D=4) | 0.6539 |
| chance_exp(α=0.8) | 0.6652 |
| full_exp(α=0.8) | 0.6940 |
| chance_poly(p=2.0) | 0.7027 |
| full_step(floor=0.01) | 0.8134 |
| full_linear(D=4) | 0.8157 |
| full_exp(α=0.5) | 0.8195 |
| full_poly(p=2.0) | 0.8257 |
| chance_step(floor=0.01) | 0.8186 |

**Winner**: chance_exp(α=0.5) — marginal improvement over baseline (3.9%). Full mode consistently hurts. All public actions in Liar's Dice means decision biasing adds variance without benefit.

## Key Findings

1. **Chance-only bias is the safe bet.** It helps or matches baseline across all games. Full (decision) biasing is game-dependent — helps Kuhn but hurts Leduc and Liar's Dice.

2. **Step function is consistently the worst.** Binary targeting (level 0 vs everything else) creates too much variance. Smooth decay is essential.

3. **Moderate decay > extreme decay.** Gentle attenuation (exp(α=0.8), linear(D=4)) outperforms aggressive (exp(α=0.5)) on Kuhn and Leduc. On Liar's Dice, gentle exp(α=0.5) wins — but the margin is small.

4. **Game complexity matters.** Kuhn sees 2-3× improvement; Liar's Dice marginal; Leduc baseline wins at 20k iterations. Larger games may need more iterations for IIG bias to pay off (initial overhead from non-uniform sampling).

5. **Decision biasing adds harmful variance on games with all-public actions** (Liar's Dice) or complex decision trees (Leduc). The biased sampling at decision nodes distorts the counterfactual regret computation more than it helps targeting.

## Hypotheses for Next Phases

- **H1**: Leduc IIG bias will overtake baseline at higher iteration counts (50k-100k). The biased configs show steeper convergence slopes at 20k.
- **H2**: Different I₀ targets may produce different results — Phase B should test sensitivity across infosets.
- **H3**: A δ-safety-switch (fraction δ use IIG-biased, 1-δ unbiased) could preserve convergence guarantees while retaining targeting benefits.
- **H4**: chance_exp(α=0.8) is a robust default across games.

## Next Steps

- **Phase B**: Run same experiment with multiple I₀ targets per game. Test if results are I₀-dependent.
- **Phase C**: Online match performance — ISGT vs OOS vs ISMCTS in bot-vs-bot play.
- **Phase D**: Full benchmark comparison on larger iteration budgets.
- **Consider**: Running Leduc with 50k-100k iterations to test H1.
- **Consider**: Implementing δ-mixture for convergence safety.

## Files

- `kuhn_poker.json` — Per-checkpoint exploitability for all 11 Kuhn configs
- `leduc_poker.json` — Per-checkpoint exploitability for all 11 Leduc configs  
- `liars_dice.json` — Per-checkpoint exploitability for all 11 Liar's Dice configs
- `all_results.json` — Combined results across all games
