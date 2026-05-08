# 011b — Sampling Mode Sanity Check

**Date**: 2026-04-06  
**Archive**: 011_validation / 011b_sampling_mode_sanity  
**Status**: ✅ PASS  
**Game**: Kuhn Poker (12 infosets, 30 terminals)

## Motivation

011a verifies individual traces. This experiment runs small batches and checks that the statistical properties of the sampling match expectations: δ controls targeting frequency, ε shapes the exploration policy, mode determines decision-node behavior, and untargeted iterations behave like vanilla OOS.

**Configuration**: ε=0.6, γ=0.01, ConstantWeight, seed=42.

## Results

### Check 1 — Targeting frequency matches δ (N=50 per δ)

| δ | Observed targeted | 95% CI (binomial) | Result |
|-----|-------------------|---------------------|--------|
| 0.0 | 0 / 50 | [0, 0] | ✅ PASS |
| 0.2 | 13 / 50 | [5, 16] | ✅ PASS |
| 0.5 | 22 / 50 | [18, 32] | ✅ PASS |
| 0.9 | 46 / 50 | [41, 49] | ✅ PASS |
| 1.0 | 50 / 50 | [50, 50] | ✅ PASS |

Boundary values (δ=0 and δ=1) are exact. Interior values fall within 95% binomial confidence intervals.

### Check 2 — ε-on-policy computation (deterministic)

Rather than a statistical frequency check (which would be confounded by the evolving policy), this check deterministically verifies the ε-on-policy formula at every traced decision node:

$$\sigma_{\text{sample}}(a) = \frac{\varepsilon}{|A|} + (1 - \varepsilon) \cdot \sigma(a)$$

**Result**: 42 decision nodes checked, all exact. ✅ PASS

This is a stronger test than a frequency-based approach — it confirms the formula holds on every single node, not just in aggregate.

### Check 3 — Full mode locks decisions to z* (N=30, δ=1.0)

| Mode | Decision nodes | z* matches | Fraction | Expected | Result |
|--------|----------------|------------|----------|----------|--------|
| chance | 62 | 31 | 50% | < 100% | ✅ PASS |
| full | 72 | 72 | 100% | = 100% | ✅ PASS |

**Key observation**: In chance mode, decision nodes sample from ε-on-policy independently of z*, so matching z* is probabilistic (~50% with uniform initialization and |A|=2). In full mode, targeted iterations deterministically follow z* at all nodes (chance AND decision), producing the expected 100% match rate.

The node count difference (62 vs 72) is expected — full mode always follows z* to completion, while chance mode can diverge at decisions and reach different (sometimes shorter) terminal paths.

### Check 4 — Untargeted iterations (δ=0): s1 ≡ s2

| Property | Observed | Result |
|----------|----------|--------|
| Targeted iterations | 0 / 50 | ✅ as expected |
| l = s2 at all terminals | Yes (all 50) | ✅ PASS |

At δ=0, no iteration is targeted, so z* never constrains the walk. s1 tracks the targeted-scenario reach, which is always 0 when the walk isn't targeted. The tail probability l = δ·s1 + (1−δ)·s2 = s2. This confirms untargeted ISGT is equivalent to vanilla OOS.

## Summary

| Check | What it validates | Result |
|-------|-------------------|--------|
| 1 | δ controls targeting frequency correctly | ✅ |
| 2 | ε-on-policy formula applied exactly at every decision | ✅ |
| 3 | Full mode locks all decisions to z*; chance mode does not | ✅ |
| 4 | δ=0 ⟹ s1=0, l=s2, equivalent to baseline OOS | ✅ |

**Overall: PASS**
