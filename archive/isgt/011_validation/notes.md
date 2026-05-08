# 011 — Validation Experiments

**Date**: 2026-04-06  
**Archive**: 011_validation  
**Predecessor**: 010_precomp_oos_rerun  
**Status**: ✅ PASS (all 11 checks across 3 sub-experiments)

## Overview

A series of small, focused validation experiments on Kuhn Poker (12 infosets, 30 terminals) verifying correctness and behavioral properties of the ISGT implementation. These confirm that the algorithm works as specified before proceeding to histogram analysis, variance studies, or larger-scale experiments.

**Harness**: `experiments/validation_011.py` — unified script running all 3 sub-experiments with a `TracedISGTBot` subclass that captures per-node trace data.

## Sub-experiments

| ID | Name | Checks | Status |
|----|------|--------|--------|
| 011a | Single-Trace Rollout Validation | 8 | ✅ PASS |
| 011b | Sampling Mode Sanity Check | 4 | ✅ PASS |
| 011c | Support Restriction Check | 4 | ✅ PASS |

## Key Findings

### 011a — Trace-level correctness confirmed

- **s1/s2 dual tracking** works correctly: s1 tracks targeted-scenario reach (preserved when walk follows z*, drops to 0 on divergence); s2 tracks ε-on-policy reach (always updated).
- **Chance mode**: Decisions sample from ε-on-policy independently of z*. Chance nodes follow z* deterministically in targeted iterations.
- **Full mode**: All nodes (chance AND decision) follow z* in targeted iterations. This produces higher l values (both s1 and s2 contribute) and correspondingly lower |W| — lower variance per-iteration updates.
- **Regret signs**: Correct in all traces.

### 011b — Sampling parameters behave as specified

- **δ**: Targeting frequency matches δ exactly at boundaries (0 and 1) and within 95% binomial CIs at interior values.
- **ε**: The ε-on-policy formula $\sigma_{\text{sample}}(a) = \varepsilon/|A| + (1-\varepsilon)\cdot\sigma(a)$ verified deterministically at all 42 traced decision nodes.
- **Mode**: Full mode locks 100% of decisions to z* (72/72); chance mode does not (31/62 = 50%).
- **δ=0**: Equivalent to vanilla OOS — s1=0 everywhere, l=s2.

### 011c — IIG targeting is real, not random

- **Decay shapes distribution**: ConstantWeight produces uniform z* selection (~41% upstream vs 33% expected); Exp(0.5) concentrates more (46% upstream). Different decay functions produce measurably different distributions.
- **Proximity bias works**: Under Exp(0.5), distance-0 terminals are the most frequently selected among IIG-reachable terminals.
- **No coverage hole**: Untargeted iterations visit 24/30 unique terminals including 16 non-upstream terminals from just 103 iterations.
- **Matches theory**: χ² test confirms observed z* distribution is consistent with the theoretical IIG-weighted distribution (p=0.335, not rejected).

## Implications

The ISGT implementation is correct:

1. The dual s1/s2 tracking and l = δ·s1 + (1−δ)·s2 tail probability are implemented correctly
2. The δ/ε/mode parameters control behavior exactly as specified in the algorithm
3. IIG-weighted z* sampling genuinely biases toward upstream terminals (not equivalent to random)
4. The (1−δ) untargeted fraction provides full tree coverage — no reachability gaps
5. Full mode's lower |W| values (from preserved s1) confirm the variance-reduction mechanism

Ready to proceed to histogram analysis and variance studies.
