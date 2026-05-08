# 011a — Single-Trace Rollout Validation

**Date**: 2026-04-06  
**Archive**: 011_validation / 011a_single_trace_rollout  
**Status**: ✅ PASS  
**Game**: Kuhn Poker (12 infosets, 30 terminals)

## Setup

Ran 5 fully traced ISGT iterations in both **chance** and **full** mode. Each iteration logged every node visited during `_walk` with s1/s2 sample reach, strategy probabilities, z* matching, and regret updates.

**Configuration**: δ=0.9, ε=0.6, γ=0.01, ConstantWeight, seed=42.  
**Active infoset**: P0|"0" (player 0 dealt card 0).  
**IIG neighborhood**: Level 0 only (P0|"0"). Upstream terminals: 10/30.

## Results — Chance Mode

| Iter | Update | Targeted? | z* tid | IIG dist | z* actions | Terminal u | l | W |
|------|--------|-----------|--------|----------|------------|-----------|------|-------|
| 0 | P0 | NO | #11 | -1 (floor) | (1,0,0,1,0) | 1.00 | 0.0042 | 20.00 |
| 1 | P1 | YES | #1 | 0 | (0,1,0,1,0) | 2.00 | 0.0117 | 14.29 |
| 2 | P0 | NO | #0 | 0 | (0,1,0,0) | 1.00 | 0.0058 | 14.33 |
| 3 | P1 | YES | #9 | 0 | (0,2,1,1) | 1.00 | 0.0117 | 7.14 |
| 4 | P0 | YES | #18 | -1 (floor) | (1,2,1,0) | -1.00 | 0.0232 | -7.14 |

### Key observations — chance mode

- **Untargeted iterations (0, 2)**: Chance actions diverge from z* immediately → s1=0 throughout → l depends entirely on s2. Walk samples from natural chance probabilities.
- **Targeted iterations (1, 3, 4)**: Chance nodes follow z* deterministically (s1 preserved). Decision nodes still sample from ε-on-policy — observed divergence from z* at decisions (iter 1 depth 2: chose action 1 vs z*=0, iter 3 depth 2: chose 0 vs z*=1).
- **s1 × s2 tracking**: s2 always multiplied by σ_sample at decisions and ρ at chance. s1 tracks targeted-scenario reach correctly.
- **Regret signs**: Correct — sampled action gets (c−x)·W, others get −x·W.

## Results — Full Mode

| Iter | Update | Targeted? | z* tid | IIG dist | z* actions | Terminal u | l | W |
|------|--------|-----------|--------|----------|------------|-----------|------|-------|
| 0 | P0 | NO | #11 | -1 (floor) | (1,0,0,1,0) | 1.00 | 0.0042 | 20.00 |
| 1 | P1 | YES | #1 | 0 | (0,1,0,1,0) | 1.00 | 0.0321 | 1.30 |
| 2 | P0 | YES | #18 | -1 (floor) | (1,2,1,0) | 1.00 | 0.0342 | 2.44 |
| 3 | P1 | NO | #0 | 0 | (0,1,0,0) | -1.00 | 0.0083 | -20.00 |
| 4 | P0 | YES | #9 | 0 | (0,2,1,1) | -2.00 | 0.0342 | -4.88 |

### Key observations — full mode

- **Targeted iterations (1, 2, 4)**: ALL nodes (chance AND decision) follow z* — every action matches z*. s1 is preserved at 0.0333 throughout. This produces much higher l values (0.032–0.034 vs 0.004–0.023 in chance mode) and correspondingly lower |W| (1.3–4.9 vs 7.1–20.0).
- **Untargeted iterations (0, 3)**: Identical behavior to chance mode — s1=0 from chance divergence, walk samples normally.
- **W magnitude**: Full mode targeted iterations produce dramatically lower |W| values because l is larger (s1 preserved → l = δ·s1 + (1−δ)·s2 has both terms contributing). This means lower variance per-iteration updates.
- **Regret updates**: Smaller absolute magnitude in full mode targeted iterations (±0.65 to ±2.44) vs chance mode (±3.57 to ±14.26).

## Verification Checklist

| Check | Chance | Full |
|-------|--------|------|
| s1, s2 initialized to correct values | ✓ | ✓ |
| s1 preserved when walk follows z* | ✓ | ✓ |
| s1 → 0 on divergence from z* | ✓ | ✓ |
| s2 *= ρ at chance, σ_sample at decision | ✓ | ✓ |
| l = δ·s1 + (1−δ)·s2 > 0 at terminal | ✓ | ✓ |
| Chance mode: decisions sample ε-on-policy (not locked to z*) | ✓ | N/A |
| Full mode: targeted decisions always match z* | N/A | ✓ |
| Regret update signs correct | ✓ | ✓ |

**Result**: PASS (both modes)

Kuhn Poker is the ideal game for this: 12 infosets, 30 terminals, small enough to reason about every node in a trace.
- ISGT bias_mode: run both "chance" and "full" (separate traces)
- ISGT decay: ConstantWeight (simplest to verify)
- Seed: fixed (e.g., 42) for reproducibility

## Implementation Notes

Neither bot currently has per-iteration tracing. The script will need to either:
- **(a)** Subclass `ISGTBot`/`OOSBot` and override `_walk` to add logging, or
- **(b)** Add a `verbose=False` flag to `_walk` that enables print statements, or
- **(c)** Monkey-patch the walk methods with wrappers

Option (a) is cleanest — a `TracedISGTBot` that wraps `_walk` calls with logging but delegates all logic to the parent. The trace output goes to stdout with clean formatting.

Existing infrastructure to leverage:
- `ISGTDebugInfo` / `last_debug` — captures active infoset and IIG weights (header info)
- `iig.print_levels(iid)` — prints upstream BFS levels
- `iig._fmt_id(iid)` — human-readable infoset labels

## Script

`experiments/011a_single_trace.py`

## Expected Output

Clean tables like:

```
=== ISGT Iteration 3/10 (chance mode, δ=0.9) ===
Active: P0|"0"    Update player: 0
Targeted: YES (coin=0.23 < δ=0.9)
z*: terminal #7, actions=(1, 0, 1), IIG dist=1, p(z*)=0.0714

Depth | Type     | Player | Info      | Action | z*  | Match? | s1     | s2     | σ(a)  | σ_s(a)
------|----------|--------|-----------|--------|-----|--------|--------|--------|-------|-------
  0   | chance   | C      | —         | 1      | 1   | ✓      | 1.0000 | 0.3333 | —     | —
  1   | decision | P0     | "0"       | 0      | 0   | ✓      | 0.6500 | 0.2167 | 0.500 | 0.650
  2   | decision | P1     | "1 cb"    | 1      | 1   | ✓      | 0.6500 | 0.1408 | 0.500 | 0.650
  3   | terminal | —      | —         | —      | —   | —      | —      | —      | —     | —

Terminal: u=1.0, l=δ·s1+(1-δ)·s2 = 0.9·0.65 + 0.1·0.14 = 0.599
W = u · π₋ᵢ / l = 1.0 · 0.500 / 0.599 = 0.835
Regrets: a0 += +0.418 (sampled), a1 += -0.418

Checks: s1_init=✓ s2_track=✓ l_pos=✓ eps_policy=✓ mode=✓ regret=✓
```

## Results

*(To be filled after execution.)*
