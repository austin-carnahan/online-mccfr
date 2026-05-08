# Archive 016 — Anchor-Split Walk Redesign & Full Ablation

## Motivation

Experiment 015C found two independent properties in ISGT's walk modes:

| Property | Full mode | Chance mode |
|----------|----------|-------------|
| Proximity gradient (r) | -0.8 (strong) | ~0 (flat) |
| Per-node regret magnitude | ~0.004 (tiny) | ~0.25 (large) |
| Importance weight (l) | ~0.5 (flat) | ~0.04 (small) |
| Exploration | None | Full ε-on-policy |

Full mode has the gradient but can't exploit it — deterministic walks
keep s1=1.0 the entire path, giving l ≈ 0.5 and W ≈ 2 (tiny updates).
Chance mode has throughput but no gradient — ε-on-policy at every
decision erases the geometric coupling between z*'s IIG level and I₀.

## The Anchor-Split Walk

The redesign combines both: **deterministic prefix, ε-on-policy suffix,
split at the IIG anchor point.**

### Walk Structure

```
Root ──── [deterministic: follow z*] ──── Anchor ──── [ε-on-policy] ──── Terminal
                                          depth ≈ D₀ − k
```

1. **Sample z\*** as before (two-stage: pick level k, then terminal).
2. **Identify the anchor** — the shallowest infoset at level k on z\*'s
   decision sequence. Its game-tree depth is the split point.
3. **Deterministic prefix** (root → anchor): follow z\*'s actions exactly.
   s1 stays 1.0. Locks the walk into the IIG level-k neighborhood.
4. **ε-on-policy suffix** (anchor → terminal): sample like chance mode.
   s1 decays at each decision node. Exploration happens here.

### Why It Works — IIG Geometry

IIG edges are *upstream*: level k from I₀ means k player decisions
*back* in the game tree. The anchor infoset at level k sits at roughly
depth D₀ − k. Every terminal continuation forward from the anchor
still passes through the anchor, so any ε-on-policy suffix stays at
IIG level ≤ k. The walk never leaves the targeted neighborhood.

### Comparison to OOS

OOS does a similar split, but its split point is the current game state
(depth D₀). The anchor-split point is at depth D₀ − k, giving k extra
player turns of ε-on-policy exploration compared to OOS.

### Unification of Existing Modes

| Mode | Split point | Prefix | Suffix |
|------|------------|--------|--------|
| Old full | Never (∞) | Entire walk | None |
| Chance | Root (0) | Chance nodes only | All decisions |
| **Anchor** | **Level-k infoset** | **Root → anchor** | **Anchor → terminal** |
| OOS | Game state (D₀) | Root → game state | Game state → terminal |

Old full = anchor with k = ∞. Chance = anchor with k = 0 (or no anchor).

## Implementation Changes (src/isgt.py)

### `_compute_anchor_depth(active_iid, tid)`

New method. Replays z\*'s terminal history to find the game-tree depth
of the shallowest infoset at the minimum IIG level on z\*'s decision
sequence. Handles both P1 and P2 anchor nodes correctly.

Returns 0 for floor terminals (no upstream infosets) — defaults to
fully ε-on-policy walk (no reason to deterministically replay an
irrelevant terminal).

### Floor terminals excluded from targeting distribution

`_get_target_dist` now skips level -1 (floor) terminals entirely.
100% of targeting probability goes to IIG neighborhood terminals.
Full-support guarantee is maintained by the (1−δ) untargeted iterations.

Previously, floor terminals consumed ~28-33% of targeting budget
(with ConstantWeight) doing zero useful neighborhood work.

### Decision node logic: two phases

**Before anchor (in prefix):**
- Targeted iteration: deterministic — follow z\*, s1 stays 1.0
- Untargeted iteration: ε-on-policy, s1 tracks pass/fail against z\*

**After anchor (suffix):**
- Both scenarios: ε-on-policy, s1 and s2 evolve identically
  (s1 = s1 × sample_policy, s2 = s2 × sample_policy)

### Chance node logic: same two phases

Before anchor: deterministic follow (targeted) or natural sample with
s1 tracking (untargeted). After anchor: natural distribution, s1 × ρ
for both scenarios.

## Verified Results (P2 decision in Leduc, 200 sims)

| IIG Level | Old full l | Anchor l | Old full per-node | Anchor per-node |
|-----------|-----------|----------|-------------------|-----------------|
| 0 | ~0.500 | 0.049 | ~0.004 | 0.278 |
| 1 | ~0.500 | 0.065 | ~0.004 | 0.058 |
| -1 (floor) | ~0.500 | excluded | 0.000 | excluded |

Proximity gradient preserved (0.278 → 0.058). Regret magnitudes
boosted ~70x from old full mode into the same regime as chance mode.

All games tested and passing: Kuhn, Leduc, Goofspiel, Liar's Dice.
22/22 existing tests pass.

---

## Experiment Plan: Full Ablation

Now that the walk mechanics are redesigned, we need a full ablation
to validate the anchor-split walk against the baselines.

### 16A — Anchor vs Baselines (Aggregate Exploitability)

**Purpose:** Does the anchor-split walk improve aggregate exploitability
over old full mode, chance mode, and OOS?

**Configs (4):**

| # | Algorithm | Mode | Label |
|---|-----------|------|-------|
| 1 | OOS (IST) | — | `oos` |
| 2 | ISGT | chance | `isgt_chance` |
| 3 | ISGT | full (anchor) | `isgt_anchor` |
| 4 | ISGT | full (old, if retained) | `isgt_full_old` |

**Parameters:**
- Fixed: δ=0.5, ε=0.4, γ=0.01 (paper §4.4.2)
- Decay: constant (fair baseline — no proximity bias in level selection)
- Games: kuhn_poker, leduc_poker, goofspiel, liars_dice
- Budgets: [100, 250, 500, 1000] sims/move
- Matches: 500 per config

**Key question:** Does anchor mode beat OOS and chance mode at aggregate
exploitability, especially at low budgets?

### 16B — Anchor Proximity Probe

**Purpose:** Repeat 015C with the anchor walk to confirm the proximity
gradient now carries through to meaningful regret magnitudes.

**Configs (3):**

| # | Mode | Decay | Label |
|---|------|-------|-------|
| 1 | anchor | constant | `anchor/const` |
| 2 | anchor | sqrt | `anchor/sqrt` |
| 3 | anchor | tbal | `anchor/tbal` |

**Parameters:**
- Fixed: δ=0.5, ε=0.4, γ=0.01
- Game: Leduc poker
- Budgets: [5, 10, 20, 50] sims/move
- Matches: 500 per config

**Key metrics:**
- Per-node profile by IIG level (expect declining gradient like old full,
  but with magnitudes in the ~0.2 range like chance mode)
- Importance weights (expect l << 0.5, confirming s1 decay in suffix)
- Slope summary (expect r ≈ −0.8, confirming proximity persists)

### 16C — Delta × Decay Sweep

**Purpose:** Map the anchor walk's parameter surface. Does the decay
function finally matter now that proximity-weighted updates have
meaningful throughput?

**Grid:**
- Games: leduc_poker, liars_dice, goofspiel
- δ: [0.2, 0.5, 0.7, 0.9]
- Decay: [constant, sqrt, tbal]
- Budgets: [100, 250, 500, 1000]
- Matches: 500 per config

**Key question:** With anchor mode, does constant decay finally outperform
tbal? (015C showed constant has built-in proximity bias via per-terminal
oversampling of low levels. With anchor mode's proximity gradient,
this oversampling should now translate into better exploitability.)

---

## Predictions

1. **16A:** Anchor mode should outperform old full mode substantially
   (the l ≈ 0.5 flat importance weight was a direct handicap). Should
   compete with or beat chance mode, especially in Leduc where full
   mode's proximity gradient was strongest. May beat OOS at matched δ
   because anchor's split is geometrically motivated (IIG level) rather
   than arbitrarily placed at the game state.

2. **16B:** Per-node profiles should show both gradient AND magnitude —
   declining from ~0.2-0.3 at level 0 to ~0.05-0.1 at level 6, with
   importance weights in the l ≈ 0.03-0.05 range (matching chance mode).

3. **16C:** Decay function may finally produce a meaningful signal.
   Constant decay's proximity oversampling (level 0 terminals get ~278x
   more samples) combined with anchor mode's proximity gradient could
   create synergy that wasn't possible before.
