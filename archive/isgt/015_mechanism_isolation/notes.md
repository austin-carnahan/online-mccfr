# Archive 015 — Mechanism Isolation, Proximity Probe & Discovery

## Context

Experiment 014 established that under the aggregate exploitability metric:
- δ is the dominant parameter (lower δ → better aggregate exploitability)
- Decay function produces no meaningful signal (washed out by aggregation)
- Chance mode is a no-op on Goofspiel (zero interior chance nodes)
- Full mode degrades with δ due to importance-weight variance
- The paper uses δ=0.5, ε=0.4 for aggregate exploitability (§4.4.2)

Experiment 015 strips ISGT down to its core mechanism — sampling full
terminal histories z\* for targeting — and compares directly against OOS's
partial-history targeting.  With decay held to just 2 choices (constant
and t_balanced), we isolate what z\*-based targeting buys us before
introducing more complex decay functions.

**015C was the breakthrough experiment.** It found the first real evidence
of a proximity gradient in the IIG — specifically in ISGT full mode on
Leduc poker — leading directly to the anchor-split walk redesign in
experiment 016.

## Paper Parameter Reference

| Parameter | Paper §4.4.2 | Code param | Exp 014 used |
|-----------|:------------:|:----------:|:------------:|
| Targeting (δ) | **0.5** | `delta` | 0.2/0.5/0.9 |
| Exploration (ε) | **0.4** | `epsilon` | 0.6 |
| RM floor (γ) | **0.01** | `gamma` | 0.01 |

Note: the paper writes "γ = 0.4 for exploration" in §4.4.2 but means ε.
Confirmed by §4.4.3 which says "exploration ε = 0.4".  The γ = 0.01
regret matching floor is a separate parameter.

---

## 15A — Mechanism Isolation (Aggregate Exploitability)

### Purpose

At matched parameters (δ=0.5, ε=0.4, γ=0.01), compare OOS's partial-
history targeting against ISGT's full-terminal-history targeting.  Two
decay baselines (constant = no proximity bias, t_balanced = tree-shape-
aware) isolate the targeting mechanism from proximity shaping.

### Configs (5)

| # | Algorithm | Mode | Decay | Label |
|---|-----------|------|-------|-------|
| 1 | OOS (IST) | — | — | `oos` |
| 2 | ISGT | chance | constant | `isgt_chance_constant` |
| 3 | ISGT | chance | t_balanced | `isgt_chance_tbal` |
| 4 | ISGT | full | constant | `isgt_full_constant` |
| 5 | ISGT | full | t_balanced | `isgt_full_tbal` |

### Parameters

- **Fixed:** δ=0.5, ε=0.4, γ=0.01 (paper §4.4.2 settings)
- **Games (4):** kuhn_poker, leduc_poker, liars_dice, goofspiel
- **Budgets (4):** [100, 250, 500, 1000] sims/move
- **Jobs:** 4 games × 5 configs × 4 budgets = **80**

### Metrics

| Metric | Measures | Applies to | Implementation |
|--------|----------|------------|----------------|
| **Exploitability** | Aggregate strategy quality | All 5 configs | Existing `aggregate_exploitability()` |
| **Infoset coverage** | Fraction of IIG infosets with nonzero weight | All 5 configs | Post-hoc count on `global_strategy` vs IIG size |
| **Level-stratified weight profile** | Coverage + mean weight at each IIG distance from active infosets | All 5 configs | Per active infoset: `iig.levels()` → weight by distance |
| **Active regret by z\* level** | Mean |Δregret| at I₀, bucketed by z\*'s IIG level | ISGT only (4 configs) | Snapshot I₀ regret before/after targeted `_walk()` |
| **Neighborhood regret by z\* level** | Total |Δregret| across upstream IIG neighborhood, bucketed by z\*'s IIG level | ISGT only (4 configs) | Snapshot all neighborhood regrets before/after targeted `_walk()` |

### Key Questions

1. At matched δ=0.5, does ISGT differ from OOS in aggregate exploitability?
2. Does chance vs full mode matter at δ=0.5?
3. Does constant vs t_balanced produce different weight profiles despite
   similar exploitability?
4. **Does z\* proximity matter for regret work?** — At each IIG level,
   does the targeted walk produce regret updates in I₀'s neighborhood?
   The active metric shows a predictable cliff (distant z\* can't reach
   I₀ directly).  The neighborhood metric reveals whether distant z\*
   still does useful work on nearby infosets.

---

## 15B — δ × ε Parameter Sweep for ISGT

### Purpose

Map ISGT's aggregate response surface over (δ, ε), mirroring paper
Figures 3(c,f,j).  Determines ISGT's optimal operating point per game.

### Grid

- **Games (3):** leduc_poker, liars_dice, goofspiel (drop kuhn — trivially small)
- **Modes (2):** chance + full (conditionally drop one after 15A results)
- **Decay (1):** constant
- **δ:** [0.2, 0.5, 0.7, 0.9] (4 values)
- **ε:** [0.4, 0.6] (2 values — paper's value vs our prior default)
- **γ:** 0.01 (fixed)
- **Budgets (4):** [100, 250, 500, 1000] sims/move (matched to 15A)
- **Jobs:** 3 games × 2 modes × 4δ × 2ε × 4 budgets = **192** (96 if one mode dropped)

### Metrics

| Metric | Measures | Implementation |
|--------|----------|----------------|
| **Exploitability** | Aggregate strategy quality | Existing `aggregate_exploitability()` |
| **Infoset coverage** | Fraction of IIG infosets with nonzero weight | Post-hoc count |

### Key Questions

5. What is ISGT's optimal (δ, ε) for aggregate exploitability per game?
6. How does coverage scale with δ? (validates coverage → exploitability
   theory from exp 014)
7. Does the optimal δ change with simulation budget?

---

## Metric Details

### Infoset Coverage (both 15A and 15B)

Per player, after all matches complete:

    coverage_p = |{I in global_strategy[p] : sum(weights) > 0}| / |IIG infosets for player p|

Single scalar per (config, game, budget).

### Level-Stratified Weight Profile (15A only)

For each active infoset encountered during matches, use `iig.levels()`
to partition all infosets by IIG distance.  Per level k, compute:

- **Coverage:** fraction of level-k infosets with nonzero weight in
  `global_strategy`
- **Mean weight:** average total strategy mass at level-k infosets

Output: `{level_k: {coverage, mean_weight, n_infosets}}` per job.

The main value is the **cross-algorithm** comparison — OOS vs ISGT
chance vs ISGT full produce weight profiles via fundamentally different
mechanisms.  The cross-decay comparison (constant vs t_balanced) is
largely redundant with decay histograms from earlier experiments.

### Regret Coupling by z\* Level (15A, ISGT only)

Two paired metrics measure how much "useful regret work" a targeted MCCFR
iteration performs, bucketed by IIG distance of the sampled z\* from the
active infoset I₀.

#### What gets measured

Each ISGT iteration samples one z\* and runs `_walk()` from root to a
terminal.  Along that sampled path, `_walk()` updates the regret array
at **every** decision node belonging to `update_player` — not just I₀.

Before and after the walk, we snapshot regret arrays at two scopes:

1. **`active_regret_by_level`** — |Δregret| at I₀ only.
2. **`neighborhood_regret_by_level`** — Sum of |Δregret| across I₀'s
   entire upstream IIG neighborhood (all infosets reachable via
   `iig.levels(I₀)` that have been initialised in `_infostates`).

Both are bucketed by the z\*'s IIG level (minimum IIG distance from any
infoset on z\*'s decision sequence to I₀).

#### Why two metrics

The active-only metric produces a predictable cliff: a z\* at IIG
distance ≥ 2 from I₀ almost never passes through I₀, so |Δregret| at
I₀ is mechanically zero.  This tells us nothing interesting.

The neighborhood metric captures work that a distant z\* does on infosets
*near* I₀ in the IIG.  This work benefits I₀ indirectly: when neighbor
strategies improve, future walks passing through both the neighbor and
I₀ will propagate better counterfactual values to I₀.  The mechanism
is multi-iteration information propagation through overlapping terminal
sets, not single-walk ancestor updates.

#### IIG distance semantics (important nuance)

IIG level k means: there exists a chain of k pairwise consecutive-in-
some-terminal infosets connecting J to I₀.  For k ≥ 2, J and I₀ need
**not** share any terminal history — they are connected indirectly
through intermediate infosets with overlapping terminal sets.

So "nearby in the IIG" does **not** mean "on the same game-tree path."
It means "in a region of the information-set graph connected through
chains of mutually informative terminal sets."  Shorter chains = fewer
intermediate propagation steps needed for z\*'s regret updates to
eventually influence I₀'s strategy.

#### How it's calculated

Per targeted iteration where `update_player == bot._player_id`:

```python
# Before _walk():
active_before = infostates[I₀][REGRET].copy()
neighborhood_snapshot = {
    iid_key: infostates[iid_key][REGRET].copy()
    for iid in iig.levels(active_iid)
    if iid_key in infostates
}

# After _walk():
active_delta = sum(|active_after[a] - active_before[a]|)
neighborhood_delta = sum(
    sum(|after[a] - before[a]|)  for each iid in snapshot
)

# Bucket both by z*'s IIG level
```

#### Output per level k

    active_regret_by_level:       {mean_abs_delta, std_delta, n_samples}
    neighborhood_regret_by_level: {mean_abs_delta, std_delta, n_samples}
    neighborhood_touched_by_level: {mean_touched, std_touched, n_samples}

The **ratio** neighborhood / active at each level is the "neighborhood
benefit multiplier":

- **Level 0:** ratio ≈ 1.1–1.8x (z\* passes through I₀ + some neighbors)
- **Level 1–2:** ratio grows sharply (direct I₀ update fades, neighbor
  updates persist)
- **Level 3+:** active → 0, neighborhood often nonzero → ratio → ∞

The **per-infoset average** (neighborhood_delta / mean_touched) controls
for the fan-out effect.  The upstream BFS cone grows with level, so the
raw neighborhood sum includes more infosets at higher levels.  Dividing
by the number of actually-touched infosets isolates the per-infoset
regret intensity from the set-size effect.

#### What the results tell us

- If `neighborhood_regret` drops sharply with IIG distance, the game has
  strong locality: even neighbor benefit decays with distance, justifying
  aggressive decay functions.

- If `neighborhood_regret` stays relatively flat, distant z\* does as much
  neighborhood work as nearby z\*.  Proximity-weighted decay wastes budget
  by over-sampling level-0 terminals that don't provide proportionally
  more benefit.  Constant weighting is optimal.

- The drop-off shape directly informs what decay function (if any) is
  justified — exponential, polynomial, or step.

#### Interpreting constant vs t_balanced (sampling bias)

The two decay baselines in 15A have fundamentally different sampling
distributions, which matters for interpreting the per-infoset profiles.

**Constant weighting** assigns equal probability to each IIG *level*, then
uniform within that level.  Because terminal counts grow sharply with
level (e.g. Leduc: 8, 56, 36, 80, 745, 1295, 1080, 2220 terminals at
levels 0–7), an individual level-0 terminal gets sampled ~278x more often
than a level-7 terminal.  **Constant is already aggressively proximity-
biased at the terminal level** — it's only "constant" at the level level.

**Terminal_balanced weighting** sets level weight ∝ terminal count, which
cancels the 1/|T_ℓ| within-level factor to produce P(z) = 1/N for all
terminals.  This is genuinely unbiased over terminals.

The interpretation framework:

| Observation (per-infoset view) | Under constant | Under t_balanced |
|-------------------------------|:---:|:---:|
| Flat across levels | Ambiguous: could be real OR proximity bias masking decay | **Strong evidence** proximity doesn't matter |
| Drops with level | Strong locality signal (even with built-in bias, distant z\* is worse) | Moderate locality (but sampling was unbiased, so effect is genuine) |
| Constant flatter than t_balanced | Constant's proximity oversampling is compensating for a natural decay | — |
| Both profiles identical | **Definitive:** proximity doesn't matter for this game | — |

The key comparison is between the two curves:

1. **If constant ≈ t_balanced per-infoset profiles:** The built-in
   proximity bias of constant weighting doesn't help or hurt.  The
   coupling structure is genuinely flat.  Constant (or any decay) is
   equivalent.  This is consistent with experiment 014's finding that
   decay function doesn't affect aggregate exploitability.

2. **If constant shows flatter per-infoset profile than t_balanced:**
   Constant's proximity oversampling is compensating for a real decay
   in the natural coupling.  An intermediate decay function (between
   constant and t_balanced) would be optimal — it would concentrate
   budget at the levels where per-infoset return is highest without
   the extreme oversampling constant does at level 0.

3. **If t_balanced shows flatter per-infoset profile than constant:**
   Counter-intuitive — would suggest distant terminals contribute
   *more* per-infoset and constant's proximity bias is actively harmful.
   This would argue for t_balanced or even inverse-proximity weighting.

#### Why constant decay is important for measurement

Under constant decay, z\* samples are uniform across IIG levels.  This
gives unbiased level coverage and a clean measurement of the natural
coupling structure.  t_balanced provides a second baseline where z\* is
uniform per-terminal (biased toward high-terminal-count levels).

---

## Design Rationale

### Why no OOS in 15B's sweep

15A gives the direct OOS vs ISGT comparison at matched δ=0.5.  OOS and
ISGT use δ differently (partial-history vs full-terminal targeting), so
sweeping δ across both conflates "how does targeting intensity affect
ISGT?" with "how does ISGT compare to OOS?"  Keep 15A = mechanism
comparison, 15B = parameter optimization.

### Why only constant + t_balanced in 15A

Experiment 014 showed all 7 decays produce nearly identical exploitability.
Two decays suffice:  constant = no proximity bias (flat baseline),
t_balanced = tree-shape-aware (structural baseline).  Any difference
between them is purely the effect of terminal-count-weighted z\* selection.

### Why drop kuhn from 15B

Trivially small game where everything converges quickly.  Kept in 15A
for sanity checking only.

### Why ε sweep is minimal

The paper notes exploration impact is "not very strong" (§4.4.1).  ISGT
chance mode doesn't even use ε during targeting (only chance nodes are
constrained).  Two values (0.4 vs 0.6) are sufficient to detect any
meaningful ε effect.

---

## 15C — Low-Budget Proximity Probe

### Purpose

015A showed that the per-node neighborhood regret profile was **flat**
across IIG levels in chance mode (~0.25 at all levels) and showed tiny
magnitudes in full mode (~0.004).  The proximity signal we hypothesized
wasn't visible under standard simulation budgets.

The insight: at high sim counts, regret arrays converge and the per-
iteration |Δregret| shrinks toward zero — the signal is buried in the
noise floor.  At low budgets (sims=5 to 50), regret tables are still
volatile, so each iteration's contribution is larger and easier to
measure.

015C was designed to probe proximity in this low-budget regime, with
additional controls for early-vs-late iteration effects.

### Configs (6)

| # | Mode | Decay | Label |
|---|------|-------|-------|
| 1 | chance | constant | `chance/const` |
| 2 | chance | sqrt | `chance/sqrt` |
| 3 | chance | tbal | `chance/tbal` |
| 4 | full | constant | `full/const` |
| 5 | full | sqrt | `full/sqrt` |
| 6 | full | tbal | `full/tbal` |

### Parameters

- **Fixed:** δ=0.5, ε=0.4, γ=0.01
- **Game:** Leduc poker only (richest IIG structure: 7 levels, 936 infosets)
- **Budgets:** sims_per_move = [5, 10, 20, 50]
- **Matches:** 500 per config (full run)

### Key Metrics

1. **Per-node profile:** neighborhood_delta / touched_count at each IIG
   level.  Controls for the fan-out effect (higher levels have more
   infosets in the upstream BFS cone).

2. **Early vs late bucketing:** First half vs second half of simulations
   within each step() call.  Tests whether the proximity signal changes
   as regrets accumulate within a decision.

3. **Importance weight (l) by level:** Reveals the s1/s2 mechanics.
   Full mode had l ≈ 0.5 at all levels (deterministic walk → s1=1.0).
   Chance mode had l ≈ 0.04 (stochastic → s1 decays).

4. **Slope summary (Pearson r):** Correlation of (IIG level, per-node
   regret).  Negative r = proximity gradient exists.  Single number
   per (config, budget, early/late) combination.

### Results & Discovery Narrative

**The proximity gradient is real — but only visible in full mode.**

Full mode per-node profiles decline from level 0 to level 6:
- full/const sims=50: 0.0064 → 0.0014 (5x drop)
- full/const sims=5:  0.0050 → 0.0026 (2x drop)

Chance mode per-node profiles are flat at ~0.25 across all levels.

**Table 8 (Slope Summary) was the smoking gun:**

| Config | s=5 early | s=5 late | s=50 early | s=50 late |
| :--- | ---: | ---: | ---: | ---: |
| chance/const | 0.024 | 0.809 | 0.647 | 0.556 |
| full/const | **-0.826** | **-0.692** | **-0.837** | **-0.906** |
| full/sqrt | **-0.586** | **-0.590** | **-0.936** | -0.328 |

Full mode: consistently negative r (−0.7 to −0.9).  Chance mode: near
zero or positive.  The proximity gradient exists in full mode and is
absent in chance mode, across all budgets and early/late splits.

**But there was a catch:** Full mode's per-node magnitudes (~0.004) were
50x smaller than chance mode's (~0.25).  Full mode had the gradient but
not the throughput.  Chance mode had the throughput but not the gradient.

**Root cause: importance weight mechanics.**  Table 7 revealed:
- Full mode: l ≈ 0.5000 ± 0.0002 at every level (flat, because s1=1.0
  when walks are fully deterministic → l = δ·1.0 + (1-δ)·s2 ≈ 0.5)
- Chance mode: l ≈ 0.04 ± 0.04 (stochastic walks → s1 decays)

Since W = u·π₋ᵢ/l, larger l → smaller W → smaller regret updates.
Full mode's deterministic walk kept s1=1.0 the whole way, making l
large and updates tiny.  It has the proximity signal but can't act
on it with meaningful regret magnitudes.

**This directly motivated the anchor-split walk redesign (experiment 016):**
Switch from deterministic to ε-on-policy at the IIG anchor point. This
preserves the proximity gradient (deterministic prefix locks the IIG
level) while introducing stochasticity in the suffix (s1 decays, l
drops, W increases, updates become meaningful).

### Archived Results

- `015c_results/probe_results.json` — raw per-iteration data
- `015c_results/tables.md` — all 8 tables in markdown
- `015c_plots/fig1_per_node_profiles.png` — per-node by level
- `015c_plots/fig2_early_vs_late.png` — early/late comparison
- `015c_plots/fig3_importance_weights.png` — l by level/mode
- `015c_plots/fig4_chance_vs_full.png` — chance vs full overlay

---

## Job Totals

| Experiment | Games | Configs | Budgets | Jobs | Metrics |
|:----------:|:-----:|:-------:|:-------:|-----:|---------|
| **15A** | 4 | 5 | 4 | **80** | All 4 metrics |
| **15B** | 3 | 2 modes × 4δ × 2ε | 4 | **192** | Exploitability + coverage |
| **15C** | 1 | 6 | 4 | **24** | Per-node, IW, slope |
| **Total** | | | | **296** | |

---

## Commands

```bash
# 15A
python -m experiments.mechanism_isolation_015a --quick   # ~5 min
python -m experiments.mechanism_isolation_015a --full    # ~30 min

# 15C
python -m experiments.proximity_probe_015c --quick      # ~2 min
python -m experiments.proximity_probe_015c --full        # ~15 min

# 15C plots + tables
python -m experiments.plot_proximity_probe_015c
```
