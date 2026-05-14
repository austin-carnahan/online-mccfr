# 002 — LOTR Schedule Validation

Pre-ablation diagnostic for the four canonical LOTR schedule shapes under the
`(ρ, w)` parameterization. Validates that the schedule machinery in
`src/lotr.py` produces the divergence distributions it claims to, and surfaces
the corpus-level structural features that downstream ablations will need to
control for.

## Setup

- Bot: `LOTRBot` (canonical, standalone — no `depth_lotr` dependency)
- ρ = 0.5 (total per-state divergence budget)
- Schedules tested (depth-frame, four shapes):
  - `step(ρ, depth=0)` — single coin at root; OOS-targeted dichotomy
  - `early_linear(ρ)` — `w(d) = D − d` (mass at root in depth coords ↔ large k)
  - `uniform(ρ)` — `w(d) = 1` (flat)
  - `late_linear(ρ)` — `w(d) = d + 1` (mass at active in depth coords ↔ small k)
- Games: kuhn_poker, leduc_poker, goofspiel, liars_dice
- Corpus: 40 random-play decision states per game (24 for kuhn — tree exhausted)
- 500 sims per state, 3 seeds, ε=0.4, γ=0.01
- Total wall time: ~14 s (4 games × 4 schedules × ~40 states × 500 sims)

## Plots (per game)

Each game produces four panels (one per schedule, in the order
step → early_linear → uniform → late_linear):

1. **`<game>_divergence_by_D.png`** — **PRIMARY**. Per-D stratified
   `P(backoff=k | diverged, D=d)` with analytic `w(k|D)/Σ_j w(j|D)` overlay
   as black ticks. Each D-stratum is a clean controlled experiment; bars
   should land on the ticks.
2. **`<game>_divergence_aggregate_by_k.png`** — D-mixture
   `P(k | diverged, D≥k)`. Empirical bars + analytic mixture line. The
   D-invariant landing view *given the corpus's D distribution* — not a
   pure shape view (see "Reading the aggregate plot" below).
3. **`<game>_divergence_normalized.png`** — `P(diverge at k | k available)`.
   Same shape as aggregate-by-k, scaled by ρ. Kept for completeness.
4. **`<game>_divergence_raw.png`** — unconditional `P(backoff=k)` including
   the grey no-divergence bar at k=0. Heights weighted by both ρ and D
   availability; useful for sanity-checking the budget invariant.

## Findings

### 1. The schedule machinery is correct

Across all 16 (game, schedule) panels, empirical `P(backoff=k | diverged, D)`
matches analytic `w(k|D)/Σ_j w(j|D)` to within Monte Carlo noise at every D.
Verified concretely on leduc late_linear:

| k | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
|---|---:|---:|---:|---:|---:|---:|---:|
| empirical | 0.578 | 0.293 | 0.194 | 0.143 | 0.087 | 0.060 | 0.033 |
| analytic  | 0.584 | 0.295 | 0.187 | 0.136 | 0.082 | 0.051 | 0.036 |

The per-D plot is the honest view of shape — each D-stratum recovers the
defined `w` exactly.

### 2. ρ holds independent of shape

All four schedules deliver `P(diverged) ≈ 0.5` per state across all four games
(observed range: 0.490–0.503). The total-budget invariant of the `(ρ, w)`
parameterization is empirically confirmed.

### 3. `step(ρ, depth=0)` recovers OOS-targeted coin behavior

The step schedule's landing distribution genuinely depends on D — it places
100% of mass at the maximum k for each D. Empirically, `P(k | diverged, D)`
hits 1.0 exactly at the largest k available in each game (kuhn k=3, leduc k=7,
goofspiel k=5, liars_dice k=5). This is the only canonical shape whose
landing position is D-dependent.

### 4. Reading the aggregate plot: shape is a per-D object

The aggregate-by-k view is intuitive for *some* schedules (late_linear shows a
clean monotone decay toward larger k) but misleading for others. Two
mechanisms cause this:

- **Per-D shape mixing.** For uniform, `w(k|D) = 1/D`. Low-D states dump all
  their mass into small k without contributing to large k. The D-mixture is
  therefore monotonically decreasing in k even though the per-D shape is
  flat. Mathematically correct; intuitively misleading.
- **Per-k peak smearing.** early_linear puts its peak at k=D (different
  position per D). Averaging over D smears the peak across the k axis,
  flattening what was a strong shape at each fixed D.

late_linear is the only one of the four whose per-D peak position (k=1) is
D-invariant, so its aggregate-by-k curve faithfully mirrors the per-D shape.

**Conclusion:** the per-D plot is the right place to read off shape. The
aggregate-by-k plot is the right place to read off what the corpus actually
*does* under each schedule — useful for downstream effort-budget analysis but
not for shape intuition.

### 5. Leduc's flop is a structural signature in the data

The leduc step plot has a sharp dip at k=4: `P(k=4) ≈ 0.07` between k=3 (0.32)
and k=5 (0.50). For step, `P(k | diverged, D≥k)` reduces to
`iters_at_D=k / iters_at_D≥k`, so the curve is a direct readout of the corpus's
per-D iteration histogram:

| D | iters_at_D |
|---|---:|
| 1 | 4500 |
| 2 | 5000 |
| 3 | 3000 |
| **4** | **500** |
| 5 | 3500 |
| 6 | 3000 |
| 7 | 500  |

D=4 is rare because it sits exactly at the leduc round boundary — post-flop
states with very particular history lengths. Pre-flop (D=2/3) and deep
post-flop (D=5/6) are both common; the D=4 band is narrow.

Once you've seen it in the step panel, the same deficit is visible — diluted
— in every other leduc schedule:

- **uniform at k=4** sits below trend (off-monotone vs k=3 and k=5)
- **early_linear at k=4** breaks what should be a monotone-rising curve
  (k=4 ≈ 0.24 vs k=5 ≈ 0.30)
- **late_linear at k=4** is almost on-trend because so little mass lands at
  large k

None of the other three games show a comparable inflection: kuhn has no
chance interior, goofspiel's D distribution is smooth, and liars_dice's
chance is all at the root (so observability gates uniformly).

This is a genuine game-structural feature, not a probe artifact. Leduc is a
natural mini-stress test for any prefix-sensitive LOTR estimator: if a
quantity behaves anomalously only on leduc and only at the post-flop band,
the D=4 deficit is probably involved.

## Files

- `summary.md` — auto-generated probe report (config + empirical table + plot index)
- `state_corpus.json` — the 40-states-per-game corpus that all schedules share
- `raw_step_summaries.jsonl` — per-(state, schedule) bot tracking summary
- `histograms.csv` — aggregate `P(backoff=k)` table (D='all' marker)
- `histograms_by_D.csv` — per-D stratified table backing the PRIMARY plot
- `histograms_aggregate_by_k.csv` — D-mixture landing table (empirical + analytic) backing the aggregate-by-k plot
- `prefix_context.csv` — prefix-position kind counts (chance / p0 / p1) per depth
- `plots/` — all four plot variants for each game

## Status

Schedule validation complete. The `(ρ, w)` machinery in `src/lotr.py` is
correct, ρ-invariance holds, and per-D shapes match analytic predictions
across all four games. The corpus's D distribution (especially leduc's flop
inflection) is well-characterized and noted for downstream ablation
interpretation.

Next: schedule ablation on convergence / exploitability under the four shapes.
