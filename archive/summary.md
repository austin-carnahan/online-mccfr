# Experiment Index

## Retrospective Sampling Experiments

| # | Name | Summary |
|---|------|---------|
| 001 | Retro Decay Ablation | 6 decay functions × 4 games; found uniform wins 3/4 games, step catastrophically bad in goofspiel |
| 002 | Retro Uniform vs OOS | Full 4-game AE comparison; Retro dominates: goofspiel 47%, kuhn 31%, leduc 14%, liars_dice 12% |
| 003 | Retro Epsilon Ablation | ε sweep {0.1–0.8} × 4 games; optimal ε is game-dependent: goofspiel=0.4, leduc=0.2, liars_dice=0.4 |

## ISGT Experiments (Historical)

| # | Name | Summary |
|---|------|---------|
| 001 | Root Convergence (Phase A) | Initial ISGT vs ISMCTS exploitability comparison across games |
| 002 | Root Convergence (Iteration) | Iteration-based convergence comparison, controlling for sim count |
| 003 | Root Convergence (Time) | Time-based convergence — wall-clock fairness between algorithms |
| 004 | OOS Debug (Liar's Dice) | Debugging OOS implementation against ISMCTS on Liar's Dice |
| 005 | ISGT Optimization Validation | Quick sanity check that ISGT optimizations don't change results |
| 006 | ISGT Ablation (Quick) | Fast sweep of decay functions × bias modes on aggregate exploitability |
| 007 | ISGT Ablation (Full) | Full-scale decay × mode ablation — 7 decays, 2 modes, 4 games |
| 008 | Delta Ablation | δ parameter sweep × decay × mode on aggregate exploitability |
| 009 | IIG Reachability | Upstream BFS terminal coverage — validating IIG level structure |
| 010 | Pre-initialized OOS Rerun | OOS with pre-built infostate tree to eliminate tree-building confound |
| 011 | Validation Suite | Correctness checks: single-trace rollout (a), sampling mode sanity (b), support restriction (c) |
| 012 | Sampling Distribution | Analysis of how two-stage targeting distributes z* samples across IIG levels |
| 013 | Two-Stage Decay Histograms | Visualizing realized z* distributions under different decay functions |
| 014 | Post-Fix Delta Baseline | δ × decay grid after s1 fix; found δ dominant, decay signal flat, paper uses δ=0.5 |
| 015A | Mechanism Isolation | OOS vs ISGT at paper params (δ=0.5, ε=0.4); per-node neighborhood regret profiles |
| 015B | Parameter Surface | δ × ε sweep for ISGT optimal operating point per game (designed, not yet run) |
| 015C | Low-Budget Proximity Probe | chance/full × 3 decays × sims 5–50; found real proximity gradient in full mode (r ≈ −0.8) |
| 016 | Anchor Walk Redesign | Anchor-split walk: deterministic prefix to IIG anchor, ε-on-policy suffix; full ablation planned |

---

## Key Terms

| Term | Meaning |
|------|---------|
| IIG | Information-set Intersection Graph — nodes are infosets, edges connect infosets that share a terminal history |
| ISGT | Information Set Graph Targeting — our algorithm; samples a terminal z* from the IIG to target MCCFR walks |
| ISGT full | Full-mode targeting (anchor-split): deterministic from root to z*'s IIG anchor infoset, then ε-on-policy for the suffix |
| ISGT chance | Chance-mode targeting: walk follows z* only at chance nodes, samples ε-on-policy at decision nodes |
| OOS | Online Outcome Sampling (Lisý et al. 2015) — targets a subgame Z_sub (many terminals), not a single z* |
| Constant decay | Level weights w(ℓ) = 1; equal probability per IIG level, heavy per-terminal bias toward nearby levels |
| Terminal-balanced (tbal) | Level weights w(ℓ) ∝ \|T_ℓ\|; cancels within-level bias so P(z) = 1/N uniform over all terminals |
| Sqrt-balanced | Level weights w(ℓ) = √\|T_ℓ\|; intermediate — less proximity bias than constant, less flat than tbal |

---

## Paper Experiments We're Replicating

Source: Lisý, Lanctot, Bowling 2015 — "Online Monte Carlo Counterfactual Regret Minimization for Search in Imperfect Information Games"

### §4.4.1 — Convergence at the Root

**What it tests:** Does OOS converge to Nash equilibrium when run from the
root of the game (no match history, no targeting)? Compared against two
ISMCTS variants: UCT (doesn't converge) and Regret Matching (converges
slowly).

**Setup:** Run each algorithm from the empty history with no targeting
(δ=0, or equivalently OOS becomes plain MCCFR with incremental tree
building). Measure exploitability of the strategy over time. ε=0.6.

**Result:** OOS converges to exact NE in all three games.  ISMCTS-UCT
initially improves then diverges. ISMCTS-RM converges but slower.

**Our take:** This is a straightforward offline convergence proof. No
targeting is active — the algorithm is just MCCFR. This tells us the
base algorithm works; it doesn't tell us anything about targeting.

### §4.4.2 — Aggregated Strategy Exploitability

**What it tests:** Does OOS produce less exploitable strategies than ISMCTS
when playing many matches online (with targeting active)?

**Setup — the aggregate method:** Run 500 matches of each algorithm against
a random opponent. In each match, the algorithm targets from the current
match history (I(m)) using IST with δ=0.5, ε=0.4. After each match, the
computed strategy data (cumulative average strategy sᴵ for OOS, visit
counts for ISMCTS) is merged into a single global strategy structure that
accumulates across all 500 matches. Successor infoset data is also folded
in, weighted by visit count. Unvisited infosets default to uniform random.
Exploitability is then computed on this global strategy.

**What persists across matches — and what doesn't:** Each of the 500
matches creates a *fresh* bot instance with empty regret tables. The bot
runs its MCCFR iterations for that match's budget, accumulating regrets
and an average strategy (sᴵ) internally. When the match ends, only the
average strategy is extracted and merged into the global structure —
regrets are discarded. So:

- **Regrets:** Reset every match. No iterative refinement across matches.
- **Average strategy (sᴵ):** Accumulated into `global_strategy`. Each
  infoset's strategy is a weighted sum of contributions from every match
  that happened to visit it.

This means the aggregate is not one long MCCFR run broken into 500
segments — it's 500 independent short MCCFR runs whose average
strategies are stitched together.

**What "aggregate" actually means:** Each of the 500 matches has a
different deal, different match prefix, different targeting region. The
global strategy is a patchwork — each infoset's strategy comes from
whichever match(es) happened to visit it, weighted by how often. This
is *not* the strategy from any single match. It's an approximation of
"if this algorithm played many games, what overall strategy would
emerge across all the infosets it touched?"

**Why the random opponent doesn't matter:** The random opponent
determines the match history (deal + action sequence), which sets
the targeting region I(m). But within each match, MCCFR runs its own
internal self-play walks — the opponent's strategy during the match
doesn't affect the MCCFR update rule. The random opponent is just a
device for generating diverse match prefixes so that different regions
of the game tree get visited across the 500 matches.

**Why this is basically a convergence test:** Targeting helps OOS focus
computation on the current match prefix, which improves play locally
within that match. But across 500 matches with different deals, every
match targets a different region. The aggregate strategy converges for
the same reason plain MCCFR converges — enough matches eventually cover
the infoset space. The targeting just changes *which* infosets get
visited in each match, not the fundamental convergence mechanism. With
δ=0.5, half the iterations are untargeted MCCFR anyway. And since
regrets don't carry over, each match is independently "starting from
scratch" — there's no compounding benefit from match to match, just
broader infoset coverage.

**Result:** OOS beats ISMCTS in Goofspiel and Liar's Dice. In Poker,
ISMCTS is less exploitable at low budgets (more iterations to avoid
blunders outweighs NE convergence). All algorithms improve with more
computation time except ISMCTS-UCT in Goofspiel (gets worse).

**Our take:** This experiment validates that OOS doesn't degrade under
online play (targeting doesn't break convergence). It's not designed
to show that targeting *helps* — that's §4.4.3 (head-to-head matches),
where δ=0.9 and the benefit of focusing computation on the current
game state matters.

---

## Key Figures

### Exp 012 — Sampling Distributions by Decay Function

![Leduc Poker decay distributions](012_sampling_distribution/leduc_poker_decay.png)

![Liar's Dice decay distributions](012_sampling_distribution/liars_dice_decay.png)

### Exp 014 — Delta × Decay Ranking

![Bar ranking](014_postfix_delta_baseline/bar_ranking.png)

### Exp 015C — Per-Node Proximity Profiles (Table 3)

Each ISGT iteration targets infoset I₀ by sampling a terminal z\* at some
IIG level k (distance from I₀ in the information-set intersection graph).
Before and after the walk, we snapshot regret arrays across I₀'s entire
IIG neighborhood and measure the total |Δregret| (how much regret changed)
and touched count (how many infosets were actually updated). The **per-node
value** = total |Δregret| / touched count — this controls for the fact that
higher IIG levels have larger neighborhoods, isolating the regret work done
*per infoset*. Plotting this per-node value across levels 0–6 gives the
"profile." A declining profile means nearby z\* produces more useful regret
work per infoset than distant z\* — i.e., proximity in the IIG matters.

Full mode shows a declining gradient; chance mode is flat.

**full/const, sims=50** — proximity gradient visible (0.0064 → 0.0014):

| Lvl | early_pn | n_e | late_pn | n_l | all_pn | n_a | e−l diff |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 0.0062 | 3599 | 0.0065 | 3401 | 0.0064 | 17480 | -0.0003 |
| 1 | 0.0044 | 2177 | 0.0046 | 2223 | 0.0045 | 11170 | -0.0001 |
| 2 | 0.0040 | 1494 | 0.0042 | 1437 | 0.0044 | 7236 | -0.0002 |
| 3 | 0.0052 | 721 | 0.0045 | 716 | 0.0044 | 3651 | 0.0007 |
| 4 | 0.0046 | 269 | 0.0043 | 302 | 0.0043 | 1429 | 0.0002 |
| 5 | 0.0025 | 85 | 0.0032 | 79 | 0.0041 | 392 | -0.0007 |
| 6 | 0.0021 | 10 | 0.0024 | 13 | 0.0014 | 66 | -0.0002 |

**full/const, sims=5** — same gradient, fewer samples (0.0047 → 0.0019):

| Lvl | early_pn | n_e | late_pn | n_l | all_pn | n_a | e−l diff |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 0.0047 | 455 | 0.0047 | 440 | 0.0050 | 1816 | -0.0000 |
| 1 | 0.0038 | 184 | 0.0036 | 197 | 0.0038 | 1060 | 0.0002 |
| 2 | 0.0037 | 198 | 0.0047 | 173 | 0.0038 | 734 | -0.0010 |
| 3 | 0.0025 | 45 | 0.0039 | 47 | 0.0029 | 318 | -0.0013 |
| 4 | 0.0033 | 24 | 0.0029 | 23 | 0.0034 | 116 | 0.0004 |
| 5 | 0.0019 | 6 | 0.0015 | 6 | 0.0026 | 29 | 0.0004 |

**chance/const, sims=50** — flat profile for comparison (~0.25 at all levels):

| Lvl | early_pn | n_e | late_pn | n_l | all_pn | n_a | e−l diff |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 0.2533 | 3528 | 0.2651 | 3582 | 0.2558 | 17805 | -0.0118 |
| 1 | 0.2427 | 2264 | 0.2455 | 2259 | 0.2389 | 11191 | -0.0028 |
| 2 | 0.2489 | 1433 | 0.2431 | 1393 | 0.2397 | 7059 | 0.0058 |
| 3 | 0.2505 | 712 | 0.2724 | 777 | 0.2541 | 3649 | -0.0218 |
| 4 | 0.2354 | 334 | 0.2911 | 303 | 0.2729 | 1560 | -0.0558 |
| 5 | 0.3580 | 70 | 0.2041 | 77 | 0.2808 | 399 | 0.1539 |
| 6 | 0.3065 | 18 | 0.5837 | 19 | 0.3621 | 88 | -0.2771 |

### Exp 015C — Importance Weight: Chance vs Full (Table 7, sims=5)

Full mode l ≈ 0.5000 ± 0.0002 at every level. Chance mode l ≈ 0.04 ± 0.04.

| Lvl | ch/const | ±std | ch/tbal | ±std | fu/const | ±std | fu/tbal | ±std |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 0.0382 | 0.0395 | 0.0367 | 0.0391 | 0.5001 | 0.0002 | 0.5001 | 0.0002 |
| 1 | 0.0416 | 0.0441 | 0.0388 | 0.0408 | 0.5001 | 0.0003 | 0.5001 | 0.0003 |
| 2 | 0.0436 | 0.0466 | 0.0437 | 0.0441 | 0.5001 | 0.0003 | 0.5001 | 0.0002 |
| 3 | 0.0491 | 0.0509 | 0.0480 | 0.0489 | 0.5001 | 0.0001 | 0.5001 | 0.0002 |
| 4 | 0.0454 | 0.0479 | 0.0431 | 0.0457 | 0.5001 | 0.0003 | 0.5001 | 0.0002 |
| 5 | 0.0697 | 0.0613 | 0.0583 | 0.0597 | 0.5001 | 0.0002 | 0.5002 | 0.0007 |
| 6 | 0.0767 | 0.0455 | 0.0479 | 0.0345 | 0.5002 | 0.0001 | 0.5001 | 0.0001 |

### Exp 015C — Early-vs-Late Slope Summary (Table 8)

Pearson r of (IIG level, per-node regret). Negative r = proximity gradient exists. Full mode consistently negative; chance mode near zero or positive.

| Config | s=5 early | s=5 late | s=10 early | s=10 late | s=20 early | s=20 late | s=50 early | s=50 late |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| chance/const | 0.024 | 0.809 | -0.016 | 0.719 | 0.610 | 0.700 | 0.647 | 0.556 |
| chance/sqrt | 0.670 | 0.050 | 0.915 | -0.229 | 0.758 | 0.606 | 0.795 | 0.520 |
| chance/tbal | 0.500 | -0.804 | 0.293 | 0.614 | 0.857 | 0.839 | 0.928 | 0.760 |
| full/const | -0.826 | -0.692 | -0.219 | -0.688 | 0.382 | 0.526 | -0.837 | -0.906 |
| full/sqrt | -0.586 | -0.590 | 0.551 | -0.821 | -0.351 | -0.684 | -0.936 | -0.328 |
| full/tbal | -0.629 | 0.620 | -0.037 | -0.137 | -0.830 | -0.003 | 0.156 | 0.333 |