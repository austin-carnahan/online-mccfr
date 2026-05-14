# LOTR / OOS Parity Archive

Date: 2026-05-13

This archive collects the result artifacts for the point where the current
LOTR implementation reached Kuhn parity with OOS, then survived a short
cross-game sanity check on Leduc, Goofspiel, and Liar's Dice.

## Archived Artifacts

- `mixture_lotr_illegal_fix_medium/`: Kuhn-only parity check after the
  illegal observable chance target fix. The result directory still uses the
  old `mixture_step_r05` label, but this is the implementation that was later
  renamed to canonical LOTR.
- `lotr_step_vs_oos_sanity/`: Cross-game sanity sweep with canonical
  `lotr_step_r05` naming.

Each result directory contains:

- `incremental_results.jsonl`
- `results.json`
- `summary.csv`
- `summary.md`

## Kuhn Parity Check

Source: `mixture_lotr_illegal_fix_medium/summary.md`

Configuration:

- Game: `kuhn_poker`
- Algorithms: OOS `delta=0.5` vs LOTR `step(0.5, depth=0)`
- Seeds: 30 paired seeds
- Matches per seed: 100
- Budgets: 5k, 10k, 20k, 50k simulations per move

The paired differences are `LOTR - OOS` exploitability. Negative means LOTR
had lower exploitability for that paired seed.

| sims | LOTR mean | OOS mean | mean diff | mean 95% CI | LOTR better | OOS better |
|---:|---:|---:|---:|---:|---:|---:|
| 5,000 | 0.0373917 | 0.0379142 | -0.00052254 | [-0.00180029, 0.000692664] | 14 | 16 |
| 10,000 | 0.0248779 | 0.0250608 | -0.000182898 | [-0.00147086, 0.00104255] | 13 | 17 |
| 20,000 | 0.0163434 | 0.0166011 | -0.000257702 | [-0.000908051, 0.000407489] | 16 | 14 |
| 50,000 | 0.00977609 | 0.00999031 | -0.000214213 | [-0.000631883, 0.000162592] | 16 | 14 |

Takeaway: LOTR and OOS track each other closely in Kuhn after the fix. The
mean paired differences are small, the confidence intervals include zero at
all budgets, and neither algorithm consistently dominates by seed count.

## Cross-Game Sanity Sweep

Source: `lotr_step_vs_oos_sanity/summary.md`

Configuration:

- Games: `leduc_poker`, `goofspiel`, `liars_dice`
- Algorithms: OOS `delta=0.5` vs LOTR `step(0.5, depth=0)`
- Seeds: 3 paired seeds
- Matches per seed: 100 for Leduc and Goofspiel, 50 for Liar's Dice
- Budgets: 5k, 10k, 20k, 50k simulations per move
- Tracking: enabled

### Paired Differences

The paired differences are `LOTR - OOS` exploitability. Negative means LOTR
had lower exploitability for that paired seed.

| game | 5k | 10k | 20k | 50k | 50k seed split |
|---|---:|---:|---:|---:|---:|
| goofspiel | -0.000177 | -0.000264 | 0.000465 | 0.000169 | 0 / 3 LOTR better |
| leduc_poker | -0.016724 | -0.008066 | 0.002838 | 0.000255 | 2 / 3 LOTR better |
| liars_dice | 0.007366 | 0.002893 | 0.001727 | 0.010921 | 1 / 3 LOTR better |

Takeaway: Leduc and Goofspiel look consistent with the Kuhn parity story.
Liar's Dice shows a small OOS-favoring mean gap, especially at 50k, but with
only 3 seeds and 50 matches per seed this is likely dominated by statistical
noise rather than a clean algorithmic separation.

### High-Budget Tracking Snapshot

| game | LOTR mean | OOS mean | mean diff | LOTR coverage | OOS coverage | LOTR ESS/sim | OOS ESS/sim | LOTR prefix survival | OOS prefix survival |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| goofspiel | 0.0170646 | 0.0168956 | 0.000169 | 1.000 | 1.000 | 0.000789826 | 0.000795661 | 0.583270 | 0.626615 |
| leduc_poker | 0.544623 | 0.544369 | 0.000255 | 1.000 | 1.000 | 0.00118561 | 0.00122179 | 0.500061 | 0.545718 |
| liars_dice | 0.403409 | 0.392488 | 0.010921 | 0.496297 | 0.498793 | 0.000548621 | 0.000522254 | 0.499923 | 0.525978 |

Tracking interpretation:

- Coverage is complete for Leduc and Goofspiel. Liar's Dice coverage reaches
  about 50 percent at 50k for both algorithms, which is expected for the larger
  game and only 50 matches per seed.
- ESS/sim is low for both algorithms in Liar's Dice, but not clearly worse for
  LOTR.
- The prefix survival mismatch is expected. LOTR's sticky-by-coin tracking is
  close to the forced coin survival rate, while OOS can remain on prefix during
  untargeted iterations when the natural sample happens to match the target.

## Overall Conclusion

The current LOTR implementation matches OOS behavior in Kuhn under the
`step(0.5, depth=0)` schedule and remains well behaved on Leduc, Goofspiel,
and Liar's Dice. The non-Kuhn sanity run did not reveal a special-case failure
in the new IST observability machinery. Liar's Dice remains the noisiest game
and should be treated cautiously unless future runs use more seeds or more
matches per seed.
