"""Explore how decay functions allocate simulation budget across IIG levels.

Under level-budget semantics, the decay function controls what fraction
of the simulation budget goes to each IIG level:
  P(level=ℓ) ∝ f(ℓ)   — fraction of budget at each level
  P(z | level=ℓ) = 1/|T_ℓ|  — uniform within level

Runs LevelUniform, LevelExponential, LevelPolynomial, LevelStep across
all benchmark games.  Produces:
  1. Per-level terminal structure table
  2. Theoretical vs empirical P(level) table + deviation
  3. Per-level budget ratio check (empirical vs f(ℓ)/f(ref))
  4. Matplotlib bar charts (saved to results/ if matplotlib available)

Usage:
    python -m experiments.explore_decay_budget
    python -m experiments.explore_decay_budget --game leduc_poker --num-sims 2000
    python -m experiments.explore_decay_budget --no-plots
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pyspiel

from src.games import load_game
from src.iig import IIG
from src.isgt import (
    LevelUniform,
    LevelExponential,
    LevelPolynomial,
    LevelStep,
)
from experiments.sampling_analysis_012 import (
    _AnalysisBot,
    _deepest_infoset,
    _iig_distance,
)


# ═════════════════════════════════════════════════════════════════════════
# Decay suite
# ═════════════════════════════════════════════════════════════════════════

DECAY_SUITE = [
    ("level_uniform",  LevelUniform()),
    ("level_exp(0.7)", LevelExponential(0.7)),
    ("level_exp(0.5)", LevelExponential(0.5)),
    ("level_exp(0.3)", LevelExponential(0.3)),
    ("level_step(0.01)", LevelStep(0.01)),
    ("level_poly(2)",  LevelPolynomial(2.0)),
]

ALL_GAMES = ["kuhn_poker", "leduc_poker", "goofspiel", "liars_dice"]


# ═════════════════════════════════════════════════════════════════════════
# Terminal-level assignment (mirrors _get_target_dist logic exactly)
# ═════════════════════════════════════════════════════════════════════════

def assign_terminal_levels(iig: IIG, active_iid: tuple) -> np.ndarray:
    """Assign each terminal to its closest upstream IIG level.

    Returns array of shape (num_terminals,) with level ints.
    -1 = floor (not in upstream neighborhood).
    """
    level_map = iig.levels(active_iid)
    num_t = iig.num_terminals
    terminal_level = np.full(num_t, -1, dtype=int)
    for tid in range(num_t):
        seq = iig._terminal_seqs[tid]
        if seq:
            best = None
            for iid in seq:
                if iid in level_map:
                    lvl = level_map[iid]
                    if best is None or lvl < best:
                        best = lvl
            if best is not None:
                terminal_level[tid] = best
    return terminal_level


def terminal_counts_by_level(terminal_level: np.ndarray) -> dict[int, int]:
    """Count terminals per level. Returns {level: count}."""
    counts: dict[int, int] = {}
    for lvl in terminal_level:
        lvl = int(lvl)
        counts[lvl] = counts.get(lvl, 0) + 1
    return dict(sorted(counts.items()))


# ═════════════════════════════════════════════════════════════════════════
# Theoretical distribution (new terminal-normalized semantics)
# ═════════════════════════════════════════════════════════════════════════

def compute_theoretical_dist(
    level_counts: dict[int, int],
    decay_fn,
    max_level: int,
) -> dict[int, float]:
    """Theoretical P(level) under level-budget two-stage sampling.

    Under level-budget semantics:
        P(level=ℓ) ∝ f(ℓ)
        P(z | level=ℓ) = 1/|T_ℓ|

    Floor terminals (level -1) are excluded.
    """
    level_weights: dict[int, float] = {}
    for lvl in level_counts:
        if lvl == -1:
            continue  # floor excluded
        w = decay_fn.weight(lvl, max_level)
        level_weights[lvl] = w

    total = sum(level_weights.values())
    if total == 0:
        return {}
    return {lvl: w / total for lvl, w in sorted(level_weights.items())}


def compute_theoretical_per_terminal(
    level_counts: dict[int, int],
    decay_fn,
    max_level: int,
) -> dict[int, float]:
    """Theoretical per-terminal probability at each level.

    Under level-budget semantics:
        P(z at level ℓ) = P(level=ℓ) / |T_ℓ| = f(ℓ) / (Σ_k f(k)) / |T_ℓ|

    Returns: {level: per_terminal_prob}
    """
    total_weight = 0.0
    for lvl in level_counts:
        if lvl == -1:
            continue
        total_weight += decay_fn.weight(lvl, max_level)

    if total_weight == 0:
        return {}

    result: dict[int, float] = {}
    for lvl, count in sorted(level_counts.items()):
        if lvl == -1:
            continue
        result[lvl] = decay_fn.weight(lvl, max_level) / (total_weight * count)
    return result


# ═════════════════════════════════════════════════════════════════════════
# Empirical sampling
# ═════════════════════════════════════════════════════════════════════════

def sample_empirical(
    iig: IIG,
    game: pyspiel.Game,
    active_iid: tuple,
    terminal_level: np.ndarray,
    decay_fn,
    num_sims: int = 2000,
    seed: int = 42,
) -> dict[int, int]:
    """Sample z* terminals and histogram by level.

    Only samples from the target distribution (no walk needed).
    Returns {level: count}.
    """
    bot = _AnalysisBot(
        game, player_id=0, num_simulations=1,
        epsilon=0.6, gamma=0.01,
        level_weight_fn=decay_fn,
        bias_mode="chance",
        seed=seed, iig=iig, delta=0.9,
    )

    iig_weights = bot._get_iig_weights(active_iid)
    target_dist = bot._get_target_dist(active_iid, iig_weights)

    rng = np.random.default_rng(seed)
    counts: dict[int, int] = Counter()
    for _ in range(num_sims):
        tid = rng.choice(len(target_dist), p=target_dist)
        lvl = int(terminal_level[tid])
        counts[lvl] += 1

    return dict(sorted(counts.items()))


# ═════════════════════════════════════════════════════════════════════════
# Verification: per-terminal ratio check
# ═════════════════════════════════════════════════════════════════════════

MIN_SAMPLES_FOR_RATIO = 30  # levels with fewer samples are marked unreliable


def verify_level_budget_ratio(
    empirical_counts: dict[int, int],
    level_counts: dict[int, int],
    decay_fn,
    max_level: int,
    num_sims: int,
) -> list[dict]:
    """Check that level budget ratios match f(ℓ)/f(ref).

    Under level-budget semantics, the fraction of samples at each level
    should be proportional to f(ℓ), regardless of terminal count.

    Normalizes to the level with the most samples (most reliable estimate).

    Returns list of dicts with level, expected_ratio, empirical_ratio, etc.
    """
    levels = sorted(l for l in level_counts if l >= 0)
    if not levels:
        return []

    # Per-level empirical rate = count / num_sims
    rates = {}
    for lvl in levels:
        cnt = empirical_counts.get(lvl, 0)
        rates[lvl] = cnt / num_sims

    # Find reference level: the one with the most empirical samples
    ref_lvl = max(levels, key=lambda l: empirical_counts.get(l, 0))
    ref_rate = rates[ref_lvl]
    if ref_rate == 0:
        return []
    f_ref = decay_fn.weight(ref_lvl, max_level)
    if f_ref == 0:
        return []

    results = []
    for lvl in levels:
        cnt = empirical_counts.get(lvl, 0)
        expected = decay_fn.weight(lvl, max_level) / f_ref
        empirical = rates[lvl] / ref_rate
        results.append({
            "level": lvl,
            "f_value": decay_fn.weight(lvl, max_level),
            "expected_ratio": expected,
            "empirical_ratio": empirical,
            "count": cnt,
            "n_terminals": level_counts[lvl],
            "reliable": cnt >= MIN_SAMPLES_FOR_RATIO,
            "is_ref": lvl == ref_lvl,
        })
    return results


# ═════════════════════════════════════════════════════════════════════════
# Display
# ═════════════════════════════════════════════════════════════════════════

def _lvl(d: int) -> str:
    return f"L{d}" if d >= 0 else "Floor"


def print_game_header(game_name, iig, active_iid, level_counts, max_level, num_sims):
    total_upstream = sum(c for l, c in level_counts.items() if l >= 0)
    print(f"\n{'═' * 78}")
    print(f"  {game_name.upper()}")
    print(f"  Active infoset: {iig._fmt_id(active_iid)}")
    print(f"  Terminals: {iig.num_terminals}, Upstream: {total_upstream}, "
          f"Max level: {max_level}")
    levels = sorted(l for l in level_counts if l >= 0)
    print(f"  Terminals per level: "
          + ", ".join(f"{_lvl(l)}={level_counts[l]}" for l in levels))
    floor = level_counts.get(-1, 0)
    if floor > 0:
        print(f"  Floor terminals (excluded from targeting): {floor}")
    # Warn about under-sampled levels (levels with tiny f(ℓ) get few samples)
    n_levels = len(levels)
    min_expected = num_sims / n_levels  # uniform: each level gets this many
    print(f"  Samples/level (uniform): ~{min_expected:.0f}")
    print(f"{'═' * 78}")


def print_distribution_tables(
    labels, level_counts, theoretical, empirical, num_sims
):
    levels = sorted(l for l in level_counts if l >= 0)
    col_w = max(10, max(len(lb) for lb in labels) + 1)

    # Theoretical P(level)
    print(f"\n  Theoretical P(level) — level-budget semantics:")
    hdr = f"  {'Level':>7s}  {'|T_ℓ|':>6s}"
    for lb in labels:
        hdr += f"  {lb:>{col_w}s}"
    print(hdr)
    print(f"  {'─' * 7}  {'─' * 6}" + f"  {'─' * col_w}" * len(labels))

    for d in levels:
        row = f"  {_lvl(d):>7s}  {level_counts[d]:>6d}"
        for lb in labels:
            p = theoretical[lb].get(d, 0.0)
            row += f"  {p*100:>{col_w - 1}.2f}%"
        print(row)

    # Empirical P(level)
    print(f"\n  Empirical P(level) — {num_sims} samples:")
    print(hdr)
    print(f"  {'─' * 7}  {'─' * 6}" + f"  {'─' * col_w}" * len(labels))

    for d in levels:
        row = f"  {_lvl(d):>7s}  {level_counts[d]:>6d}"
        for lb in labels:
            c = empirical[lb].get(d, 0)
            pct = 100 * c / num_sims
            row += f"  {pct:>{col_w - 1}.2f}%"
        print(row)

    # Deviation
    print(f"\n  Deviation (empirical − theoretical, pp):")
    max_dev = 0.0
    hdr2 = f"  {'Level':>7s}"
    for lb in labels:
        hdr2 += f"  {lb:>{col_w}s}"
    print(hdr2)
    print(f"  {'─' * 7}" + f"  {'─' * col_w}" * len(labels))

    for d in levels:
        row = f"  {_lvl(d):>7s}"
        for lb in labels:
            emp_pct = 100 * empirical[lb].get(d, 0) / num_sims
            theo_pct = 100 * theoretical[lb].get(d, 0.0)
            dev = emp_pct - theo_pct
            max_dev = max(max_dev, abs(dev))
            sign = "+" if dev >= 0 else ""
            row += f"  {sign}{dev:>{col_w - 2}.2f}%"
        print(row)

    expected_noise = 100 / num_sims ** 0.5
    print(f"\n  Max absolute deviation: {max_dev:.2f} pp "
          f"(expected noise ≈ {expected_noise:.1f} pp for N={num_sims})")


def print_ratio_table(label, ratios):
    """Print level-budget ratio verification for one decay function."""
    if not ratios:
        return
    ref = next((r for r in ratios if r.get("is_ref")), None)
    ref_str = f" (ref=L{ref['level']}, n={ref['count']})" if ref else ""
    print(f"\n  Level-budget ratio check [{label}]{ref_str}:")
    print(f"  {'Level':>7s}  {'f(ℓ)':>8s}  {'|T_ℓ|':>6s}  {'Expected':>10s}  "
          f"{'Empirical':>10s}  {'Δ':>8s}  {'n':>6s}")
    print(f"  {'─' * 7}  {'─' * 8}  {'─' * 6}  {'─' * 10}  {'─' * 10}  {'─' * 8}  {'─' * 6}")
    for r in ratios:
        delta = r["empirical_ratio"] - r["expected_ratio"]
        flag = " *" if not r["reliable"] else (" ←" if r.get("is_ref") else "  ")
        print(f"  {_lvl(r['level']):>7s}  {r['f_value']:>8.4f}  "
              f"{r['n_terminals']:>6d}  "
              f"{r['expected_ratio']:>10.4f}  {r['empirical_ratio']:>10.4f}  "
              f"{delta:>+8.4f}  {r['count']:>5d}{flag}")
    low_count = sum(1 for r in ratios if not r["reliable"])
    if low_count:
        print(f"  (* = fewer than {MIN_SAMPLES_FOR_RATIO} samples — ratio unreliable)")


# ═════════════════════════════════════════════════════════════════════════
# Matplotlib histograms
# ═════════════════════════════════════════════════════════════════════════

def make_plots(
    game_name: str,
    labels: list[str],
    level_counts: dict[int, int],
    theoretical: dict[str, dict[int, float]],
    empirical: dict[str, dict[int, int]],
    num_sims: int,
    out_dir: Path,
):
    """Create grouped bar charts comparing theoretical vs empirical."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("  [matplotlib not available — skipping plots]")
        return

    levels = sorted(l for l in level_counts if l >= 0)
    x = np.arange(len(levels))
    n_decay = len(labels)

    fig, axes = plt.subplots(2, (n_decay + 1) // 2, figsize=(5 * ((n_decay + 1) // 2), 8))
    axes = axes.flatten()

    for i, label in enumerate(labels):
        ax = axes[i]
        theo_vals = [theoretical[label].get(l, 0.0) * 100 for l in levels]
        emp_vals = [100 * empirical[label].get(l, 0) / num_sims for l in levels]

        w = 0.35
        ax.bar(x - w / 2, theo_vals, w, label="Theoretical", color="#4C72B0", alpha=0.8)
        ax.bar(x + w / 2, emp_vals, w, label="Empirical", color="#DD8452", alpha=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels([_lvl(l) for l in levels], fontsize=8)
        ax.set_ylabel("P(level) %", fontsize=8)
        ax.set_title(label, fontsize=10)
        ax.legend(fontsize=7)

    # Hide unused axes
    for i in range(n_decay, len(axes)):
        axes[i].set_visible(False)

    fig.suptitle(f"{game_name} — Level-Budget Decay Validation (N={num_sims})",
                 fontsize=12, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])

    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"decay_validation_{game_name}.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {path}")

    # Second plot: per-terminal probability (shows the "shape" directly)
    fig2, axes2 = plt.subplots(2, (n_decay + 1) // 2, figsize=(5 * ((n_decay + 1) // 2), 8))
    axes2 = axes2.flatten()

    for i, label in enumerate(labels):
        ax = axes2[i]
        # Per-terminal = P(level) / |T_ℓ|
        theo_per_t = []
        emp_per_t = []
        emp_err = []
        low_sample_levels = []
        for j, l in enumerate(levels):
            n_t = level_counts[l]
            cnt = empirical[label].get(l, 0)
            theo_per_t.append(theoretical[label].get(l, 0.0) / n_t * 1000)
            emp_per_t.append(cnt / (num_sims * n_t) * 1000)
            # Binomial stderr for per-terminal rate
            p_hat = cnt / num_sims if num_sims > 0 else 0
            se = (p_hat * (1 - p_hat) / num_sims) ** 0.5 if num_sims > 0 else 0
            emp_err.append(se / n_t * 1000)
            if cnt < MIN_SAMPLES_FOR_RATIO:
                low_sample_levels.append(j)

        w = 0.35
        ax.bar(x - w / 2, theo_per_t, w, label="Theoretical", color="#4C72B0", alpha=0.8)
        ax.bar(x + w / 2, emp_per_t, w, yerr=emp_err,
               label="Empirical", color="#DD8452", alpha=0.8,
               capsize=2, error_kw={"linewidth": 0.8})
        # Mark low-sample levels
        for j in low_sample_levels:
            ax.annotate("*", (x[j] + w / 2, emp_per_t[j]),
                        ha="center", va="bottom", fontsize=8, color="red")
        ax.set_xticks(x)
        ax.set_xticklabels([_lvl(l) for l in levels], fontsize=8)
        ax.set_ylabel("P(z) × 1000", fontsize=8)
        ax.set_title(f"{label} — per-terminal shape", fontsize=10)
        ax.legend(fontsize=7)

    for i in range(n_decay, len(axes2)):
        axes2[i].set_visible(False)

    fig2.suptitle(f"{game_name} — Per-Terminal Probability (context)",
                  fontsize=12, fontweight="bold")
    fig2.tight_layout(rect=[0, 0, 1, 0.95])

    path2 = out_dir / f"decay_shape_{game_name}.png"
    fig2.savefig(path2, dpi=150)
    plt.close(fig2)
    print(f"  Saved: {path2}")


# ═════════════════════════════════════════════════════════════════════════
# Key invariant checks (programmatic pass/fail)
# ═════════════════════════════════════════════════════════════════════════

def check_invariants(
    game_name: str,
    labels: list[str],
    level_counts: dict[int, int],
    theoretical: dict[str, dict[int, float]],
    empirical: dict[str, dict[int, int]],
    num_sims: int,
) -> list[str]:
    """Run invariant checks. Returns list of failure messages (empty = pass)."""
    failures = []
    levels = sorted(l for l in level_counts if l >= 0)

    for label in labels:
        # Check 1: LevelUniform should give equal P(level) across levels
        if label == "level_uniform":
            theo = theoretical[label]
            n_levels = len(levels)
            expected_frac = 1.0 / n_levels
            for lvl in levels:
                actual = theo.get(lvl, 0)
                if abs(actual - expected_frac) > 1e-10:
                    failures.append(
                        f"[{game_name}] level_uniform: theoretical P(L{lvl}) = "
                        f"{actual:.6f}, expected {expected_frac:.6f}")

        # Check 2: Empirical deviations within per-level noise tolerance
        # Use binomial std: σ = sqrt(p(1-p)/N), tolerance = 4σ
        for lvl in levels:
            emp_pct = empirical[label].get(lvl, 0) / num_sims
            theo_pct = theoretical[label].get(lvl, 0.0)
            sigma = (theo_pct * (1 - theo_pct) / num_sims) ** 0.5 if num_sims > 0 else 0
            tol = max(4 * sigma, 2.0 / num_sims)  # 4σ, minimum 2/N
            if abs(emp_pct - theo_pct) > tol:
                failures.append(
                    f"[{game_name}] {label}: L{lvl} deviation "
                    f"{abs(emp_pct - theo_pct)*100:.1f}pp > "
                    f"{tol*100:.1f}pp tolerance (4σ, n_exp="
                    f"{theo_pct * num_sims:.0f})")

        # Check 3: No floor terminals sampled
        floor_count = empirical[label].get(-1, 0)
        if floor_count > 0:
            failures.append(
                f"[{game_name}] {label}: {floor_count} floor terminals sampled!")

    return failures


# ═════════════════════════════════════════════════════════════════════════
# Main
# ═════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Explore how decay functions allocate budget across IIG levels")
    parser.add_argument("--game", nargs="*", default=None,
                        choices=ALL_GAMES,
                        help="Games to run. Default: all.")
    parser.add_argument("--num-sims", type=int, default=5000,
                        help="Samples per decay function (default: 5000)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-plots", action="store_true",
                        help="Skip matplotlib histograms")
    args = parser.parse_args()

    games = args.game if args.game else ALL_GAMES

    print("=" * 78)
    print("  Decay Budget Explorer")
    print(f"  Games: {', '.join(games)}")
    print(f"  Sims/config: {args.num_sims}, Seed: {args.seed}")
    print(f"  Decay functions: {len(DECAY_SUITE)}")
    print("=" * 78)

    all_failures = []

    for game_name in games:
        game = load_game(game_name)
        iig = IIG(game)
        active_iid = _deepest_infoset(iig)

        # Terminal structure
        terminal_level = assign_terminal_levels(iig, active_iid)
        level_counts = terminal_counts_by_level(terminal_level)
        level_sets = iig.level_sets(active_iid)
        max_level = max(level_sets) if level_sets else 0

        print_game_header(game_name, iig, active_iid, level_counts, max_level,
                          args.num_sims)

        labels = [lb for lb, _ in DECAY_SUITE]
        theoretical = {}
        empirical = {}

        for label, decay_fn in DECAY_SUITE:
            theoretical[label] = compute_theoretical_dist(
                level_counts, decay_fn, max_level)
            empirical[label] = sample_empirical(
                iig, game, active_iid, terminal_level, decay_fn,
                num_sims=args.num_sims, seed=args.seed)

        # Distribution tables
        print_distribution_tables(
            labels, level_counts, theoretical, empirical, args.num_sims)

        # Per-level budget ratio check for non-uniform decays
        for label, decay_fn in DECAY_SUITE:
            if label == "level_uniform":
                continue
            ratios = verify_level_budget_ratio(
                empirical[label], level_counts, decay_fn,
                max_level, args.num_sims)
            print_ratio_table(label, ratios)

        # Invariant checks
        failures = check_invariants(
            game_name, labels, level_counts,
            theoretical, empirical, args.num_sims)
        all_failures.extend(failures)

        if failures:
            print(f"\n  FAILURES ({len(failures)}):")
            for f in failures:
                print(f"    ✗ {f}")
        else:
            print(f"\n  All invariant checks passed.")

        # Plots
        if not args.no_plots:
            out_dir = Path("results") / "decay_validation"
            make_plots(game_name, labels, level_counts,
                       theoretical, empirical, args.num_sims, out_dir)

    # Summary
    print(f"\n{'=' * 78}")
    if all_failures:
        print(f"  FAILED — {len(all_failures)} invariant violations:")
        for f in all_failures:
            print(f"    ✗ {f}")
    else:
        print(f"  ALL PASSED — level-budget semantics verified across "
              f"{len(games)} games × {len(DECAY_SUITE)} decay functions")
    print(f"{'=' * 78}\n")

    return 1 if all_failures else 0


if __name__ == "__main__":
    sys.exit(main())
