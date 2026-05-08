"""013 — Two-Stage Decay Histograms.

Validates that two-stage level-first sampling gives decay functions their
intended semantics: the empirical z* level distribution should match the
theoretical (weight-derived) distribution, independent of per-level
terminal counts.

Runs all 4 benchmark games × 10 decay functions.

Usage:
    python -m experiments.twostage_histograms_013
    python -m experiments.twostage_histograms_013 --game liars_dice
    python -m experiments.twostage_histograms_013 --json results.json
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter

import numpy as np
import pyspiel

from src.games import load_game
from src.iig import IIG
from src.isgt import (
    ISGTBot,
    ConstantWeight,
    ExponentialDecay,
    PolynomialDecay,
    StepFunction,
    TerminalBalancedWeight,
)
from experiments.sampling_analysis_012 import (
    Config,
    _AnalysisBot,
    _deepest_infoset,
    _iig_distance,
)


# ═════════════════════════════════════════════════════════════════════════
# Decay function suite
# ═════════════════════════════════════════════════════════════════════════

DECAY_SUITE = [
    ("t_balanced",  TerminalBalancedWeight()),
    ("constant",    ConstantWeight()),
    ("exp(0.7)",    ExponentialDecay(0.7)),
    ("exp(0.5)",    ExponentialDecay(0.5)),
    ("exp(0.3)",    ExponentialDecay(0.3)),
    ("exp(0.1)",    ExponentialDecay(0.1)),
    ("step(0.25)",  StepFunction(0.25)),
    ("step(0.1)",   StepFunction(0.1)),
    ("step(0.01)",  StepFunction(0.01)),
    ("poly(2)",     PolynomialDecay(2.0)),
]

ALL_GAMES = ["kuhn_poker", "leduc_poker", "goofspiel", "liars_dice"]


# ═════════════════════════════════════════════════════════════════════════
# Theoretical level distribution
# ═════════════════════════════════════════════════════════════════════════

def compute_theoretical_dist(
    iig: IIG,
    active_iid: tuple,
    decay_fn,
) -> dict[int, float]:
    """Compute the theoretical P(level) under two-stage sampling.

    Returns {level: probability} including the floor group (-1).
    """
    level_map = iig.levels(active_iid)
    level_sets = iig.level_sets(active_iid)
    max_level = max(level_sets) if level_sets else 0
    num_t = iig.num_terminals

    # Assign terminals to levels (same logic as _get_target_dist)
    level_counts: dict[int, int] = {}
    for tid in range(num_t):
        seq = iig._terminal_seqs[tid]
        best = None
        if seq:
            for iid in seq:
                if iid in level_map:
                    lvl = level_map[iid]
                    if best is None or lvl < best:
                        best = lvl
        lvl = best if best is not None else -1
        level_counts[lvl] = level_counts.get(lvl, 0) + 1

    # Compute level weights
    level_weights: dict[int, float] = {}
    for lvl in level_counts:
        if isinstance(decay_fn, TerminalBalancedWeight):
            # w(ℓ) = |T_ℓ| → uniform P(z) = 1/N
            level_weights[lvl] = float(level_counts[lvl])
        elif lvl == -1:
            w = decay_fn.weight(max_level + 1, max_level)
            level_weights[lvl] = max(w, 1e-6)
        else:
            level_weights[lvl] = decay_fn.weight(lvl, max_level)

    total_w = sum(level_weights.values())
    return {lvl: level_weights[lvl] / total_w for lvl in sorted(level_weights)}


def compute_level_terminal_counts(iig: IIG, active_iid: tuple) -> dict[int, int]:
    """Count terminals per IIG level. Returns {level: count}."""
    level_map = iig.levels(active_iid)
    num_t = iig.num_terminals
    counts: dict[int, int] = {}
    for tid in range(num_t):
        seq = iig._terminal_seqs[tid]
        best = None
        if seq:
            for iid in seq:
                if iid in level_map:
                    lvl = level_map[iid]
                    if best is None or lvl < best:
                        best = lvl
        lvl = best if best is not None else -1
        counts[lvl] = counts.get(lvl, 0) + 1
    return dict(sorted(counts.items()))


# ═════════════════════════════════════════════════════════════════════════
# Empirical sampling
# ═════════════════════════════════════════════════════════════════════════

def sample_z_star_histogram(
    iig: IIG,
    game: pyspiel.Game,
    active_iid: tuple,
    decay_fn,
    num_sims: int = 500,
    delta: float = 0.9,
    seed: int = 42,
) -> dict[int, int]:
    """Sample z* terminals and histogram their IIG levels.

    Only counts targeted iterations (which use the IIG-weighted distribution).
    Returns {level: count}.
    """
    level_map = iig.levels(active_iid)

    bot = _AnalysisBot(
        game, player_id=0, num_simulations=1,
        epsilon=0.6, gamma=0.01,
        level_weight_fn=decay_fn,
        bias_mode="chance",
        seed=seed, iig=iig, delta=delta,
    )

    iig_weights = bot._get_iig_weights(active_iid)
    target_dist = bot._get_target_dist(active_iid, iig_weights)

    counts: dict[int, int] = Counter()
    rng = np.random.default_rng(seed)

    for _ in range(num_sims):
        # Only sample z* — we don't need to run the full walk
        tid = rng.choice(len(target_dist), p=target_dist)
        d = _iig_distance(iig, level_map, tid)
        counts[d] += 1

    return dict(sorted(counts.items()))


# ═════════════════════════════════════════════════════════════════════════
# Display
# ═════════════════════════════════════════════════════════════════════════

def _lvl(d: int) -> str:
    return f"L{d}" if d >= 0 else "Floor"


def print_game_results(
    game_name: str,
    iig: IIG,
    active_iid: tuple,
    level_counts: dict[int, int],
    theoretical: dict[str, dict[int, float]],
    empirical: dict[str, dict[int, int]],
    num_sims: int,
):
    """Print results for one game: structure, theoretical, empirical."""
    level_map = iig.levels(active_iid)
    level_sets = iig.level_sets(active_iid)
    max_level = max(level_sets) if level_sets else 0
    all_levels = sorted(level_counts.keys())

    print(f"\n{'═' * 78}")
    print(f"  {game_name.upper()}")
    print(f"  Active infoset: {iig._fmt_id(active_iid)}")
    print(f"  Terminals: {iig.num_terminals}, Infosets: {iig.num_infosets}, "
          f"Max level: {max_level}")
    print(f"  Terminals per level: "
          + ", ".join(f"{_lvl(l)}={level_counts[l]}" for l in all_levels))
    print(f"{'═' * 78}")

    # Column layout
    labels = list(theoretical.keys())
    col_w = max(10, max(len(lb) for lb in labels) + 1)

    # ── Theoretical ──
    print(f"\n  Theoretical P(level) under two-stage sampling:")
    hdr = f"  {'Level':>7s}"
    for lb in labels:
        hdr += f"  {lb:>{col_w}s}"
    print(hdr)
    print(f"  {'─' * 7}" + f"  {'─' * col_w}" * len(labels))

    for d in all_levels:
        row = f"  {_lvl(d):>7s}"
        for lb in labels:
            p = theoretical[lb].get(d, 0.0)
            row += f"  {p*100:>{col_w - 1}.1f}%"
        print(row)

    # ── Empirical ──
    print(f"\n  Empirical z* distribution ({num_sims} samples, targeted only):")
    print(hdr)
    print(f"  {'─' * 7}" + f"  {'─' * col_w}" * len(labels))

    for d in all_levels:
        row = f"  {_lvl(d):>7s}"
        for lb in labels:
            c = empirical[lb].get(d, 0)
            pct = 100 * c / num_sims if num_sims > 0 else 0
            row += f"  {pct:>{col_w - 1}.1f}%"
        print(row)

    # ── Deviation (empirical – theoretical) ──
    print(f"\n  Deviation (empirical − theoretical, pp):")
    print(hdr)
    print(f"  {'─' * 7}" + f"  {'─' * col_w}" * len(labels))

    max_dev = 0.0
    for d in all_levels:
        row = f"  {_lvl(d):>7s}"
        for lb in labels:
            emp_pct = 100 * empirical[lb].get(d, 0) / num_sims if num_sims > 0 else 0
            theo_pct = 100 * theoretical[lb].get(d, 0.0)
            dev = emp_pct - theo_pct
            max_dev = max(max_dev, abs(dev))
            sign = "+" if dev >= 0 else ""
            row += f"  {sign}{dev:>{col_w - 2}.1f}%"
        print(row)

    print(f"\n  Max absolute deviation: {max_dev:.1f} pp "
          f"(expected ≈ {100 / num_sims**0.5:.1f} pp for N={num_sims})")


# ═════════════════════════════════════════════════════════════════════════
# Main
# ═════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="013 — Two-Stage Decay Histograms")
    parser.add_argument("--game", nargs="*", default=None,
                        choices=ALL_GAMES,
                        help="Games to run. Default: all.")
    parser.add_argument("--num-sims", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--json", default=None)
    args = parser.parse_args()

    games = args.game if args.game else ALL_GAMES

    print("=" * 78)
    print("  013 — Two-Stage Decay Histograms")
    print(f"  Games: {', '.join(games)}")
    print(f"  Sims/config: {args.num_sims}, Seed: {args.seed}")
    print(f"  Decay functions: {len(DECAY_SUITE)}")
    print("=" * 78)

    all_json = {}

    for game_name in games:
        game = load_game(game_name)
        iig = IIG(game)
        active_iid = _deepest_infoset(iig)

        level_counts = compute_level_terminal_counts(iig, active_iid)

        theoretical = {}
        empirical = {}

        for label, decay_fn in DECAY_SUITE:
            # Theoretical
            theoretical[label] = compute_theoretical_dist(
                iig, active_iid, decay_fn)

            # Empirical
            empirical[label] = sample_z_star_histogram(
                iig, game, active_iid, decay_fn,
                num_sims=args.num_sims, seed=args.seed)

        print_game_results(
            game_name, iig, active_iid, level_counts,
            theoretical, empirical, args.num_sims)

        if args.json:
            all_json[game_name] = {
                "active_infoset": iig._fmt_id(active_iid),
                "num_terminals": iig.num_terminals,
                "num_infosets": iig.num_infosets,
                "terminals_per_level": {str(k): v for k, v in level_counts.items()},
                "theoretical": {lb: {str(k): v for k, v in d.items()}
                                for lb, d in theoretical.items()},
                "empirical": {lb: {str(k): v for k, v in d.items()}
                              for lb, d in empirical.items()},
            }

    if args.json:
        with open(args.json, "w") as f:
            json.dump(all_json, f, indent=2)
        print(f"\n  Results exported to {args.json}")

    print()


if __name__ == "__main__":
    main()
