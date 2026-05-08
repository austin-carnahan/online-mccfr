#!/usr/bin/env python3
"""Print ISGT ablation results in nicely formatted tables.

Usage:
    python -m experiments.print_ablation [path/to/ablation_summary.json]

Defaults to results/isgt_ablation_full/ablation_summary.json
"""

import json
import sys

GAME_ORDER = ["kuhn_poker", "leduc_poker", "liars_dice", "goofspiel"]

GAME_DISPLAY = {
    "kuhn_poker": "Kuhn Poker",
    "leduc_poker": "Leduc Poker",
    "liars_dice": "Liar's Dice",
    "goofspiel": "Goofspiel",
}


def load_data(path):
    with open(path) as f:
        return json.load(f)


def print_game_table(game_name, configs, budgets):
    """Print a single game's results as a ranked table."""
    display = GAME_DISPLAY.get(game_name, game_name)

    # Build rows: (config_name, {budget: expl})
    rows = []
    for config, results in configs.items():
        budget_map = {r["sims_per_move"]: r["exploitability"] for r in results}
        rows.append((config, budget_map))

    # Sort by exploitability at the highest budget (best first)
    max_budget = max(budgets)
    rows.sort(key=lambda r: r[1].get(max_budget, 999))

    # Find best value per column for highlighting
    best_per_budget = {}
    for b in budgets:
        vals = [r[1].get(b, 999) for r in rows]
        best_per_budget[b] = min(vals)

    # Header
    budget_header = "".join(f"{'sims=' + str(b):>14s}" for b in budgets)
    title = f"  {display}"
    width = 28 + 14 * len(budgets)
    print()
    print("=" * width)
    print(title)
    print("-" * width)
    print(f"  {'#':>3s}  {'Config':<22s}{budget_header}")
    print("-" * width)

    # Rows
    for rank, (config, budget_map) in enumerate(rows, 1):
        # Determine mode tag
        if config == "oos":
            tag = "BASE"
        elif config.startswith("chance_"):
            tag = " C  "
        elif config.startswith("full_"):
            tag = " F  "
        else:
            tag = "    "

        vals = ""
        for b in budgets:
            v = budget_map.get(b)
            if v is None:
                vals += f"{'---':>14s}"
            elif abs(v - best_per_budget[b]) < 1e-9:
                vals += f"  {v:>10.6f} *"
            else:
                vals += f"  {v:>10.6f}  "

        print(f"  {rank:>3d}  {config:<22s}{vals}")

    print("=" * width)


def print_mode_comparison(summary, budgets):
    """Print a condensed mode-vs-mode comparison across games."""
    width = 28 + 14 * len(budgets)
    budget_header = "".join(f"{'sims=' + str(b):>14s}" for b in budgets)

    print()
    print()
    print("=" * width)
    print("  MODE COMPARISON: Best per mode at each budget")
    print("-" * width)
    print(f"  {'Game':<15s} {'Mode':<10s}{budget_header}")
    print("-" * width)

    for game in GAME_ORDER:
        if game not in summary:
            continue
        configs = summary[game]
        display = GAME_DISPLAY.get(game, game)

        # Group by mode
        modes = {"chance": [], "full": [], "oos": []}
        for config, results in configs.items():
            budget_map = {r["sims_per_move"]: r["exploitability"] for r in results}
            if config == "oos":
                modes["oos"].append(budget_map)
            elif config.startswith("chance_"):
                modes["chance"].append(budget_map)
            elif config.startswith("full_"):
                modes["full"].append(budget_map)

        for mode_name in ["chance", "full", "oos"]:
            if not modes[mode_name]:
                continue
            vals = ""
            for b in budgets:
                best = min(m.get(b, 999) for m in modes[mode_name])
                vals += f"  {best:>10.6f}  "
            label = f"{display}" if mode_name == "chance" else ""
            print(f"  {label:<15s} {mode_name:<10s}{vals}")
        print()

    print("=" * width)


def print_convergence_rates(summary, budgets):
    """Print improvement ratio from lowest to highest budget."""
    if len(budgets) < 2:
        return

    lo, hi = min(budgets), max(budgets)
    width = 50

    print()
    print()
    print("=" * width)
    print(f"  CONVERGENCE: Improvement ratio sims={lo} → {hi}")
    print("-" * width)
    print(f"  {'Game':<15s} {'Config':<22s} {'Ratio':>8s}")
    print("-" * width)

    for game in GAME_ORDER:
        if game not in summary:
            continue
        display = GAME_DISPLAY.get(game, game)

        rows = []
        for config, results in summary[game].items():
            budget_map = {r["sims_per_move"]: r["exploitability"] for r in results}
            v_lo = budget_map.get(lo)
            v_hi = budget_map.get(hi)
            if v_lo and v_hi and v_lo > 0:
                ratio = v_lo / v_hi
                rows.append((config, ratio))

        rows.sort(key=lambda r: -r[1])  # highest improvement first
        for config, ratio in rows:
            label = display if config == rows[0][0] else ""
            print(f"  {label:<15s} {config:<22s} {ratio:>7.2f}x")
        print()

    print("=" * width)


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "results/isgt_ablation_full/ablation_summary.json"

    try:
        summary = load_data(path)
    except FileNotFoundError:
        print(f"File not found: {path}")
        print("Usage: python -m experiments.print_ablation [path/to/ablation_summary.json]")
        sys.exit(1)

    # Determine budgets from data
    budgets = set()
    for game_data in summary.values():
        for results in game_data.values():
            for r in results:
                budgets.add(r["sims_per_move"])
    budgets = sorted(budgets)

    # Print per-game tables (ranked by best at highest budget)
    for game in GAME_ORDER:
        if game in summary:
            print_game_table(game, summary[game], budgets)

    # Print cross-game mode comparison
    print_mode_comparison(summary, budgets)

    # Print convergence rates
    print_convergence_rates(summary, budgets)


if __name__ == "__main__":
    main()
