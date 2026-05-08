"""Plot experiment 015A results — mechanism isolation.

Generates figures and prints tables from 015a results.

Figures (paper §4.4.2 style — aggregate exploitability vs sims/move):

  Figure 1 — Aggregate Exploitability (1×4 grid, one per game)
    All 5 configs on same axes.  Paper-style: x=sims/move, y=exploitability.

  Figure 2 — Aggregate Exploitability by Mode (2×4 grid)
    Row 1: chance configs + OOS.  Row 2: full configs + OOS.
    Separates the two targeting mechanisms for clarity.

  Figure 3 — Best ISGT Config vs OOS (1×4 grid)
    Per game, picks best-performing ISGT config at highest budget.

  Figure 4 — Coverage by Config (1×4, Liar's Dice is the main story)
    Mean of P0/P1 coverage at each budget, all configs.

Tables (printed to stdout):

  Table 1 — Exploitability summary (all games × configs × budgets)
  Table 2 — Active regret by IIG level (per game × config, sims=1000)
  Table 3 — Neighborhood regret by IIG level (per game × config, sims=1000)
  Table 4 — Neighborhood regret per node (normalized by touched count)
  Table 5 — Coverage (Liar's Dice focus, all budgets)

Usage:
    python -m experiments.plot_mechanism_isolation_015a
    python -m experiments.plot_mechanism_isolation_015a --results path/to/dir
    python -m experiments.plot_mechanism_isolation_015a --tables-only
    python -m experiments.plot_mechanism_isolation_015a --plots-only
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

# ── Paths ────────────────────────────────────────────────────────────────

DEFAULT_RESULTS_DIR = Path("results/015a_mechanism_isolation_full")
OUTPUT_DIR = Path("results/plots/015a_mechanism_isolation")

GAMES = ["kuhn_poker", "leduc_poker", "goofspiel", "liars_dice"]
GAME_LABELS = {
    "kuhn_poker": "Kuhn Poker",
    "leduc_poker": "Leduc Poker",
    "goofspiel": "Goofspiel",
    "liars_dice": "Liar's Dice",
}

# Configs in the order we want them plotted/printed
CONFIG_ORDER = [
    "oos",
    "isgt_chance_constant",
    "isgt_chance_tbal",
    "isgt_full_constant",
    "isgt_full_tbal",
]

CONFIG_LABELS = {
    "oos": "OOS",
    "isgt_chance_constant": "ISGT chance const",
    "isgt_chance_tbal": "ISGT chance t_bal",
    "isgt_full_constant": "ISGT full const",
    "isgt_full_tbal": "ISGT full t_bal",
}

CONFIG_SHORT = {
    "oos": "OOS",
    "isgt_chance_constant": "ch_const",
    "isgt_chance_tbal": "ch_tbal",
    "isgt_full_constant": "fl_const",
    "isgt_full_tbal": "fl_tbal",
}

# ── Visual style ─────────────────────────────────────────────────────────

CONFIG_COLORS = {
    "oos":                   "#dc2626",  # red
    "isgt_chance_constant":  "#16a34a",  # green
    "isgt_chance_tbal":      "#65a30d",  # lime
    "isgt_full_constant":    "#7c3aed",  # violet
    "isgt_full_tbal":        "#a855f7",  # purple
}

CONFIG_MARKERS = {
    "oos": "*",
    "isgt_chance_constant": "o",
    "isgt_chance_tbal": "s",
    "isgt_full_constant": "^",
    "isgt_full_tbal": "D",
}

CONFIG_LINESTYLES = {
    "oos": "--",
    "isgt_chance_constant": "-",
    "isgt_chance_tbal": (0, (3, 1, 1, 1)),
    "isgt_full_constant": "-",
    "isgt_full_tbal": (0, (3, 1, 1, 1)),
}

CONFIG_LINEWIDTHS = {
    "oos": 2.2,
    "isgt_chance_constant": 1.8,
    "isgt_chance_tbal": 1.8,
    "isgt_full_constant": 1.8,
    "isgt_full_tbal": 1.8,
}

CONFIG_ZORDER = {
    "oos": 10,
    "isgt_chance_constant": 5,
    "isgt_chance_tbal": 5,
    "isgt_full_constant": 5,
    "isgt_full_tbal": 5,
}


# ── Data loading ─────────────────────────────────────────────────────────

def load_summary(results_dir):
    path = Path(results_dir) / "ablation_summary.json"
    with open(path) as f:
        return json.load(f)


def load_detailed(results_dir):
    path = Path(results_dir) / "ablation_results.json"
    with open(path) as f:
        return json.load(f)


def extract_series(summary, game, config):
    """Pull sims and exploitability arrays."""
    series = summary.get(game, {}).get(config, [])
    sims = [d["sims_per_move"] for d in series]
    expl = [d["exploitability"] for d in series]
    cov = [d.get("infoset_coverage", [1.0, 1.0]) for d in series]
    return sims, expl, cov


def get_detailed_entry(detailed, game, config, sims):
    """Find a specific entry in detailed results."""
    for entry in detailed:
        if (entry["game"] == game and entry["config"] == config
                and entry["sims_per_move"] == sims):
            return entry
    return None


# ── Plot helpers ─────────────────────────────────────────────────────────

def format_axes(ax, x_label=True):
    ax.set_xscale("log")
    ax.set_xticks([100, 250, 500, 1000])
    ax.get_xaxis().set_major_formatter(ticker.ScalarFormatter())
    ax.tick_params(axis="both", which="both", labelsize=9)
    ax.grid(True, alpha=0.25, linewidth=0.5)
    if x_label:
        ax.set_xlabel("Sims / move", fontsize=10)


def plot_config(ax, summary, game, config, label=None):
    sims, expl, _ = extract_series(summary, game, config)
    if not sims:
        return
    ax.plot(sims, expl,
            color=CONFIG_COLORS[config],
            linestyle=CONFIG_LINESTYLES[config],
            marker=CONFIG_MARKERS[config],
            markersize=7 if config == "oos" else 5,
            linewidth=CONFIG_LINEWIDTHS[config],
            zorder=CONFIG_ZORDER[config],
            label=label or CONFIG_LABELS[config])


# ════════════════════════════════════════════════════════════════════════
# Figure 1: All configs on one panel per game (1×4)
# ════════════════════════════════════════════════════════════════════════

def fig_exploitability_all(summary):
    """Paper §4.4.2 style — one panel per game, all 5 configs."""
    fig, axes = plt.subplots(1, 4, figsize=(20, 4.5), sharey=False)

    for col, game in enumerate(GAMES):
        ax = axes[col]
        for config in CONFIG_ORDER:
            plot_config(ax, summary, game, config)
        format_axes(ax)
        ax.set_title(GAME_LABELS[game], fontsize=12, fontweight="bold")
        if col == 0:
            ax.set_ylabel("Exploitability", fontsize=10)
        if col == 0:
            ax.legend(fontsize=7, loc="upper right",
                      framealpha=0.9, edgecolor="0.8")

    fig.suptitle("Experiment 15A — Mechanism Isolation: Aggregate Exploitability\n"
                 "(δ=0.5, ε=0.4, γ=0.01)",
                 fontsize=13, fontweight="bold", y=1.04)
    fig.tight_layout()
    return fig


# ════════════════════════════════════════════════════════════════════════
# Figure 2: Split by mode (2×4)
# ════════════════════════════════════════════════════════════════════════

def fig_exploitability_by_mode(summary):
    """2×4: row 1 = chance configs + OOS, row 2 = full configs + OOS."""
    fig, axes = plt.subplots(2, 4, figsize=(20, 8), sharey=False)

    mode_configs = {
        "chance": ["oos", "isgt_chance_constant", "isgt_chance_tbal"],
        "full": ["oos", "isgt_full_constant", "isgt_full_tbal"],
    }

    for col, game in enumerate(GAMES):
        for row, mode in enumerate(["chance", "full"]):
            ax = axes[row, col]
            for config in mode_configs[mode]:
                plot_config(ax, summary, game, config)
            format_axes(ax, x_label=(row == 1))
            if row == 0:
                ax.set_title(GAME_LABELS[game], fontsize=12, fontweight="bold")
                ax.set_xlabel("")
            if col == 0:
                mode_label = "Chance" if mode == "chance" else "Full"
                ax.set_ylabel(f"{mode_label} mode\nExploitability", fontsize=10)
            if row == 0 and col == 0:
                ax.legend(fontsize=7, loc="upper right",
                          framealpha=0.9, edgecolor="0.8")

    fig.suptitle("Experiment 15A — Exploitability by Targeting Mode\n"
                 "(δ=0.5, ε=0.4, γ=0.01)",
                 fontsize=13, fontweight="bold", y=1.02)
    fig.tight_layout()
    return fig


# ════════════════════════════════════════════════════════════════════════
# Figure 3: Best ISGT vs OOS (1×4)
# ════════════════════════════════════════════════════════════════════════

def fig_best_vs_oos(summary):
    """Per game, pick best-performing ISGT at sims=1000 and plot vs OOS."""
    fig, axes = plt.subplots(1, 4, figsize=(20, 4.5), sharey=False)

    isgt_configs = [c for c in CONFIG_ORDER if c != "oos"]

    for col, game in enumerate(GAMES):
        ax = axes[col]

        # Find best ISGT at highest budget
        best_config = None
        best_expl = float("inf")
        for config in isgt_configs:
            sims, expl, _ = extract_series(summary, game, config)
            if sims and expl[-1] < best_expl:
                best_expl = expl[-1]
                best_config = config

        # Plot OOS + best ISGT
        plot_config(ax, summary, game, "oos")
        if best_config:
            plot_config(ax, summary, game, best_config,
                        label=f"Best ISGT: {CONFIG_LABELS[best_config]}")

        format_axes(ax)
        ax.set_title(GAME_LABELS[game], fontsize=12, fontweight="bold")
        if col == 0:
            ax.set_ylabel("Exploitability", fontsize=10)
        ax.legend(fontsize=7, loc="upper right",
                  framealpha=0.9, edgecolor="0.8")

    fig.suptitle("Experiment 15A — Best ISGT Config vs OOS\n"
                 "(δ=0.5, ε=0.4, γ=0.01)",
                 fontsize=13, fontweight="bold", y=1.04)
    fig.tight_layout()
    return fig


# ════════════════════════════════════════════════════════════════════════
# Figure 4: Coverage (1×4)
# ════════════════════════════════════════════════════════════════════════

def fig_coverage(summary):
    """Coverage (mean of P0/P1) vs sims/move. Main story is Liar's Dice."""
    fig, axes = plt.subplots(1, 4, figsize=(20, 4.5), sharey=False)

    for col, game in enumerate(GAMES):
        ax = axes[col]
        for config in CONFIG_ORDER:
            sims, _, cov_pairs = extract_series(summary, game, config)
            if not sims:
                continue
            mean_cov = [(c[0] + c[1]) / 2.0 for c in cov_pairs]
            ax.plot(sims, mean_cov,
                    color=CONFIG_COLORS[config],
                    linestyle=CONFIG_LINESTYLES[config],
                    marker=CONFIG_MARKERS[config],
                    markersize=7 if config == "oos" else 5,
                    linewidth=CONFIG_LINEWIDTHS[config],
                    zorder=CONFIG_ZORDER[config],
                    label=CONFIG_LABELS[config])
        format_axes(ax)
        ax.set_title(GAME_LABELS[game], fontsize=12, fontweight="bold")
        ax.set_ylim(-0.05, 1.05)
        if col == 0:
            ax.set_ylabel("Infoset Coverage", fontsize=10)
        if col == 3:  # legend on Liar's Dice where it matters
            ax.legend(fontsize=7, loc="lower right",
                      framealpha=0.9, edgecolor="0.8")

    fig.suptitle("Experiment 15A — Infoset Coverage\n"
                 "(δ=0.5, ε=0.4, γ=0.01)",
                 fontsize=13, fontweight="bold", y=1.04)
    fig.tight_layout()
    return fig


# ════════════════════════════════════════════════════════════════════════
# Tables
# ════════════════════════════════════════════════════════════════════════

def print_separator(title, width=120):
    print()
    print("=" * width)
    print(f"  {title}")
    print("=" * width)


# ── Table 1: Exploitability summary ──────────────────────────────────

def table_exploitability(summary):
    print_separator("Table 1: Aggregate Exploitability (all games × configs × budgets)")
    print()

    for game in GAMES:
        print(f"  {GAME_LABELS[game]}")
        print(f"  {'Config':<18} {'100':>10} {'250':>10} {'500':>10} {'1000':>10}")
        print(f"  {'-'*18} {'-'*10} {'-'*10} {'-'*10} {'-'*10}")

        for config in CONFIG_ORDER:
            sims, expl, _ = extract_series(summary, game, config)
            if not sims:
                continue
            expl_by_sims = dict(zip(sims, expl))
            vals = [f"{expl_by_sims.get(s, float('nan')):10.4f}" for s in [100, 250, 500, 1000]]
            print(f"  {CONFIG_SHORT[config]:<18} {''.join(vals)}")

        # Highlight best ISGT vs OOS at sims=1000
        oos_expl = dict(zip(*extract_series(summary, game, "oos")[:2])).get(1000)
        best_label, best_val = None, float("inf")
        for config in CONFIG_ORDER:
            if config == "oos":
                continue
            sims, expl, _ = extract_series(summary, game, config)
            e = dict(zip(sims, expl)).get(1000)
            if e is not None and e < best_val:
                best_val = e
                best_label = CONFIG_SHORT[config]

        if oos_expl and best_label:
            delta_pct = (best_val - oos_expl) / oos_expl * 100
            winner = "ISGT" if best_val < oos_expl else "OOS"
            print(f"  → Best ISGT ({best_label}) vs OOS @ 1000: "
                  f"{best_val:.4f} vs {oos_expl:.4f} ({delta_pct:+.1f}%, {winner} wins)")
        print()


# ── Table 2: Active regret by IIG level ──────────────────────────────

def _regret_table(detailed, game, sims, metric_key, title_suffix, show_per_node=False):
    """Generic regret table for a given game and metric."""
    isgt_configs = [c for c in CONFIG_ORDER if c != "oos"]

    # Collect all levels across configs
    all_levels = set()
    entries = {}
    for config in isgt_configs:
        entry = get_detailed_entry(detailed, game, config, sims)
        if entry:
            entries[config] = entry
            all_levels.update(entry.get(metric_key, {}).keys())

    if not entries:
        print(f"  No data for {game} {title_suffix}")
        return

    levels = sorted(all_levels, key=lambda x: int(x))
    # Skip level -1 (floor group) in display
    levels = [l for l in levels if int(l) >= 0]

    if show_per_node:
        # Per-node table: mean, per_node, CV, n_samples
        header = f"  {'Lvl':>4}"
        for config in isgt_configs:
            short = CONFIG_SHORT[config]
            header += f"  {short+' mean':>12} {short+' /node':>12} {short+' CV':>10} {short+' n':>8}"
        print(header)
        print(f"  {'-'*4}" + (f"  {'-'*12} {'-'*12} {'-'*10} {'-'*8}" * len(isgt_configs)))

        for lvl in levels:
            row = f"  {lvl:>4}"
            for config in isgt_configs:
                entry = entries.get(config)
                if not entry:
                    row += f"  {'—':>12} {'—':>12} {'—':>10} {'—':>8}"
                    continue
                r = entry.get(metric_key, {}).get(lvl)
                t = entry.get("neighborhood_touched_by_level", {}).get(lvl)
                if r is None:
                    row += f"  {'—':>12} {'—':>12} {'—':>10} {'—':>8}"
                    continue
                mean_val = r["mean_abs_delta"]
                std_val = r["std_delta"]
                n = r["n_samples"]
                cv = std_val / mean_val if mean_val > 0 else float("inf")
                if t and t["mean_touched"] > 0:
                    per_node = mean_val / t["mean_touched"]
                else:
                    per_node = 0.0
                if mean_val == 0 and std_val == 0:
                    row += f"  {'0':>12} {'0':>12} {'—':>10} {n:>8}"
                else:
                    row += f"  {mean_val:>12.4f} {per_node:>12.4f} {cv:>10.2f} {n:>8}"
            print(row)
    else:
        # Standard table: mean, std, variance, CV, n_samples
        header = f"  {'Lvl':>4}"
        for config in isgt_configs:
            short = CONFIG_SHORT[config]
            header += f"  {short+' mean':>12} {short+' std':>10} {short+' var':>12} {short+' CV':>8} {short+' n':>8}"
        print(header)
        print(f"  {'-'*4}" + (f"  {'-'*12} {'-'*10} {'-'*12} {'-'*8} {'-'*8}" * len(isgt_configs)))

        for lvl in levels:
            row = f"  {lvl:>4}"
            for config in isgt_configs:
                entry = entries.get(config)
                if not entry:
                    row += f"  {'—':>12} {'—':>10} {'—':>12} {'—':>8} {'—':>8}"
                    continue
                r = entry.get(metric_key, {}).get(lvl)
                if r is None:
                    row += f"  {'—':>12} {'—':>10} {'—':>12} {'—':>8} {'—':>8}"
                    continue
                mean_val = r["mean_abs_delta"]
                std_val = r["std_delta"]
                var_val = std_val ** 2
                cv = std_val / mean_val if mean_val > 0 else float("inf")
                n = r["n_samples"]
                if mean_val == 0 and std_val == 0:
                    row += f"  {'0':>12} {'0':>10} {'0':>12} {'—':>8} {n:>8}"
                else:
                    row += f"  {mean_val:>12.4f} {std_val:>10.4f} {var_val:>12.6f} {cv:>8.2f} {n:>8}"
            print(row)


def table_active_regret(detailed, sims=1000):
    print_separator(f"Table 2: Active Regret |Δregret| at I₀ by IIG Level (sims={sims})")
    print("  Mean = mean |Δregret| per walk. Variance/CV measure update stability.")
    print("  High CV = dominated by rare large updates (most walks produce zero).")
    print()

    for game in ["goofspiel", "leduc_poker", "liars_dice"]:
        print(f"\n  ── {GAME_LABELS[game]} ──")
        _regret_table(detailed, game, sims, "regret_by_level", "active regret")


def table_neighborhood_regret(detailed, sims=1000):
    print_separator(f"Table 3: Neighborhood Regret (sum |Δregret| across upstream nbr) by IIG Level (sims={sims})")
    print("  Mean = total neighborhood |Δregret| per walk. Lower CV = more reliable signal.")
    print()

    for game in ["goofspiel", "leduc_poker", "liars_dice"]:
        print(f"\n  ── {GAME_LABELS[game]} ──")
        _regret_table(detailed, game, sims, "neighborhood_regret_by_level", "neighborhood regret")


def table_neighborhood_per_node(detailed, sims=1000):
    print_separator(f"Table 4: Neighborhood Regret Per Touched Node by IIG Level (sims={sims})")
    print("  /node = mean_nbr_delta / mean_touched. Controls for BFS fan-out.")
    print("  Flat /node across levels → proximity doesn't matter for per-infoset work.")
    print()

    for game in ["goofspiel", "leduc_poker", "liars_dice"]:
        print(f"\n  ── {GAME_LABELS[game]} ──")
        _regret_table(detailed, game, sims, "neighborhood_regret_by_level",
                      "nbr per node", show_per_node=True)


# ── Table 5: Coverage ────────────────────────────────────────────────

def table_coverage(summary):
    print_separator("Table 5: Infoset Coverage (mean of P0/P1)")
    print("  Liar's Dice is the main story — other games hit 100% at all budgets.")
    print()

    for game in GAMES:
        any_below_1 = False
        for config in CONFIG_ORDER:
            sims, _, cov_pairs = extract_series(summary, game, config)
            for c in cov_pairs:
                if (c[0] + c[1]) / 2.0 < 0.999:
                    any_below_1 = True
                    break

        if not any_below_1 and game != "liars_dice":
            print(f"  {GAME_LABELS[game]}: 100% coverage at all budgets for all configs.")
            continue

        print(f"\n  {GAME_LABELS[game]}")
        print(f"  {'Config':<18} {'100':>10} {'250':>10} {'500':>10} {'1000':>10}")
        print(f"  {'-'*18} {'-'*10} {'-'*10} {'-'*10} {'-'*10}")

        for config in CONFIG_ORDER:
            sims, _, cov_pairs = extract_series(summary, game, config)
            if not sims:
                continue
            cov_by_sims = {s: (c[0] + c[1]) / 2.0 for s, c in zip(sims, cov_pairs)}
            vals = [f"{cov_by_sims.get(s, 0):10.1%}" for s in [100, 250, 500, 1000]]
            print(f"  {CONFIG_SHORT[config]:<18} {''.join(vals)}")
        print()


# ── Table 6: Level profile (weight distribution) ────────────────────

def table_level_profile(detailed, sims=1000):
    print_separator(f"Table 6: Level-Stratified Weight Profile (sims={sims})")
    print("  Per IIG level: coverage, mean weight, num infosets at that level.")
    print("  Shows how each algorithm distributes learning effort across the IIG.")
    print()

    for game in ["leduc_poker", "liars_dice"]:
        print(f"\n  ── {GAME_LABELS[game]} ──")
        for config in CONFIG_ORDER:
            entry = get_detailed_entry(detailed, game, config, sims)
            if not entry:
                continue
            lp = entry.get("level_profile", {})
            if not lp:
                continue
            print(f"\n    {CONFIG_LABELS[config]}:")
            print(f"    {'Lvl':>4} {'Coverage':>10} {'Mean Wt':>12} {'# Infosets':>12}")
            print(f"    {'-'*4} {'-'*10} {'-'*12} {'-'*12}")
            for lvl in sorted(lp.keys(), key=lambda x: int(x)):
                p = lp[lvl]
                print(f"    {lvl:>4} {p['coverage']:>10.4f} {p['mean_weight']:>12.1f} {p['n_infosets']:>12}")


# ════════════════════════════════════════════════════════════════════════
# Main
# ════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Plot 015A mechanism isolation results")
    parser.add_argument("--results", type=str, default=str(DEFAULT_RESULTS_DIR),
                        help="Path to results directory")
    parser.add_argument("--tables-only", action="store_true",
                        help="Print tables only, no plots")
    parser.add_argument("--plots-only", action="store_true",
                        help="Generate plots only, no tables")
    parser.add_argument("--sims", type=int, default=1000,
                        help="Sims/move for regret tables (default: 1000)")
    args = parser.parse_args()

    results_dir = Path(args.results)
    summary = load_summary(results_dir)
    detailed = load_detailed(results_dir)

    if not args.plots_only:
        table_exploitability(summary)
        table_active_regret(detailed, sims=args.sims)
        table_neighborhood_regret(detailed, sims=args.sims)
        table_neighborhood_per_node(detailed, sims=args.sims)
        table_coverage(summary)
        table_level_profile(detailed, sims=args.sims)

    if not args.tables_only:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

        fig1 = fig_exploitability_all(summary)
        fig1.savefig(OUTPUT_DIR / "fig1_exploitability_all.png",
                     dpi=150, bbox_inches="tight")
        print(f"\n  Saved: {OUTPUT_DIR / 'fig1_exploitability_all.png'}")

        fig2 = fig_exploitability_by_mode(summary)
        fig2.savefig(OUTPUT_DIR / "fig2_exploitability_by_mode.png",
                     dpi=150, bbox_inches="tight")
        print(f"  Saved: {OUTPUT_DIR / 'fig2_exploitability_by_mode.png'}")

        fig3 = fig_best_vs_oos(summary)
        fig3.savefig(OUTPUT_DIR / "fig3_best_vs_oos.png",
                     dpi=150, bbox_inches="tight")
        print(f"  Saved: {OUTPUT_DIR / 'fig3_best_vs_oos.png'}")

        fig4 = fig_coverage(summary)
        fig4.savefig(OUTPUT_DIR / "fig4_coverage.png",
                     dpi=150, bbox_inches="tight")
        print(f"  Saved: {OUTPUT_DIR / 'fig4_coverage.png'}")

        plt.close("all")
        print(f"\n  All plots saved to {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
