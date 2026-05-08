"""Plot experiment 014 results — postfix delta baseline (aggregate exploitability).

Generates four figures from 014_postfix_delta_baseline_quick/ablation_summary.json:

  Figure 1 — Delta Sensitivity by Mode (2×4 grid)
    Rows: chance / full mode.  Columns: one per game.
    Lines: one per δ value (constant decay) + OOS baseline.
    Shows δ is the dominant parameter for aggregate exploitability.

  Figure 2 — Decay Function Spread (2×4 grid)
    Rows: chance / full mode.  Columns: one per game.
    All 7 decays plotted at δ=0.2 (best performing δ).
    Shows decay function has negligible effect on this metric.

  Figure 3 — Mode Comparison (1×4 grid)
    Best chance config vs best full config vs OOS per game.
    Shows chance ≈ full at low δ, full degrades at high δ.

  Figure 4 — Bar chart — sims=1000, all configs ranked per game

Usage:
    python -m experiments.plot_postfix_baseline_014
    python -m experiments.plot_postfix_baseline_014 --results path/to/ablation_summary.json
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

RESULTS_PATH = Path("results/014_postfix_delta_baseline_quick/ablation_summary.json")
OUTPUT_DIR = Path("results/plots/014_postfix_baseline")

GAMES = ["kuhn_poker", "leduc_poker", "goofspiel", "liars_dice"]
GAME_LABELS = {
    "kuhn_poker": "Kuhn Poker",
    "leduc_poker": "Leduc Poker",
    "goofspiel": "Goofspiel",
    "liars_dice": "Liar's Dice",
}

# ── Palettes (consistent with project conventions) ──────────────────

DELTA_COLORS = {
    "d0.2": "#93c5fd",   # light blue
    "d0.5": "#3b82f6",   # medium blue
    "d0.9": "#1d4ed8",   # dark blue
}
DELTA_LINESTYLES = {
    "d0.2": (0, (5, 3)),        # dashed
    "d0.5": (0, (3, 1, 1, 1)),  # dash-dot
    "d0.9": "-",                 # solid
}
DELTA_MARKERS = {"d0.2": "o", "d0.5": "s", "d0.9": "^"}

OOS_STYLE = dict(color="#dc2626", linestyle="--", marker="*",
                 markersize=10, linewidth=2.2, zorder=10)

MODE_COLORS = {"chance": "#16a34a", "full": "#9333ea"}

DECAY_NAMES = ["t_balanced", "constant", "exp_0.7", "exp_0.5",
               "step_0.25", "step_0.1", "poly_2"]
DECAY_COLORS = {
    "t_balanced": "#10b981",  # emerald
    "constant":   "#6b7280",  # gray
    "exp_0.7":    "#22d3ee",  # cyan
    "exp_0.5":    "#06b6d4",  # teal
    "step_0.25":  "#a78bfa",  # violet
    "step_0.1":   "#8b5cf6",  # purple
    "poly_2":     "#f97316",  # orange
}
DECAY_MARKERS = {
    "t_balanced": "o", "constant": "s", "exp_0.7": "^",
    "exp_0.5": "D", "step_0.25": "v", "step_0.1": "<", "poly_2": ">",
}
DECAY_LABELS = {
    "t_balanced": "t_balanced", "constant": "constant",
    "exp_0.7": "exp(0.7)", "exp_0.5": "exp(0.5)",
    "step_0.25": "step(0.25)", "step_0.1": "step(0.1)", "poly_2": "poly(2)",
}


def load_data(path):
    with open(path) as f:
        return json.load(f)


def extract(data, game, config_key):
    """Pull sims and exploitability arrays for a config."""
    series = data[game].get(config_key, [])
    sims = [d["sims_per_move"] for d in series]
    expl = [d["exploitability"] for d in series]
    return sims, expl


def format_axes(ax, x_label=True):
    ax.set_xscale("log")
    ax.set_xticks([250, 1000])
    ax.get_xaxis().set_major_formatter(ticker.ScalarFormatter())
    ax.tick_params(axis="both", which="both", labelsize=9)
    ax.grid(True, alpha=0.25, linewidth=0.5)
    if x_label:
        ax.set_xlabel("Sims / move", fontsize=10)


# ════════════════════════════════════════════════════════════════════════
# Figure 1: Delta Sensitivity by Mode (2×4)
# ════════════════════════════════════════════════════════════════════════

def fig_delta_sensitivity(data):
    """2×4 grid: rows=mode, cols=game. Lines per δ (constant decay) + OOS."""
    fig, axes = plt.subplots(2, 4, figsize=(18, 8), sharey=False)

    for col, game in enumerate(GAMES):
        for row, mode in enumerate(["chance", "full"]):
            ax = axes[row, col]

            for d_label in ["d0.2", "d0.5", "d0.9"]:
                key = f"isgt_{mode}_constant_{d_label}"
                sims, expl = extract(data, game, key)
                if sims:
                    ax.plot(sims, expl,
                            color=DELTA_COLORS[d_label],
                            linestyle=DELTA_LINESTYLES[d_label],
                            marker=DELTA_MARKERS[d_label],
                            markersize=6, linewidth=1.8,
                            label=f"δ={d_label[1:]}")

            # OOS baseline
            sims, expl = extract(data, game, "oos_d0.9")
            if sims:
                ax.plot(sims, expl, label="OOS (δ=0.9)", **OOS_STYLE)

            format_axes(ax, x_label=(row == 1))
            if row == 0:
                ax.set_title(GAME_LABELS[game], fontsize=12, fontweight="bold")
                ax.set_xlabel("")
            if col == 0:
                mode_label = "Chance" if mode == "chance" else "Full"
                ax.set_ylabel(f"{mode_label} mode\nExploitability", fontsize=10)
            if row == 0 and col == 0:
                ax.legend(fontsize=8, loc="upper right",
                          framealpha=0.9, edgecolor="0.8")

    fig.suptitle("δ Sensitivity — Constant Decay (Aggregate Exploitability)",
                 fontsize=14, fontweight="bold", y=1.01)
    fig.tight_layout()
    return fig


# ════════════════════════════════════════════════════════════════════════
# Figure 2: Decay Function Spread at δ=0.2 (2×4)
# ════════════════════════════════════════════════════════════════════════

def fig_decay_spread(data):
    """2×4 grid: rows=mode, cols=game. All 7 decays at δ=0.2 + OOS."""
    fig, axes = plt.subplots(2, 4, figsize=(18, 8), sharey=False)

    for col, game in enumerate(GAMES):
        for row, mode in enumerate(["chance", "full"]):
            ax = axes[row, col]

            for decay in DECAY_NAMES:
                key = f"isgt_{mode}_{decay}_d0.2"
                sims, expl = extract(data, game, key)
                if sims:
                    ax.plot(sims, expl,
                            color=DECAY_COLORS[decay],
                            marker=DECAY_MARKERS[decay],
                            markersize=6, linewidth=1.5,
                            label=DECAY_LABELS[decay])

            # OOS baseline
            sims, expl = extract(data, game, "oos_d0.9")
            if sims:
                ax.plot(sims, expl, label="OOS (δ=0.9)", **OOS_STYLE)

            format_axes(ax, x_label=(row == 1))
            if row == 0:
                ax.set_title(GAME_LABELS[game], fontsize=12, fontweight="bold")
                ax.set_xlabel("")
            if col == 0:
                mode_label = "Chance" if mode == "chance" else "Full"
                ax.set_ylabel(f"{mode_label} mode\nExploitability", fontsize=10)
            if row == 0 and col == 0:
                ax.legend(fontsize=7, loc="upper right",
                          framealpha=0.9, edgecolor="0.8", ncol=2)

    fig.suptitle("Decay Function Spread at δ=0.2 (Aggregate Exploitability)",
                 fontsize=14, fontweight="bold", y=1.01)
    fig.tight_layout()
    return fig


# ════════════════════════════════════════════════════════════════════════
# Figure 3: Mode Comparison (1×4)
# ════════════════════════════════════════════════════════════════════════

def _best_config(data, game, mode):
    """Find the (decay, delta) with lowest exploitability at sims=1000."""
    best_key, best_e = None, float("inf")
    for decay in DECAY_NAMES:
        for d in ["d0.2", "d0.5", "d0.9"]:
            key = f"isgt_{mode}_{decay}_{d}"
            series = data[game].get(key, [])
            for s in series:
                if s["sims_per_move"] == 1000 and s["exploitability"] < best_e:
                    best_e = s["exploitability"]
                    best_key = key
    return best_key, best_e


def fig_mode_comparison(data):
    """1×4 grid: best chance vs best full vs OOS per game."""
    fig, axes = plt.subplots(1, 4, figsize=(18, 4.5))

    for col, game in enumerate(GAMES):
        ax = axes[col]

        for mode, ls in [("chance", "-"), ("full", "-.")]:
            best_key, _ = _best_config(data, game, mode)
            if best_key:
                sims, expl = extract(data, game, best_key)
                # Parse delta from key
                delta_str = best_key.split("_d")[-1]
                decay_str = "_".join(best_key.split("_")[2:-1])
                ax.plot(sims, expl, color=MODE_COLORS[mode],
                        marker="o" if mode == "chance" else "s",
                        markersize=7, linewidth=2, linestyle=ls,
                        label=f"{mode.title()} ({decay_str}, δ={delta_str})")

        sims, expl = extract(data, game, "oos_d0.9")
        if sims:
            ax.plot(sims, expl, label="OOS (δ=0.9)", **OOS_STYLE)

        format_axes(ax)
        ax.set_title(GAME_LABELS[game], fontsize=12, fontweight="bold")
        if col == 0:
            ax.set_ylabel("Exploitability", fontsize=10)
            ax.legend(fontsize=7.5, loc="upper right",
                      framealpha=0.9, edgecolor="0.8")

    fig.suptitle("Best ISGT Config per Mode vs OOS (Aggregate Exploitability)",
                 fontsize=14, fontweight="bold", y=1.01)
    fig.tight_layout()
    return fig


# ════════════════════════════════════════════════════════════════════════
# Figure 4: Bar chart — sims=1000, all configs ranked per game
# ════════════════════════════════════════════════════════════════════════

def _short_label(config_name):
    """Shorten config name for bar chart labels."""
    if config_name.startswith("oos"):
        return "OOS (δ=0.9)"
    # isgt_chance_constant_d0.2 → Ch const δ=0.2
    parts = config_name.split("_")
    # parts: ['isgt', mode, decay..., 'dX.X']
    mode = "Ch" if parts[1] == "chance" else "Fu"
    delta = parts[-1]  # d0.2, d0.5, d0.9
    decay_parts = parts[2:-1]
    decay = "_".join(decay_parts)
    # Shorten decay names
    short_decay = {
        "t_balanced": "t_bal", "constant": "const",
        "exp_0.5": "exp.5", "exp_0.7": "exp.7",
        "step_0.1": "stp.1", "step_0.25": "stp.25", "poly_2": "poly2",
    }.get(decay, decay)
    return f"{mode} {short_decay} δ={delta[1:]}"


def _bar_color(config_name):
    if config_name.startswith("oos"):
        return "#dc2626"
    parts = config_name.split("_")
    mode = parts[1]
    delta = parts[-1]
    if mode == "chance":
        return {"d0.2": "#86efac", "d0.5": "#16a34a", "d0.9": "#166534"}.get(delta, "#16a34a")
    else:
        return {"d0.2": "#c4b5fd", "d0.5": "#9333ea", "d0.9": "#581c87"}.get(delta, "#9333ea")


def fig_bar_ranking(data):
    """2×2 grid: horizontal bar chart per game, all configs at sims=1000."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 14))
    axes_flat = axes.flatten()

    for idx, game in enumerate(GAMES):
        ax = axes_flat[idx]
        game_data = data[game]

        configs = []
        for key, series in game_data.items():
            e1000 = [s["exploitability"] for s in series
                     if s["sims_per_move"] == 1000]
            if e1000:
                configs.append((key, e1000[0]))
        configs.sort(key=lambda x: x[1])

        names = [c[0] for c in configs]
        vals = [c[1] for c in configs]
        colors = [_bar_color(n) for n in names]

        bars = ax.barh(range(len(names)), vals, color=colors,
                       edgecolor="white", linewidth=0.5, height=0.7)

        for bar, val in zip(bars, vals):
            ax.text(bar.get_width() + (max(vals) - min(vals)) * 0.01,
                    bar.get_y() + bar.get_height() / 2,
                    f"{val:.4f}", va="center", fontsize=6.5)

        ax.set_yticks(range(len(names)))
        ax.set_yticklabels([_short_label(n) for n in names], fontsize=7)
        ax.invert_yaxis()
        ax.set_xlabel("Exploitability (sims=1000)", fontsize=10)
        ax.set_title(GAME_LABELS[game], fontsize=12, fontweight="bold")
        ax.grid(axis="x", alpha=0.2)

    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#86efac", label="Chance δ=0.2"),
        Patch(facecolor="#16a34a", label="Chance δ=0.5"),
        Patch(facecolor="#166534", label="Chance δ=0.9"),
        Patch(facecolor="#c4b5fd", label="Full δ=0.2"),
        Patch(facecolor="#9333ea", label="Full δ=0.5"),
        Patch(facecolor="#581c87", label="Full δ=0.9"),
        Patch(facecolor="#dc2626", label="OOS"),
    ]
    fig.legend(handles=legend_elements, loc="lower center",
               ncol=7, fontsize=9, framealpha=0.9,
               bbox_to_anchor=(0.5, -0.01))

    fig.suptitle("All Configs Ranked at sims=1000 (Aggregate Exploitability)",
                 fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0.03, 1, 0.97])
    return fig


# ════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="014 — Plot postfix delta baseline results")
    parser.add_argument("--results", type=str, default=str(RESULTS_PATH),
                        help="Path to ablation_summary.json")
    parser.add_argument("--output", type=str, default=str(OUTPUT_DIR),
                        help="Output directory for plots")
    parser.add_argument("--dpi", type=int, default=150)
    args = parser.parse_args()

    results_path = Path(args.results)
    output_dir = Path(args.output)

    if not results_path.exists():
        print(f"Error: {results_path} not found")
        return

    data = load_data(results_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    plots = [
        ("delta_sensitivity", "Delta Sensitivity (2×4)", fig_delta_sensitivity),
        ("decay_spread", "Decay Spread at δ=0.2 (2×4)", fig_decay_spread),
        ("mode_comparison", "Mode Comparison (1×4)", fig_mode_comparison),
        ("bar_ranking", "Bar Rankings (2×2)", fig_bar_ranking),
    ]

    for name, desc, fn in plots:
        print(f"Generating {desc}...")
        fig = fn(data)
        path = output_dir / f"{name}.png"
        fig.savefig(path, dpi=args.dpi, bbox_inches="tight")
        print(f"  → {path}")

    plt.close("all")
    print(f"\nAll plots saved to {output_dir}/")


if __name__ == "__main__":
    main()
