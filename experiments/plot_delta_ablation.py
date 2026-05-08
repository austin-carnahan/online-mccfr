"""Plot delta ablation results.

Generates three figures from isgt_delta_ablation_full/ablation_summary.json:

  Figure 1 — Delta Sensitivity (2×4 grid)
    Rows: chance / full mode.  Columns: one per game.
    Lines: δ ∈ {0.2, 0.5, 0.9, 1.0} (constant decay) + OOS baseline.
    Shows how δ controls exploration vs targeting in each mode.

  Figure 2 — Mode Comparison (1×4 grid)
    Best chance config vs best full config vs OOS per game.
    Demonstrates that chance mode is the robust default.

  Figure 3 — IIG Decay Signal (1×4 grid)
    Exp(0.5) vs Constant at matched (mode, δ) + OOS.
    Reveals whether IIG proximity-based weighting adds value.

Usage:
    python -m experiments.plot_delta_ablation
"""

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

RESULTS_PATH = Path("results/isgt_delta_ablation_full/ablation_summary.json")
OUTPUT_DIR = Path("results/plots")

# Game display order and labels
GAMES = ["kuhn_poker", "leduc_poker", "goofspiel", "liars_dice"]
GAME_LABELS = {
    "kuhn_poker": "Kuhn Poker",
    "leduc_poker": "Leduc Poker",
    "goofspiel": "Goofspiel",
    "liars_dice": "Liar's Dice",
}

# ── Color palettes ──────────────────────────────────────────────────────

# Delta values: sequential blue palette (light→dark = more targeting)
DELTA_COLORS = {
    "d0.2": "#93c5fd",   # light blue
    "d0.5": "#3b82f6",   # medium blue
    "d0.9": "#1d4ed8",   # dark blue
    "d1.0": "#1e3a5f",   # navy
}
DELTA_LINESTYLES = {
    "d0.2": (0, (5, 3)),       # dashed
    "d0.5": (0, (3, 1, 1, 1)), # dash-dot
    "d0.9": "-",                # solid
    "d1.0": (0, (1, 1)),       # dotted
}
DELTA_MARKERS = {"d0.2": "o", "d0.5": "s", "d0.9": "^", "d1.0": "D"}

OOS_STYLE = dict(color="#dc2626", linestyle="--", marker="*",
                 markersize=10, linewidth=2.2, zorder=10)

# Mode comparison palette
MODE_COLORS = {
    "chance": "#16a34a",   # green
    "full":   "#9333ea",   # purple
}

# Decay comparison palette
DECAY_COLORS = {
    "constant": "#f59e0b",  # amber
    "exp_0.5": "#06b6d4",   # cyan
}
DECAY_MARKERS = {"constant": "o", "exp_0.5": "^"}


def load_data():
    with open(RESULTS_PATH) as f:
        return json.load(f)


def extract(data, game, config_key):
    """Pull sims and exploitability arrays for a config."""
    series = data[game][config_key]
    sims = [d["sims_per_move"] for d in series]
    expl = [d["exploitability"] for d in series]
    return sims, expl


def best_delta_for(data, game, mode, decay):
    """Find the δ that gives lowest exploitability at sims=1000."""
    best_d, best_e = None, float("inf")
    for d in ["d0.2", "d0.5", "d0.9", "d1.0"]:
        key = f"{mode}_{decay}_{d}"
        series = data[game].get(key)
        if series:
            e1000 = [s for s in series if s["sims_per_move"] == 1000]
            if e1000 and e1000[0]["exploitability"] < best_e:
                best_e = e1000[0]["exploitability"]
                best_d = d
    return best_d


def format_axes(ax, x_label=True):
    """Common axis formatting."""
    ax.set_xscale("log")
    ax.set_xticks([100, 250, 500, 1000])
    ax.get_xaxis().set_major_formatter(ticker.ScalarFormatter())
    ax.tick_params(axis="both", which="both", labelsize=9)
    ax.grid(True, alpha=0.25, linewidth=0.5)
    if x_label:
        ax.set_xlabel("Sims / move", fontsize=10)


# ════════════════════════════════════════════════════════════════════════
# Figure 1: Delta Sensitivity (2×4)
# ════════════════════════════════════════════════════════════════════════

def fig_delta_sensitivity(data):
    fig, axes = plt.subplots(2, 4, figsize=(18, 8), sharey=False)

    for col, game in enumerate(GAMES):
        for row, mode in enumerate(["chance", "full"]):
            ax = axes[row, col]

            # Plot δ variants (constant decay)
            for d_label in ["d0.2", "d0.5", "d0.9", "d1.0"]:
                key = f"{mode}_constant_{d_label}"
                sims, expl = extract(data, game, key)
                ax.plot(sims, expl,
                        color=DELTA_COLORS[d_label],
                        linestyle=DELTA_LINESTYLES[d_label],
                        marker=DELTA_MARKERS[d_label],
                        markersize=6, linewidth=1.8,
                        label=f"δ={d_label[1:]}")

            # OOS baseline
            sims, expl = extract(data, game, "oos_precomp")
            ax.plot(sims, expl, label="OOS", **OOS_STYLE)

            format_axes(ax, x_label=(row == 1))
            if row == 0:
                ax.set_title(GAME_LABELS[game], fontsize=12, fontweight="bold")
                ax.set_xlabel("")
            if col == 0:
                ax.set_ylabel(f"{'Chance' if mode == 'chance' else 'Full'} mode\n"
                              f"Exploitability", fontsize=10)

            # Legend only on first subplot
            if row == 0 and col == 0:
                ax.legend(fontsize=8, loc="upper right",
                          framealpha=0.9, edgecolor="0.8")

    fig.suptitle("Delta (δ) Sensitivity — Constant Decay",
                 fontsize=14, fontweight="bold", y=1.01)
    fig.tight_layout()
    return fig


# ════════════════════════════════════════════════════════════════════════
# Figure 2: Mode Comparison (1×4)
# ════════════════════════════════════════════════════════════════════════

def fig_mode_comparison(data):
    fig, axes = plt.subplots(1, 4, figsize=(18, 4.5))

    for col, game in enumerate(GAMES):
        ax = axes[col]

        # Best chance config (constant decay, best δ)
        best_d_ch = best_delta_for(data, game, "chance", "constant")
        key_ch = f"chance_constant_{best_d_ch}"
        sims, expl = extract(data, game, key_ch)
        ax.plot(sims, expl, color=MODE_COLORS["chance"],
                marker="o", markersize=7, linewidth=2,
                linestyle="-",
                label=f"Chance (δ={best_d_ch[1:]})")

        # Best full config (constant decay, best δ)
        best_d_fu = best_delta_for(data, game, "full", "constant")
        key_fu = f"full_constant_{best_d_fu}"
        sims, expl = extract(data, game, key_fu)
        ax.plot(sims, expl, color=MODE_COLORS["full"],
                marker="s", markersize=7, linewidth=2,
                linestyle="-.",
                label=f"Full (δ={best_d_fu[1:]})")

        # Best chance exp config
        best_d_ch_exp = best_delta_for(data, game, "chance", "exp_0.5")
        key_ch_exp = f"chance_exp_0.5_{best_d_ch_exp}"
        sims, expl = extract(data, game, key_ch_exp)
        ax.plot(sims, expl, color=MODE_COLORS["chance"],
                marker="^", markersize=7, linewidth=1.5,
                linestyle="--", alpha=0.7,
                label=f"Chance+Exp (δ={best_d_ch_exp[1:]})")

        # OOS
        sims, expl = extract(data, game, "oos_precomp")
        ax.plot(sims, expl, label="OOS", **OOS_STYLE)

        format_axes(ax)
        ax.set_title(GAME_LABELS[game], fontsize=12, fontweight="bold")
        if col == 0:
            ax.set_ylabel("Exploitability", fontsize=10)
            ax.legend(fontsize=8, loc="upper right",
                      framealpha=0.9, edgecolor="0.8")

    fig.suptitle("Best ISGT Configs vs OOS",
                 fontsize=14, fontweight="bold", y=1.01)
    fig.tight_layout()
    return fig


# ════════════════════════════════════════════════════════════════════════
# Figure 3: IIG Decay Signal (1×4)
# ════════════════════════════════════════════════════════════════════════

def fig_decay_signal(data):
    fig, axes = plt.subplots(1, 4, figsize=(18, 4.5))

    for col, game in enumerate(GAMES):
        ax = axes[col]

        # Pick the δ where exp vs constant difference is most interesting
        # Use chance mode, compare at δ=0.9 (where targeting matters most)
        test_delta = "d0.9"

        for decay, style_d in [("constant", "constant"), ("exp_0.5", "exp_0.5")]:
            key = f"chance_{decay}_{test_delta}"
            sims, expl = extract(data, game, key)
            ax.plot(sims, expl,
                    color=DECAY_COLORS[style_d],
                    marker=DECAY_MARKERS[style_d],
                    markersize=7, linewidth=2,
                    linestyle="-",
                    label=f"Constant (δ=0.9)" if decay == "constant"
                          else f"Exp(0.5) (δ=0.9)")

        # Also show at δ=0.5 for comparison
        test_delta2 = "d0.5"
        for decay, style_d in [("constant", "constant"), ("exp_0.5", "exp_0.5")]:
            key = f"chance_{decay}_{test_delta2}"
            sims, expl = extract(data, game, key)
            ax.plot(sims, expl,
                    color=DECAY_COLORS[style_d],
                    marker=DECAY_MARKERS[style_d],
                    markersize=5, linewidth=1.3,
                    linestyle="--", alpha=0.5,
                    label=f"Constant (δ=0.5)" if decay == "constant"
                          else f"Exp(0.5) (δ=0.5)")

        # OOS
        sims, expl = extract(data, game, "oos_precomp")
        ax.plot(sims, expl, label="OOS", **OOS_STYLE)

        format_axes(ax)
        ax.set_title(GAME_LABELS[game], fontsize=12, fontweight="bold")
        if col == 0:
            ax.set_ylabel("Exploitability", fontsize=10)
            ax.legend(fontsize=7.5, loc="upper right",
                      framealpha=0.9, edgecolor="0.8")

    fig.suptitle("IIG Decay Signal — Exp(0.5) vs Constant (Chance Mode)",
                 fontsize=14, fontweight="bold", y=1.01)
    fig.tight_layout()
    return fig


# ════════════════════════════════════════════════════════════════════════
# Figure 4: Bar chart — sims=1000, all configs ranked per game
# ════════════════════════════════════════════════════════════════════════

def fig_bar_ranking(data):
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes_flat = axes.flatten()

    for idx, game in enumerate(GAMES):
        ax = axes_flat[idx]
        game_data = data[game]

        # Collect sims=1000 results
        configs = []
        for key, series in game_data.items():
            e1000 = [s["exploitability"] for s in series
                     if s["sims_per_move"] == 1000]
            if e1000:
                configs.append((key, e1000[0]))

        # Sort by exploitability
        configs.sort(key=lambda x: x[1])

        names = [c[0] for c in configs]
        vals = [c[1] for c in configs]

        # Color bars by type
        colors = []
        for name in names:
            if name == "oos_precomp":
                colors.append("#dc2626")
            elif name.startswith("chance_exp"):
                colors.append("#06b6d4")
            elif name.startswith("chance_const"):
                colors.append("#16a34a")
            elif name.startswith("full_exp"):
                colors.append("#a78bfa")
            else:
                colors.append("#9333ea")

        bars = ax.barh(range(len(names)), vals, color=colors, edgecolor="white",
                       linewidth=0.5, height=0.7)

        # Annotate with values
        for bar, val in zip(bars, vals):
            ax.text(bar.get_width() + (max(vals) - min(vals)) * 0.01,
                    bar.get_y() + bar.get_height() / 2,
                    f"{val:.4f}", va="center", fontsize=7.5)

        ax.set_yticks(range(len(names)))
        ax.set_yticklabels([_short_label(n) for n in names], fontsize=8)
        ax.invert_yaxis()
        ax.set_xlabel("Exploitability (sims=1000)", fontsize=10)
        ax.set_title(GAME_LABELS[game], fontsize=12, fontweight="bold")
        ax.grid(axis="x", alpha=0.2)

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#16a34a", label="Chance + Constant"),
        Patch(facecolor="#06b6d4", label="Chance + Exp(0.5)"),
        Patch(facecolor="#9333ea", label="Full + Constant"),
        Patch(facecolor="#a78bfa", label="Full + Exp(0.5)"),
        Patch(facecolor="#dc2626", label="OOS"),
    ]
    fig.legend(handles=legend_elements, loc="lower center",
               ncol=5, fontsize=10, framealpha=0.9,
               bbox_to_anchor=(0.5, -0.02))

    fig.suptitle("All Configs Ranked at sims=1000",
                 fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0.03, 1, 0.97])
    return fig


def _short_label(config_name):
    """Shorten config name for bar chart labels."""
    if config_name == "oos_precomp":
        return "OOS (δ=0.9)"
    # e.g. "chance_constant_d0.5" → "Ch Const δ=0.5"
    parts = config_name.split("_")
    mode = "Ch" if parts[0] == "chance" else "Fu"
    if parts[1] == "constant":
        decay = "Const"
        delta = parts[2]
    else:
        decay = f"Exp({parts[2]})"
        delta = parts[3]
    return f"{mode} {decay} δ={delta[1:]}"


# ════════════════════════════════════════════════════════════════════════

def main():
    data = load_data()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Generating Figure 1: Delta Sensitivity (2×4 grid)...")
    fig1 = fig_delta_sensitivity(data)
    p1 = OUTPUT_DIR / "delta_sensitivity.png"
    fig1.savefig(p1, dpi=150, bbox_inches="tight")
    print(f"  → {p1}")

    print("Generating Figure 2: Mode Comparison (1×4)...")
    fig2 = fig_mode_comparison(data)
    p2 = OUTPUT_DIR / "mode_comparison.png"
    fig2.savefig(p2, dpi=150, bbox_inches="tight")
    print(f"  → {p2}")

    print("Generating Figure 3: IIG Decay Signal (1×4)...")
    fig3 = fig_decay_signal(data)
    p3 = OUTPUT_DIR / "decay_signal.png"
    fig3.savefig(p3, dpi=150, bbox_inches="tight")
    print(f"  → {p3}")

    print("Generating Figure 4: Bar Rankings (2×2)...")
    fig4 = fig_bar_ranking(data)
    p4 = OUTPUT_DIR / "bar_ranking.png"
    fig4.savefig(p4, dpi=150, bbox_inches="tight")
    print(f"  → {p4}")

    print(f"\nAll plots saved to {OUTPUT_DIR}/")
    plt.close("all")


if __name__ == "__main__":
    main()
