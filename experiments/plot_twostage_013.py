"""Plot two-stage decay histograms from archive 013 JSON results.

Generates one figure per game: grouped bars showing the empirical
fraction of targeted samples per IIG level for all 10 decay functions.

Usage:
    python -m experiments.plot_twostage_013
    python -m experiments.plot_twostage_013 --game kuhn_poker
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

JSON_PATH = Path("archive/013_twostage_decay_histograms/results.json")
PLOT_DIR = Path("results/plots/twostage_013")

GAME_LABELS = {
    "kuhn_poker": "Kuhn Poker",
    "leduc_poker": "Leduc Poker",
    "liars_dice": "Liar's Dice",
    "goofspiel": "Goofspiel",
}

DECAY_COLORS = {
    "t_balanced": "#10b981",
    "constant":   "#6b7280",
    "exp(0.7)":   "#22d3ee",
    "exp(0.5)":   "#06b6d4",
    "exp(0.3)":   "#0284c7",
    "exp(0.1)":   "#3b82f6",
    "step(0.25)": "#a78bfa",
    "step(0.1)":  "#8b5cf6",
    "step(0.01)": "#7c3aed",
    "poly(2)":    "#f97316",
}


def plot_game(game_name: str, game_data: dict, num_sims: int = 500,
              dpi: int = 150) -> plt.Figure:
    """Single grouped-bar chart of empirical z* level distribution."""
    levels = sorted(int(k) for k in game_data["terminals_per_level"])
    decay_labels = list(game_data["empirical"].keys())
    n_levels = len(levels)
    n_decays = len(decay_labels)
    bar_w = 0.8 / n_decays
    x = np.arange(n_levels)

    fig, ax = plt.subplots(figsize=(max(9, 2.2 * n_levels), 5))

    for i, label in enumerate(decay_labels):
        data = game_data["empirical"][label]
        total = sum(data.values())
        fractions = [data.get(str(l), 0) / total * 100 if total else 0
                     for l in levels]

        offset = (i - n_decays / 2 + 0.5) * bar_w
        color = DECAY_COLORS.get(label, f"C{i}")
        bars = ax.bar(x + offset, fractions, bar_w, label=label,
                      color=color, edgecolor="white", linewidth=0.4)

        # Annotate bars with sample counts
        for bar, l in zip(bars, levels):
            count = data.get(str(l), 0)
            if count > 0:
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 1,
                        str(count), ha="center", va="bottom",
                        fontsize=6, color="#374151")

    lvl_labels = [f"L{l}" if l >= 0 else "Floor" for l in levels]
    ax.set_xticks(x)
    ax.set_xticklabels(lvl_labels)
    ax.set_xlabel("IIG Level")
    ax.set_ylabel("Fraction of targeted samples (%)")
    ax.set_ylim(0, 105)
    ax.grid(axis="y", alpha=0.3)
    ax.legend(loc="upper right", fontsize=7, ncol=3, framealpha=0.9)

    # Title with game metadata
    t_per_lvl = game_data["terminals_per_level"]
    t_str = ", ".join(f"L{l}={t_per_lvl[str(l)]}" for l in levels)
    game_label = GAME_LABELS.get(game_name, game_name)
    ax.set_title(
        f"{game_label} — Two-Stage Decay Comparison\n"
        f"Active: {game_data['active_infoset']}  |  "
        f"Terminals: {game_data['num_terminals']}  |  {t_str}  |  "
        f"N={num_sims}",
        fontsize=11, fontweight="bold",
    )
    fig.tight_layout()
    return fig


def main():
    parser = argparse.ArgumentParser(description="013 — Plot two-stage histograms")
    parser.add_argument("--game", nargs="*", default=None)
    parser.add_argument("--dpi", type=int, default=150)
    parser.add_argument("--json", default=str(JSON_PATH))
    args = parser.parse_args()

    with open(args.json) as f:
        data = json.load(f)

    games = args.game if args.game else list(data.keys())
    PLOT_DIR.mkdir(parents=True, exist_ok=True)

    for game_name in games:
        if game_name not in data:
            print(f"  Skipping {game_name}: not in JSON")
            continue
        fig = plot_game(game_name, data[game_name], dpi=args.dpi)
        path = PLOT_DIR / f"{game_name}_twostage.png"
        fig.savefig(path, dpi=args.dpi, bbox_inches="tight")
        print(f"  → {path}")
    plt.close("all")
    print(f"\n  Done. Plots in {PLOT_DIR}/")


if __name__ == "__main__":
    main()
