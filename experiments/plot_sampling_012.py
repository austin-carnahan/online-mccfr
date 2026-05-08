"""Plot sampling distribution histograms for archive 012.

Generates data (if needed) and plots for all four benchmark games.
Produces four figures per game (one per parameter sweep):

  Figure 1 — Decay Function Comparison
  Figure 2 — δ (Targeting Probability) Sweep
  Figure 3 — Bias Mode Comparison (chance vs full)
  Figure 4 — ε (Exploration) Sweep

Data is cached as JSON in results/sampling_012/.
Plots are saved as PNG in results/plots/sampling_012/.

Usage:
    python -m experiments.plot_sampling_012           # all games, generate + plot
    python -m experiments.plot_sampling_012 --quick   # Kuhn only (fast check)
    python -m experiments.plot_sampling_012 --plot-only  # skip data generation
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

from experiments.sampling_analysis_012 import (
    Config, HistogramResult,
    decay_comparison, delta_sweep, mode_comparison, epsilon_sweep,
    run_parameter_sweep, _deepest_infoset,
)
from src.games import load_game
from src.iig import IIG


# ── Color palettes (consistent with plot_delta_ablation.py) ─────────

# Decay functions: distinct categorical palette
DECAY_COLORS = {
    "constant":   "#6b7280",  # gray
    "exp(0.5)":   "#06b6d4",  # cyan
    "exp(0.1)":   "#3b82f6",  # blue
    "step(0.01)": "#8b5cf6",  # violet
}

# Delta values: sequential blue gradient (more targeting = darker)
DELTA_COLORS = {
    "δ=0.0": "#dbeafe",   # lightest blue
    "δ=0.2": "#93c5fd",
    "δ=0.5": "#3b82f6",
    "δ=0.9": "#1d4ed8",
    "δ=1.0": "#1e3a5f",   # navy
}

# Mode: green/purple
MODE_COLORS = {
    "chance": "#16a34a",
    "full":   "#9333ea",
}

# Epsilon: warm gradient
EPSILON_COLORS = {
    "ε=0.0": "#fbbf24",  # amber
    "ε=0.2": "#f97316",  # orange
    "ε=0.6": "#ef4444",  # red
    "ε=1.0": "#dc2626",  # dark red
}

SWEEP_PALETTES = {
    "decay": DECAY_COLORS,
    "delta": DELTA_COLORS,
    "mode":  MODE_COLORS,
    "epsilon": EPSILON_COLORS,
}

SWEEP_TITLES = {
    "decay":   "Decay Function Comparison",
    "delta":   "δ (Targeting Probability) Sweep",
    "mode":    "Bias Mode Comparison",
    "epsilon": "ε (Exploration) Sweep",
}


# ── Data loading ────────────────────────────────────────────────────────

def load_data(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def _parse_histogram(hist_dict: dict) -> dict[int, int]:
    """Convert JSON string keys back to ints."""
    return {int(k): v for k, v in hist_dict.items()}


# ── Plotting ────────────────────────────────────────────────────────────

def plot_sweep(
    sweep_data: list[dict],
    palette: dict[str, str],
    title: str,
    subtitle: str = "",
) -> plt.Figure:
    """Plot a grouped bar chart of z* IIG distance distributions.

    Args:
        sweep_data: List of result dicts (from HistogramResult.to_dict()).
        palette: label → color mapping.
        title: Figure title.
        subtitle: Optional subtitle (game name, etc.).

    Returns:
        matplotlib Figure.
    """
    # Extract histograms (targeted-only z* distribution)
    labels = [d["label"] for d in sweep_data]
    hists = [_parse_histogram(d["z_star_targeted"]) for d in sweep_data]

    # Collect all distance levels across configs
    all_dists = sorted(set().union(*hists))

    # Build bar positions
    n_groups = len(all_dists)
    n_bars = len(labels)
    bar_width = 0.8 / n_bars
    x = np.arange(n_groups)

    fig, ax = plt.subplots(figsize=(max(7, 2 * n_groups + 2), 5))

    for i, (label, hist) in enumerate(zip(labels, hists)):
        total = sum(hist.values())
        fractions = [hist.get(d, 0) / total * 100 if total > 0 else 0
                     for d in all_dists]
        offset = (i - n_bars / 2 + 0.5) * bar_width
        color = palette.get(label, f"C{i}")
        bars = ax.bar(x + offset, fractions, bar_width,
                      label=label, color=color, edgecolor="white",
                      linewidth=0.5)

        # Annotate bars with counts
        for bar, d in zip(bars, all_dists):
            count = hist.get(d, 0)
            if count > 0:
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 1,
                        str(count), ha="center", va="bottom",
                        fontsize=7, color="#374151")

    # X-axis labels
    dist_labels = ["Floor" if d < 0 else f"Level {d}" for d in all_dists]
    ax.set_xticks(x)
    ax.set_xticklabels(dist_labels)
    ax.set_xlabel("IIG Distance")
    ax.set_ylabel("Fraction of targeted samples (%)")
    ax.set_ylim(0, 105)

    # Title
    full_title = title
    if subtitle:
        full_title += f"\n{subtitle}"
    ax.set_title(full_title, fontsize=12, fontweight="bold")

    # Metadata annotation
    if sweep_data:
        game = sweep_data[0].get("config", {}).get("game", "")
        upstream = sweep_data[0].get("num_upstream_terminals", "?")
        total_t = sweep_data[0].get("num_terminals", "?")
        n_sims = sweep_data[0].get("config", {}).get("num_sims", "?")
        ax.text(0.99, 0.97,
                f"Upstream: {upstream}/{total_t}  |  N={n_sims}/config",
                transform=ax.transAxes, ha="right", va="top",
                fontsize=8, color="#6b7280",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                          edgecolor="#e5e7eb", alpha=0.9))

    ax.legend(loc="best", fontsize=9, framealpha=0.9)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    return fig


def plot_all_sweeps(data: dict, game_label: str = "") -> dict[str, plt.Figure]:
    """Plot all available sweeps from a JSON results file.

    Args:
        data: Dict of sweep_name → list of result dicts.
        game_label: Human-readable game name for subtitles.

    Returns:
        Dict of sweep_name → Figure.
    """
    figures = {}
    for sweep_name, sweep_data in data.items():
        title = SWEEP_TITLES.get(sweep_name, sweep_name)
        palette = SWEEP_PALETTES.get(sweep_name, {})
        fig = plot_sweep(sweep_data, palette, title, subtitle=game_label)
        figures[sweep_name] = fig
    return figures


# ── Data generation ─────────────────────────────────────────────────────

DATA_DIR = Path("results/sampling_012")
PLOT_DIR = Path("results/plots/sampling_012")

GAMES = ["kuhn_poker", "leduc_poker", "goofspiel", "liars_dice"]
GAME_LABELS = {
    "kuhn_poker": "Kuhn Poker",
    "leduc_poker": "Leduc Poker",
    "liars_dice": "Liar's Dice",
    "goofspiel": "Goofspiel",
}

SWEEPS = {
    "decay": decay_comparison,
    "delta": delta_sweep,
    "mode": mode_comparison,
    "epsilon": epsilon_sweep,
}


def generate_data(games: list[str], num_sims: int = 500, seed: int = 42):
    """Run all sweeps for each game and save JSON results."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    for game_name in games:
        out_path = DATA_DIR / f"{game_name}.json"
        print(f"  Generating: {game_name} ({num_sims} sims/config)")

        game = load_game(game_name)
        iig = IIG(game)

        # Pick deepest infoset (max upstream BFS depth) for all sweeps
        active_iid = _deepest_infoset(iig)
        print(f"    Active infoset: {iig._fmt_id(active_iid)} "
              f"(depth {max(iig.levels(active_iid).values())})")

        export = {}
        for sweep_name, config_fn in SWEEPS.items():
            configs = config_fn(game_name, num_sims, seed)
            results = run_parameter_sweep(configs, game=game, iig=iig,
                                          active_iid=active_iid)
            export[sweep_name] = []
            for r in results:
                d = r.to_dict()
                for k in ["z_star_all", "z_star_targeted",
                           "reached_all", "reached_targeted"]:
                    d[k] = {str(kk): v for kk, v in d[k].items()}
                export[sweep_name].append(d)

        with open(out_path, "w") as f:
            json.dump(export, f, indent=2)
        print(f"    → {out_path}")


def generate_plots(games: list[str], dpi: int = 150):
    """Load JSON data and produce plots for each game."""
    PLOT_DIR.mkdir(parents=True, exist_ok=True)

    for game_name in games:
        json_path = DATA_DIR / f"{game_name}.json"
        if not json_path.exists():
            print(f"  Skipping {game_name}: {json_path} not found")
            continue

        data = load_data(str(json_path))
        game_label = GAME_LABELS.get(game_name, game_name)
        figures = plot_all_sweeps(data, game_label=game_label)

        for sweep_name, fig in figures.items():
            path = PLOT_DIR / f"{game_name}_{sweep_name}.png"
            fig.savefig(path, dpi=dpi, bbox_inches="tight")
            print(f"    → {path}")
        plt.close("all")


# ── CLI ─────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="012 — Sampling distribution analysis: generate data + plots.")
    parser.add_argument("--quick", action="store_true",
                        help="Kuhn Poker only (fast sanity check)")
    parser.add_argument("--games", nargs="*", default=None,
                        choices=GAMES,
                        help="Specific games to run (default: all)")
    parser.add_argument("--num-sims", type=int, default=500,
                        help="Simulations per config (default: 500)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dpi", type=int, default=150)
    parser.add_argument("--plot-only", action="store_true",
                        help="Skip data generation, plot existing JSON")
    args = parser.parse_args()

    if args.quick:
        games = ["kuhn_poker"]
    elif args.games:
        games = args.games
    else:
        games = GAMES

    print("=" * 64)
    print("  012 — Sampling Distribution Analysis")
    print(f"  Games: {', '.join(games)}")
    print(f"  Sims/config: {args.num_sims}, Seed: {args.seed}")
    print("=" * 64)

    if not args.plot_only:
        print("\n── Data Generation ──")
        generate_data(games, num_sims=args.num_sims, seed=args.seed)

    print("\n── Plotting ──")
    generate_plots(games, dpi=args.dpi)

    print(f"\n  Done. Plots in {PLOT_DIR}/")


if __name__ == "__main__":
    main()
