"""Plotting functions for convergence curves and benchmarks.

Generates matplotlib figures matching the paper's Figure 3 layout.
All functions accept results dicts as produced by the experiment harnesses
and return matplotlib Figure objects for saving or display.

Usage:
    from eval.plots import plot_root_convergence, plot_aggregate_exploit
    fig = plot_root_convergence(results)
    fig.savefig("root_convergence.png")
"""

import json

# Distinct visual styles so overlapping series remain distinguishable
_ALGO_STYLES = [
    {"marker": "o", "linestyle": "-",  "markevery": (0, 1), "markersize": 7},
    {"marker": "s", "linestyle": "--", "markevery": (0, 1), "markersize": 7},
    {"marker": "^", "linestyle": "-.", "markevery": (0, 1), "markersize": 7},
    {"marker": "D", "linestyle": ":",  "markevery": (0, 1), "markersize": 6},
    {"marker": "v", "linestyle": "-",  "markevery": (0, 1), "markersize": 7},
]


def load_results(path):
    """Load experiment results from a JSON file."""
    with open(path) as f:
        return json.load(f)


def plot_root_convergence(results, title=None):
    """Plot exploitability vs iterations for root convergence experiment.

    Args:
        results: Dict {algo_name: [{"iterations": N, "exploitability": X}, ...]}.
        title: Optional plot title.

    Returns:
        matplotlib.figure.Figure
    """
    import matplotlib.pyplot as plt
    from matplotlib.ticker import ScalarFormatter

    fig, ax = plt.subplots(figsize=(8, 5))

    for i, (algo_name, data) in enumerate(results.items()):
        if "time" in data[0]:
            xs = [d["time"] for d in data]
            x_label = "Time (s)"
        else:
            xs = [d["iterations"] for d in data]
            x_label = "Iterations"
        expls = [d["exploitability"] for d in data]
        style = _ALGO_STYLES[i % len(_ALGO_STYLES)]
        ax.plot(xs, expls, label=algo_name.upper(), linewidth=2, **style)

    ax.set_xlabel(x_label)
    ax.set_ylabel("Exploitability")
    ax.set_xscale("log")
    ax.set_yscale("log")
    yfmt = ScalarFormatter()
    yfmt.set_scientific(False)
    ax.yaxis.set_major_formatter(yfmt)
    ax.legend()
    ax.grid(True, alpha=0.3)
    if title:
        ax.set_title(title)
    fig.tight_layout()
    return fig


def plot_aggregate_exploit(results, title=None):
    """Plot aggregate exploitability vs sims-per-move.

    Args:
        results: Dict {algo_name: [{"sims_per_move": N, "exploitability": X}, ...]}.
        title: Optional plot title.

    Returns:
        matplotlib.figure.Figure
    """
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 5))

    for i, (algo_name, data) in enumerate(results.items()):
        sims = [d["sims_per_move"] for d in data]
        expls = [d["exploitability"] for d in data]
        style = _ALGO_STYLES[i % len(_ALGO_STYLES)]
        ax.plot(sims, expls, label=algo_name.upper(), linewidth=2, **style)

    ax.set_xlabel("Simulations per move")
    ax.set_ylabel("Aggregate Exploitability")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.legend()
    ax.grid(True, alpha=0.3)
    if title:
        ax.set_title(title)
    fig.tight_layout()
    return fig


def plot_all_games(all_results, plot_fn, suptitle=None, save_path=None):
    """Plot a grid of subplots, one per game.

    Args:
        all_results: Dict {game_name: {algo_name: [data]}}.
        plot_fn: Either plot_root_convergence or plot_aggregate_exploit.
        suptitle: Overall figure title.
        save_path: If provided, save figure to this path.

    Returns:
        matplotlib.figure.Figure
    """
    import matplotlib.pyplot as plt
    from matplotlib.ticker import ScalarFormatter

    games = list(all_results.keys())
    n = len(games)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 5))
    if n == 1:
        axes = [axes]

    for ax, game_name in zip(axes, games):
        game_data = all_results[game_name]
        for i, (algo_name, data) in enumerate(game_data.items()):
            if "time" in data[0]:
                x_key, x_label = "time", "Time (s)"
            elif "iterations" in data[0]:
                x_key, x_label = "iterations", "Iterations"
            else:
                x_key, x_label = "sims_per_move", "Sims/move"
            xs = [d[x_key] for d in data]
            ys = [d["exploitability"] for d in data]
            style = _ALGO_STYLES[i % len(_ALGO_STYLES)]
            ax.plot(xs, ys, label=algo_name.upper(), linewidth=2, **style)

        ax.set_xlabel(x_label)
        ax.set_ylabel("Exploitability")
        ax.set_xscale("log")
        ax.set_yscale("log")
        yfmt = ScalarFormatter()
        yfmt.set_scientific(False)
        ax.yaxis.set_major_formatter(yfmt)
        ax.set_title(game_name)
        ax.legend()
        ax.grid(True, alpha=0.3)

    if suptitle:
        fig.suptitle(suptitle, fontsize=14)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig
