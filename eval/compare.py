"""Generate comparison plots from saved experiment results.

Usage:
    python run.py plot                        # plot everything (both experiment types, all games)
    python run.py plot root_convergence       # root convergence only
    python run.py plot aggregate_exploit      # aggregate exploitability only
    python run.py plot --results-dir results  # custom results directory
    python run.py plot --output-dir plots     # custom output directory
"""

import os
import sys

from eval.plots import (
    load_results,
    plot_all_games,
    plot_root_convergence,
    plot_aggregate_exploit,
)

DEFAULT_RESULTS_DIR = "results"
DEFAULT_OUTPUT_DIR = "results/plots"


def generate_plots(experiment_types=None, results_dir=None, output_dir=None):
    """Load saved results and generate comparison plots.

    Args:
        experiment_types: List of experiment types to plot.
            Options: "root_convergence", "aggregate_exploit". Defaults to both.
        results_dir: Base directory containing result subdirectories.
        output_dir: Where to save plot images.
    """
    if results_dir is None:
        results_dir = DEFAULT_RESULTS_DIR
    if output_dir is None:
        output_dir = DEFAULT_OUTPUT_DIR
    if experiment_types is None:
        experiment_types = ["root_convergence", "aggregate_exploit"]

    os.makedirs(output_dir, exist_ok=True)

    for exp_type in experiment_types:
        results_path = os.path.join(results_dir, exp_type, "all_results.json")
        if not os.path.exists(results_path):
            print(f"  No results found at {results_path}, skipping {exp_type}")
            continue

        all_results = load_results(results_path)

        if exp_type == "root_convergence":
            plot_fn = plot_root_convergence
            suptitle = "Root Convergence: Exploitability vs Iterations"
        else:
            plot_fn = plot_aggregate_exploit
            suptitle = "Aggregate Exploitability vs Sims/Move"

        # Multi-game comparison plot
        save_path = os.path.join(output_dir, f"{exp_type}.png")
        fig = plot_all_games(all_results, plot_fn, suptitle=suptitle,
                             save_path=save_path)
        print(f"  Saved {save_path}")

        # Individual per-game plots
        for game_name, game_data in all_results.items():
            title = f"{exp_type.replace('_', ' ').title()}: {game_name}"
            if exp_type == "root_convergence":
                fig = plot_root_convergence(game_data, title=title)
            else:
                fig = plot_aggregate_exploit(game_data, title=title)

            per_game_path = os.path.join(output_dir,
                                         f"{exp_type}_{game_name}.png")
            fig.savefig(per_game_path, dpi=150, bbox_inches="tight")
            print(f"  Saved {per_game_path}")

        import matplotlib.pyplot as plt
        plt.close("all")

    print(f"\nAll plots saved to {output_dir}/")


def main():
    """CLI entry point."""
    args = sys.argv[1:]

    experiment_types = None
    results_dir = None
    output_dir = None
    positional = []
    i = 0
    while i < len(args):
        if args[i] == "--results-dir" and i + 1 < len(args):
            results_dir = args[i + 1]
            i += 2
        elif args[i] == "--output-dir" and i + 1 < len(args):
            output_dir = args[i + 1]
            i += 2
        else:
            positional.append(args[i])
            i += 1

    if positional:
        experiment_types = positional

    generate_plots(experiment_types=experiment_types,
                   results_dir=results_dir, output_dir=output_dir)


if __name__ == "__main__":
    main()
