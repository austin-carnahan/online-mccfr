"""Offline Outcome Sampling MCCFR baseline using OpenSpiel's built-in implementation.

Usage:
    python run.py outcome_sampling                            # leduc_poker, 10K iterations
    python run.py outcome_sampling leduc_poker 50000
    python run.py outcome_sampling leduc_poker 10000 -s       # + strategy table
    python run.py outcome_sampling leduc_poker 10000 -s -r    # + strategy + regret tables
    python run.py outcome_sampling leduc_poker 10000 -w       # + visit weights
"""

import sys
import time

from open_spiel.python.algorithms.outcome_sampling_mccfr import OutcomeSamplingSolver

from src.games import load_game, GAME_SPECS
from src.metrics import exploitability
from src.display import parse_display_flags, print_results, print_output


def make_checkpoints(num_iterations):
    """Generate log-spaced checkpoint iterations."""
    checkpoints = []
    val = 10
    while val < num_iterations:
        checkpoints.append(val)
        val *= 10
    checkpoints.append(num_iterations)
    return checkpoints


def run_os_baseline(game_name, num_iterations, checkpoints=None):
    """Run OS-MCCFR and measure exploitability at checkpoints.

    Returns list of (iteration, exploitability, elapsed_seconds) tuples.
    """
    game = load_game(game_name)
    solver = OutcomeSamplingSolver(game)

    if checkpoints is None:
        checkpoints = make_checkpoints(num_iterations)
    checkpoint_set = set(checkpoints)

    results = []
    t0 = time.time()

    for i in range(1, num_iterations + 1):
        solver.iteration()
        if i in checkpoint_set:
            expl = exploitability(game, solver.average_policy())
            elapsed = time.time() - t0
            results.append((i, expl, elapsed))

    return results, solver


def main():
    positional, flags = parse_display_flags(sys.argv[1:])

    game_name = positional[0] if len(positional) >= 1 else "leduc_poker"
    num_iterations = int(positional[1]) if len(positional) >= 2 else 10_000

    if game_name not in GAME_SPECS:
        print(f"Unknown game '{game_name}'. Available: {list(GAME_SPECS.keys())}")
        sys.exit(1)

    results, solver = run_os_baseline(game_name, num_iterations)
    game = load_game(game_name)
    print_results("OS-MCCFR", game_name, results)
    print_output(game, solver, flags)


if __name__ == "__main__":
    main()
