"""Run all benchmark experiments.

Convenience script to run both root convergence and aggregate exploitability.

Usage:
    python experiments/run_benchmark.py [--games leduc_poker,liars_dice]
"""

import sys

from experiments.root_convergence import run_experiment as run_root
from experiments.aggregate_exploit import run_experiment as run_aggregate


def main():
    games = None
    if len(sys.argv) > 1 and sys.argv[1] == "--games":
        games = sys.argv[2].split(",")

    print("\n" + "="*60)
    print("  RUNNING ROOT CONVERGENCE")
    print("="*60)
    run_root(game_names=games)

    print("\n" + "="*60)
    print("  RUNNING AGGREGATE EXPLOITABILITY")
    print("="*60)
    run_aggregate(game_names=games)


if __name__ == "__main__":
    main()
