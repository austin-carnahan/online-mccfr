"""Unified CLI entry point.

Usage:
    python run.py play [game]                                  # random playout
    python run.py outcome_sampling [game] [iters]              # offline OS-MCCFR
    python run.py online <game> <bot0> <bot1> [num_games]      # head-to-head match
    python run.py root_convergence [game] [--algos oos,ismcts] # root convergence
    python run.py aggregate_exploit [game] [--budgets 100,500] # aggregate exploit

Examples:
    python run.py play all
    python run.py outcome_sampling leduc_poker 50000
    python run.py online leduc_poker oos ismcts 100 --sims 1000
    python run.py root_convergence leduc_poker --checkpoints 100,1000,10000
    python run.py aggregate_exploit leduc_poker --budgets 100,500 --matches 100
"""

import sys
import importlib

COMMANDS = {
    "play":               "play_game",
    "outcome_sampling":   "src.outcome_sampling",
    "online":             "src.online",
    "root_convergence":   "experiments.root_convergence",
    "aggregate_exploit":  "experiments.aggregate_exploit",
    "plot":               "eval.compare",
}


def usage():
    print("Usage: python run.py <command> [args...]")
    print(f"\nAvailable commands: {', '.join(COMMANDS)}")
    print("\nRun 'python run.py <command> --help' for command-specific usage.")
    sys.exit(1)


def main():
    if len(sys.argv) < 2:
        usage()

    cmd = sys.argv[1]
    if cmd not in COMMANDS:
        print(f"Unknown command '{cmd}'.")
        usage()

    # Remove 'run.py <cmd>' from argv so the module sees its own args
    sys.argv = [COMMANDS[cmd]] + sys.argv[2:]

    module = importlib.import_module(COMMANDS[cmd])
    module.main()


if __name__ == "__main__":
    main()
