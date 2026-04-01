"""Root convergence experiment.

Runs each algorithm from the game root (empty match history) with increasing
simulation budgets, measuring exploitability at each checkpoint. Reproduces
the paper's Figures 3(a,d,h).

For OOS: runs N iterations of _oos_episode() from root, extracts OOSPolicy,
    computes exploitability.
For ISMCTS: runs run_search() with N max_simulations from the root state,
    extracts empirical action frequencies from the tree, computes exploitability.

Usage:
    python run.py root_convergence [game] [--algos oos,ismcts] [--checkpoints 100,500,1000]
"""

import json
import os
import sys
import time

import numpy as np

from src.games import load_game, GAME_SPECS
from src.metrics import exploitability
from src.oos import OOSBot
from src.ismcts import make_ismcts_bot
from experiments.configs import (
    ALGORITHM_REGISTRY,
    DEFAULT_ALGO_CONFIGS,
    EXPERIMENT_GAMES,
    ROOT_CHECKPOINTS,
)


def run_oos_root_convergence(game, checkpoints, seed=None, **kwargs):
    """Run OOS from root, recording exploitability at each checkpoint.

    OOS iterations accumulate — we run from 0 up to max(checkpoints),
    checkpointing at each threshold.

    Returns:
        list of (iterations, exploitability) tuples.
    """
    delta = kwargs.get("delta", 0.9)
    epsilon = kwargs.get("epsilon", 0.6)
    gamma = kwargs.get("gamma", 0.01)

    bot = OOSBot(game, player_id=0, num_simulations=0,
                 delta=delta, epsilon=epsilon, gamma=gamma, seed=seed)

    sorted_checkpoints = sorted(checkpoints)
    results = []
    iterations_done = 0

    for target in sorted_checkpoints:
        iters_needed = target - iterations_done
        for _ in range(iters_needed):
            update_player = iterations_done % game.num_players()
            root = game.new_initial_state()
            bot._oos_episode(root, update_player)
            iterations_done += 1

        policy = bot.average_policy()
        expl = exploitability(game, policy)
        results.append({"iterations": target, "exploitability": expl})
        print(f"    OOS  iter={target:>6d}  expl={expl:.6f}")

    return results


def run_ismcts_root_convergence(game, checkpoints, seed=None, **kwargs):
    """Run ISMCTS from root, recording exploitability at each checkpoint.

    Calls run_simulation() directly from the game root, accumulating
    tree nodes across all simulations. This handles chance-starting games
    (e.g. Leduc, Liar's Dice) correctly — run_simulation() samples
    chance outcomes internally and builds info-set nodes for all players.

    Simulations accumulate (like OOS) — we run from 0 up to max(checkpoints),
    checkpointing at each threshold.

    Returns:
        list of (iterations, exploitability) tuples.
    """
    from experiments.configs import ISMCTSPolicy

    rng = np.random.RandomState(seed)
    bot = make_ismcts_bot(game, player_id=0, max_simulations=1,
                          random_state=rng)

    sorted_checkpoints = sorted(checkpoints)
    results = []
    sims_done = 0

    for target in sorted_checkpoints:
        sims_needed = target - sims_done
        for _ in range(sims_needed):
            root = game.new_initial_state()
            bot.run_simulation(root)
            sims_done += 1

        policy = ISMCTSPolicy(game, bot)
        expl = exploitability(game, policy)
        results.append({"iterations": target, "exploitability": expl})
        print(f"    UCT  iter={target:>6d}  expl={expl:.6f}")

    return results


# Registry of root convergence runners
ROOT_RUNNERS = {
    "oos": run_oos_root_convergence,
    "ismcts": run_ismcts_root_convergence,
}


def run_experiment(game_names=None, algo_names=None, checkpoints=None,
                   seed=42, output_dir="results/root_convergence"):
    """Run the full root convergence experiment.

    Args:
        game_names: List of game short names. Defaults to all.
        algo_names: List of algorithm short names. Defaults to all.
        checkpoints: List of iteration counts. Defaults to ROOT_CHECKPOINTS.
        seed: Random seed.
        output_dir: Where to save JSON results.

    Returns:
        dict: {game: {algo: [results]}}
    """
    if game_names is None:
        game_names = EXPERIMENT_GAMES
    if algo_names is None:
        algo_names = list(ROOT_RUNNERS.keys())
    if checkpoints is None:
        checkpoints = ROOT_CHECKPOINTS

    os.makedirs(output_dir, exist_ok=True)
    all_results = {}

    for game_name in game_names:
        print(f"\n{'='*50}")
        print(f"  Root Convergence: {game_name}")
        print(f"{'='*50}")
        game = load_game(game_name)
        all_results[game_name] = {}

        for algo_name in algo_names:
            if algo_name not in ROOT_RUNNERS:
                print(f"  Skipping unknown algorithm: {algo_name}")
                continue

            print(f"\n  Algorithm: {algo_name}")
            t0 = time.time()
            kwargs = DEFAULT_ALGO_CONFIGS.get(algo_name, {})
            results = ROOT_RUNNERS[algo_name](game, checkpoints,
                                              seed=seed, **kwargs)
            elapsed = time.time() - t0
            print(f"  Done in {elapsed:.1f}s")

            all_results[game_name][algo_name] = results

            # Save per-game-algo results
            out_path = os.path.join(output_dir, f"{game_name}_{algo_name}.json")
            with open(out_path, "w") as f:
                json.dump({"game": game_name, "algorithm": algo_name,
                           "checkpoints": results}, f, indent=2)

    # Save combined results
    out_path = os.path.join(output_dir, "all_results.json")
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {output_dir}/")

    return all_results


def main():
    """CLI entry point."""
    args = sys.argv[1:]

    # Parse flags
    game_names = None
    algo_names = None
    checkpoints = None
    seed = 42
    positional = []
    i = 0
    while i < len(args):
        if args[i] == "--algos" and i + 1 < len(args):
            algo_names = args[i + 1].split(",")
            i += 2
        elif args[i] == "--checkpoints" and i + 1 < len(args):
            checkpoints = [int(x) for x in args[i + 1].split(",")]
            i += 2
        elif args[i] == "--seed" and i + 1 < len(args):
            seed = int(args[i + 1])
            i += 2
        else:
            positional.append(args[i])
            i += 1

    if positional:
        game_names = positional

    run_experiment(game_names=game_names, algo_names=algo_names,
                   checkpoints=checkpoints, seed=seed)


if __name__ == "__main__":
    main()
