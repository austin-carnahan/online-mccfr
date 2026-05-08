"""Root convergence experiment.

Runs each algorithm from the game root with a wall-clock time budget,
measuring exploitability at time checkpoints. Reproduces the paper's
Figures 3(a,d,h) where the x-axis is time in seconds.

For ISGT: runs _walk() episodes from root with uniform weights (= vanilla MCCFR).
For ISMCTS: runs run_simulation() from root, accumulating tree nodes.

Usage:
    python run.py root_convergence [game] [--algos ismcts,isgt] [--checkpoints 1,5,10,30,60,120,300]
"""

import json
import os
import sys
import time

import numpy as np

from src.games import load_game, GAME_SPECS
from src.metrics import exploitability
from src.ismcts import make_ismcts_bot
from src.isgt import ISGTBot, LevelUniform
from src.oos import OOSBot
from experiments.configs import (
    DEFAULT_ALGO_CONFIGS,
    EXPERIMENT_GAMES,
    TIME_CHECKPOINTS,
)


def run_ismcts_root_convergence(game, time_checkpoints, seed=None, **kwargs):
    """Run ISMCTS from root with a time budget, checkpointing at time thresholds."""
    from experiments.configs import ISMCTSPolicy

    rng = np.random.RandomState(seed)
    bot = make_ismcts_bot(game, player_id=0, max_simulations=1,
                          random_state=rng)

    sorted_checkpoints = sorted(time_checkpoints)
    results = []
    checkpoint_idx = 0
    iterations = 0
    t_start = time.monotonic()

    while checkpoint_idx < len(sorted_checkpoints):
        root = game.new_initial_state()
        bot.run_simulation(root)
        iterations += 1

        elapsed = time.monotonic() - t_start
        if elapsed >= sorted_checkpoints[checkpoint_idx]:
            policy = ISMCTSPolicy(game, bot)
            expl = exploitability(game, policy)
            results.append({
                "time": round(elapsed, 2),
                "iterations": iterations,
                "exploitability": expl,
            })
            print(f"    UCT  t={elapsed:>7.1f}s  iter={iterations:>8d}  expl={expl:.6f}")
            checkpoint_idx += 1

    return results


def run_isgt_root_convergence(game, time_checkpoints, seed=None, **kwargs):
    """Run ISGT from root with delta=0 (= vanilla MCCFR, no targeting), time-budgeted.

    Paper §4.4.1: OOS from root is just MCCFR with incremental tree building.
    Same applies to ISGT with delta=0 — no z* targeting, pure ε-on-policy walks.
    Both full and chance modes should behave identically at delta=0.
    """
    epsilon = kwargs.get("epsilon", 0.6)
    gamma = kwargs.get("gamma", 0.01)
    bias_mode = kwargs.get("bias_mode", "full")
    label = kwargs.get("label", f"ISGT-{bias_mode}")

    bot = ISGTBot(game, player_id=0, num_simulations=0,
                  epsilon=epsilon, gamma=gamma,
                  level_weight_fn=LevelUniform(),
                  bias_mode=bias_mode, delta=0.0,
                  seed=seed)

    sorted_checkpoints = sorted(time_checkpoints)
    results = []
    checkpoint_idx = 0
    iterations = 0
    t_start = time.monotonic()

    while checkpoint_idx < len(sorted_checkpoints):
        update_player = iterations % game.num_players()
        root = game.new_initial_state()

        # delta=0 → all iterations untargeted. Set per-iteration state.
        bot._is_targeted_iter = False
        bot._z_star_history = ()
        bot._z_star_chance_actions = []
        bot._z_star_chance_idx = 0
        bot._anchor_depth = 0

        bot._walk(root, update_player,
                  my_reach=1.0, opp_reach=1.0,
                  s1=1.0, s2=1.0)
        iterations += 1

        elapsed = time.monotonic() - t_start
        if elapsed >= sorted_checkpoints[checkpoint_idx]:
            policy = bot.average_policy()
            expl = exploitability(game, policy)
            results.append({
                "time": round(elapsed, 2),
                "iterations": iterations,
                "exploitability": expl,
            })
            print(f"    {label:8s} t={elapsed:>7.1f}s  iter={iterations:>8d}  expl={expl:.6f}")
            checkpoint_idx += 1

    return results


def run_oos_root_convergence(game, time_checkpoints, seed=None, **kwargs):
    """Run OOS from root with delta=0 (= vanilla MCCFR, no targeting), time-budgeted."""
    delta = kwargs.get("delta", 0.0)
    epsilon = kwargs.get("epsilon", 0.6)
    gamma = kwargs.get("gamma", 0.01)

    bot = OOSBot(game, player_id=0, num_simulations=0,
                 delta=delta, epsilon=epsilon, gamma=gamma, seed=seed)

    sorted_checkpoints = sorted(time_checkpoints)
    results = []
    checkpoint_idx = 0
    iterations = 0
    t_start = time.monotonic()

    while checkpoint_idx < len(sorted_checkpoints):
        update_player = iterations % game.num_players()
        root = game.new_initial_state()
        bot._oos_episode(root, update_player)
        iterations += 1

        elapsed = time.monotonic() - t_start
        if elapsed >= sorted_checkpoints[checkpoint_idx]:
            policy = bot.average_policy()
            expl = exploitability(game, policy)
            results.append({
                "time": round(elapsed, 2),
                "iterations": iterations,
                "exploitability": expl,
            })
            print(f"    OOS  t={elapsed:>7.1f}s  iter={iterations:>8d}  expl={expl:.6f}")
            checkpoint_idx += 1

    return results


ROOT_RUNNERS = {
    "ismcts": run_ismcts_root_convergence,
    "isgt": run_isgt_root_convergence,
    "isgt_full": lambda game, tc, seed=None, **kw: run_isgt_root_convergence(
        game, tc, seed=seed, bias_mode="full", label="ISGT-ful", **kw),
    "isgt_chance": lambda game, tc, seed=None, **kw: run_isgt_root_convergence(
        game, tc, seed=seed, bias_mode="chance", label="ISGT-ch", **kw),
    "oos": run_oos_root_convergence,
}


def run_experiment(game_names=None, algo_names=None, time_checkpoints=None,
                   seed=42, output_dir="results/root_convergence"):
    """Run the full root convergence experiment with time budgets."""
    if game_names is None:
        game_names = EXPERIMENT_GAMES
    if algo_names is None:
        algo_names = list(ROOT_RUNNERS.keys())
    if time_checkpoints is None:
        time_checkpoints = TIME_CHECKPOINTS

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
            results = ROOT_RUNNERS[algo_name](game, time_checkpoints,
                                              seed=seed, **kwargs)
            elapsed = time.time() - t0
            print(f"  Done in {elapsed:.1f}s")

            all_results[game_name][algo_name] = results

            out_path = os.path.join(output_dir, f"{game_name}_{algo_name}.json")
            with open(out_path, "w") as f:
                json.dump({"game": game_name, "algorithm": algo_name,
                           "checkpoints": results}, f, indent=2)

    out_path = os.path.join(output_dir, "all_results.json")
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {output_dir}/")

    return all_results


def main():
    """CLI entry point."""
    args = sys.argv[1:]

    game_names = None
    algo_names = None
    time_checkpoints = None
    seed = 42
    positional = []
    i = 0
    while i < len(args):
        if args[i] == "--algos" and i + 1 < len(args):
            algo_names = args[i + 1].split(",")
            i += 2
        elif args[i] == "--checkpoints" and i + 1 < len(args):
            time_checkpoints = [float(x) for x in args[i + 1].split(",")]
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
                   time_checkpoints=time_checkpoints, seed=seed)


if __name__ == "__main__":
    main()
