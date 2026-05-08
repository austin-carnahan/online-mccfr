"""Experiment: DD vs OOS diagnostics on Kuhn poker.

Kuhn shows the clearest DD pathology — crossover at ~1.5k sims, then DD
plateaus at ~0.042 while OOS continues dropping. This experiment collects
the full diagnostic suite at budgets spanning the crossover to understand
exactly what's going wrong in DD's variance profile.

Games: kuhn_poker only.
Algorithms: OOS (δ=0.5), DD constant(δ=0.5).
Budgets: 500, 1000, 2000, 5000.
Matches: 100 per config.

Usage:
    python -m experiments.ddelta_kuhn_diagnostics [--workers 4]
"""

import argparse
import json
import os
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pyspiel

from open_spiel.python import policy as policy_lib
from open_spiel.python.bots.policy import PolicyBot

from src.games import load_game
from src.oos import OOSBot, AVG_POLICY_INDEX
from src.depth_delta import DepthDeltaBot, ConstantSchedule
from src.metrics import exploitability
from eval.aggregate_exploitability import AggregatePolicy, _merge_bot_strategy


# ═════════════════════════════════════════════════════════════════════════
# Configuration
# ═════════════════════════════════════════════════════════════════════════

GAMES = ["kuhn_poker"]

BUDGETS = [500, 1_000, 2_000, 5_000]

MATCHES_BY_BUDGET = {
    500: 100,
    1_000: 100,
    2_000: 100,
    5_000: 100,
}

ALGORITHMS = ["oos", "depth_delta"]

EPSILON = 0.4
GAMMA = 0.01
DELTA = 0.5
SEED = 42

OUTPUT_DIR = "results/ddelta_kuhn_diagnostics"


# ═════════════════════════════════════════════════════════════════════════
# Job execution (same structure as high_budget_crossover)
# ═════════════════════════════════════════════════════════════════════════

def _count_total_info_sets(game):
    """Count total distinct information state strings in the game."""
    info_keys = set()

    def _traverse(state):
        if state.is_terminal():
            return
        if state.is_chance_node():
            for action, _ in state.chance_outcomes():
                _traverse(state.child(action))
        else:
            player = state.current_player()
            info_keys.add(state.information_state_string(player))
            for action in state.legal_actions():
                _traverse(state.child(action))

    _traverse(game.new_initial_state())
    return len(info_keys)


def _run_job(job: dict) -> dict:
    """Run a single job: play matches, collect exploitability + diagnostics."""
    game = load_game(job["game"])
    num_matches = job["num_matches"]
    sims = job["sims_per_move"]
    algo = job["algorithm"]
    seed = job["seed"]

    rng = np.random.RandomState(seed)
    global_strategy = [{}, {}]
    counter = [0]

    # Aggregated tracking stats across all steps
    all_tracking = {
        "log_W_mean": [],
        "log_W_p95": [],
        "log_W_p99": [],
        "log_W_max": [],
        "ess": [],
        "ess_per_sim": [],
        "log_l_mean": [],
        "log_l_p5": [],
        "mean_inv_l_sq": [],
        "mean_l": [],
        "min_l": [],
        "max_inv_l": [],
        "mean_sum_abs_regret": [],
        "depth_neg_log_q_sum": {},
        "depth_neg_log_q_count": {},
        "visit_count_min": [],
        "visit_count_median": [],
        "n_iters_total": 0,
    }
    if algo == "depth_delta":
        all_tracking["on_path_fraction"] = []
        all_tracking["mean_diverge_depth"] = []
    if algo == "oos":
        all_tracking["targeted_fraction"] = []

    for match_idx in range(num_matches):
        search_player = match_idx % 2
        random_player = 1 - search_player

        s = seed + counter[0]
        counter[0] += 1

        if algo == "oos":
            search_bot = OOSBot(
                game, search_player,
                num_simulations=sims,
                delta=DELTA, epsilon=EPSILON, gamma=GAMMA, seed=s,
                tracking=True,
            )
        else:
            search_bot = DepthDeltaBot(
                game, search_player,
                num_simulations=sims,
                epsilon=EPSILON, gamma=GAMMA,
                schedule=ConstantSchedule(DELTA),
                seed=s, tracking=True,
            )

        random_policy = policy_lib.UniformRandomPolicy(game)
        random_bot = PolicyBot(random_player, rng, random_policy)

        bots = [None, None]
        bots[search_player] = search_bot
        bots[random_player] = random_bot

        state = game.new_initial_state()
        for bot in bots:
            bot.restart()

        while not state.is_terminal():
            if state.is_chance_node():
                outcomes, probs = zip(*state.chance_outcomes())
                action = rng.choice(outcomes, p=probs)
                state.apply_action(action)
            else:
                player = state.current_player()
                action = bots[player].step(state)
                for p, bot in enumerate(bots):
                    if p != player:
                        bot.inform_action(state, player, action)

                if player == search_player:
                    summary = search_bot.get_tracking_summary()
                    if summary:
                        all_tracking["log_W_mean"].append(summary["log_W_mean"])
                        all_tracking["log_W_p95"].append(summary["log_W_p95"])
                        all_tracking["log_W_p99"].append(summary["log_W_p99"])
                        all_tracking["log_W_max"].append(summary["log_W_max"])
                        all_tracking["ess"].append(summary["ess"])
                        all_tracking["ess_per_sim"].append(summary["ess_per_sim"])
                        all_tracking["log_l_mean"].append(summary["log_l_mean"])
                        all_tracking["log_l_p5"].append(summary["log_l_p5"])
                        all_tracking["mean_inv_l_sq"].append(summary["mean_inv_l_sq"])
                        all_tracking["mean_l"].append(summary["mean_l"])
                        all_tracking["min_l"].append(summary["min_l"])
                        all_tracking["max_inv_l"].append(summary["max_inv_l"])
                        all_tracking["mean_sum_abs_regret"].append(summary["mean_sum_abs_regret"])
                        for d, mean_nlq in summary.get("depth_neg_log_q", {}).items():
                            d_key = str(d)
                            if d_key not in all_tracking["depth_neg_log_q_sum"]:
                                all_tracking["depth_neg_log_q_sum"][d_key] = 0.0
                                all_tracking["depth_neg_log_q_count"][d_key] = 0
                            all_tracking["depth_neg_log_q_sum"][d_key] += mean_nlq
                            all_tracking["depth_neg_log_q_count"][d_key] += 1
                        all_tracking["visit_count_min"].append(summary.get("visit_count_min", 0))
                        all_tracking["visit_count_median"].append(summary.get("visit_count_median", 0))
                        all_tracking["n_iters_total"] += summary["n_iters"]
                        if algo == "depth_delta":
                            all_tracking["on_path_fraction"].append(summary["on_path_fraction"])
                            all_tracking["mean_diverge_depth"].append(summary["mean_diverge_depth"])
                        if algo == "oos":
                            all_tracking["targeted_fraction"].append(summary["targeted_fraction"])

                state.apply_action(action)

        _merge_bot_strategy(search_bot, search_player, global_strategy)

    # Compute final exploitability
    agg_policy = AggregatePolicy(game, global_strategy)
    expl = exploitability(game, agg_policy)

    # Coverage
    total_info = _count_total_info_sets(game)
    covered = len(set(global_strategy[0].keys()) | set(global_strategy[1].keys()))

    result = {
        "game": job["game"],
        "algorithm": algo,
        "sims_per_move": sims,
        "num_matches": num_matches,
        "exploitability": expl,
        "coverage": covered,
        "total_info_sets": total_info,
        "coverage_frac": covered / total_info if total_info > 0 else 0.0,
    }

    if all_tracking["mean_inv_l_sq"]:
        arr_log_W_mean = np.array(all_tracking["log_W_mean"])
        arr_log_W_p95 = np.array(all_tracking["log_W_p95"])
        arr_log_W_p99 = np.array(all_tracking["log_W_p99"])
        arr_log_W_max = np.array(all_tracking["log_W_max"])
        arr_ess = np.array(all_tracking["ess"])
        arr_ess_per_sim = np.array(all_tracking["ess_per_sim"])
        arr_log_l_mean = np.array(all_tracking["log_l_mean"])
        arr_log_l_p5 = np.array(all_tracking["log_l_p5"])
        arr_w2 = np.array(all_tracking["mean_inv_l_sq"])
        arr_l = np.array(all_tracking["mean_l"])
        arr_min_l = np.array(all_tracking["min_l"])
        arr_max_inv = np.array(all_tracking["max_inv_l"])
        arr_regret = np.array(all_tracking["mean_sum_abs_regret"])
        arr_vc_min = np.array(all_tracking["visit_count_min"])
        arr_vc_med = np.array(all_tracking["visit_count_median"])

        depth_neg_log_q = {}
        for d_key in sorted(all_tracking["depth_neg_log_q_sum"].keys(), key=lambda x: int(x)):
            count = all_tracking["depth_neg_log_q_count"][d_key]
            if count > 0:
                depth_neg_log_q[d_key] = all_tracking["depth_neg_log_q_sum"][d_key] / count

        result["tracking"] = {
            "log_W_mean": float(arr_log_W_mean.mean()),
            "log_W_p95_mean": float(arr_log_W_p95.mean()),
            "log_W_p99_mean": float(arr_log_W_p99.mean()),
            "log_W_max_mean": float(arr_log_W_max.mean()),
            "log_W_max_max": float(arr_log_W_max.max()),
            "ess_mean": float(arr_ess.mean()),
            "ess_per_sim_mean": float(arr_ess_per_sim.mean()),
            "ess_per_sim_min": float(arr_ess_per_sim.min()),
            "log_l_mean": float(arr_log_l_mean.mean()),
            "log_l_p5_mean": float(arr_log_l_p5.mean()),
            "mean_E_W2": float(arr_w2.mean()),
            "std_E_W2": float(arr_w2.std()),
            "max_E_W2": float(arr_w2.max()),
            "p95_E_W2": float(np.percentile(arr_w2, 95)),
            "mean_l": float(arr_l.mean()),
            "mean_min_l": float(arr_min_l.mean()),
            "p5_min_l": float(np.percentile(arr_min_l, 5)),
            "mean_max_inv_l": float(arr_max_inv.mean()),
            "p95_max_inv_l": float(np.percentile(arr_max_inv, 95)),
            "depth_neg_log_q": depth_neg_log_q,
            "mean_abs_regret": float(arr_regret.mean()),
            "std_abs_regret": float(arr_regret.std()),
            "visit_count_min_mean": float(arr_vc_min.mean()),
            "visit_count_median_mean": float(arr_vc_med.mean()),
            "n_steps": len(all_tracking["mean_inv_l_sq"]),
            "n_iters_total": all_tracking["n_iters_total"],
        }

        if algo == "depth_delta" and all_tracking["on_path_fraction"]:
            arr_on_path = np.array(all_tracking["on_path_fraction"])
            arr_div = np.array(all_tracking["mean_diverge_depth"])
            result["tracking"]["mean_on_path_frac"] = float(arr_on_path.mean())
            result["tracking"]["mean_diverge_depth"] = float(arr_div[arr_div >= 0].mean()) if (arr_div >= 0).any() else -1.0
            result["tracking"]["std_diverge_depth"] = float(arr_div[arr_div >= 0].std()) if (arr_div >= 0).any() else 0.0

        if algo == "oos" and all_tracking["targeted_fraction"]:
            arr_targeted = np.array(all_tracking["targeted_fraction"])
            result["tracking"]["mean_targeted_frac"] = float(arr_targeted.mean())

    return result


# ═════════════════════════════════════════════════════════════════════════
# Orchestration
# ═════════════════════════════════════════════════════════════════════════

def build_jobs():
    """Expand experiment grid into job dicts."""
    jobs = []
    for game_name in GAMES:
        for algo in ALGORITHMS:
            for sims in BUDGETS:
                jobs.append({
                    "game": game_name,
                    "algorithm": algo,
                    "sims_per_move": sims,
                    "num_matches": MATCHES_BY_BUDGET[sims],
                    "seed": SEED,
                })
    return jobs


def main():
    parser = argparse.ArgumentParser(description="DD vs OOS Kuhn diagnostics")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--output", type=str, default=OUTPUT_DIR)
    args = parser.parse_args()

    jobs = build_jobs()
    output_dir = args.output
    os.makedirs(output_dir, exist_ok=True)

    total = len(jobs)
    print(f"DD Kuhn Diagnostics — {total} jobs")
    print(f"  Game: kuhn_poker")
    print(f"  Algorithms: {ALGORITHMS}")
    print(f"  Budgets: {BUDGETS}")
    print(f"  Workers: {args.workers}")
    print()

    results = []
    errors = []
    completed = 0
    t_start = time.time()

    incremental_path = os.path.join(output_dir, "incremental_results.jsonl")

    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        future_to_job = {}
        for job in jobs:
            f = pool.submit(_run_job, job)
            future_to_job[f] = job

        for future in as_completed(future_to_job):
            completed += 1
            elapsed_total = time.time() - t_start
            est_remaining = (elapsed_total / completed) * (total - completed) if completed > 0 else 0

            try:
                result = future.result()
                results.append(result)

                with open(incremental_path, "a") as incr:
                    incr.write(json.dumps(result) + "\n")

                tracking_str = ""
                if "tracking" in result:
                    t = result["tracking"]
                    if "mean_on_path_frac" in t:
                        tracking_str = (
                            f"  E[W²]={t['mean_E_W2']:.1f} "
                            f"on_path={t['mean_on_path_frac']:.2f}"
                        )
                    elif "mean_targeted_frac" in t:
                        tracking_str = (
                            f"  E[W²]={t['mean_E_W2']:.1f} "
                            f"targeted={t['mean_targeted_frac']:.2f}"
                        )
                    else:
                        tracking_str = f"  E[W²]={t['mean_E_W2']:.1f}"

                print(
                    f"  [{completed:>3d}/{total}] "
                    f"{result['algorithm']:12s} "
                    f"sims={result['sims_per_move']:>5d}  "
                    f"expl={result['exploitability']:.6f}  "
                    f"cov={result['coverage']}/{result['total_info_sets']}"
                    f"{tracking_str}"
                    f"  ({elapsed_total:.0f}s elapsed)"
                )
            except Exception as exc:
                job = future_to_job[future]
                print(
                    f"  [{completed:>3d}/{total}] FAILED "
                    f"{job['algorithm']} sims={job['sims_per_move']}: {exc}"
                )
                errors.append({
                    "job": job,
                    "error": str(exc),
                    "traceback": traceback.format_exc(),
                })

    # Sort and save
    results.sort(key=lambda r: (r["algorithm"], r["sims_per_move"]))

    results_path = os.path.join(output_dir, "results.json")
    with open(results_path, "w") as f:
        json.dump({
            "config": {
                "games": GAMES,
                "algorithms": ALGORITHMS,
                "budgets": BUDGETS,
                "matches_by_budget": MATCHES_BY_BUDGET,
                "epsilon": EPSILON,
                "gamma": GAMMA,
                "delta": DELTA,
                "seed": SEED,
                "dd_schedule": "constant(δ=0.5)",
            },
            "results": results,
        }, f, indent=2)

    if errors:
        errors_path = os.path.join(output_dir, "errors.json")
        with open(errors_path, "w") as f:
            json.dump(errors, f, indent=2)
        print(f"\n  {len(errors)} jobs FAILED — see {errors_path}")

    total_elapsed = time.time() - t_start
    print(f"\nDone. {len(results)}/{total} jobs completed in {total_elapsed:.1f}s.")
    print(f"Results saved to {results_path}")

    # Print summary
    print("\n" + "=" * 80)
    print("  SUMMARY")
    print("=" * 80)
    print(f"\n  kuhn_poker:")
    for algo in ALGORITHMS:
        algo_results = sorted(
            [r for r in results if r["algorithm"] == algo],
            key=lambda r: r["sims_per_move"]
        )
        if algo_results:
            line = f"    {algo:12s}  "
            line += "  ".join(
                f"{r['sims_per_move']}:{r['exploitability']:.4f}"
                for r in algo_results
            )
            print(line)


if __name__ == "__main__":
    main()
