"""Pilot: Observable-Node / Every-Node / Depth-LOTR vs OOS on Kuhn poker.

Four algorithms, Kuhn-only, four budgets, 4 workers.  This pilot now
includes v3 (Observable-Node LOTR) — same τ-blend as v2 but with the
IST observability gate restored so hidden chance is sampled from
natural π exactly like OOS-IST.

Algorithms:
  - oos_d05      OOS-IST, δ=0.5
  - v1_step_r05  Depth-LOTR step(ρ=0.5,d=0): force observable chance
                 every iter, τ-blend at first decision.
  - v2_step_r05  Every-Node LOTR step(ρ=0.5,d=0): τ-blend at every
                 prefix node including hidden chance (degenerate).
  - v3_step_r05  Observable-Node LOTR step(ρ=0.5,d=0): τ-blend at
                 decisions and observable chance only; hidden chance
                 ~ natural π.  Apples-to-apples with OOS_d05 modulo
                 q-form (per-iter coin vs per-node coin).

ρ_LOTR = 0.5 ↔ δ_OOS = 0.5 — same prefix-divergence rate.

Budgets: 10k, 20k, 50k, 100k.  100 matches each.
"""

import argparse
import json
import os
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np

from open_spiel.python import policy as policy_lib
from open_spiel.python.bots.policy import PolicyBot

from src.games import load_game
from src.oos import OOSBot
from src.depth_lotr import DepthLOTRBot, step as depth_step
from src.every_node_lotr import EveryNodeLOTRBot, step as every_step
from src.observable_node_lotr import ObservableNodeLOTRBot, step as obs_step
from src.lotr import LOTRBot, step as lotr_step
from src.metrics import exploitability
from eval.aggregate_exploitability import AggregatePolicy, _merge_bot_strategy


# ═════════════════════════════════════════════════════════════════════════
# Configuration
# ═════════════════════════════════════════════════════════════════════════

GAMES = ["kuhn_poker"]
BUDGETS = [10_000, 20_000, 50_000, 100_000]
MATCHES_PER_BUDGET = 100

ALGORITHMS = ["oos_d05", "v1_step_r05", "v2_step_r05", "v3_step_r05", "lotr_step_r05"]

OOS_DELTAS = {"oos_d05": 0.5}
LOTR_RHOS = {"v1_step_r05": 0.5, "v2_step_r05": 0.5, "v3_step_r05": 0.5, "lotr_step_r05": 0.5}

EPSILON = 0.4
GAMMA = 0.01
SEED = 42

OUTPUT_DIR = "results/every_node_lotr_kuhn_pilot"


# ═════════════════════════════════════════════════════════════════════════
# Helpers
# ═════════════════════════════════════════════════════════════════════════

def _count_total_info_sets(game):
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


def _build_bot(algo, game, player, sims, seed):
    if algo.startswith("oos"):
        return OOSBot(
            game, player,
            num_simulations=sims,
            delta=OOS_DELTAS[algo], epsilon=EPSILON, gamma=GAMMA, seed=seed,
            tracking=True,
        )
    if algo.startswith("v1"):
        schedule = depth_step(LOTR_RHOS[algo], depth=0)
        return DepthLOTRBot(
            game, player,
            num_simulations=sims,
            epsilon=EPSILON, gamma=GAMMA, schedule=schedule,
            seed=seed, tracking=True,
        )
    if algo.startswith("v2"):
        schedule = every_step(LOTR_RHOS[algo], depth=0)
        return EveryNodeLOTRBot(
            game, player,
            num_simulations=sims,
            epsilon=EPSILON, gamma=GAMMA, schedule=schedule,
            seed=seed, tracking=True,
        )
    if algo.startswith("v3"):
        schedule = obs_step(LOTR_RHOS[algo], depth=0)
        return ObservableNodeLOTRBot(
            game, player,
            num_simulations=sims,
            epsilon=EPSILON, gamma=GAMMA, schedule=schedule,
            seed=seed, tracking=True,
        )
    if algo.startswith("lotr"):
        schedule = lotr_step(LOTR_RHOS[algo], depth=0)
        return LOTRBot(
            game, player,
            num_simulations=sims,
            epsilon=EPSILON, gamma=GAMMA, schedule=schedule,
            seed=seed, tracking=True,
        )
    raise ValueError(f"Unknown algorithm: {algo}")


# ═════════════════════════════════════════════════════════════════════════
# Job
# ═════════════════════════════════════════════════════════════════════════

def _run_job(job: dict) -> dict:
    game = load_game(job["game"])
    num_matches = job["num_matches"]
    sims = job["sims_per_move"]
    algo = job["algorithm"]
    seed = job["seed"]

    rng = np.random.RandomState(seed)
    global_strategy = [{}, {}]
    counter = [0]

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
        "mean_sum_sq_regret": [],
        "p95_sum_sq_regret": [],
        "max_sum_sq_regret": [],
        "depth_neg_log_q_sum": {},
        "depth_neg_log_q_count": {},
        "visit_count_min": [],
        "visit_count_median": [],
        "n_iters_total": 0,
        "prefix_survival_fraction": [],
    }
    if algo.startswith(("v1", "v2", "v3", "lotr")):
        all_tracking["on_path_fraction"] = []
        all_tracking["on_path_given_chance_stayed"] = []
        all_tracking["chance_diverged_fraction"] = []
        all_tracking["mean_diverge_depth"] = []
        all_tracking["prefix_D"] = []
    if algo.startswith("oos"):
        all_tracking["targeted_fraction"] = []

    for match_idx in range(num_matches):
        search_player = match_idx % 2
        random_player = 1 - search_player

        s = seed + counter[0]
        counter[0] += 1

        search_bot = _build_bot(algo, game, search_player, sims, s)

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
                        all_tracking["mean_sum_sq_regret"].append(summary["mean_sum_sq_regret"])
                        all_tracking["p95_sum_sq_regret"].append(summary["p95_sum_sq_regret"])
                        all_tracking["max_sum_sq_regret"].append(summary["max_sum_sq_regret"])
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
                        if "prefix_survival_fraction" in summary:
                            all_tracking["prefix_survival_fraction"].append(
                                summary["prefix_survival_fraction"])
                        if algo.startswith(("v1", "v2", "v3", "lotr")):
                            all_tracking["on_path_fraction"].append(summary["on_path_fraction"])
                            all_tracking["on_path_given_chance_stayed"].append(
                                summary["on_path_given_chance_stayed"])
                            all_tracking["chance_diverged_fraction"].append(
                                summary["chance_diverged_fraction"])
                            all_tracking["mean_diverge_depth"].append(summary["mean_diverge_depth"])
                            all_tracking["prefix_D"].append(summary["prefix_D"])
                        if algo.startswith("oos") and "targeted_fraction" in summary:
                            all_tracking["targeted_fraction"].append(summary["targeted_fraction"])

                state.apply_action(action)

        _merge_bot_strategy(search_bot, search_player, global_strategy)

    agg_policy = AggregatePolicy(game, global_strategy)
    expl = exploitability(game, agg_policy)

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
        arr_log_W_p99 = np.array(all_tracking["log_W_p99"])
        arr_log_W_max = np.array(all_tracking["log_W_max"])
        arr_ess = np.array(all_tracking["ess"])
        arr_ess_per_sim = np.array(all_tracking["ess_per_sim"])
        arr_w2 = np.array(all_tracking["mean_inv_l_sq"])
        arr_regret = np.array(all_tracking["mean_sum_abs_regret"])
        arr_sq_regret = np.array(all_tracking["mean_sum_sq_regret"])
        arr_p95_sq = np.array(all_tracking["p95_sum_sq_regret"])
        arr_max_sq = np.array(all_tracking["max_sum_sq_regret"])

        depth_neg_log_q = {}
        for d_key in sorted(all_tracking["depth_neg_log_q_sum"].keys(), key=lambda x: int(x)):
            count = all_tracking["depth_neg_log_q_count"][d_key]
            if count > 0:
                depth_neg_log_q[d_key] = all_tracking["depth_neg_log_q_sum"][d_key] / count

        result["tracking"] = {
            "log_W_mean": float(arr_log_W_mean.mean()),
            "log_W_p99_mean": float(arr_log_W_p99.mean()),
            "log_W_max_mean": float(arr_log_W_max.mean()),
            "log_W_max_max": float(arr_log_W_max.max()),
            "ess_mean": float(arr_ess.mean()),
            "ess_per_sim_mean": float(arr_ess_per_sim.mean()),
            "ess_per_sim_min": float(arr_ess_per_sim.min()),
            "mean_E_W2": float(arr_w2.mean()),
            "p95_E_W2": float(np.percentile(arr_w2, 95)),
            "max_E_W2": float(arr_w2.max()),
            "depth_neg_log_q": depth_neg_log_q,
            "mean_abs_regret": float(arr_regret.mean()),
            "mean_E_dR2": float(arr_sq_regret.mean()),
            "p95_E_dR2": float(arr_p95_sq.mean()),
            "max_E_dR2": float(arr_max_sq.max()),
            "n_iters_total": all_tracking["n_iters_total"],
        }

        if all_tracking["prefix_survival_fraction"]:
            arr_psurv = np.array(all_tracking["prefix_survival_fraction"])
            result["tracking"]["mean_prefix_survival"] = float(arr_psurv.mean())
            result["tracking"]["std_prefix_survival"] = float(arr_psurv.std())

        if algo.startswith(("v1", "v2", "v3", "lotr")) and all_tracking["on_path_fraction"]:
            arr_on_path = np.array(all_tracking["on_path_fraction"])
            arr_on_path_cond = np.array(all_tracking["on_path_given_chance_stayed"])
            arr_chance_div = np.array(all_tracking["chance_diverged_fraction"])
            arr_div = np.array(all_tracking["mean_diverge_depth"])
            arr_D = np.array(all_tracking["prefix_D"])
            result["tracking"]["mean_on_path_frac"] = float(arr_on_path.mean())
            result["tracking"]["mean_on_path_given_chance"] = float(arr_on_path_cond.mean())
            result["tracking"]["mean_chance_diverged_frac"] = float(arr_chance_div.mean())
            result["tracking"]["mean_diverge_depth"] = (
                float(arr_div[arr_div >= 0].mean()) if (arr_div >= 0).any() else -1.0)
            result["tracking"]["mean_prefix_D"] = float(arr_D.mean())
            result["tracking"]["max_prefix_D"] = int(arr_D.max())

        if algo.startswith("oos") and all_tracking["targeted_fraction"]:
            arr_targeted = np.array(all_tracking["targeted_fraction"])
            result["tracking"]["mean_targeted_frac"] = float(arr_targeted.mean())

    return result


# ═════════════════════════════════════════════════════════════════════════
# Orchestration
# ═════════════════════════════════════════════════════════════════════════

def build_jobs():
    jobs = []
    for game_name in GAMES:
        for algo in ALGORITHMS:
            for sims in BUDGETS:
                jobs.append({
                    "game": game_name,
                    "algorithm": algo,
                    "sims_per_move": sims,
                    "num_matches": MATCHES_PER_BUDGET,
                    "seed": SEED,
                })
    return jobs


def main():
    parser = argparse.ArgumentParser(description="Every-Node LOTR Kuhn pilot")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--output", type=str, default=OUTPUT_DIR)
    args = parser.parse_args()

    jobs = build_jobs()
    output_dir = args.output
    os.makedirs(output_dir, exist_ok=True)

    total = len(jobs)
    print(f"Every-Node LOTR Kuhn pilot — {total} jobs")
    print(f"  Games: {GAMES}")
    print(f"  Algorithms: {ALGORITHMS}")
    print(f"    OOS δ: {OOS_DELTAS}")
    print(f"    LOTR ρ (step depth profile @ d=0): {LOTR_RHOS}")
    print(f"  Budgets: {BUDGETS}")
    print(f"  Matches per (algo,budget): {MATCHES_PER_BUDGET}")
    print(f"  Workers: {args.workers}")
    print()

    results = []
    errors = []
    completed = 0
    t_start = time.time()

    incremental_path = os.path.join(output_dir, "incremental_results.jsonl")

    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        future_to_job = {pool.submit(_run_job, j): j for j in jobs}

        for future in as_completed(future_to_job):
            completed += 1
            elapsed_total = time.time() - t_start
            est_remaining = (elapsed_total / completed) * (total - completed)

            try:
                result = future.result()
                results.append(result)

                with open(incremental_path, "a") as incr:
                    incr.write(json.dumps(result) + "\n")

                tracking_str = ""
                if "tracking" in result:
                    t = result["tracking"]
                    dr2 = t.get("mean_E_dR2", float("nan"))
                    psurv = t.get("mean_prefix_survival", float("nan"))
                    if "mean_on_path_given_chance" in t:
                        tracking_str = (
                            f"  E[W²]={t['mean_E_W2']:.1e} E[ΔR²]={dr2:.1e} "
                            f"on_path|c={t['mean_on_path_given_chance']:.2f} "
                            f"surv={psurv:.2f} D={t['mean_prefix_D']:.1f}"
                        )
                    elif "mean_targeted_frac" in t:
                        tracking_str = (
                            f"  E[W²]={t['mean_E_W2']:.1e} E[ΔR²]={dr2:.1e} "
                            f"targeted={t['mean_targeted_frac']:.2f} "
                            f"surv={psurv:.2f}"
                        )

                print(
                    f"  [{completed:>3d}/{total}] "
                    f"{result['game']:12s} {result['algorithm']:13s} "
                    f"sims={result['sims_per_move']:>7d}  "
                    f"expl={result['exploitability']:.6f}  "
                    f"cov={result['coverage']}/{result['total_info_sets']}"
                    f"{tracking_str}"
                    f"  ({elapsed_total:.0f}s, ETA {est_remaining/60:.1f}m)"
                )
            except Exception as exc:
                job = future_to_job[future]
                print(
                    f"  [{completed:>3d}/{total}] FAILED "
                    f"{job['game']} {job['algorithm']} sims={job['sims_per_move']}: {exc}"
                )
                errors.append({
                    "job": job,
                    "error": str(exc),
                    "traceback": traceback.format_exc(),
                })

    results.sort(key=lambda r: (r["game"], r["algorithm"], r["sims_per_move"]))

    results_path = os.path.join(output_dir, "results.json")
    with open(results_path, "w") as f:
        json.dump({
            "config": {
                "games": GAMES,
                "algorithms": ALGORITHMS,
                "budgets": BUDGETS,
                "matches_per_budget": MATCHES_PER_BUDGET,
                "epsilon": EPSILON,
                "gamma": GAMMA,
                "oos_deltas": OOS_DELTAS,
                "lotr_rhos": LOTR_RHOS,
                "lotr_depth_profile": "step(d=0)",
                "seed": SEED,
            },
            "results": results,
        }, f, indent=2)

    if errors:
        errors_path = os.path.join(output_dir, "errors.json")
        with open(errors_path, "w") as f:
            json.dump(errors, f, indent=2)
        print(f"\n  {len(errors)} jobs FAILED — see {errors_path}")

    total_elapsed = time.time() - t_start
    print(f"\nDone. {len(results)}/{total} jobs in {total_elapsed/60:.1f} minutes.")
    print(f"Results saved to {results_path}")

    print("\n" + "=" * 80)
    print("  EXPLOITABILITY SUMMARY")
    print("=" * 80)
    for game_name in GAMES:
        print(f"\n  {game_name}:")
        game_results = [r for r in results if r["game"] == game_name]
        for algo in ALGORITHMS:
            algo_results = sorted(
                [r for r in game_results if r["algorithm"] == algo],
                key=lambda r: r["sims_per_move"]
            )
            if algo_results:
                line = f"    {algo:13s}  "
                line += "  ".join(
                    f"{r['sims_per_move']//1000}k:{r['exploitability']:.4f}"
                    for r in algo_results
                )
                print(line)


if __name__ == "__main__":
    main()
