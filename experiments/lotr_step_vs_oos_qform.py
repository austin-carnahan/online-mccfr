"""Experiment: LOTR(step at root) vs OOS — isolating the q-form advantage.

When the LOTR depth profile is a step concentrated at d=0:

    w(0) = 1,   w(d>0) = 0    ⇒   τ(0) = 1−ρ,  τ(d>0) = 1

each episode flips a single coin at the prefix root — with probability ρ
it diverges and samples freely from σ_ε for the rest of the trajectory,
otherwise it follows the full target prefix.  This recreates OOS's
targeted/untargeted dichotomy as a special case of LOTR.

What differs is the *sampling distribution* on the targeted-mode
episodes:

  - OOS targeted:  forces target a* every prefix step.  Local q over
    the rest of the trajectory comes from σ_ε mixed with the targeting
    coin at root level (per-iteration coin).
  - LOTR step:     samples target with prob τ(0)=1−ρ at the root, then
    τ=1 (always target) at every subsequent prefix node — i.e. q is a
    product of realized local probabilities throughout, no global
    mixture coin.

Same expected on-path rate (δ_OOS ↔ 1−ρ_LOTR), same divergence
allocation (single root coin), same other knobs (ε, γ).  Any
difference in exploitability / E[W²] / ESS is therefore attributable
to the *product-form local q* alone — demonstrating (or refuting)
whether LOTR's denominator structure is a real advantage.

If LOTR-step matches OOS, OOS is just a specialization of LOTR.  If it
beats OOS, the product-form q is the mechanism.

Apples-to-apples note (prefix survival):
    Both bots emit `prefix_survival_fraction` as a secondary diagnostic
    — the realized fraction of episodes whose actions matched the
    match-history target at every prefix decision and observable chance
    node.  This metric is *not* a direct cross-algorithm calibration:
    LOTR-step at ρ only flips its d=0 coin on episodes that have a
    player decision on the prefix (others trivially score 1.0), while
    OOS at δ gates all observable prefix nodes with one global coin.
    Use the per-algo nominal knobs (δ for OOS, 1−ρ for LOTR-step) for
    matched-pair claims; use `mean_prefix_survival` as a sanity check
    that each bot is hitting its expected exploration regime.

Games: leduc_poker, goofspiel, liars_dice.
Budgets: 5k, 20k, 100k, 500k.
Matches: 100 (≤100k), 30 (500k).
Total: 3 games × 4 arms × 4 budgets = 48 jobs.

Usage:
    python -m experiments.lotr_step_vs_oos_qform [--workers 5] [--pilot]
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
from src.depth_lotr import DepthLOTRBot, step
from src.metrics import exploitability
from eval.aggregate_exploitability import AggregatePolicy, _merge_bot_strategy


# ═════════════════════════════════════════════════════════════════════════
# Configuration
# ═════════════════════════════════════════════════════════════════════════

GAMES = ["kuhn_poker", "leduc_poker", "goofspiel", "liars_dice"]

BUDGETS = [1_000, 5_000, 20_000, 100_000, 250_000, 500_000]

# Per-game, per-budget match counts.  None = skip that cell.
# Tapered: 100 at low/mid budgets, 50 at 250k, 30 at 500k.
# Kuhn is small enough to be converged by 100k; skip 250k and 500k there.
MATCHES_BY_GAME_BUDGET = {
    "kuhn_poker":  {1_000: 100, 5_000: 100, 20_000: 100, 100_000: 50,
                    250_000: None, 500_000: None},
    "leduc_poker": {1_000: 100, 5_000: 100, 20_000: 100, 100_000: 100,
                    250_000: 50, 500_000: 30},
    "goofspiel":   {1_000: 100, 5_000: 100, 20_000: 100, 100_000: 100,
                    250_000: 50, 500_000: 30},
    "liars_dice":  {1_000: 100, 5_000: 100, 20_000: 100, 100_000: 100,
                    250_000: 50, 500_000: 30},
}

ALGORITHMS = ["oos_d05", "oos_d09", "lotr_step_r05", "lotr_step_r01"]

# Per-algorithm parameters.  Pairs share prefix-divergence rate:
#   oos_d09 ↔ lotr_step_r01  (10% divergence at root only)
#   oos_d05 ↔ lotr_step_r05  (50% divergence at root only)
OOS_DELTAS = {"oos_d05": 0.5, "oos_d09": 0.9}
LOTR_RHOS  = {"lotr_step_r05": 0.5, "lotr_step_r01": 0.1}

EPSILON = 0.4
GAMMA = 0.01
SEED = 42

OUTPUT_DIR = "results/lotr_step_vs_oos_qform"


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
        # Unified apples-to-apples metric (computed identically in both
        # bots): fraction of episodes whose realized actions matched the
        # match-history target at every player decision and observable
        # chance node on the prefix.  For LOTR-step at ρ this should
        # equal 1−ρ to within sampling noise; for OOS at δ it has a
        # small positive slack vs δ because untargeted episodes can
        # accidentally play the target sequence.
        "prefix_survival_fraction": [],
    }
    if algo.startswith("lotr"):
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

        if algo.startswith("oos"):
            search_bot = OOSBot(
                game, search_player,
                num_simulations=sims,
                delta=OOS_DELTAS[algo], epsilon=EPSILON, gamma=GAMMA, seed=s,
                tracking=True,
            )
        elif algo.startswith("lotr"):
            schedule = step(LOTR_RHOS[algo], depth=0)
            search_bot = DepthLOTRBot(
                game, search_player,
                num_simulations=sims,
                epsilon=EPSILON, gamma=GAMMA,
                schedule=schedule,
                seed=s, tracking=True,
            )
        else:
            raise ValueError(f"Unknown algorithm: {algo}")

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
                        # Unified metric — both bots emit it.
                        if "prefix_survival_fraction" in summary:
                            all_tracking["prefix_survival_fraction"].append(
                                summary["prefix_survival_fraction"])
                        if algo.startswith("lotr"):
                            all_tracking["on_path_fraction"].append(summary["on_path_fraction"])
                            all_tracking["on_path_given_chance_stayed"].append(
                                summary["on_path_given_chance_stayed"])
                            all_tracking["chance_diverged_fraction"].append(
                                summary["chance_diverged_fraction"])
                            all_tracking["mean_diverge_depth"].append(summary["mean_diverge_depth"])
                            all_tracking["prefix_D"].append(summary["prefix_D"])
                        if algo.startswith("oos"):
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
        arr_sq_regret = np.array(all_tracking["mean_sum_sq_regret"])
        arr_p95_sq = np.array(all_tracking["p95_sum_sq_regret"])
        arr_max_sq = np.array(all_tracking["max_sum_sq_regret"])
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
            "mean_E_dR2": float(arr_sq_regret.mean()),
            "p95_E_dR2": float(arr_p95_sq.mean()),
            "max_E_dR2": float(arr_max_sq.max()),
            "std_abs_regret": float(arr_regret.std()),
            "visit_count_min_mean": float(arr_vc_min.mean()),
            "visit_count_median_mean": float(arr_vc_med.mean()),
            "n_steps": len(all_tracking["mean_inv_l_sq"]),
            "n_iters_total": all_tracking["n_iters_total"],
        }

        # Unified apples-to-apples prefix-survival metric.
        if all_tracking["prefix_survival_fraction"]:
            arr_psurv = np.array(all_tracking["prefix_survival_fraction"])
            result["tracking"]["mean_prefix_survival"] = float(arr_psurv.mean())
            result["tracking"]["std_prefix_survival"] = float(arr_psurv.std())

        if algo.startswith("lotr") and all_tracking["on_path_fraction"]:
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
            result["tracking"]["std_diverge_depth"] = (
                float(arr_div[arr_div >= 0].std()) if (arr_div >= 0).any() else 0.0)
            result["tracking"]["mean_prefix_D"] = float(arr_D.mean())
            result["tracking"]["max_prefix_D"] = int(arr_D.max())

        if algo.startswith("oos") and all_tracking["targeted_fraction"]:
            arr_targeted = np.array(all_tracking["targeted_fraction"])
            result["tracking"]["mean_targeted_frac"] = float(arr_targeted.mean())

    return result


# ═════════════════════════════════════════════════════════════════════════
# Orchestration
# ═════════════════════════════════════════════════════════════════════════

def build_jobs(games=None):
    if games is None:
        games = GAMES
    jobs = []
    for game_name in games:
        budget_map = MATCHES_BY_GAME_BUDGET[game_name]
        for algo in ALGORITHMS:
            for sims in BUDGETS:
                num_matches = budget_map.get(sims)
                if num_matches is None:
                    continue
                jobs.append({
                    "game": game_name,
                    "algorithm": algo,
                    "sims_per_move": sims,
                    "num_matches": num_matches,
                    "seed": SEED,
                })
    return jobs


def main():
    parser = argparse.ArgumentParser(description="LOTR(step) vs OOS — q-form isolation")
    parser.add_argument("--workers", type=int, default=5)
    parser.add_argument("--games", type=str, default=None)
    parser.add_argument("--output", type=str, default=OUTPUT_DIR)
    parser.add_argument("--pilot", action="store_true",
                        help="Tiny pilot: leduc only, 5k+20k, 10 matches each")
    args = parser.parse_args()

    games = args.games.split(",") if args.games else None
    if args.pilot:
        games = ["leduc_poker"]
        global BUDGETS, MATCHES_BY_GAME_BUDGET
        BUDGETS = [5_000, 20_000]
        MATCHES_BY_GAME_BUDGET = {
            "leduc_poker": {5_000: 10, 20_000: 10},
        }
    jobs = build_jobs(games)
    output_dir = args.output
    os.makedirs(output_dir, exist_ok=True)

    total = len(jobs)
    print(f"LOTR(step) vs OOS — q-form isolation — {total} jobs")
    print(f"  Games: {games or GAMES}")
    print(f"  Algorithms: {ALGORITHMS}")
    print(f"    OOS δ: {OOS_DELTAS}")
    print(f"    LOTR ρ (step depth profile @ d=0): {LOTR_RHOS}")
    print(f"  Budgets: {BUDGETS}")
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
                    f"{result['game']:15s} {result['algorithm']:13s} "
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
                "games": games or GAMES,
                "algorithms": ALGORITHMS,
                "budgets": BUDGETS,
                "matches_by_game_budget": MATCHES_BY_GAME_BUDGET,
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
    for game_name in (games or GAMES):
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
