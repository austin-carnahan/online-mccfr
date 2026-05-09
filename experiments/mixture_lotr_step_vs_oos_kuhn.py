"""Paired-seed Kuhn benchmark: Mixture LOTR step(0.5, 0) vs OOS delta=0.5.

This is the V4 follow-up to the older LOTR step/q-form experiment.  The
LOTR arm intentionally uses ``src.mixture_lotr.MixtureLOTRBot`` rather than
``DepthLOTRBot``.

Default grid:
    game: kuhn_poker
    algorithms: oos_d05, mixture_step_r05
    sims_per_move: 1k, 5k, 20k, 50k, 100k
    seeds: 100 paired seeds
    matches per seed: 100

Usage:
    PYTHONPATH=. python -m experiments.mixture_lotr_step_vs_oos_kuhn --workers 8

Outputs:
    results/mixture_lotr_step_vs_oos_kuhn/incremental_results.jsonl
    results/mixture_lotr_step_vs_oos_kuhn/results.json
    results/mixture_lotr_step_vs_oos_kuhn/summary.csv
    results/mixture_lotr_step_vs_oos_kuhn/summary.md
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import time
import traceback
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np

from open_spiel.python import policy as policy_lib
from open_spiel.python.bots.policy import PolicyBot

from eval.aggregate_exploitability import AggregatePolicy, _merge_bot_strategy
from src.games import load_game
from src.metrics import exploitability
from src.mixture_lotr import MixtureLOTRBot, step as mixture_step
from src.oos import OOSBot


GAME = "kuhn_poker"
ALGORITHMS = ["oos_d05", "mixture_step_r05"]
BUDGETS = [1_000, 5_000, 20_000, 50_000, 100_000]
NUM_SEEDS = 100
MATCHES_PER_SEED = 100

EPSILON = 0.4
GAMMA = 0.01
OOS_DELTA = 0.5
MIXTURE_RHO = 0.5
OUTPUT_DIR = "results/mixture_lotr_step_vs_oos_kuhn"
BOOTSTRAP_REPS = 10_000


def _count_total_info_sets(game):
    info_keys = set()

    def traverse(state):
        if state.is_terminal():
            return
        if state.is_chance_node():
            for action, _probability in state.chance_outcomes():
                traverse(state.child(action))
        else:
            player_id = state.current_player()
            info_keys.add(state.information_state_string(player_id))
            for action in state.legal_actions():
                traverse(state.child(action))

    traverse(game.new_initial_state())
    return len(info_keys)


def _make_bot(algorithm, game, player_id, sims_per_move, seed):
    if algorithm == "oos_d05":
        return OOSBot(
            game,
            player_id,
            num_simulations=sims_per_move,
            delta=OOS_DELTA,
            epsilon=EPSILON,
            gamma=GAMMA,
            seed=seed,
            tracking=False,
        )
    if algorithm == "mixture_step_r05":
        return MixtureLOTRBot(
            game,
            player_id,
            num_simulations=sims_per_move,
            epsilon=EPSILON,
            gamma=GAMMA,
            schedule=mixture_step(MIXTURE_RHO, depth=0),
            seed=seed,
            tracking=False,
        )
    raise ValueError(f"Unknown algorithm: {algorithm}")


def _run_job(job):
    game = load_game(GAME)
    algorithm = job["algorithm"]
    sims_per_move = job["sims_per_move"]
    seed = job["seed"]
    num_matches = job["num_matches"]

    base_seed = seed * 1_000_003 + 17
    rng = np.random.RandomState(base_seed)
    random_policy = policy_lib.UniformRandomPolicy(game)
    global_strategy = [{}, {}]

    for match_index in range(num_matches):
        search_player = match_index % game.num_players()
        random_player = 1 - search_player
        bot_seed = base_seed + match_index

        search_bot = _make_bot(
            algorithm, game, search_player, sims_per_move, bot_seed)
        random_bot = PolicyBot(random_player, rng, random_policy)
        bots = [None, None]
        bots[search_player] = search_bot
        bots[random_player] = random_bot

        state = game.new_initial_state()
        for bot in bots:
            bot.restart()

        while not state.is_terminal():
            if state.is_chance_node():
                outcomes, probabilities = zip(*state.chance_outcomes())
                action = rng.choice(outcomes, p=probabilities)
                state.apply_action(action)
            else:
                player_id = state.current_player()
                action = bots[player_id].step(state)
                for observer_id, bot in enumerate(bots):
                    if observer_id != player_id:
                        bot.inform_action(state, player_id, action)
                state.apply_action(action)

        _merge_bot_strategy(search_bot, search_player, global_strategy)

    aggregate_policy = AggregatePolicy(game, global_strategy)
    expl = exploitability(game, aggregate_policy)
    total_info_sets = _count_total_info_sets(game)
    covered_info_sets = len(
        set(global_strategy[0].keys()) | set(global_strategy[1].keys()))

    return {
        "game": GAME,
        "algorithm": algorithm,
        "sims_per_move": sims_per_move,
        "seed": seed,
        "num_matches": num_matches,
        "exploitability": float(expl),
        "coverage": covered_info_sets,
        "total_info_sets": total_info_sets,
        "coverage_frac": (
            covered_info_sets / total_info_sets if total_info_sets else 0.0),
    }


def _build_jobs(budgets, seeds, num_matches):
    jobs = []
    for sims_per_move in budgets:
        for seed in seeds:
            for algorithm in ALGORITHMS:
                jobs.append({
                    "game": GAME,
                    "algorithm": algorithm,
                    "sims_per_move": sims_per_move,
                    "seed": seed,
                    "num_matches": num_matches,
                })
    return jobs


def _job_key(record):
    return (
        int(record["sims_per_move"]),
        int(record["seed"]),
        record["algorithm"],
    )


def _load_incremental(path):
    completed = {}
    if not os.path.exists(path):
        return completed
    with open(path) as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            completed[_job_key(record)] = record
    return completed


def _mean_ci(values, reps=BOOTSTRAP_REPS):
    array = np.asarray(values, dtype=np.float64)
    if len(array) == 0:
        return (math.nan, math.nan)
    if len(array) == 1:
        return (float(array[0]), float(array[0]))
    rng = np.random.RandomState(20260508)
    sample_indices = rng.randint(0, len(array), size=(reps, len(array)))
    means = array[sample_indices].mean(axis=1)
    return (
        float(np.percentile(means, 2.5)),
        float(np.percentile(means, 97.5)),
    )


def _summarize_algorithm_rows(results):
    grouped = defaultdict(list)
    for result in results:
        key = (result["sims_per_move"], result["algorithm"])
        grouped[key].append(float(result["exploitability"]))

    rows = []
    for (sims_per_move, algorithm), values in sorted(grouped.items()):
        array = np.asarray(values, dtype=np.float64)
        ci_low, ci_high = _mean_ci(array)
        rows.append({
            "kind": "algorithm",
            "sims_per_move": sims_per_move,
            "algorithm": algorithm,
            "n": len(array),
            "mean_exploitability": float(array.mean()),
            "median_exploitability": float(np.median(array)),
            "std_exploitability": float(array.std(ddof=1)) if len(array) > 1 else 0.0,
            "p10_exploitability": float(np.percentile(array, 10)),
            "p90_exploitability": float(np.percentile(array, 90)),
            "mean_ci95_low": ci_low,
            "mean_ci95_high": ci_high,
        })
    return rows


def _summarize_paired_rows(results):
    by_key = {}
    for result in results:
        key = (result["sims_per_move"], result["seed"])
        by_key.setdefault(key, {})[result["algorithm"]] = result

    grouped_diffs = defaultdict(list)
    for (sims_per_move, seed), records in by_key.items():
        if "oos_d05" not in records or "mixture_step_r05" not in records:
            continue
        diff = (
            float(records["mixture_step_r05"]["exploitability"])
            - float(records["oos_d05"]["exploitability"])
        )
        grouped_diffs[sims_per_move].append((seed, diff))

    rows = []
    for sims_per_move, seed_diffs in sorted(grouped_diffs.items()):
        diffs = np.asarray([diff for _seed, diff in seed_diffs], dtype=np.float64)
        ci_low, ci_high = _mean_ci(diffs)
        mixture_better = int((diffs < 0.0).sum())
        oos_better = int((diffs > 0.0).sum())
        ties = int((diffs == 0.0).sum())
        rows.append({
            "kind": "paired_diff",
            "sims_per_move": sims_per_move,
            "algorithm": "mixture_step_r05_minus_oos_d05",
            "n": len(diffs),
            "mean_diff": float(diffs.mean()),
            "median_diff": float(np.median(diffs)),
            "std_diff": float(diffs.std(ddof=1)) if len(diffs) > 1 else 0.0,
            "p10_diff": float(np.percentile(diffs, 10)),
            "p90_diff": float(np.percentile(diffs, 90)),
            "mean_ci95_low": ci_low,
            "mean_ci95_high": ci_high,
            "mixture_better_count": mixture_better,
            "oos_better_count": oos_better,
            "tie_count": ties,
        })
    return rows


def _write_csv(path, algorithm_rows, paired_rows):
    fieldnames = [
        "kind",
        "sims_per_move",
        "algorithm",
        "n",
        "mean_exploitability",
        "median_exploitability",
        "std_exploitability",
        "p10_exploitability",
        "p90_exploitability",
        "mean_diff",
        "median_diff",
        "std_diff",
        "p10_diff",
        "p90_diff",
        "mean_ci95_low",
        "mean_ci95_high",
        "mixture_better_count",
        "oos_better_count",
        "tie_count",
    ]
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in algorithm_rows + paired_rows:
            writer.writerow(row)


def _format_float(value):
    if value is None or not math.isfinite(float(value)):
        return "nan"
    return f"{float(value):.6g}"


def _write_markdown(path, algorithm_rows, paired_rows, config):
    with open(path, "w") as handle:
        handle.write("# Mixture LOTR step vs OOS on Kuhn\n\n")
        handle.write("## Config\n\n")
        for key, value in config.items():
            handle.write(f"- {key}: {value}\n")

        handle.write("\n## Exploitability By Algorithm\n\n")
        handle.write(
            "| sims_per_move | algorithm | n | mean | median | p10 | p90 | mean 95% CI |\n")
        handle.write("|---:|---|---:|---:|---:|---:|---:|---:|\n")
        for row in algorithm_rows:
            ci = f"[{_format_float(row['mean_ci95_low'])}, {_format_float(row['mean_ci95_high'])}]"
            handle.write(
                f"| {row['sims_per_move']} | {row['algorithm']} | {row['n']} | "
                f"{_format_float(row['mean_exploitability'])} | "
                f"{_format_float(row['median_exploitability'])} | "
                f"{_format_float(row['p10_exploitability'])} | "
                f"{_format_float(row['p90_exploitability'])} | {ci} |\n")

        handle.write("\n## Paired Difference\n\n")
        handle.write("Diff = mixture_step_r05 exploitability - oos_d05 exploitability. Negative means Mixture did better for that seed.\n\n")
        handle.write(
            "| sims_per_move | n | mean diff | median diff | p10 | p90 | mean 95% CI | mixture better | oos better | ties |\n")
        handle.write("|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n")
        for row in paired_rows:
            ci = f"[{_format_float(row['mean_ci95_low'])}, {_format_float(row['mean_ci95_high'])}]"
            handle.write(
                f"| {row['sims_per_move']} | {row['n']} | "
                f"{_format_float(row['mean_diff'])} | "
                f"{_format_float(row['median_diff'])} | "
                f"{_format_float(row['p10_diff'])} | "
                f"{_format_float(row['p90_diff'])} | {ci} | "
                f"{row['mixture_better_count']} | "
                f"{row['oos_better_count']} | {row['tie_count']} |\n")


def _write_reports(output_dir, results, config):
    results = sorted(results, key=lambda item: (
        item["sims_per_move"], item["seed"], item["algorithm"]))
    algorithm_rows = _summarize_algorithm_rows(results)
    paired_rows = _summarize_paired_rows(results)

    results_path = os.path.join(output_dir, "results.json")
    with open(results_path, "w") as handle:
        json.dump({"config": config, "results": results}, handle, indent=2)

    csv_path = os.path.join(output_dir, "summary.csv")
    markdown_path = os.path.join(output_dir, "summary.md")
    _write_csv(csv_path, algorithm_rows, paired_rows)
    _write_markdown(markdown_path, algorithm_rows, paired_rows, config)
    return results_path, csv_path, markdown_path, algorithm_rows, paired_rows


def _print_summary(algorithm_rows, paired_rows):
    print("\nExploitability by algorithm")
    for row in algorithm_rows:
        print(
            f"  sims={row['sims_per_move']:>6d} {row['algorithm']:18s} "
            f"n={row['n']:>3d} mean={row['mean_exploitability']:.6g} "
            f"median={row['median_exploitability']:.6g} "
            f"ci=[{row['mean_ci95_low']:.6g}, {row['mean_ci95_high']:.6g}]"
        )

    print("\nPaired diff: mixture - oos")
    for row in paired_rows:
        print(
            f"  sims={row['sims_per_move']:>6d} n={row['n']:>3d} "
            f"mean={row['mean_diff']:.6g} median={row['median_diff']:.6g} "
            f"ci=[{row['mean_ci95_low']:.6g}, {row['mean_ci95_high']:.6g}] "
            f"mix_better={row['mixture_better_count']} "
            f"oos_better={row['oos_better_count']} ties={row['tie_count']}"
        )


def _parse_budgets(value):
    return [int(part.replace("_", "")) for part in value.split(",") if part.strip()]


def main():
    parser = argparse.ArgumentParser(
        description="Paired-seed Kuhn sweep: Mixture LOTR step(0.5,0) vs OOS delta=0.5")
    parser.add_argument("--workers", type=int, default=5)
    parser.add_argument("--output", type=str, default=OUTPUT_DIR)
    parser.add_argument("--budgets", type=str, default=None,
                        help="Comma-separated sims-per-move budgets, e.g. 20000 or 20000,100000.")
    parser.add_argument("--num-seeds", type=int, default=NUM_SEEDS)
    parser.add_argument("--seed-offset", type=int, default=0)
    parser.add_argument("--num-matches", type=int, default=MATCHES_PER_SEED)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--pilot", action="store_true",
                        help="Small smoke grid: 2 seeds, 10 matches, 1k budget.")
    args = parser.parse_args()

    budgets = _parse_budgets(args.budgets) if args.budgets else list(BUDGETS)
    num_seeds = args.num_seeds
    num_matches = args.num_matches
    if args.pilot:
        budgets = [1_000]
        num_seeds = 2
        num_matches = 10

    seeds = list(range(args.seed_offset, args.seed_offset + num_seeds))
    jobs = _build_jobs(budgets, seeds, num_matches)

    output_dir = args.output
    os.makedirs(output_dir, exist_ok=True)
    incremental_path = os.path.join(output_dir, "incremental_results.jsonl")

    completed = _load_incremental(incremental_path) if args.resume else {}
    pending_jobs = [job for job in jobs if _job_key(job) not in completed]

    config = {
        "game": GAME,
        "algorithms": ALGORITHMS,
        "budgets": budgets,
        "num_seeds": num_seeds,
        "seed_offset": args.seed_offset,
        "seeds": seeds,
        "num_matches": num_matches,
        "epsilon": EPSILON,
        "gamma": GAMMA,
        "oos_delta": OOS_DELTA,
        "mixture_schedule": f"step({MIXTURE_RHO}, depth=0)",
        "bootstrap_reps": BOOTSTRAP_REPS,
    }

    print("Mixture LOTR step vs OOS on Kuhn")
    print(f"  Algorithms: {ALGORITHMS}")
    print(f"  Budgets: {budgets}")
    print(f"  Seeds: {seeds[0]}..{seeds[-1]} ({len(seeds)} total)")
    print(f"  Matches per seed: {num_matches}")
    print(f"  Workers: {args.workers}")
    print(f"  Pending jobs: {len(pending_jobs)} / {len(jobs)}")
    print(f"  Output: {output_dir}\n")

    results_by_key = dict(completed)
    errors = []
    completed_count = 0
    start_time = time.time()

    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        future_to_job = {pool.submit(_run_job, job): job for job in pending_jobs}
        for future in as_completed(future_to_job):
            completed_count += 1
            job = future_to_job[future]
            elapsed = time.time() - start_time
            eta_seconds = ((elapsed / completed_count) *
                           (len(pending_jobs) - completed_count)
                           if completed_count else 0.0)
            try:
                result = future.result()
                results_by_key[_job_key(result)] = result
                with open(incremental_path, "a") as handle:
                    handle.write(json.dumps(result) + "\n")
                print(
                    f"  [{completed_count:>4d}/{len(pending_jobs):>4d}] "
                    f"seed={result['seed']:>3d} "
                    f"sims={result['sims_per_move']:>6d} "
                    f"{result['algorithm']:18s} "
                    f"expl={result['exploitability']:.6g} "
                    f"cov={result['coverage']}/{result['total_info_sets']} "
                    f"elapsed={elapsed/60:.1f}m eta={eta_seconds/60:.1f}m"
                )
            except Exception as exc:
                errors.append({
                    "job": job,
                    "error": str(exc),
                    "traceback": traceback.format_exc(),
                })
                print(
                    f"  [{completed_count:>4d}/{len(pending_jobs):>4d}] "
                    f"FAILED seed={job['seed']} sims={job['sims_per_move']} "
                    f"{job['algorithm']}: {exc}"
                )

    results = list(results_by_key.values())
    paths = _write_reports(output_dir, results, config)
    _results_path, csv_path, markdown_path, algorithm_rows, paired_rows = paths

    if errors:
        errors_path = os.path.join(output_dir, "errors.json")
        with open(errors_path, "w") as handle:
            json.dump(errors, handle, indent=2)
        print(f"\n{len(errors)} jobs failed. See {errors_path}")

    _print_summary(algorithm_rows, paired_rows)
    print(f"\nReports written to {csv_path} and {markdown_path}")


if __name__ == "__main__":
    main()