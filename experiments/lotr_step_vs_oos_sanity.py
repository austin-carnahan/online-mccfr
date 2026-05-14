"""Paired-seed sanity sweep: canonical LOTR step(0.5, 0) vs OOS delta=0.5.

This is the non-Kuhn follow-up to experiments.lotr_step_vs_oos_kuhn. It
keeps the parity schedule fixed:

    OOSBot(delta=0.5)  vs  LOTRBot(schedule=step(0.5, depth=0))

Default grid:
    games: leduc_poker, goofspiel, liars_dice
    algorithms: oos_d05, lotr_step_r05
    sims_per_move: 5k, 10k, 20k, 50k
    seeds: 3 paired seeds
    matches: leduc/goofspiel 100 per seed, liars_dice 50 per seed

Usage:
    python -m experiments.lotr_step_vs_oos_sanity --workers 6
    python -m experiments.lotr_step_vs_oos_sanity --pilot --workers 1

Outputs:
    results/lotr_step_vs_oos_sanity/incremental_results.jsonl
    results/lotr_step_vs_oos_sanity/results.json
    results/lotr_step_vs_oos_sanity/summary.csv
    results/lotr_step_vs_oos_sanity/summary.md
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
from src.games import GAME_SPECS, load_game
from src.lotr import LOTRBot, step as lotr_step
from src.metrics import exploitability
from src.oos import OOSBot


DEFAULT_GAMES = ["leduc_poker", "goofspiel", "liars_dice"]
ALGORITHMS = ["oos_d05", "lotr_step_r05"]
BUDGETS = [5_000, 10_000, 20_000, 50_000]
NUM_SEEDS = 3
SEED_OFFSET = 0

DEFAULT_MATCHES_BY_GAME = {
    "leduc_poker": 100,
    "goofspiel": 100,
    "liars_dice": 50,
}

EPSILON = 0.4
GAMMA = 0.01
OOS_DELTA = 0.5
LOTR_RHO = 0.5
OUTPUT_DIR = "results/lotr_step_vs_oos_sanity"
BOOTSTRAP_REPS = 10_000

_TOTAL_INFO_CACHE = {}

COMMON_TRACKING_KEYS = [
    "log_W_mean",
    "log_W_p95",
    "log_W_p99",
    "log_W_max",
    "ess_per_sim",
    "log_l_mean",
    "log_l_p5",
    "mean_inv_l_sq",
    "mean_l",
    "min_l",
    "max_inv_l",
    "mean_sum_abs_regret",
    "mean_sum_sq_regret",
    "p95_sum_sq_regret",
    "max_sum_sq_regret",
    "prefix_survival_fraction",
    "visit_count_min",
    "visit_count_median",
    "n_iters",
]

OOS_TRACKING_KEYS = ["targeted_fraction"]
LOTR_TRACKING_KEYS = [
    "on_path_fraction",
    "on_path_given_chance_stayed",
    "chance_diverged_fraction",
    "mean_diverge_depth",
    "prefix_D",
]

TRACKING_ROW_FIELDS = [
    "mean_prefix_survival",
    "mean_targeted_frac",
    "mean_on_path_frac",
    "mean_on_path_given_chance",
    "mean_chance_diverged_frac",
    "mean_prefix_D",
    "max_prefix_D",
    "mean_ess_per_sim",
    "min_ess_per_sim",
    "mean_E_W2",
    "p95_E_W2",
    "mean_log_l",
    "mean_log_l_p5",
    "mean_abs_regret",
    "mean_E_dR2",
    "p95_E_dR2",
    "max_E_dR2",
    "mean_visit_count_min",
    "mean_visit_count_median",
    "tracking_steps",
    "tracking_iters_total",
]


def _parse_int_list(value):
    return [int(part.strip().replace("_", "")) for part in value.split(",") if part.strip()]


def _parse_games(value):
    if not value:
        return list(DEFAULT_GAMES)
    parts = [part.strip() for part in value.split(",") if part.strip()]
    if parts == ["all"]:
        return list(GAME_SPECS.keys())
    unknown = [name for name in parts if name not in GAME_SPECS]
    if unknown:
        raise ValueError(f"Unknown games {unknown}. Available: {list(GAME_SPECS)}")
    return parts


def _parse_matches_by_game(value, games, default_map, num_matches_override=None):
    if num_matches_override is not None:
        return {game_name: int(num_matches_override) for game_name in games}

    matches = {game_name: int(default_map.get(game_name, 100)) for game_name in games}
    if not value:
        return matches

    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        if ":" not in item:
            raise ValueError(
                "--matches-by-game entries must look like leduc_poker:100")
        game_name, count = item.split(":", 1)
        game_name = game_name.strip()
        if game_name not in games:
            raise ValueError(f"Match count provided for unselected game: {game_name}")
        matches[game_name] = int(count.strip().replace("_", ""))
    return matches


def _count_total_info_sets(game_name, game):
    if game_name in _TOTAL_INFO_CACHE:
        return _TOTAL_INFO_CACHE[game_name]

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
    total = len(info_keys)
    _TOTAL_INFO_CACHE[game_name] = total
    return total


def _make_bot(algorithm, game, player_id, sims_per_move, seed, tracking):
    if algorithm == "oos_d05":
        return OOSBot(
            game,
            player_id,
            num_simulations=sims_per_move,
            delta=OOS_DELTA,
            epsilon=EPSILON,
            gamma=GAMMA,
            seed=seed,
            tracking=tracking,
        )
    if algorithm == "lotr_step_r05":
        return LOTRBot(
            game,
            player_id,
            num_simulations=sims_per_move,
            epsilon=EPSILON,
            gamma=GAMMA,
            schedule=lotr_step(LOTR_RHO, depth=0),
            seed=seed,
            tracking=tracking,
        )
    raise ValueError(f"Unknown algorithm: {algorithm}")


def _new_tracking_accumulator(algorithm):
    keys = list(COMMON_TRACKING_KEYS)
    if algorithm.startswith("oos"):
        keys.extend(OOS_TRACKING_KEYS)
    if algorithm.startswith("lotr"):
        keys.extend(LOTR_TRACKING_KEYS)
    return {key: [] for key in keys}


def _record_tracking(accumulator, summary):
    if not summary:
        return
    for key in accumulator:
        if key in summary:
            accumulator[key].append(summary[key])


def _mean(values):
    if not values:
        return math.nan
    return float(np.asarray(values, dtype=np.float64).mean())


def _percentile(values, q):
    if not values:
        return math.nan
    return float(np.percentile(np.asarray(values, dtype=np.float64), q))


def _finalize_tracking(accumulator):
    if not accumulator.get("mean_inv_l_sq"):
        return {}

    result = {
        "mean_prefix_survival": _mean(accumulator.get("prefix_survival_fraction", [])),
        "mean_targeted_frac": _mean(accumulator.get("targeted_fraction", [])),
        "mean_on_path_frac": _mean(accumulator.get("on_path_fraction", [])),
        "mean_on_path_given_chance": _mean(accumulator.get("on_path_given_chance_stayed", [])),
        "mean_chance_diverged_frac": _mean(accumulator.get("chance_diverged_fraction", [])),
        "mean_prefix_D": _mean(accumulator.get("prefix_D", [])),
        "max_prefix_D": (
            int(max(accumulator["prefix_D"]))
            if accumulator.get("prefix_D") else None
        ),
        "mean_ess_per_sim": _mean(accumulator.get("ess_per_sim", [])),
        "min_ess_per_sim": (
            float(min(accumulator["ess_per_sim"]))
            if accumulator.get("ess_per_sim") else math.nan
        ),
        "mean_E_W2": _mean(accumulator.get("mean_inv_l_sq", [])),
        "p95_E_W2": _percentile(accumulator.get("mean_inv_l_sq", []), 95),
        "mean_log_l": _mean(accumulator.get("log_l_mean", [])),
        "mean_log_l_p5": _mean(accumulator.get("log_l_p5", [])),
        "mean_abs_regret": _mean(accumulator.get("mean_sum_abs_regret", [])),
        "mean_E_dR2": _mean(accumulator.get("mean_sum_sq_regret", [])),
        "p95_E_dR2": _mean(accumulator.get("p95_sum_sq_regret", [])),
        "max_E_dR2": (
            float(max(accumulator["max_sum_sq_regret"]))
            if accumulator.get("max_sum_sq_regret") else math.nan
        ),
        "mean_visit_count_min": _mean(accumulator.get("visit_count_min", [])),
        "mean_visit_count_median": _mean(accumulator.get("visit_count_median", [])),
        "tracking_steps": len(accumulator.get("mean_inv_l_sq", [])),
        "tracking_iters_total": int(sum(accumulator.get("n_iters", []))),
    }
    cleaned = {}
    for key, value in result.items():
        if value is None:
            continue
        if isinstance(value, (float, int)) and not math.isfinite(float(value)):
            continue
        cleaned[key] = value
    return cleaned


def _run_job(job):
    game_name = job["game"]
    game = load_game(game_name)
    if game.num_players() != 2:
        raise ValueError(f"{game_name} has {game.num_players()} players; expected 2")

    algorithm = job["algorithm"]
    sims_per_move = job["sims_per_move"]
    seed = job["seed"]
    num_matches = job["num_matches"]
    tracking = bool(job.get("tracking", True))

    base_seed = seed * 1_000_003 + 17
    rng = np.random.RandomState(base_seed)
    random_policy = policy_lib.UniformRandomPolicy(game)
    global_strategy = [{}, {}]
    tracking_accumulator = _new_tracking_accumulator(algorithm)

    start_time = time.time()
    for match_index in range(num_matches):
        search_player = match_index % game.num_players()
        random_player = 1 - search_player
        bot_seed = base_seed + match_index

        search_bot = _make_bot(
            algorithm, game, search_player, sims_per_move, bot_seed, tracking)
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

                if tracking and player_id == search_player:
                    _record_tracking(
                        tracking_accumulator, search_bot.get_tracking_summary())

                state.apply_action(action)

        _merge_bot_strategy(search_bot, search_player, global_strategy)

    aggregate_policy = AggregatePolicy(game, global_strategy)
    expl = exploitability(game, aggregate_policy)
    total_info_sets = _count_total_info_sets(game_name, game)
    covered_info_sets = len(
        set(global_strategy[0].keys()) | set(global_strategy[1].keys()))

    result = {
        "game": game_name,
        "algorithm": algorithm,
        "sims_per_move": sims_per_move,
        "seed": seed,
        "num_matches": num_matches,
        "exploitability": float(expl),
        "coverage": covered_info_sets,
        "total_info_sets": total_info_sets,
        "coverage_frac": (
            covered_info_sets / total_info_sets if total_info_sets else 0.0),
        "elapsed_s": round(time.time() - start_time, 3),
    }
    tracking_summary = _finalize_tracking(tracking_accumulator)
    if tracking_summary:
        result["tracking"] = tracking_summary
    return result


def _build_jobs(games, budgets, seeds, matches_by_game, tracking):
    jobs = []
    for game_name in games:
        for sims_per_move in budgets:
            for seed in seeds:
                for algorithm in ALGORITHMS:
                    jobs.append({
                        "game": game_name,
                        "algorithm": algorithm,
                        "sims_per_move": sims_per_move,
                        "seed": seed,
                        "num_matches": int(matches_by_game[game_name]),
                        "tracking": bool(tracking),
                    })
    return jobs


def _job_key(record):
    return (
        record["game"],
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
    rng = np.random.RandomState(20260513)
    sample_indices = rng.randint(0, len(array), size=(reps, len(array)))
    means = array[sample_indices].mean(axis=1)
    return (
        float(np.percentile(means, 2.5)),
        float(np.percentile(means, 97.5)),
    )


def _summarize_algorithm_rows(results):
    grouped = defaultdict(list)
    for result in results:
        key = (result["game"], result["sims_per_move"], result["algorithm"])
        grouped[key].append(result)

    rows = []
    for (game_name, sims_per_move, algorithm), group in sorted(grouped.items()):
        values = np.asarray([float(row["exploitability"]) for row in group], dtype=np.float64)
        ci_low, ci_high = _mean_ci(values)
        row = {
            "kind": "algorithm",
            "game": game_name,
            "sims_per_move": sims_per_move,
            "algorithm": algorithm,
            "n": len(values),
            "num_matches": int(group[0]["num_matches"]),
            "mean_exploitability": float(values.mean()),
            "median_exploitability": float(np.median(values)),
            "std_exploitability": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
            "p10_exploitability": float(np.percentile(values, 10)),
            "p90_exploitability": float(np.percentile(values, 90)),
            "mean_ci95_low": ci_low,
            "mean_ci95_high": ci_high,
            "mean_coverage": float(np.mean([row["coverage"] for row in group])),
            "mean_coverage_frac": float(np.mean([row["coverage_frac"] for row in group])),
            "mean_elapsed_s": float(np.mean([row.get("elapsed_s", 0.0) for row in group])),
        }
        tracking_rows = [row.get("tracking", {}) for row in group]
        for field in TRACKING_ROW_FIELDS:
            vals = [track[field] for track in tracking_rows
                    if field in track and track[field] is not None
                    and math.isfinite(float(track[field]))]
            if vals:
                row[field] = float(np.mean(vals))
        rows.append(row)
    return rows


def _summarize_paired_rows(results):
    by_key = {}
    for result in results:
        key = (result["game"], result["sims_per_move"], result["seed"])
        by_key.setdefault(key, {})[result["algorithm"]] = result

    grouped_diffs = defaultdict(list)
    for (game_name, sims_per_move, seed), records in by_key.items():
        if "oos_d05" not in records or "lotr_step_r05" not in records:
            continue
        diff = (
            float(records["lotr_step_r05"]["exploitability"])
            - float(records["oos_d05"]["exploitability"])
        )
        grouped_diffs[(game_name, sims_per_move)].append((seed, diff))

    rows = []
    for (game_name, sims_per_move), seed_diffs in sorted(grouped_diffs.items()):
        diffs = np.asarray([diff for _seed, diff in seed_diffs], dtype=np.float64)
        ci_low, ci_high = _mean_ci(diffs)
        rows.append({
            "kind": "paired_diff",
            "game": game_name,
            "sims_per_move": sims_per_move,
            "algorithm": "lotr_step_r05_minus_oos_d05",
            "n": len(diffs),
            "mean_diff": float(diffs.mean()),
            "median_diff": float(np.median(diffs)),
            "std_diff": float(diffs.std(ddof=1)) if len(diffs) > 1 else 0.0,
            "p10_diff": float(np.percentile(diffs, 10)),
            "p90_diff": float(np.percentile(diffs, 90)),
            "mean_ci95_low": ci_low,
            "mean_ci95_high": ci_high,
            "lotr_better_count": int((diffs < 0.0).sum()),
            "oos_better_count": int((diffs > 0.0).sum()),
            "tie_count": int((diffs == 0.0).sum()),
        })
    return rows


def _format_float(value):
    if value is None:
        return ""
    try:
        value = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not math.isfinite(value):
        return ""
    return f"{value:.6g}"


def _write_csv(path, algorithm_rows, paired_rows):
    rows = algorithm_rows + paired_rows
    base_fields = [
        "kind", "game", "sims_per_move", "algorithm", "n", "num_matches",
        "mean_exploitability", "median_exploitability", "std_exploitability",
        "p10_exploitability", "p90_exploitability", "mean_diff",
        "median_diff", "std_diff", "p10_diff", "p90_diff",
        "mean_ci95_low", "mean_ci95_high", "mean_coverage",
        "mean_coverage_frac", "mean_elapsed_s", "lotr_better_count",
        "oos_better_count", "tie_count",
    ]
    extra_fields = sorted(set().union(*(row.keys() for row in rows)) - set(base_fields))
    fieldnames = base_fields + extra_fields
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _write_markdown(path, algorithm_rows, paired_rows, config):
    with open(path, "w") as handle:
        handle.write("# LOTR step vs OOS sanity sweep\n\n")
        handle.write("## Config\n\n")
        for key, value in config.items():
            handle.write(f"- {key}: {value}\n")

        handle.write("\n## Exploitability By Algorithm\n\n")
        handle.write(
            "| game | sims | algorithm | seeds | matches | mean | median | mean 95% CI | coverage | prefix_survival | ess/sim | E[W^2] | prefix_D |\n")
        handle.write("|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n")
        for row in algorithm_rows:
            ci = f"[{_format_float(row.get('mean_ci95_low'))}, {_format_float(row.get('mean_ci95_high'))}]"
            handle.write(
                f"| {row['game']} | {row['sims_per_move']} | {row['algorithm']} | "
                f"{row['n']} | {row['num_matches']} | "
                f"{_format_float(row.get('mean_exploitability'))} | "
                f"{_format_float(row.get('median_exploitability'))} | {ci} | "
                f"{_format_float(row.get('mean_coverage_frac'))} | "
                f"{_format_float(row.get('mean_prefix_survival'))} | "
                f"{_format_float(row.get('mean_ess_per_sim'))} | "
                f"{_format_float(row.get('mean_E_W2'))} | "
                f"{_format_float(row.get('mean_prefix_D'))} |\n"
            )

        handle.write("\n## Paired Difference\n\n")
        handle.write("Diff = lotr_step_r05 exploitability - oos_d05 exploitability. Negative means LOTR did better for that seed.\n\n")
        handle.write(
            "| game | sims | seeds | mean diff | median diff | mean 95% CI | LOTR better | OOS better | ties |\n")
        handle.write("|---|---:|---:|---:|---:|---:|---:|---:|---:|\n")
        for row in paired_rows:
            ci = f"[{_format_float(row.get('mean_ci95_low'))}, {_format_float(row.get('mean_ci95_high'))}]"
            handle.write(
                f"| {row['game']} | {row['sims_per_move']} | {row['n']} | "
                f"{_format_float(row.get('mean_diff'))} | "
                f"{_format_float(row.get('median_diff'))} | {ci} | "
                f"{row['lotr_better_count']} | {row['oos_better_count']} | "
                f"{row['tie_count']} |\n"
            )


def _write_reports(output_dir, results, config):
    results = sorted(results, key=lambda item: (
        item["game"], item["sims_per_move"], item["seed"], item["algorithm"]))
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
            f"  {row['game']:12s} sims={row['sims_per_move']:>6d} "
            f"{row['algorithm']:18s} n={row['n']:>2d} "
            f"mean={row['mean_exploitability']:.6g} "
            f"ci=[{row['mean_ci95_low']:.6g}, {row['mean_ci95_high']:.6g}] "
            f"cov={row['mean_coverage_frac']:.3f}"
        )

    print("\nPaired diff: LOTR - OOS")
    for row in paired_rows:
        print(
            f"  {row['game']:12s} sims={row['sims_per_move']:>6d} "
            f"n={row['n']:>2d} mean={row['mean_diff']:.6g} "
            f"ci=[{row['mean_ci95_low']:.6g}, {row['mean_ci95_high']:.6g}] "
            f"lotr_better={row['lotr_better_count']} "
            f"oos_better={row['oos_better_count']} ties={row['tie_count']}"
        )


def main():
    parser = argparse.ArgumentParser(
        description="Paired-seed sanity sweep: LOTR step(0.5,0) vs OOS delta=0.5")
    parser.add_argument("--workers", type=int, default=5)
    parser.add_argument("--games", type=str, default=None,
                        help="Comma-separated games. Default: leduc_poker,goofspiel,liars_dice")
    parser.add_argument("--budgets", type=str, default=None,
                        help="Comma-separated sims-per-move budgets, e.g. 5000,10000,20000,50000")
    parser.add_argument("--num-seeds", type=int, default=NUM_SEEDS)
    parser.add_argument("--seed-offset", type=int, default=SEED_OFFSET)
    parser.add_argument("--num-matches", type=int, default=None,
                        help="Override match count for every selected game.")
    parser.add_argument("--matches-by-game", type=str, default=None,
                        help="Comma-separated overrides, e.g. leduc_poker:100,liars_dice:50")
    parser.add_argument("--output", type=str, default=OUTPUT_DIR)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--no-tracking", action="store_true",
                        help="Disable per-step tracking summaries to reduce overhead.")
    parser.add_argument("--pilot", action="store_true",
                        help="Tiny all-game smoke grid: one seed, 100 sims, 2 matches.")
    args = parser.parse_args()

    games = _parse_games(args.games)
    budgets = _parse_int_list(args.budgets) if args.budgets else list(BUDGETS)
    num_seeds = args.num_seeds
    seed_offset = args.seed_offset
    matches_by_game = _parse_matches_by_game(
        args.matches_by_game, games, DEFAULT_MATCHES_BY_GAME, args.num_matches)
    tracking = not args.no_tracking

    if args.pilot:
        budgets = [100]
        num_seeds = 1
        matches_by_game = {game_name: 2 for game_name in games}
        tracking = True

    seeds = list(range(seed_offset, seed_offset + num_seeds))
    jobs = _build_jobs(games, budgets, seeds, matches_by_game, tracking)

    output_dir = args.output
    os.makedirs(output_dir, exist_ok=True)
    incremental_path = os.path.join(output_dir, "incremental_results.jsonl")

    completed = _load_incremental(incremental_path) if args.resume else {}
    pending_jobs = [job for job in jobs if _job_key(job) not in completed]

    config = {
        "games": games,
        "algorithms": ALGORITHMS,
        "budgets": budgets,
        "num_seeds": num_seeds,
        "seed_offset": seed_offset,
        "seeds": seeds,
        "matches_by_game": matches_by_game,
        "epsilon": EPSILON,
        "gamma": GAMMA,
        "oos_delta": OOS_DELTA,
        "lotr_schedule": f"step({LOTR_RHO}, depth=0)",
        "tracking": tracking,
        "bootstrap_reps": BOOTSTRAP_REPS,
    }

    print("LOTR step vs OOS sanity sweep")
    print(f"  Games: {games}")
    print(f"  Algorithms: {ALGORITHMS}")
    print(f"  Budgets: {budgets}")
    print(f"  Seeds: {seeds}")
    print(f"  Matches by game: {matches_by_game}")
    print(f"  Tracking: {tracking}")
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

                tracking_str = ""
                if "tracking" in result:
                    track = result["tracking"]
                    tracking_str = (
                        f" psurv={track.get('mean_prefix_survival', math.nan):.3f}"
                        f" ess/sim={track.get('mean_ess_per_sim', math.nan):.3f}"
                    )
                print(
                    f"  [{completed_count:>4d}/{len(pending_jobs):>4d}] "
                    f"{result['game']:12s} seed={result['seed']:>3d} "
                    f"sims={result['sims_per_move']:>6d} "
                    f"{result['algorithm']:18s} expl={result['exploitability']:.6g} "
                    f"cov={result['coverage']}/{result['total_info_sets']}"
                    f"{tracking_str} elapsed={elapsed/60:.1f}m eta={eta_seconds/60:.1f}m"
                )
            except Exception as exc:
                errors.append({
                    "job": job,
                    "error": str(exc),
                    "traceback": traceback.format_exc(),
                })
                print(
                    f"  [{completed_count:>4d}/{len(pending_jobs):>4d}] "
                    f"FAILED {job['game']} seed={job['seed']} "
                    f"sims={job['sims_per_move']} {job['algorithm']}: {exc}"
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
