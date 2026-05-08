"""Experiment 15C: Low-Budget Proximity Probe.

Does IIG proximity predict per-walk regret productivity when the sim
budget is genuinely scarce (5-50 sims)?  The 15A profiles were flat at
sims>=100.  If proximity ever matters, it should be visible at tiny
budgets where the local tree is unsaturated.

Key additions over 15A:
  - Iteration-bucketed neighborhood regret (early 20% vs late 20% of sims)
  - Realized importance weight statistics by z* level
  - SqrtBalancedWeight decay (intermediate between constant and t_balanced)
  - Much lower sim budgets: 5, 10, 20, 50
  - Leduc only (richest IIG structure, 8 levels)
  - High match count (2000) to compensate for fewer observations per match

Configs (6):
  1. chance + constant       — proximity-biased at terminal level
  2. chance + t_balanced     — uniform over terminals
  3. chance + sqrt_balanced  — intermediate
  4. full + constant         — full mode comparison
  5. full + t_balanced       — full mode comparison
  6. full + sqrt_balanced    — full mode comparison

Fixed: δ=0.5, ε=0.4, γ=0.01 (matched to 15A)
Game: leduc_poker only
Budgets: 5, 10, 20, 50 sims/move
Matches: 2000

Jobs: 6 configs × 4 budgets = 24

Usage:
    python -m experiments.proximity_probe_015c
    python -m experiments.proximity_probe_015c --quick     # 500 matches
    python -m experiments.proximity_probe_015c --budgets 5 10
"""

import sys
import os
if not sys.stdout.line_buffering:
    sys.stdout.reconfigure(line_buffering=True)

import argparse
import json
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed

from src.games import load_game
from src.iig import IIG
from src.isgt import (ISGTBot, ConstantWeight, TerminalBalancedWeight,
                      SqrtBalancedWeight)
from eval.aggregate_metrics import aggregate_with_metrics


# ── Configuration ────────────────────────────────────────────────────────

GAME = "leduc_poker"

# Paper §4.4.2 parameters
DELTA = 0.5
EPSILON = 0.4
GAMMA = 0.01

BUDGETS = [5, 10, 20, 50]
NUM_MATCHES = 2000
QUICK_MATCHES = 500
WORKERS = 6

# ISGT configs: (label, bias_mode, decay_name, decay_fn)
CONFIGS = [
    ("chance_constant",      "chance", "constant",        ConstantWeight()),
    ("chance_tbal",          "chance", "t_balanced",       TerminalBalancedWeight()),
    ("chance_sqrt",          "chance", "sqrt_balanced",    SqrtBalancedWeight()),
    ("full_constant",        "full",   "constant",        ConstantWeight()),
    ("full_tbal",            "full",   "t_balanced",       TerminalBalancedWeight()),
    ("full_sqrt",            "full",   "sqrt_balanced",    SqrtBalancedWeight()),
]


# ── Single job ───────────────────────────────────────────────────────────

def _run_job(label, bias_mode, decay_fn, sims_per_move,
             num_matches, seed):
    """Run one probe configuration."""
    game = load_game(GAME)
    iig = IIG(game)
    counter = [0]

    def bot_factory(player_id):
        s = seed + counter[0]
        counter[0] += 1
        bot = ISGTBot(game, player_id, num_simulations=sims_per_move,
                      epsilon=EPSILON, gamma=GAMMA,
                      level_weight_fn=decay_fn,
                      bias_mode=bias_mode, seed=s, iig=iig,
                      delta=DELTA)
        bot._regret_tracking = True
        return bot

    t0 = time.time()
    metrics = aggregate_with_metrics(
        game, bot_factory, iig,
        num_matches=num_matches, seed=seed,
        level_profile=False, num_sims=sims_per_move)
    elapsed = time.time() - t0

    return {
        "game": GAME,
        "config": label,
        "bias_mode": bias_mode,
        "decay": decay_fn.name(),
        "delta": DELTA,
        "epsilon": EPSILON,
        "sims_per_move": sims_per_move,
        "num_matches": num_matches,
        "exploitability": metrics["exploitability"],
        "infoset_coverage": metrics["infoset_coverage"],
        "regret_by_level": metrics.get("regret_by_level", {}),
        "neighborhood_regret_by_level": metrics.get(
            "neighborhood_regret_by_level", {}),
        "neighborhood_touched_by_level": metrics.get(
            "neighborhood_touched_by_level", {}),
        "importance_weight_by_level": metrics.get(
            "importance_weight_by_level", {}),
        "bucketed_neighborhood_regret": metrics.get(
            "bucketed_neighborhood_regret", {}),
        "elapsed_s": round(elapsed, 1),
    }


# ── Main harness ─────────────────────────────────────────────────────────

def run_probe(budgets=None, num_matches=NUM_MATCHES,
              workers=WORKERS, seed=42, output_dir=None):
    """Run the 15C proximity probe experiment."""
    budgets = budgets or BUDGETS
    if output_dir is None:
        suffix = "quick" if num_matches <= QUICK_MATCHES else "full"
        output_dir = f"results/015c_proximity_probe_{suffix}"
    os.makedirs(output_dir, exist_ok=True)

    jobs = []
    for label, bias_mode, decay_name, decay_fn in CONFIGS:
        for sims in sorted(budgets):
            jobs.append((label, bias_mode, decay_fn, sims,
                         num_matches, seed))

    total = len(jobs)
    print(f"Experiment 15C — Low-Budget Proximity Probe: {total} jobs")
    print(f"  Game: {GAME}")
    print(f"  Configs: {len(CONFIGS)} ({', '.join(c[0] for c in CONFIGS)})")
    print(f"  Budgets: {sorted(budgets)}")
    print(f"  Matches: {num_matches}")
    print(f"  Fixed: δ={DELTA}, ε={EPSILON}, γ={GAMMA}")
    print(f"  Workers: {workers}")
    print(f"  Metrics: neighborhood regret (bucketed by iteration phase),")
    print(f"           importance weights, per-node profiles")
    print()

    results = []
    errors = []
    completed = 0
    t_start = time.time()

    incremental_path = os.path.join(output_dir, "incremental_results.jsonl")

    with ProcessPoolExecutor(max_workers=workers) as pool:
        future_to_job = {}
        for args in jobs:
            f = pool.submit(_run_job, *args)
            future_to_job[f] = args

        for future in as_completed(future_to_job):
            completed += 1
            elapsed_total = time.time() - t_start
            est_remaining = ((elapsed_total / completed)
                             * (total - completed))

            try:
                result = future.result()
                results.append(result)

                with open(incremental_path, "a") as incr:
                    incr.write(json.dumps(result) + "\n")

                print(f"  [{completed:>2d}/{total}] "
                      f"{result['config']:20s} "
                      f"sims={result['sims_per_move']:>3d}  "
                      f"expl={result['exploitability']:.4f}  "
                      f"({result['elapsed_s']:.1f}s)  "
                      f"ETA {est_remaining/60:.1f}m")
            except Exception as exc:
                job_args = future_to_job[future]
                err_msg = f"  [{completed:>2d}/{total}] FAILED: {exc}"
                print(err_msg)
                errors.append({"args": str(job_args), "error": str(exc),
                               "traceback": traceback.format_exc()})

    # Sort results
    results.sort(key=lambda r: (r["config"], r["sims_per_move"]))

    # Save full results
    results_path = os.path.join(output_dir, "probe_results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)

    elapsed_total = time.time() - t_start
    print(f"\nDone in {elapsed_total/60:.1f}m. "
          f"{len(results)} completed, {len(errors)} errors.")
    print(f"Results: {results_path}")
    if errors:
        err_path = os.path.join(output_dir, "errors.json")
        with open(err_path, "w") as f:
            json.dump(errors, f, indent=2)
        print(f"Errors: {err_path}")


# ── CLI ──────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="15C: Low-budget proximity probe (Leduc only)")
    parser.add_argument("--quick", action="store_true",
                        help=f"Use {QUICK_MATCHES} matches (faster)")
    parser.add_argument("--budgets", nargs="+", type=int,
                        help="Override sim budgets (default: 5 10 20 50)")
    parser.add_argument("--matches", type=int,
                        help="Override match count")
    parser.add_argument("--workers", type=int, default=WORKERS,
                        help=f"Worker count (default: {WORKERS})")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed (default: 42)")
    parser.add_argument("--output", type=str,
                        help="Output directory override")

    args = parser.parse_args()

    num_matches = NUM_MATCHES
    if args.quick:
        num_matches = QUICK_MATCHES
    if args.matches:
        num_matches = args.matches

    run_probe(
        budgets=args.budgets,
        num_matches=num_matches,
        workers=args.workers,
        seed=args.seed,
        output_dir=args.output,
    )


if __name__ == "__main__":
    main()
