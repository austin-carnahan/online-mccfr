"""ISGT ablation study: decay functions × bias modes × games.

Runs aggregate exploitability (online play vs random) for every
combination of (game, bias_mode, decay_function, sims_per_move).
Multi-threaded via concurrent.futures.ProcessPoolExecutor.

Usage:
    # Quick sanity check (~2 min)
    python -m experiments.isgt_ablation --quick

    # Full ablation (~20 min)
    python -m experiments.isgt_ablation --full

    # Custom
    python -m experiments.isgt_ablation --budgets 100,500 --match-scale 0.5 --workers 6
"""

# Force unbuffered stdout so progress prints appear immediately,
# even when piped through tee or redirected to a file.
import sys
import os
if not sys.stdout.line_buffering:
    sys.stdout.reconfigure(line_buffering=True)

import json
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np

from src.games import load_game
from src.iig import IIG
from src.oos import OOSBot
from src.isgt import (
    ISGTBot, ExponentialDecay, PolynomialDecay, LinearDecay,
    ConstantWeight,
)
from eval.aggregate_exploitability import aggregate_exploitability


# ── Configuration ────────────────────────────────────────────────────────

GAMES = ["kuhn_poker", "leduc_poker", "liars_dice", "goofspiel"]

BIAS_MODES = ["chance", "full"]

# Each entry: (short_name, LevelWeightFn instance)
DECAY_FUNCTIONS = [
    ("exp_0.3",   ExponentialDecay(0.3)),
    ("exp_0.5",   ExponentialDecay(0.5)),
    ("exp_0.7",   ExponentialDecay(0.7)),
    ("poly_2",    PolynomialDecay(2.0)),
    ("linear_4",  LinearDecay(4)),
    ("constant",  ConstantWeight()),
]

# Per-game match counts (liar's dice is expensive)
GAME_MATCHES = {
    "kuhn_poker": 500,
    "leduc_poker": 300,
    "goofspiel": 300,
    "liars_dice": 100,
}

QUICK_MATCH_SCALE = 0.1  # quick mode uses 10% of full matches

BUDGETS = [100, 250, 500, 1000]
WORKERS = 6


# ── Single job (runs in worker process) ──────────────────────────────────

def _run_isgt_job(game_name, bias_mode, decay_name, decay_fn,
                  sims_per_move, num_matches, seed):
    """Run one ISGT (game, mode, decay, budget) combination."""
    game = load_game(game_name)
    iig = IIG(game)
    counter = [0]

    def bot_factory(player_id):
        s = seed + counter[0]
        counter[0] += 1
        return ISGTBot(game, player_id, num_simulations=sims_per_move,
                       epsilon=0.6, gamma=0.01,
                       level_weight_fn=decay_fn,
                       bias_mode=bias_mode, seed=s, iig=iig)

    t0 = time.time()
    expl = aggregate_exploitability(game, bot_factory,
                                    num_matches=num_matches, seed=seed)
    elapsed = time.time() - t0

    config_name = f"{bias_mode}_{decay_name}"
    return {
        "game": game_name,
        "config": config_name,
        "bias_mode": bias_mode,
        "decay": decay_name,
        "sims_per_move": sims_per_move,
        "exploitability": expl,
        "elapsed_s": round(elapsed, 1),
        "num_matches": num_matches,
    }


def _run_oos_job(game_name, sims_per_move, num_matches, seed):
    """Run one OOS baseline job."""
    game = load_game(game_name)
    counter = [0]

    def bot_factory(player_id):
        s = seed + counter[0]
        counter[0] += 1
        return OOSBot(game, player_id, num_simulations=sims_per_move,
                      delta=0.9, epsilon=0.6, gamma=0.01, seed=s)

    t0 = time.time()
    expl = aggregate_exploitability(game, bot_factory,
                                    num_matches=num_matches, seed=seed)
    elapsed = time.time() - t0

    return {
        "game": game_name,
        "config": "oos",
        "bias_mode": "n/a",
        "decay": "n/a",
        "sims_per_move": sims_per_move,
        "exploitability": expl,
        "elapsed_s": round(elapsed, 1),
        "num_matches": num_matches,
    }


# ── Main harness ─────────────────────────────────────────────────────────

def run_ablation(budgets, match_scale=1.0, workers=WORKERS, seed=42,
                 output_dir="results/isgt_ablation"):
    """Run the full ablation grid."""
    os.makedirs(output_dir, exist_ok=True)

    # Build job list: (fn, args) tuples
    jobs = []  # (callable, args, description)
    for game_name in GAMES:
        num_matches = max(10, int(GAME_MATCHES[game_name] * match_scale))
        # ISGT configs
        for bias_mode in BIAS_MODES:
            for decay_name, decay_fn in DECAY_FUNCTIONS:
                for sims in sorted(budgets):
                    jobs.append((
                        _run_isgt_job,
                        (game_name, bias_mode, decay_name, decay_fn,
                         sims, num_matches, seed),
                    ))
        # OOS baseline
        for sims in sorted(budgets):
            jobs.append((
                _run_oos_job,
                (game_name, sims, num_matches, seed),
            ))

    total = len(jobs)
    isgt_count = len(GAMES) * len(BIAS_MODES) * len(DECAY_FUNCTIONS) * len(budgets)
    oos_count = len(GAMES) * len(budgets)
    match_info = ', '.join(
        f'{g}: {max(10, int(GAME_MATCHES[g] * match_scale))}' for g in GAMES
    )
    print(f"ISGT Ablation: {total} jobs "
          f"({isgt_count} ISGT + {oos_count} OOS baseline)")
    print(f"  Games: {GAMES}")
    print(f"  Matches per game: {{{match_info}}}")
    print(f"  Workers: {workers}")
    print(f"  Budgets: {sorted(budgets)}")
    print(f"  Decays: {[n for n, _ in DECAY_FUNCTIONS]}")
    print()

    results = []
    errors = []
    completed = 0
    t_start = time.time()

    # Incremental results file — each completed job appends a line
    incremental_path = os.path.join(output_dir, "incremental_results.jsonl")

    with ProcessPoolExecutor(max_workers=workers) as pool:
        future_to_job = {}
        for fn, args in jobs:
            f = pool.submit(fn, *args)
            future_to_job[f] = args

        for future in as_completed(future_to_job):
            completed += 1
            elapsed_total = time.time() - t_start
            est_remaining = (elapsed_total / completed) * (total - completed)

            try:
                result = future.result()
                results.append(result)

                # Save incrementally (one JSON object per line)
                with open(incremental_path, "a") as incr:
                    incr.write(json.dumps(result) + "\n")

                print(f"  [{completed:>4d}/{total}] "
                      f"{result['game']:15s} {result['config']:25s} "
                      f"sims={result['sims_per_move']:>5d}  "
                      f"expl={result['exploitability']:.6f}  "
                      f"({result['elapsed_s']:.1f}s)  "
                      f"ETA {est_remaining/60:.1f}m")
            except Exception as exc:
                job_args = future_to_job[future]
                err_msg = f"  [{completed:>4d}/{total}] FAILED: {exc}"
                print(err_msg)
                errors.append({"args": str(job_args), "error": str(exc),
                               "traceback": traceback.format_exc()})

    if errors:
        print(f"\n*** {len(errors)} job(s) failed ***")
        for e in errors:
            print(f"  {e['args']}: {e['error']}")

    # Sort results for readability
    results.sort(key=lambda r: (r["game"], r["bias_mode"], r["decay"],
                                 r["sims_per_move"]))

    # Save raw results
    out_path = os.path.join(output_dir, "ablation_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    # Build summary: game → config → [budget results]
    summary = {}
    for r in results:
        game = r["game"]
        config = r["config"]
        if game not in summary:
            summary[game] = {}
        if config not in summary[game]:
            summary[game][config] = []
        summary[game][config].append({
            "sims_per_move": r["sims_per_move"],
            "exploitability": r["exploitability"],
        })

    # Sort each config's results by budget
    for game in summary:
        for config in summary[game]:
            summary[game][config].sort(key=lambda x: x["sims_per_move"])

    summary_path = os.path.join(output_dir, "ablation_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    total_time = time.time() - t_start
    print(f"\nDone in {total_time/60:.1f} minutes "
          f"({len(results)} completed, {len(errors)} failed).")
    print(f"Results saved to {output_dir}/")

    # Print summary table
    _print_summary(summary, budgets)

    return results


def _print_summary(summary, budgets):
    """Print a compact summary table."""
    sorted_budgets = sorted(budgets)
    budget_cols = " ".join(f"{'sims='+str(b):>12s}" for b in sorted_budgets)
    header = f"{'Game':15s} {'Config':25s} {budget_cols}"
    print(f"\n{'='*len(header)}")
    print(header)
    print(f"{'='*len(header)}")

    for game in GAMES:
        if game not in summary:
            continue
        for config in sorted(summary[game]):
            row = f"{game:15s} {config:25s} "
            budget_map = {r["sims_per_move"]: r["exploitability"]
                          for r in summary[game][config]}
            for b in sorted_budgets:
                if b in budget_map:
                    row += f" {budget_map[b]:>11.6f}"
                else:
                    row += f" {'---':>11s}"
            print(row)
        print()


def main():
    args = sys.argv[1:]

    # Defaults (full)
    budgets = list(BUDGETS)
    match_scale = 1.0
    workers = WORKERS
    seed = 42
    output_dir = "results/isgt_ablation"

    i = 0
    while i < len(args):
        if args[i] == "--quick":
            budgets = [100, 500]
            match_scale = QUICK_MATCH_SCALE
            i += 1
        elif args[i] == "--full":
            budgets = list(BUDGETS)
            match_scale = 1.0
            i += 1
        elif args[i] == "--budgets" and i + 1 < len(args):
            budgets = [int(x) for x in args[i + 1].split(",")]
            i += 2
        elif args[i] == "--match-scale" and i + 1 < len(args):
            match_scale = float(args[i + 1])
            i += 2
        elif args[i] == "--workers" and i + 1 < len(args):
            workers = int(args[i + 1])
            i += 2
        elif args[i] == "--seed" and i + 1 < len(args):
            seed = int(args[i + 1])
            i += 2
        elif args[i] == "--output" and i + 1 < len(args):
            output_dir = args[i + 1]
            i += 2
        else:
            print(f"Unknown arg: {args[i]}")
            i += 1

    run_ablation(budgets=budgets, match_scale=match_scale,
                 workers=workers, seed=seed, output_dir=output_dir)


if __name__ == "__main__":
    main()
