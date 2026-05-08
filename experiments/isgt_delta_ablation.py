"""ISGT delta ablation: targeting probability × decay functions × games.

Tests the hypothesis that ISGT's exploration deficit (δ=1.0, all iterations
targeted) explains the OOS crossover at high budgets. Sweeps δ ∈ {0.2, 0.5,
0.9, 1.0} to find the right targeting/exploration balance.

Grid:
  - Games: kuhn_poker, leduc_poker, liars_dice, goofspiel
  - Bias modes: chance, full
  - Decay functions: constant (baseline), exp_0.5 (shaped representative)
  - Delta values: 0.2, 0.5, 0.9, 1.0
  - Sim budgets: 100, 250, 500, 1000
  - Baseline: OOS (δ=0.9)

Total: 4 games × 2 modes × 2 decays × 4 deltas × 4 budgets + 4×4 OOS
     = 256 + 16 = 272 jobs

Key questions answered:
  1. Does δ < 1.0 fix the OOS crossover on Leduc and Liar's Dice?
  2. Does proper exploration unmask decay function signal
     (i.e., does exp_0.5 beat constant at the right δ)?

Usage:
    # Quick sanity check (~3 min)
    python -m experiments.isgt_delta_ablation --quick

    # Full run (~60 min)
    python -m experiments.isgt_delta_ablation --full

    # Custom
    python -m experiments.isgt_delta_ablation --budgets 100,500 --match-scale 0.5
"""

# Force unbuffered stdout for piped output (tee, redirect)
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
    ISGTBot, ExponentialDecay, ConstantWeight,
)
from eval.aggregate_exploitability import aggregate_exploitability


# ── Configuration ────────────────────────────────────────────────────────

GAMES = ["kuhn_poker", "leduc_poker", "liars_dice", "goofspiel"]

BIAS_MODES = ["chance", "full"]

# Focused decay set: constant (baseline) + one shaped representative
DECAY_FUNCTIONS = [
    ("constant",  ConstantWeight()),
    ("exp_0.5",   ExponentialDecay(0.5)),
]

DELTA_VALUES = [0.2, 0.5, 0.9, 1.0]

# Per-game match counts (same as archive 007)
GAME_MATCHES = {
    "kuhn_poker": 750,
    "leduc_poker": 450,
    "goofspiel": 450,
    "liars_dice": 150,
}

QUICK_MATCH_SCALE = 0.1
BUDGETS = [100, 250, 500, 1000]
WORKERS = 6


# ── Single job (runs in worker process) ──────────────────────────────────

def _run_isgt_job(game_name, bias_mode, decay_name, decay_fn,
                  delta, sims_per_move, num_matches, seed):
    """Run one ISGT (game, mode, decay, delta, budget) combination."""
    game = load_game(game_name)
    iig = IIG(game)
    counter = [0]

    def bot_factory(player_id):
        s = seed + counter[0]
        counter[0] += 1
        return ISGTBot(game, player_id, num_simulations=sims_per_move,
                       epsilon=0.6, gamma=0.01,
                       level_weight_fn=decay_fn,
                       bias_mode=bias_mode, seed=s, iig=iig,
                       delta=delta)

    t0 = time.time()
    expl = aggregate_exploitability(game, bot_factory,
                                    num_matches=num_matches, seed=seed)
    elapsed = time.time() - t0

    config_name = f"{bias_mode}_{decay_name}_d{delta}"
    return {
        "game": game_name,
        "config": config_name,
        "bias_mode": bias_mode,
        "decay": decay_name,
        "delta": delta,
        "sims_per_move": sims_per_move,
        "exploitability": expl,
        "elapsed_s": round(elapsed, 1),
        "num_matches": num_matches,
    }


def _run_oos_job(game_name, sims_per_move, num_matches, seed,
                 precomp=False):
    """Run one OOS baseline job."""
    game = load_game(game_name)
    iig = IIG(game) if precomp else None
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

    config_label = "oos_precomp" if precomp else "oos"
    return {
        "game": game_name,
        "config": config_label,
        "bias_mode": "n/a",
        "decay": "n/a",
        "delta": 0.9,
        "sims_per_move": sims_per_move,
        "exploitability": expl,
        "elapsed_s": round(elapsed, 1),
        "num_matches": num_matches,
    }


# ── Main harness ─────────────────────────────────────────────────────────

def run_ablation(budgets, match_scale=1.0, workers=WORKERS, seed=42,
                 output_dir="results/isgt_delta_ablation", precomp=False):
    """Run the delta ablation grid."""
    os.makedirs(output_dir, exist_ok=True)

    jobs = []
    for game_name in GAMES:
        num_matches = max(10, int(GAME_MATCHES[game_name] * match_scale))
        # ISGT configs
        for bias_mode in BIAS_MODES:
            for decay_name, decay_fn in DECAY_FUNCTIONS:
                for delta in DELTA_VALUES:
                    for sims in sorted(budgets):
                        jobs.append((
                            _run_isgt_job,
                            (game_name, bias_mode, decay_name, decay_fn,
                             delta, sims, num_matches, seed),
                        ))
        # OOS baseline
        for sims in sorted(budgets):
            jobs.append((
                _run_oos_job,
                (game_name, sims, num_matches, seed, precomp),
            ))

    total = len(jobs)
    isgt_count = (len(GAMES) * len(BIAS_MODES) * len(DECAY_FUNCTIONS)
                  * len(DELTA_VALUES) * len(budgets))
    oos_count = len(GAMES) * len(budgets)
    match_info = ', '.join(
        f'{g}: {max(10, int(GAME_MATCHES[g] * match_scale))}' for g in GAMES
    )
    precomp_tag = " [precomp]" if precomp else ""
    print(f"ISGT Delta Ablation: {total} jobs "
          f"({isgt_count} ISGT + {oos_count} OOS baseline){precomp_tag}")
    print(f"  Games: {GAMES}")
    print(f"  Matches per game: {{{match_info}}}")
    print(f"  Workers: {workers}")
    print(f"  Budgets: {sorted(budgets)}")
    print(f"  Deltas: {DELTA_VALUES}")
    print(f"  Decays: {[n for n, _ in DECAY_FUNCTIONS]}")
    print(f"  Modes: {BIAS_MODES}")
    print()

    results = []
    errors = []
    completed = 0
    t_start = time.time()

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

                with open(incremental_path, "a") as incr:
                    incr.write(json.dumps(result) + "\n")

                print(f"  [{completed:>4d}/{total}] "
                      f"{result['game']:15s} {result['config']:30s} "
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

    results.sort(key=lambda r: (r["game"], r["bias_mode"], r["decay"],
                                 r["delta"], r["sims_per_move"]))

    out_path = os.path.join(output_dir, "ablation_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

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

    _print_summary(summary, budgets)
    return results


def _print_summary(summary, budgets):
    """Print a compact summary table."""
    sorted_budgets = sorted(budgets)
    budget_cols = " ".join(f"{'sims='+str(b):>12s}" for b in sorted_budgets)
    header = f"{'Game':15s} {'Config':30s} {budget_cols}"
    print(f"\n{'='*len(header)}")
    print(header)
    print(f"{'='*len(header)}")

    for game in GAMES:
        if game not in summary:
            continue
        for config in sorted(summary[game]):
            row = f"{game:15s} {config:30s} "
            budget_map = {r["sims_per_move"]: r["exploitability"]
                          for r in summary[game][config]}
            for b in sorted_budgets:
                if b in budget_map:
                    row += f"{budget_map[b]:>12.4f} "
                else:
                    row += f"{'---':>12s} "
            print(row)
        print()


# ── CLI ──────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="ISGT delta ablation study")
    parser.add_argument("--quick", action="store_true",
                        help="Quick mode: 10%% matches, all budgets")
    parser.add_argument("--full", action="store_true",
                        help="Full mode: 100%% matches, all budgets")
    parser.add_argument("--budgets", type=str, default=None,
                        help="Comma-separated sim budgets (default: 100,250,500,1000)")
    parser.add_argument("--match-scale", type=float, default=None,
                        help="Scale factor for match counts")
    parser.add_argument("--workers", type=int, default=WORKERS,
                        help=f"Worker processes (default: {WORKERS})")
    parser.add_argument("--output", type=str, default=None,
                        help="Output directory")
    parser.add_argument("--precomp", action="store_true",
                        help="Pre-initialize OOS infostates from IIG ")
    args = parser.parse_args()

    budgets = BUDGETS
    if args.budgets:
        budgets = [int(x) for x in args.budgets.split(",")]

    match_scale = 1.0
    output_dir = "results/isgt_delta_ablation"

    if args.quick:
        match_scale = QUICK_MATCH_SCALE
        output_dir = "results/isgt_delta_ablation_quick"
    elif args.full:
        match_scale = 1.0
        output_dir = "results/isgt_delta_ablation_full"

    if args.match_scale is not None:
        match_scale = args.match_scale
    if args.output is not None:
        output_dir = args.output

    run_ablation(budgets, match_scale=match_scale,
                 workers=args.workers, output_dir=output_dir,
                 precomp=args.precomp)
