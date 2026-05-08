"""Experiment 014: Post-Fix Delta × Decay Baseline.

Full post-fix ablation with corrected OOS and corrected ISGT (s1=1.0).
Sweeps all 7 decay functions from archive 013 across 3 δ values for ISGT,
with OOS at δ=0.9 (paper setting) as the fixed benchmark.

Corrections in place:
  - OOS: per-iteration δ flip, deterministic targeting at chance + decision nodes
  - ISGT: s1=1.0 (conditional importance weighting; z* is context, not weight)

Even with the s1 fix, ISGT routes targeted iterations toward a sampled z*,
and unlikely z* terminals produce large importance-weighted updates via the
s2 path.  This experiment checks how δ × decay interact for ISGT, and
whether full mode is now competitive with chance mode.

Grid:
  - Games: kuhn_poker, leduc_poker, liars_dice, goofspiel
  - ISGT δ×decay sweep:
      Decays (7): t_balanced, constant, exp_0.7, exp_0.5,
                  step_0.25, step_0.1, poly_2
      Deltas (3): 0.2, 0.5, 0.9
      Modes (2):  chance, full
      Budgets:    250, 1000
  - OOS baseline: δ=0.9 (paper setting) × 2 budgets
  - ε=0.6, γ=0.01 (fixed)

Jobs:
  ISGT δ×decay:  4 games × 2 modes × 7 decays × 3 deltas × 2 budgets = 336
  OOS baseline:  4 games × 2 budgets                                  =   8
  Total:                                                               = 344

Key questions:
  1. Sanity check: do all algorithms produce similar results at δ=0?
  2. Does the s1 fix change which decay functions win?
     (archive 013 found t_balanced best, but under the old buggy code)
  3. Optimal δ for each decay × mode combination — does it vary?
  4. Full vs chance mode — does full mode now compete at the right δ?
  5. ISGT vs OOS — at what δ/decay does ISGT match or beat OOS (δ=0.9)?
  6. Game-size dependence — does Liar's Dice (147K terminals) penalize ISGT?

Usage:
    # Quick (~10 min, 25% matches)
    python -m experiments.postfix_delta_baseline_014 --quick

    # Full (~90 min, 100% matches)
    python -m experiments.postfix_delta_baseline_014 --full
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

from src.games import load_game
from src.iig import IIG
from src.oos import OOSBot
from src.isgt import (
    ISGTBot, TerminalBalancedWeight, ConstantWeight, ExponentialDecay,
    StepFunction, PolynomialDecay,
)
from eval.aggregate_exploitability import aggregate_exploitability


# ── Configuration ────────────────────────────────────────────────────────

GAMES = ["kuhn_poker", "leduc_poker", "liars_dice", "goofspiel"]

BIAS_MODES = ["chance", "full"]

# Same 7 decay functions as archive 013 two-stage screening
DECAY_FUNCTIONS = [
    ("t_balanced", TerminalBalancedWeight()),
    ("constant",   ConstantWeight()),
    ("exp_0.7",    ExponentialDecay(0.7)),
    ("exp_0.5",    ExponentialDecay(0.5)),
    ("step_0.25",  StepFunction(0.25)),
    ("step_0.1",   StepFunction(0.1)),
    ("poly_2",     PolynomialDecay(2.0)),
]

ISGT_DELTA_VALUES = [0.2, 0.5, 0.9]  # δ sweep for ISGT
OOS_DELTA = 0.5                       # Paper uses δ=0.5 for aggregate exploitability (§4.4.2)

BUDGETS = [250, 1000]

# Per-game match counts (same as prior archives)
GAME_MATCHES = {
    "kuhn_poker": 750,
    "leduc_poker": 450,
    "goofspiel": 450,
    "liars_dice": 150,
}

QUICK_MATCH_SCALE = 0.25
WORKERS = 6


# ── Single job (runs in worker process) ──────────────────────────────────

def _run_isgt_job(game_name, bias_mode, decay_name, decay_fn,
                  delta, sims_per_move, num_matches, seed):
    """Run one ISGT configuration."""
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

    config_name = f"isgt_{bias_mode}_{decay_name}_d{delta}"
    return {
        "game": game_name,
        "algorithm": "isgt",
        "config": config_name,
        "bias_mode": bias_mode,
        "decay": decay_name,
        "delta": delta,
        "sims_per_move": sims_per_move,
        "exploitability": expl,
        "elapsed_s": round(elapsed, 1),
        "num_matches": num_matches,
    }


def _run_oos_job(game_name, delta, sims_per_move, num_matches, seed):
    """Run one OOS configuration."""
    game = load_game(game_name)
    iig = IIG(game)
    counter = [0]

    def bot_factory(player_id):
        s = seed + counter[0]
        counter[0] += 1
        return OOSBot(game, player_id, num_simulations=sims_per_move,
                      delta=delta, epsilon=0.6, gamma=0.01, seed=s)

    t0 = time.time()
    expl = aggregate_exploitability(game, bot_factory,
                                    num_matches=num_matches, seed=seed)
    elapsed = time.time() - t0

    config_name = f"oos_d{delta}"
    return {
        "game": game_name,
        "algorithm": "oos",
        "config": config_name,
        "bias_mode": "n/a",
        "decay": "n/a",
        "delta": delta,
        "sims_per_move": sims_per_move,
        "exploitability": expl,
        "elapsed_s": round(elapsed, 1),
        "num_matches": num_matches,
    }


# ── Main harness ─────────────────────────────────────────────────────────

def run_ablation(budgets, match_scale=1.0, workers=WORKERS, seed=42,
                 output_dir="results/014_postfix_delta_baseline"):
    """Run the post-fix delta baseline grid."""
    os.makedirs(output_dir, exist_ok=True)

    jobs = []
    for game_name in GAMES:
        num_matches = max(10, int(GAME_MATCHES[game_name] * match_scale))

        # ISGT δ × decay sweep
        for bias_mode in BIAS_MODES:
            for decay_name, decay_fn in DECAY_FUNCTIONS:
                for delta in ISGT_DELTA_VALUES:
                    for sims in sorted(budgets):
                        jobs.append((
                            _run_isgt_job,
                            (game_name, bias_mode, decay_name, decay_fn,
                             delta, sims, num_matches, seed),
                        ))

        # OOS baseline at δ=0.9 (paper setting)
        for sims in sorted(budgets):
            jobs.append((
                _run_oos_job,
                (game_name, OOS_DELTA, sims, num_matches, seed),
            ))

    total = len(jobs)
    n_decays = len(DECAY_FUNCTIONS)
    isgt_sweep = len(GAMES) * len(BIAS_MODES) * n_decays * len(ISGT_DELTA_VALUES) * len(budgets)
    oos_baseline = len(GAMES) * len(budgets)
    match_info = ', '.join(
        f'{g}: {max(10, int(GAME_MATCHES[g] * match_scale))}' for g in GAMES
    )
    decay_names = [n for n, _ in DECAY_FUNCTIONS]
    print(f"Experiment 014 — Post-Fix Delta × Decay Baseline: {total} jobs")
    print(f"  ISGT δ×decay: {isgt_sweep} "
          f"({n_decays} decays × {len(ISGT_DELTA_VALUES)} deltas "
          f"× {len(BIAS_MODES)} modes × {len(budgets)} budgets × {len(GAMES)} games)")
    print(f"  OOS baseline: {oos_baseline} (δ={OOS_DELTA})")
    print(f"  Games: {GAMES}")
    print(f"  Matches per game: {{{match_info}}}")
    print(f"  Workers: {workers}")
    print(f"  Budgets: {sorted(budgets)}")
    print(f"  Deltas: {ISGT_DELTA_VALUES}")
    print(f"  Decays: {decay_names}")
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
                      f"{result['game']:15s} {result['config']:35s} "
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

    results.sort(key=lambda r: (r["game"], r["algorithm"], r["bias_mode"],
                                 r["decay"], r["delta"], r["sims_per_move"]))

    # ── Save results ──
    out_path = os.path.join(output_dir, "ablation_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    # ── Build summary ──
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
    """Print a compact summary table grouped by game."""
    sorted_budgets = sorted(budgets)
    budget_cols = " ".join(f"{'sims='+str(b):>12s}" for b in sorted_budgets)
    header = f"{'Game':15s} {'Config':35s} {budget_cols}"
    sep = "=" * len(header)
    print(f"\n{sep}")
    print(header)
    print(sep)

    for game in GAMES:
        if game not in summary:
            continue
        for config in sorted(summary[game]):
            row = f"{game:15s} {config:35s} "
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
    parser = argparse.ArgumentParser(
        description="Experiment 014: Post-Fix Delta Baseline")
    parser.add_argument("--quick", action="store_true",
                        help="Quick mode: 25%% matches")
    parser.add_argument("--full", action="store_true",
                        help="Full mode: 100%% matches")
    parser.add_argument("--budgets", type=str, default=None,
                        help="Comma-separated sim budgets (default: 250,1000)")
    parser.add_argument("--match-scale", type=float, default=None,
                        help="Scale factor for match counts")
    parser.add_argument("--workers", type=int, default=WORKERS,
                        help=f"Worker processes (default: {WORKERS})")
    parser.add_argument("--output", type=str, default=None,
                        help="Output directory")
    args = parser.parse_args()

    budgets = BUDGETS
    if args.budgets:
        budgets = [int(x) for x in args.budgets.split(",")]

    match_scale = 1.0
    output_dir = "results/014_postfix_delta_baseline"

    if args.quick:
        match_scale = QUICK_MATCH_SCALE
        output_dir = "results/014_postfix_delta_baseline_quick"
    elif args.full:
        match_scale = 1.0
        output_dir = "results/014_postfix_delta_baseline_full"

    if args.match_scale is not None:
        match_scale = args.match_scale
    if args.output is not None:
        output_dir = args.output

    run_ablation(budgets, match_scale=match_scale,
                 workers=args.workers, output_dir=output_dir)
