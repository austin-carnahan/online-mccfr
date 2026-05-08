"""Experiment 15A: Mechanism Isolation — Aggregate Exploitability.

At matched parameters (δ=0.5, ε=0.4, γ=0.01), compare OOS's partial-
history targeting against ISGT's full-terminal-history targeting.  Two
decay baselines (constant, t_balanced) isolate the targeting mechanism
from proximity shaping.

Configs (5):
  1. OOS (IST)             — paper's algorithm, the benchmark
  2. ISGT chance constant   — no proximity bias, chance targeting only
  3. ISGT chance t_balanced — tree-shape-aware, chance targeting only
  4. ISGT full constant     — no proximity bias, full targeting
  5. ISGT full t_balanced   — tree-shape-aware, full targeting

Fixed: δ=0.5, ε=0.4, γ=0.01 (paper §4.4.2 settings)
Games: kuhn_poker, leduc_poker, liars_dice, goofspiel
Budgets: 100, 250, 500, 1000 sims/move

Metrics per job:
  - Exploitability (primary)
  - Infoset coverage fraction (per player)
  - Level-stratified weight profile (coverage + mean weight by IIG distance)
  - Regret update magnitude by z* level (ISGT only)

Jobs: 4 games × 5 configs × 4 budgets = 80

Usage:
    python -m experiments.mechanism_isolation_015a --quick
    python -m experiments.mechanism_isolation_015a --full
"""

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
from src.isgt import ISGTBot, ConstantWeight, TerminalBalancedWeight
from eval.aggregate_metrics import aggregate_with_metrics


# ── Configuration ────────────────────────────────────────────────────────

GAMES = ["kuhn_poker", "leduc_poker", "liars_dice", "goofspiel"]

# Paper §4.4.2 parameters
DELTA = 0.5
EPSILON = 0.4
GAMMA = 0.01

BUDGETS = [100, 250, 500, 1000]

# ISGT configs: (label, bias_mode, decay_name, decay_fn)
ISGT_CONFIGS = [
    ("isgt_chance_constant",  "chance", "constant",   ConstantWeight()),
    ("isgt_chance_tbal",      "chance", "t_balanced",  TerminalBalancedWeight()),
    ("isgt_full_constant",    "full",   "constant",   ConstantWeight()),
    ("isgt_full_tbal",        "full",   "t_balanced",  TerminalBalancedWeight()),
]

# Per-game match counts (same as prior experiments)
GAME_MATCHES = {
    "kuhn_poker": 750,
    "leduc_poker": 450,
    "goofspiel": 450,
    "liars_dice": 150,
}

QUICK_MATCH_SCALE = 0.25
WORKERS = 6


# ── Single jobs (run in worker processes) ────────────────────────────────

def _run_isgt_job(game_name, label, bias_mode, decay_fn,
                  sims_per_move, num_matches, seed):
    """Run one ISGT configuration with full metrics."""
    game = load_game(game_name)
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
        num_matches=num_matches, seed=seed, level_profile=True)
    elapsed = time.time() - t0

    return {
        "game": game_name,
        "algorithm": "isgt",
        "config": label,
        "bias_mode": bias_mode,
        "decay": decay_fn.name(),
        "delta": DELTA,
        "epsilon": EPSILON,
        "sims_per_move": sims_per_move,
        "exploitability": metrics["exploitability"],
        "infoset_coverage": metrics["infoset_coverage"],
        "level_profile": metrics.get("level_profile", {}),
        "regret_by_level": metrics.get("regret_by_level", {}),
        "neighborhood_regret_by_level": metrics.get("neighborhood_regret_by_level", {}),
        "neighborhood_touched_by_level": metrics.get("neighborhood_touched_by_level", {}),
        "elapsed_s": round(elapsed, 1),
        "num_matches": num_matches,
    }


def _run_oos_job(game_name, sims_per_move, num_matches, seed):
    """Run OOS baseline with metrics."""
    game = load_game(game_name)
    iig = IIG(game)
    counter = [0]

    def bot_factory(player_id):
        s = seed + counter[0]
        counter[0] += 1
        return OOSBot(game, player_id, num_simulations=sims_per_move,
                      delta=DELTA, epsilon=EPSILON, gamma=GAMMA,
                      seed=s, iig=iig)

    t0 = time.time()
    metrics = aggregate_with_metrics(
        game, bot_factory, iig,
        num_matches=num_matches, seed=seed, level_profile=True)
    elapsed = time.time() - t0

    return {
        "game": game_name,
        "algorithm": "oos",
        "config": "oos",
        "bias_mode": "n/a",
        "decay": "n/a",
        "delta": DELTA,
        "epsilon": EPSILON,
        "sims_per_move": sims_per_move,
        "exploitability": metrics["exploitability"],
        "infoset_coverage": metrics["infoset_coverage"],
        "level_profile": metrics.get("level_profile", {}),
        "regret_by_level": {},
        "neighborhood_regret_by_level": {},
        "neighborhood_touched_by_level": {},
        "elapsed_s": round(elapsed, 1),
        "num_matches": num_matches,
    }


# ── Main harness ─────────────────────────────────────────────────────────

def run_ablation(budgets, match_scale=1.0, workers=WORKERS, seed=42,
                 output_dir="results/015a_mechanism_isolation"):
    """Run the 15A mechanism isolation experiment."""
    os.makedirs(output_dir, exist_ok=True)

    jobs = []
    for game_name in GAMES:
        num_matches = max(10, int(GAME_MATCHES[game_name] * match_scale))

        # ISGT configs
        for label, bias_mode, decay_name, decay_fn in ISGT_CONFIGS:
            for sims in sorted(budgets):
                jobs.append((
                    _run_isgt_job,
                    (game_name, label, bias_mode, decay_fn,
                     sims, num_matches, seed),
                ))

        # OOS baseline
        for sims in sorted(budgets):
            jobs.append((
                _run_oos_job,
                (game_name, sims, num_matches, seed),
            ))

    total = len(jobs)
    isgt_jobs = len(GAMES) * len(ISGT_CONFIGS) * len(budgets)
    oos_jobs = len(GAMES) * len(budgets)
    match_info = ', '.join(
        f'{g}: {max(10, int(GAME_MATCHES[g] * match_scale))}' for g in GAMES
    )
    print(f"Experiment 15A — Mechanism Isolation: {total} jobs")
    print(f"  ISGT: {isgt_jobs} (4 configs × {len(budgets)} budgets "
          f"× {len(GAMES)} games)")
    print(f"  OOS:  {oos_jobs} (1 config × {len(budgets)} budgets "
          f"× {len(GAMES)} games)")
    print(f"  Fixed: δ={DELTA}, ε={EPSILON}, γ={GAMMA}")
    print(f"  Games: {GAMES}")
    print(f"  Matches per game: {{{match_info}}}")
    print(f"  Workers: {workers}")
    print(f"  Budgets: {sorted(budgets)}")
    print(f"  Metrics: exploitability, coverage, level_profile, "
          f"regret_by_level (ISGT)")
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

                cov = result["infoset_coverage"]
                cov_str = f"cov=[{cov[0]:.2f},{cov[1]:.2f}]"
                print(f"  [{completed:>3d}/{total}] "
                      f"{result['game']:15s} {result['config']:25s} "
                      f"sims={result['sims_per_move']:>5d}  "
                      f"expl={result['exploitability']:.6f}  "
                      f"{cov_str}  "
                      f"({result['elapsed_s']:.1f}s)  "
                      f"ETA {est_remaining/60:.1f}m")
            except Exception as exc:
                job_args = future_to_job[future]
                err_msg = f"  [{completed:>3d}/{total}] FAILED: {exc}"
                print(err_msg)
                errors.append({"args": str(job_args), "error": str(exc),
                               "traceback": traceback.format_exc()})

    if errors:
        print(f"\n*** {len(errors)} job(s) failed ***")
        for e in errors:
            print(f"  {e['args']}: {e['error']}")

    results.sort(key=lambda r: (r["game"], r["config"], r["sims_per_move"]))

    # ── Save full results ──
    out_path = os.path.join(output_dir, "ablation_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    # ── Build summary (primary metric only) ──
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
            "infoset_coverage": r["infoset_coverage"],
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
    header = f"{'Game':15s} {'Config':25s} {budget_cols}"
    sep = "=" * len(header)
    print(f"\n{sep}")
    print(header)
    print(sep)

    for game in GAMES:
        if game not in summary:
            continue
        for config in sorted(summary[game]):
            row = f"{game:15s} {config:25s} "
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
        description="Experiment 15A: Mechanism Isolation")
    parser.add_argument("--quick", action="store_true",
                        help="Quick mode: 25%% matches")
    parser.add_argument("--full", action="store_true",
                        help="Full mode: 100%% matches")
    parser.add_argument("--budgets", type=str, default=None,
                        help="Comma-separated sim budgets "
                             "(default: 100,250,500,1000)")
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
    output_dir = "results/015a_mechanism_isolation"

    if args.quick:
        match_scale = QUICK_MATCH_SCALE
        output_dir = "results/015a_mechanism_isolation_quick"
    elif args.full:
        match_scale = 1.0
        output_dir = "results/015a_mechanism_isolation_full"

    if args.match_scale is not None:
        match_scale = args.match_scale
    if args.output is not None:
        output_dir = args.output

    run_ablation(budgets, match_scale=match_scale,
                 workers=args.workers, output_dir=output_dir)
