"""OOS vs ISGT(max_depth=0) equivalence check.

Sanity check: ISGT with max_iig_depth=0 should behave nearly identically
to Online Outcome Sampling — both deterministically walk root→active
infoset, then ε-on-policy forward.  Aggregate exploitability curves
should overlap.

Also establishes ISGT as a baseline for subsequent decay function ablation.

Usage:
    python -m experiments.oos_vs_isgt_depth0
    python -m experiments.oos_vs_isgt_depth0 --quick
    python -m experiments.oos_vs_isgt_depth0 --budgets 50,200,1000
"""

import sys
import os
if not sys.stdout.line_buffering:
    sys.stdout.reconfigure(line_buffering=True)

import json
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np

from src.games import load_game
from src.iig import IIG
from src.oos import OOSBot
from src.isgt import ISGTBot, LevelUniform
from eval.aggregate_exploitability import aggregate_exploitability


# ── Configuration ────────────────────────────────────────────────────────

GAMES = ["kuhn_poker", "leduc_poker", "goofspiel", "liars_dice"]

# Shared parameters (must be identical for both algorithms)
DELTA = 0.9
EPSILON = 0.6
GAMMA = 0.01

BUDGETS = [50, 200, 1000]

GAME_MATCHES = {
    "kuhn_poker": 500,
    "leduc_poker": 300,
    "goofspiel": 300,
    "liars_dice": 100,
}

QUICK_MATCH_SCALE = 0.1
WORKERS = 4


# ── Job functions (run in worker processes) ──────────────────────────────

def _run_oos_job(game_name, sims_per_move, num_matches, seed):
    """Run one OOS aggregate exploitability evaluation."""
    game = load_game(game_name)
    counter = [0]

    def bot_factory(player_id):
        s = seed + counter[0]
        counter[0] += 1
        return OOSBot(game, player_id, num_simulations=sims_per_move,
                      delta=DELTA, epsilon=EPSILON, gamma=GAMMA, seed=s)

    t0 = time.time()
    expl = aggregate_exploitability(game, bot_factory,
                                    num_matches=num_matches, seed=seed)
    elapsed = time.time() - t0

    return {
        "game": game_name,
        "algorithm": "oos",
        "sims_per_move": sims_per_move,
        "exploitability": expl,
        "elapsed_s": round(elapsed, 1),
        "num_matches": num_matches,
    }


def _run_isgt_depth0_job(game_name, sims_per_move, num_matches, seed):
    """Run one ISGT(max_depth=0) aggregate exploitability evaluation."""
    game = load_game(game_name)
    iig = IIG(game)
    counter = [0]

    def bot_factory(player_id):
        s = seed + counter[0]
        counter[0] += 1
        return ISGTBot(game, player_id, num_simulations=sims_per_move,
                       epsilon=EPSILON, gamma=GAMMA,
                       level_weight_fn=LevelUniform(),
                       bias_mode="full", seed=s, iig=iig,
                       delta=DELTA, max_iig_depth=0)

    t0 = time.time()
    expl = aggregate_exploitability(game, bot_factory,
                                    num_matches=num_matches, seed=seed)
    elapsed = time.time() - t0

    return {
        "game": game_name,
        "algorithm": "isgt_depth0",
        "sims_per_move": sims_per_move,
        "exploitability": expl,
        "elapsed_s": round(elapsed, 1),
        "num_matches": num_matches,
    }


# ── Plotting ─────────────────────────────────────────────────────────────

def make_plots(summary, budgets, output_dir):
    """Generate per-game exploitability comparison plots."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("  [matplotlib not available — skipping plots]")
        return

    sorted_budgets = sorted(budgets)

    fig, axes = plt.subplots(1, len(GAMES), figsize=(4.5 * len(GAMES), 4),
                             squeeze=False)
    axes = axes[0]

    colors = {"oos": "#4C72B0", "isgt_depth0": "#DD8452"}
    markers = {"oos": "o", "isgt_depth0": "s"}
    labels = {"oos": "OOS", "isgt_depth0": "ISGT (depth=0)"}

    for i, game_name in enumerate(GAMES):
        ax = axes[i]
        if game_name not in summary:
            ax.set_title(game_name)
            continue

        for algo in ["oos", "isgt_depth0"]:
            if algo not in summary[game_name]:
                continue
            points = summary[game_name][algo]
            xs = [p["sims_per_move"] for p in points]
            ys = [p["exploitability"] for p in points]
            ax.plot(xs, ys, marker=markers[algo], color=colors[algo],
                    label=labels[algo], linewidth=1.5, markersize=6)

        ax.set_xlabel("Sims / move")
        ax.set_ylabel("Exploitability")
        ax.set_title(game_name.replace("_", " ").title())
        ax.legend(fontsize=8)
        ax.set_xscale("log")
        ax.grid(True, alpha=0.3)

    fig.suptitle("OOS vs ISGT(depth=0) — Aggregate Exploitability",
                 fontsize=12, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.93])

    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, "oos_vs_isgt_depth0.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Plot saved: {path}")


# ── Main harness ─────────────────────────────────────────────────────────

def run_experiment(budgets=None, match_scale=1.0, workers=WORKERS, seed=42,
                   output_dir="results/oos_vs_isgt_depth0"):
    """Run OOS vs ISGT(depth=0) comparison."""
    if budgets is None:
        budgets = list(BUDGETS)

    os.makedirs(output_dir, exist_ok=True)

    # Build jobs
    jobs = []
    for game_name in GAMES:
        num_matches = max(10, int(GAME_MATCHES[game_name] * match_scale))
        for sims in sorted(budgets):
            jobs.append((_run_oos_job,
                         (game_name, sims, num_matches, seed)))
            jobs.append((_run_isgt_depth0_job,
                         (game_name, sims, num_matches, seed)))

    total = len(jobs)
    match_info = ", ".join(
        f"{g}: {max(10, int(GAME_MATCHES[g] * match_scale))}" for g in GAMES)

    print("=" * 70)
    print("  OOS vs ISGT(depth=0) Equivalence Check")
    print(f"  Games: {GAMES}")
    print(f"  Budgets: {sorted(budgets)}")
    print(f"  Matches: {{{match_info}}}")
    print(f"  Shared params: δ={DELTA}, ε={EPSILON}, γ={GAMMA}")
    print(f"  Jobs: {total}, Workers: {workers}")
    print("=" * 70)

    results = []
    completed = 0
    t_start = time.time()

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
                print(f"  [{completed:>3d}/{total}] "
                      f"{result['game']:15s} {result['algorithm']:15s} "
                      f"sims={result['sims_per_move']:>5d}  "
                      f"expl={result['exploitability']:.6f}  "
                      f"({result['elapsed_s']:.1f}s)  "
                      f"ETA {est_remaining/60:.1f}m")
            except Exception as exc:
                print(f"  [{completed:>3d}/{total}] FAILED: {exc}")

    # Sort results
    results.sort(key=lambda r: (r["game"], r["algorithm"], r["sims_per_move"]))

    # Save raw results
    raw_path = os.path.join(output_dir, "results.json")
    with open(raw_path, "w") as f:
        json.dump(results, f, indent=2)

    # Build summary: game → algo → [{sims, expl}]
    summary = {}
    for r in results:
        g, a = r["game"], r["algorithm"]
        summary.setdefault(g, {}).setdefault(a, []).append({
            "sims_per_move": r["sims_per_move"],
            "exploitability": r["exploitability"],
        })
    for g in summary:
        for a in summary[g]:
            summary[g][a].sort(key=lambda x: x["sims_per_move"])

    summary_path = os.path.join(output_dir, "summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    total_time = time.time() - t_start

    # Print summary table
    sorted_budgets = sorted(budgets)
    budget_cols = " ".join(f"{'sims='+str(b):>12s}" for b in sorted_budgets)
    header = f"{'Game':15s} {'Algorithm':15s} {budget_cols}"
    print(f"\n{'='*len(header)}")
    print(header)
    print(f"{'='*len(header)}")

    for game in GAMES:
        if game not in summary:
            continue
        for algo in ["oos", "isgt_depth0"]:
            if algo not in summary[game]:
                continue
            row = f"{game:15s} {algo:15s} "
            budget_map = {p["sims_per_move"]: p["exploitability"]
                          for p in summary[game][algo]}
            for b in sorted_budgets:
                if b in budget_map:
                    row += f" {budget_map[b]:>11.6f}"
                else:
                    row += f" {'---':>11s}"
            print(row)
        print()

    print(f"Done in {total_time/60:.1f} minutes.")
    print(f"Results saved to {output_dir}/")

    # Generate plots
    make_plots(summary, budgets, output_dir)

    return results


def main():
    args = sys.argv[1:]

    budgets = list(BUDGETS)
    match_scale = 1.0
    workers = WORKERS
    seed = 42

    i = 0
    while i < len(args):
        if args[i] == "--quick":
            budgets = [50, 200]
            match_scale = QUICK_MATCH_SCALE
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
        else:
            print(f"Unknown arg: {args[i]}")
            i += 1

    run_experiment(budgets=budgets, match_scale=match_scale,
                   workers=workers, seed=seed)


if __name__ == "__main__":
    main()
