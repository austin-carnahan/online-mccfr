"""OOS vs ISGT(depth=0) — Match-count sensitivity.

Tests the hypothesis that ISGT's within-match chance diversity matters
more at low match counts, where OOS can't rely on match diversity to
cover the private-state space.

Usage:
    python -m experiments.oos_vs_isgt_matchcount
    python -m experiments.oos_vs_isgt_matchcount --quick
"""

import sys
import os
if not sys.stdout.line_buffering:
    sys.stdout.reconfigure(line_buffering=True)

import json
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

from src.games import load_game
from src.iig import IIG
from src.oos import OOSBot
from src.isgt import ISGTBot, LevelUniform
from eval.aggregate_exploitability import aggregate_exploitability


# ── Configuration ────────────────────────────────────────────────────────

GAMES = ["kuhn_poker", "leduc_poker", "goofspiel", "liars_dice"]

DELTA = 0.9
EPSILON = 0.6
GAMMA = 0.01

# Sim budgets where original experiment showed the clearest separation
BUDGETS = [200, 1000]

# Match counts to sweep — low counts stress-test chance coverage
MATCH_COUNTS = [10, 30, 100]

WORKERS = 4
OUTPUT_DIR = "results/oos_vs_isgt_matchcount"


# ── Job functions ────────────────────────────────────────────────────────

def _run_oos_job(game_name, sims_per_move, num_matches, seed):
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
        "game": game_name, "algorithm": "oos",
        "sims_per_move": sims_per_move, "num_matches": num_matches,
        "exploitability": expl, "elapsed_s": round(elapsed, 1),
    }


def _run_isgt_depth0_job(game_name, sims_per_move, num_matches, seed):
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
        "game": game_name, "algorithm": "isgt_depth0",
        "sims_per_move": sims_per_move, "num_matches": num_matches,
        "exploitability": expl, "elapsed_s": round(elapsed, 1),
    }


# ── Plotting ─────────────────────────────────────────────────────────────

def make_plots(summary, output_dir):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("  [matplotlib not available — skipping plots]")
        return

    budgets = sorted(BUDGETS)
    colors = {"oos": "#4C72B0", "isgt_depth0": "#DD8452"}
    markers = {"oos": "o", "isgt_depth0": "s"}
    labels = {"oos": "OOS", "isgt_depth0": "ISGT (depth=0)"}

    fig, axes = plt.subplots(len(GAMES), len(budgets),
                             figsize=(5 * len(budgets), 4 * len(GAMES)),
                             squeeze=False)

    for row, game_name in enumerate(GAMES):
        if game_name not in summary:
            continue
        for col, sims in enumerate(budgets):
            ax = axes[row][col]
            for algo in ["oos", "isgt_depth0"]:
                if algo not in summary[game_name]:
                    continue
                points = [p for p in summary[game_name][algo]
                          if p["sims_per_move"] == sims]
                points.sort(key=lambda p: p["num_matches"])
                if not points:
                    continue
                xs = [p["num_matches"] for p in points]
                ys = [p["exploitability"] for p in points]
                ax.plot(xs, ys, marker=markers[algo], color=colors[algo],
                        label=labels[algo], linewidth=1.5, markersize=6)

            ax.set_xlabel("Matches")
            ax.set_ylabel("Exploitability")
            title = f"{game_name.replace('_', ' ').title()} — {sims} sims/move"
            ax.set_title(title, fontsize=10)
            ax.legend(fontsize=8)
            ax.set_xscale("log")
            ax.grid(True, alpha=0.3)

    fig.suptitle("OOS vs ISGT(depth=0) — Match-Count Sensitivity",
                 fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, "matchcount_sensitivity.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Plot saved: {path}")


# ── Main ─────────────────────────────────────────────────────────────────

def run_experiment(budgets=None, match_counts=None, workers=WORKERS,
                   seed=42, output_dir=OUTPUT_DIR):
    if budgets is None:
        budgets = list(BUDGETS)
    if match_counts is None:
        match_counts = list(MATCH_COUNTS)

    os.makedirs(output_dir, exist_ok=True)

    jobs = []
    for game_name in GAMES:
        for nm in sorted(match_counts):
            for sims in sorted(budgets):
                jobs.append((_run_oos_job,
                             (game_name, sims, nm, seed)))
                jobs.append((_run_isgt_depth0_job,
                             (game_name, sims, nm, seed)))

    total = len(jobs)

    print("=" * 70)
    print("  OOS vs ISGT(depth=0) — Match-Count Sensitivity")
    print(f"  Games: {GAMES}")
    print(f"  Budgets: {sorted(budgets)}")
    print(f"  Match counts: {sorted(match_counts)}")
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
                      f"matches={result['num_matches']:>4d}  "
                      f"expl={result['exploitability']:.6f}  "
                      f"({result['elapsed_s']:.1f}s)  "
                      f"ETA {est_remaining/60:.1f}m")
            except Exception as exc:
                print(f"  [{completed:>3d}/{total}] FAILED: {exc}")

    results.sort(key=lambda r: (r["game"], r["algorithm"],
                                r["sims_per_move"], r["num_matches"]))

    # Save raw
    raw_path = os.path.join(output_dir, "results.json")
    with open(raw_path, "w") as f:
        json.dump(results, f, indent=2)

    # Build summary: game → algo → [{sims, matches, expl}]
    summary = {}
    for r in results:
        g, a = r["game"], r["algorithm"]
        summary.setdefault(g, {}).setdefault(a, []).append({
            "sims_per_move": r["sims_per_move"],
            "num_matches": r["num_matches"],
            "exploitability": r["exploitability"],
        })

    summary_path = os.path.join(output_dir, "summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    total_time = time.time() - t_start

    # Print summary table — one per budget level
    for sims in sorted(budgets):
        match_cols = " ".join(f"{'m='+str(m):>12s}" for m in sorted(match_counts))
        header = f"{'Game':15s} {'Algorithm':15s} {match_cols}"
        print(f"\n  sims_per_move = {sims}")
        print(f"  {'='*len(header)}")
        print(f"  {header}")
        print(f"  {'='*len(header)}")

        for game in GAMES:
            if game not in summary:
                continue
            for algo in ["oos", "isgt_depth0"]:
                if algo not in summary[game]:
                    continue
                row = f"  {game:15s} {algo:15s} "
                algo_pts = summary[game][algo]
                m_map = {p["num_matches"]: p["exploitability"]
                         for p in algo_pts if p["sims_per_move"] == sims}
                for m in sorted(match_counts):
                    if m in m_map:
                        row += f" {m_map[m]:>11.6f}"
                    else:
                        row += f" {'---':>11s}"
                print(row)
            print()

    # Print gap analysis (ISGT - OOS) for each cell
    print("  Gap analysis: ISGT(d=0) − OOS  (negative = ISGT better)")
    print("  " + "=" * 60)
    for sims in sorted(budgets):
        print(f"\n  sims = {sims}:")
        for game in GAMES:
            if game not in summary:
                continue
            oos_pts = {p["num_matches"]: p["exploitability"]
                       for p in summary[game].get("oos", [])
                       if p["sims_per_move"] == sims}
            isgt_pts = {p["num_matches"]: p["exploitability"]
                        for p in summary[game].get("isgt_depth0", [])
                        if p["sims_per_move"] == sims}
            gaps = []
            for m in sorted(match_counts):
                if m in oos_pts and m in isgt_pts:
                    gap = isgt_pts[m] - oos_pts[m]
                    gaps.append(f"m={m}: {gap:+.4f}")
            print(f"    {game:15s} {', '.join(gaps)}")

    print(f"\n  Done in {total_time/60:.1f} minutes.")
    print(f"  Results saved to {output_dir}/")

    make_plots(summary, output_dir)

    return results


def main():
    args = sys.argv[1:]
    budgets = list(BUDGETS)
    match_counts = list(MATCH_COUNTS)
    workers = WORKERS
    seed = 42

    i = 0
    while i < len(args):
        if args[i] == "--quick":
            budgets = [200]
            match_counts = [10, 30]
            i += 1
        elif args[i] == "--budgets" and i + 1 < len(args):
            budgets = [int(x) for x in args[i + 1].split(",")]
            i += 2
        elif args[i] == "--matches" and i + 1 < len(args):
            match_counts = [int(x) for x in args[i + 1].split(",")]
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

    run_experiment(budgets=budgets, match_counts=match_counts,
                   workers=workers, seed=seed)


if __name__ == "__main__":
    main()
