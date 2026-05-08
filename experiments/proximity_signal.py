"""Proximity Signal Test.

Demonstrates the IIG proximity gradient: closer levels produce more
regret throughput per node.  Provides clean diagnostics across games
and bias modes (full=anchor-split, chance).

Tables produced:
  1. Per-node proximity profile (mean, std, n, touched count)
  2. Touch rate (touched / neighborhood_size)
  3. Raw per-walk total neighborhood regret
  4. Weighted Pearson r + p-value summary
  5. Importance weight diagnostics (mean, std, median)

Fixed params: δ=0.7, ε=0.4, γ=0.01, LevelUniform, 100 sims/move.
Variables: {full, chance} × {leduc_poker, goofspiel, liars_dice} = 6 jobs.

Usage:
    python -m experiments.proximity_signal
    python -m experiments.proximity_signal --quick
    python -m experiments.proximity_signal --games leduc_poker
    python -m experiments.proximity_signal --matches 2000
"""

import sys
if not sys.stdout.line_buffering:
    sys.stdout.reconfigure(line_buffering=True)

import argparse
import json
import os
import time
import traceback
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
from scipy import stats

from src.games import load_game
from src.iig import IIG
from src.isgt import (ISGTBot, LevelUniform, LevelExponential,
                      LevelPolynomial)
from eval.aggregate_metrics import aggregate_with_metrics

DECAY_FUNCTIONS = {
    "level_uniform": LevelUniform,
    "level_exp50": lambda: LevelExponential(0.5),
    "level_exp75": lambda: LevelExponential(0.75),
    "level_poly2": lambda: LevelPolynomial(2.0),
}


# ── Configuration ────────────────────────────────────────────────────────

GAMES = ["leduc_poker", "goofspiel", "liars_dice"]
MODES = ["full", "chance"]

# High delta to maximize targeting signal, constant weight for max proximity bias
DELTA = 0.7
EPSILON = 0.4
GAMMA = 0.01
SIMS_PER_MOVE = 100

NUM_MATCHES = 1000
QUICK_MATCHES = 200
WORKERS = 6


# ── Single job ───────────────────────────────────────────────────────────

def _run_job(game_name, bias_mode, sims_per_move, num_matches, seed,
             decay_name="constant"):
    """Run one (game, mode) configuration."""
    game = load_game(game_name)
    iig = IIG(game)
    counter = [0]
    decay_cls = DECAY_FUNCTIONS[decay_name]

    def bot_factory(player_id):
        s = seed + counter[0]
        counter[0] += 1
        bot = ISGTBot(game, player_id, num_simulations=sims_per_move,
                      epsilon=EPSILON, gamma=GAMMA,
                      level_weight_fn=decay_cls(),
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
        "game": game_name,
        "bias_mode": bias_mode,
        "delta": DELTA,
        "epsilon": EPSILON,
        "sims_per_move": sims_per_move,
        "num_matches": num_matches,
        "exploitability": metrics["exploitability"],
        "neighborhood_regret_by_level": metrics.get(
            "neighborhood_regret_by_level", {}),
        "neighborhood_touched_by_level": metrics.get(
            "neighborhood_touched_by_level", {}),
        "neighborhood_size_by_level": metrics.get(
            "neighborhood_size_by_level", {}),
        "importance_weight_by_level": metrics.get(
            "importance_weight_by_level", {}),
        "decay": decay_name,
        "elapsed_s": round(elapsed, 1),
    }


# ── Main harness ─────────────────────────────────────────────────────────

def run_experiment(games=None, num_matches=NUM_MATCHES,
                   workers=WORKERS, seed=42, output_dir=None,
                   sims_per_move=None, decay_name="constant"):
    """Run the proximity signal experiment."""
    games = games or GAMES
    sims = sims_per_move or SIMS_PER_MOVE
    if output_dir is None:
        suffix = "quick" if num_matches <= QUICK_MATCHES else "full"
        output_dir = f"results/proximity_signal_{suffix}"
    os.makedirs(output_dir, exist_ok=True)

    jobs = []
    for game_name in games:
        for mode in MODES:
            jobs.append((game_name, mode, sims, num_matches, seed,
                         decay_name))

    total = len(jobs)
    print(f"Proximity Signal Test — {total} jobs")
    print(f"  Games: {games}")
    print(f"  Modes: {MODES}")
    print(f"  Fixed: δ={DELTA}, ε={EPSILON}, γ={GAMMA}, sims={sims}, decay={decay_name}")
    print(f"  Matches: {num_matches}, Workers: {workers}")
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
                      f"{result['game']:15s} {result['bias_mode']:8s} "
                      f"expl={result['exploitability']:.4f}  "
                      f"({result['elapsed_s']:.1f}s)  "
                      f"ETA {est_remaining/60:.1f}m")
            except Exception as exc:
                job_args = future_to_job[future]
                print(f"  [{completed:>2d}/{total}] FAILED {job_args[:2]}: {exc}")
                errors.append({"args": str(job_args), "error": str(exc),
                               "traceback": traceback.format_exc()})

    results.sort(key=lambda r: (r["game"], r["bias_mode"]))

    results_path = os.path.join(output_dir, "results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)

    elapsed_total = time.time() - t_start
    print(f"\nData collection done in {elapsed_total/60:.1f}m. "
          f"{len(results)} completed, {len(errors)} errors.")

    if errors:
        err_path = os.path.join(output_dir, "errors.json")
        with open(err_path, "w") as f:
            json.dump(errors, f, indent=2)
        print(f"Errors: {err_path}")

    # ── Generate tables ──────────────────────────────────────────────────
    print_tables(results)
    save_tables(results, output_dir)

    print(f"\nResults: {results_path}")
    print(f"Tables:  {os.path.join(output_dir, 'tables.md')}")

    return results


# ── Reporting helpers ────────────────────────────────────────────────────

def _index_results(results):
    """Index results by (game, mode)."""
    idx = {}
    for r in results:
        idx[(r["game"], r["bias_mode"])] = r
    return idx


def _get_levels(r, field="neighborhood_regret_by_level"):
    """Get sorted non-negative level keys from a result dict."""
    data = r.get(field, {})
    return sorted(int(k) for k in data if int(k) >= 0)


def _per_node(r, level):
    """Compute per-node regret for a result at a given level."""
    lvl_s = str(level)
    nbr = r.get("neighborhood_regret_by_level", {}).get(lvl_s, {})
    tch = r.get("neighborhood_touched_by_level", {}).get(lvl_s, {})
    delta = nbr.get("mean_abs_delta", 0)
    touched = tch.get("mean_touched", 0)
    return delta / touched if touched > 0 else 0.0


def _per_node_std(r, level):
    """Approximate std of per-node regret via propagation."""
    lvl_s = str(level)
    nbr = r.get("neighborhood_regret_by_level", {}).get(lvl_s, {})
    tch = r.get("neighborhood_touched_by_level", {}).get(lvl_s, {})
    mean_d = nbr.get("mean_abs_delta", 0)
    std_d = nbr.get("std_delta", 0)
    mean_t = tch.get("mean_touched", 0)
    if mean_t <= 0:
        return 0.0
    # delta/sigma propagation for ratio: sigma(d/t) ≈ std_d / mean_t
    return std_d / mean_t


def _n_samples(r, level):
    """Sample count for a level."""
    lvl_s = str(level)
    nbr = r.get("neighborhood_regret_by_level", {}).get(lvl_s, {})
    return nbr.get("n_samples", 0)


def _touch_rate(r, level):
    """Touch rate: mean_touched / mean_neighborhood_size."""
    lvl_s = str(level)
    tch = r.get("neighborhood_touched_by_level", {}).get(lvl_s, {})
    sz = r.get("neighborhood_size_by_level", {}).get(lvl_s, {})
    mean_t = tch.get("mean_touched", 0)
    mean_s = sz.get("mean_touched", 0)  # reuses _summarize_touched_by_level
    if mean_s <= 0:
        return 0.0
    return mean_t / mean_s


def _raw_total(r, level):
    """Raw per-walk total neighborhood regret."""
    lvl_s = str(level)
    nbr = r.get("neighborhood_regret_by_level", {}).get(lvl_s, {})
    return nbr.get("mean_abs_delta", 0)


def _raw_total_std(r, level):
    """Std of raw per-walk total neighborhood regret."""
    lvl_s = str(level)
    nbr = r.get("neighborhood_regret_by_level", {}).get(lvl_s, {})
    return nbr.get("std_delta", 0)


def _mean_l(r, level):
    """Mean importance weight l at a given level."""
    lvl_s = str(level)
    iw = r.get("importance_weight_by_level", {}).get(lvl_s, {})
    return iw.get("mean_l", 0)


def _per_node_l_normalized(r, level):
    """Per-node × mean_l: removes W=1/l amplification."""
    pn = _per_node(r, level)
    ml = _mean_l(r, level)
    return pn * ml if ml > 0 else 0.0


def _weighted_pearson(r, min_n=20):
    """Weighted Pearson r of (level, per_node), weighted by sqrt(n).

    Returns (r, p_value, n_levels) or (None, None, 0) if insufficient data.
    """
    levels = _get_levels(r)
    xs, ys, ws = [], [], []
    for lvl in levels:
        n = _n_samples(r, lvl)
        if n < min_n:
            continue
        pn = _per_node(r, lvl)
        xs.append(lvl)
        ys.append(pn)
        ws.append(np.sqrt(n))

    if len(xs) < 3:
        return None, None, len(xs)

    # Weighted Pearson: correlate sqrt(w)*x with sqrt(w)*y
    xs = np.array(xs, dtype=float)
    ys = np.array(ys, dtype=float)
    ws = np.array(ws, dtype=float)
    ws /= ws.sum()  # normalize weights

    mx = np.average(xs, weights=ws)
    my = np.average(ys, weights=ws)
    cov_xy = np.average((xs - mx) * (ys - my), weights=ws)
    var_x = np.average((xs - mx) ** 2, weights=ws)
    var_y = np.average((ys - my) ** 2, weights=ws)

    if var_x <= 0 or var_y <= 0:
        return 0.0, 1.0, len(xs)

    r_val = cov_xy / np.sqrt(var_x * var_y)

    # Approximate p-value using effective sample size
    n_eff = 1.0 / np.sum(ws ** 2)  # Kish's effective sample size
    if n_eff <= 2:
        p_val = 1.0
    else:
        t_stat = r_val * np.sqrt((n_eff - 2) / (1 - r_val ** 2 + 1e-15))
        p_val = 2.0 * (1.0 - stats.t.cdf(abs(t_stat), df=max(1, n_eff - 2)))

    return float(r_val), float(p_val), len(xs)


# ── Markdown helpers ─────────────────────────────────────────────────────

def _md_table(headers, rows, alignments=None):
    """Build markdown table lines."""
    if alignments is None:
        alignments = ["r"] * len(headers)
    lines = []
    lines.append("| " + " | ".join(headers) + " |")
    sep = []
    for a in alignments:
        if a == "l":
            sep.append(":---")
        elif a == "c":
            sep.append(":---:")
        else:
            sep.append("---:")
    lines.append("| " + " | ".join(sep) + " |")
    for row in rows:
        lines.append("| " + " | ".join(str(c) for c in row) + " |")
    return lines


def _f(val, d=4):
    """Format float for markdown."""
    if val is None or val == 0:
        return "-"
    return f"{val:.{d}f}"


# ── Table generators ─────────────────────────────────────────────────────

def _table1_per_node_profile(results, md, games):
    """Table 1: Per-Node Proximity Profile."""
    md.append("## Table 1: Per-Node Proximity Profile\n")
    md.append("Per-node = mean(neighborhood_Δregret) / mean(touched_count). "
              "pn×l normalizes out importance weight amplification. "
              "Negative gradient (higher at level 0) = proximity signal.\n")

    for game_name in games:
        game_results = [r for r in results if r["game"] == game_name]
        if not game_results:
            continue

        md.append(f"### {game_name}\n")

        # Determine max level across modes
        max_lvl = 0
        for r in game_results:
            lvls = _get_levels(r)
            if lvls:
                max_lvl = max(max_lvl, max(lvls))

        headers = ["Lvl"]
        for mode in MODES:
            headers.extend([f"{mode} pn", "pn×l", "std", "touched", "n"])
        rows = []

        for lvl in range(max_lvl + 1):
            row = [str(lvl)]
            for mode in MODES:
                r = next((x for x in game_results
                          if x["bias_mode"] == mode), None)
                if r:
                    pn = _per_node(r, lvl)
                    pnl = _per_node_l_normalized(r, lvl)
                    std = _per_node_std(r, lvl)
                    n = _n_samples(r, lvl)
                    tch = r.get("neighborhood_touched_by_level", {}
                                ).get(str(lvl), {}).get("mean_touched", 0)
                    row.extend([_f(pn), _f(pnl), _f(std), _f(tch, 1), str(n)])
                else:
                    row.extend(["-", "-", "-", "-", "-"])
            rows.append(row)

        md.extend(_md_table(headers, rows,
                            ["r"] + ["r", "r", "r", "r", "r"] * len(MODES)))
        md.append("")


def _table2_touch_rate(results, md, games):
    """Table 2: Touch Rate by Level."""
    md.append("## Table 2: Touch Rate (touched / neighborhood_size)\n")
    md.append("Fraction of materialized neighborhood that gets a nonzero "
              "regret update per walk. Higher = broader coverage.\n")

    for game_name in games:
        game_results = [r for r in results if r["game"] == game_name]
        if not game_results:
            continue

        md.append(f"### {game_name}\n")

        max_lvl = 0
        for r in game_results:
            lvls = _get_levels(r)
            if lvls:
                max_lvl = max(max_lvl, max(lvls))

        headers = ["Lvl"]
        for mode in MODES:
            headers.extend([f"{mode} rate", "touched", "nbr_size", "n"])
        rows = []

        for lvl in range(max_lvl + 1):
            row = [str(lvl)]
            for mode in MODES:
                r = next((x for x in game_results
                          if x["bias_mode"] == mode), None)
                if r:
                    rate = _touch_rate(r, lvl)
                    tch = r.get("neighborhood_touched_by_level", {}
                                ).get(str(lvl), {}).get("mean_touched", 0)
                    sz = r.get("neighborhood_size_by_level", {}
                               ).get(str(lvl), {}).get("mean_touched", 0)
                    n = _n_samples(r, lvl)
                    row.extend([_f(rate, 3), _f(tch, 1), _f(sz, 1), str(n)])
                else:
                    row.extend(["-", "-", "-", "-"])
            rows.append(row)

        md.extend(_md_table(headers, rows,
                            ["r"] + ["r", "r", "r", "r"] * len(MODES)))
        md.append("")


def _table3_raw_total(results, md, games):
    """Table 3: Raw Per-Walk Total Neighborhood Regret."""
    md.append("## Table 3: Raw Per-Walk Total Neighborhood Regret\n")
    md.append("Total |Δregret| across entire upstream neighborhood per "
              "walk. Combines intensity × breadth = net walk productivity.\n")

    for game_name in games:
        game_results = [r for r in results if r["game"] == game_name]
        if not game_results:
            continue

        md.append(f"### {game_name}\n")

        max_lvl = 0
        for r in game_results:
            lvls = _get_levels(r)
            if lvls:
                max_lvl = max(max_lvl, max(lvls))

        headers = ["Lvl"]
        for mode in MODES:
            headers.extend([f"{mode} total", "std", "n"])
        rows = []

        for lvl in range(max_lvl + 1):
            row = [str(lvl)]
            for mode in MODES:
                r = next((x for x in game_results
                          if x["bias_mode"] == mode), None)
                if r:
                    total = _raw_total(r, lvl)
                    std = _raw_total_std(r, lvl)
                    n = _n_samples(r, lvl)
                    row.extend([_f(total), _f(std), str(n)])
                else:
                    row.extend(["-", "-", "-"])
            rows.append(row)

        md.extend(_md_table(headers, rows,
                            ["r"] + ["r", "r", "r"] * len(MODES)))
        md.append("")


def _table4_pearson(results, md, games):
    """Table 4: Weighted Pearson r Summary."""
    md.append("## Table 4: Weighted Pearson Correlation (level vs per-node)\n")
    md.append("Weighted by √n at each level. Negative r = proximity "
              "predicts higher per-node regret (gradient exists). "
              "p-value from t-distribution with Kish effective df.\n")

    headers = ["Game", "Mode", "Pearson r", "p-value", "n_levels"]
    rows = []

    for game_name in games:
        for mode in MODES:
            r = next((x for x in results
                      if x["game"] == game_name
                      and x["bias_mode"] == mode), None)
            if not r:
                rows.append([game_name, mode, "-", "-", "-"])
                continue
            r_val, p_val, n_lvls = _weighted_pearson(r)
            rows.append([
                game_name, mode,
                _f(r_val, 3) if r_val is not None else "n/a",
                _f(p_val, 4) if p_val is not None else "n/a",
                str(n_lvls),
            ])

    md.extend(_md_table(headers, rows, ["l", "l", "r", "r", "r"]))
    md.append("")

    # Also add unweighted Pearson for comparison
    md.append("### Unweighted Pearson r (for reference)\n")
    headers2 = ["Game", "Mode", "Pearson r", "p-value", "n_levels"]
    rows2 = []

    for game_name in games:
        for mode in MODES:
            r = next((x for x in results
                      if x["game"] == game_name
                      and x["bias_mode"] == mode), None)
            if not r:
                rows2.append([game_name, mode, "-", "-", "-"])
                continue
            levels = _get_levels(r)
            xs, ys = [], []
            for lvl in levels:
                n = _n_samples(r, lvl)
                if n < 20:
                    continue
                xs.append(lvl)
                ys.append(_per_node(r, lvl))
            if len(xs) >= 3:
                r_val, p_val = stats.pearsonr(xs, ys)
                rows2.append([game_name, mode, _f(r_val, 3),
                              _f(p_val, 4), str(len(xs))])
            else:
                rows2.append([game_name, mode, "n/a", "n/a",
                              str(len(xs))])

    md.extend(_md_table(headers2, rows2, ["l", "l", "r", "r", "r"]))
    md.append("")


def _table5_importance_weights(results, md, games):
    """Table 5: Importance Weight Diagnostics."""
    md.append("## Table 5: Importance Weight (l) Diagnostics\n")
    md.append("l = δ·s1 + (1−δ)·s2. Verifies anchor-split mechanics: "
              "full mode l should be low (~0.03-0.05 post-fix), "
              "not flat at 0.5.\n")

    for game_name in games:
        game_results = [r for r in results if r["game"] == game_name]
        if not game_results:
            continue

        md.append(f"### {game_name}\n")

        max_lvl = 0
        for r in game_results:
            iw = r.get("importance_weight_by_level", {})
            for k in iw:
                if int(k) >= 0:
                    max_lvl = max(max_lvl, int(k))

        headers = ["Lvl"]
        for mode in MODES:
            headers.extend([f"{mode} mean_l", "std_l", "median_l", "n"])
        rows = []

        for lvl in range(max_lvl + 1):
            row = [str(lvl)]
            for mode in MODES:
                r = next((x for x in game_results
                          if x["bias_mode"] == mode), None)
                if r:
                    iw = r.get("importance_weight_by_level", {}
                               ).get(str(lvl), {})
                    row.extend([
                        _f(iw.get("mean_l", 0)),
                        _f(iw.get("std_l", 0)),
                        _f(iw.get("median_l", 0)),
                        str(iw.get("n_samples", 0)),
                    ])
                else:
                    row.extend(["-", "-", "-", "-"])
            rows.append(row)

        md.extend(_md_table(headers, rows,
                            ["r"] + ["r", "r", "r", "r"] * len(MODES)))
        md.append("")


# ── Print / Save ─────────────────────────────────────────────────────────

def _header_line(results):
    """Build header line from actual run config."""
    r0 = results[0] if results else {}
    sims = r0.get("sims_per_move", SIMS_PER_MOVE)
    decay = r0.get("decay", "constant")
    return f"δ={DELTA}, ε={EPSILON}, γ={GAMMA}, sims={sims}, decay={decay}"


def print_tables(results):
    """Print all tables to terminal."""
    games = sorted(set(r["game"] for r in results))
    md = []
    md.append("# Proximity Signal Test Results\n")
    md.append(_header_line(results) + "\n")

    _table1_per_node_profile(results, md, games)
    _table2_touch_rate(results, md, games)
    _table3_raw_total(results, md, games)
    _table4_pearson(results, md, games)
    _table5_importance_weights(results, md, games)

    print("\n" + "\n".join(md))


def save_tables(results, output_dir):
    """Save markdown tables to file."""
    games = sorted(set(r["game"] for r in results))
    md = []
    md.append("# Proximity Signal Test Results\n")
    md.append(_header_line(results) + "\n")

    _table1_per_node_profile(results, md, games)
    _table2_touch_rate(results, md, games)
    _table3_raw_total(results, md, games)
    _table4_pearson(results, md, games)
    _table5_importance_weights(results, md, games)

    path = os.path.join(output_dir, "tables.md")
    with open(path, "w") as f:
        f.write("\n".join(md) + "\n")


def load_and_report(results_dir):
    """Load saved results and regenerate tables."""
    path = os.path.join(results_dir, "results.json")
    with open(path) as f:
        results = json.load(f)
    print_tables(results)
    save_tables(results, results_dir)
    print(f"\nTables regenerated at {os.path.join(results_dir, 'tables.md')}")


# ── CLI ──────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Proximity Signal Test — IIG gradient across games/modes")
    parser.add_argument("--quick", action="store_true",
                        help=f"Use {QUICK_MATCHES} matches (faster)")
    parser.add_argument("--games", nargs="+",
                        help=f"Override games (default: {GAMES})")
    parser.add_argument("--matches", type=int,
                        help="Override match count")
    parser.add_argument("--workers", type=int, default=WORKERS,
                        help=f"Worker count (default: {WORKERS})")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed (default: 42)")
    parser.add_argument("--output", type=str,
                        help="Output directory override")
    parser.add_argument("--sims", type=int,
                        help=f"Sims per move (default: {SIMS_PER_MOVE})")
    parser.add_argument("--decay", type=str, default="constant",
                        choices=list(DECAY_FUNCTIONS.keys()),
                        help="Decay function (default: constant)")
    parser.add_argument("--report-only", type=str, metavar="DIR",
                        help="Regenerate tables from saved results")

    args = parser.parse_args()

    if args.report_only:
        load_and_report(args.report_only)
        return

    num_matches = NUM_MATCHES
    if args.quick:
        num_matches = QUICK_MATCHES
    if args.matches:
        num_matches = args.matches

    run_experiment(
        games=args.games,
        num_matches=num_matches,
        workers=args.workers,
        seed=args.seed,
        output_dir=args.output,
        sims_per_move=args.sims,
        decay_name=args.decay,
    )


if __name__ == "__main__":
    main()
