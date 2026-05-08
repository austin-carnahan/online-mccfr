"""Regret Gradient — IIG-level regret profile evaluation module.

Measures how per-node regret throughput varies with IIG distance from
the active infoset.  The core question: does proximity to the decision
point predict higher regret update magnitude per node?

Uses the aggregate multi-match evaluation method (Section 4.2 of Lisý
et al. 2015) with ISGT's regret tracking enabled.

Accepts an EvalConfig and produces:
  results.json  — raw per-job metrics + frozen config
  tables.md     — formatted summary tables
  plots/        — gradient curves (one per game)

Tables produced:
  1. Per-node regret profile  (mean, pn×l, std, touched, n)
  2. Touch rate               (touched / neighborhood_size)
  3. Raw per-walk total       (mean, std, n)
  4. Pearson correlation      (level vs per-node, weighted + unweighted)
  5. Importance weight        (mean_l, std_l, median_l, n)

Usage from an experiment script:
    from eval.config import EvalConfig
    from eval.regret_gradient import run
    from src.isgt import LevelUniform, LevelExponential

    config = EvalConfig(
        games=["leduc_poker", "goofspiel", "liars_dice"],
        algorithm="isgt",
        delta=0.7,
        epsilon=0.4,
        sims_per_move=100,
        num_matches=1000,
        decay_fn=[LevelUniform(), LevelExponential(0.7)],
        bias_mode="full",
        output_dir="results/regret_gradient_example",
    )

    results = run(config)

Usage from command line (with an experiment script):
    python -m experiments.my_regret_gradient_experiment

Regenerate tables/plots from saved results:
    from eval.regret_gradient import report
    report("results/regret_gradient_example")
"""

from __future__ import annotations

import json
import os
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
from scipy import stats

from eval.config import EvalConfig, JobSpec
from src.games import load_game
from src.iig import IIG
from src.isgt import ISGTBot, LevelWeightFn, LevelUniform
from src.oos import OOSBot
from eval.aggregate_metrics import aggregate_with_metrics


# ═════════════════════════════════════════════════════════════════════════
# Job execution
# ═════════════════════════════════════════════════════════════════════════

def _run_job(job: dict) -> dict:
    """Run a single (game, algorithm, params) job.

    Accepts and returns plain dicts for multiprocessing serialization.
    """
    game = load_game(job["game"])
    iig = IIG(game)
    counter = [0]

    # Reconstruct decay_fn from name (for pickling across processes)
    decay_fn = _resolve_decay_fn(job["decay_fn"])

    def bot_factory(player_id):
        s = job["seed"] + counter[0]
        counter[0] += 1

        if job["algorithm"] == "oos":
            return OOSBot(
                game, player_id,
                num_simulations=job["sims_per_move"],
                delta=job["delta"],
                epsilon=job["epsilon"],
                gamma=job["gamma"],
                seed=s,
            )
        else:
            bot = ISGTBot(
                game, player_id,
                num_simulations=job["sims_per_move"],
                epsilon=job["epsilon"],
                gamma=job["gamma"],
                level_weight_fn=decay_fn,
                bias_mode=job["bias_mode"],
                seed=s, iig=iig,
                delta=job["delta"],
                max_iig_depth=job["max_iig_depth"],
            )
            bot._regret_tracking = True
            return bot

    t0 = time.time()
    metrics = aggregate_with_metrics(
        game, bot_factory, iig,
        num_matches=job["num_matches"],
        seed=job["seed"],
        level_profile=False,
        num_sims=job["sims_per_move"],
    )
    elapsed = time.time() - t0

    result = {**job, "elapsed_s": round(elapsed, 1)}
    result["exploitability"] = metrics["exploitability"]
    for key in ("neighborhood_regret_by_level",
                "neighborhood_touched_by_level",
                "neighborhood_size_by_level",
                "importance_weight_by_level"):
        result[key] = metrics.get(key, {})

    return result


# ═════════════════════════════════════════════════════════════════════════
# Decay function serialization (for multiprocessing)
# ═════════════════════════════════════════════════════════════════════════

_DECAY_REGISTRY: dict[str, LevelWeightFn] = {}


def _register_decay_fns(config: EvalConfig) -> None:
    """Register decay functions by name for cross-process reconstruction."""
    from eval.config import _ensure_list
    for fn in _ensure_list(config.decay_fn):
        if fn is None:
            fn = LevelUniform()
        _DECAY_REGISTRY[fn.name()] = fn


def _resolve_decay_fn(name: str) -> LevelWeightFn:
    """Reconstruct a decay function from its name."""
    if name in _DECAY_REGISTRY:
        return _DECAY_REGISTRY[name]
    # Fallback: try to parse common patterns
    if name == "level_uniform":
        return LevelUniform()
    from src.isgt import LevelExponential, LevelPolynomial, LevelStep
    if name.startswith("level_exp("):
        alpha = float(name.split("=")[1].rstrip(")"))
        return LevelExponential(alpha)
    if name.startswith("level_poly("):
        p = float(name.split("=")[1].rstrip(")"))
        return LevelPolynomial(p)
    if name.startswith("level_step("):
        floor = float(name.split("=")[1].rstrip(")"))
        return LevelStep(floor)
    raise ValueError(f"Unknown decay function: {name}")


# ═════════════════════════════════════════════════════════════════════════
# Main entry point
# ═════════════════════════════════════════════════════════════════════════

def run(config: EvalConfig) -> list[dict]:
    """Run the regret gradient evaluation.

    Args:
        config: EvalConfig with experiment parameters.

    Returns:
        List of result dicts (one per job).
    """
    _register_decay_fns(config)
    all_jobs = config.expand()

    # Filter out non-ISGT algorithms (regret gradient requires IIG tracking)
    jobs = [j for j in all_jobs if j.algorithm == "isgt"]
    skipped = len(all_jobs) - len(jobs)
    if skipped:
        print(f"Note: skipped {skipped} non-ISGT jobs "
              f"(regret gradient requires ISGT regret tracking)")
    if not jobs:
        raise ValueError("No ISGT jobs in config — regret gradient "
                         "requires algorithm='isgt'")

    output_dir = config.output_dir or "results/regret_gradient"
    os.makedirs(output_dir, exist_ok=True)

    # Convert JobSpecs to plain dicts for multiprocessing
    job_dicts = [j.to_dict() for j in jobs]

    total = len(job_dicts)
    swept = config.swept_fields()
    print(f"Regret Gradient — {total} jobs")
    print(f"  Games: {sorted(set(j['game'] for j in job_dicts))}")
    print(f"  Swept: {swept or ['(none)']}")
    print(f"  Config: {config.to_dict()}")
    print(f"  Workers: {config.workers}")
    print()

    results = []
    errors = []
    completed = 0
    t_start = time.time()

    incremental_path = os.path.join(output_dir, "incremental_results.jsonl")

    with ProcessPoolExecutor(max_workers=config.workers) as pool:
        future_to_job = {}
        for jd in job_dicts:
            f = pool.submit(_run_job, jd)
            future_to_job[f] = jd

        for future in as_completed(future_to_job):
            completed += 1
            elapsed_total = time.time() - t_start
            est_remaining = (
                (elapsed_total / completed) * (total - completed)
            )

            try:
                result = future.result()
                results.append(result)

                with open(incremental_path, "a") as incr:
                    incr.write(json.dumps(result) + "\n")

                print(
                    f"  [{completed:>3d}/{total}] "
                    f"{result['game']:15s} {result['label']:20s} "
                    f"expl={result['exploitability']:.4f}  "
                    f"({result['elapsed_s']:.1f}s)  "
                    f"ETA {est_remaining / 60:.1f}m"
                )
            except Exception as exc:
                jd = future_to_job[future]
                print(
                    f"  [{completed:>3d}/{total}] FAILED "
                    f"{jd['game']} {jd['label']}: {exc}"
                )
                errors.append({
                    "job": jd,
                    "error": str(exc),
                    "traceback": traceback.format_exc(),
                })

    results.sort(key=lambda r: (r["game"], r["label"]))

    # Save results + config snapshot
    results_path = os.path.join(output_dir, "results.json")
    with open(results_path, "w") as f:
        json.dump({
            "config": config.to_dict(),
            "results": results,
        }, f, indent=2)

    elapsed_total = time.time() - t_start
    print(
        f"\nData collection done in {elapsed_total / 60:.1f}m. "
        f"{len(results)} completed, {len(errors)} errors."
    )

    if errors:
        err_path = os.path.join(output_dir, "errors.json")
        with open(err_path, "w") as f:
            json.dump(errors, f, indent=2)
        print(f"Errors: {err_path}")

    # Generate tables and plots
    print_tables(results)
    save_tables(results, output_dir)
    try:
        save_plots(results, output_dir)
    except Exception as exc:
        print(f"Plot generation failed: {exc}")

    print(f"\nResults: {results_path}")
    print(f"Tables:  {os.path.join(output_dir, 'tables.md')}")

    return results


def report(results_dir: str) -> None:
    """Regenerate tables and plots from saved results."""
    path = os.path.join(results_dir, "results.json")
    with open(path) as f:
        data = json.load(f)

    results = data["results"] if "results" in data else data
    print_tables(results)
    save_tables(results, results_dir)
    try:
        save_plots(results, results_dir)
    except Exception as exc:
        print(f"Plot generation failed: {exc}")
    print(f"\nTables regenerated at {os.path.join(results_dir, 'tables.md')}")


# ═════════════════════════════════════════════════════════════════════════
# Data extraction helpers
# ═════════════════════════════════════════════════════════════════════════

def _get_levels(r: dict, field: str = "neighborhood_regret_by_level") -> list[int]:
    data = r.get(field, {})
    return sorted(int(k) for k in data if int(k) >= 0)


def _per_node(r: dict, level: int) -> float:
    lvl_s = str(level)
    nbr = r.get("neighborhood_regret_by_level", {}).get(lvl_s, {})
    tch = r.get("neighborhood_touched_by_level", {}).get(lvl_s, {})
    delta = nbr.get("mean_abs_delta", 0)
    touched = tch.get("mean_touched", 0)
    return delta / touched if touched > 0 else 0.0


def _per_node_std(r: dict, level: int) -> float:
    lvl_s = str(level)
    nbr = r.get("neighborhood_regret_by_level", {}).get(lvl_s, {})
    tch = r.get("neighborhood_touched_by_level", {}).get(lvl_s, {})
    std_d = nbr.get("std_delta", 0)
    mean_t = tch.get("mean_touched", 0)
    if mean_t <= 0:
        return 0.0
    return std_d / mean_t


def _n_samples(r: dict, level: int) -> int:
    lvl_s = str(level)
    nbr = r.get("neighborhood_regret_by_level", {}).get(lvl_s, {})
    return nbr.get("n_samples", 0)


def _touch_rate(r: dict, level: int) -> float:
    lvl_s = str(level)
    tch = r.get("neighborhood_touched_by_level", {}).get(lvl_s, {})
    sz = r.get("neighborhood_size_by_level", {}).get(lvl_s, {})
    mean_t = tch.get("mean_touched", 0)
    mean_s = sz.get("mean_touched", 0)
    if mean_s <= 0:
        return 0.0
    return mean_t / mean_s


def _raw_total(r: dict, level: int) -> float:
    lvl_s = str(level)
    nbr = r.get("neighborhood_regret_by_level", {}).get(lvl_s, {})
    return nbr.get("mean_abs_delta", 0)


def _raw_total_std(r: dict, level: int) -> float:
    lvl_s = str(level)
    nbr = r.get("neighborhood_regret_by_level", {}).get(lvl_s, {})
    return nbr.get("std_delta", 0)


def _mean_l(r: dict, level: int) -> float:
    lvl_s = str(level)
    iw = r.get("importance_weight_by_level", {}).get(lvl_s, {})
    return iw.get("mean_l", 0)


def _per_node_l_normalized(r: dict, level: int) -> float:
    pn = _per_node(r, level)
    ml = _mean_l(r, level)
    return pn * ml if ml > 0 else 0.0


def _weighted_pearson(r: dict, min_n: int = 20):
    """Weighted Pearson r of (level, per_node), weighted by sqrt(n).

    Returns (r, p_value, n_levels) or (None, None, 0).
    """
    levels = _get_levels(r)
    xs, ys, ws = [], [], []
    for lvl in levels:
        n = _n_samples(r, lvl)
        if n < min_n:
            continue
        xs.append(lvl)
        ys.append(_per_node(r, lvl))
        ws.append(np.sqrt(n))

    if len(xs) < 3:
        return None, None, len(xs)

    xs = np.array(xs, dtype=float)
    ys = np.array(ys, dtype=float)
    ws = np.array(ws, dtype=float)
    ws /= ws.sum()

    mx = np.average(xs, weights=ws)
    my = np.average(ys, weights=ws)
    cov_xy = np.average((xs - mx) * (ys - my), weights=ws)
    var_x = np.average((xs - mx) ** 2, weights=ws)
    var_y = np.average((ys - my) ** 2, weights=ws)

    if var_x <= 0 or var_y <= 0:
        return 0.0, 1.0, len(xs)

    r_val = cov_xy / np.sqrt(var_x * var_y)

    n_eff = 1.0 / np.sum(ws ** 2)
    if n_eff <= 2:
        p_val = 1.0
    else:
        t_stat = r_val * np.sqrt((n_eff - 2) / (1 - r_val ** 2 + 1e-15))
        p_val = 2.0 * (1.0 - stats.t.cdf(abs(t_stat), df=max(1, n_eff - 2)))

    return float(r_val), float(p_val), len(xs)


# ═════════════════════════════════════════════════════════════════════════
# Markdown formatting
# ═════════════════════════════════════════════════════════════════════════

def _md_table(headers: list[str], rows: list[list[str]],
              alignments: list[str] | None = None) -> list[str]:
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


def _f(val, d: int = 4) -> str:
    if val is None or val == 0:
        return "-"
    return f"{val:.{d}f}"


# ═════════════════════════════════════════════════════════════════════════
# Table generators
# ═════════════════════════════════════════════════════════════════════════

def _unique_labels(results: list[dict]) -> list[str]:
    """Ordered unique labels across results."""
    seen = set()
    out = []
    for r in results:
        lbl = r.get("label", "")
        if lbl not in seen:
            seen.add(lbl)
            out.append(lbl)
    return out


def _table_per_node_profile(results: list[dict], md: list[str],
                            games: list[str]) -> None:
    md.append("## Table 1: Per-Node Regret Profile\n")
    md.append(
        "Per-node = mean(neighborhood_Δregret) / mean(touched_count). "
        "pn×l normalizes out importance weight amplification. "
        "Negative gradient (higher at level 0) = proximity signal.\n"
    )

    labels = _unique_labels(results)

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
        for lbl in labels:
            headers.extend([f"{lbl} pn", "pn×l", "std", "touched", "n"])
        rows = []

        for lvl in range(max_lvl + 1):
            row = [str(lvl)]
            for lbl in labels:
                r = next((x for x in game_results
                          if x.get("label") == lbl), None)
                if r:
                    pn = _per_node(r, lvl)
                    pnl = _per_node_l_normalized(r, lvl)
                    std_val = _per_node_std(r, lvl)
                    n = _n_samples(r, lvl)
                    tch = r.get("neighborhood_touched_by_level", {}
                                ).get(str(lvl), {}).get("mean_touched", 0)
                    row.extend([_f(pn), _f(pnl), _f(std_val),
                                _f(tch, 1), str(n)])
                else:
                    row.extend(["-"] * 5)
            rows.append(row)

        aligns = ["r"] + ["r", "r", "r", "r", "r"] * len(labels)
        md.extend(_md_table(headers, rows, aligns))
        md.append("")


def _table_touch_rate(results: list[dict], md: list[str],
                      games: list[str]) -> None:
    md.append("## Table 2: Touch Rate (touched / neighborhood_size)\n")
    md.append(
        "Fraction of materialized neighborhood that gets a nonzero "
        "regret update per walk. Higher = broader coverage.\n"
    )

    labels = _unique_labels(results)

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
        for lbl in labels:
            headers.extend([f"{lbl} rate", "touched", "nbr_size", "n"])
        rows = []

        for lvl in range(max_lvl + 1):
            row = [str(lvl)]
            for lbl in labels:
                r = next((x for x in game_results
                          if x.get("label") == lbl), None)
                if r:
                    rate = _touch_rate(r, lvl)
                    tch = r.get("neighborhood_touched_by_level", {}
                                ).get(str(lvl), {}).get("mean_touched", 0)
                    sz = r.get("neighborhood_size_by_level", {}
                               ).get(str(lvl), {}).get("mean_touched", 0)
                    n = _n_samples(r, lvl)
                    row.extend([_f(rate, 3), _f(tch, 1),
                                _f(sz, 1), str(n)])
                else:
                    row.extend(["-"] * 4)
            rows.append(row)

        aligns = ["r"] + ["r", "r", "r", "r"] * len(labels)
        md.extend(_md_table(headers, rows, aligns))
        md.append("")


def _table_raw_total(results: list[dict], md: list[str],
                     games: list[str]) -> None:
    md.append("## Table 3: Raw Per-Walk Total Neighborhood Regret\n")
    md.append(
        "Total |Δregret| across entire upstream neighborhood per "
        "walk. Combines intensity × breadth = net walk productivity.\n"
    )

    labels = _unique_labels(results)

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
        for lbl in labels:
            headers.extend([f"{lbl} total", "std", "n"])
        rows = []

        for lvl in range(max_lvl + 1):
            row = [str(lvl)]
            for lbl in labels:
                r = next((x for x in game_results
                          if x.get("label") == lbl), None)
                if r:
                    total = _raw_total(r, lvl)
                    std_val = _raw_total_std(r, lvl)
                    n = _n_samples(r, lvl)
                    row.extend([_f(total), _f(std_val), str(n)])
                else:
                    row.extend(["-"] * 3)
            rows.append(row)

        aligns = ["r"] + ["r", "r", "r"] * len(labels)
        md.extend(_md_table(headers, rows, aligns))
        md.append("")


def _table_pearson(results: list[dict], md: list[str],
                   games: list[str]) -> None:
    md.append("## Table 4: Weighted Pearson Correlation (level vs per-node)\n")
    md.append(
        "Weighted by √n at each level. Negative r = proximity "
        "predicts higher per-node regret (gradient exists). "
        "p-value from t-distribution with Kish effective df.\n"
    )

    labels = _unique_labels(results)

    headers = ["Game", "Label", "Pearson r", "p-value", "n_levels"]
    rows = []

    for game_name in games:
        for lbl in labels:
            r = next((x for x in results
                      if x["game"] == game_name
                      and x.get("label") == lbl), None)
            if not r:
                rows.append([game_name, lbl, "-", "-", "-"])
                continue
            r_val, p_val, n_lvls = _weighted_pearson(r)
            rows.append([
                game_name, lbl,
                _f(r_val, 3) if r_val is not None else "n/a",
                _f(p_val, 4) if p_val is not None else "n/a",
                str(n_lvls),
            ])

    md.extend(_md_table(headers, rows, ["l", "l", "r", "r", "r"]))
    md.append("")

    # Unweighted for reference
    md.append("### Unweighted Pearson r (for reference)\n")
    rows2 = []
    for game_name in games:
        for lbl in labels:
            r = next((x for x in results
                      if x["game"] == game_name
                      and x.get("label") == lbl), None)
            if not r:
                rows2.append([game_name, lbl, "-", "-", "-"])
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
                rows2.append([game_name, lbl, _f(r_val, 3),
                              _f(p_val, 4), str(len(xs))])
            else:
                rows2.append([game_name, lbl, "n/a", "n/a",
                              str(len(xs))])

    md.extend(_md_table(headers, rows2, ["l", "l", "r", "r", "r"]))
    md.append("")


def _table_importance_weights(results: list[dict], md: list[str],
                              games: list[str]) -> None:
    md.append("## Table 5: Importance Weight (l) Diagnostics\n")
    md.append("l = δ·s1 + (1−δ)·s2.\n")

    labels = _unique_labels(results)

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
        for lbl in labels:
            headers.extend([f"{lbl} mean_l", "std_l", "median_l", "n"])
        rows = []

        for lvl in range(max_lvl + 1):
            row = [str(lvl)]
            for lbl in labels:
                r = next((x for x in game_results
                          if x.get("label") == lbl), None)
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
                    row.extend(["-"] * 4)
            rows.append(row)

        aligns = ["r"] + ["r", "r", "r", "r"] * len(labels)
        md.extend(_md_table(headers, rows, aligns))
        md.append("")


# ═════════════════════════════════════════════════════════════════════════
# Table assembly
# ═════════════════════════════════════════════════════════════════════════

def _build_tables(results: list[dict]) -> list[str]:
    games = sorted(set(r["game"] for r in results))
    md: list[str] = []
    md.append("# Regret Gradient Results\n")

    # Config summary from first result
    if results:
        r0 = results[0]
        md.append(
            f"δ={r0.get('delta', '?')}, ε={r0.get('epsilon', '?')}, "
            f"γ={r0.get('gamma', '?')}, "
            f"sims={r0.get('sims_per_move', '?')}, "
            f"matches={r0.get('num_matches', '?')}\n"
        )

    _table_per_node_profile(results, md, games)
    _table_touch_rate(results, md, games)
    _table_raw_total(results, md, games)
    _table_pearson(results, md, games)
    _table_importance_weights(results, md, games)

    return md


def print_tables(results: list[dict]) -> None:
    md = _build_tables(results)
    print("\n" + "\n".join(md))


def save_tables(results: list[dict], output_dir: str) -> None:
    md = _build_tables(results)
    path = os.path.join(output_dir, "tables.md")
    with open(path, "w") as f:
        f.write("\n".join(md) + "\n")


# ═════════════════════════════════════════════════════════════════════════
# Plots
# ═════════════════════════════════════════════════════════════════════════

def save_plots(results: list[dict], output_dir: str) -> None:
    """Generate per-game gradient plots: per-node regret vs IIG level."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plots_dir = os.path.join(output_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    games = sorted(set(r["game"] for r in results))
    labels = _unique_labels(results)

    # Color cycle
    colors = plt.cm.tab10.colors

    for game_name in games:
        game_results = [r for r in results if r["game"] == game_name]
        if not game_results:
            continue

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Left: per-node regret vs level
        ax = axes[0]
        for i, lbl in enumerate(labels):
            r = next((x for x in game_results
                      if x.get("label") == lbl), None)
            if not r:
                continue
            lvls = _get_levels(r)
            if not lvls:
                continue
            xs = [lvl for lvl in lvls if _n_samples(r, lvl) >= 5]
            ys = [_per_node(r, lvl) for lvl in xs]
            ax.plot(xs, ys, "o-", color=colors[i % len(colors)],
                    label=lbl, markersize=5)

        ax.set_xlabel("IIG Level")
        ax.set_ylabel("Per-Node Regret")
        ax.set_title(f"{game_name} — Per-Node Regret by Level")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        # Right: per-node × l (normalized)
        ax = axes[1]
        for i, lbl in enumerate(labels):
            r = next((x for x in game_results
                      if x.get("label") == lbl), None)
            if not r:
                continue
            lvls = _get_levels(r)
            if not lvls:
                continue
            xs = [lvl for lvl in lvls if _n_samples(r, lvl) >= 5]
            ys = [_per_node_l_normalized(r, lvl) for lvl in xs]
            ax.plot(xs, ys, "o-", color=colors[i % len(colors)],
                    label=lbl, markersize=5)

        ax.set_xlabel("IIG Level")
        ax.set_ylabel("Per-Node × mean(l)")
        ax.set_title(f"{game_name} — l-Normalized Per-Node Regret")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        path = os.path.join(plots_dir, f"{game_name}_gradient.png")
        fig.savefig(path, dpi=150)
        plt.close(fig)
        print(f"  Plot: {path}")
