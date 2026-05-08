"""Tables and plots for Experiment 15C — Low-Budget Proximity Probe.

Prints comprehensive tables for every metric and ablation dimension.
Optionally generates plots.

Usage:
    python -m experiments.plot_proximity_probe_015c
    python -m experiments.plot_proximity_probe_015c --results results/015c_proximity_probe_quick
    python -m experiments.plot_proximity_probe_015c --tables-only
    python -m experiments.plot_proximity_probe_015c --plots-only
"""

import argparse
import json
import os
import sys
from collections import defaultdict

import numpy as np


# ── Defaults ─────────────────────────────────────────────────────────────

DEFAULT_RESULTS = "results/015c_proximity_probe_full"
OUTPUT_DIR = "results/plots/015c_proximity_probe"

CONFIG_ORDER = [
    "chance_constant", "chance_sqrt", "chance_tbal",
    "full_constant", "full_sqrt", "full_tbal",
]

BUDGET_ORDER = [5, 10, 20, 50]

MODE_LABELS = {
    "chance_constant": "chance/const",
    "chance_sqrt":     "chance/sqrt",
    "chance_tbal":     "chance/tbal",
    "full_constant":   "full/const",
    "full_sqrt":       "full/sqrt",
    "full_tbal":       "full/tbal",
}


# ── Data loading ─────────────────────────────────────────────────────────

def load_results(results_dir):
    path = os.path.join(results_dir, "probe_results.json")
    with open(path) as f:
        data = json.load(f)
    # Index by (config, sims)
    indexed = {}
    for r in data:
        key = (r["config"], r["sims_per_move"])
        indexed[key] = r
    return data, indexed


# ── Table helpers ────────────────────────────────────────────────────────

def fmt(val, width=10, decimals=4):
    if val is None or val == 0:
        return "-".center(width)
    return f"{val:>{width}.{decimals}f}"


def fmt_int(val, width=6):
    return f"{val:>{width}d}"


def section(title):
    print(f"\n{'=' * 80}")
    print(f"  {title}")
    print(f"{'=' * 80}")


def subsection(title):
    print(f"\n  --- {title} ---")


# ── Markdown helpers ─────────────────────────────────────────────────────

def mfmt(val, decimals=4):
    """Format a value for markdown table cells."""
    if val is None or val == 0:
        return "-"
    return f"{val:.{decimals}f}"


def md_table(headers, rows, alignments=None):
    """Build a markdown table from headers and rows (list of lists).
    Returns list of lines.  alignments: list of 'l', 'r', 'c'."""
    if alignments is None:
        alignments = ["r"] * len(headers)
    lines = []
    lines.append("| " + " | ".join(headers) + " |")
    sep_parts = []
    for a in alignments:
        if a == "l":
            sep_parts.append(":---")
        elif a == "c":
            sep_parts.append(":---:")
        else:
            sep_parts.append("---:")
    lines.append("| " + " | ".join(sep_parts) + " |")
    for row in rows:
        lines.append("| " + " | ".join(str(c) for c in row) + " |")
    return lines


# ── Table 1: Exploitability Summary ─────────────────────────────────────

def table_exploitability(indexed, md):
    section("Table 1: Exploitability (config × budget)")
    header = f"  {'Config':<18s}" + "".join(f"{'sims='+str(s):>12s}" for s in BUDGET_ORDER)
    print(header)
    print("  " + "-" * (18 + 12 * len(BUDGET_ORDER)))
    for cfg in CONFIG_ORDER:
        row = f"  {MODE_LABELS[cfg]:<18s}"
        for sims in BUDGET_ORDER:
            r = indexed.get((cfg, sims))
            if r:
                row += f"{r['exploitability']:>12.4f}"
            else:
                row += f"{'—':>12s}"
        print(row)

    # Markdown
    md.append("## Table 1: Exploitability (config × budget)\n")
    md_headers = ["Config"] + [f"sims={s}" for s in BUDGET_ORDER]
    md_rows = []
    for cfg in CONFIG_ORDER:
        md_row = [MODE_LABELS[cfg]]
        for sims in BUDGET_ORDER:
            r = indexed.get((cfg, sims))
            md_row.append(mfmt(r["exploitability"]) if r else "—")
        md_rows.append(md_row)
    md.extend(md_table(md_headers, md_rows, ["l"] + ["r"] * len(BUDGET_ORDER)))
    md.append("")


# ── Table 2: Per-Node Neighborhood Regret Profile ───────────────────────

def compute_per_node(r, level_str):
    nbr = r.get("neighborhood_regret_by_level", {}).get(level_str, {})
    tch = r.get("neighborhood_touched_by_level", {}).get(level_str, {})
    delta = nbr.get("mean_abs_delta", 0)
    touched = tch.get("mean_touched", 0)
    n = nbr.get("n_samples", 0)
    pn = delta / touched if touched > 0 else 0
    return pn, n


def table_per_node_profile(indexed, md):
    section("Table 2: Per-Node Neighborhood Regret by Level")
    print("  (neighborhood_delta / touched_count — controls for fan-out)")

    md.append("## Table 2: Per-Node Neighborhood Regret by Level\n")
    md.append("neighborhood_delta / touched_count — controls for fan-out.\n")

    for cfg in CONFIG_ORDER:
        subsection(f"{MODE_LABELS[cfg]}")
        md.append(f"### {MODE_LABELS[cfg]}\n")

        # Determine max level across budgets
        max_lvl = 0
        for sims in BUDGET_ORDER:
            r = indexed.get((cfg, sims))
            if r:
                for k in r.get("neighborhood_regret_by_level", {}):
                    if k != "-1":
                        max_lvl = max(max_lvl, int(k))

        header = f"    {'Lvl':>4s}" + "".join(
            f"  {'sims='+str(s)+' pn':>10s} {'n':>6s}" for s in BUDGET_ORDER)
        print(header)
        print("    " + "-" * (4 + 18 * len(BUDGET_ORDER)))

        md_headers = ["Lvl"]
        for s in BUDGET_ORDER:
            md_headers.extend([f"s={s} pn", "n"])
        md_rows = []

        for lvl in range(max_lvl + 1):
            row = f"    {lvl:>4d}"
            md_row = [str(lvl)]
            for sims in BUDGET_ORDER:
                r = indexed.get((cfg, sims))
                if r:
                    pn, n = compute_per_node(r, str(lvl))
                    row += f"  {fmt(pn)} {fmt_int(n)}"
                    md_row.extend([mfmt(pn), str(n)])
                else:
                    row += f"  {'-':>10s} {'-':>6s}"
                    md_row.extend(["-", "-"])
            print(row)
            md_rows.append(md_row)

        md.extend(md_table(md_headers, md_rows, ["r"] * len(md_headers)))
        md.append("")


# ── Table 3: Bucketed Early/Late Per-Node Profile ───────────────────────

def table_bucketed_profile(indexed, md):
    section("Table 3: Bucketed Per-Node Profile (Early 20% vs Late 20% of sims)")
    print("  Primary test: if proximity matters when tree is unsaturated,")
    print("  early should show steeper gradient than late.")

    md.append("## Table 3: Bucketed Per-Node Profile (Early 20% vs Late 20%)\n")
    md.append("Primary test: if proximity matters when tree is unsaturated, "
              "early should show steeper gradient than late.\n")

    for cfg in CONFIG_ORDER:
        subsection(f"{MODE_LABELS[cfg]}")
        md.append(f"### {MODE_LABELS[cfg]}\n")

        for sims in BUDGET_ORDER:
            r = indexed.get((cfg, sims))
            if not r:
                continue
            buck = r.get("bucketed_neighborhood_regret", {})
            if not buck:
                continue

            print(f"    sims={sims}:")
            header = (f"      {'Lvl':>4s}"
                      f"  {'early_pn':>10s} {'n_e':>6s}"
                      f"  {'late_pn':>10s} {'n_l':>6s}"
                      f"  {'all_pn':>10s} {'n_a':>6s}"
                      f"  {'e-l diff':>10s}")
            print(header)
            print("      " + "-" * 70)

            md.append(f"**sims={sims}:**\n")
            md_headers = ["Lvl", "early_pn", "n_e", "late_pn", "n_l",
                          "all_pn", "n_a", "e−l diff"]
            md_rows = []

            for lvl_s in sorted(buck.keys(), key=lambda x: int(x)):
                lvl = int(lvl_s)
                if lvl < 0:
                    continue
                b = buck[lvl_s]
                e = b.get("early", {})
                l = b.get("late", {})
                a = b.get("all", {})
                ep = e.get("per_node", 0)
                lp = l.get("per_node", 0)
                ap = a.get("per_node", 0)
                en = e.get("n_samples", 0)
                ln = l.get("n_samples", 0)
                an = a.get("n_samples", 0)
                diff = ep - lp if (en > 5 and ln > 5) else None
                row = (f"      {lvl:>4d}"
                       f"  {fmt(ep)} {fmt_int(en)}"
                       f"  {fmt(lp)} {fmt_int(ln)}"
                       f"  {fmt(ap)} {fmt_int(an)}"
                       f"  {fmt(diff)}")
                print(row)
                md_rows.append([str(lvl), mfmt(ep), str(en), mfmt(lp),
                                str(ln), mfmt(ap), str(an), mfmt(diff)])

            md.extend(md_table(md_headers, md_rows, ["r"] * len(md_headers)))
            md.append("")


# ── Table 4: Importance Weight Statistics by Level ──────────────────────

def table_importance_weights(indexed, md):
    section("Table 4: Importance Weight (l) Statistics by Level")
    print("  l = δ·s1 + (1-δ)·s2 — the combined sampling probability.")
    print("  Full mode: l ≈ 0.5 (s1=1.0 dominates). Chance mode: l varies.")

    md.append("## Table 4: Importance Weight (l) Statistics by Level\n")
    md.append("l = δ·s1 + (1−δ)·s2 — the combined sampling probability. "
              "Full mode: l ≈ 0.5 (s1=1.0 dominates). Chance mode: l varies.\n")

    for cfg in CONFIG_ORDER:
        subsection(f"{MODE_LABELS[cfg]}")
        md.append(f"### {MODE_LABELS[cfg]}\n")

        for sims in BUDGET_ORDER:
            r = indexed.get((cfg, sims))
            if not r:
                continue
            iw = r.get("importance_weight_by_level", {})
            if not iw:
                continue

            print(f"    sims={sims}:")
            header = (f"      {'Lvl':>4s}"
                      f"  {'mean_l':>10s} {'std_l':>10s}"
                      f"  {'median_l':>10s}"
                      f"  {'min_l':>10s} {'max_l':>10s}"
                      f"  {'n':>6s}")
            print(header)
            print("      " + "-" * 68)

            md.append(f"**sims={sims}:**\n")
            md_headers = ["Lvl", "mean_l", "std_l", "median_l",
                          "min_l", "max_l", "n"]
            md_rows = []

            for lvl_s in sorted(iw.keys(), key=lambda x: int(x)):
                lvl = int(lvl_s)
                if lvl < 0:
                    continue
                w = iw[lvl_s]
                row = (f"      {lvl:>4d}"
                       f"  {fmt(w.get('mean_l', 0))}"
                       f"  {fmt(w.get('std_l', 0))}"
                       f"  {fmt(w.get('median_l', 0))}"
                       f"  {fmt(w.get('min_l', 0))}"
                       f"  {fmt(w.get('max_l', 0))}"
                       f"  {fmt_int(w.get('n_samples', 0))}")
                print(row)
                md_rows.append([
                    str(lvl),
                    mfmt(w.get("mean_l", 0)),
                    mfmt(w.get("std_l", 0)),
                    mfmt(w.get("median_l", 0)),
                    mfmt(w.get("min_l", 0)),
                    mfmt(w.get("max_l", 0)),
                    str(w.get("n_samples", 0)),
                ])

            md.extend(md_table(md_headers, md_rows, ["r"] * len(md_headers)))
            md.append("")


# ── Table 5: Active Regret by Level ─────────────────────────────────────

def table_active_regret(indexed, md):
    section("Table 5: Active Regret (|Δregret| at I₀) by Level")
    print("  Expected: cliff to zero at level 2+ (mechanical).")

    md.append("## Table 5: Active Regret (|Δregret| at I₀) by Level\n")
    md.append("Expected: cliff to zero at level 2+ (mechanical).\n")

    for cfg in CONFIG_ORDER:
        subsection(f"{MODE_LABELS[cfg]}")
        md.append(f"### {MODE_LABELS[cfg]}\n")

        header = f"    {'Lvl':>4s}" + "".join(
            f"  {'s='+str(s)+' mean':>10s} {'n':>6s}" for s in BUDGET_ORDER)
        print(header)
        print("    " + "-" * (4 + 18 * len(BUDGET_ORDER)))

        max_lvl = 0
        for sims in BUDGET_ORDER:
            r = indexed.get((cfg, sims))
            if r:
                for k in r.get("regret_by_level", {}):
                    if k != "-1":
                        max_lvl = max(max_lvl, int(k))

        md_headers = ["Lvl"]
        for s in BUDGET_ORDER:
            md_headers.extend([f"s={s} mean", "n"])
        md_rows = []

        for lvl in range(max_lvl + 1):
            row = f"    {lvl:>4d}"
            md_row = [str(lvl)]
            for sims in BUDGET_ORDER:
                r = indexed.get((cfg, sims))
                if r:
                    reg = r.get("regret_by_level", {}).get(str(lvl), {})
                    m = reg.get("mean_abs_delta", 0)
                    n = reg.get("n_samples", 0)
                    row += f"  {fmt(m)} {fmt_int(n)}"
                    md_row.extend([mfmt(m), str(n)])
                else:
                    row += f"  {'-':>10s} {'-':>6s}"
                    md_row.extend(["-", "-"])
            print(row)
            md_rows.append(md_row)

        md.extend(md_table(md_headers, md_rows, ["r"] * len(md_headers)))
        md.append("")


# ── Table 6: Cross-Config Per-Node Comparison (fixed budget) ────────────

def table_cross_config_per_node(indexed, md):
    section("Table 6: Cross-Config Per-Node Comparison at Each Budget")
    print("  Compare constant vs sqrt vs tbal within each mode.")
    print("  If decay matters, these should differ.")

    md.append("## Table 6: Cross-Config Per-Node Comparison at Each Budget\n")
    md.append("Compare constant vs sqrt vs tbal within each mode. "
              "If decay matters, these should differ.\n")

    for sims in BUDGET_ORDER:
        subsection(f"sims={sims}")
        md.append(f"### sims={sims}\n")

        # Determine max level
        max_lvl = 0
        for cfg in CONFIG_ORDER:
            r = indexed.get((cfg, sims))
            if r:
                for k in r.get("neighborhood_regret_by_level", {}):
                    if k != "-1":
                        max_lvl = max(max_lvl, int(k))

        header = f"    {'Lvl':>4s}" + "".join(
            f"  {MODE_LABELS[c]:>12s}" for c in CONFIG_ORDER)
        print(header)
        print("    " + "-" * (4 + 14 * len(CONFIG_ORDER)))

        md_headers = ["Lvl"] + [MODE_LABELS[c] for c in CONFIG_ORDER]
        md_rows = []

        for lvl in range(min(max_lvl + 1, 8)):
            row = f"    {lvl:>4d}"
            md_row = [str(lvl)]
            for cfg in CONFIG_ORDER:
                r = indexed.get((cfg, sims))
                if r:
                    pn, _ = compute_per_node(r, str(lvl))
                    row += f"  {pn:>12.4f}"
                    md_row.append(mfmt(pn))
                else:
                    row += f"  {'—':>12s}"
                    md_row.append("—")
            print(row)
            md_rows.append(md_row)

        md.extend(md_table(md_headers, md_rows, ["r"] * len(md_headers)))
        md.append("")


# ── Table 7: Importance Weight — Chance vs Full ─────────────────────────

def table_iw_mode_comparison(indexed, md):
    section("Table 7: Importance Weight Mean & Std — Chance vs Full")
    print("  Shows the variance structure that explains exploitability wins.")

    md.append("## Table 7: Importance Weight Mean & Std — Chance vs Full\n")
    md.append("Shows the variance structure that explains exploitability wins.\n")

    compare_cfgs = ["chance_constant", "chance_tbal",
                    "full_constant", "full_tbal"]

    for sims in BUDGET_ORDER:
        subsection(f"sims={sims}")
        md.append(f"### sims={sims}\n")

        header = (f"    {'Lvl':>4s}"
                  f"  {'ch/const':>10s} {'±std':>8s}"
                  f"  {'ch/tbal':>10s} {'±std':>8s}"
                  f"  {'fu/const':>10s} {'±std':>8s}"
                  f"  {'fu/tbal':>10s} {'±std':>8s}")
        print(header)
        print("    " + "-" * 84)

        max_lvl = 0
        for cfg in compare_cfgs:
            r = indexed.get((cfg, sims))
            if r:
                for k in r.get("importance_weight_by_level", {}):
                    if k != "-1":
                        max_lvl = max(max_lvl, int(k))

        md_headers = ["Lvl", "ch/const", "±std", "ch/tbal", "±std",
                      "fu/const", "±std", "fu/tbal", "±std"]
        md_rows = []

        for lvl in range(min(max_lvl + 1, 8)):
            row = f"    {lvl:>4d}"
            md_row = [str(lvl)]
            for cfg in compare_cfgs:
                r = indexed.get((cfg, sims))
                if r:
                    iw = r.get("importance_weight_by_level", {}).get(str(lvl), {})
                    m = iw.get("mean_l", 0)
                    s = iw.get("std_l", 0)
                    row += f"  {m:>10.4f} {s:>8.4f}"
                    md_row.extend([mfmt(m), mfmt(s)])
                else:
                    row += f"  {'—':>10s} {'—':>8s}"
                    md_row.extend(["—", "—"])
            print(row)
            md_rows.append(md_row)

        md.extend(md_table(md_headers, md_rows, ["r"] * len(md_headers)))
        md.append("")


# ── Table 8: Bucketed Early-Late Slope Summary ──────────────────────────

def table_slope_summary(indexed, md):
    section("Table 8: Early-vs-Late Slope Summary")
    print("  For each config × budget: Pearson r of (level, per_node)")
    print("  for early and late buckets. Negative r = proximity helps.")
    print("  Zero or positive = proximity doesn't help.")

    md.append("## Table 8: Early-vs-Late Slope Summary\n")
    md.append("For each config × budget: Pearson r of (level, per_node) "
              "for early and late buckets. Negative r = proximity helps. "
              "Zero or positive = proximity doesn't help.\n")

    header = f"  {'Config':<18s}" + "".join(
        f"  {'s='+str(s)+' e':>7s} {'l':>7s}" for s in BUDGET_ORDER)
    print(header)
    print("  " + "-" * (18 + 16 * len(BUDGET_ORDER)))

    md_headers = ["Config"]
    for s in BUDGET_ORDER:
        md_headers.extend([f"s={s} early", f"s={s} late"])
    md_rows = []

    for cfg in CONFIG_ORDER:
        row = f"  {MODE_LABELS[cfg]:<18s}"
        md_row = [MODE_LABELS[cfg]]
        for sims in BUDGET_ORDER:
            r = indexed.get((cfg, sims))
            if not r:
                row += f"  {'—':>7s} {'—':>7s}"
                md_row.extend(["—", "—"])
                continue
            buck = r.get("bucketed_neighborhood_regret", {})

            for bucket_name in ["early", "late"]:
                levels = []
                pn_vals = []
                for lvl_s in sorted(buck.keys(), key=lambda x: int(x)):
                    lvl = int(lvl_s)
                    if lvl < 0:
                        continue
                    b = buck[lvl_s].get(bucket_name, {})
                    n = b.get("n_samples", 0)
                    pn = b.get("per_node", 0)
                    if n >= 10:  # Only include levels with enough samples
                        levels.append(lvl)
                        pn_vals.append(pn)

                if len(levels) >= 3:
                    r_val = float(np.corrcoef(levels, pn_vals)[0, 1])
                    row += f"  {r_val:>7.3f}"
                    md_row.append(f"{r_val:.3f}")
                else:
                    row += f"  {'n/a':>7s}"
                    md_row.append("n/a")

        print(row)
        md_rows.append(md_row)

    md.extend(md_table(md_headers, md_rows,
                       ["l"] + ["r"] * (len(md_headers) - 1)))
    md.append("")


# ── Plots ────────────────────────────────────────────────────────────────

def make_plots(indexed, output_dir):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available, skipping plots")
        return

    os.makedirs(output_dir, exist_ok=True)

    COLORS = {
        "chance_constant": "#2ca02c",
        "chance_sqrt":     "#98df8a",
        "chance_tbal":     "#17becf",
        "full_constant":   "#9467bd",
        "full_sqrt":       "#c5b0d5",
        "full_tbal":       "#d62728",
    }

    MARKERS = {
        "chance_constant": "o",
        "chance_sqrt":     "s",
        "chance_tbal":     "^",
        "full_constant":   "D",
        "full_sqrt":       "v",
        "full_tbal":       "x",
    }

    # ── Fig 1: Per-node profiles by budget ───────────────────────────
    fig, axes = plt.subplots(1, 4, figsize=(20, 5), sharey=True)
    fig.suptitle("Per-Node Neighborhood Regret by IIG Level", fontsize=14)

    for ax, sims in zip(axes, BUDGET_ORDER):
        for cfg in CONFIG_ORDER:
            r = indexed.get((cfg, sims))
            if not r:
                continue
            levels = []
            pn_vals = []
            for lvl in range(8):
                pn, n = compute_per_node(r, str(lvl))
                if n >= 10:
                    levels.append(lvl)
                    pn_vals.append(pn)
            if levels:
                ax.plot(levels, pn_vals, color=COLORS[cfg],
                        marker=MARKERS[cfg], label=MODE_LABELS[cfg],
                        linewidth=1.5, markersize=5, alpha=0.8)

        ax.set_title(f"sims={sims}")
        ax.set_xlabel("IIG Level")
        if sims == BUDGET_ORDER[0]:
            ax.set_ylabel("Per-Node |Δregret|")
        ax.grid(True, alpha=0.3)

    axes[-1].legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "fig1_per_node_profiles.png"),
                dpi=150)
    print(f"  Saved fig1_per_node_profiles.png")
    plt.close(fig)

    # ── Fig 2: Early vs Late comparison ──────────────────────────────
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    fig.suptitle("Early (20%) vs Late (20%) Per-Node Profiles", fontsize=14)

    for col, sims in enumerate(BUDGET_ORDER):
        for row, (bucket, label) in enumerate([("early", "Early 20%"),
                                                 ("late", "Late 20%")]):
            ax = axes[row][col]
            for cfg in CONFIG_ORDER:
                r = indexed.get((cfg, sims))
                if not r:
                    continue
                buck = r.get("bucketed_neighborhood_regret", {})
                levels = []
                pn_vals = []
                for lvl in range(8):
                    b = buck.get(str(lvl), {}).get(bucket, {})
                    n = b.get("n_samples", 0)
                    pn = b.get("per_node", 0)
                    if n >= 10:
                        levels.append(lvl)
                        pn_vals.append(pn)
                if levels:
                    ax.plot(levels, pn_vals, color=COLORS[cfg],
                            marker=MARKERS[cfg], label=MODE_LABELS[cfg],
                            linewidth=1.5, markersize=5, alpha=0.8)

            ax.set_title(f"{label}, sims={sims}")
            if col == 0:
                ax.set_ylabel("Per-Node |Δregret|")
            if row == 1:
                ax.set_xlabel("IIG Level")
            ax.grid(True, alpha=0.3)

    axes[0][-1].legend(loc="upper left", fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "fig2_early_vs_late.png"), dpi=150)
    print(f"  Saved fig2_early_vs_late.png")
    plt.close(fig)

    # ── Fig 3: Importance weight distributions ───────────────────────
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    fig.suptitle("Importance Weight (l) — Mean ± Std by Level", fontsize=14)

    for ax, sims in zip(axes, BUDGET_ORDER):
        for cfg in CONFIG_ORDER:
            r = indexed.get((cfg, sims))
            if not r:
                continue
            iw = r.get("importance_weight_by_level", {})
            levels = []
            means = []
            stds = []
            for lvl in range(8):
                w = iw.get(str(lvl), {})
                n = w.get("n_samples", 0)
                if n >= 10:
                    levels.append(lvl)
                    means.append(w["mean_l"])
                    stds.append(w["std_l"])
            if levels:
                means = np.array(means)
                stds = np.array(stds)
                ax.plot(levels, means, color=COLORS[cfg],
                        marker=MARKERS[cfg], label=MODE_LABELS[cfg],
                        linewidth=1.5, markersize=5, alpha=0.8)
                ax.fill_between(levels, means - stds, means + stds,
                                color=COLORS[cfg], alpha=0.1)

        ax.set_title(f"sims={sims}")
        ax.set_xlabel("IIG Level")
        if sims == BUDGET_ORDER[0]:
            ax.set_ylabel("l = δ·s1 + (1-δ)·s2")
        ax.grid(True, alpha=0.3)

    axes[-1].legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "fig3_importance_weights.png"),
                dpi=150)
    print(f"  Saved fig3_importance_weights.png")
    plt.close(fig)

    # ── Fig 4: Chance vs Full per-node (separated y-axes) ────────────
    fig, axes = plt.subplots(2, 4, figsize=(20, 10), sharey="row")
    fig.suptitle("Per-Node Profile: Chance Mode (top) vs Full Mode (bottom)",
                 fontsize=14)

    chance_cfgs = [c for c in CONFIG_ORDER if c.startswith("chance")]
    full_cfgs = [c for c in CONFIG_ORDER if c.startswith("full")]

    for col, sims in enumerate(BUDGET_ORDER):
        for row, cfgs in enumerate([chance_cfgs, full_cfgs]):
            ax = axes[row][col]
            for cfg in cfgs:
                r = indexed.get((cfg, sims))
                if not r:
                    continue
                levels = []
                pn_vals = []
                for lvl in range(8):
                    pn, n = compute_per_node(r, str(lvl))
                    if n >= 10:
                        levels.append(lvl)
                        pn_vals.append(pn)
                if levels:
                    ax.plot(levels, pn_vals, color=COLORS[cfg],
                            marker=MARKERS[cfg], label=MODE_LABELS[cfg],
                            linewidth=1.5, markersize=5, alpha=0.8)

            mode_label = "Chance" if row == 0 else "Full"
            ax.set_title(f"{mode_label}, sims={sims}")
            if col == 0:
                ax.set_ylabel("Per-Node |Δregret|")
            if row == 1:
                ax.set_xlabel("IIG Level")
            ax.grid(True, alpha=0.3)
            ax.legend(loc="upper right", fontsize=7)

    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "fig4_chance_vs_full.png"), dpi=150)
    print(f"  Saved fig4_chance_vs_full.png")
    plt.close(fig)


# ── Main ─────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Tables and plots for 015c proximity probe results")
    parser.add_argument("--results", type=str, default=DEFAULT_RESULTS,
                        help="Results directory")
    parser.add_argument("--tables-only", action="store_true")
    parser.add_argument("--plots-only", action="store_true")
    parser.add_argument("--output", type=str, default=OUTPUT_DIR,
                        help="Plot output directory")
    args = parser.parse_args()

    data, indexed = load_results(args.results)
    print(f"Loaded {len(data)} entries from {args.results}")

    show_tables = not args.plots_only
    show_plots = not args.tables_only

    md = []  # Accumulate markdown lines

    if show_tables:
        md.append("# Experiment 15C — Low-Budget Proximity Probe Results\n")

        table_exploitability(indexed, md)
        table_per_node_profile(indexed, md)
        table_bucketed_profile(indexed, md)
        table_importance_weights(indexed, md)
        table_active_regret(indexed, md)
        table_cross_config_per_node(indexed, md)
        table_iw_mode_comparison(indexed, md)
        table_slope_summary(indexed, md)

        # Write markdown file
        md_path = os.path.join(args.results, "tables.md")
        with open(md_path, "w") as f:
            f.write("\n".join(md) + "\n")
        print(f"\nMarkdown tables written to {md_path}")

    if show_plots:
        print(f"\nGenerating plots → {args.output}")
        make_plots(indexed, args.output)

    print("\nDone.")


if __name__ == "__main__":
    main()
