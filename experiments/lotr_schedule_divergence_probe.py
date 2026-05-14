"""Empirical LOTR schedule divergence probe (backoff-distance framing).

This is a small pre-ablation diagnostic for LOTR schedule shapes. It builds a
shared corpus of random-play decision states, runs real LOTR searches from each
state under several schedules, and records where the schedule coin actually
diverges along the observable prefix — expressed as *backoff distance* from the
active state (backoff=0 means no divergence, backoff=k means diverged k steps
before active).

Outputs:
    results/lotr_schedule_divergence_probe/raw_step_summaries.jsonl
    results/lotr_schedule_divergence_probe/state_corpus.json
    results/lotr_schedule_divergence_probe/histograms.csv
    results/lotr_schedule_divergence_probe/prefix_context.csv
    results/lotr_schedule_divergence_probe/summary.md
    results/lotr_schedule_divergence_probe/plots/<game>_divergence_hist.png

Usage:
    python -m experiments.lotr_schedule_divergence_probe --pilot
    python -m experiments.lotr_schedule_divergence_probe
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import time
from collections import Counter, defaultdict

import numpy as np

from src.games import GAME_SPECS, load_game
from src.lotr import LOTRBot, uniform, early_linear, late_linear, step


DEFAULT_GAMES = ["kuhn_poker", "leduc_poker", "goofspiel", "liars_dice"]
DEFAULT_STATES_PER_GAME = 40
DEFAULT_SIMS_PER_STATE = 500
DEFAULT_NUM_SEEDS = 3
DEFAULT_SEED_OFFSET = 0
DEFAULT_RHO = 0.5
DEFAULT_OUTPUT_DIR = "results/lotr_schedule_probe_by_D"

EPSILON = 0.4
GAMMA = 0.01
NODE_TYPES = ["chance", "p0", "p1", "unknown"]


def _parse_games(value):
    if not value:
        return list(DEFAULT_GAMES)
    games = [part.strip() for part in value.split(",") if part.strip()]
    if games == ["all"]:
        return list(GAME_SPECS.keys())
    unknown = [game for game in games if game not in GAME_SPECS]
    if unknown:
        raise ValueError(f"Unknown games {unknown}. Available: {list(GAME_SPECS)}")
    return games


def _rho_label(rho):
    text = f"{rho:.3g}".replace(".", "")
    return f"r{text}"


def _schedule_specs(rho):
    rho_label = _rho_label(rho)
    return [
        (f"step_{rho_label}_d0", lambda: step(rho, depth=0)),
        (f"early_linear_{rho_label}", lambda: early_linear(rho)),
        (f"uniform_{rho_label}", lambda: uniform(rho)),
        (f"late_linear_{rho_label}", lambda: late_linear(rho)),
    ]


def _sample_chance_action(state, rng):
    outcomes = state.chance_outcomes()
    actions, probabilities = zip(*outcomes)
    index = int(rng.choice(len(actions), p=probabilities))
    return actions[index]


def _state_from_history(game, history):
    state = game.new_initial_state()
    for action in history:
        if state.is_terminal():
            raise ValueError("History continues past terminal state")
        state.apply_action(int(action))
    return state


def _collect_state_corpus(game_name, states_per_game, seeds):
    game = load_game(game_name)
    corpus = []
    seen = set()
    max_episodes_per_seed = max(20, states_per_game * 8)

    for seed_index, seed in enumerate(seeds):
        if len(corpus) >= states_per_game:
            break
        rng = np.random.RandomState(seed * 1_000_003 + 97)
        episode = 0
        while len(corpus) < states_per_game and episode < max_episodes_per_seed:
            state = game.new_initial_state()
            while not state.is_terminal() and len(corpus) < states_per_game:
                if state.is_chance_node():
                    state.apply_action(_sample_chance_action(state, rng))
                    continue

                history = tuple(int(action) for action in state.history())
                current_player = int(state.current_player())
                if history and history not in seen and current_player >= 0:
                    seen.add(history)
                    corpus.append({
                        "game": game_name,
                        "history": list(history),
                        "current_player": current_player,
                        "source_seed": int(seed),
                        "source_seed_index": int(seed_index),
                        "source_episode": int(episode),
                    })

                legal_actions = state.legal_actions()
                action = legal_actions[int(rng.randint(len(legal_actions)))]
                state.apply_action(action)
            episode += 1

    return corpus


def _new_aggregate():
    return {
        "n_states": 0,
        "n_iters": 0,
        "coin_diverged_count": 0,
        "coin_no_divergence_count": 0,
        "prefix_D_sum": 0.0,
        "prefix_D_min": None,
        "prefix_D_max": None,
        "backoff_kind_counts": Counter(),  # (backoff, kind) → count
        "D_backoff_kind_counts": Counter(),  # (D, backoff, kind) → count
        "D_iters": Counter(),  # D → total iterations from states with that D
        "prefix_position_counts": Counter(),
    }


def _merge_summary(aggregate, summary):
    n_iters = int(summary.get("n_iters", 0))
    no_divergence = int(summary.get("coin_no_divergence_count", 0))
    coin_diverged = max(0, n_iters - no_divergence)
    prefix_D = int(summary.get("prefix_D", 0))

    aggregate["n_states"] += 1
    aggregate["n_iters"] += n_iters
    aggregate["coin_diverged_count"] += coin_diverged
    aggregate["coin_no_divergence_count"] += no_divergence
    aggregate["prefix_D_sum"] += prefix_D
    aggregate["prefix_D_min"] = (
        prefix_D if aggregate["prefix_D_min"] is None
        else min(aggregate["prefix_D_min"], prefix_D)
    )
    aggregate["prefix_D_max"] = (
        prefix_D if aggregate["prefix_D_max"] is None
        else max(aggregate["prefix_D_max"], prefix_D)
    )
    aggregate["D_iters"][prefix_D] += n_iters

    for depth, kind_counts in summary.get("coin_diverge_depth_kind_counts", {}).items():
        depth_int = int(depth)
        backoff = prefix_D - depth_int  # distance from active state
        for kind, count in kind_counts.items():
            c = int(count)
            aggregate["backoff_kind_counts"][(backoff, kind)] += c
            aggregate["D_backoff_kind_counts"][(prefix_D, backoff, kind)] += c

    for depth, kind in summary.get("prefix_position_kinds", {}).items():
        aggregate["prefix_position_counts"][(int(depth), kind)] += 1


def _run_probe(games, schedules, states_per_game, sims_per_state,
               seeds, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    raw_path = os.path.join(output_dir, "raw_step_summaries.jsonl")
    corpus_path = os.path.join(output_dir, "state_corpus.json")

    corpus_by_game = {}
    aggregates = defaultdict(_new_aggregate)
    start_time = time.time()

    with open(raw_path, "w") as raw_handle:
        for game_index, game_name in enumerate(games):
            game = load_game(game_name)
            corpus = _collect_state_corpus(game_name, states_per_game, seeds)
            corpus_by_game[game_name] = corpus
            print(f"{game_name}: collected {len(corpus)} decision states")

            for schedule_index, (schedule_label, schedule_factory) in enumerate(schedules):
                print(f"  {schedule_label}: probing {len(corpus)} states")
                for state_index, state_record in enumerate(corpus):
                    state = _state_from_history(game, state_record["history"])
                    bot_seed = (
                        10_000_019
                        + game_index * 1_000_003
                        + schedule_index * 100_003
                        + state_index
                    )
                    bot = LOTRBot(
                        game,
                        int(state.current_player()),
                        num_simulations=sims_per_state,
                        epsilon=EPSILON,
                        gamma=GAMMA,
                        schedule=schedule_factory(),
                        seed=bot_seed,
                        tracking=True,
                    )
                    bot.step_with_policy(state)
                    summary = bot.get_tracking_summary()
                    _merge_summary(aggregates[(game_name, schedule_label)], summary)

                    record = {
                        "game": game_name,
                        "schedule": schedule_label,
                        "state_index": state_index,
                        "history": state_record["history"],
                        "current_player": int(state.current_player()),
                        "bot_seed": bot_seed,
                        "summary": {
                            "n_iters": summary.get("n_iters", 0),
                            "prefix_D": summary.get("prefix_D", 0),
                            "prefix_position_kinds": summary.get("prefix_position_kinds", {}),
                            "coin_diverge_fraction": summary.get("coin_diverge_fraction", 0.0),
                            "coin_no_divergence_count": summary.get("coin_no_divergence_count", 0),
                            "coin_diverge_depth_kind_counts": summary.get(
                                "coin_diverge_depth_kind_counts", {}
                            ),
                            "prefix_survival_fraction": summary.get(
                                "prefix_survival_fraction", 0.0
                            ),
                            "ess_per_sim": summary.get("ess_per_sim", 0.0),
                        },
                    }
                    raw_handle.write(json.dumps(record) + "\n")

    with open(corpus_path, "w") as handle:
        json.dump({
            "games": games,
            "states_per_game": states_per_game,
            "seeds": seeds,
            "corpus": corpus_by_game,
        }, handle, indent=2)

    elapsed_s = time.time() - start_time
    return aggregates, corpus_by_game, elapsed_s


def _max_backoff_for_game(aggregates, game_name):
    max_backoff_val = 0
    for (agg_game, _schedule), aggregate in aggregates.items():
        if agg_game != game_name:
            continue
        for backoff, _kind in aggregate["backoff_kind_counts"]:
            max_backoff_val = max(max_backoff_val, backoff)
    return max_backoff_val


def _availability(aggregate, backoff_k):
    """Number of iterations where backoff=k was a valid divergence position.

    backoff=k means d = D-k, so it's available when D >= k.
    """
    return sum(
        iters for D, iters in aggregate["D_iters"].items() if D >= backoff_k
    )


def _write_histogram_csv(path, aggregates):
    """Aggregate (over all D) histogram.  D='all' marker for clarity."""
    fieldnames = [
        "game", "schedule", "D", "backoff", "node_type", "count",
        "prob_unconditional", "availability", "prob_normalized",
    ]
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for (game_name, schedule_label), aggregate in sorted(aggregates.items()):
            n_iters = max(1, aggregate["n_iters"])
            max_backoff_val = max(
                [b for b, _kind in aggregate["backoff_kind_counts"]] or [0]
            )
            # backoff=0 → no divergence (reached active)
            writer.writerow({
                "game": game_name,
                "schedule": schedule_label,
                "D": "all",
                "backoff": 0,
                "node_type": "none",
                "count": aggregate["coin_no_divergence_count"],
                "prob_unconditional": aggregate["coin_no_divergence_count"] / n_iters,
                "availability": n_iters,
                "prob_normalized": aggregate["coin_no_divergence_count"] / n_iters,
            })
            # backoff=1..max → divergence at that distance from active
            for backoff in range(1, max_backoff_val + 1):
                avail = _availability(aggregate, backoff)
                for kind in NODE_TYPES:
                    count = aggregate["backoff_kind_counts"].get((backoff, kind), 0)
                    writer.writerow({
                        "game": game_name,
                        "schedule": schedule_label,
                        "D": "all",
                        "backoff": backoff,
                        "node_type": kind,
                        "count": count,
                        "prob_unconditional": count / n_iters,
                        "availability": avail,
                        "prob_normalized": count / avail if avail > 0 else 0.0,
                    })


def _write_histogram_by_D_csv(path, aggregates):
    """Per-D stratified histogram.

    For each (game, schedule, D), reports counts of divergences at each
    backoff k=1..D and node kind, normalized by iterations from states
    of that exact D.  Per-D conditional landing probability.
    """
    fieldnames = [
        "game", "schedule", "D", "backoff", "node_type", "count",
        "iters_at_D", "prob_at_D",
    ]
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for (game_name, schedule_label), aggregate in sorted(aggregates.items()):
            for D in sorted(aggregate["D_iters"]):
                iters_at_D = aggregate["D_iters"][D]
                if iters_at_D <= 0 or D <= 0:
                    continue
                # no-divergence count at this D = iters_at_D - sum of diverges at this D
                diverged_at_D = sum(
                    cnt for (D2, _b, _k), cnt in aggregate["D_backoff_kind_counts"].items()
                    if D2 == D
                )
                writer.writerow({
                    "game": game_name,
                    "schedule": schedule_label,
                    "D": D,
                    "backoff": 0,
                    "node_type": "none",
                    "count": iters_at_D - diverged_at_D,
                    "iters_at_D": iters_at_D,
                    "prob_at_D": (iters_at_D - diverged_at_D) / iters_at_D,
                })
                for backoff in range(1, D + 1):
                    for kind in NODE_TYPES:
                        count = aggregate["D_backoff_kind_counts"].get(
                            (D, backoff, kind), 0
                        )
                        writer.writerow({
                            "game": game_name,
                            "schedule": schedule_label,
                            "D": D,
                            "backoff": backoff,
                            "node_type": kind,
                            "count": count,
                            "iters_at_D": iters_at_D,
                            "prob_at_D": count / iters_at_D,
                        })


def _write_histogram_aggregate_by_k_csv(path, aggregates, schedules):
    """Aggregate-by-k histogram: pool over D≥k for a D-invariant landing view.

    For each (game, schedule, k):
      count(k)               = Σ_{D≥k} Σ_kind D_backoff_kind_counts[(D,k,kind)]
      diverged_at_D≥k        = Σ_{D≥k} (total diverged iters from states of that D)
      iters_at_D≥k           = Σ_{D≥k} D_iters[D]
      prob_emp(k)            = count(k) / diverged_at_D≥k
      prob_analytic(k)       = Σ_{D≥k} iters_at_D · w(k|D) / Σ_{D≥k} iters_at_D
    """
    schedule_factories = {label: factory for label, factory in schedules}
    fieldnames = [
        "game", "schedule", "k", "count",
        "diverged_at_D_ge_k", "iters_at_D_ge_k",
        "prob_emp", "prob_analytic",
    ]
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for (game_name, schedule_label), aggregate in sorted(aggregates.items()):
            max_k = max(
                [b for (_D, b, _k) in aggregate["D_backoff_kind_counts"]] or [0]
            )
            if max_k <= 0:
                continue
            sched = schedule_factories[schedule_label]()
            diverged_at_D = {}
            for (D, _b, _k), cnt in aggregate["D_backoff_kind_counts"].items():
                diverged_at_D[D] = diverged_at_D.get(D, 0) + cnt
            for k in range(1, max_k + 1):
                count_k = sum(
                    cnt for (D, b, _kind), cnt in aggregate["D_backoff_kind_counts"].items()
                    if b == k and D >= k
                )
                div_ge_k = sum(v for D, v in diverged_at_D.items() if D >= k)
                iters_ge_k = sum(v for D, v in aggregate["D_iters"].items() if D >= k)
                # analytic mixture
                num = 0.0
                den = 0.0
                for D, iters in aggregate["D_iters"].items():
                    if D < k or iters <= 0:
                        continue
                    raw = np.array(
                        [max(float(sched._weight_fn(j, D)), 0.0) for j in range(D)],
                        dtype=np.float64,
                    )
                    total = raw.sum()
                    w = raw / total if total > 0 else np.ones(D) / D
                    # depth d = D - k corresponds to backoff k
                    w_k = float(w[D - k])
                    num += iters * w_k
                    den += iters
                prob_analytic = num / den if den > 0 else 0.0
                writer.writerow({
                    "game": game_name,
                    "schedule": schedule_label,
                    "k": k,
                    "count": count_k,
                    "diverged_at_D_ge_k": div_ge_k,
                    "iters_at_D_ge_k": iters_ge_k,
                    "prob_emp": count_k / div_ge_k if div_ge_k > 0 else 0.0,
                    "prob_analytic": prob_analytic,
                })


def _write_prefix_context_csv(path, aggregates):
    fieldnames = [
        "game", "schedule", "depth", "node_type", "states_observed",
        "share_of_probed_states",
    ]
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for (game_name, schedule_label), aggregate in sorted(aggregates.items()):
            n_states = max(1, aggregate["n_states"])
            max_depth = max(
                [depth for depth, _kind in aggregate["prefix_position_counts"]] or [-1]
            )
            for depth in range(max_depth + 1):
                for kind in NODE_TYPES:
                    count = aggregate["prefix_position_counts"].get((depth, kind), 0)
                    writer.writerow({
                        "game": game_name,
                        "schedule": schedule_label,
                        "depth": depth,
                        "node_type": kind,
                        "states_observed": count,
                        "share_of_probed_states": count / n_states,
                    })


def _format_float(value):
    if value is None or not math.isfinite(float(value)):
        return "nan"
    return f"{float(value):.6g}"


def _write_summary(path, games, schedules, aggregates, config, elapsed_s):
    with open(path, "w") as handle:
        handle.write("# LOTR schedule divergence probe\n\n")
        handle.write("## Config\n\n")
        for key, value in config.items():
            handle.write(f"- {key}: {value}\n")
        handle.write(f"- elapsed_s: {_format_float(elapsed_s)}\n")

        handle.write("\n## Empirical Coin Divergence\n\n")
        handle.write(
            "| game | schedule | states | iters | coin diverged | no divergence | mean D | D range |\n"
        )
        handle.write("|---|---|---:|---:|---:|---:|---:|---|\n")
        for game_name in games:
            for schedule_label, _schedule_factory in schedules:
                aggregate = aggregates[(game_name, schedule_label)]
                n_iters = max(1, aggregate["n_iters"])
                mean_D = aggregate["prefix_D_sum"] / max(1, aggregate["n_states"])
                d_range = f"{aggregate['prefix_D_min']}..{aggregate['prefix_D_max']}"
                handle.write(
                    f"| {game_name} | {schedule_label} | "
                    f"{aggregate['n_states']} | {aggregate['n_iters']} | "
                    f"{_format_float(aggregate['coin_diverged_count'] / n_iters)} | "
                    f"{_format_float(aggregate['coin_no_divergence_count'] / n_iters)} | "
                    f"{_format_float(mean_D)} | {d_range} |\n"
                )

        handle.write("\n## Plots\n\n")
        for game_name in games:
            handle.write(
                f"- `plots/{game_name}_divergence_by_D.png`: "
                "PRIMARY — per-D P(backoff=k | diverged, D) with analytic w(k)/Σw(j) overlay.\n"
            )
            handle.write(
                f"- `plots/{game_name}_divergence_aggregate_by_k.png`: "
                "D-mixture P(k | diverged, D≥k) — D-invariant landing view of the schedule shape.\n"
            )
            handle.write(
                f"- `plots/{game_name}_divergence_normalized.png`: "
                "aggregate over D (slope reflects availability, not shape).\n"
            )
            handle.write(
                f"- `plots/{game_name}_divergence_raw.png`: raw P(backoff) including no-divergence mass.\n"
            )


def _make_plots(output_dir, games, schedules, aggregates):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib is not available; skipping PNG plots")
        return []

    plot_dir = os.path.join(output_dir, "plots")
    os.makedirs(plot_dir, exist_ok=True)
    colors = {
        "none": "#B0B0B0",
        "chance": "#4C78A8",
        "p0": "#F58518",
        "p1": "#54A24B",
    }
    kind_list = ["chance", "p0", "p1"]  # drop "unknown" (always 0)
    plot_paths = []

    for game_name in games:
        max_backoff_val = _max_backoff_for_game(aggregates, game_name)
        backoffs = list(range(max_backoff_val + 1))

        # --- Raw (unconditional) plot ---
        fig, axes = plt.subplots(
            1, len(schedules), figsize=(4.2 * len(schedules), 4.0),
            sharey=True, squeeze=False,
        )
        axes = axes[0]
        for axis, (schedule_label, _schedule_factory) in zip(axes, schedules):
            aggregate = aggregates[(game_name, schedule_label)]
            n_iters = max(1, aggregate["n_iters"])
            bottom = np.zeros(len(backoffs), dtype=np.float64)

            no_div_values = np.zeros(len(backoffs), dtype=np.float64)
            no_div_values[0] = aggregate["coin_no_divergence_count"] / n_iters
            axis.bar(
                backoffs, no_div_values, bottom=bottom, color=colors["none"],
                label="no divergence", width=0.75,
            )
            bottom += no_div_values

            for kind in kind_list:
                values = np.zeros(len(backoffs), dtype=np.float64)
                for i, b in enumerate(backoffs):
                    if b == 0:
                        continue
                    values[i] = aggregate["backoff_kind_counts"].get(
                        (b, kind), 0
                    ) / n_iters
                axis.bar(
                    backoffs, values, bottom=bottom, color=colors[kind],
                    label=kind, width=0.75,
                )
                bottom += values

            div_rate = aggregate["coin_diverged_count"] / n_iters
            no_div_rate = aggregate["coin_no_divergence_count"] / n_iters
            mean_D = aggregate["prefix_D_sum"] / max(1, aggregate["n_states"])
            axis.set_title(schedule_label, fontsize=9)
            axis.set_xlabel("backoff from active")
            axis.set_ylim(0.0, 1.05)
            axis.set_xticks(backoffs)
            axis.grid(axis="y", alpha=0.25)
            axis.text(
                0.02, 0.98,
                f"P(div)={div_rate:.3f}\nP(no div)={no_div_rate:.3f}\nmean D={mean_D:.2f}",
                transform=axis.transAxes,
                ha="left", va="top", fontsize=8,
                bbox={"facecolor": "white", "alpha": 0.75, "edgecolor": "none"},
            )
        axes[0].set_ylabel("P(backoff distance)")
        handles, labels = axes[-1].get_legend_handles_labels()
        seen_labels = set()
        unique_handles, unique_labels = [], []
        for h, l in zip(handles, labels):
            if l not in seen_labels:
                seen_labels.add(l)
                unique_handles.append(h)
                unique_labels.append(l)
        fig.legend(unique_handles, unique_labels, loc="lower center",
                   ncol=len(unique_labels))
        fig.suptitle(f"{game_name}: LOTR coin divergence (raw)")
        fig.tight_layout(rect=[0, 0.11, 1, 0.92])
        path = os.path.join(plot_dir, f"{game_name}_divergence_raw.png")
        fig.savefig(path, dpi=160)
        plt.close(fig)
        plot_paths.append(path)

        # --- Normalized by availability plot ---
        backoffs_norm = list(range(1, max_backoff_val + 1))
        if not backoffs_norm:
            continue
        fig, axes = plt.subplots(
            1, len(schedules), figsize=(4.2 * len(schedules), 4.0),
            sharey=True, squeeze=False,
        )
        axes = axes[0]
        for axis, (schedule_label, _schedule_factory) in zip(axes, schedules):
            aggregate = aggregates[(game_name, schedule_label)]
            bottom = np.zeros(len(backoffs_norm), dtype=np.float64)

            for kind in kind_list:
                values = np.zeros(len(backoffs_norm), dtype=np.float64)
                for i, b in enumerate(backoffs_norm):
                    avail = _availability(aggregate, b)
                    if avail > 0:
                        values[i] = aggregate["backoff_kind_counts"].get(
                            (b, kind), 0
                        ) / avail
                axis.bar(
                    backoffs_norm, values, bottom=bottom, color=colors[kind],
                    label=kind, width=0.75,
                )
                bottom += values

            mean_D = aggregate["prefix_D_sum"] / max(1, aggregate["n_states"])
            axis.set_title(schedule_label, fontsize=9)
            axis.set_xlabel("backoff from active")
            axis.set_ylim(bottom=0.0)
            axis.set_xticks(backoffs_norm)
            axis.grid(axis="y", alpha=0.25)
            axis.text(
                0.02, 0.98,
                f"mean D={mean_D:.2f}",
                transform=axis.transAxes,
                ha="left", va="top", fontsize=8,
                bbox={"facecolor": "white", "alpha": 0.75, "edgecolor": "none"},
            )
        axes[0].set_ylabel("P(diverge at k | k available)")
        handles, labels = axes[-1].get_legend_handles_labels()
        seen_labels = set()
        unique_handles, unique_labels = [], []
        for h, l in zip(handles, labels):
            if l not in seen_labels:
                seen_labels.add(l)
                unique_handles.append(h)
                unique_labels.append(l)
        fig.legend(unique_handles, unique_labels, loc="lower center",
                   ncol=len(unique_labels))
        fig.suptitle(
            f"{game_name}: LOTR divergence (normalized by availability)\n"
            f"aggregates over the game's D distribution — slope reflects availability, not schedule shape"
        )
        fig.tight_layout(rect=[0, 0.11, 1, 0.88])
        path = os.path.join(plot_dir, f"{game_name}_divergence_normalized.png")
        fig.savefig(path, dpi=160)
        plt.close(fig)
        plot_paths.append(path)

        # --- Per-D stratified plot (PRIMARY): P(backoff=k | diverged, D=d) ---
        Ds = sorted(d for d in set(
            D for (D, _b, _k) in aggregates[(game_name, schedules[0][0])]["D_backoff_kind_counts"]
        ) if d > 0)
        if not Ds:
            continue
        # Get analytic w(k)/Σw(j) per schedule by D
        n_schedules = len(schedules)
        fig, axes = plt.subplots(
            1, n_schedules, figsize=(4.2 * n_schedules, 4.0),
            sharey=True, squeeze=False,
        )
        axes = axes[0]
        cmap = plt.get_cmap("viridis")
        for axis, (schedule_label, schedule_factory) in zip(axes, schedules):
            aggregate = aggregates[(game_name, schedule_label)]
            sched = schedule_factory()
            n_groups = len(Ds)
            # bar width per D within a backoff group
            for d_idx, D in enumerate(Ds):
                iters_at_D = aggregate["D_iters"].get(D, 0)
                if iters_at_D <= 0:
                    continue
                ks = np.arange(1, D + 1)
                emp = np.zeros(D, dtype=np.float64)
                for i, k in enumerate(ks):
                    cnt = sum(
                        aggregate["D_backoff_kind_counts"].get((D, int(k), kind), 0)
                        for kind in kind_list
                    )
                    diverged_total = sum(
                        cnt2
                        for (D2, _b, _kk), cnt2 in aggregate["D_backoff_kind_counts"].items()
                        if D2 == D
                    )
                    emp[i] = (cnt / diverged_total) if diverged_total > 0 else 0.0
                # analytic w(k)/Σw(j) — recover w from schedule by reading raw weights
                # via tau_array: w(d) is recoverable from S(d) chain, but easier:
                # reconstruct w from the weight_fn directly.
                raw = np.array(
                    [max(float(sched._weight_fn(j, D)), 0.0) for j in range(D)],
                    dtype=np.float64,
                )
                if raw.sum() > 0:
                    w_norm = raw / raw.sum()
                else:
                    w_norm = np.ones(D) / D
                # w_norm is indexed by d=0..D-1; convert to backoff k=D-d
                analytic = w_norm[::-1]  # analytic[i] = w(k=i+1)

                color = cmap(d_idx / max(1, n_groups - 1))
                x = ks + (d_idx - (n_groups - 1) / 2) * (0.8 / n_groups)
                axis.bar(
                    x, emp, width=0.8 / n_groups, color=color,
                    label=f"D={D} (n={iters_at_D})", alpha=0.85,
                )
                # analytic reference: scatter markers at same x
                axis.scatter(
                    x, analytic, marker="_", s=80, color="black",
                    linewidths=1.5, zorder=5,
                )
            axis.set_title(schedule_label, fontsize=9)
            axis.set_xlabel("backoff k from active")
            axis.set_xticks(range(1, max(Ds) + 1))
            axis.grid(axis="y", alpha=0.25)
            axis.set_ylim(0.0, 1.05)
        axes[0].set_ylabel("P(backoff=k | diverged, D)")
        handles, labels = axes[-1].get_legend_handles_labels()
        seen_labels = set()
        unique_handles, unique_labels = [], []
        for h, l in zip(handles, labels):
            if l not in seen_labels:
                seen_labels.add(l)
                unique_handles.append(h)
                unique_labels.append(l)
        fig.legend(unique_handles, unique_labels, loc="lower center",
                   ncol=min(len(unique_labels), 6), fontsize=8)
        fig.suptitle(
            f"{game_name}: per-D conditional landing P(backoff=k | diverged, D)\n"
            f"black ticks = analytic w(k)/Σw(j); bars = empirical"
        )
        fig.tight_layout(rect=[0, 0.13, 1, 0.88])
        path = os.path.join(plot_dir, f"{game_name}_divergence_by_D.png")
        fig.savefig(path, dpi=160)
        plt.close(fig)
        plot_paths.append(path)

        # --- Aggregate-by-k plot: D-invariant landing view P(k | diverged, D>=k) ---
        if not Ds:
            continue
        max_k = max(Ds)
        fig, axes = plt.subplots(
            1, len(schedules), figsize=(4.2 * len(schedules), 4.0),
            sharey=True, squeeze=False,
        )
        axes = axes[0]
        for axis, (schedule_label, schedule_factory) in zip(axes, schedules):
            aggregate = aggregates[(game_name, schedule_label)]
            sched = schedule_factory()
            diverged_at_D = {}
            for (D, _b, _k), cnt in aggregate["D_backoff_kind_counts"].items():
                diverged_at_D[D] = diverged_at_D.get(D, 0) + cnt
            ks = np.arange(1, max_k + 1)
            emp = np.zeros(max_k, dtype=np.float64)
            analytic = np.zeros(max_k, dtype=np.float64)
            for i, k in enumerate(ks):
                count_k = sum(
                    cnt for (D, b, _kind), cnt in aggregate["D_backoff_kind_counts"].items()
                    if b == k and D >= k
                )
                div_ge_k = sum(v for D, v in diverged_at_D.items() if D >= k)
                emp[i] = (count_k / div_ge_k) if div_ge_k > 0 else 0.0
                num = 0.0
                den = 0.0
                for D, iters in aggregate["D_iters"].items():
                    if D < k or iters <= 0:
                        continue
                    raw = np.array(
                        [max(float(sched._weight_fn(j, D)), 0.0) for j in range(D)],
                        dtype=np.float64,
                    )
                    total = raw.sum()
                    w = raw / total if total > 0 else np.ones(D) / D
                    num += iters * float(w[D - int(k)])
                    den += iters
                analytic[i] = num / den if den > 0 else 0.0
            axis.bar(ks, emp, width=0.7, color="#4C78A8", label="empirical")
            axis.plot(ks, analytic, "o-", color="black", lw=1.2,
                      markersize=5, label="analytic")
            axis.set_title(schedule_label, fontsize=9)
            axis.set_xlabel("backoff k from active")
            axis.set_xticks(list(ks))
            axis.grid(axis="y", alpha=0.25)
            axis.set_ylim(0.0, 1.05)
        axes[0].set_ylabel("P(backoff=k | diverged, D≥k)")
        handles, labels = axes[-1].get_legend_handles_labels()
        seen_labels = set()
        unique_handles, unique_labels = [], []
        for h, l in zip(handles, labels):
            if l not in seen_labels:
                seen_labels.add(l)
                unique_handles.append(h)
                unique_labels.append(l)
        fig.legend(unique_handles, unique_labels, loc="lower center",
                   ncol=len(unique_labels), fontsize=8)
        fig.suptitle(
            f"{game_name}: aggregate-by-k landing  P(k | diverged, D≥k)\n"
            f"D-mixture pooled over states where k is reachable"
        )
        fig.tight_layout(rect=[0, 0.11, 1, 0.88])
        path = os.path.join(plot_dir, f"{game_name}_divergence_aggregate_by_k.png")
        fig.savefig(path, dpi=160)
        plt.close(fig)
        plot_paths.append(path)

    return plot_paths


def main():
    parser = argparse.ArgumentParser(
        description="Empirical LOTR schedule divergence probe")
    parser.add_argument("--games", type=str, default=None,
                        help="Comma-separated games, or all.")
    parser.add_argument("--rho", type=float, default=DEFAULT_RHO)
    parser.add_argument("--states-per-game", type=int,
                        default=DEFAULT_STATES_PER_GAME)
    parser.add_argument("--sims-per-state", type=int,
                        default=DEFAULT_SIMS_PER_STATE)
    parser.add_argument("--num-seeds", type=int, default=DEFAULT_NUM_SEEDS)
    parser.add_argument("--seed-offset", type=int, default=DEFAULT_SEED_OFFSET)
    parser.add_argument("--output", type=str, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--pilot", action="store_true",
                        help="Tiny smoke run: 5 states/game, 100 sims/state.")
    args = parser.parse_args()

    games = _parse_games(args.games)
    states_per_game = args.states_per_game
    sims_per_state = args.sims_per_state
    if args.pilot:
        states_per_game = 5
        sims_per_state = 100

    seeds = list(range(args.seed_offset, args.seed_offset + args.num_seeds))
    schedules = _schedule_specs(args.rho)

    config = {
        "games": games,
        "schedules": [label for label, _factory in schedules],
        "rho": args.rho,
        "states_per_game": states_per_game,
        "sims_per_state": sims_per_state,
        "num_seeds": args.num_seeds,
        "seed_offset": args.seed_offset,
        "seeds": seeds,
        "epsilon": EPSILON,
        "gamma": GAMMA,
    }

    print("LOTR schedule divergence probe")
    for key, value in config.items():
        print(f"  {key}: {value}")
    print(f"  output: {args.output}\n")

    aggregates, _corpus_by_game, elapsed_s = _run_probe(
        games, schedules, states_per_game, sims_per_state, seeds, args.output)

    _write_histogram_csv(os.path.join(args.output, "histograms.csv"), aggregates)
    _write_histogram_by_D_csv(
        os.path.join(args.output, "histograms_by_D.csv"), aggregates
    )
    _write_histogram_aggregate_by_k_csv(
        os.path.join(args.output, "histograms_aggregate_by_k.csv"),
        aggregates, schedules,
    )
    _write_prefix_context_csv(os.path.join(args.output, "prefix_context.csv"), aggregates)
    if not args.no_plots:
        _make_plots(args.output, games, schedules, aggregates)
    _write_summary(
        os.path.join(args.output, "summary.md"),
        games, schedules, aggregates, config, elapsed_s,
    )

    print(f"\nDone in {elapsed_s / 60:.2f} minutes")
    print(f"Reports written to {args.output}")


if __name__ == "__main__":
    main()