"""Head-to-head match evaluation module.

Runs round-robin tournaments between bot configurations, alternating
positions. Reports win rates (for symmetric games) or mean payoff
(for asymmetric games like poker) with 95% confidence intervals.

Reproduces the methodology from Lisý, Lanctot & Bowling 2015, §4.4.3.

Core functions:
    play_match(game, bot_factories, seed) -> dict
        Play a single match, return payoffs.

    round_robin(game, bot_configs, num_matches, sims_per_move, seed)
        All-pairs tournament at a given sims budget.

Config-driven entry point:
    run(config: HeadsUpConfig) -> list[dict]
        Accepts a HeadsUpConfig, runs all matchups, writes results
        and tables.

Usage:
    from eval.headsup import HeadsUpConfig, run

    config = HeadsUpConfig(
        games=["kuhn_poker", "leduc_poker"],
        bots=[
            BotSpec(algorithm="oos", label="OOS", epsilon=0.4, delta=0.9),
            BotSpec(algorithm="retro", label="Retro", epsilon=0.4),
            BotSpec(algorithm="random", label="RND"),
        ],
        sims_per_move=[100, 500, 1000],
        num_matches=500,
        output_dir="results/headsup",
    )
    run(config)
"""

from __future__ import annotations

import json
import math
import os
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pyspiel

from open_spiel.python import policy as policy_lib
from open_spiel.python.bots.policy import PolicyBot

from src.games import load_game


# ═════════════════════════════════════════════════════════════════════════
# Bot specification
# ═════════════════════════════════════════════════════════════════════════

@dataclass
class BotSpec:
    """Specification for a bot in head-to-head evaluation.

    algorithm: "oos", "retro", "isgt", or "random"
    label: display name for tables (auto-generated if empty)
    """
    algorithm: str = "oos"
    label: str = ""
    delta: float = 0.9
    epsilon: float = 0.4
    gamma: float = 0.01
    decay_fn: Any = None   # LevelWeightFn for retro/isgt
    targeting: str = "IST"  # IST or PST, for OOS

    def __post_init__(self):
        if not self.label:
            self.label = self.algorithm

    def to_dict(self) -> dict:
        from src.isgt import LevelWeightFn
        d = {
            "algorithm": self.algorithm,
            "label": self.label,
            "delta": self.delta,
            "epsilon": self.epsilon,
            "gamma": self.gamma,
            "targeting": self.targeting,
        }
        if self.decay_fn is not None and isinstance(self.decay_fn, LevelWeightFn):
            d["decay_fn"] = self.decay_fn.name()
        elif self.decay_fn is not None:
            d["decay_fn"] = str(self.decay_fn)
        else:
            d["decay_fn"] = None
        return d


# ═════════════════════════════════════════════════════════════════════════
# Head-to-head config
# ═════════════════════════════════════════════════════════════════════════

@dataclass
class HeadsUpConfig:
    """Configuration for head-to-head evaluation."""

    games: str | list[str] = "kuhn_poker"
    bots: list[BotSpec] = field(default_factory=list)
    sims_per_move: int | list[int] = 500
    num_matches: int = 500
    seed: int = 42
    workers: int = 4
    output_dir: str = "results/headsup"

    def to_dict(self) -> dict:
        sims = self.sims_per_move
        if not isinstance(sims, list):
            sims = [sims]
        games = self.games
        if not isinstance(games, list):
            games = [games]
        return {
            "games": games,
            "bots": [b.to_dict() for b in self.bots],
            "sims_per_move": sims,
            "num_matches": self.num_matches,
            "seed": self.seed,
            "workers": self.workers,
            "output_dir": self.output_dir,
        }


# ═════════════════════════════════════════════════════════════════════════
# Bot construction
# ═════════════════════════════════════════════════════════════════════════

def _make_bot(game, player_id, spec: dict, sims_per_move: int, seed: int):
    """Construct a bot from a serialized BotSpec dict."""
    algo = spec["algorithm"]

    if algo == "random":
        rng = np.random.RandomState(seed)
        return PolicyBot(player_id, rng, policy_lib.UniformRandomPolicy(game))

    if algo == "oos":
        from src.oos import OOSBot
        return OOSBot(
            game, player_id,
            num_simulations=sims_per_move,
            delta=spec["delta"],
            epsilon=spec["epsilon"],
            gamma=spec["gamma"],
            targeting=spec.get("targeting", "IST"),
            seed=seed,
        )

    if algo == "retro":
        from src.retrospective import RetroBot
        from src.isgt import LevelUniform
        decay_fn = _resolve_decay_fn(spec.get("decay_fn")) or LevelUniform()
        return RetroBot(
            game, player_id,
            num_simulations=sims_per_move,
            epsilon=spec["epsilon"],
            gamma=spec["gamma"],
            decay_fn=decay_fn,
            seed=seed,
        )

    if algo == "depth_delta":
        from src.depth_delta import DepthDeltaBot, resolve_schedule
        sched = resolve_schedule(spec.get("schedule", "linear"))
        return DepthDeltaBot(
            game, player_id,
            num_simulations=sims_per_move,
            epsilon=spec["epsilon"],
            gamma=spec["gamma"],
            schedule=sched,
            seed=seed,
        )

    if algo == "isgt":
        from src.isgt import ISGTBot, LevelUniform
        from src.iig import IIG
        iig = IIG(game)
        decay_fn = _resolve_decay_fn(spec.get("decay_fn")) or LevelUniform()
        return ISGTBot(
            game, player_id,
            num_simulations=sims_per_move,
            epsilon=spec["epsilon"],
            gamma=spec["gamma"],
            level_weight_fn=decay_fn,
            bias_mode=spec.get("bias_mode", "full"),
            seed=seed,
            iig=iig,
            delta=spec.get("delta", 0.9),
            max_iig_depth=spec.get("max_iig_depth"),
        )

    raise ValueError(f"Unknown algorithm: {algo}")


def _resolve_decay_fn(name):
    """Reconstruct a LevelWeightFn from its string name."""
    if name is None:
        return None
    from src.isgt import LevelUniform, LevelExponential, LevelStep, LevelPolynomial
    if name == "level_uniform":
        return LevelUniform()
    if name.startswith("level_exp("):
        alpha = float(name.split("=")[1].rstrip(")"))
        return LevelExponential(alpha)
    if name.startswith("level_step("):
        floor = float(name.split("=")[1].rstrip(")"))
        return LevelStep(floor)
    if name.startswith("level_poly("):
        p = float(name.split("=")[1].rstrip(")"))
        return LevelPolynomial(p)
    return None


# ═════════════════════════════════════════════════════════════════════════
# Match play
# ═════════════════════════════════════════════════════════════════════════

def play_match(game, bot_p0, bot_p1, rng):
    """Play a single match, return payoffs [u0, u1].

    Handles chance nodes via the provided rng.
    """
    state = game.new_initial_state()
    bots = [bot_p0, bot_p1]

    for bot in bots:
        bot.restart()

    while not state.is_terminal():
        if state.is_chance_node():
            outcomes, probs = zip(*state.chance_outcomes())
            action = rng.choice(outcomes, p=probs)
            state.apply_action(action)
        else:
            player = state.current_player()
            action = bots[player].step(state)
            for p, bot in enumerate(bots):
                if p != player:
                    bot.inform_action(state, player, action)
            state.apply_action(action)

    return [state.returns()[p] for p in range(2)]


def _play_matchup(job: dict) -> dict:
    """Run a full matchup between two bots, alternating positions.

    job keys:
        game, row_bot, col_bot, sims_per_move, num_matches, seed

    Returns dict with:
        row_label, col_label, game, sims_per_move,
        mean_payoff, win_rate, ci95, payoff_ci95, num_matches,
        payoffs (list of per-match row player payoffs)
    """
    game = load_game(job["game"])
    rng = np.random.RandomState(job["seed"])

    row_spec = job["row_bot"]
    col_spec = job["col_bot"]
    sims = job["sims_per_move"]
    n = job["num_matches"]

    payoffs = []
    wins = 0
    draws = 0
    bot_counter = [0]

    def next_seed():
        s = job["seed"] + bot_counter[0]
        bot_counter[0] += 1
        return s

    for match_idx in range(n):
        # Alternate positions each match
        if match_idx % 2 == 0:
            p0_spec, p1_spec = row_spec, col_spec
        else:
            p0_spec, p1_spec = col_spec, row_spec

        bot0 = _make_bot(game, 0, p0_spec, sims, next_seed())
        bot1 = _make_bot(game, 1, p1_spec, sims, next_seed())

        returns = play_match(game, bot0, bot1, rng)

        # Payoff from row player's perspective
        if match_idx % 2 == 0:
            row_payoff = returns[0]
        else:
            row_payoff = returns[1]

        payoffs.append(row_payoff)

        if row_payoff > 0:
            wins += 1
        elif row_payoff == 0:
            draws += 1

    payoffs_arr = np.array(payoffs)
    mean_payoff = float(payoffs_arr.mean())
    std_payoff = float(payoffs_arr.std(ddof=1)) if n > 1 else 0.0
    payoff_ci95 = 1.96 * std_payoff / math.sqrt(n)

    # Win rate: ties count as 0.5
    win_rate = (wins + 0.5 * draws) / n * 100.0
    # CI for win rate (binomial approximation)
    p_hat = win_rate / 100.0
    wr_ci95 = 1.96 * math.sqrt(p_hat * (1 - p_hat) / n) * 100.0

    return {
        "game": job["game"],
        "sims_per_move": sims,
        "row_label": row_spec["label"],
        "col_label": col_spec["label"],
        "mean_payoff": round(mean_payoff, 4),
        "payoff_ci95": round(payoff_ci95, 4),
        "win_rate": round(win_rate, 2),
        "win_rate_ci95": round(wr_ci95, 2),
        "num_matches": n,
        "wins": wins,
        "draws": draws,
        "losses": n - wins - draws,
    }


# ═════════════════════════════════════════════════════════════════════════
# Tables
# ═════════════════════════════════════════════════════════════════════════

def _build_tables(results: list[dict]) -> list[str]:
    """Build markdown tables from head-to-head results.

    Generates one table per (game, sims_per_move) combination,
    formatted as a matrix like the paper's Figure 4.
    """
    md: list[str] = []
    md.append("# Head-to-Head Results\n")

    games = sorted(set(r["game"] for r in results))
    sims_values = sorted(set(r["sims_per_move"] for r in results))
    all_labels = []
    seen = set()
    for r in results:
        for lbl in [r["row_label"], r["col_label"]]:
            if lbl not in seen:
                seen.add(lbl)
                all_labels.append(lbl)

    for game in games:
        for sims in sims_values:
            subset = [r for r in results
                      if r["game"] == game and r["sims_per_move"] == sims]
            if not subset:
                continue

            # Collect labels present in this subset
            labels = []
            label_set = set()
            for r in subset:
                for lbl in [r["row_label"], r["col_label"]]:
                    if lbl not in label_set:
                        label_set.add(lbl)
                        labels.append(lbl)

            # Build lookup: (row, col) -> result
            lookup = {}
            for r in subset:
                lookup[(r["row_label"], r["col_label"])] = r

            # ── Win Rate Table ──
            md.append(f"## {game} — s={sims} (Win Rate %)\n")

            headers = [f"s={sims}"] + labels
            rows = []
            for row_lbl in labels:
                row = [row_lbl]
                for col_lbl in labels:
                    if row_lbl == col_lbl:
                        row.append("-")
                    elif (row_lbl, col_lbl) in lookup:
                        r = lookup[(row_lbl, col_lbl)]
                        row.append(
                            f"{r['win_rate']:.1f}({r['win_rate_ci95']:.1f})"
                        )
                    else:
                        row.append("-")
                rows.append(row)

            aligns = ["l"] + ["r"] * len(labels)
            md.extend(_md_table(headers, rows, aligns))
            md.append("")

            # ── Mean Payoff Table ──
            md.append(f"## {game} — s={sims} (Mean Payoff)\n")

            rows = []
            for row_lbl in labels:
                row = [row_lbl]
                for col_lbl in labels:
                    if row_lbl == col_lbl:
                        row.append("-")
                    elif (row_lbl, col_lbl) in lookup:
                        r = lookup[(row_lbl, col_lbl)]
                        row.append(
                            f"{r['mean_payoff']:+.4f}({r['payoff_ci95']:.4f})"
                        )
                    else:
                        row.append("-")
                rows.append(row)

            md.extend(_md_table(headers, rows, aligns))
            md.append("")

    return md


def _md_table(headers, rows, alignments=None):
    """Render a markdown table."""
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


def print_tables(results: list[dict]) -> None:
    """Print markdown tables to stdout."""
    lines = _build_tables(results)
    print("\n".join(lines))


def save_tables(results: list[dict], output_dir: str) -> None:
    """Save markdown tables to output_dir/tables.md."""
    lines = _build_tables(results)
    path = os.path.join(output_dir, "tables.md")
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


# ═════════════════════════════════════════════════════════════════════════
# Main entry point
# ═════════════════════════════════════════════════════════════════════════

def run(config: HeadsUpConfig) -> list[dict]:
    """Run head-to-head evaluation.

    Generates all pairwise matchups across games and sims budgets,
    runs them in parallel, produces results.json and tables.md.
    """
    output_dir = config.output_dir
    os.makedirs(output_dir, exist_ok=True)

    games = config.games if isinstance(config.games, list) else [config.games]
    sims_list = (config.sims_per_move
                 if isinstance(config.sims_per_move, list)
                 else [config.sims_per_move])
    bots = config.bots
    n_bots = len(bots)
    bot_dicts = [b.to_dict() for b in bots]

    # Build all matchup jobs: for each (game, sims, row_bot, col_bot) pair
    jobs = []
    for game in games:
        for sims in sims_list:
            for i in range(n_bots):
                for j in range(n_bots):
                    if i == j:
                        continue
                    jobs.append({
                        "game": game,
                        "sims_per_move": sims,
                        "row_bot": bot_dicts[i],
                        "col_bot": bot_dicts[j],
                        "num_matches": config.num_matches,
                        "seed": config.seed,
                    })

    total = len(jobs)
    print(f"Head-to-Head Evaluation — {total} matchups")
    print(f"  Games: {games}")
    print(f"  Bots: {[b.label for b in bots]}")
    print(f"  Sims/move: {sims_list}")
    print(f"  Matches per matchup: {config.num_matches}")
    print(f"  Workers: {config.workers}")
    print()

    results = []
    errors = []
    completed = 0
    t_start = time.time()

    incremental_path = os.path.join(output_dir, "incremental_results.jsonl")

    with ProcessPoolExecutor(max_workers=config.workers) as pool:
        future_to_job = {}
        for jd in jobs:
            f = pool.submit(_play_matchup, jd)
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
                    f"{result['game']:15s} s={result['sims_per_move']:>5d}  "
                    f"{result['row_label']:>10s} vs {result['col_label']:<10s}  "
                    f"WR={result['win_rate']:.1f}%  "
                    f"pay={result['mean_payoff']:+.4f}  "
                    f"ETA {est_remaining / 60:.1f}m"
                )
            except Exception as exc:
                jd = future_to_job[future]
                print(
                    f"  [{completed:>3d}/{total}] FAILED "
                    f"{jd['game']} "
                    f"{jd['row_bot']['label']} vs {jd['col_bot']['label']}: "
                    f"{exc}"
                )
                errors.append({
                    "job": jd,
                    "error": str(exc),
                    "traceback": traceback.format_exc(),
                })

    results.sort(key=lambda r: (
        r["game"], r["sims_per_move"], r["row_label"], r["col_label"]
    ))

    results_path = os.path.join(output_dir, "results.json")
    with open(results_path, "w") as f:
        json.dump({
            "config": config.to_dict(),
            "results": results,
        }, f, indent=2)

    elapsed_total = time.time() - t_start
    print(
        f"\nDone in {elapsed_total / 60:.1f}m. "
        f"{len(results)} completed, {len(errors)} errors."
    )

    if errors:
        err_path = os.path.join(output_dir, "errors.json")
        with open(err_path, "w") as f:
            json.dump(errors, f, indent=2)

    print_tables(results)
    save_tables(results, output_dir)

    print(f"\nResults: {results_path}")
    print(f"Tables:  {os.path.join(output_dir, 'tables.md')}")

    return results


def report(results_dir: str) -> None:
    """Regenerate tables from saved results."""
    path = os.path.join(results_dir, "results.json")
    with open(path) as f:
        data = json.load(f)
    results = data["results"] if "results" in data else data
    print_tables(results)
    save_tables(results, results_dir)
    print(f"\nTables regenerated at {os.path.join(results_dir, 'tables.md')}")
