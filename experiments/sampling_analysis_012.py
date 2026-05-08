"""012 — Sampling Distribution Analysis.

Runs ISGT iterations and histograms the IIG distance of sampled z*
terminals (and reached terminals) across parameter configurations.
Answers: does decay concentrate sampling? How does δ dilute targeting?
How do chance-only vs full mode differ in effective distribution?

Usage:
    python -m experiments.sampling_analysis_012
    python -m experiments.sampling_analysis_012 --game leduc_poker
    python -m experiments.sampling_analysis_012 --json results.json
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from dataclasses import asdict, dataclass, field

import numpy as np
import pyspiel

from src.games import load_game
from src.iig import IIG
from src.isgt import (
    ISGTBot,
    ConstantWeight,
    ExponentialDecay,
    PolynomialDecay,
    StepFunction,
    REGRET_INDEX,
    AVG_POLICY_INDEX,
)


# ═════════════════════════════════════════════════════════════════════════
# Configuration
# ═════════════════════════════════════════════════════════════════════════

@dataclass
class Config:
    """One histogram experiment configuration."""
    game: str = "kuhn_poker"
    decay_fn: object = None
    delta: float = 0.9
    epsilon: float = 0.6
    bias_mode: str = "chance"
    num_sims: int = 100
    seed: int = 42
    label: str = ""

    def __post_init__(self):
        if self.decay_fn is None:
            self.decay_fn = ConstantWeight()
        if not self.label:
            self.label = f"{self.decay_fn.name()}, δ={self.delta}, {self.bias_mode}"


# ═════════════════════════════════════════════════════════════════════════
# Per-iteration record
# ═════════════════════════════════════════════════════════════════════════

@dataclass
class SampleRecord:
    """Data from one ISGT iteration."""
    targeted: bool
    z_star_tid: int
    z_star_iig_distance: int   # -1 = non-upstream (floor)
    reached_tid: int
    reached_iig_distance: int  # -1 = non-upstream (floor)


# ═════════════════════════════════════════════════════════════════════════
# Result container
# ═════════════════════════════════════════════════════════════════════════

@dataclass
class HistogramResult:
    """Aggregated results from one histogram experiment."""
    label: str
    records: list[SampleRecord]
    active_infoset: str
    num_upstream_terminals: int
    num_terminals: int
    max_level: int
    config_summary: dict

    def _histogram(self, key: str, targeted_only: bool) -> dict[int, int]:
        recs = self.records
        if targeted_only:
            recs = [r for r in recs if r.targeted]
        counts = Counter(getattr(r, key) for r in recs)
        return dict(sorted(counts.items()))

    def z_star_histogram(self, targeted_only: bool = False) -> dict[int, int]:
        return self._histogram("z_star_iig_distance", targeted_only)

    def reached_histogram(self, targeted_only: bool = False) -> dict[int, int]:
        return self._histogram("reached_iig_distance", targeted_only)

    @property
    def num_targeted(self) -> int:
        return sum(1 for r in self.records if r.targeted)

    @property
    def num_untargeted(self) -> int:
        return len(self.records) - self.num_targeted

    def to_dict(self) -> dict:
        return {
            "label": self.label,
            "active_infoset": self.active_infoset,
            "num_upstream_terminals": self.num_upstream_terminals,
            "num_terminals": self.num_terminals,
            "max_level": self.max_level,
            "config": self.config_summary,
            "num_targeted": self.num_targeted,
            "num_untargeted": self.num_untargeted,
            "z_star_all": self.z_star_histogram(False),
            "z_star_targeted": self.z_star_histogram(True),
            "reached_all": self.reached_histogram(False),
            "reached_targeted": self.reached_histogram(True),
        }


# ═════════════════════════════════════════════════════════════════════════
# Analysis bot — thin subclass exposing z* terminal ID
# ═════════════════════════════════════════════════════════════════════════

class _AnalysisBot(ISGTBot):

    def _sample_target(self, target_dist):
        tid = self._rng.choice(len(target_dist), p=target_dist)
        self._last_z_star_tid = tid
        history = self._iig._terminal_histories[tid]
        chance_actions = self._extract_chance_actions(history)
        return history, chance_actions, target_dist[tid]


# ═════════════════════════════════════════════════════════════════════════
# IIG distance computation
# ═════════════════════════════════════════════════════════════════════════

def _deepest_infoset(iig: IIG) -> tuple:
    """Find the infoset with the maximum upstream BFS depth.

    This ensures histograms reveal the full multi-level IIG structure
    rather than collapsing to a flat L0/L-1 binary split (which is
    what happens at root infosets with no predecessors).
    """
    best_iid = None
    best_depth = -1
    for iid in sorted(iig.infosets):
        level_map = iig.levels(iid)
        max_level = max(level_map.values())
        if max_level > best_depth:
            best_depth = max_level
            best_iid = iid
    return best_iid


def _iig_distance(iig: IIG, level_map: dict, tid: int) -> int:
    """Min upstream BFS distance among infosets on terminal tid's path.

    Returns -1 if no infoset on this terminal is in the upstream
    neighborhood (i.e., it's a floor-weight terminal).
    """
    seq = iig._terminal_seqs[tid]
    min_d = None
    for iid in seq:
        d = level_map.get(iid)
        if d is not None and (min_d is None or d < min_d):
            min_d = d
    return min_d if min_d is not None else -1


# ═════════════════════════════════════════════════════════════════════════
# Core experiment runner
# ═════════════════════════════════════════════════════════════════════════

def run_histogram_experiment(
    config: Config,
    iig: IIG | None = None,
    game: pyspiel.Game | None = None,
    active_iid: tuple | None = None,
) -> HistogramResult:
    """Run N ISGT iterations and record z*/reached IIG distances.

    Args:
        config: Experiment configuration.
        iig: Pre-built IIG (avoids redundant construction across configs).
        game: Pre-loaded game (avoids redundant loading).
        active_iid: Active infoset. Defaults to the deepest infoset
            (max upstream BFS depth) to reveal multi-level structure.

    Returns:
        HistogramResult with per-iteration records.
    """
    if game is None:
        game = load_game(config.game)
    if iig is None:
        iig = IIG(game)

    # Pick active infoset — choose deepest (max upstream BFS depth)
    if active_iid is None:
        active_iid = _deepest_infoset(iig)

    # Precompute IIG structure
    level_map = iig.levels(active_iid)
    level_sets = iig.level_sets(active_iid)
    max_level = max(level_sets) if level_sets else 0
    upstream_tids = set()
    for iid in level_map:
        upstream_tids |= iig.z_set(iid)

    # Reverse lookup: history → tid
    history_to_tid = {h: tid for tid, h in enumerate(iig._terminal_histories)}

    # Create analysis bot
    bot = _AnalysisBot(
        game, player_id=0, num_simulations=1,
        epsilon=config.epsilon, gamma=0.01,
        level_weight_fn=config.decay_fn,
        bias_mode=config.bias_mode,
        seed=config.seed, iig=iig, delta=config.delta,
    )

    # Precompute targeting infrastructure
    iig_weights = bot._get_iig_weights(active_iid)
    target_dist = bot._get_target_dist(active_iid, iig_weights)

    # Run iterations
    records = []
    for sim in range(config.num_sims):
        update_player = sim % game.num_players()
        root = game.new_initial_state()

        # Sample z*
        history, chance_actions, target_prob = bot._sample_target(target_dist)
        z_star_tid = bot._last_z_star_tid

        # Set up iteration state
        bot._z_star_history = history
        bot._z_star_chance_actions = chance_actions
        bot._z_star_chance_idx = 0
        bot._is_targeted_iter = (bot._rng.random() < bot._delta)

        # Walk
        bot._walk(root, update_player, 1.0, 1.0, target_prob, 1.0)

        # Capture reached terminal
        reached_history = tuple(root.history())
        reached_tid = history_to_tid.get(reached_history, -1)

        # Compute IIG distances
        z_dist = _iig_distance(iig, level_map, z_star_tid)
        r_dist = _iig_distance(iig, level_map, reached_tid) if reached_tid >= 0 else -1

        records.append(SampleRecord(
            targeted=bot._is_targeted_iter,
            z_star_tid=z_star_tid,
            z_star_iig_distance=z_dist,
            reached_tid=reached_tid,
            reached_iig_distance=r_dist,
        ))

    return HistogramResult(
        label=config.label,
        records=records,
        active_infoset=iig._fmt_id(active_iid),
        num_upstream_terminals=len(upstream_tids),
        num_terminals=iig.num_terminals,
        max_level=max_level,
        config_summary={
            "game": config.game,
            "decay": config.decay_fn.name(),
            "delta": config.delta,
            "epsilon": config.epsilon,
            "bias_mode": config.bias_mode,
            "num_sims": config.num_sims,
            "seed": config.seed,
        },
    )


# ═════════════════════════════════════════════════════════════════════════
# Parameter sweep
# ═════════════════════════════════════════════════════════════════════════

def run_parameter_sweep(
    configs: list[Config],
    game: pyspiel.Game | None = None,
    iig: IIG | None = None,
    active_iid: tuple | None = None,
) -> list[HistogramResult]:
    """Run a grid of configs, sharing the game/IIG across runs."""
    if configs and game is None:
        game = load_game(configs[0].game)
    if iig is None:
        iig = IIG(game)
    return [
        run_histogram_experiment(c, iig=iig, game=game, active_iid=active_iid)
        for c in configs
    ]


# ═════════════════════════════════════════════════════════════════════════
# Display helpers
# ═════════════════════════════════════════════════════════════════════════

def _level_label(d: int) -> str:
    return f"Level {d}" if d >= 0 else "Floor"


def print_single_histogram(result: HistogramResult):
    """Print a single experiment's histograms."""
    print(f"\n  Config: {result.label}")
    print(f"  Active infoset: {result.active_infoset}")
    print(f"  Upstream terminals: {result.num_upstream_terminals}/{result.num_terminals}")
    print(f"  Max IIG level: {result.max_level}")
    print(f"  Iterations: {len(result.records)} "
          f"({result.num_targeted} targeted, {result.num_untargeted} untargeted)")

    for hist_name, method in [("z* sampled", "z_star_histogram"),
                              ("Reached terminal", "reached_histogram")]:
        all_hist = getattr(result, method)(targeted_only=False)
        tgt_hist = getattr(result, method)(targeted_only=True)

        print(f"\n  {hist_name} — IIG Distance Distribution:")
        # Collect all distance keys
        all_dists = sorted(set(all_hist) | set(tgt_hist))
        n_all = sum(all_hist.values())
        n_tgt = sum(tgt_hist.values())

        print(f"  {'Distance':>12s}  {'Targeted':>10s}  {'All':>10s}")
        print(f"  {'─' * 12}  {'─' * 10}  {'─' * 10}")
        for d in all_dists:
            t = tgt_hist.get(d, 0)
            a = all_hist.get(d, 0)
            lbl = _level_label(d)
            t_pct = f"{t} ({100*t/n_tgt:.0f}%)" if n_tgt > 0 else "—"
            a_pct = f"{a} ({100*a/n_all:.0f}%)" if n_all > 0 else "—"
            print(f"  {lbl:>12s}  {t_pct:>10s}  {a_pct:>10s}")


def print_histogram_comparison(
    results: list[HistogramResult],
    title: str = "Comparison",
    metric: str = "z_star",
    targeted_only: bool = True,
):
    """Print side-by-side histogram comparison across configs.

    Args:
        results: List of HistogramResults to compare.
        title: Section title.
        metric: "z_star" or "reached".
        targeted_only: If True, show targeted-only histogram.
    """
    if not results:
        return

    method = f"{metric}_histogram"
    subset_label = "Targeted Only" if targeted_only else "All Samples"
    hist_label = "z* sampled" if metric == "z_star" else "Reached terminal"

    print(f"\n{'=' * 72}")
    print(f"  {title}")
    print(f"  {hist_label} — {subset_label}")
    print(f"  Game: {results[0].config_summary['game']}, "
          f"Active: {results[0].active_infoset}, "
          f"Upstream: {results[0].num_upstream_terminals}/{results[0].num_terminals}")
    print(f"{'=' * 72}")

    # Gather histograms and all distance keys
    hists = [getattr(r, method)(targeted_only=targeted_only) for r in results]
    all_dists = sorted(set().union(*hists))
    totals = [sum(h.values()) for h in hists]

    # Column widths
    labels = [r.label for r in results]
    col_w = max(14, max(len(lb) for lb in labels) + 2)

    # Header
    dist_col = "Distance"
    header = f"  {dist_col:<12s}"
    for lb in labels:
        header += f"  {lb:>{col_w}s}"
    print(f"\n{header}")
    sep = f"  {'─' * 12}"
    for _ in labels:
        sep += f"  {'─' * col_w}"
    print(sep)

    # Rows
    for d in all_dists:
        lbl = _level_label(d)
        row = f"  {lbl:<12s}"
        for i, h in enumerate(hists):
            c = h.get(d, 0)
            n = totals[i]
            if n > 0:
                cell = f"{c} ({100*c/n:.0f}%)"
            else:
                cell = "—"
            row += f"  {cell:>{col_w}s}"
        print(row)

    # Totals
    row = f"  {'Total':<12s}"
    for n in totals:
        row += f"  {str(n):>{col_w}s}"
    print(sep)
    print(row)


# ═════════════════════════════════════════════════════════════════════════
# Predefined sweeps
# ═════════════════════════════════════════════════════════════════════════

def decay_comparison(game: str = "kuhn_poker", n: int = 100, seed: int = 42):
    """Sweep 1: Constant vs shaped decays (fixed δ=0.9, chance mode)."""
    return [
        Config(game=game, decay_fn=ConstantWeight(), delta=0.9,
               num_sims=n, seed=seed, label="constant"),
        Config(game=game, decay_fn=ExponentialDecay(0.5), delta=0.9,
               num_sims=n, seed=seed, label="exp(0.5)"),
        Config(game=game, decay_fn=ExponentialDecay(0.1), delta=0.9,
               num_sims=n, seed=seed, label="exp(0.1)"),
        Config(game=game, decay_fn=StepFunction(0.01), delta=0.9,
               num_sims=n, seed=seed, label="step(0.01)"),
    ]


def delta_sweep(game: str = "kuhn_poker", n: int = 100, seed: int = 42):
    """Sweep 2: δ values (fixed Exp(0.5), chance mode)."""
    return [
        Config(game=game, decay_fn=ExponentialDecay(0.5), delta=d,
               num_sims=n, seed=seed, label=f"δ={d}")
        for d in [0.0, 0.2, 0.5, 0.9, 1.0]
    ]


def mode_comparison(game: str = "kuhn_poker", n: int = 100, seed: int = 42):
    """Sweep 3: Chance-only vs full mode (fixed Exp(0.5), δ=0.9)."""
    return [
        Config(game=game, decay_fn=ExponentialDecay(0.5), delta=0.9,
               bias_mode="chance", num_sims=n, seed=seed, label="chance"),
        Config(game=game, decay_fn=ExponentialDecay(0.5), delta=0.9,
               bias_mode="full", num_sims=n, seed=seed, label="full"),
    ]


def epsilon_sweep(game: str = "kuhn_poker", n: int = 100, seed: int = 42):
    """Sweep 4: ε exploration values (fixed Exp(0.5), δ=0.9, chance)."""
    return [
        Config(game=game, decay_fn=ExponentialDecay(0.5), delta=0.9,
               epsilon=e, num_sims=n, seed=seed, label=f"ε={e}")
        for e in [0.0, 0.2, 0.6, 1.0]
    ]


# ═════════════════════════════════════════════════════════════════════════
# Main
# ═════════════════════════════════════════════════════════════════════════

SWEEPS = {
    "decay": ("Decay Function Comparison", decay_comparison),
    "delta": ("δ (Targeting Probability) Sweep", delta_sweep),
    "mode": ("Bias Mode Comparison (chance vs full)", mode_comparison),
    "epsilon": ("ε (Exploration) Sweep", epsilon_sweep),
}


def main():
    parser = argparse.ArgumentParser(
        description="012 — Sampling Distribution Analysis")
    parser.add_argument("--game", default="kuhn_poker",
                        choices=["kuhn_poker", "leduc_poker", "liars_dice", "goofspiel"])
    parser.add_argument("--sweep", nargs="*", default=None,
                        help="Sweeps to run (decay, delta, mode, epsilon). "
                             "Default: all.")
    parser.add_argument("--num-sims", type=int, default=100,
                        help="Simulations per config (default: 100)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--json", default=None,
                        help="Export results to JSON file")
    args = parser.parse_args()

    sweep_names = args.sweep if args.sweep else list(SWEEPS.keys())

    print("=" * 72)
    print(f"  012 — Sampling Distribution Analysis")
    print(f"  Game: {args.game}, Sims/config: {args.num_sims}, Seed: {args.seed}")
    print("=" * 72)

    # Load game & IIG once
    game = load_game(args.game)
    iig = IIG(game)
    print(f"  Infosets: {iig.num_infosets}, Terminals: {iig.num_terminals}")

    all_results = {}

    for sweep_name in sweep_names:
        if sweep_name not in SWEEPS:
            print(f"\n  Unknown sweep: {sweep_name}. Skipping.")
            continue

        title, config_fn = SWEEPS[sweep_name]
        configs = config_fn(args.game, args.num_sims, args.seed)
        results = run_parameter_sweep(configs, game=game, iig=iig)
        all_results[sweep_name] = results

        # z* distribution (targeted only) — the primary observable
        print_histogram_comparison(results, title=title,
                                   metric="z_star", targeted_only=True)

        # Reached terminal (targeted only) — shows effective targeting
        print_histogram_comparison(results, title=f"{title} — Reached",
                                   metric="reached", targeted_only=True)

        # Reached terminal (all) — shows full algorithm behavior
        print_histogram_comparison(results, title=f"{title} — Reached (All)",
                                   metric="reached", targeted_only=False)

    # JSON export
    if args.json:
        export = {}
        for name, results in all_results.items():
            # Convert int keys to str for JSON
            export[name] = []
            for r in results:
                d = r.to_dict()
                for k in ["z_star_all", "z_star_targeted",
                          "reached_all", "reached_targeted"]:
                    d[k] = {str(kk): v for kk, v in d[k].items()}
                export[name].append(d)
        with open(args.json, "w") as f:
            json.dump(export, f, indent=2)
        print(f"\n  Results exported to {args.json}")

    print()


if __name__ == "__main__":
    main()
