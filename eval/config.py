"""Shared experiment configuration with automatic grid expansion.

EvalConfig is a flat dataclass where any sweepable field can accept a
scalar or a list.  The expand() method takes the cartesian product of
all list-valued sweepable fields, producing one JobSpec per combination.

Sweepable fields (scalar or list):
    algorithm, delta, epsilon, gamma, bias_mode, decay_fn,
    max_iig_depth, sims_per_move, num_matches

Infrastructure fields (never swept):
    games, seed, workers, output_dir

Algorithm-aware deduplication:
    When algorithm="oos", the fields decay_fn, max_iig_depth, and
    bias_mode are ignored during grid expansion — OOS jobs are
    deduplicated so that varying these ISGT-only parameters doesn't
    produce redundant OOS runs.

Auto-labeling:
    Each JobSpec gets a label built from the fields that actually vary
    across the grid.  If only sims_per_move and num_matches vary, a
    job's label might be "sims=200_m=10".  Decay functions use their
    .name() method.

Usage:
    from eval.config import EvalConfig
    from src.isgt import LevelUniform, LevelExponential

    config = EvalConfig(
        games=["leduc_poker", "goofspiel", "liars_dice"],
        algorithm="isgt",
        delta=0.7,
        epsilon=0.4,
        sims_per_move=100,
        num_matches=1000,
        decay_fn=[LevelUniform(), LevelExponential(0.7)],
        bias_mode=["full", "chance"],
        output_dir="results/my_experiment",
    )

    jobs = config.expand()  # 2 decays × 2 modes = 4 JobSpecs per game
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from typing import Any

from src.isgt import LevelWeightFn, LevelUniform
from src.depth_delta import DeltaSchedule, LinearSchedule


# ═════════════════════════════════════════════════════════════════════════
# JobSpec — one fully-resolved point in the parameter grid
# ═════════════════════════════════════════════════════════════════════════

@dataclass
class JobSpec:
    """A single resolved experiment configuration (no lists)."""
    game: str
    algorithm: str
    delta: float
    epsilon: float
    gamma: float
    bias_mode: str
    decay_fn: LevelWeightFn
    max_iig_depth: int | None
    sims_per_move: int
    num_matches: int
    schedule: DeltaSchedule | None = None
    checkpoints: list[int] = field(default_factory=list)
    seed: int = 42
    label: str = ""

    def decay_name(self) -> str:
        return self.decay_fn.name()

    def schedule_name(self) -> str:
        return self.schedule.name() if self.schedule is not None else "linear"

    def to_dict(self) -> dict:
        """Serializable dict for JSON output."""
        return {
            "game": self.game,
            "algorithm": self.algorithm,
            "delta": self.delta,
            "epsilon": self.epsilon,
            "gamma": self.gamma,
            "bias_mode": self.bias_mode,
            "decay_fn": self.decay_name(),
            "max_iig_depth": self.max_iig_depth,
            "sims_per_move": self.sims_per_move,
            "num_matches": self.num_matches,
            "schedule": self.schedule_name(),
            "checkpoints": self.checkpoints,
            "seed": self.seed,
            "label": self.label,
        }


# ═════════════════════════════════════════════════════════════════════════
# EvalConfig — flat config with grid expansion
# ═════════════════════════════════════════════════════════════════════════

# Fields that are irrelevant per algorithm — collapsed during grid expansion
_ALGO_IGNORE_FIELDS = {
    "oos": {"decay_fn", "max_iig_depth", "bias_mode", "schedule"},
    "retro": {"max_iig_depth", "bias_mode", "schedule"},
    "depth_delta": {"decay_fn", "max_iig_depth", "bias_mode", "delta"},
}

# Union of all algorithm-specific fields (for label building)
_ALGO_SPECIFIC_FIELDS = {"decay_fn", "max_iig_depth", "bias_mode", "schedule"}

# Fields eligible for grid sweep (num_matches handled separately as checkpoints)
_SWEEPABLE_FIELDS = [
    "algorithm", "delta", "epsilon", "gamma", "bias_mode",
    "decay_fn", "max_iig_depth", "schedule", "sims_per_move",
]


def _ensure_list(val: Any) -> list:
    """Wrap scalars and None in a list."""
    if val is None:
        return [None]
    if isinstance(val, list):
        return val
    return [val]


def _label_value(field_name: str, val: Any) -> str:
    """Short label fragment for a single field value."""
    if field_name == "decay_fn" and isinstance(val, LevelWeightFn):
        return val.name()
    if field_name == "schedule" and isinstance(val, DeltaSchedule):
        return val.name()
    if field_name == "max_iig_depth":
        return f"d{val}" if val is not None else "d∞"
    if field_name == "sims_per_move":
        return f"sims={val}"
    if field_name == "num_matches":
        return f"m={val}"
    if field_name == "delta":
        return f"δ={val}"
    if field_name == "epsilon":
        return f"ε={val}"
    if field_name == "gamma":
        return f"γ={val}"
    return str(val)


@dataclass
class EvalConfig:
    """Flat experiment configuration with automatic grid expansion.

    Any sweepable field accepts a scalar or list.  expand() returns the
    cartesian product as a list of JobSpec objects.
    """

    # ── Infrastructure (never swept) ──
    games: str | list[str] = "leduc_poker"
    seed: int = 42
    workers: int = 4
    output_dir: str = ""

    # ── Sweepable (scalar or list) ──
    algorithm: str | list[str] = "isgt"
    delta: float | list[float] = 0.9
    epsilon: float | list[float] = 0.6
    gamma: float | list[float] = 0.01
    bias_mode: str | list[str] = "full"
    decay_fn: LevelWeightFn | list[LevelWeightFn] | None = None
    max_iig_depth: int | list[int | None] | None = None
    schedule: DeltaSchedule | list[DeltaSchedule] | None = None
    sims_per_move: int | list[int] = 100
    num_matches: int | list[int] = 500

    def expand(self) -> list[JobSpec]:
        """Expand config into a list of JobSpecs (cartesian product).

        OOS jobs are deduplicated: ISGT-only fields (decay_fn,
        max_iig_depth, bias_mode) are collapsed for algorithm="oos".

        num_matches is NOT swept.  When given as a list (e.g. [10, 30, 100])
        the values become checkpoints and num_matches is set to the maximum.
        """
        games = _ensure_list(self.games)

        # num_matches → checkpoints (not part of the cartesian sweep)
        match_list = sorted(_ensure_list(self.num_matches))
        max_matches = match_list[-1]
        checkpoints = match_list

        # Build per-field value lists
        field_values = {}
        for fname in _SWEEPABLE_FIELDS:
            raw = getattr(self, fname)
            if fname == "decay_fn" and raw is None:
                field_values[fname] = [LevelUniform()]
            elif fname == "schedule" and raw is None:
                field_values[fname] = [LinearSchedule()]
            else:
                field_values[fname] = _ensure_list(raw)

        # Identify which fields actually vary (for auto-labeling)
        varying = {f for f in _SWEEPABLE_FIELDS if len(field_values[f]) > 1}

        # Cartesian product of all sweepable fields
        keys = _SWEEPABLE_FIELDS
        value_lists = [field_values[k] for k in keys]

        jobs: list[JobSpec] = []
        seen_dedup: dict[str, set[tuple]] = {}  # algorithm -> seen combos

        for combo in itertools.product(*value_lists):
            params = dict(zip(keys, combo))
            algo = params["algorithm"]
            ignore_fields = _ALGO_IGNORE_FIELDS.get(algo, set())

            # Dedup: collapse irrelevant fields for this algorithm
            if ignore_fields:
                dedup_key = tuple(
                    params[f] for f in keys if f not in ignore_fields
                )
                if algo not in seen_dedup:
                    seen_dedup[algo] = set()
                if dedup_key in seen_dedup[algo]:
                    continue
                seen_dedup[algo].add(dedup_key)
                # Normalize irrelevant fields
                if "bias_mode" in ignore_fields:
                    params["bias_mode"] = "full"
                if "decay_fn" in ignore_fields:
                    params["decay_fn"] = LevelUniform()
                if "max_iig_depth" in ignore_fields:
                    params["max_iig_depth"] = None
                if "schedule" in ignore_fields:
                    params["schedule"] = LinearSchedule()
                if "delta" in ignore_fields:
                    params["delta"] = 0.5

            # Build label from varying fields (skip irrelevant for this algo)
            # sims_per_move is excluded because it's the x-axis variable
            label_parts = []
            if "algorithm" in varying:
                label_parts.append(params["algorithm"])
            for f in keys:
                if f == "algorithm":
                    continue
                if f == "sims_per_move":
                    continue
                if f in ignore_fields:
                    continue
                if f in varying:
                    label_parts.append(_label_value(f, params[f]))
            label = "_".join(label_parts) if label_parts else params["algorithm"]

            for game in games:
                jobs.append(JobSpec(
                    game=game,
                    algorithm=params["algorithm"],
                    delta=params["delta"],
                    epsilon=params["epsilon"],
                    gamma=params["gamma"],
                    bias_mode=params["bias_mode"],
                    decay_fn=params["decay_fn"],
                    max_iig_depth=params["max_iig_depth"],
                    schedule=params["schedule"],
                    sims_per_move=params["sims_per_move"],
                    num_matches=max_matches,
                    checkpoints=checkpoints,
                    seed=self.seed,
                    label=label,
                ))

        return jobs

    def swept_fields(self) -> list[str]:
        """Return names of fields that have multiple values."""
        result = []
        for fname in _SWEEPABLE_FIELDS:
            raw = getattr(self, fname)
            vals = _ensure_list(raw)
            if len(vals) > 1:
                result.append(fname)
        return result

    def to_dict(self) -> dict:
        """Serializable dict for JSON output (config snapshot)."""
        d = {}
        d["games"] = _ensure_list(self.games)
        d["seed"] = self.seed
        d["workers"] = self.workers
        d["output_dir"] = self.output_dir
        for fname in _SWEEPABLE_FIELDS:
            raw = getattr(self, fname)
            vals = _ensure_list(raw)
            if fname == "decay_fn":
                vals = [v.name() if isinstance(v, LevelWeightFn) else str(v)
                        for v in vals]
            elif fname == "schedule":
                vals = [v.name() if isinstance(v, DeltaSchedule) else str(v)
                        for v in vals]
            d[fname] = vals if len(vals) > 1 else vals[0]
        d["num_matches"] = _ensure_list(self.num_matches)
        return d
