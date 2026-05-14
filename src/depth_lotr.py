"""Depth-LOTR — Local Online Target-Rate sampling for MCCFR with (ρ, w) control.

LOTR is the family of online MCCFR samplers that, at each undiverged
prefix node, take the target action with a *locally specified* probability
(rather than a single global targeting coin like OOS).  Depth-LOTR is
the variant where this local rate τ(d) depends only on the depth d
along the active match prefix of length D:

    q(a*) = τ(d)
    q(a)  = (1 − τ(d)) · σ_ε(a) / (1 − σ_ε(a*))   for a ≠ a*

    q(a*) = τ(d)
    q(a)  = (1 − τ(d)) · σ_ε(a) / (1 − σ_ε(a*))   for a ≠ a*

Once an episode diverges from the target (or after the prefix ends), it
samples freely from σ_ε.  Sampling is therefore product-form

    q(z) = Π_t q_t(a_t)

so the importance weight W = u·π_opp / q stays well behaved.

──────────────────────────────────────────────────────────────────────
(ρ, w) parameterization — "the LOTR schedule"
──────────────────────────────────────────────────────────────────────

Raw τ schedules are hard to interpret.  We instead expose two knobs:

    ρ ∈ [0, 1]        total probability the episode diverges somewhere
                      on the prefix (1 − ρ = stays on full target prefix).
                      Inverse of OOS's δ in spirit:  ρ_LOTR ↔ 1 − δ_OOS.

    w(d) ≥ 0          "depth profile" — a probability distribution over
                      prefix depth (Σ_d w(d) = 1) describing *where*
                      divergence is allocated.

ρ is a global exploration knob playing the role of OOS's δ; w(d) gives
a new axis OOS lacks: *where* on the prefix the divergence occurs.
τ(d) is then derived rather than chosen:

    S(d) = Π_{j<d} τ(j) = 1 − ρ · Σ_{j<d} w(j)

    τ(d) = 1 − ρ · w(d) / S(d)

Properties (all clean):
  - ρ = 0          → τ(d) = 1 always (no exploration, every episode on path)
  - w concentrated → forced divergence at one specific depth
  - cumulative survival to D is exactly 1 − ρ
  - τ(d) ∈ [0, 1] for all ρ ≤ 1 and any nonneg w

Built-in depth profiles (indexed on backoff k = D − d, distance from active):
  - uniform(ρ)        w(k) = 1     (flat — equal mass per backoff position)
  - near_active(ρ)    w(k) = 1/k   (mass concentrates at active end)
  - far_backoff(ρ)    w(k) = k     (mass concentrates at root end)
  - step_at(ρ, k0)    w(k) = δ_{k,k0}  (all mass at one backoff position)
  - max_backoff(ρ)    w(k) = δ_{k,D}   (all mass at the root; OOS-equivalent)

Conditional on diverging, P(backoff=k | diverged, D) = w(k) / Σ_j w(j).
The shape w is what the user specifies; ρ controls the total budget.

q(z) is *not* affected by the (ρ, w) reparameterization — it remains a
product of realized local probabilities.  In particular w(d) never
appears as a global mixture term, so we avoid the retrospective-levels
zero-component blowup.
"""

from __future__ import annotations

import abc
import math

import numpy as np
import pyspiel

try:
    from src.fast_ops import (
        FastRNG, regret_match_sample, regret_match_sample_eps,
        update_regrets, update_avg_strategy,
    )
    _HAS_FAST_OPS = True
except ImportError:
    _HAS_FAST_OPS = False

REGRET_INDEX = 0
AVG_POLICY_INDEX = 1


# ═════════════════════════════════════════════════════════════════════════
# Schedules
# ═════════════════════════════════════════════════════════════════════════

class LOTRSchedule(abc.ABC):
    """Abstract base for LOTR schedules — return per-node target rate τ(d)."""

    @abc.abstractmethod
    def tau(self, d: int, D: int) -> float:
        """Local target-action probability at decision d of D total."""

    @abc.abstractmethod
    def name(self) -> str:
        """Short serializable label."""


class WeightedExplorationSchedule(LOTRSchedule):
    """Depth-LOTR (ρ, w) schedule.

    τ(d) is derived from total divergence budget ρ and a depth profile w(d).
    weight_fn(d, D) returns nonnegative raw weights; they are normalized
    per D into w(d) (the depth profile).  τ is computed once per D and cached.
    """

    def __init__(self, rho, weight_fn, label):
        if not (0.0 <= rho <= 1.0):
            raise ValueError(f"rho must be in [0,1], got {rho}")
        self._rho = float(rho)
        self._weight_fn = weight_fn
        self._label = label
        self._cache = {}

    @property
    def rho(self):
        return self._rho

    def tau(self, d, D):
        if D <= 0:
            return 1.0
        if D not in self._cache:
            self._cache[D] = self._compute(D)
        return float(self._cache[D][d])

    def tau_array(self, D):
        if D <= 0:
            return np.zeros(0, dtype=np.float64)
        if D not in self._cache:
            self._cache[D] = self._compute(D)
        return self._cache[D]

    def _compute(self, D):
        raw = np.array(
            [max(float(self._weight_fn(j, D)), 0.0) for j in range(D)],
            dtype=np.float64,
        )
        total = raw.sum()
        if total > 0:
            w = raw / total
        else:
            w = np.ones(D, dtype=np.float64) / D
        # cum[d] = Σ_{j<d} w(j),  cum[0]=0
        cum = np.concatenate([[0.0], np.cumsum(w)[:-1]])
        S = 1.0 - self._rho * cum
        S = np.maximum(S, 1e-12)
        tau_arr = 1.0 - self._rho * w / S
        return np.clip(tau_arr, 0.0, 1.0)

    def name(self):
        return self._label


def uniform(rho):
    """Uniform shape on backoff: w(k) = 1.

    Conditional on diverging, each available backoff position is equally
    likely.  P(backoff=k | diverged, D) = 1/D for k = 1..D.
    """
    return WeightedExplorationSchedule(
        rho, lambda d, D: 1.0, f"uniform(ρ={rho})"
    )


def near_active(rho):
    """Near-active shape on backoff: w(k) = 1/k.

    Mass concentrates at the active end.  P(backoff=k | diverged, D) ∝ 1/k
    so the step before active (k=1) gets the most mass, falling toward
    the root (k=D).
    """
    return WeightedExplorationSchedule(
        rho, lambda d, D: 1.0 / max(D - d, 1), f"near_active(ρ={rho})"
    )


def far_backoff(rho):
    """Far-backoff shape on backoff: w(k) = k.

    Mass concentrates at the root end.  P(backoff=k | diverged, D) ∝ k
    so the root (k=D) gets the most mass, falling toward active (k=1).
    """
    return WeightedExplorationSchedule(
        rho, lambda d, D: float(D - d), f"far_backoff(ρ={rho})"
    )


def step_at(rho, k0):
    """Step at a specific backoff distance: w(k) = 1 if k==k0 else 0.

    All divergence budget concentrated at one backoff position.  At
    states with D < k0 this falls back to a single coin at the root.
    """
    return WeightedExplorationSchedule(
        rho,
        lambda d, D, _k0=k0: 1.0 if (D - d) == _k0 else (
            1.0 if d == 0 and _k0 > D else 0.0
        ),
        f"step_at(ρ={rho},k={k0})",
    )


def max_backoff(rho):
    """All divergence at maximum backoff (the root): w(k) = 1 if k==D else 0.

    Equivalent to step(ρ, depth=0): every state places its entire ρ budget
    on a single coin at d=0.  This recovers OOS's targeted/untargeted
    dichotomy as a LOTR special case and is the only canonical shape
    that genuinely depends on D.
    """
    return WeightedExplorationSchedule(
        rho, lambda d, D: 1.0 if d == 0 else 0.0, f"max_backoff(ρ={rho})"
    )


def step(rho, depth=0):
    """Step depth profile: all divergence budget concentrated at one depth.

    Depth-indexed counterpart of step_at.  step(ρ, depth=0) is identical
    to max_backoff(ρ) and recovers OOS's targeted/untargeted dichotomy.
    """
    return WeightedExplorationSchedule(
        rho,
        lambda d, D, _k=depth: 1.0 if d == _k else 0.0,
        f"step(ρ={rho},d={depth})",
    )


# ─────────────────────────────────────────────────────────────────────────
# Legacy aliases (depth-indexed shapes, retained for compatibility)
# ─────────────────────────────────────────────────────────────────────────

def late_linear(rho):
    """LEGACY: w(d) ∝ d + 1.  Depth-indexed linear-toward-active.

    Prefer near_active(rho) for the cleaner backoff-indexed form.
    """
    return WeightedExplorationSchedule(
        rho, lambda d, D: d + 1, f"late_linear(ρ={rho})"
    )


def early_linear(rho):
    """LEGACY: w(d) ∝ D − d.  Depth-indexed linear-toward-root.

    Prefer far_backoff(rho) for the cleaner backoff-indexed form.
    """
    return WeightedExplorationSchedule(
        rho, lambda d, D: D - d, f"early_linear(ρ={rho})"
    )


def local_uniform(rho):
    """LEGACY alias for uniform(rho)."""
    return uniform(rho)


def resolve_schedule(name):
    """Reconstruct a schedule from its serialized name string."""
    if name is None:
        return uniform(0.5)
    if name.startswith("uniform("):
        rho = float(name.split("=")[1].rstrip(")"))
        return uniform(rho)
    if name.startswith("late_linear("):
        rho = float(name.split("=")[1].rstrip(")"))
        return late_linear(rho)
    if name.startswith("early_linear("):
        rho = float(name.split("=")[1].rstrip(")"))
        return early_linear(rho)
    if name.startswith("step("):
        body = name[len("step("):-1]
        parts = dict(p.split("=") for p in body.split(","))
        return step(float(parts["ρ"]), int(parts.get("d", 0)))
    if name.startswith("step_at("):
        body = name[len("step_at("):-1]
        parts = dict(p.split("=") for p in body.split(","))
        return step_at(float(parts["ρ"]), int(parts.get("k", 1)))
    if name.startswith("near_active("):
        rho = float(name.split("=")[1].rstrip(")"))
        return near_active(rho)
    if name.startswith("far_backoff("):
        rho = float(name.split("=")[1].rstrip(")"))
        return far_backoff(rho)
    if name.startswith("max_backoff("):
        rho = float(name.split("=")[1].rstrip(")"))
        return max_backoff(rho)
    if name.startswith("local_uniform("):
        rho = float(name.split("=")[1].rstrip(")"))
        return local_uniform(rho)
    raise ValueError(f"Unknown schedule: {name}")


# ═════════════════════════════════════════════════════════════════════════
# DepthLOTRBot
# ═════════════════════════════════════════════════════════════════════════

class DepthLOTRBot(pyspiel.Bot):
    """Depth-LOTR MCCFR bot with (ρ, w) prefix exploration control."""

    def __init__(self, game, player_id, num_simulations=1000,
                 epsilon=0.4, gamma=0.01, schedule=None, seed=None,
                 tracking=False):
        pyspiel.Bot.__init__(self)
        self._game = game
        self._player_id = player_id
        self._num_simulations = num_simulations
        self._epsilon = epsilon
        self._gamma = gamma
        self._schedule = schedule if schedule is not None else uniform(0.5)
        if _HAS_FAST_OPS:
            self._rng = FastRNG(seed if seed is not None else 42)
        else:
            self._rng = np.random.RandomState(seed)

        self._infostates = {}

        self._match_history = []
        self._chance_forced = {}
        self._depth_to_decision = {}
        self._D = 0
        self._num_players = game.num_players()

        self._tracking = tracking
        if tracking:
            self._reset_tracking()

    # -- pyspiel.Bot interface ------------------------------------------------

    def step(self, state):
        _, action = self.step_with_policy(state)
        return action

    def step_with_policy(self, state):
        self._build_targeting_info(state)

        if self._tracking:
            self._reset_tracking()

        for sim in range(self._num_simulations):
            update_player = sim % self._num_players
            root = self._game.new_initial_state()
            if self._tracking:
                self._cur_diverge_depth = -1
                self._cur_chance_diverged = False
                self._cur_prefix_survived = True
                self._cur_sum_abs_regret = 0.0
                self._cur_sum_sq_regret = 0.0
                self._cur_log_W = None
            self._episode(root, update_player)

        info_key = state.information_state_string(self._player_id)
        legal_actions = state.legal_actions()
        policy = self._get_average_policy(info_key, len(legal_actions))

        action_probs = list(zip(legal_actions, policy))
        action = self._rng.choice(legal_actions, p=policy)

        self._match_history.append(action)
        return action_probs, action

    def inform_action(self, state, player_id, action):
        self._match_history = list(state.history()) + [action]

    def restart(self):
        self._match_history = []
        self._chance_forced = {}
        self._depth_to_decision = {}
        self._D = 0

    # -- Tracking -------------------------------------------------------------

    def _reset_tracking(self):
        self._iter_log_l = []
        self._iter_log_W = []
        self._iter_diverge_depth = []
        self._iter_chance_diverged = []
        # Unified apples-to-apples metric (matches OOS): True iff every
        # realized prefix decision and every observable prefix chance
        # outcome matched the match-history target.  Hidden (unobservable)
        # chance is excluded — same convention in OOS.
        self._iter_prefix_survived = []
        self._iter_prefix_D = []
        self._iter_sum_abs_regret = []
        self._iter_sum_sq_regret = []
        self._iter_utility = []

        self._depth_neg_log_q_sum = {}
        self._depth_neg_log_q_count = {}

        self._infostate_stats = {}

    def _ensure_infostate_stats(self, info_key, num_actions):
        if info_key not in self._infostate_stats:
            self._infostate_stats[info_key] = {
                "visits": 0,
                "sum_W2": 0.0,
                "sum_abs_delta": np.zeros(num_actions, dtype=np.float64),
                "sum_sq_delta": np.zeros(num_actions, dtype=np.float64),
            }

    def get_tracking_summary(self):
        if not self._tracking:
            return {}

        n_iters = len(self._iter_log_l)
        if n_iters == 0:
            return {}

        log_l_arr = np.array(self._iter_log_l)
        log_W_arr = np.array(self._iter_log_W) if self._iter_log_W else np.array([0.0])
        div_arr = np.array(self._iter_diverge_depth)
        chance_div_arr = np.array(self._iter_chance_diverged)
        prefix_arr = (np.array(self._iter_prefix_survived)
                      if self._iter_prefix_survived else np.array([True]))
        abs_r_arr = np.array(self._iter_sum_abs_regret)
        sq_r_arr = np.array(self._iter_sum_sq_regret) if self._iter_sum_sq_regret else np.array([0.0])
        util_arr = np.array(self._iter_utility)

        log_abs_W = log_W_arr
        max_log = log_abs_W.max() if len(log_abs_W) > 0 else 0.0
        shifted = log_abs_W - max_log
        sum_W = np.exp(shifted).sum()
        sum_W2 = np.exp(2 * shifted).sum()
        ess = (sum_W ** 2) / sum_W2 if sum_W2 > 0 else float(n_iters)
        ess_per_sim = ess / n_iters

        depth_neg_log_q = {}
        for d in sorted(self._depth_neg_log_q_sum.keys()):
            count = self._depth_neg_log_q_count[d]
            if count > 0:
                depth_neg_log_q[d] = self._depth_neg_log_q_sum[d] / count

        infostate_summary = {}
        visit_counts = []
        for info_key, stats in self._infostate_stats.items():
            v = stats["visits"]
            if v == 0:
                continue
            visit_counts.append(v)
            infostate_summary[info_key] = {
                "visits": v,
                "mean_W2": stats["sum_W2"] / v,
                "mean_abs_delta": (stats["sum_abs_delta"] / v).tolist(),
                "mean_sq_delta": (stats["sum_sq_delta"] / v).tolist(),
            }

        visit_counts_arr = np.array(visit_counts) if visit_counts else np.array([0])
        n_infostates = len(visit_counts)
        on_path_mask = (div_arr == -1) & (~chance_div_arr)
        on_path_count = int(on_path_mask.sum())
        chance_diverged_count = int(chance_div_arr.sum())
        chance_stayed_count = n_iters - chance_diverged_count
        on_path_given_chance = (on_path_count / chance_stayed_count
                                if chance_stayed_count > 0 else 0.0)

        return {
            "log_W_mean": float(log_W_arr.mean()),
            "log_W_std": float(log_W_arr.std()),
            "log_W_median": float(np.median(log_W_arr)),
            "log_W_p95": float(np.percentile(log_W_arr, 95)),
            "log_W_p99": float(np.percentile(log_W_arr, 99)),
            "log_W_max": float(log_W_arr.max()),
            "ess": float(ess),
            "ess_per_sim": float(ess_per_sim),
            "log_l_mean": float(log_l_arr.mean()),
            "log_l_min": float(log_l_arr.min()),
            "log_l_p5": float(np.percentile(log_l_arr, 5)),
            "log_l_median": float(np.median(log_l_arr)),
            "mean_l": float(np.exp(log_l_arr).mean()),
            "min_l": float(np.exp(log_l_arr.min())),
            "mean_inv_l_sq": float(np.exp(-2 * log_l_arr).mean()),
            "max_inv_l": float(np.exp(-log_l_arr.min())),
            "on_path_fraction": on_path_count / n_iters,
            "chance_diverged_fraction": chance_diverged_count / n_iters,
            "on_path_given_chance_stayed": on_path_given_chance,
            # Unified apples-to-apples metric (same definition in OOSBot):
            # fraction of episodes whose realized actions matched the
            # match-history target at every player decision and every
            # observable chance node on the prefix.
            "prefix_survival_fraction": float(prefix_arr.mean()),
            "mean_diverge_depth": float(div_arr[div_arr >= 0].mean()) if (div_arr >= 0).any() else -1.0,
            "prefix_D": int(self._D),
            "depth_neg_log_q": depth_neg_log_q,
            "mean_sum_abs_regret": float(abs_r_arr.mean()),
            "std_sum_abs_regret": float(abs_r_arr.std()),
            "mean_sum_sq_regret": float(sq_r_arr.mean()),
            "p95_sum_sq_regret": float(np.percentile(sq_r_arr, 95)),
            "max_sum_sq_regret": float(sq_r_arr.max()),
            "mean_utility": float(util_arr.mean()),
            "n_infostates_updated": n_infostates,
            "visit_count_min": int(visit_counts_arr.min()) if n_infostates > 0 else 0,
            "visit_count_p5": int(np.percentile(visit_counts_arr, 5)) if n_infostates > 0 else 0,
            "visit_count_median": int(np.median(visit_counts_arr)) if n_infostates > 0 else 0,
            "visit_count_max": int(visit_counts_arr.max()) if n_infostates > 0 else 0,
            "infostate_stats": infostate_summary,
            "n_iters": n_iters,
        }

    def provides_policy(self):
        return True

    # -- Core algorithm -------------------------------------------------------

    def _episode(self, state, update_player):
        x, l_out, u = self._walk(state, update_player,
                                 my_reach=1.0, opp_reach=1.0,
                                 l=1.0, diverged=False)
        if self._tracking:
            self._iter_log_l.append(math.log(max(l_out, 1e-300)))
            self._iter_diverge_depth.append(self._cur_diverge_depth)
            self._iter_chance_diverged.append(self._cur_chance_diverged)
            self._iter_prefix_survived.append(self._cur_prefix_survived)
            self._iter_prefix_D.append(self._D)
            self._iter_sum_abs_regret.append(self._cur_sum_abs_regret)
            self._iter_sum_sq_regret.append(self._cur_sum_sq_regret)
            self._iter_utility.append(u)
            if self._cur_log_W is not None:
                self._iter_log_W.append(self._cur_log_W)
        return (x, l_out, u)

    def _walk(self, state, update_player, my_reach, opp_reach, l, diverged):
        if state.is_terminal():
            return (1.0, l, state.player_return(update_player))

        if state.is_chance_node():
            return self._handle_chance(state, update_player,
                                       my_reach, opp_reach, l, diverged)

        cur_player = state.current_player()
        info_key = state.information_state_string(cur_player)
        legal_actions = state.legal_actions()
        num_actions = len(legal_actions)
        depth = len(state.history())

        in_tree = info_key in self._infostates
        if not in_tree:
            self._infostates[info_key] = [
                np.zeros(num_actions, dtype=np.float64),
                np.zeros(num_actions, dtype=np.float64),
            ]
            action_idx = self._rng.randint(num_actions)
            # Unified prefix-survival tracking: even though we don't
            # τ-blend at not-in-tree nodes (we sample uniformly and drop
            # to playout), the *realized* action still either matches the
            # target or doesn't.  Without this, on-path episodes get
            # over-counted by the tracker.  Algorithm unchanged.
            if self._tracking and not diverged and depth in self._depth_to_decision:
                t_idx = self._get_targeted_action_idx(state, legal_actions)
                if t_idx is not None and action_idx != t_idx:
                    self._cur_prefix_survived = False
                    if self._cur_diverge_depth == -1:
                        self._cur_diverge_depth = self._depth_to_decision[depth]
            state.apply_action(legal_actions[action_idx])
            l_playout = l / num_actions
            x, l_out, u = self._playout(state, update_player, l_playout)
            return (x / num_actions, l_out, u)

        # Regret matching + sampling distribution σ_ε
        if _HAS_FAST_OPS:
            regrets_view = self._infostates[info_key][REGRET_INDEX]
            if cur_player == update_player:
                policy, sample_policy, _auto_idx = regret_match_sample_eps(
                    regrets_view, self._gamma, self._epsilon, self._rng)
            else:
                policy, _auto_idx = regret_match_sample(
                    regrets_view, self._gamma, self._rng)
                sample_policy = policy
        else:
            policy = self._regret_matching(self._infostates[info_key][REGRET_INDEX])
            if cur_player == update_player:
                uniform_p = np.ones(num_actions, dtype=np.float64) / num_actions
                sample_policy = self._epsilon * uniform_p + (1.0 - self._epsilon) * policy
            else:
                sample_policy = policy.copy()

        target_idx = self._get_targeted_action_idx(state, legal_actions)
        in_prefix = (not diverged) and (target_idx is not None)

        if in_prefix:
            d = self._depth_to_decision[depth]
            tau_d = self._schedule.tau(d, self._D)

            sigma_target = sample_policy[target_idx]
            non_target_mass = 1.0 - sigma_target

            blend = np.zeros(num_actions, dtype=np.float64)
            blend[target_idx] = tau_d

            if non_target_mass > 1e-12:
                scale = (1.0 - tau_d) / non_target_mass
                for a in range(num_actions):
                    if a != target_idx:
                        blend[a] = sample_policy[a] * scale
            elif num_actions > 1:
                share = (1.0 - tau_d) / (num_actions - 1)
                for a in range(num_actions):
                    if a != target_idx:
                        blend[a] = share

            sampled_idx = self._rng.choice(num_actions, p=blend)
            action = legal_actions[sampled_idx]
            new_l = l * blend[sampled_idx]
            new_diverged = (sampled_idx != target_idx)
            if self._tracking and new_diverged:
                self._cur_prefix_survived = False
                if self._cur_diverge_depth == -1:
                    self._cur_diverge_depth = d
            if self._tracking:
                q_a = blend[sampled_idx]
                neg_log_q = -math.log(max(q_a, 1e-300))
                if d not in self._depth_neg_log_q_sum:
                    self._depth_neg_log_q_sum[d] = 0.0
                    self._depth_neg_log_q_count[d] = 0
                self._depth_neg_log_q_sum[d] += neg_log_q
                self._depth_neg_log_q_count[d] += 1
        else:
            sampled_idx = self._rng.choice(num_actions, p=sample_policy)
            action = legal_actions[sampled_idx]
            new_l = l * sample_policy[sampled_idx]
            new_diverged = diverged
            if self._tracking and depth in self._depth_to_decision:
                d = self._depth_to_decision[depth]
                q_a = sample_policy[sampled_idx]
                neg_log_q = -math.log(max(q_a, 1e-300))
                if d not in self._depth_neg_log_q_sum:
                    self._depth_neg_log_q_sum[d] = 0.0
                    self._depth_neg_log_q_count[d] = 0
                self._depth_neg_log_q_sum[d] += neg_log_q
                self._depth_neg_log_q_count[d] += 1

        if cur_player == update_player:
            new_my_reach = my_reach * policy[sampled_idx]
            new_opp_reach = opp_reach
        else:
            new_my_reach = my_reach
            new_opp_reach = opp_reach * policy[sampled_idx]

        state.apply_action(action)
        x, l_out, u = self._walk(state, update_player,
                                 new_my_reach, new_opp_reach,
                                 new_l, new_diverged)

        c = x
        x = x * policy[sampled_idx]

        if cur_player == update_player:
            W = u * opp_reach / max(l_out, 1e-30)
            if _HAS_FAST_OPS:
                update_regrets(self._infostates[info_key][REGRET_INDEX],
                               sampled_idx, c, x, W, num_actions)
            else:
                for a_idx in range(num_actions):
                    if a_idx == sampled_idx:
                        self._infostates[info_key][REGRET_INDEX][a_idx] += (c - x) * W
                    else:
                        self._infostates[info_key][REGRET_INDEX][a_idx] += -x * W

            if self._tracking:
                abs_W = abs(W)
                cur_log_W = math.log(max(abs_W, 1e-300))
                if self._cur_log_W is None or cur_log_W > self._cur_log_W:
                    self._cur_log_W = cur_log_W

                self._ensure_infostate_stats(info_key, num_actions)
                stats = self._infostate_stats[info_key]
                stats["visits"] += 1
                stats["sum_W2"] += W * W
                delta_sampled = (c - x) * W
                stats["sum_abs_delta"][sampled_idx] += abs(delta_sampled)
                stats["sum_sq_delta"][sampled_idx] += delta_sampled * delta_sampled
                iter_abs = abs(delta_sampled)
                iter_sq = delta_sampled * delta_sampled
                for a_idx in range(num_actions):
                    if a_idx != sampled_idx:
                        d_a = -x * W
                        stats["sum_abs_delta"][a_idx] += abs(d_a)
                        stats["sum_sq_delta"][a_idx] += d_a * d_a
                        iter_abs += abs(d_a)
                        iter_sq += d_a * d_a
                self._cur_sum_abs_regret += iter_abs
                self._cur_sum_sq_regret += iter_sq
        else:
            if _HAS_FAST_OPS:
                update_avg_strategy(self._infostates[info_key][AVG_POLICY_INDEX],
                                    policy, opp_reach, l, num_actions)
            else:
                for a_idx in range(num_actions):
                    self._infostates[info_key][AVG_POLICY_INDEX][a_idx] += (
                        opp_reach * policy[a_idx] / max(l, 1e-30)
                    )

        return (x, l_out, u)

    def _handle_chance(self, state, update_player, my_reach, opp_reach,
                       l, diverged):
        outcomes = state.chance_outcomes()
        chance_probs = dict(outcomes)
        depth = len(state.history())
        in_history = depth < len(self._match_history)

        forced = (not diverged
                  and in_history
                  and self._chance_forced.get(depth, True)
                  and self._match_history[depth] in chance_probs)

        if forced:
            action = self._match_history[depth]
            rho = chance_probs[action]
            state.apply_action(action)
            x, l_out, u = self._walk(state, update_player,
                                     my_reach, rho * opp_reach,
                                     l, diverged)
            return (rho * x, l_out, u)
        else:
            actions, probs = zip(*outcomes)
            action = self._rng.choice(actions, p=probs)
            rho = chance_probs[action]

            new_diverged = diverged
            if not diverged and in_history:
                if action != self._match_history[depth]:
                    new_diverged = True
                    if self._tracking:
                        self._cur_chance_diverged = True

            state.apply_action(action)
            x, l_out, u = self._walk(state, update_player,
                                     my_reach, rho * opp_reach,
                                     l * rho, new_diverged)
            return (rho * x, l_out, u)

    def _playout(self, state, update_player, sample_reach):
        x = 1.0
        while not state.is_terminal():
            if state.is_chance_node():
                actions, probs = zip(*state.chance_outcomes())
                idx = self._rng.choice(len(actions), p=probs)
                x *= probs[idx]
                sample_reach *= probs[idx]
                state.apply_action(actions[idx])
            else:
                legal = state.legal_actions()
                n = len(legal)
                action = legal[self._rng.randint(n)]
                x *= 1.0 / n
                sample_reach *= 1.0 / n
                state.apply_action(action)

        return (x, sample_reach, state.player_return(update_player))

    # -- Targeting helpers ----------------------------------------------------

    def _build_targeting_info(self, current_state):
        self._match_history = list(current_state.history())
        self._chance_forced = {}
        self._depth_to_decision = {}

        if not self._match_history:
            self._D = 0
            return

        history = self._match_history
        n = len(history)

        target_info = current_state.information_state_string(self._player_id)
        s = self._game.new_initial_state()
        d_count = 0
        for d in range(n):
            if s.is_chance_node():
                self._chance_forced[d] = self._is_chance_observable(
                    s, d, history, n, target_info
                )
            elif not s.is_terminal():
                self._depth_to_decision[d] = d_count
                d_count += 1
            s.apply_action(history[d])

        self._D = d_count

    def _is_chance_observable(self, state_at_depth, depth, history, n,
                              target_info):
        actual = history[depth]
        for alt, _ in state_at_depth.chance_outcomes():
            if alt == actual:
                continue
            alt_s = state_at_depth.clone()
            alt_s.apply_action(alt)
            reachable = True
            for i in range(depth + 1, n):
                if alt_s.is_terminal():
                    reachable = False
                    break
                if alt_s.is_chance_node():
                    co = dict(alt_s.chance_outcomes())
                    if history[i] in co:
                        alt_s.apply_action(history[i])
                    else:
                        reachable = False
                        break
                else:
                    if history[i] in alt_s.legal_actions():
                        alt_s.apply_action(history[i])
                    else:
                        reachable = False
                        break
            if reachable and not alt_s.is_terminal():
                try:
                    alt_info = alt_s.information_state_string(self._player_id)
                    if alt_info == target_info:
                        return False
                except Exception:
                    pass
        return True

    def _get_targeted_action_idx(self, state, legal_actions):
        depth = len(state.history())
        if depth < len(self._match_history) and depth in self._depth_to_decision:
            target = self._match_history[depth]
            if target in legal_actions:
                return legal_actions.index(target)
        return None

    # -- Strategy computation -------------------------------------------------

    def _regret_matching(self, regrets):
        positive = np.maximum(regrets, 0)
        total = positive.sum()
        n = len(regrets)
        if total > 0:
            policy = positive / total
        else:
            policy = np.ones(n, dtype=np.float64) / n
        if self._gamma > 0:
            policy = self._gamma / n + (1.0 - self._gamma) * policy
        return policy

    def _get_average_policy(self, info_key, num_actions):
        if info_key not in self._infostates:
            return np.ones(num_actions, dtype=np.float64) / num_actions
        avg = self._infostates[info_key][AVG_POLICY_INDEX]
        total = avg.sum()
        if total > 0:
            return avg / total
        return np.ones(num_actions, dtype=np.float64) / num_actions

    def average_policy(self):
        return DepthLOTRPolicy(self._game, self._infostates)


class DepthLOTRPolicy:
    """Adapter so DepthLOTRBot info states look like an OpenSpiel Policy."""

    def __init__(self, game, infostates):
        self._game = game
        self._infostates = infostates

    def action_probabilities(self, state, player_id=None):
        if player_id is None:
            player_id = state.current_player()
        info_key = state.information_state_string(player_id)
        legal_actions = state.legal_actions()
        n = len(legal_actions)

        if info_key in self._infostates:
            avg = self._infostates[info_key][AVG_POLICY_INDEX]
            total = avg.sum()
            if total > 0:
                probs = avg / total
            else:
                probs = np.ones(n, dtype=np.float64) / n
        else:
            probs = np.ones(n, dtype=np.float64) / n

        return dict(zip(legal_actions, probs))
