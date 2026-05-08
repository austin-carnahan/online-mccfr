"""Mixture LOTR (v4) — per-node OOS-style coin at every observable prefix node.

Refines Observable-Node LOTR (v3) by replacing the redistribution
τ-blend at the divergence coin with an OOS-style *mixture* coin and
adopting **sticky-by-coin** divergence semantics.  Importance weights
are computed via **arm-marginal IS** (the natural generalization of
OOS's `l = δ·s1 + (1-δ)·s2` to D+1 latent scenarios under sticky-by-
coin divergence).

At each undiverged observable prefix position with target a*, schedule
rate τ(d), and natural distribution π (σ_ε at decisions, chance probs
at observable chance):

    flip a coin:
      with prob τ(d):     mode = force.   play target a*.
      with prob 1 − τ(d): mode = diverge. sample from π.

Sticky-by-coin: divergence is determined by the latent coin, not by
whether the realized action happens to match target.  A diverge coin
that accidentally samples a* still flips us off-prefix for all later
nodes.

Arm-marginal IS denominator
───────────────────────────
The latent random variable is K = first diverge position ∈ {0,…,D−1,∞}.
Define D+1 "arms":

    arm k (k<D): forces target at positions 0..k−1, samples π at
                 position k, samples π at all positions > k
                 (including off-prefix tail).
    arm D:       forces target at every prefix position 0..D−1.

Per-arm probabilities:

    P(arm k, k<D) = (∏_{d<k} τ(d)) · (1 − τ(k))
    P(arm D)      = ∏_{d<D} τ(d)            (Σ over k = 1)

Walking the realized trajectory we maintain a length-(D+1) reach
vector s.  At each observable prefix position d with realized action a
and natural probability π_d(a):

    for k ∈ 0..d:     s[k] *= π_d(a)             (arm k diverged ≤ d, samples)
    for k ∈ d+1..D:   s[k] *= 1{a == target_d}   (arm k still forcing)

At off-prefix nodes (hidden chance, beyond match history, or playout):

    s[:] *= π(a)                                  (all arms identical here)

Terminal IS denominator:

    l(z) = Σ_k P(arm k) · s[k]                    (true marginal)

This is the per-iter mixture marginal under the v4 sampling
distribution, generalizing OOS's δ·s1 + (1−δ)·s2 to D+1 scenarios.
For D=1 with τ(0)=ρ, l reduces exactly to OOS_dρ.

Why per-step "marginal q" was wrong (sticky-by-coin + per-step
pooling gives a biased pooling at D≥2): under sticky-by-coin the
force arm and diverge arm have *different* downstream sampling
distributions (force arm continues forcing, diverge arm samples).
A per-step marginal at d=0 like τ + (1−τ)·π pretends the two arms
behave the same downstream — they don't.  The IS estimator stayed
unbiased only because the per-step product happens to cancel out in
expectation, but per-iter recorded l varied wildly between F and D
modes (≥4× for D=2 on Kuhn), inflating Var(W) and Var(ΔR).

Why this is different from v3
─────────────────────────────
v3 (observable-node) at each in-prefix site computes
    q(a*) = τ,   q(a) = (1 − τ) · π(a) / (1 − π(a*))   for a ≠ a*
which redistributes (1 − τ) only over non-target actions and
multiplies a per-step q (no arm pooling).  That is biased at D=1
(OOS marginal would credit π(a*) to the target action) and
catastrophic at D≥2 (no scenario tracking at all).

──────────────────────────────────────────────────────────────────────
Pluggable observability
──────────────────────────────────────────────────────────────────────

An ObservabilityOracle decides which chance nodes count as prefix
positions (contributing to D and gating the mixture coin).  Player
decisions are always observable.  Built-in oracles:

  - ISTObservability  (default) — outcome distinguishable from any
    alternative in the searching player's information state along
    the realized history.  Same convention as v3 / OOS-IST.
  - EveryChanceObservable — every chance node counts (recovers v2's
    chance treatment under the mixture coin).
  - NoChanceObservable — no chance node counts; decisions only.

Hidden chance nodes (those rejected by the oracle) are sampled from
their natural distribution π in every iteration — never forced, never
counted in D — and contribute identically to all arms.

──────────────────────────────────────────────────────────────────────
(ρ, w) parameterization — unchanged from Depth-LOTR
──────────────────────────────────────────────────────────────────────

Schedule classes (LOTRSchedule, WeightedExplorationSchedule, uniform,
late_linear, early_linear, step) are imported verbatim from
src.depth_lotr.  τ(d, D) signature is unchanged; D counts observable
prefix positions per the active oracle.

Under sticky-by-coin, the realized prefix-survival rate over D coins
is exactly Π_d τ(d), independent of σ_ε.
"""

from __future__ import annotations

import abc
import math

import numpy as np
import pyspiel

from src.depth_lotr import (
    LOTRSchedule, WeightedExplorationSchedule,
    uniform, late_linear, early_linear, step, resolve_schedule,
)

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
# Observability oracles
# ═════════════════════════════════════════════════════════════════════════

class ObservabilityOracle(abc.ABC):
    """Decides which chance nodes contribute to the observable prefix.

    Player decisions are always observable (keyed by
    information_state_string by definition).  This oracle only governs
    chance nodes.
    """

    @abc.abstractmethod
    def is_observable_chance(self, game, state_at_depth, depth, history,
                             n, target_info, player_id) -> bool:
        ...

    @abc.abstractmethod
    def name(self) -> str:
        ...


class ISTObservability(ObservabilityOracle):
    """Information-state-tree observability (default).

    A chance outcome at depth `d` is observable iff every alternative
    outcome that is still consistent with the realized history would
    leave the searching player in a *different* information state at
    the active step.  Matches OOS-IST and v3 / Depth-LOTR.
    """

    def is_observable_chance(self, game, state_at_depth, depth, history,
                             n, target_info, player_id) -> bool:
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
                    alt_info = alt_s.information_state_string(player_id)
                    if alt_info == target_info:
                        return False
                except Exception:
                    pass
        return True

    def name(self) -> str:
        return "IST"


class EveryChanceObservable(ObservabilityOracle):
    """Treat every chance node as observable (v2-like chance treatment)."""

    def is_observable_chance(self, game, state_at_depth, depth, history,
                             n, target_info, player_id) -> bool:
        return True

    def name(self) -> str:
        return "every_chance"


class NoChanceObservable(ObservabilityOracle):
    """Treat no chance node as observable; decisions-only prefix."""

    def is_observable_chance(self, game, state_at_depth, depth, history,
                             n, target_info, player_id) -> bool:
        return False

    def name(self) -> str:
        return "no_chance"


def resolve_observability(name):
    """Reconstruct an oracle from its serialized name string."""
    if name is None or name == "IST":
        return ISTObservability()
    if name == "every_chance":
        return EveryChanceObservable()
    if name == "no_chance":
        return NoChanceObservable()
    raise ValueError(f"Unknown observability oracle: {name}")


# ═════════════════════════════════════════════════════════════════════════
# MixtureLOTRBot
# ═════════════════════════════════════════════════════════════════════════

class MixtureLOTRBot(pyspiel.Bot):
    """Mixture-LOTR bot — per-node OOS-style coin at every observable position."""

    def __init__(self, game, player_id, num_simulations=1000,
                 epsilon=0.4, gamma=0.01, schedule=None,
                 observability=None, seed=None, tracking=False):
        pyspiel.Bot.__init__(self)
        self._game = game
        self._player_id = player_id
        self._num_simulations = num_simulations
        self._epsilon = epsilon
        self._gamma = gamma
        self._schedule = schedule if schedule is not None else uniform(0.5)
        self._observability = (observability if observability is not None
                               else ISTObservability())
        if _HAS_FAST_OPS:
            self._rng = FastRNG(seed if seed is not None else 42)
        else:
            self._rng = np.random.RandomState(seed)

        self._infostates = {}

        self._match_history = []
        # Position index along observable prefix (0..D-1), populated only
        # at decisions and chance nodes the oracle accepts.
        self._depth_to_position = {}
        self._D = 0
        self._arm_probs = np.array([1.0], dtype=np.float64)
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
        self._depth_to_position = {}
        self._D = 0
        self._arm_probs = np.array([1.0], dtype=np.float64)

    # -- Tracking -------------------------------------------------------------

    def _reset_tracking(self):
        self._iter_log_l = []
        self._iter_log_W = []
        self._iter_diverge_depth = []
        self._iter_chance_diverged = []
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

    def _l_from_s(self, s):
        """Compute scalar IS denominator l = Σ_k P(arm_k) · s[k]."""
        if self._D == 0:
            return float(s[0])
        return float(np.dot(self._arm_probs, s))

    def _episode(self, state, update_player):
        s_init = np.ones(self._D + 1, dtype=np.float64)
        x, l_out, u = self._walk(state, update_player,
                                 my_reach=1.0, opp_reach=1.0,
                                 s=s_init, diverged=False)
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

    def _rng_uniform(self):
        """Portable [0,1) draw: FastRNG exposes .random(); RandomState
        exposes .random_sample()."""
        try:
            return float(self._rng.random())
        except (AttributeError, TypeError):
            return float(self._rng.random_sample())

    def _sample_realized_action(self, target_idx, pi, tau, num_actions):
        """Sticky-by-coin per-node coin (realized walk only).

        Flip a coin; if force, play target; else sample π.  Returns
        (sampled_idx, coin_diverge).  s-update is performed by the
        caller using the prefix rule (see module docstring).
        """
        u_coin = self._rng_uniform()
        coin_diverge = (u_coin >= tau)
        if not coin_diverge:
            sampled_idx = target_idx
        else:
            sampled_idx = int(self._rng.choice(num_actions, p=pi))
        return sampled_idx, coin_diverge

    def _update_s_prefix(self, s, d, sampled_idx, target_idx, pi_a):
        """Apply observable-prefix-position s-update in place."""
        # arms 0..d already past their diverge point (or diverging here):
        # contribute π_d(a)
        s[:d + 1] *= pi_a
        # arms d+1..D still forcing at this position; require a == target_d
        if sampled_idx != target_idx:
            s[d + 1:] = 0.0

    def _walk(self, state, update_player, my_reach, opp_reach, s, diverged):
        if state.is_terminal():
            l_leaf = self._l_from_s(s)
            return (1.0, l_leaf, state.player_return(update_player))

        if state.is_chance_node():
            return self._handle_chance(state, update_player,
                                       my_reach, opp_reach, s, diverged)

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
            uniform_p = 1.0 / num_actions

            # Not-in-tree: realized walk samples uniformly without
            # consulting the coin, so all arms (including arm D) assign
            # 1/n to the realized action.  Mirrors OOS exactly.
            new_s = s * uniform_p

            # Tracking only — record divergence if the realized random
            # action missed the prefix target.  Does NOT affect s.
            target_idx = self._get_targeted_action_idx(state, legal_actions)
            if self._tracking and (not diverged) \
                    and depth in self._depth_to_position \
                    and target_idx is not None and action_idx != target_idx:
                d = self._depth_to_position[depth]
                self._cur_prefix_survived = False
                if self._cur_diverge_depth == -1:
                    self._cur_diverge_depth = d

            state.apply_action(legal_actions[action_idx])
            x, l_out, u = self._playout(state, update_player, new_s)
            return (x * uniform_p, l_out, u)

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
            d = self._depth_to_position[depth]
            tau_d = self._schedule.tau(d, self._D)
            sampled_idx, miss = self._sample_realized_action(
                target_idx, sample_policy, tau_d, num_actions)
            new_diverged = miss
            if self._tracking and miss:
                self._cur_prefix_survived = False
                if self._cur_diverge_depth == -1:
                    self._cur_diverge_depth = d
        else:
            sampled_idx = self._rng.choice(num_actions, p=sample_policy)
            new_diverged = diverged
            d = self._depth_to_position.get(depth, None)

        # s-update (applies at observable prefix position regardless of
        # whether the realized walk forced or sampled — the rule depends
        # on the *position*, not the realized arm).
        new_s = s.copy()
        pi_a = float(sample_policy[sampled_idx])
        if depth in self._depth_to_position and target_idx is not None:
            d_pos = self._depth_to_position[depth]
            self._update_s_prefix(new_s, d_pos, sampled_idx, target_idx, pi_a)
        else:
            new_s *= pi_a

        if self._tracking and depth in self._depth_to_position:
            d_pos = self._depth_to_position[depth]
            neg_log_q = -math.log(max(pi_a, 1e-300))
            if d_pos not in self._depth_neg_log_q_sum:
                self._depth_neg_log_q_sum[d_pos] = 0.0
                self._depth_neg_log_q_count[d_pos] = 0
            self._depth_neg_log_q_sum[d_pos] += neg_log_q
            self._depth_neg_log_q_count[d_pos] += 1

        action = legal_actions[sampled_idx]

        if cur_player == update_player:
            new_my_reach = my_reach * policy[sampled_idx]
            new_opp_reach = opp_reach
        else:
            new_my_reach = my_reach
            new_opp_reach = opp_reach * policy[sampled_idx]

        # Marginal IS denominator at this node (for avg_strategy update)
        l_here = self._l_from_s(s)

        state.apply_action(action)
        x, l_out, u = self._walk(state, update_player,
                                 new_my_reach, new_opp_reach,
                                 new_s, new_diverged)

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
                                    policy, opp_reach, l_here, num_actions)
            else:
                for a_idx in range(num_actions):
                    self._infostates[info_key][AVG_POLICY_INDEX][a_idx] += (
                        opp_reach * policy[a_idx] / max(l_here, 1e-30)
                    )

        return (x, l_out, u)

    def _handle_chance(self, state, update_player, my_reach, opp_reach,
                       s, diverged):
        outcomes = state.chance_outcomes()
        actions, probs_list = zip(*outcomes)
        actions = list(actions)
        probs_arr = np.array(probs_list, dtype=np.float64)
        num_actions = len(actions)
        depth = len(state.history())

        in_prefix = (not diverged) and (depth in self._depth_to_position) \
            and (depth < len(self._match_history)) \
            and (self._match_history[depth] in actions)

        if in_prefix:
            target = self._match_history[depth]
            target_idx = actions.index(target)
            d = self._depth_to_position[depth]
            tau_d = self._schedule.tau(d, self._D)

            sampled_idx, miss = self._sample_realized_action(
                target_idx, probs_arr, tau_d, num_actions)
            action = actions[sampled_idx]
            rho = float(probs_arr[sampled_idx])
            new_diverged = miss

            if self._tracking and miss:
                self._cur_prefix_survived = False
                self._cur_chance_diverged = True
                if self._cur_diverge_depth == -1:
                    self._cur_diverge_depth = d
            if self._tracking:
                neg_log_q = -math.log(max(rho, 1e-300))
                if d not in self._depth_neg_log_q_sum:
                    self._depth_neg_log_q_sum[d] = 0.0
                    self._depth_neg_log_q_count[d] = 0
                self._depth_neg_log_q_sum[d] += neg_log_q
                self._depth_neg_log_q_count[d] += 1

            new_s = s.copy()
            self._update_s_prefix(new_s, d, sampled_idx, target_idx, rho)
            state.apply_action(action)
            x, l_out, u = self._walk(state, update_player,
                                     my_reach, rho * opp_reach,
                                     new_s, new_diverged)
            return (rho * x, l_out, u)
        else:
            # Hidden chance, beyond match history, or already diverged:
            # sample from natural π.  All arms multiply by ρ.
            sampled_idx = self._rng.choice(num_actions, p=probs_arr)
            action = actions[sampled_idx]
            rho = float(probs_arr[sampled_idx])
            new_s = s * rho
            # If we *are* at an observable prefix position but already
            # diverged, the s-update should still apply the prefix rule
            # to preserve the per-arm semantics.  (For arms that were
            # still alive when realized walk diverged at K<d, this node
            # is past their diverge point so they sample.  For arm D and
            # arms k>K, the realized action a_d may not equal target_d,
            # zeroing those s[k].)
            if depth in self._depth_to_position \
                    and depth < len(self._match_history) \
                    and self._match_history[depth] in actions:
                d = self._depth_to_position[depth]
                target_idx = actions.index(self._match_history[depth])
                # Re-derive new_s under the prefix rule (overwrite the
                # naive ρ-multiplied version).
                new_s = s.copy()
                self._update_s_prefix(new_s, d, sampled_idx, target_idx, rho)
            state.apply_action(action)
            x, l_out, u = self._walk(state, update_player,
                                     my_reach, rho * opp_reach,
                                     new_s, diverged)
            return (rho * x, l_out, u)

    def _playout(self, state, update_player, s):
        """Random playout to terminal, threading the per-arm reach vector.

        Playout is by definition past the in-tree boundary.  We treat
        every step as off-prefix (multiplies all arms uniformly), which
        matches OOS's playout treatment.
        """
        x = 1.0
        s = s.copy()
        while not state.is_terminal():
            if state.is_chance_node():
                actions, probs = zip(*state.chance_outcomes())
                idx = self._rng.choice(len(actions), p=probs)
                p = float(probs[idx])
                x *= p
                s *= p
                state.apply_action(actions[idx])
            else:
                legal = state.legal_actions()
                n = len(legal)
                action = legal[self._rng.randint(n)]
                p = 1.0 / n
                x *= p
                s *= p
                state.apply_action(action)

        l_leaf = self._l_from_s(s)
        return (x, l_leaf, state.player_return(update_player))

    # -- Targeting helpers ----------------------------------------------------

    def _build_targeting_info(self, current_state):
        """Walk match history once.  Decisions are always observable;
        chance nodes are gated by the configured ObservabilityOracle.

        D = number of observable prefix positions.
        Also precomputes self._arm_probs (length D+1) for arm-marginal IS.
        """
        self._match_history = list(current_state.history())
        self._depth_to_position = {}

        if not self._match_history:
            self._D = 0
            self._arm_probs = np.array([1.0], dtype=np.float64)
            return

        history = self._match_history
        n = len(history)

        target_info = current_state.information_state_string(self._player_id)
        s = self._game.new_initial_state()
        pos = 0
        for d in range(n):
            if s.is_terminal():
                break
            if s.is_chance_node():
                if self._observability.is_observable_chance(
                        self._game, s, d, history, n,
                        target_info, self._player_id):
                    self._depth_to_position[d] = pos
                    pos += 1
                # Hidden chance: skip (no position assigned).
            else:
                self._depth_to_position[d] = pos
                pos += 1
            s.apply_action(history[d])

        self._D = pos

        # Precompute arm probabilities:
        #   P(arm k, k<D) = (∏_{d<k} τ(d)) · (1 − τ(k))
        #   P(arm D)      = ∏_{d<D} τ(d)
        D = self._D
        arm_probs = np.zeros(D + 1, dtype=np.float64)
        if D == 0:
            arm_probs[0] = 1.0
        else:
            taus = np.array([self._schedule.tau(d, D) for d in range(D)],
                            dtype=np.float64)
            running = 1.0
            for k in range(D):
                arm_probs[k] = running * (1.0 - taus[k])
                running *= taus[k]
            arm_probs[D] = running
        self._arm_probs = arm_probs

    def _get_targeted_action_idx(self, state, legal_actions):
        depth = len(state.history())
        if depth < len(self._match_history) and depth in self._depth_to_position:
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
        return MixtureLOTRPolicy(self._game, self._infostates)


class MixtureLOTRPolicy:
    """Adapter so MixtureLOTRBot info states look like an OpenSpiel Policy."""

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


__all__ = [
    "MixtureLOTRBot",
    "MixtureLOTRPolicy",
    "ObservabilityOracle",
    "ISTObservability",
    "EveryChanceObservable",
    "NoChanceObservable",
    "resolve_observability",
    "LOTRSchedule",
    "WeightedExplorationSchedule",
    "uniform",
    "late_linear",
    "early_linear",
    "step",
    "resolve_schedule",
]
