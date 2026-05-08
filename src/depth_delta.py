"""Depth-δ — per-node graded targeting for online MCCFR.

Replaces OOS's binary per-iteration δ coin flip with a depth-dependent
δ(d) schedule applied independently at each player decision in the match
history prefix.  Once the walk diverges from the match history (any
sampled action ≠ target), all subsequent prefix decisions are free
(ε-on-policy), matching retro's "full counterfactual branch" behavior.

At each undiverted prefix decision d (of D total player decisions):

    σ_d(a) = δ(d) · 𝟙[a = target] + (1 − δ(d)) · σ_ε(a)

The importance weight is a single running product l = Π σ_d(a_d),
always positive — no level death, no s1/s2 mixture.

Key properties:
  - Convergence: guaranteed (every terminal has nonzero probability).
  - Variance: bounded by the product structure; no O((D+1)²) blowup.
  - Exploration: post-divergence free walk gives retro-like counterfactual
    branches without retro's level-mixture variance penalty.

Schedules control how δ(d) varies with depth:
  - LinearSchedule:  δ(d) = (D − d) / (D + 1)  — high targeting early
  - ReverseLinearSchedule:  δ(d) = (d + 1) / (D + 1)  — low targeting early
  - ConstantSchedule(δ₀):  δ(d) = δ₀ for all d
  - ExponentialSchedule(α):  δ(d) = α^(d+1)  — sharp early decay
"""

from __future__ import annotations

import abc

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

# Index constants matching OpenSpiel's mccfr module
REGRET_INDEX = 0
AVG_POLICY_INDEX = 1


# ═════════════════════════════════════════════════════════════════════════
# Delta schedules
# ═════════════════════════════════════════════════════════════════════════

class DeltaSchedule(abc.ABC):
    """Abstract base for depth-dependent δ schedules."""

    @abc.abstractmethod
    def delta(self, d: int, D: int) -> float:
        """Targeting probability at player decision d of D total.

        Args:
            d: 0-based index of this player decision within the prefix.
            D: Total number of player decisions in the match prefix.

        Returns:
            δ ∈ [0, 1].  Higher = more likely to force the target action.
        """

    @abc.abstractmethod
    def name(self) -> str:
        """Short serializable name for config/label output."""


class LinearSchedule(DeltaSchedule):
    """δ(d) = (D − d) / (D + 1).  High targeting early, fading toward current node."""

    def delta(self, d: int, D: int) -> float:
        if D == 0:
            return 0.0
        return (D - d) / (D + 1)

    def name(self) -> str:
        return "linear"


class ReverseLinearSchedule(DeltaSchedule):
    """δ(d) = (d + 1) / (D + 1).  Low targeting early, rising toward current node."""

    def delta(self, d: int, D: int) -> float:
        if D == 0:
            return 0.0
        return (d + 1) / (D + 1)

    def name(self) -> str:
        return "reverse_linear"


class ConstantSchedule(DeltaSchedule):
    """δ(d) = δ₀ for all d.  Closest analog to OOS's flat coin flip."""

    def __init__(self, delta_0: float = 0.5):
        self._delta_0 = delta_0

    def delta(self, d: int, D: int) -> float:
        return self._delta_0

    def name(self) -> str:
        return f"constant(δ={self._delta_0})"


class ExponentialSchedule(DeltaSchedule):
    """δ(d) = α^(d+1).  Sharp decay from root."""

    def __init__(self, alpha: float = 0.5):
        self._alpha = alpha

    def delta(self, d: int, D: int) -> float:
        return self._alpha ** (d + 1)

    def name(self) -> str:
        return f"exp(α={self._alpha})"


# ═════════════════════════════════════════════════════════════════════════
# Schedule resolution (string → object)
# ═════════════════════════════════════════════════════════════════════════

def resolve_schedule(name: str | None) -> DeltaSchedule:
    """Reconstruct a DeltaSchedule from its serialized name string."""
    if name is None:
        return LinearSchedule()
    if name == "linear":
        return LinearSchedule()
    if name == "reverse_linear":
        return ReverseLinearSchedule()
    if name.startswith("constant("):
        # "constant(δ=0.5)"
        val = float(name.split("=")[1].rstrip(")"))
        return ConstantSchedule(val)
    if name.startswith("exp("):
        # "exp(α=0.5)"
        val = float(name.split("=")[1].rstrip(")"))
        return ExponentialSchedule(val)
    raise ValueError(f"Unknown schedule: {name}")


# ═════════════════════════════════════════════════════════════════════════
# DepthDeltaBot
# ═════════════════════════════════════════════════════════════════════════

class DepthDeltaBot(pyspiel.Bot):
    """Depth-δ online MCCFR bot.

    Uses per-node graded targeting with sticky divergence: once the walk
    departs from the match history, all remaining prefix decisions are
    sampled freely (ε-on-policy).
    """

    def __init__(self, game, player_id, num_simulations=1000,
                 epsilon=0.4, gamma=0.01, schedule=None, seed=None,
                 tracking=False):
        """Initialize Depth-δ bot.

        Args:
            game: OpenSpiel game instance.
            player_id: Which player this bot controls.
            num_simulations: Iterations per decision point.
            epsilon: Exploration factor for update player sampling.
            gamma: Floor exploration in regret matching (paper uses 0.01).
            schedule: DeltaSchedule controlling δ(d). Defaults to LinearSchedule.
            seed: Random seed for reproducibility.
            tracking: If True, collect per-iteration and per-infoset diagnostics.
        """
        pyspiel.Bot.__init__(self)
        self._game = game
        self._player_id = player_id
        self._num_simulations = num_simulations
        self._epsilon = epsilon
        self._gamma = gamma
        self._schedule = schedule or LinearSchedule()
        if _HAS_FAST_OPS:
            self._rng = FastRNG(seed if seed is not None else 42)
        else:
            self._rng = np.random.RandomState(seed)

        # Persistent strategy tables: info_key -> [regrets, avg_strategy]
        self._infostates = {}

        # Current match state tracking
        self._match_history = []     # full action history (chance + player)
        self._chance_forced = {}     # depth -> bool: observable to searching player?
        self._depth_to_decision = {} # game-tree depth -> player decision index (0-based)
        self._D = 0                  # total player decisions in match history
        self._num_players = game.num_players()

        # Diagnostics tracking (opt-in)
        self._tracking = tracking
        if tracking:
            self._reset_tracking()

    # -- pyspiel.Bot interface ------------------------------------------------

    def step(self, state):
        """Run depth-δ, return sampled action for current state."""
        _, action = self.step_with_policy(state)
        return action

    def step_with_policy(self, state):
        """Run simulations, return (policy, sampled_action)."""
        self._build_targeting_info(state)

        if self._tracking:
            self._reset_tracking()

        for sim in range(self._num_simulations):
            update_player = sim % self._num_players
            root = self._game.new_initial_state()
            if self._tracking:
                self._cur_diverge_depth = -1  # -1 means stayed on path
                self._cur_sum_abs_regret = 0.0
                self._cur_log_W = None
            self._episode(root, update_player)

        # Extract current policy at this info set
        info_key = state.information_state_string(self._player_id)
        legal_actions = state.legal_actions()
        policy = self._get_average_policy(info_key, len(legal_actions))

        action_probs = list(zip(legal_actions, policy))
        action = self._rng.choice(legal_actions, p=policy)

        # Record our action in match history
        self._match_history.append(action)

        return action_probs, action

    def inform_action(self, state, player_id, action):
        """Record an action taken in the real game (by another player)."""
        self._match_history = list(state.history()) + [action]

    def restart(self):
        """Reset per-game state for a new game. Keeps learned strategy."""
        self._match_history = []
        self._chance_forced = {}
        self._depth_to_decision = {}
        self._D = 0

    def _reset_tracking(self):
        """Initialize/reset all diagnostics accumulators."""
        # Per-iteration scalars (appended each sim)
        self._iter_log_l = []           # log(terminal sample reach)
        self._iter_log_W = []           # log(|W|) at regret update
        self._iter_diverge_depth = []   # decision index at divergence (-1 if stayed on path)
        self._iter_prefix_D = []        # prefix length D for this iteration
        self._iter_sum_abs_regret = []  # total |ΔR| this iteration
        self._iter_utility = []         # terminal utility

        # Per-depth -log q(a) accumulators (indexed by decision depth d)
        self._depth_neg_log_q_sum = {}   # d -> sum of -log(q(a)) at that depth
        self._depth_neg_log_q_count = {} # d -> count of visits at that depth

        # Per-infoset accumulators: info_key -> dict
        self._infostate_stats = {}

    def _ensure_infostate_stats(self, info_key, num_actions):
        """Lazily create per-infoset stats entry."""
        if info_key not in self._infostate_stats:
            self._infostate_stats[info_key] = {
                "visits": 0,
                "sum_W2": 0.0,
                "sum_abs_delta": np.zeros(num_actions, dtype=np.float64),
                "sum_sq_delta": np.zeros(num_actions, dtype=np.float64),
            }

    def get_tracking_summary(self):
        """Return a summary dict of all tracked diagnostics.

        Call after step_with_policy() to get diagnostics for that step.
        """
        if not self._tracking:
            return {}

        n_iters = len(self._iter_log_l)
        if n_iters == 0:
            return {}

        log_l_arr = np.array(self._iter_log_l)
        log_W_arr = np.array(self._iter_log_W) if self._iter_log_W else np.array([0.0])
        div_arr = np.array(self._iter_diverge_depth)
        abs_r_arr = np.array(self._iter_sum_abs_regret)
        util_arr = np.array(self._iter_utility)

        # Compute W from log_W for ESS
        # ESS = (Σ|W|)² / Σ W² — use log-space to avoid overflow
        log_abs_W = log_W_arr  # already log(|W|)
        max_log = log_abs_W.max() if len(log_abs_W) > 0 else 0.0
        # Stable computation: shift by max
        shifted = log_abs_W - max_log
        sum_W = np.exp(shifted).sum()
        sum_W2 = np.exp(2 * shifted).sum()
        ess = (sum_W ** 2) / sum_W2 if sum_W2 > 0 else float(n_iters)
        ess_per_sim = ess / n_iters

        # Per-depth -log q summary
        depth_neg_log_q = {}
        for d in sorted(self._depth_neg_log_q_sum.keys()):
            count = self._depth_neg_log_q_count[d]
            if count > 0:
                depth_neg_log_q[d] = self._depth_neg_log_q_sum[d] / count

        # Per-infoset summary
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

        # Update coverage imbalance
        visit_counts_arr = np.array(visit_counts) if visit_counts else np.array([0])
        total_visits = visit_counts_arr.sum()
        n_infostates = len(visit_counts)

        on_path_count = int((div_arr == -1).sum())

        return {
            # Log-space weight diagnostics
            "log_W_mean": float(log_W_arr.mean()),
            "log_W_std": float(log_W_arr.std()),
            "log_W_median": float(np.median(log_W_arr)),
            "log_W_p95": float(np.percentile(log_W_arr, 95)),
            "log_W_p99": float(np.percentile(log_W_arr, 99)),
            "log_W_max": float(log_W_arr.max()),
            # ESS
            "ess": float(ess),
            "ess_per_sim": float(ess_per_sim),
            # Sample reach (log-space)
            "log_l_mean": float(log_l_arr.mean()),
            "log_l_min": float(log_l_arr.min()),
            "log_l_p5": float(np.percentile(log_l_arr, 5)),
            "log_l_median": float(np.median(log_l_arr)),
            # Legacy (for backward compat with experiment script)
            "mean_l": float(np.exp(log_l_arr).mean()),
            "min_l": float(np.exp(log_l_arr.min())),
            "mean_inv_l_sq": float(np.exp(-2 * log_l_arr).mean()),
            "max_inv_l": float(np.exp(-log_l_arr.min())),
            # Divergence diagnostics
            "on_path_fraction": on_path_count / n_iters,
            "mean_diverge_depth": float(div_arr[div_arr >= 0].mean()) if (div_arr >= 0).any() else -1.0,
            "prefix_D": int(self._D),
            # Per-depth proposal risk (-log q)
            "depth_neg_log_q": depth_neg_log_q,
            # Regret magnitude
            "mean_sum_abs_regret": float(abs_r_arr.mean()),
            "std_sum_abs_regret": float(abs_r_arr.std()),
            # Utility
            "mean_utility": float(util_arr.mean()),
            # Update coverage imbalance
            "n_infostates_updated": n_infostates,
            "visit_count_min": int(visit_counts_arr.min()) if n_infostates > 0 else 0,
            "visit_count_p5": int(np.percentile(visit_counts_arr, 5)) if n_infostates > 0 else 0,
            "visit_count_median": int(np.median(visit_counts_arr)) if n_infostates > 0 else 0,
            "visit_count_max": int(visit_counts_arr.max()) if n_infostates > 0 else 0,
            # Per-infoset (for offline proxy comparison)
            "infostate_stats": infostate_summary,
            "n_iters": n_iters,
        }

    def provides_policy(self):
        return True

    # -- Core algorithm -------------------------------------------------------

    def _episode(self, state, update_player):
        """Run one depth-δ iteration from root."""
        x, l_out, u = self._walk(state, update_player,
                                  my_reach=1.0, opp_reach=1.0,
                                  l=1.0, diverged=False)
        if self._tracking:
            import math
            self._iter_log_l.append(math.log(max(l_out, 1e-300)))
            self._iter_diverge_depth.append(self._cur_diverge_depth)
            self._iter_prefix_D.append(self._D)
            self._iter_sum_abs_regret.append(self._cur_sum_abs_regret)
            self._iter_utility.append(u)
            # log_W is recorded at the regret update site (may be multiple per walk)
            # Use the max |W| from this iteration as the representative
            if self._cur_log_W is not None:
                self._iter_log_W.append(self._cur_log_W)
        return (x, l_out, u)

    def _walk(self, state, update_player, my_reach, opp_reach, l, diverged):
        """Recursive walk from root to terminal.

        Args:
            state: Current game state (modified in-place via apply_action).
            update_player: Player whose regrets are updated this iteration.
            my_reach: Reach probability of the update player's strategy.
            opp_reach: Reach probability of opponents + chance.
            l: Running sample reach product (always positive).
            diverged: Whether we have already left the match history path.

        Returns:
            (x, l, u): suffix reach, sample reach at terminal, utility.
        """
        # Terminal
        if state.is_terminal():
            return (1.0, l, state.player_return(update_player))

        # Chance node — IST semantics
        if state.is_chance_node():
            return self._handle_chance(state, update_player,
                                       my_reach, opp_reach, l, diverged)

        # Decision node
        cur_player = state.current_player()
        info_key = state.information_state_string(cur_player)
        legal_actions = state.legal_actions()
        num_actions = len(legal_actions)
        depth = len(state.history())

        # Look up or create info state
        in_tree = info_key in self._infostates
        if not in_tree:
            self._infostates[info_key] = [
                np.zeros(num_actions, dtype=np.float64),
                np.zeros(num_actions, dtype=np.float64),
            ]
            action_idx = self._rng.randint(num_actions)
            state.apply_action(legal_actions[action_idx])
            l_playout = l / num_actions
            x, l_out, u = self._playout(state, update_player, l_playout)
            return (x / num_actions, l_out, u)

        # Regret matching + sampling distribution
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
                uniform = np.ones(num_actions, dtype=np.float64) / num_actions
                sample_policy = self._epsilon * uniform + (1.0 - self._epsilon) * policy
            else:
                sample_policy = policy.copy()

        # Determine targeting for this decision
        target_idx = self._get_targeted_action_idx(state, legal_actions)
        in_prefix = (not diverged) and (target_idx is not None)

        if in_prefix:
            # Depth-δ blended targeting
            d = self._depth_to_decision[depth]
            delta_d = self._schedule.delta(d, self._D)
            # σ_d(a) = δ(d)·𝟙[a=target] + (1−δ(d))·σ_ε(a)
            blend = sample_policy * (1.0 - delta_d)
            blend[target_idx] += delta_d

            sampled_idx = self._rng.choice(num_actions, p=blend)
            action = legal_actions[sampled_idx]
            new_l = l * blend[sampled_idx]
            new_diverged = (sampled_idx != target_idx)
            # Track first divergence point
            if self._tracking and new_diverged and self._cur_diverge_depth == -1:
                self._cur_diverge_depth = d
            # Track per-depth -log q(a)
            if self._tracking:
                import math
                q_a = blend[sampled_idx]
                neg_log_q = -math.log(max(q_a, 1e-300))
                if d not in self._depth_neg_log_q_sum:
                    self._depth_neg_log_q_sum[d] = 0.0
                    self._depth_neg_log_q_count[d] = 0
                self._depth_neg_log_q_sum[d] += neg_log_q
                self._depth_neg_log_q_count[d] += 1
        else:
            # Free sampling (post-divergence, post-prefix, or no target)
            sampled_idx = self._rng.choice(num_actions, p=sample_policy)
            action = legal_actions[sampled_idx]
            new_l = l * sample_policy[sampled_idx]
            new_diverged = diverged  # stay diverged once diverged
            # Track per-depth -log q for free nodes too (use game-tree depth as key)
            if self._tracking and depth in self._depth_to_decision:
                import math
                d = self._depth_to_decision[depth]
                q_a = sample_policy[sampled_idx]
                neg_log_q = -math.log(max(q_a, 1e-300))
                if d not in self._depth_neg_log_q_sum:
                    self._depth_neg_log_q_sum[d] = 0.0
                    self._depth_neg_log_q_count[d] = 0
                self._depth_neg_log_q_sum[d] += neg_log_q
                self._depth_neg_log_q_count[d] += 1

        # Update strategy reach probabilities
        if cur_player == update_player:
            new_my_reach = my_reach * policy[sampled_idx]
            new_opp_reach = opp_reach
        else:
            new_my_reach = my_reach
            new_opp_reach = opp_reach * policy[sampled_idx]

        # Recurse
        state.apply_action(action)
        x, l_out, u = self._walk(state, update_player,
                                 new_my_reach, new_opp_reach,
                                 new_l, new_diverged)

        # Suffix reach
        c = x
        x = x * policy[sampled_idx]

        # Regret and average strategy updates (same as OOS Algorithm 1)
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

            # Per-infoset and per-iteration tracking
            if self._tracking:
                import math
                abs_W = abs(W)
                cur_log_W = math.log(max(abs_W, 1e-300))
                # Keep the max log|W| across all update sites in this iteration
                if self._cur_log_W is None or cur_log_W > self._cur_log_W:
                    self._cur_log_W = cur_log_W

                self._ensure_infostate_stats(info_key, num_actions)
                stats = self._infostate_stats[info_key]
                stats["visits"] += 1
                stats["sum_W2"] += W * W
                # Regret deltas per action
                delta_sampled = (c - x) * W
                stats["sum_abs_delta"][sampled_idx] += abs(delta_sampled)
                stats["sum_sq_delta"][sampled_idx] += delta_sampled * delta_sampled
                iter_abs = abs(delta_sampled)
                for a_idx in range(num_actions):
                    if a_idx != sampled_idx:
                        d_a = -x * W
                        stats["sum_abs_delta"][a_idx] += abs(d_a)
                        stats["sum_sq_delta"][a_idx] += d_a * d_a
                        iter_abs += abs(d_a)
                self._cur_sum_abs_regret += iter_abs
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
        """Handle chance nodes with IST targeting.

        Observable chance (searching player's private info) is forced when
        on the match history path. Hidden chance is sampled naturally.
        """
        outcomes = state.chance_outcomes()
        chance_probs = dict(outcomes)
        depth = len(state.history())
        in_history = depth < len(self._match_history)

        forced = (not diverged
                  and in_history
                  and self._chance_forced.get(depth, True)
                  and self._match_history[depth] in chance_probs)

        if forced:
            # Observable chance on the match path: deterministic
            action = self._match_history[depth]
            # l unchanged (deterministic, probability = 1 under targeting)
            # but untargeted probability is the chance prob
            # For depth-δ, chance forcing is deterministic — the sample prob
            # under the blended scheme is effectively 1.0 for observable chance.
            # We still need to account for the natural chance probability in
            # opp_reach (for correct counterfactual value weighting).
            rho = chance_probs[action]
            state.apply_action(action)
            x, l_out, u = self._walk(state, update_player,
                                     my_reach, rho * opp_reach,
                                     l, diverged)
            return (rho * x, l_out, u)
        else:
            # Sample from natural distribution
            actions, probs = zip(*outcomes)
            action = self._rng.choice(actions, p=probs)
            rho = chance_probs[action]

            # If we were on the match path but this was hidden chance,
            # check if we've now diverged from the match history
            new_diverged = diverged
            if not diverged and in_history:
                if action != self._match_history[depth]:
                    new_diverged = True

            state.apply_action(action)
            x, l_out, u = self._walk(state, update_player,
                                     my_reach, rho * opp_reach,
                                     l * rho, new_diverged)
            return (rho * x, l_out, u)

    def _playout(self, state, update_player, sample_reach):
        """Random playout to terminal (outside the tree)."""
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
        """Sync match history, compute decision map and chance observability."""
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
        """Test whether a chance outcome is observable to the searching player."""
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
        """Return the index of the match history action at this depth, or None."""
        depth = len(state.history())
        if depth < len(self._match_history) and depth in self._depth_to_decision:
            target = self._match_history[depth]
            if target in legal_actions:
                return legal_actions.index(target)
        return None

    # -- Strategy computation -------------------------------------------------

    def _regret_matching(self, regrets):
        """Convert cumulative regrets to a policy via regret matching."""
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
        """Get the average strategy at an info set (for action selection)."""
        if info_key not in self._infostates:
            return np.ones(num_actions, dtype=np.float64) / num_actions
        avg = self._infostates[info_key][AVG_POLICY_INDEX]
        total = avg.sum()
        if total > 0:
            return avg / total
        return np.ones(num_actions, dtype=np.float64) / num_actions

    # -- For exploitability computation ---------------------------------------

    def average_policy(self):
        """Return a policy-like object for exploitability computation."""
        return DepthDeltaPolicy(self._game, self._infostates)


class DepthDeltaPolicy:
    """Adapter to make DepthDeltaBot info states look like an OpenSpiel Policy."""

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
