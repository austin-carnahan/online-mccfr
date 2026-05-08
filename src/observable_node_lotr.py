"""Observable-Node LOTR — τ-blend Local Online Target-Rate at every
*observable* prefix node.

Refines Every-Node LOTR (which τ-blended at all prefix positions
including hidden chance) by restoring the IST observability gate from
Depth-LOTR.  τ-blend now fires at:

  - every player decision on the prefix
  - every chance node on the prefix that is *observable* to the
    searching player (information_state_string distinguishes the actual
    outcome from any alternative)

Hidden chance nodes (e.g. the opponent's private card in Kuhn / Leduc)
are sampled from the natural distribution π in every iteration —
exactly as OOS-IST does.  Sticky divergence still triggers on any
miss at an observable prefix position (chance or decision).

Apples-to-apples comparison with OOS-IST:

  OOS-IST per-iteration coin:
    - With prob δ: force every observable prefix position to target,
      hidden chance ~ natural π.
    - With prob 1−δ: σ_ε at decisions, natural π at all chance.

  v3 Observable-Node LOTR with step(ρ, d=0):
    - At each observable prefix position: independent τ-blend coin.
      d=0 fires at rate τ(0)=1−ρ, d>0 at rate τ=1.
    - Hidden chance ~ natural π.

Both target the *same* set of nodes with the *same* expected on-path
rate (1−ρ ≡ δ).  The only structural difference is q-form: OOS uses a
per-iteration mixture coin, v3 uses per-node product-form sampling.
Any exploitability gap between v3_step_r05 and oos_d05 isolates the
q-form effect.

──────────────────────────────────────────────────────────────────────
(ρ, w) parameterization — unchanged from Depth-LOTR
──────────────────────────────────────────────────────────────────────

Schedule classes (LOTRSchedule, WeightedExplorationSchedule, uniform,
late_linear, early_linear, step) are imported verbatim from
src.depth_lotr.  Their tau(d, D) signature is unchanged but D now
counts *observable* prefix positions (decisions + observable chance).
"""

from __future__ import annotations

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
# ObservableNodeLOTRBot
# ═════════════════════════════════════════════════════════════════════════

class ObservableNodeLOTRBot(pyspiel.Bot):
    """Observable-node LOTR bot — τ-blend at every observable prefix node."""

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
        # Position index along observable prefix (0..D-1); populated only
        # at decision nodes and chance nodes that pass the observability
        # test.  Hidden chance is *absent* from this dict.
        self._depth_to_position = {}
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
        self._depth_to_position = {}
        self._D = 0

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
            # Unified apples-to-apples metric.
            "prefix_survival_fraction": float(prefix_arr.mean()),
            "mean_diverge_depth": float(div_arr[div_arr >= 0].mean()) if (div_arr >= 0).any() else -1.0,
            # D = number of *observable* prefix positions.
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
            if self._tracking and not diverged and depth in self._depth_to_position:
                t_idx = self._get_targeted_action_idx(state, legal_actions)
                if t_idx is not None and action_idx != t_idx:
                    self._cur_prefix_survived = False
                    if self._cur_diverge_depth == -1:
                        self._cur_diverge_depth = self._depth_to_position[depth]
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
            d = self._depth_to_position[depth]
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
            if self._tracking and depth in self._depth_to_position:
                d = self._depth_to_position[depth]
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
        actions, probs_list = zip(*outcomes)
        actions = list(actions)
        probs_arr = np.array(probs_list, dtype=np.float64)
        num_actions = len(actions)
        depth = len(state.history())

        # Observable prefix chance node iff present in _depth_to_position
        # AND in match history AND target outcome is in current outcomes.
        in_prefix = (not diverged) and (depth in self._depth_to_position) \
            and (depth < len(self._match_history)) \
            and (self._match_history[depth] in actions)

        if in_prefix:
            target = self._match_history[depth]
            target_idx = actions.index(target)
            d = self._depth_to_position[depth]
            tau_d = self._schedule.tau(d, self._D)

            sigma_target = probs_arr[target_idx]
            non_target_mass = 1.0 - sigma_target

            blend = np.zeros(num_actions, dtype=np.float64)
            blend[target_idx] = tau_d

            if non_target_mass > 1e-12:
                scale = (1.0 - tau_d) / non_target_mass
                for a in range(num_actions):
                    if a != target_idx:
                        blend[a] = probs_arr[a] * scale
            elif num_actions > 1:
                share = (1.0 - tau_d) / (num_actions - 1)
                for a in range(num_actions):
                    if a != target_idx:
                        blend[a] = share

            sampled_idx = self._rng.choice(num_actions, p=blend)
            action = actions[sampled_idx]
            rho = probs_arr[sampled_idx]
            new_diverged = (sampled_idx != target_idx)

            if self._tracking and new_diverged:
                self._cur_prefix_survived = False
                self._cur_chance_diverged = True
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

            new_l = l * blend[sampled_idx]
            state.apply_action(action)
            x, l_out, u = self._walk(state, update_player,
                                     my_reach, rho * opp_reach,
                                     new_l, new_diverged)
            return (rho * x, l_out, u)
        else:
            # Hidden chance, off-prefix chance, or already diverged:
            # sample from natural π (same as OOS-IST hidden chance).
            sampled_idx = self._rng.choice(num_actions, p=probs_arr)
            action = actions[sampled_idx]
            rho = probs_arr[sampled_idx]
            state.apply_action(action)
            x, l_out, u = self._walk(state, update_player,
                                     my_reach, rho * opp_reach,
                                     l * rho, diverged)
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
        """Walk match history once, populate _depth_to_position at every
        decision and every *observable* chance node.  Hidden chance is
        deliberately omitted so it falls through the off-prefix branch
        in _handle_chance and gets sampled from natural π.

        D = number of observable prefix positions.
        """
        self._match_history = list(current_state.history())
        self._depth_to_position = {}

        if not self._match_history:
            self._D = 0
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
                if self._is_chance_observable(s, d, history, n, target_info):
                    self._depth_to_position[d] = pos
                    pos += 1
                # Hidden chance: skip (no position assigned).
            else:
                self._depth_to_position[d] = pos
                pos += 1
            s.apply_action(history[d])

        self._D = pos

    def _is_chance_observable(self, state_at_depth, depth, history, n,
                              target_info):
        """Return True iff the actual outcome at this chance node is
        distinguishable from every alternative outcome in the searching
        player's information state at the current step.  Same test as
        Depth-LOTR / OOS-IST.
        """
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
        return ObservableNodeLOTRPolicy(self._game, self._infostates)


class ObservableNodeLOTRPolicy:
    """Adapter so ObservableNodeLOTRBot info states look like an OpenSpiel Policy."""

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
    "ObservableNodeLOTRBot",
    "ObservableNodeLOTRPolicy",
    "LOTRSchedule",
    "WeightedExplorationSchedule",
    "uniform",
    "late_linear",
    "early_linear",
    "step",
    "resolve_schedule",
]
