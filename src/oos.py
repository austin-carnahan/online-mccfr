"""Online Outcome Sampling (OOS) — Lisý, Lanctot, Bowling 2015.

Implements Algorithm 1 from the paper as a pyspiel.Bot. The bot runs
OOS iterations from the game root at each decision, accumulating regrets
across decisions and games. Uses IST (Information Set Targeting) or
PST (Public State Targeting) to bias sampling toward the current match state.

Key differences from offline OS-MCCFR:
  1. Incremental tree building — info sets created on first visit
  2. Targeted sampling — δ blend of targeted vs untargeted trajectories
  3. Stateful across decisions — regrets persist within a game (and optionally across)

"""

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


class OOSBot(pyspiel.Bot):
    """Online Outcome Sampling bot extending pyspiel.Bot.

    Runs root-to-terminal OOS simulations when asked to act, then returns
    an action sampled from the current regret-matched strategy.
    """

    def __init__(self, game, player_id, num_simulations=1000, delta=0.9,
                 epsilon=0.6, gamma=0.01, targeting="IST", seed=None,
                 tracking=False):
        """Initialize OOS bot.

        Args:
            game: OpenSpiel game instance.
            player_id: Which player this bot controls.
            num_simulations: OOS iterations per decision point.
            delta: Targeting probability (0 = pure MCCFR, 1 = full targeting).
            epsilon: Exploration factor for update player sampling.
            gamma: Floor exploration in regret matching (paper uses 0.01).
            targeting: "IST" or "PST".
            seed: Random seed for reproducibility.
            tracking: If True, collect per-iteration and per-infoset diagnostics.
        """
        pyspiel.Bot.__init__(self)
        self._game = game
        self._player_id = player_id
        self._num_simulations = num_simulations
        self._delta = delta
        self._epsilon = epsilon
        self._gamma = gamma
        self._targeting = targeting
        if _HAS_FAST_OPS:
            self._rng = FastRNG(seed if seed is not None else 42)
        else:
            self._rng = np.random.RandomState(seed)

        # Persistent strategy tables: info_key -> [regrets, avg_strategy]
        self._infostates = {}

        # Current match state tracking
        self._match_history = []  # full action history (chance + player)
        self._chance_forced = {}  # depth -> bool: observable to searching player?
        self._num_players = game.num_players()

        # Diagnostics tracking (opt-in)
        self._tracking = tracking
        if tracking:
            self._reset_tracking()

    # -- pyspiel.Bot interface -------------------------------------------------

    def step(self, state):
        """Run OOS, return sampled action for current state."""
        policy, action = self.step_with_policy(state)
        return action

    def step_with_policy(self, state):
        """Run OOS simulations, return (policy, sampled_action)."""
        # Sync match history from actual game state (includes chance actions)
        self._build_targeting_info(state)

        if self._tracking:
            self._reset_tracking()

        # Run OOS iterations (alternating update player)
        for sim in range(self._num_simulations):
            update_player = sim % self._num_players
            root = self._game.new_initial_state()
            if self._tracking:
                self._cur_sum_abs_regret = 0.0
                self._cur_sum_sq_regret = 0.0
                self._cur_log_W = None
                # Unified prefix-survival flag (matches DepthLOTRBot).
                # Becomes False if any realized prefix decision or any
                # observable prefix chance outcome misses the match-history
                # target.  Hidden chance is excluded.  Instrumentation only;
                # the OOS sampling/regret update logic is unchanged.
                self._cur_prefix_survived = True
            self._oos_episode(root, update_player)

        # Extract current policy at this info set
        info_key = state.information_state_string(self._player_id)
        legal_actions = state.legal_actions()
        policy = self._get_average_policy(info_key, len(legal_actions))

        # Build (action, prob) list and sample
        action_probs = list(zip(legal_actions, policy))
        action = self._rng.choice(legal_actions, p=policy)

        # Record our own action in match history
        self._match_history.append(action)

        return action_probs, action

    def inform_action(self, state, player_id, action):
        """Record an action taken in the real game (by another player).

        Syncs from state.history() to capture any chance actions applied
        since the last bot interaction, then appends this player action.
        """
        self._match_history = list(state.history()) + [action]

    def restart(self):
        """Reset per-game state for a new game. Keeps learned strategy."""
        self._match_history = []
        self._chance_forced = {}

    def _reset_tracking(self):
        """Initialize/reset all diagnostics accumulators."""
        self._iter_log_l = []            # log(terminal sample prob)
        self._iter_log_W = []            # log(|W|) at regret update
        self._iter_targeted = []         # whether iteration was targeted
        # Realized prefix-survival per episode (apples-to-apples with LOTR).
        self._iter_prefix_survived = []
        self._iter_sum_abs_regret = []   # total |ΔR| this iteration
        self._iter_sum_sq_regret = []    # total Σ (ΔR)² this iteration
        self._iter_utility = []          # terminal utility

        # Per-depth -log q(a) accumulators
        self._depth_neg_log_q_sum = {}
        self._depth_neg_log_q_count = {}

        # Per-infoset accumulators
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
        """Return a summary dict of tracked diagnostics for this step."""
        if not self._tracking:
            return {}

        n_iters = len(self._iter_log_l)
        if n_iters == 0:
            return {}

        log_l_arr = np.array(self._iter_log_l)
        log_W_arr = np.array(self._iter_log_W) if self._iter_log_W else np.array([0.0])
        targeted_arr = np.array(self._iter_targeted)
        prefix_arr = (np.array(self._iter_prefix_survived)
                      if self._iter_prefix_survived else np.array([True]))
        abs_r_arr = np.array(self._iter_sum_abs_regret)
        sq_r_arr = np.array(self._iter_sum_sq_regret) if self._iter_sum_sq_regret else np.array([0.0])
        util_arr = np.array(self._iter_utility)

        # ESS from log-space weights
        log_abs_W = log_W_arr
        max_log = log_abs_W.max() if len(log_abs_W) > 0 else 0.0
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
        n_infostates = len(visit_counts)
        targeted_frac = float(targeted_arr.mean()) if len(targeted_arr) > 0 else 0.0

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
            # OOS-specific
            "targeted_fraction": targeted_frac,
            # Unified apples-to-apples metric (same definition in DepthLOTRBot):
            # fraction of episodes whose realized actions matched the
            # match-history target at every player decision and every
            # observable chance node on the prefix.  Differs from
            # targeted_fraction because untargeted iterations can
            # still land on the target by chance.
            "prefix_survival_fraction": float(prefix_arr.mean()),
            # Per-depth proposal risk (-log q)
            "depth_neg_log_q": depth_neg_log_q,
            # Regret magnitude
            "mean_sum_abs_regret": float(abs_r_arr.mean()),
            "std_sum_abs_regret": float(abs_r_arr.std()),
            "mean_sum_sq_regret": float(sq_r_arr.mean()),
            "p95_sum_sq_regret": float(np.percentile(sq_r_arr, 95)),
            "max_sum_sq_regret": float(sq_r_arr.max()),
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

    # -- Core OOS algorithm (Algorithm 1 from paper) --------------------------

    def _oos_episode(self, state, update_player):
        """Run one OOS iteration from the given state."""
        # Per-iteration scenario flip (Algorithm 1: "Before each iteration,
        # a scenario is decided")
        self._is_targeted_iter = self._rng.random() < self._delta
        x, l, u = self._walk(state, update_player,
                              my_reach=1.0, opp_reach=1.0,
                              s1=1.0, s2=1.0)
        if self._tracking:
            import math
            self._iter_log_l.append(math.log(max(l, 1e-300)))
            self._iter_targeted.append(self._is_targeted_iter)
            self._iter_prefix_survived.append(self._cur_prefix_survived)
            self._iter_sum_abs_regret.append(self._cur_sum_abs_regret)
            self._iter_sum_sq_regret.append(self._cur_sum_sq_regret)
            self._iter_utility.append(u)
            if self._cur_log_W is not None:
                self._iter_log_W.append(self._cur_log_W)
        return (x, l, u)

    def _walk(self, state, update_player, my_reach, opp_reach, s1, s2):
        """Recursive OOS walk from root to terminal.

        Args:
            state: Current game state (modified in-place via apply_action).
            update_player: Player whose regrets are updated this iteration.
            my_reach: Reach probability of the update player's strategy.
            opp_reach: Reach probability of opponents + chance.
            s1: Sample reach probability under targeted scenario.
            s2: Sample reach probability under untargeted scenario.

        Returns:
            (x, l, u): suffix reach, combined sample probability, utility.
        """
        # Terminal — Algorithm 1 line 2-3
        if state.is_terminal():
            l = self._delta * s1 + (1.0 - self._delta) * s2
            return (1.0, l, state.player_return(update_player))

        # Chance node — Algorithm 1 line 4-7
        if state.is_chance_node():
            return self._handle_chance(state, update_player,
                                       my_reach, opp_reach, s1, s2)

        # Decision node — Algorithm 1 line 8 onwards
        cur_player = state.current_player()
        info_key = state.information_state_string(cur_player)
        legal_actions = state.legal_actions()
        num_actions = len(legal_actions)

        # Look up or create info state
        in_tree = info_key in self._infostates
        if not in_tree:
            # Algorithm 1 line 10-13: add to memory, playout
            self._infostates[info_key] = [
                np.zeros(num_actions, dtype=np.float64),
                np.zeros(num_actions, dtype=np.float64),
            ]
            action_idx = self._rng.randint(num_actions)
            # Unified prefix-survival tracking (instrumentation only).
            # Realized non-target action at a not-in-tree prefix node
            # constitutes a divergence from the match-history target.
            if self._tracking:
                t_idx_local = self._get_targeted_action_idx(state, legal_actions)
                if t_idx_local is not None and action_idx != t_idx_local:
                    self._cur_prefix_survived = False
            state.apply_action(legal_actions[action_idx])
            l_playout = (self._delta * s1 + (1.0 - self._delta) * s2) / num_actions
            x, l, u = self._playout(state, update_player, l_playout)
            return (x / num_actions, l, u)

        # Algorithm 1 line 15: policy via regret matching
        # Algorithm 1 line 9: compute sampling distribution
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

        # Targeting: find action consistent with real match history
        target_idx = self._get_targeted_action_idx(state, legal_actions)

        # Sample action — targeted iterations deterministically follow
        # match history (Algorithm 1 line 9: restrict to actions in Z_sub)
        if self._is_targeted_iter and target_idx is not None:
            sampled_idx = target_idx
            action = legal_actions[target_idx]
        else:
            sampled_idx = self._rng.choice(num_actions, p=sample_policy)
            action = legal_actions[sampled_idx]

        # Unified prefix-survival tracking (instrumentation only).  In
        # untargeted iterations on the prefix, sampling can still land on
        # the target by chance (the realized trajectory is what matters).
        if self._tracking and target_idx is not None and sampled_idx != target_idx:
            self._cur_prefix_survived = False

        # Track per-depth -log q(a)
        if self._tracking:
            import math
            depth = len(state.history())
            # Effective q(a): in targeted iter on-target, q=1; else sample_policy
            if self._is_targeted_iter and target_idx is not None and sampled_idx == target_idx:
                q_a = 1.0
            else:
                q_a = sample_policy[sampled_idx]
            neg_log_q = -math.log(max(q_a, 1e-300))
            if depth not in self._depth_neg_log_q_sum:
                self._depth_neg_log_q_sum[depth] = 0.0
                self._depth_neg_log_q_count[depth] = 0
            self._depth_neg_log_q_sum[depth] += neg_log_q
            self._depth_neg_log_q_count[depth] += 1

        # Update sample reaches (s1, s2) — Algorithm 1 line 9
        new_s2 = s2 * sample_policy[sampled_idx]
        if target_idx is not None and sampled_idx == target_idx:
            # Deterministic in targeted scenario: s1 unchanged
            # (Φ(a)/sum = 1 when only one action is in Z_sub)
            new_s1 = s1
        elif target_idx is not None:
            new_s1 = 0.0  # left the targeted subgame
        else:
            new_s1 = s1 * sample_policy[sampled_idx]

        # Update reach probabilities — Algorithm 1 line 16-17
        if cur_player == update_player:
            new_my_reach = my_reach * policy[sampled_idx]
            new_opp_reach = opp_reach
        else:
            new_my_reach = my_reach
            new_opp_reach = opp_reach * policy[sampled_idx]

        # Recurse — Algorithm 1 line 18
        state.apply_action(action)
        x, l, u = self._walk(state, update_player,
                             new_my_reach, new_opp_reach,
                             new_s1, new_s2)

        # Algorithm 1 line 19-20
        c = x
        x = x * policy[sampled_idx]

        # Algorithm 1 line 21-29: update regrets and average strategy
        if cur_player == update_player:
            # Line 23: W = u * π_{-i} / l
            W = u * opp_reach / max(l, 1e-30)
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
            # Line 29: update average strategy at opponent nodes
            sample_prob = self._delta * s1 + (1.0 - self._delta) * s2
            if _HAS_FAST_OPS:
                update_avg_strategy(self._infostates[info_key][AVG_POLICY_INDEX],
                                    policy, opp_reach, sample_prob, num_actions)
            else:
                for a_idx in range(num_actions):
                    self._infostates[info_key][AVG_POLICY_INDEX][a_idx] += (
                        opp_reach * policy[a_idx] / max(sample_prob, 1e-30)
                    )

        # Line 30
        return (x, l, u)

    def _handle_chance(self, state, update_player, my_reach, opp_reach, s1, s2):
        """Handle chance nodes with IST targeting.

        For correct IST, only chance outcomes *observable* to the searching
        player are forced during targeted iterations.  Hidden chance (e.g.
        opponent's private cards) is sampled from the natural distribution
        even in the targeted scenario.
        """
        outcomes = state.chance_outcomes()
        chance_probs = dict(outcomes)
        depth = len(state.history())
        in_history = depth < len(self._match_history)

        # Only force chance nodes that are observable to the searching player
        forced = (in_history
                  and self._chance_forced.get(depth, True)
                  and self._match_history[depth] in chance_probs)
        target_action = self._match_history[depth] if forced else None

        if forced and self._is_targeted_iter:
            # Observable chance in targeted mode: force the match outcome
            action = target_action
            rho1 = 1.0
        else:
            # Sample from natural distribution
            actions, probs = zip(*outcomes)
            action = self._rng.choice(actions, p=probs)
            if forced:
                # Track what the targeted scenario would have done
                rho1 = 1.0 if action == target_action else 0.0
            else:
                rho1 = chance_probs[action]

        rho2 = chance_probs[action]
        # Unified prefix-survival tracking: observable chance on the
        # prefix can miss the target only in untargeted iterations
        # (forced+targeted always plays target_action).
        if self._tracking and forced and action != target_action:
            self._cur_prefix_survived = False
        state.apply_action(action)
        x, l, u = self._walk(state, update_player,
                             my_reach, rho2 * opp_reach,
                             rho1 * s1, rho2 * s2)
        return (rho2 * x, l, u)

    def _playout(self, state, update_player, sample_reach):
        """Random playout to terminal (outside the tree).

        Returns (x, l, u) matching the walk return signature.
        """
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
        """Sync match history from game state and precompute IST chance forcing.

        Called at the start of step_with_policy.  Captures the full action
        history (chance + player) from the actual game state, then replays
        it to determine which chance nodes are *observable* to the searching
        player (and thus must be forced during targeted iterations).
        """
        self._match_history = list(current_state.history())
        self._chance_forced = {}

        if not self._match_history:
            return

        target_info = current_state.information_state_string(self._player_id)
        history = self._match_history
        n = len(history)

        # Single forward pass: test each chance node for observability
        s = self._game.new_initial_state()
        for d in range(n):
            if s.is_chance_node():
                self._chance_forced[d] = self._is_chance_observable(
                    s, d, history, n, target_info
                )
            s.apply_action(history[d])

    def _is_chance_observable(self, state_at_depth, depth, history, n,
                              target_info):
        """Test whether a chance outcome is observable to the searching player.

        Tries swapping the actual chance outcome for an alternative and
        replaying the rest of the match history.  If any alternative reaches
        the *same* information set, the outcome is not observable (hidden
        private info like the opponent's card) and need not be forced.
        """
        actual = history[depth]
        for alt, _ in state_at_depth.chance_outcomes():
            if alt == actual:
                continue
            # Replay the rest of the match with this alternative outcome
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
                        return False  # not observable — alternative reaches same info set
                except Exception:
                    pass
        return True  # observable — no alternative reaches the same info set

    def _get_targeted_action_idx(self, state, legal_actions):
        """Return the index of the match history action at this depth, or None.

        Now works correctly because match_history is synced from
        state.history() (includes chance actions), so depth indexing aligns.
        """
        depth = len(state.history())
        if depth < len(self._match_history):
            target = self._match_history[depth]
            if target in legal_actions:
                return legal_actions.index(target)
        return None

    # -- Strategy computation -------------------------------------------------

    def _regret_matching(self, regrets):
        """Convert cumulative regrets to a policy via regret matching.

        Uses γ-floor exploration as in the paper (default γ=0.01).
        """
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

    # -- For display/exploitability compatibility -----------------------------

    def average_policy(self):
        """Return a policy-like object for exploitability computation."""
        return OOSPolicy(self._game, self._infostates)


class OOSPolicy:
    """Adapter to make OOS info states look like an OpenSpiel Policy.

    Needed for exploitability() computation via OpenSpiel's exploitability module.
    """

    def __init__(self, game, infostates):
        self._game = game
        self._infostates = infostates

    def action_probabilities(self, state, player_id=None):
        """Return action probabilities at the given state."""
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
