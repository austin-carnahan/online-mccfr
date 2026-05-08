"""Retrospective Sampling — multi-level divergence online MCCFR.

Replaces OOS's binary δ (targeted/untargeted) with a spectrum of D+1
divergence levels, where D is the number of player decisions in the
current match history.

Level k means:
  - Force the first D-k player decisions to follow the match history
  - ε-on-policy for all decisions after the divergence point

    k = 0 → fully targeted (all D player decisions forced)
    k = D → fully untargeted (no decisions forced)

Each iteration samples an active level k from a decay-weighted
distribution P(k) ∝ f(k), but tracks D+1 sample-reach accumulators
simultaneously. The combined importance weight is:

    l = Σ_k w_k · s_k

where w_k = f(k) / Σ_j f(j) and s_k is the cumulative sample
probability under level k's hypothetical sampling scheme.

Chance nodes use IST semantics: observable chance (searching player's
private info) is forced; hidden chance (opponent's private info) is
sampled from the natural distribution. This is consistent across all
divergence levels.

The decay function (LevelWeightFn from src.isgt) controls budget
allocation across divergence levels:
  - LevelExponential(0.5): 50% at k=0, 25% at k=1, ...
  - LevelUniform(): equal budget per level
  - LevelStep(0.01): nearly all budget at k=0 (≈ OOS with high δ)

Architecture:
  RetroBot(pyspiel.Bot)
    ├── IST chance observability (shared with OOS)
    ├── LevelWeightFn — pluggable decay over divergence levels
    ├── _retro_episode() — sample active level, recurse
    ├── _walk() — recursive MCCFR with D+1 sample-reach accumulators
    └── RetroPolicy — adapter for exploitability computation
"""

import numpy as np
import pyspiel

from src.isgt import LevelWeightFn, LevelExponential

# Index constants matching OpenSpiel's mccfr module
REGRET_INDEX = 0
AVG_POLICY_INDEX = 1


class RetroBot(pyspiel.Bot):
    """Retrospective Sampling bot extending pyspiel.Bot.

    Runs multi-level divergence MCCFR simulations when asked to act,
    then returns an action sampled from the current average strategy.
    """

    def __init__(self, game, player_id, num_simulations=1000,
                 epsilon=0.6, gamma=0.01, decay_fn=None, seed=None):
        """Initialize Retrospective Sampling bot.

        Args:
            game: OpenSpiel game instance.
            player_id: Which player this bot controls.
            num_simulations: Iterations per decision point.
            epsilon: Exploration factor for update player sampling.
            gamma: Floor exploration in regret matching.
            decay_fn: LevelWeightFn controlling divergence level budget.
                      Defaults to LevelExponential(0.5).
            seed: Random seed for reproducibility.
        """
        pyspiel.Bot.__init__(self)
        self._game = game
        self._player_id = player_id
        self._num_simulations = num_simulations
        self._epsilon = epsilon
        self._gamma = gamma
        self._decay_fn = decay_fn or LevelExponential(0.5)
        self._rng = np.random.RandomState(seed)

        # Persistent strategy tables: info_key -> [regrets, avg_strategy]
        self._infostates = {}

        # Current match state — rebuilt each step_with_policy
        self._match_history = []       # full action history (chance + player)
        self._chance_forced = {}       # depth -> bool: observable to searching player?
        self._depth_to_decision = {}   # depth -> player decision index (0-based)
        self._D = 0                    # total player decisions in match history
        self._level_weights = np.array([1.0])  # normalized w_k

        self._num_players = game.num_players()

        # Opt-in diagnostics for experiment tracking
        self._diagnostics = False
        self._diag_active_levels = []     # active level per iteration
        self._diag_l_values = []          # realized l per iteration
        self._diag_surviving_s = []       # count of nonzero s_k per iteration
        self._diag_s_vectors = []         # full s_k vector at terminal per iteration
        self._diag_regret_by_level = {}   # level -> list of |Δregret| at active infoset

    # -- pyspiel.Bot interface -------------------------------------------------

    def step(self, state):
        """Run retrospective sampling, return sampled action."""
        _, action = self.step_with_policy(state)
        return action

    def step_with_policy(self, state):
        """Run simulations, return (policy, sampled_action)."""
        self._build_targeting_info(state)

        for sim in range(self._num_simulations):
            update_player = sim % self._num_players
            root = self._game.new_initial_state()
            self._retro_episode(root, update_player)

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
        """Record an action taken in the real game (by another player).

        Syncs from state.history() to capture any chance actions applied
        since the last bot interaction, then appends this player action.
        """
        self._match_history = list(state.history()) + [action]

    def restart(self):
        """Reset per-game state for a new game. Keeps learned strategy."""
        self._match_history = []
        self._chance_forced = {}
        self._depth_to_decision = {}
        self._D = 0
        self._level_weights = np.array([1.0])

    def provides_policy(self):
        return True

    # -- Core Retrospective Sampling algorithm ---------------------------------

    def _retro_episode(self, state, update_player):
        """Run one iteration: sample active level, recurse with D+1 accumulators."""
        D = self._D
        active_level = self._rng.choice(D + 1, p=self._level_weights)

        # D+1 sample reach accumulators, one per divergence level
        s = np.ones(D + 1, dtype=np.float64)

        # Snapshot regret before walk for diagnostics
        if self._diagnostics and update_player == self._player_id:
            info_key = state.information_state_string(self._player_id)
            if info_key in self._infostates:
                regret_before = self._infostates[info_key][REGRET_INDEX].copy()
            else:
                regret_before = None
        else:
            regret_before = None

        x, l, u = self._walk(state, update_player,
                              my_reach=1.0, opp_reach=1.0,
                              s=s, active_level=active_level)

        # Record diagnostics
        if self._diagnostics:
            self._diag_active_levels.append(active_level)
            self._diag_l_values.append(float(l))
            if regret_before is not None:
                info_key = state.information_state_string(self._player_id)
                if info_key in self._infostates:
                    regret_after = self._infostates[info_key][REGRET_INDEX]
                    delta = float(np.abs(regret_after - regret_before).sum())
                else:
                    delta = 0.0
                if active_level not in self._diag_regret_by_level:
                    self._diag_regret_by_level[active_level] = []
                self._diag_regret_by_level[active_level].append(delta)

        return x, l, u

    def _walk(self, state, update_player, my_reach, opp_reach, s, active_level):
        """Recursive walk with multi-level sample-reach tracking.

        Args:
            state: Current game state (modified in-place).
            update_player: Player whose regrets are updated.
            my_reach: Update player's strategy reach probability.
            opp_reach: Opponents' + chance reach probability.
            s: Array of D+1 sample-reach accumulators.
            active_level: The divergence level sampled for this iteration
                          (controls action selection).

        Returns:
            (x, l, u): suffix reach, combined sample probability, utility.
        """
        D = self._D

        # Terminal
        if state.is_terminal():
            l = np.dot(self._level_weights, s)
            if self._diagnostics:
                self._diag_surviving_s.append(int(np.count_nonzero(s)))
                self._diag_s_vectors.append(s.copy())
            return (1.0, l, state.player_return(update_player))

        # Chance node — IST semantics (same for all levels)
        if state.is_chance_node():
            return self._handle_chance(state, update_player,
                                       my_reach, opp_reach, s, active_level)

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
            l_playout = np.dot(self._level_weights, s) / num_actions
            x, l, u = self._playout(state, update_player, l_playout)
            return (x / num_actions, l, u)

        # Regret matching
        policy = self._regret_matching(self._infostates[info_key][REGRET_INDEX])

        # Sampling distribution
        if cur_player == update_player:
            uniform = np.ones(num_actions, dtype=np.float64) / num_actions
            sample_policy = self._epsilon * uniform + (1.0 - self._epsilon) * policy
        else:
            sample_policy = policy.copy()

        # Determine match history target and forced threshold
        target_idx = None
        forced_threshold = 0  # levels 0..forced_threshold-1 force this decision

        if depth in self._depth_to_decision:
            d = self._depth_to_decision[depth]
            forced_threshold = D - d  # levels 0..D-d-1 force this decision
            target = self._match_history[depth]
            if target in legal_actions:
                target_idx = legal_actions.index(target)
            else:
                forced_threshold = 0  # target not legal — no levels can force

        # Action selection: active level determines whether we force or sample
        if active_level < forced_threshold and target_idx is not None:
            sampled_idx = target_idx
            action = legal_actions[target_idx]
        else:
            sampled_idx = self._rng.choice(num_actions, p=sample_policy)
            action = legal_actions[sampled_idx]

        # Update D+1 sample reach accumulators
        new_s = s.copy()
        for k in range(D + 1):
            if k < forced_threshold:
                # Level k forces this decision
                if sampled_idx == target_idx:
                    pass  # s_k unchanged (deterministic match)
                else:
                    new_s[k] = 0.0  # level k's trajectory is dead
            else:
                # Level k treats this decision as free
                new_s[k] *= sample_policy[sampled_idx]

        # Update strategy reach probabilities
        if cur_player == update_player:
            new_my_reach = my_reach * policy[sampled_idx]
            new_opp_reach = opp_reach
        else:
            new_my_reach = my_reach
            new_opp_reach = opp_reach * policy[sampled_idx]

        # Recurse
        state.apply_action(action)
        x, l, u = self._walk(state, update_player,
                             new_my_reach, new_opp_reach,
                             new_s, active_level)

        # Suffix reach
        c = x
        x = x * policy[sampled_idx]

        # Regret and average strategy updates (same as OOS Algorithm 1)
        if cur_player == update_player:
            W = u * opp_reach / max(l, 1e-30)
            for a_idx in range(num_actions):
                if a_idx == sampled_idx:
                    self._infostates[info_key][REGRET_INDEX][a_idx] += (c - x) * W
                else:
                    self._infostates[info_key][REGRET_INDEX][a_idx] += -x * W
        else:
            sample_prob = np.dot(self._level_weights, s)
            for a_idx in range(num_actions):
                self._infostates[info_key][AVG_POLICY_INDEX][a_idx] += (
                    opp_reach * policy[a_idx] / max(sample_prob, 1e-30)
                )

        return (x, l, u)

    def _handle_chance(self, state, update_player, my_reach, opp_reach,
                       s, active_level):
        """Handle chance nodes with IST semantics.

        Observable chance (searching player's private info) is forced;
        hidden chance is sampled naturally. Same across all divergence levels.
        """
        outcomes = state.chance_outcomes()
        chance_probs = dict(outcomes)
        depth = len(state.history())
        in_history = depth < len(self._match_history)

        forced = (in_history
                  and self._chance_forced.get(depth, True)
                  and self._match_history[depth] in chance_probs)
        target_action = self._match_history[depth] if forced else None

        if forced:
            # Observable chance: force the match outcome
            action = target_action
            # s_k unchanged for all k (deterministic, probability 1)
        else:
            # Hidden or post-history chance: sample naturally
            actions, probs = zip(*outcomes)
            action = self._rng.choice(actions, p=probs)
            # All s_k *= chance_prob (same for all levels)
            s = s * chance_probs[action]

        rho = chance_probs[action]
        state.apply_action(action)
        x, l, u = self._walk(state, update_player,
                             my_reach, rho * opp_reach,
                             s, active_level)
        return (rho * x, l, u)

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
        """Sync match history, compute decision map, chance observability, and level weights."""
        self._match_history = list(current_state.history())
        self._chance_forced = {}
        self._depth_to_decision = {}

        if not self._match_history:
            self._D = 0
            self._level_weights = np.array([1.0])
            return

        history = self._match_history
        n = len(history)

        # Forward pass: identify chance observability and player decision depths
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

        # Compute normalized level weights
        D = self._D
        raw = np.array([self._decay_fn.weight(k, D) for k in range(D + 1)],
                       dtype=np.float64)
        total = raw.sum()
        self._level_weights = raw / total if total > 0 else np.ones(D + 1) / (D + 1)

    def _is_chance_observable(self, state_at_depth, depth, history, n,
                              target_info):
        """Test whether a chance outcome is observable to the searching player.

        Swaps the actual chance outcome for alternatives and replays the
        rest of the match. If any alternative reaches the same information
        set, the outcome is hidden (not observable) and need not be forced.
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

    # -- For display/exploitability compatibility -----------------------------

    def average_policy(self):
        """Return a policy-like object for exploitability computation."""
        return RetroPolicy(self._game, self._infostates)

    def enable_diagnostics(self):
        """Turn on per-iteration diagnostics collection."""
        self._diagnostics = True

    def reset_diagnostics(self):
        """Clear all collected diagnostics data."""
        self._diag_active_levels = []
        self._diag_l_values = []
        self._diag_surviving_s = []
        self._diag_s_vectors = []
        self._diag_regret_by_level = {}

    def get_diagnostics(self):
        """Return a dict of all diagnostics data."""
        return {
            "active_levels": self._diag_active_levels,
            "l_values": self._diag_l_values,
            "surviving_s_counts": self._diag_surviving_s,
            "s_vectors": self._diag_s_vectors,
            "regret_by_level": self._diag_regret_by_level,
        }


class RetroPolicy:
    """Adapter to make RetroBot info states look like an OpenSpiel Policy."""

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
