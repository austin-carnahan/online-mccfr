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

# Index constants matching OpenSpiel's mccfr module
REGRET_INDEX = 0
AVG_POLICY_INDEX = 1


class OOSBot(pyspiel.Bot):
    """Online Outcome Sampling bot extending pyspiel.Bot.

    Runs root-to-terminal OOS simulations when asked to act, then returns
    an action sampled from the current regret-matched strategy.
    """

    def __init__(self, game, player_id, num_simulations=1000, delta=0.9,
                 epsilon=0.6, gamma=0.01, targeting="IST", seed=None):
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
        """
        pyspiel.Bot.__init__(self)
        self._game = game
        self._player_id = player_id
        self._num_simulations = num_simulations
        self._delta = delta
        self._epsilon = epsilon
        self._gamma = gamma
        self._targeting = targeting
        self._rng = np.random.RandomState(seed)

        # Persistent strategy tables: info_key -> [regrets, avg_strategy]
        self._infostates = {}

        # Current match state tracking
        self._match_history = []  # action ints taken in the real game so far
        self._num_players = game.num_players()

    # -- pyspiel.Bot interface -------------------------------------------------

    def step(self, state):
        """Run OOS, return sampled action for current state."""
        policy, action = self.step_with_policy(state)
        return action

    def step_with_policy(self, state):
        """Run OOS simulations, return (policy, sampled_action)."""
        # Run OOS iterations (alternating update player)
        for sim in range(self._num_simulations):
            update_player = sim % self._num_players
            root = self._game.new_initial_state()
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
        """Record an action taken in the real game (by another player or chance)."""
        self._match_history.append(action)

    def restart(self):
        """Reset per-game state for a new game. Keeps learned strategy."""
        self._match_history = []

    def provides_policy(self):
        return True

    # -- Core OOS algorithm (Algorithm 1 from paper) --------------------------

    def _oos_episode(self, state, update_player):
        """Run one OOS iteration from the given state."""
        return self._walk(state, update_player,
                          my_reach=1.0, opp_reach=1.0,
                          s1=1.0, s2=1.0)

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
            state.apply_action(legal_actions[action_idx])
            l_playout = (self._delta * s1 + (1.0 - self._delta) * s2) / num_actions
            x, l, u = self._playout(state, update_player, l_playout)
            return (x / num_actions, l, u)

        # Algorithm 1 line 15: policy via regret matching
        policy = self._regret_matching(self._infostates[info_key][REGRET_INDEX])

        # Algorithm 1 line 9: compute sampling distribution
        if cur_player == update_player:
            uniform = np.ones(num_actions, dtype=np.float64) / num_actions
            sample_policy = self._epsilon * uniform + (1.0 - self._epsilon) * policy
        else:
            sample_policy = policy.copy()

        # Targeting: find action consistent with real match history
        target_idx = self._get_targeted_action_idx(state, legal_actions)

        # Compute s1 factor for targeted action
        if target_idx is not None:
            target_s1_factor = sample_policy[target_idx] / max(
                sample_policy[target_idx], 1e-30)
        else:
            target_s1_factor = None

        # Sample action
        sampled_idx = self._rng.choice(num_actions, p=sample_policy)
        action = legal_actions[sampled_idx]

        # Update sample reaches (s1, s2) — Algorithm 1 line 9
        new_s2 = s2 * sample_policy[sampled_idx]
        if target_idx is not None and sampled_idx == target_idx:
            new_s1 = s1 * target_s1_factor
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
            for a_idx in range(num_actions):
                if a_idx == sampled_idx:
                    # Line 25: r_I[a] += (c - x) * W
                    self._infostates[info_key][REGRET_INDEX][a_idx] += (c - x) * W
                else:
                    # Line 27: r_I[a] += -x * W
                    self._infostates[info_key][REGRET_INDEX][a_idx] += -x * W
        else:
            # Line 29: update average strategy at opponent nodes
            sample_prob = self._delta * s1 + (1.0 - self._delta) * s2
            for a_idx in range(num_actions):
                self._infostates[info_key][AVG_POLICY_INDEX][a_idx] += (
                    opp_reach * policy[a_idx] / max(sample_prob, 1e-30)
                )

        # Line 30
        return (x, l, u)

    def _handle_chance(self, state, update_player, my_reach, opp_reach, s1, s2):
        """Handle chance nodes with targeting.

        At chance nodes in the real game history, targeting forces the
        same chance outcome. Outside the history, sample normally.
        If the targeted action is not a valid chance outcome (because
        the simulation diverged earlier), fall back to untargeted.
        """
        outcomes = state.chance_outcomes()
        chance_probs = dict(outcomes)
        target_action = self._get_targeted_chance_action(state)

        # Invalidate target if it's not a valid outcome in this state
        if target_action is not None and target_action not in chance_probs:
            target_action = None

        if target_action is not None and self._rng.random() < self._delta:
            # Targeted: use the real game's chance outcome
            action = target_action
            rho1 = 1.0  # deterministic in targeted scenario
        else:
            # Untargeted: sample from natural distribution
            actions, probs = zip(*outcomes)
            action = self._rng.choice(actions, p=probs)
            if target_action is not None and action == target_action:
                rho1 = 1.0
            elif target_action is not None:
                rho1 = 0.0
            else:
                rho1 = chance_probs[action]

        rho2 = chance_probs[action]
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

    def _get_targeted_chance_action(self, state):
        """If this chance node is within the real match history, return the
        action actually taken. Otherwise None."""
        depth = len(state.history())
        if depth < len(self._match_history):
            return self._match_history[depth]
        return None

    def _get_targeted_action_idx(self, state, legal_actions):
        """For IST: return the index into legal_actions of the action that
        matches the real match history at this depth, or None."""
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
