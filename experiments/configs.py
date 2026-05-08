"""Experiment configurations: algorithm factories, game lists, parameter grids."""

import numpy as np

from src.games import load_game, GAME_SPECS
from src.oos import OOSBot, OOSPolicy
from src.ismcts import make_ismcts_bot
from src.isgt import ISGTBot, ExponentialDecay
from src.retrospective import RetroBot


# ---------------------------------------------------------------------------
# Algorithm registry
# ---------------------------------------------------------------------------

def make_oos(game, num_sims, seed=None, **kwargs):
    """Create an OOS bot for both players. Returns (bots, policy_extractor).

    The policy_extractor is a callable: () -> policy suitable for exploitability().
    """
    delta = kwargs.get("delta", 0.9)
    epsilon = kwargs.get("epsilon", 0.6)
    gamma = kwargs.get("gamma", 0.01)

    bot = OOSBot(game, player_id=0, num_simulations=num_sims,
                 delta=delta, epsilon=epsilon, gamma=gamma, seed=seed)
    return bot, lambda: bot.average_policy()


def make_ismcts_uct(game, num_sims, seed=None, **kwargs):
    """Create an ISMCTS-UCT bot. Returns (bot, policy_extractor).

    policy_extractor builds a tabular policy from ISMCTS node visit counts
    (empirical action frequencies).
    """
    rng = np.random.RandomState(seed)
    bot = make_ismcts_bot(game, player_id=0, max_simulations=num_sims,
                          random_state=rng)
    return bot, lambda: ISMCTSPolicy(game, bot)


def make_isgt(game, num_sims, seed=None, **kwargs):
    """Create an ISGT bot. Returns (bot, policy_extractor)."""
    epsilon = kwargs.get("epsilon", 0.6)
    gamma = kwargs.get("gamma", 0.01)
    bot = ISGTBot(game, player_id=0, num_simulations=num_sims,
                  epsilon=epsilon, gamma=gamma,
                  level_weight_fn=ExponentialDecay(0.5),
                  seed=seed)
    return bot, lambda: bot.average_policy()


def make_retro(game, num_sims, seed=None, **kwargs):
    """Create a Retrospective Sampling bot. Returns (bot, policy_extractor).

    kwargs:
        epsilon: Exploration factor (default 0.6).
        gamma: Regret matching floor (default 0.01).
        decay_fn: LevelWeightFn instance (default ExponentialDecay(0.5)).
        diagnostics: If True, enable per-iteration diagnostics tracking.
    """
    epsilon = kwargs.get("epsilon", 0.6)
    gamma = kwargs.get("gamma", 0.01)
    decay_fn = kwargs.get("decay_fn", ExponentialDecay(0.5))
    diagnostics = kwargs.get("diagnostics", False)

    bot = RetroBot(game, player_id=0, num_simulations=num_sims,
                   epsilon=epsilon, gamma=gamma, decay_fn=decay_fn,
                   seed=seed)
    if diagnostics:
        bot.enable_diagnostics()
    return bot, lambda: bot.average_policy()


class ISMCTSPolicy:
    """Adapter: extract a full policy from ISMCTS tree node visit counts.

    For root convergence, after run_search() has been called, each node in
    bot._nodes has child_info with visit counts. We normalize these to get
    empirical action frequencies (the strategy ISMCTS would play).
    """

    def __init__(self, game, bot):
        self._game = game
        self._bot = bot

    def action_probabilities(self, state, player_id=None):
        if player_id is None:
            player_id = state.current_player()

        key = self._bot.get_state_key(state)
        legal_actions = state.legal_actions()
        n = len(legal_actions)

        if key in self._bot._nodes:
            node = self._bot._nodes[key]
            if node.total_visits > 0:
                probs = {}
                for action in legal_actions:
                    if action in node.child_info:
                        probs[action] = node.child_info[action].visits / node.total_visits
                    else:
                        probs[action] = 0.0
                # Normalize in case not all actions were visited
                total = sum(probs.values())
                if total > 0:
                    return {a: p / total for a, p in probs.items()}
        # Fallback: uniform
        return {a: 1.0 / n for a in legal_actions}


# ---------------------------------------------------------------------------
# Registry mapping short names to factories
# ---------------------------------------------------------------------------

ALGORITHM_REGISTRY = {
    "oos": make_oos,
    "ismcts": make_ismcts_uct,
    "isgt": make_isgt,
    "retro": make_retro,
}

# Default algorithm configs for experiments
DEFAULT_ALGO_CONFIGS = {
    "oos": {"delta": 0.0, "epsilon": 0.6, "gamma": 0.01},
    "ismcts": {},
    "isgt": {"epsilon": 0.6, "gamma": 0.01},
    "retro": {"epsilon": 0.6, "gamma": 0.01},
}

# Games to use in experiments (all from GAME_SPECS)
EXPERIMENT_GAMES = [g for g in GAME_SPECS if g != "kuhn_poker"]

# Standard simulation budget checkpoints for root convergence
ROOT_CHECKPOINTS = [100, 500, 1000, 2000, 5000, 10000, 20000, 50000, 100000, 200000]

# Wall-clock time checkpoints (seconds) for root convergence
TIME_CHECKPOINTS = [1, 2, 5, 10, 20, 30, 60, 90, 120, 180, 240, 300, 420, 600]

# Sims-per-move budgets for aggregate exploitability
AGGREGATE_SIM_BUDGETS = [100, 250, 500]

# Number of matches for aggregate method
AGGREGATE_NUM_MATCHES = 500
