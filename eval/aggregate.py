"""Aggregate exploitability evaluation method.

Implements the multi-match aggregate strategy method from Section 4.2 of
Lisý, Lanctot, Bowling 2015. Plays N matches of a search bot against a
random opponent, accumulates strategy data across matches into a global
policy, then computes exploitability of the aggregated strategy.
"""

import numpy as np
import pyspiel

from open_spiel.python import policy as policy_lib
from open_spiel.python.bots.policy import PolicyBot

from src.games import load_game
from src.metrics import exploitability
from src.oos import OOSBot, AVG_POLICY_INDEX


def aggregate_exploitability(game, bot_factory, num_matches=500, seed=None):
    """Compute exploitability via aggregate multi-match method.

    Args:
        game: OpenSpiel game instance.
        bot_factory: Callable(player_id) -> pyspiel.Bot. Creates a fresh bot
            for each match. The bot must support step_with_policy() or have
            extractable strategy data.
        num_matches: Number of matches to play vs random opponent.
        seed: Random seed.

    Returns:
        float: Exploitability of the aggregated strategy.
    """
    rng = np.random.RandomState(seed)

    # Global strategy accumulator: info_key -> {action: cumulative_weight}
    global_strategy = [{}, {}]  # one per player

    for match_idx in range(num_matches):
        # Alternate which player the search bot controls
        search_player = match_idx % 2
        random_player = 1 - search_player

        # Create fresh bot and random opponent
        search_bot = bot_factory(search_player)
        random_policy = policy_lib.UniformRandomPolicy(game)
        random_bot = PolicyBot(random_player, rng, random_policy)

        bots = [None, None]
        bots[search_player] = search_bot
        bots[random_player] = random_bot

        # Play one game
        state = game.new_initial_state()
        for bot in bots:
            bot.restart()

        while not state.is_terminal():
            if state.is_chance_node():
                outcomes, probs = zip(*state.chance_outcomes())
                action = rng.choice(outcomes, p=probs)
                state.apply_action(action)
            else:
                player = state.current_player()
                action = bots[player].step(state)
                for p, bot in enumerate(bots):
                    if p != player:
                        bot.inform_action(state, player, action)
                state.apply_action(action)

        # Extract strategy data from search bot and merge into global
        _merge_bot_strategy(search_bot, search_player, global_strategy)

    # Build a policy from the global accumulator and compute exploitability
    agg_policy = AggregatePolicy(game, global_strategy)
    return exploitability(game, agg_policy)


def _merge_bot_strategy(bot, player_id, global_strategy):
    """Merge a bot's strategy data into the global accumulator.

    Handles both OOSBot (s_I tables) and ISMCTSBot (visit counts).
    """
    if isinstance(bot, OOSBot):
        _merge_oos(bot, player_id, global_strategy)
    else:
        _merge_ismcts(bot, player_id, global_strategy)


def _merge_oos(bot, player_id, global_strategy):
    """Merge OOS average strategy tables into global accumulator.

    OOS stores infostates for ALL players (both the update player and
    opponents). Since info state strings are player-specific (e.g. include
    [Observer: X]), we merge into all player buckets — the lookup in
    AggregatePolicy will only match the correct player's entries.
    """
    for info_key, tables in bot._infostates.items():
        avg_strat = tables[AVG_POLICY_INDEX]
        if avg_strat.sum() == 0:
            continue
        for p in range(len(global_strategy)):
            if info_key not in global_strategy[p]:
                global_strategy[p][info_key] = np.zeros_like(avg_strat)
            global_strategy[p][info_key] += avg_strat


def _merge_ismcts(bot, player_id, global_strategy):
    """Merge ISMCTS visit counts into global accumulator.

    ISMCTS nodes are keyed by (player, info_key). We route each node
    to the correct player's bucket in global_strategy rather than
    filtering to only the search player.
    """
    for (node_player, info_key), node in bot._nodes.items():
        if node.total_visits <= 0:
            continue
        for action, child in node.child_info.items():
            if info_key not in global_strategy[node_player]:
                global_strategy[node_player][info_key] = {}
            entry = global_strategy[node_player][info_key]
            if isinstance(entry, dict):
                entry[action] = entry.get(action, 0.0) + child.visits


class AggregatePolicy:
    """Policy built from accumulated strategy data across matches.

    At info sets with accumulated data, returns the normalized weights.
    At unvisited info sets, returns uniform random (fixed random action).
    """

    def __init__(self, game, global_strategy):
        self._game = game
        self._global_strategy = global_strategy

    def action_probabilities(self, state, player_id=None):
        if player_id is None:
            player_id = state.current_player()

        info_key = state.information_state_string(player_id)
        legal_actions = state.legal_actions()
        n = len(legal_actions)

        if info_key in self._global_strategy[player_id]:
            data = self._global_strategy[player_id][info_key]

            if isinstance(data, np.ndarray):
                # OOS-style: numpy array indexed by position
                total = data.sum()
                if total > 0:
                    probs = data / total
                    return dict(zip(legal_actions, probs))
            elif isinstance(data, dict):
                # ISMCTS-style: {action: count}
                total = sum(data.values())
                if total > 0:
                    return {a: data.get(a, 0.0) / total for a in legal_actions}

        # Unvisited: uniform
        return {a: 1.0 / n for a in legal_actions}
