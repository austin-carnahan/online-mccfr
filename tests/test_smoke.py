"""Smoke tests — verify OpenSpiel loads games and basic API works."""

import random
import pyspiel
import numpy as np
from src.games import load_game, GAME_SPECS


def test_games_load():
    """All benchmark games load without error."""
    for name in GAME_SPECS:
        game = load_game(name)
        assert game is not None


def _random_playout(game):
    """Run one random playout to terminal, return (state, returns)."""
    state = game.new_initial_state()
    while not state.is_terminal():
        if state.is_chance_node():
            outcomes = state.chance_outcomes()
            actions, probs = zip(*outcomes)
            action = np.random.choice(actions, p=probs)
        else:
            action = random.choice(state.legal_actions())
        state.apply_action(action)
    return state, state.returns()


def test_random_playout_leduc():
    """Leduc poker playout reaches terminal and is zero-sum."""
    game = load_game("leduc_poker")
    state, returns = _random_playout(game)
    assert state.is_terminal()
    assert len(returns) == 2
    assert returns[0] == -returns[1]


def test_random_playout_liars_dice():
    """Liar's dice playout reaches terminal and is zero-sum."""
    game = load_game("liars_dice")
    state, returns = _random_playout(game)
    assert state.is_terminal()
    assert len(returns) == 2
    assert returns[0] == -returns[1]


def test_random_playout_goofspiel():
    """Goofspiel (turn-based) playout reaches terminal."""
    game = load_game("goofspiel")
    state, returns = _random_playout(game)
    assert state.is_terminal()
    assert len(returns) == 2


def test_exploitability_computable():
    """Exploitability can be computed for a uniform-random policy."""
    from open_spiel.python.policy import UniformRandomPolicy
    from src.metrics import exploitability

    game = load_game("leduc_poker")
    policy = UniformRandomPolicy(game)
    expl = exploitability(game, policy)
    assert expl > 0
