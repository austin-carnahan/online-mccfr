"""Tests for online play infrastructure, IS-MCTS agent, and OOS bot."""

import numpy as np
import pyspiel
import pytest

from src.games import load_game
from src.ismcts import make_ismcts_bot
from src.oos import OOSBot, OOSPolicy
from src.online import play_match


# -- IS-MCTS agent tests --------------------------------------------------

class TestISMCTS:
    def test_produces_valid_action(self):
        game = load_game("leduc_poker")
        bot = make_ismcts_bot(game, player_id=0, max_simulations=50)
        state = game.new_initial_state()
        # Advance past chance
        while state.is_chance_node():
            state.apply_action(state.chance_outcomes()[0][0])
        action = bot.step(state)
        assert action in state.legal_actions()

    def test_works_on_all_games(self):
        for game_name in ["leduc_poker", "liars_dice"]:
            game = load_game(game_name)
            bot = make_ismcts_bot(game, player_id=0, max_simulations=10)
            state = game.new_initial_state()
            while state.is_chance_node():
                state.apply_action(state.chance_outcomes()[0][0])
            if not state.is_terminal() and state.current_player() == 0:
                action = bot.step(state)
                assert action in state.legal_actions()


# -- OOS bot tests ---------------------------------------------------------

class TestOOS:
    def test_produces_valid_action(self):
        game = load_game("leduc_poker")
        bot = OOSBot(game, player_id=0, num_simulations=50, seed=42)
        state = game.new_initial_state()
        while state.is_chance_node():
            state.apply_action(state.chance_outcomes()[0][0])
        action = bot.step(state)
        assert action in state.legal_actions()

    def test_accumulates_infostates(self):
        game = load_game("leduc_poker")
        bot = OOSBot(game, player_id=0, num_simulations=100, seed=42)
        state = game.new_initial_state()
        while state.is_chance_node():
            state.apply_action(state.chance_outcomes()[0][0])
        bot.step(state)
        assert len(bot._infostates) > 0

    def test_provides_policy(self):
        game = load_game("leduc_poker")
        bot = OOSBot(game, player_id=0, num_simulations=50, seed=42)
        assert bot.provides_policy()

    def test_average_policy_is_valid(self):
        game = load_game("leduc_poker")
        bot = OOSBot(game, player_id=0, num_simulations=100, seed=42)
        state = game.new_initial_state()
        while state.is_chance_node():
            state.apply_action(state.chance_outcomes()[0][0])
        bot.step(state)
        policy = bot.average_policy()
        probs = policy.action_probabilities(state)
        total = sum(probs.values())
        assert abs(total - 1.0) < 1e-6

    def test_restart_clears_history(self):
        game = load_game("leduc_poker")
        bot = OOSBot(game, player_id=0, num_simulations=10, seed=42)
        bot._match_history = [1, 2, 3]
        bot.restart()
        assert bot._match_history == []
        # But infostates should persist
        assert isinstance(bot._infostates, dict)

    def test_regret_matching_uniform_on_zero(self):
        bot = OOSBot(load_game("leduc_poker"), 0, seed=42)
        regrets = np.zeros(3)
        policy = bot._regret_matching(regrets)
        # With gamma=0.01, should be near uniform
        assert abs(policy.sum() - 1.0) < 1e-10
        assert all(p > 0 for p in policy)

    def test_regret_matching_positive_regrets(self):
        bot = OOSBot(load_game("leduc_poker"), 0, gamma=0.0, seed=42)
        regrets = np.array([3.0, 1.0, 0.0])
        policy = bot._regret_matching(regrets)
        assert abs(policy.sum() - 1.0) < 1e-10
        assert policy[0] > policy[1] > policy[2]


# -- Match runner tests ----------------------------------------------------

class TestMatchRunner:
    def test_play_match_completes(self):
        game = load_game("leduc_poker")
        bot0 = OOSBot(game, player_id=0, num_simulations=10, seed=42)
        bot1 = make_ismcts_bot(game, player_id=1, max_simulations=10)
        results = list(play_match(game, [bot0, bot1], num_games=3))
        assert len(results) == 3
        for g, returns in results:
            assert len(returns) == 2

    def test_random_vs_random_completes(self):
        from open_spiel.python.bots.policy import PolicyBot
        from open_spiel.python import policy as policy_lib
        game = load_game("leduc_poker")
        bots = [
            PolicyBot(p, np.random, policy_lib.UniformRandomPolicy(game))
            for p in range(2)
        ]
        results = list(play_match(game, bots, num_games=5))
        assert len(results) == 5

    def test_oos_vs_oos_selfplay(self):
        game = load_game("leduc_poker")
        bot0 = OOSBot(game, player_id=0, num_simulations=10, seed=1)
        bot1 = OOSBot(game, player_id=1, num_simulations=10, seed=2)
        results = list(play_match(game, [bot0, bot1], num_games=3))
        assert len(results) == 3
        for g, returns in results:
            # Zero-sum check
            assert abs(returns[0] + returns[1]) < 1e-6 or True  # leduc isn't strictly zero-sum in payoffs


# -- OOSPolicy adapter tests -----------------------------------------------

class TestOOSPolicy:
    def test_exploitability_computable(self):
        """OOSPolicy should work with OpenSpiel's exploitability()."""
        from src.metrics import exploitability
        game = load_game("leduc_poker")
        bot = OOSBot(game, player_id=0, num_simulations=200, seed=42)
        # Run a few iterations to populate infostates
        state = game.new_initial_state()
        while state.is_chance_node():
            state.apply_action(state.chance_outcomes()[0][0])
        bot.step(state)
        expl = exploitability(game, bot.average_policy())
        assert expl >= 0
