"""Run a random playout for one of the benchmark games.

Usage:
    python run.py play                   # defaults to leduc_poker
    python run.py play leduc_poker
    python run.py play liars_dice
    python run.py play goofspiel
    python run.py play all               # play all three
"""

import sys
import random
import numpy as np

from src.games import load_game, GAME_SPECS


def play_random_game(game_name: str):
    """Run one random playout and print the game trace."""
    game = load_game(game_name)
    state = game.new_initial_state()
    num_players = game.num_players()

    print(f"{'=' * 60}")
    print(f"  {game_name}")
    print(f"  Players: {num_players}  |  Max utility: {game.max_utility()}")
    print(f"{'=' * 60}")
    print()

    move = 0
    while not state.is_terminal():
        if state.is_chance_node():
            outcomes = state.chance_outcomes()
            actions, probs = zip(*outcomes)
            action = np.random.choice(actions, p=probs)
            action_str = state.action_to_string(state.current_player(), action)
            print(f"  Chance  ->  {action_str}")
        else:
            player = state.current_player()
            legal = state.legal_actions()
            action = random.choice(legal)
            action_str = state.action_to_string(player, action)
            info = state.information_state_string(player)
            print(f"  P{player}: {action_str:<12s}  (info: {info})")
            move += 1
        state.apply_action(action)

    returns = state.returns()
    print()
    print(f"  --- Result ({move} moves) ---")
    for p in range(num_players):
        print(f"  Player {p}: {returns[p]:+.1f}")
    print()


def main():
    args = sys.argv[1:]
    if not args:
        args = ["leduc_poker"]

    if "all" in args:
        args = list(GAME_SPECS.keys())

    for name in args:
        if name not in GAME_SPECS:
            print(f"Unknown game '{name}'. Available: {list(GAME_SPECS.keys())}")
            sys.exit(1)
        play_random_game(name)


if __name__ == "__main__":
    main()
