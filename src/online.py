"""Online match runner for pyspiel.Bot agents.

Follows the game loop protocol from OpenSpiel's spiel_bots.h:
    action = bots[current_player].step(state)
    for other players: bot.inform_action(state, player, action)
    state.apply_action(action)

Usage:
    python run.py online leduc_poker oos ismcts 100           # 100 games OOS vs ISMCTS
    python run.py online leduc_poker oos oos 500 --sims 1000  # OOS self-play
"""

import sys
import time

import numpy as np
import pyspiel

from src.games import load_game, GAME_SPECS


def play_match(game, bots, num_games, rng=None):
    """Play num_games between two pyspiel.Bot instances.

    Yields (game_index, returns) where returns is a list of per-player payoffs.
    Handles chance nodes and the InformAction protocol.
    """
    if rng is None:
        rng = np.random.RandomState()

    for g in range(num_games):
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

        yield g, state.returns()


def print_match_results(game_name, bot_names, all_returns, elapsed):
    """Print summary of match results."""
    returns_arr = np.array(all_returns)
    num_games = len(all_returns)

    print(f"\n{'=' * 55}")
    print(f"  {bot_names[0]} vs {bot_names[1]} on {game_name}")
    print(f"  {num_games} games in {elapsed:.1f}s")
    print(f"{'=' * 55}")

    for p in range(returns_arr.shape[1]):
        mean = returns_arr[:, p].mean()
        std = returns_arr[:, p].std()
        ci = 1.96 * std / np.sqrt(num_games)
        print(f"  Player {p} ({bot_names[p]}):  {mean:+.4f} ± {ci:.4f}")

    # Win rate (player 0 perspective)
    p0_wins = (returns_arr[:, 0] > returns_arr[:, 1]).sum()
    p1_wins = (returns_arr[:, 1] > returns_arr[:, 0]).sum()
    draws = num_games - p0_wins - p1_wins
    print(f"\n  P0 wins: {p0_wins}  P1 wins: {p1_wins}  Draws: {draws}")
    print(f"  P0 win rate: {(p0_wins + 0.5 * draws) / num_games:.1%}")
    print()


def main():
    """CLI: python run.py online <game> <bot0> <bot1> [num_games] [--sims N] [--delta D]"""
    args = sys.argv[1:]

    # Parse known flags
    sims = 1000
    delta = 0.9
    positional = []
    i = 0
    while i < len(args):
        if args[i] == "--sims" and i + 1 < len(args):
            sims = int(args[i + 1])
            i += 2
        elif args[i] == "--delta" and i + 1 < len(args):
            delta = float(args[i + 1])
            i += 2
        else:
            positional.append(args[i])
            i += 1

    game_name = positional[0] if len(positional) >= 1 else "leduc_poker"
    bot0_name = positional[1] if len(positional) >= 2 else "oos"
    bot1_name = positional[2] if len(positional) >= 3 else "ismcts"
    num_games = int(positional[3]) if len(positional) >= 4 else 100

    if game_name not in GAME_SPECS:
        print(f"Unknown game '{game_name}'. Available: {list(GAME_SPECS.keys())}")
        sys.exit(1)

    game = load_game(game_name)

    bot_names = [bot0_name, bot1_name]
    bots = [_make_bot(name, game, player_id, sims, delta) for player_id, name in enumerate(bot_names)]

    all_returns = []
    t0 = time.time()
    for g, returns in play_match(game, bots, num_games):
        all_returns.append(returns)
        if (g + 1) % max(1, num_games // 10) == 0:
            print(f"  game {g + 1}/{num_games}...")

    elapsed = time.time() - t0
    print_match_results(game_name, bot_names, all_returns, elapsed)


def _make_bot(name, game, player_id, sims, delta):
    """Factory: create a Bot by short name."""
    if name == "ismcts":
        from src.ismcts import make_ismcts_bot
        return make_ismcts_bot(game, player_id, max_simulations=sims)
    elif name == "oos":
        from src.oos import OOSBot
        return OOSBot(game, player_id, num_simulations=sims, delta=delta)
    elif name == "random":
        from open_spiel.python.bots.policy import PolicyBot
        from open_spiel.python import policy as policy_lib
        return PolicyBot(player_id, np.random, policy_lib.UniformRandomPolicy(game))
    else:
        raise ValueError(f"Unknown bot '{name}'. Available: ismcts, oos, random")
