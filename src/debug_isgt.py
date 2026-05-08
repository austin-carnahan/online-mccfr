"""Debug script: run ISGT on Kuhn Poker with each decay function.

1. Root convergence: run ISGT iterations and measure exploitability
2. Online match: ISGT vs random, printing debug info on first move
3. Compare all four decay functions

Usage:
    python -m src.debug_isgt
"""

import pyspiel
import numpy as np
from src.metrics import exploitability
from src.isgt import (ISGTBot, ExponentialDecay, PolynomialDecay,
                      LinearDecay, StepFunction)
from src.online import play_match


def root_convergence(game, weight_fn, num_iters=5000, seed=42):
    """Run ISGT from root with uniform weights, measure exploitability."""
    print(f"\n  {weight_fn.name()}")

    bot = ISGTBot(game, player_id=0, num_simulations=1,
                  delta=0.9, epsilon=0.6, gamma=0.01,
                  level_weight_fn=weight_fn, seed=seed)

    uniform_weights = {iid: 1.0 for iid in bot._iig.infosets}
    checkpoints = [500, 2000, 5000]

    for i in range(1, num_iters + 1):
        update_player = i % game.num_players()
        root = game.new_initial_state()
        bot._walk(root, update_player, uniform_weights,
                  my_reach=1.0, opp_reach=1.0, sample_reach=1.0)

        if i in checkpoints:
            policy = bot.average_policy()
            expl = exploitability(game, policy)
            print(f"    {i:5d} iters -> exploit = {expl:.6f}")


def online_match(game, weight_fn, num_games=50, sims=200, seed=42):
    """Run ISGT vs random, print win rate and first-move debug."""
    print(f"\n  {weight_fn.name()}  ({num_games} games, {sims} sims/move)")

    bot = ISGTBot(game, player_id=0, num_simulations=sims,
                  delta=0.9, epsilon=0.6, gamma=0.01,
                  level_weight_fn=weight_fn, seed=seed)

    from open_spiel.python.bots.policy import PolicyBot
    from open_spiel.python import policy as policy_lib
    rand_bot = PolicyBot(1, np.random, policy_lib.UniformRandomPolicy(game))

    bots = [bot, rand_bot]
    rng = np.random.RandomState(seed)

    all_returns = []
    debug_printed = False
    for g, returns in play_match(game, bots, num_games, rng=rng):
        all_returns.append(returns)
        if not debug_printed:
            bot.print_debug()
            debug_printed = True

    returns_arr = np.array(all_returns)
    p0_mean = returns_arr[:, 0].mean()
    p0_wins = (returns_arr[:, 0] > 0).sum()
    print(f"    P0 (ISGT) avg return: {p0_mean:+.4f}  "
          f"wins: {p0_wins}/{num_games}")


def main():
    game = pyspiel.load_game("kuhn_poker")

    decay_fns = [
        ExponentialDecay(0.5),
        PolynomialDecay(2.0),
        LinearDecay(4),
        StepFunction(0.01),
    ]

    # 1. Root convergence (uniform weights -- validates base engine)
    print("=" * 60)
    print("Root convergence (uniform IIG weights)")
    print("=" * 60)
    for wf in decay_fns:
        root_convergence(game, wf)

    # 2. Online matches (IIG-targeted -- exercises actual targeting)
    print()
    print("=" * 60)
    print("Online matches: ISGT vs Random")
    print("=" * 60)
    for wf in decay_fns:
        online_match(game, wf)


if __name__ == "__main__":
    main()
