"""Benchmark: fast_ops ON vs OFF for OOS and DD bots."""
import time
import numpy as np
import pyspiel


def benchmark_bot(BotClass, game, sims, label, num_games=10, **kwargs):
    """Time a bot over multiple games, return sims/second."""
    bot = BotClass(game, player_id=0, num_simulations=sims, **kwargs)
    total_sims = 0
    total_time = 0.0

    for g in range(num_games):
        state = game.new_initial_state()
        bot.restart()
        while not state.is_terminal():
            if state.is_chance_node():
                outcomes = state.chance_outcomes()
                actions, probs = zip(*outcomes)
                action = np.random.choice(actions, p=probs)
                state.apply_action(action)
            elif state.current_player() == 0:
                t0 = time.perf_counter()
                action = bot.step(state)
                total_time += time.perf_counter() - t0
                total_sims += sims
                state.apply_action(action)
            else:
                action = np.random.choice(state.legal_actions())
                bot.inform_action(state, state.current_player(), action)
                state.apply_action(action)

    sims_per_sec = total_sims / total_time
    print(f"  {label}: {sims_per_sec:,.0f} sims/s ({total_time:.2f}s total)")
    return sims_per_sec


def main():
    from src.oos import OOSBot, _HAS_FAST_OPS
    from src.depth_delta import DepthDeltaBot, ConstantSchedule

    assert _HAS_FAST_OPS, "fast_ops not loaded — can't benchmark"

    game = pyspiel.load_game("leduc_poker")
    sims = 5000
    num_games = 20

    print(f"Leduc Poker benchmark — {sims} sims/move, {num_games} games")
    print(f"(fast_ops enabled)")
    print()

    s_oos = benchmark_bot(OOSBot, game, sims, "OOS (fast)",
                          num_games=num_games,
                          delta=0.5, epsilon=0.4, gamma=0.01, seed=42)
    s_dd = benchmark_bot(DepthDeltaBot, game, sims, "DD  (fast)",
                         num_games=num_games,
                         epsilon=0.4, gamma=0.01,
                         schedule=ConstantSchedule(0.5), seed=42)

    # Now benchmark with fast_ops disabled (monkey-patch)
    print()
    print("Disabling fast_ops (pure Python fallback)...")
    import src.oos as oos_mod
    import src.depth_delta as dd_mod
    oos_mod._HAS_FAST_OPS = False
    dd_mod._HAS_FAST_OPS = False

    # Need to also revert RNG to numpy
    class OOSBotSlow(OOSBot):
        def __init__(self, *args, **kwargs):
            seed = kwargs.get('seed', 42)
            super().__init__(*args, **kwargs)
            self._rng = np.random.RandomState(seed)

    class DDBotSlow(DepthDeltaBot):
        def __init__(self, *args, **kwargs):
            seed = kwargs.get('seed', 42)
            super().__init__(*args, **kwargs)
            self._rng = np.random.RandomState(seed)

    s_oos_slow = benchmark_bot(OOSBotSlow, game, sims, "OOS (slow)",
                               num_games=num_games,
                               delta=0.5, epsilon=0.4, gamma=0.01, seed=42)
    s_dd_slow = benchmark_bot(DDBotSlow, game, sims, "DD  (slow)",
                              num_games=num_games,
                              epsilon=0.4, gamma=0.01,
                              schedule=ConstantSchedule(0.5), seed=42)

    print()
    print(f"Speedup OOS: {s_oos/s_oos_slow:.2f}x")
    print(f"Speedup DD:  {s_dd/s_dd_slow:.2f}x")


if __name__ == "__main__":
    main()
