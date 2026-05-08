"""Integration test: verify OOS and DD bots work with fast_ops enabled."""
import time
import pyspiel
from src.oos import OOSBot, _HAS_FAST_OPS as oos_fast
from src.depth_delta import DepthDeltaBot, _HAS_FAST_OPS as dd_fast
from src.depth_delta import ConstantSchedule


def run_bot_test(bot_name, bot, game, num_games=5):
    """Play a few games against a random opponent, return avg time/move."""
    times = []
    for g in range(num_games):
        state = game.new_initial_state()
        bot.restart()
        while not state.is_terminal():
            if state.is_chance_node():
                outcomes = state.chance_outcomes()
                import random
                action = random.choices(
                    [a for a, _ in outcomes],
                    weights=[p for _, p in outcomes]
                )[0]
                state.apply_action(action)
            elif state.current_player() == bot._player_id:
                t0 = time.perf_counter()
                action = bot.step(state)
                times.append(time.perf_counter() - t0)
                state.apply_action(action)
            else:
                # Random opponent
                import random
                action = random.choice(state.legal_actions())
                bot.inform_action(state, state.current_player(), action)
                state.apply_action(action)

    avg_ms = 1000 * sum(times) / len(times) if times else 0
    print(f"  {bot_name}: {len(times)} moves, avg {avg_ms:.1f} ms/move")
    return avg_ms


def main():
    print(f"fast_ops loaded: OOS={oos_fast}, DD={dd_fast}")
    assert oos_fast, "OOS fast_ops not loaded!"
    assert dd_fast, "DD fast_ops not loaded!"

    game = pyspiel.load_game("kuhn_poker")
    sims = 2000

    print(f"\nKuhn Poker, {sims} sims/move:")
    oos = OOSBot(game, player_id=0, num_simulations=sims, delta=0.5,
                 epsilon=0.4, gamma=0.01, seed=42)
    run_bot_test("OOS", oos, game)

    dd = DepthDeltaBot(game, player_id=0, num_simulations=sims,
                       epsilon=0.4, gamma=0.01,
                       schedule=ConstantSchedule(0.5), seed=42)
    run_bot_test("DD-const", dd, game)

    # Verify regret tables are populated
    print(f"\n  OOS infostates: {len(oos._infostates)}")
    print(f"  DD infostates: {len(dd._infostates)}")
    assert len(oos._infostates) > 0, "OOS no infostates!"
    assert len(dd._infostates) > 0, "DD no infostates!"

    # Quick benchmark: compare with/without fast_ops disabled
    # (We can't easily disable it since it's imported, but at least
    # we verify it runs and the timings are reasonable)
    game_leduc = pyspiel.load_game("leduc_poker")
    sims_big = 5000
    print(f"\nLeduc Poker, {sims_big} sims/move:")
    oos2 = OOSBot(game_leduc, player_id=0, num_simulations=sims_big,
                  delta=0.5, epsilon=0.4, gamma=0.01, seed=123)
    t_oos = run_bot_test("OOS", oos2, game_leduc, num_games=3)

    dd2 = DepthDeltaBot(game_leduc, player_id=0, num_simulations=sims_big,
                        epsilon=0.4, gamma=0.01,
                        schedule=ConstantSchedule(0.5), seed=123)
    t_dd = run_bot_test("DD-const", dd2, game_leduc, num_games=3)

    print(f"\n  OOS infostates: {len(oos2._infostates)}")
    print(f"  DD infostates: {len(dd2._infostates)}")

    print("\nIntegration test PASSED!")


if __name__ == "__main__":
    main()
