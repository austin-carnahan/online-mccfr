"""Smoke test for Depth-LOTR: verify observed on_path_fraction ≈ 1 − ρ.

Under the (ρ, w) parameterization, the probability of staying on the
target prefix through *all* D decisions is exactly 1 − ρ — independent
of the depth profile w(d) and of D.  This holds because

    S(D) = Π_{d<D} τ(d) = 1 − ρ · Σ_d w(d) = 1 − ρ.

So `on_path_given_chance_stayed` should match 1 − ρ within sampling
noise for any ρ and any depth profile w.
"""

from __future__ import annotations

import pyspiel

from src.depth_lotr import DepthLOTRBot, uniform, late_linear, early_linear


def play_to_depth(game, target_history):
    state = game.new_initial_state()
    for a in target_history:
        state.apply_action(a)
    return state


def run_one(game_name, history, schedule, num_sims=4000, seed=42):
    game = pyspiel.load_game(game_name)
    state = play_to_depth(game, history)
    if state.is_terminal() or state.current_player() < 0:
        raise RuntimeError("Pick a deeper non-terminal, non-chance history")

    bot = DepthLOTRBot(
        game, state.current_player(),
        num_simulations=num_sims,
        epsilon=0.4, gamma=0.01,
        schedule=schedule, seed=seed,
        tracking=True,
    )
    bot.step_with_policy(state)
    return bot.get_tracking_summary()


def main():
    print("=" * 72)
    print("Depth-LOTR smoke test: on_path|chance ≈ 1 − ρ for any depth profile")
    print("=" * 72)

    configs = [
        ("kuhn_poker", [0, 1, 0], "kuhn after [c=0, c=1, p0=pass]"),
        ("leduc_poker", [0, 1, 1, 1, 4],
         "leduc after [c=0, c=1, p0=call, p1=check, board=4]"),
    ]

    for game_name, history, label in configs:
        print(f"\n--- {label} ---")
        for rho in [0.1, 0.5, 0.9]:
            for sched_factory, sched_label in [
                (uniform, "uniform"),
                (late_linear, "late_linear"),
                (early_linear, "early_linear"),
            ]:
                schedule = sched_factory(rho)
                summary = run_one(game_name, history, schedule)
                D = summary["prefix_D"]
                obs = summary["on_path_given_chance_stayed"]
                exp = 1.0 - rho
                mark = "OK" if abs(obs - exp) < 0.04 else "DIFF"
                print(f"  ρ={rho}  w={sched_label:13s}  D={D}  "
                      f"obs|c={obs:.3f}  exp(1−ρ)={exp:.3f}  [{mark}]")

    print("\nDone.")


if __name__ == "__main__":
    main()
