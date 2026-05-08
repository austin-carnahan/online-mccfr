"""Trajectory probability match test.

Run a fixed Kuhn pre-act state through OOS_d05 and v4_step_r05 and
compare for every realized leaf:

    * Empirical P_q(z) — the bot's actual sampling distribution
    * mean(l_recorded | realized=z) — should match P_q(z) for a
        stationary proposal.
    * P_emp(z) · E[1/l_recorded | realized=z] — should be near 1 if
        l is the true marginal at every realization.
    * For v4: per-mode mean l_recorded, with step(ρ,0) modes labelled
        as arm0/free or armD/forced.

If l = P_q(z), then E[1/l_recorded · 𝟙{realized=z}] = 1 for every
trajectory.  Summing across leaves gives E[1/l] = #leaves visited.
We report this per-bot as a unit-check.

Also reports mean_l * P_emp because it was useful in earlier notes,
but the two unit-check columns are l/P_emp and P_emp*E[1/l].
"""

import math
from collections import defaultdict

import pyspiel

from src.oos import OOSBot
from src.mixture_lotr import MixtureLOTRBot, step


def install_trace(bot):
    """Monkey-patch a bot so per-iter realized leaf and recorded l are logged."""
    bot._traces = []
    bot._cur_trace = None

    is_oos = hasattr(bot, "_oos_episode")
    episode_attr = "_oos_episode" if is_oos else "_episode"
    orig_episode = getattr(bot, episode_attr)

    def episode_wrapper(state, update_player):
        bot._cur_trace = {"leaf": None, "l": None,
                  "mode": None, "diverged": None,
                  "last_diverged": None}
        result = orig_episode(state, update_player)
        if is_oos:
            bot._cur_trace["mode"] = "T" if bot._is_targeted_iter else "U"
        bot._traces.append(bot._cur_trace)
        return result
    setattr(bot, episode_attr, episode_wrapper)

    orig_walk = bot._walk

    if is_oos:
        def walk_wrapper(state, update_player, my_reach=1.0, opp_reach=1.0,
                         s1=1.0, s2=1.0):
            if state.is_terminal() and bot._cur_trace is not None \
                    and bot._cur_trace["leaf"] is None:
                bot._cur_trace["leaf"] = tuple(state.history())
                bot._cur_trace["l"] = bot._delta * s1 + (1 - bot._delta) * s2
            return orig_walk(state, update_player, my_reach, opp_reach, s1, s2)
    else:
        import numpy as _np
        def walk_wrapper(state, update_player, my_reach=1.0, opp_reach=1.0,
                         s=None, diverged=False):
            if s is None:
                s = _np.ones(bot._D + 1, dtype=_np.float64)
            if bot._cur_trace is not None:
                bot._cur_trace["last_diverged"] = diverged
            if state.is_terminal() and bot._cur_trace is not None \
                    and bot._cur_trace["leaf"] is None:
                bot._cur_trace["leaf"] = tuple(state.history())
                bot._cur_trace["l"] = float(_np.dot(bot._arm_probs, s)) \
                    if bot._D > 0 else float(s[0])
                bot._cur_trace["diverged"] = diverged
                bot._cur_trace["mode"] = (
                    "arm0/free" if diverged else f"arm{bot._D}/forced"
                )
            return orig_walk(state, update_player, my_reach, opp_reach,
                             s, diverged)
    bot._walk = walk_wrapper

    orig_playout = bot._playout

    def playout_wrapper(state, update_player, sample_reach):
        x, l, u = orig_playout(state, update_player, sample_reach)
        if bot._cur_trace is not None and bot._cur_trace["leaf"] is None:
            bot._cur_trace["leaf"] = tuple(state.history())
            bot._cur_trace["l"] = l
            if not is_oos:
                diverged = bool(bot._cur_trace.get("last_diverged"))
                bot._cur_trace["diverged"] = diverged
                bot._cur_trace["mode"] = (
                    "arm0/free" if diverged else f"arm{bot._D}/forced"
                )
        return (x, l, u)
    bot._playout = playout_wrapper


def make_state(history):
    g = pyspiel.load_game("kuhn_poker")
    s = g.new_initial_state()
    for a in history:
        s.apply_action(a)
    return g, s


def run_one(name, bot_factory, n_iters, history):
    g, s = make_state(history)
    bot = bot_factory(g)
    install_trace(bot)
    bot._num_simulations = n_iters
    bot.step_with_policy(s)
    return name, bot._traces, bot


def aggregate(traces):
    """Return per-leaf stats: count, mean_l, mean_inv_l, per-mode mean_l."""
    by_leaf = defaultdict(lambda: {"n": 0, "sum_l": 0.0, "sum_inv_l": 0.0,
                                   "modes": defaultdict(lambda: {"n": 0, "sum_l": 0.0})})
    for t in traces:
        if t["leaf"] is None or t["l"] is None or t["l"] <= 0:
            continue
        z = t["leaf"]
        by_leaf[z]["n"] += 1
        by_leaf[z]["sum_l"] += t["l"]
        by_leaf[z]["sum_inv_l"] += 1.0 / t["l"]
        m = t["mode"] or "?"
        by_leaf[z]["modes"][m]["n"] += 1
        by_leaf[z]["modes"][m]["sum_l"] += t["l"]
    return by_leaf


def report(name, traces, top_k=None):
    n = len(traces)
    by_leaf = aggregate(traces)
    # Sort leaves by count desc
    leaves = sorted(by_leaf.items(), key=lambda kv: -kv[1]["n"])
    print(f"\n=== {name}  (N={n} iters, {len(by_leaf)} unique leaves) ===")
    # Sanity: sum_z count(z)/N · E[1/l | z] over all leaves
    total_inv_l_density = 0.0
    for z, s in by_leaf.items():
        emp = s["n"] / n
        mean_inv_l = s["sum_inv_l"] / s["n"]
        total_inv_l_density += emp * mean_inv_l  # ≈ E[1/l] over all iters
    # If l = P_q(z) for every realization, this equals #leaves visited.
    # The per-leaf unit check is P_emp(z) * E[1/l | z] ~= 1.  For a
    # stationary proposal, mean_l / P_emp is another useful check.
    print(f"  Σ_z P_emp(z)·E[1/l|z]  = {total_inv_l_density:.3f}  "
          f"(== #unique leaves if l is the true marginal)")
    print()
    print(f"  {'leaf':30s} {'count':>6s} {'P_emp':>7s} "
          f"{'mean_l':>9s} {'l/Pemp':>8s} {'P*E1/l':>8s} "
          f"{'l*P':>9s}  per-mode (n, mean_l)")
    rows = leaves if top_k is None else leaves[:top_k]
    for z, s in rows:
        emp = s["n"] / n
        mean_l = s["sum_l"] / s["n"]
        mean_inv_l = s["sum_inv_l"] / s["n"]
        l_over_emp = mean_l / emp if emp > 0 else float("inf")
        inv_density = emp * mean_inv_l
        l_times_emp = mean_l * emp
        modestr = "  ".join(
            f"{m}: ({ms['n']}, {ms['sum_l']/ms['n']:.4f})"
            for m, ms in s["modes"].items()
        )
        zstr = ",".join(str(a) for a in z)
        if len(zstr) > 28:
            zstr = zstr[:28]
        print(f"  {zstr:30s} {s['n']:>6d} {emp:>7.4f} "
              f"{mean_l:>9.4f} {l_over_emp:>8.3f} {inv_density:>8.3f} "
              f"{l_times_emp:>9.4f}  {modestr}")


def main():
    n_iters = 50000
    seed = 42

    # Test multiple Kuhn states with progressively longer prefixes.
    # IST observable prefix positions D shown in comment.
    states = [
        ("root_P0_acts (D=1)", [0, 1]),               # P0 at "0"
        ("after_P0_pass (D=2)", [0, 1, 0]),           # P1 at "1p"
        ("after_pb (D=3)", [0, 1, 0, 1]),             # P0 at "0pb"
    ]

    for label, history in states:
        print(f"\n{'='*80}")
        print(f"STATE: history={history}  ({label})")
        print(f"{'='*80}")
        bots = [
            ("OOS_d05",
             lambda g: OOSBot(g, 0, num_simulations=n_iters, delta=0.5,
                              epsilon=0.4, gamma=0.01, seed=seed)),
            ("v4_step_r05",
             lambda g: MixtureLOTRBot(g, 0, num_simulations=n_iters,
                                      schedule=step(0.5, 0), epsilon=0.4,
                                      gamma=0.01, seed=seed)),
        ]
        for name, factory in bots:
            _, traces, bot = run_one(name, factory, n_iters, history)
            extra = ""
            if hasattr(bot, "_D"):
                extra = f"  D={bot._D}  depth_to_position={dict(bot._depth_to_position)}"
            print(f"\n[bot info] {name}{extra}")
            report(name, traces, top_k=None)


if __name__ == "__main__":
    main()
