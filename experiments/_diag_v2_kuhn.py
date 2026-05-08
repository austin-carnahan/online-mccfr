"""Diagnostic: per-infoset avg strategy on Kuhn for OOS / v1 / v2.

Goal: distinguish "v2 plateau is uniform variance" from "v2 plateau is
structured bias from IS-reweighting at hidden chance."

If hypothesis (hidden-chance IS reweighting) is correct, we expect:
  - v2's avg strat is *systematically* off at infosets whose updates
    sit downstream of P0's hidden card draw (every P1 infoset, plus
    P0's second decisions).
  - The deviation pattern should be coherent across matches (same
    direction), not symmetric noise.
  - OOS at the same budget should be near Nash everywhere.
  - v1 should also be off but in a different pattern (no chance
    exploration → undervisits diverged-card branches).
"""

from collections import defaultdict
import numpy as np
import pyspiel
from open_spiel.python import policy as policy_lib
from open_spiel.python.bots.policy import PolicyBot
from src.oos import OOSBot
from src.depth_lotr import DepthLOTRBot, step as depth_step
from src.every_node_lotr import EveryNodeLOTRBot, step as every_step
from src.observable_node_lotr import ObservableNodeLOTRBot, step as obs_step
from src.mixture_lotr import MixtureLOTRBot, step as mix_step

EPSILON, GAMMA = 0.4, 0.01
SIMS, MATCHES = 50_000, 50

ALPHA = 1.0 / 3.0
# Action 0 = pass / check / fold;  action 1 = bet / call.
EQ = {
    "0":   np.array([1 - ALPHA,    ALPHA]),    # P0 J first decision
    "1":   np.array([1.0,          0.0]),      # P0 Q first decision
    "2":   np.array([1 - 3*ALPHA,  3*ALPHA]),  # P0 K first decision
    "0pb": np.array([1.0,          0.0]),      # P0 J facing pass-bet
    "1pb": np.array([2/3,          1/3]),      # P0 Q facing pass-bet
    "2pb": np.array([0.0,          1.0]),      # P0 K facing pass-bet
    "0p":  np.array([2/3,          1/3]),      # P1 J facing check
    "1p":  np.array([1.0,          0.0]),      # P1 Q facing check
    "2p":  np.array([0.0,          1.0]),      # P1 K facing check
    "0b":  np.array([1.0,          0.0]),      # P1 J facing bet
    "1b":  np.array([2/3,          1/3]),      # P1 Q facing bet
    "2b":  np.array([0.0,          1.0]),      # P1 K facing bet
}


def make_bot(algo, game, player, seed):
    if algo == "oos":
        return OOSBot(game, player, num_simulations=SIMS, delta=0.5,
                      epsilon=EPSILON, gamma=GAMMA, seed=seed, tracking=False)
    if algo == "v1":
        return DepthLOTRBot(game, player, num_simulations=SIMS,
                            epsilon=EPSILON, gamma=GAMMA,
                            schedule=depth_step(0.5, 0), seed=seed, tracking=False)
    if algo == "v2":
        return EveryNodeLOTRBot(game, player, num_simulations=SIMS,
                                epsilon=EPSILON, gamma=GAMMA,
                                schedule=every_step(0.5, 0), seed=seed, tracking=False)
    if algo == "v3":
        return ObservableNodeLOTRBot(game, player, num_simulations=SIMS,
                                     epsilon=EPSILON, gamma=GAMMA,
                                     schedule=obs_step(0.5, 0), seed=seed, tracking=False)
    if algo == "v4":
        return MixtureLOTRBot(game, player, num_simulations=SIMS,
                              epsilon=EPSILON, gamma=GAMMA,
                              schedule=mix_step(0.5, 0), seed=seed, tracking=False)
    raise ValueError(algo)


def run(algo, game, seed_base=42):
    rng = np.random.RandomState(seed_base)
    accum = defaultdict(lambda: np.zeros(2))
    visits = defaultdict(int)
    for m in range(MATCHES):
        sp = m % 2
        rp = 1 - sp
        bot = make_bot(algo, game, sp, seed_base + m)
        rb = PolicyBot(rp, rng, policy_lib.UniformRandomPolicy(game))
        bots = [None, None]
        bots[sp] = bot
        bots[rp] = rb
        st = game.new_initial_state()
        for b in bots:
            b.restart()
        while not st.is_terminal():
            if st.is_chance_node():
                outs, probs = zip(*st.chance_outcomes())
                a = rng.choice(outs, p=probs)
                st.apply_action(a)
            else:
                p = st.current_player()
                a = bots[p].step(st)
                for q, b in enumerate(bots):
                    if q != p:
                        b.inform_action(st, p, a)
                st.apply_action(a)
        for k, t in bot._infostates.items():
            if t[1].sum() == 0:
                continue
            accum[k] += t[1]
            visits[k] += 1
    final = {
        k: (v / v.sum() if v.sum() > 0 else np.array([0.5, 0.5]))
        for k, v in accum.items()
    }
    return final, visits


def main():
    game = pyspiel.load_game("kuhn_poker")
    print(f"Diagnostic: sims={SIMS}  matches={MATCHES}\n")
    algos = ["oos", "v1", "v2", "v3", "v4"]
    res = {}
    for a in algos:
        print(f"  Running {a} ...", flush=True)
        res[a] = run(a, game)
    print()
    print("=" * 140)
    print("Per-infoset avg strategy vs Kuhn Nash (alpha=1/3)")
    print("=" * 140)
    print(f"{'info':6s} {'eq':>14s}  {'oos':>26s}  {'v1':>26s}  {'v2':>26s}  {'v3':>26s}  {'v4':>26s}")
    tot = {a: 0.0 for a in algos}
    for k in sorted(EQ.keys()):
        eq = EQ[k]
        cells = []
        for a in algos:
            f, v = res[a]
            if k in f:
                p = f[k]
                l1 = float(abs(p - eq).sum())
                tot[a] += l1
                cells.append(f"[{p[0]:.3f},{p[1]:.3f}] L1={l1:.3f} v={v[k]}")
            else:
                cells.append("(missing)")
        print(f"{k:6s} [{eq[0]:.3f},{eq[1]:.3f}]  " + "  ".join(f"{c:>26s}" for c in cells))
    print()
    for a in algos:
        print(f"  total_L1 {a}: {tot[a]:.4f}")


if __name__ == "__main__":
    main()
