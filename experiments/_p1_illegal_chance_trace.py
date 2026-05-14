"""Trace the P1 illegal observable-chance target branch in Kuhn.

The target state is a P1 decision after history [0, 1, 0].  From the
simulation root, the first chance outcome is hidden from P1.  If that
hidden card is 1, then P1's real target card (also 1) is illegal at the
next chance node.

This diagnostic compares:
  - OOS delta=0.5
    - production LOTR step(0.5, 0)
    - a shadow LOTR variant that also flips the coin at an illegal
        observable chance target but otherwise samples the chance outcome
        naturally and updates s by the natural probability for every arm.

The key check is the P0 action after that illegal chance branch.  OOS
should still have free-sampling mass there on untargeted iterations.
Production LOTR should now match that OOS-shaped free-sampling mass.
"""

from __future__ import annotations

from collections import Counter, defaultdict

import numpy as np
import pyspiel

from src.lotr import LOTRBot, step
from src.oos import OOSBot, REGRET_INDEX, AVG_POLICY_INDEX


HISTORY = [0, 1, 0]
PLAYER_ID = 1
TARGET_P1_CARD = HISTORY[1]
TARGET_P0_ACTION = HISTORY[2]
N_ITERS = 20000
SEED = 7


def make_state(history):
    game = pyspiel.load_game("kuhn_poker")
    state = game.new_initial_state()
    for action in history:
        state.apply_action(action)
    return game, state


def prepopulate_infostates(bot):
    """Create every Kuhn decision table up front to avoid playout noise."""
    seen = set()

    def visit(state):
        key = tuple(state.history())
        if key in seen:
            return
        seen.add(key)

        if state.is_terminal():
            return
        if state.is_chance_node():
            for action, _probability in state.chance_outcomes():
                child = state.clone()
                child.apply_action(action)
                visit(child)
            return

        info_key = state.information_state_string(state.current_player())
        num_actions = len(state.legal_actions())
        bot._infostates.setdefault(
            info_key,
            [
                np.zeros(num_actions, dtype=np.float64),
                np.zeros(num_actions, dtype=np.float64),
            ],
        )
        for action in state.legal_actions():
            child = state.clone()
            child.apply_action(action)
            visit(child)

    visit(bot._game.new_initial_state())


def install_common_trace(bot, is_oos):
    bot._p1_traces = []
    bot._cur_p1_trace = None

    episode_attr = "_oos_episode" if is_oos else "_episode"
    orig_episode = getattr(bot, episode_attr)

    def episode_wrapper(state, update_player):
        bot._cur_p1_trace = {
            "leaf": None,
            "update_player": update_player,
            "illegal_seen": False,
            "illegal_depth": None,
            "illegal_records": [],
            "post_illegal_decisions": [],
            "sample_calls": [],
            "mode": None,
        }
        result = orig_episode(state, update_player)
        if is_oos:
            bot._cur_p1_trace["mode"] = "T" if bot._is_targeted_iter else "U"
        bot._p1_traces.append(bot._cur_p1_trace)
        bot._cur_p1_trace = None
        return result

    setattr(bot, episode_attr, episode_wrapper)

    orig_walk = bot._walk
    if is_oos:
        def walk_wrapper(state, update_player, my_reach, opp_reach, s1, s2):
            trace = bot._cur_p1_trace
            if trace is not None:
                if state.is_terminal() and trace["leaf"] is None:
                    trace["leaf"] = tuple(state.history())
                elif trace["illegal_seen"] and not state.is_chance_node():
                    depth = len(state.history())
                    if depth >= 2 and not state.is_terminal():
                        target = bot._match_history[depth] if depth < len(bot._match_history) else None
                        trace["post_illegal_decisions"].append({
                            "depth": depth,
                            "target": target,
                            "legal": list(state.legal_actions()),
                            "s1": float(s1),
                            "s2": float(s2),
                        })
            return orig_walk(state, update_player, my_reach, opp_reach, s1, s2)
    else:
        def walk_wrapper(state, update_player, my_reach, opp_reach, s, diverged):
            trace = bot._cur_p1_trace
            if trace is not None:
                if state.is_terminal() and trace["leaf"] is None:
                    trace["leaf"] = tuple(state.history())
                elif trace["illegal_seen"] and not state.is_chance_node():
                    depth = len(state.history())
                    if depth >= 2 and not state.is_terminal():
                        target = bot._match_history[depth] if depth < len(bot._match_history) else None
                        trace["post_illegal_decisions"].append({
                            "depth": depth,
                            "target": target,
                            "legal": list(state.legal_actions()),
                            "diverged": bool(diverged),
                            "s": [float(value) for value in s],
                            "l": float(bot._l_from_s(s)),
                        })
            return orig_walk(state, update_player, my_reach, opp_reach, s, diverged)

    bot._walk = walk_wrapper


def install_oos_chance_trace(bot):
    orig_handle_chance = bot._handle_chance

    def handle_chance_wrapper(state, update_player, my_reach, opp_reach, s1, s2):
        outcomes = state.chance_outcomes()
        chance_probs = dict(outcomes)
        depth = len(state.history())
        in_history = depth < len(bot._match_history)
        target = bot._match_history[depth] if in_history else None
        observable = bot._chance_forced.get(depth, True)
        target_legal = target in chance_probs if target is not None else False
        illegal = in_history and observable and not target_legal
        if illegal and bot._cur_p1_trace is not None:
            bot._cur_p1_trace["illegal_seen"] = True
            bot._cur_p1_trace["illegal_depth"] = depth
            bot._cur_p1_trace["illegal_records"].append({
                "depth": depth,
                "target": target,
                "legal": list(chance_probs.keys()),
                "targeted_iter": bool(bot._is_targeted_iter),
                "s1_in": float(s1),
                "s2_in": float(s2),
            })
        return orig_handle_chance(state, update_player, my_reach, opp_reach, s1, s2)

    bot._handle_chance = handle_chance_wrapper


def install_lotr_chance_trace(bot, shadow_fix=False):
    orig_handle_chance = bot._handle_chance
    orig_sample_realized_action = bot._sample_realized_action

    def sample_realized_action_wrapper(target_idx, pi, tau, num_actions):
        sampled_idx, coin_diverge = orig_sample_realized_action(
            target_idx, pi, tau, num_actions)
        trace = bot._cur_p1_trace
        if trace is not None:
            trace["sample_calls"].append({
                "tau": float(tau),
                "target_idx": int(target_idx),
                "sampled_idx": int(sampled_idx),
                "coin_diverge": bool(coin_diverge),
                "after_illegal": bool(trace["illegal_seen"]),
            })
        return sampled_idx, coin_diverge

    bot._sample_realized_action = sample_realized_action_wrapper

    def handle_chance_wrapper(state, update_player, my_reach, opp_reach, s, diverged):
        outcomes = state.chance_outcomes()
        actions, probs = zip(*outcomes)
        actions = list(actions)
        probs_arr = np.array(probs, dtype=np.float64)
        depth = len(state.history())
        in_history = depth < len(bot._match_history)
        in_observable_position = depth in bot._depth_to_position
        target = bot._match_history[depth] if in_history else None
        target_legal = target in actions if target is not None else False
        illegal = (not diverged) and in_history and in_observable_position and not target_legal

        if not illegal:
            return orig_handle_chance(state, update_player, my_reach, opp_reach, s, diverged)

        d = bot._depth_to_position[depth]
        tau_d = bot._schedule.tau(d, bot._D)
        trace = bot._cur_p1_trace
        if trace is not None:
            trace["illegal_seen"] = True
            trace["illegal_depth"] = depth
            trace["illegal_records"].append({
                "depth": depth,
                "position": d,
                "target": target,
                "legal": actions,
                "tau": float(tau_d),
                "diverged_in": bool(diverged),
                "s_in": [float(value) for value in s],
                "l_in": float(bot._l_from_s(s)),
                "shadow_fix": bool(shadow_fix),
            })

        if not shadow_fix:
            return orig_handle_chance(state, update_player, my_reach, opp_reach, s, diverged)

        if tau_d >= 1.0:
            coin_diverge = False
        elif tau_d <= 0.0:
            coin_diverge = True
        else:
            coin_diverge = bot._rng_uniform() >= tau_d

        sampled_idx = int(bot._rng.choice(len(actions), p=probs_arr))
        action = actions[sampled_idx]
        rho = float(probs_arr[sampled_idx])
        new_s = s * rho
        new_diverged = bool(coin_diverge)

        if trace is not None:
            trace["illegal_records"][-1].update({
                "sampled_action": int(action),
                "rho": rho,
                "coin_diverge": bool(coin_diverge),
                "diverged_out": new_diverged,
                "s_out": [float(value) for value in new_s],
                "l_out_local": float(bot._l_from_s(new_s)),
            })

        state.apply_action(action)
        x, l_out, u = bot._walk(
            state, update_player, my_reach, rho * opp_reach, new_s, new_diverged)
        return (rho * x, l_out, u)

    bot._handle_chance = handle_chance_wrapper


def run_bot(name, factory, is_oos=False, shadow_fix=False):
    game, state = make_state(HISTORY)
    bot = factory(game)
    prepopulate_infostates(bot)
    install_common_trace(bot, is_oos=is_oos)
    if is_oos:
        install_oos_chance_trace(bot)
    else:
        install_lotr_chance_trace(bot, shadow_fix=shadow_fix)
    bot._num_simulations = N_ITERS
    bot.step_with_policy(state)
    return name, bot


def summarize(name, bot):
    traces = bot._p1_traces
    illegal = [trace for trace in traces if trace["illegal_seen"] and trace["leaf"]]
    by_leaf = Counter(trace["leaf"] for trace in illegal)
    p0_action_counts = Counter(trace["leaf"][2] for trace in illegal)
    mode_action_counts = defaultdict(Counter)
    after_illegal_call_counts = Counter()
    illegal_coin_counts = Counter()
    post_decision_diverged = Counter()

    for trace in illegal:
        mode = trace.get("mode") or "lotr"
        mode_action_counts[mode][trace["leaf"][2]] += 1
        calls_after = [call for call in trace["sample_calls"] if call["after_illegal"]]
        after_illegal_call_counts[len(calls_after)] += 1
        for record in trace["illegal_records"]:
            if "coin_diverge" in record:
                illegal_coin_counts[record["coin_diverge"]] += 1
        for decision in trace["post_illegal_decisions"]:
            if "diverged" in decision:
                post_decision_diverged[decision["diverged"]] += 1

    print(f"\n=== {name} ===")
    print(f"iters={len(traces)}  illegal_branch_iters={len(illegal)} "
          f"({len(illegal) / len(traces):.3f})")
    print(f"target P1 card={TARGET_P1_CARD}; illegal iff sampled P0 hidden card == {TARGET_P1_CARD}")
    print(f"P0 action after illegal branch counts: {dict(sorted(p0_action_counts.items()))}")
    if illegal:
        non_target = sum(count for action, count in p0_action_counts.items()
                         if action != TARGET_P0_ACTION)
        print(f"non-target P0 action rate after illegal branch: {non_target / len(illegal):.3f}")
    for mode, counts in sorted(mode_action_counts.items()):
        total = sum(counts.values())
        non_target = sum(count for action, count in counts.items()
                         if action != TARGET_P0_ACTION)
        print(f"  mode={mode}: counts={dict(sorted(counts.items()))} "
              f"non_target_rate={non_target / total if total else 0.0:.3f}")
    if after_illegal_call_counts:
        print(f"LOTR sample_realized_action calls after illegal: {dict(sorted(after_illegal_call_counts.items()))}")
    if illegal_coin_counts:
        print(f"shadow illegal coin diverged counts: {dict(sorted(illegal_coin_counts.items()))}")
    if post_decision_diverged:
        print(f"post-illegal decision diverged-in counts: {dict(sorted(post_decision_diverged.items()))}")

    print("top illegal leaves:")
    for leaf, count in by_leaf.most_common(8):
        print(f"  {leaf}: {count}")

    if illegal:
        example = illegal[0]
        print("example illegal trace:")
        print(f"  leaf={example['leaf']} mode={example.get('mode')}")
        print(f"  illegal_records={example['illegal_records']}")
        print(f"  post_illegal_decisions={example['post_illegal_decisions'][:3]}")
        print(f"  sample_calls={example['sample_calls'][:5]}")


def main():
    print("P1 illegal observable chance target trace")
    print(f"history={HISTORY} player={PLAYER_ID} n_iters={N_ITERS} seed={SEED}")
    print("gamma=1.0 keeps regret-matched policies uniform for this sampling trace")

    runs = [
        run_bot(
            "OOS_d05",
            lambda game: OOSBot(
                game, PLAYER_ID, num_simulations=N_ITERS, delta=0.5,
                epsilon=0.4, gamma=1.0, seed=SEED),
            is_oos=True,
        ),
        run_bot(
            "LOTR_current_step_r05",
            lambda game: LOTRBot(
                game, PLAYER_ID, num_simulations=N_ITERS,
                schedule=step(0.5, 0), epsilon=0.4, gamma=1.0, seed=SEED),
        ),
        run_bot(
            "LOTR_shadow_illegal_coin",
            lambda game: LOTRBot(
                game, PLAYER_ID, num_simulations=N_ITERS,
                schedule=step(0.5, 0), epsilon=0.4, gamma=1.0, seed=SEED),
            shadow_fix=True,
        ),
    ]

    for name, bot in runs:
        if hasattr(bot, "_D"):
            print(f"\n{name}: D={bot._D} depth_to_position={dict(bot._depth_to_position)} "
                  f"arm_probs={bot._arm_probs.tolist()}")
        summarize(name, bot)


if __name__ == "__main__":
    main()