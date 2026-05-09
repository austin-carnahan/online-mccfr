"""Frozen-policy/no-learning parity diagnostic for OOS vs Mixture LOTR.

This is the stationary-policy version of _regret_delta_match.py. It
prepopulates every Kuhn decision infoset with zero regrets, disables all
regret and average-strategy mutation, then records the shadow regret
update quantities that each algorithm would have applied.

The point is to separate formula-level sampling/reach/denominator
differences from policy drift caused by learning.
"""

from __future__ import annotations

import os
from collections import defaultdict

import numpy as np
import pyspiel

from src import mixture_lotr as mix_mod
from src import oos as oos_mod
from src.mixture_lotr import MixtureLOTRBot, step
from src.oos import OOSBot


N_ITERS = int(os.environ.get("N_ITERS", "50000"))
SEED = int(os.environ.get("SEED", "42"))
ALIGN_DETERMINISTIC_TAU = os.environ.get("ALIGN_DETERMINISTIC_TAU", "1") != "0"
FLOAT_FIELDS = ("c", "x", "u", "pi_opp", "l", "W", "delta_R")
ID_FIELDS = (
    "iter", "update_player", "info", "action", "action_idx",
    "sampled_action", "sampled_idx",
)


def make_state(history):
    game = pyspiel.load_game("kuhn_poker")
    state = game.new_initial_state()
    for action in history:
        state.apply_action(action)
    return game, state


def prepopulate_infostates(bot):
    """Create zero-regret/zero-average entries for every decision infoset."""
    seen_histories = set()

    def visit(state):
        history = tuple(state.history())
        if history in seen_histories:
            return
        seen_histories.add(history)

        if state.is_terminal():
            return
        if state.is_chance_node():
            for action, _prob in state.chance_outcomes():
                child = state.clone()
                child.apply_action(action)
                visit(child)
            return

        info_key = state.information_state_string(state.current_player())
        num_actions = len(state.legal_actions())
        existing = bot._infostates.get(info_key)
        if existing is None:
            bot._infostates[info_key] = [
                np.zeros(num_actions, dtype=np.float64),
                np.zeros(num_actions, dtype=np.float64),
            ]
        elif len(existing[0]) != num_actions:
            raise ValueError(
                f"Inconsistent action count for {info_key}: "
                f"{len(existing[0])} vs {num_actions}"
            )

        for action in state.legal_actions():
            child = state.clone()
            child.apply_action(action)
            visit(child)

    visit(bot._game.new_initial_state())


def max_table_abs(bot):
    max_abs = 0.0
    for regrets, avg_policy in bot._infostates.values():
        if len(regrets):
            max_abs = max(max_abs, float(np.max(np.abs(regrets))))
        if len(avg_policy):
            max_abs = max(max_abs, float(np.max(np.abs(avg_policy))))
    return max_abs


def _record_regret_rows(bot, update_player, info_key, legal_actions,
                        sampled_idx, c, x, u, opp_reach, l, W):
    for action_idx, action in enumerate(legal_actions):
        if action_idx == sampled_idx:
            delta_r = (c - x) * W
        else:
            delta_r = -x * W
        bot._regret_records.append({
            "iter": bot._trace_iter,
            "update_player": update_player,
            "info": info_key,
            "action": int(action),
            "action_idx": action_idx,
            "sampled_action": int(legal_actions[sampled_idx]),
            "sampled_idx": int(sampled_idx),
            "c": float(c),
            "x": float(x),
            "u": float(u),
            "pi_opp": float(opp_reach),
            "l": float(l),
            "W": float(W),
            "delta_R": float(delta_r),
        })


def install_oos_frozen_trace(bot):
    bot._regret_records = []
    bot._trace_iter = -1

    orig_episode = bot._oos_episode

    def episode_wrapper(state, update_player):
        bot._trace_iter += 1
        return orig_episode(state, update_player)

    bot._oos_episode = episode_wrapper

    def walk_wrapper(state, update_player, my_reach, opp_reach, s1, s2):
        if state.is_terminal():
            l = bot._delta * s1 + (1.0 - bot._delta) * s2
            return (1.0, l, state.player_return(update_player))

        if state.is_chance_node():
            return bot._handle_chance(state, update_player,
                                      my_reach, opp_reach, s1, s2)

        cur_player = state.current_player()
        info_key = state.information_state_string(cur_player)
        legal_actions = state.legal_actions()
        num_actions = len(legal_actions)

        if info_key not in bot._infostates:
            raise RuntimeError(f"Missing prepopulated OOS infoset: {info_key}")

        if oos_mod._HAS_FAST_OPS:
            regrets_view = bot._infostates[info_key][oos_mod.REGRET_INDEX]
            if cur_player == update_player:
                policy, sample_policy, _auto_idx = oos_mod.regret_match_sample_eps(
                    regrets_view, bot._gamma, bot._epsilon, bot._rng)
            else:
                policy, _auto_idx = oos_mod.regret_match_sample(
                    regrets_view, bot._gamma, bot._rng)
                sample_policy = policy
        else:
            policy = bot._regret_matching(
                bot._infostates[info_key][oos_mod.REGRET_INDEX])
            if cur_player == update_player:
                uniform = np.ones(num_actions, dtype=np.float64) / num_actions
                sample_policy = bot._epsilon * uniform + (1.0 - bot._epsilon) * policy
            else:
                sample_policy = policy.copy()

        target_idx = bot._get_targeted_action_idx(state, legal_actions)
        if bot._is_targeted_iter and target_idx is not None:
            sampled_idx = target_idx
            action = legal_actions[target_idx]
        else:
            sampled_idx = bot._rng.choice(num_actions, p=sample_policy)
            action = legal_actions[sampled_idx]

        new_s2 = s2 * sample_policy[sampled_idx]
        if target_idx is not None and sampled_idx == target_idx:
            new_s1 = s1
        elif target_idx is not None:
            new_s1 = 0.0
        else:
            new_s1 = s1 * sample_policy[sampled_idx]

        if cur_player == update_player:
            new_my_reach = my_reach * policy[sampled_idx]
            new_opp_reach = opp_reach
        else:
            new_my_reach = my_reach
            new_opp_reach = opp_reach * policy[sampled_idx]

        state.apply_action(action)
        suffix_x, l, u = bot._walk(state, update_player,
                                   new_my_reach, new_opp_reach,
                                   new_s1, new_s2)

        c = suffix_x
        x = suffix_x * policy[sampled_idx]

        if cur_player == update_player:
            W = u * opp_reach / max(l, 1e-30)
            _record_regret_rows(bot, update_player, info_key, legal_actions,
                                sampled_idx, c, x, u, opp_reach, l, W)

        return (x, l, u)

    bot._walk = walk_wrapper


def install_mixture_frozen_trace(bot):
    bot._regret_records = []
    bot._trace_iter = -1

    orig_sample_realized_action = bot._sample_realized_action

    def sample_realized_action_wrapper(target_idx, pi, tau, num_actions):
        if ALIGN_DETERMINISTIC_TAU:
            if tau >= 1.0:
                return target_idx, False
            if tau <= 0.0:
                return int(bot._rng.choice(num_actions, p=pi)), True
        return orig_sample_realized_action(target_idx, pi, tau, num_actions)

    bot._sample_realized_action = sample_realized_action_wrapper

    orig_episode = bot._episode

    def episode_wrapper(state, update_player):
        bot._trace_iter += 1
        return orig_episode(state, update_player)

    bot._episode = episode_wrapper

    def walk_wrapper(state, update_player, my_reach, opp_reach, s, diverged):
        if state.is_terminal():
            return (1.0, bot._l_from_s(s), state.player_return(update_player))

        if state.is_chance_node():
            return bot._handle_chance(state, update_player,
                                      my_reach, opp_reach, s, diverged)

        cur_player = state.current_player()
        info_key = state.information_state_string(cur_player)
        legal_actions = state.legal_actions()
        num_actions = len(legal_actions)
        depth = len(state.history())

        if info_key not in bot._infostates:
            raise RuntimeError(f"Missing prepopulated Mixture infoset: {info_key}")

        if mix_mod._HAS_FAST_OPS:
            regrets_view = bot._infostates[info_key][mix_mod.REGRET_INDEX]
            if cur_player == update_player:
                policy, sample_policy, _auto_idx = mix_mod.regret_match_sample_eps(
                    regrets_view, bot._gamma, bot._epsilon, bot._rng)
            else:
                policy, _auto_idx = mix_mod.regret_match_sample(
                    regrets_view, bot._gamma, bot._rng)
                sample_policy = policy
        else:
            policy = bot._regret_matching(
                bot._infostates[info_key][mix_mod.REGRET_INDEX])
            if cur_player == update_player:
                uniform = np.ones(num_actions, dtype=np.float64) / num_actions
                sample_policy = bot._epsilon * uniform + (1.0 - bot._epsilon) * policy
            else:
                sample_policy = policy.copy()

        target_idx = bot._get_targeted_action_idx(state, legal_actions)
        in_prefix = (not diverged) and (target_idx is not None)

        if in_prefix:
            d = bot._depth_to_position[depth]
            tau_d = bot._schedule.tau(d, bot._D)
            sampled_idx, miss = bot._sample_realized_action(
                target_idx, sample_policy, tau_d, num_actions)
            new_diverged = miss
        else:
            sampled_idx = bot._rng.choice(num_actions, p=sample_policy)
            new_diverged = diverged

        new_s = s.copy()
        pi_a = float(sample_policy[sampled_idx])
        if depth in bot._depth_to_position and target_idx is not None:
            d_pos = bot._depth_to_position[depth]
            bot._update_s_prefix(new_s, d_pos, sampled_idx, target_idx, pi_a)
        else:
            new_s *= pi_a

        action = legal_actions[sampled_idx]
        if cur_player == update_player:
            new_my_reach = my_reach * policy[sampled_idx]
            new_opp_reach = opp_reach
        else:
            new_my_reach = my_reach
            new_opp_reach = opp_reach * policy[sampled_idx]

        state.apply_action(action)
        suffix_x, l_out, u = bot._walk(state, update_player,
                                       new_my_reach, new_opp_reach,
                                       new_s, new_diverged)

        c = suffix_x
        x = suffix_x * policy[sampled_idx]

        if cur_player == update_player:
            W = u * opp_reach / max(l_out, 1e-30)
            _record_regret_rows(bot, update_player, info_key, legal_actions,
                                sampled_idx, c, x, u, opp_reach, l_out, W)

        return (x, l_out, u)

    bot._walk = walk_wrapper


def run_bot(name, factory, history):
    game, state = make_state(history)
    bot = factory(game)
    prepopulate_infostates(bot)
    before = max_table_abs(bot)
    if name.startswith("OOS"):
        install_oos_frozen_trace(bot)
    else:
        install_mixture_frozen_trace(bot)
    bot._num_simulations = N_ITERS
    bot.step_with_policy(state)
    after = max_table_abs(bot)
    bot._frozen_before_max_abs = before
    bot._frozen_after_max_abs = after
    return bot


def summarize(records):
    grouped = defaultdict(list)
    for record in records:
        grouped[(record["info"], record["action"])].append(record)

    total = len(records)
    summary = {}
    for key, rows in grouped.items():
        values = {field: np.array([row[field] for row in rows], dtype=np.float64)
                  for field in FLOAT_FIELDS}
        delta = values["delta_R"]
        summary[key] = {
            "n": len(rows),
            "p": len(rows) / total if total else 0.0,
            "mean_delta": float(delta.mean()),
            "mean_abs_delta": float(np.abs(delta).mean()),
            "rms_delta": float(np.sqrt(np.mean(delta * delta))),
            "mean_c": float(values["c"].mean()),
            "mean_x": float(values["x"].mean()),
            "mean_u": float(values["u"].mean()),
            "mean_pi_opp": float(values["pi_opp"].mean()),
            "mean_l": float(values["l"].mean()),
            "mean_W": float(values["W"].mean()),
            "rms_W": float(np.sqrt(np.mean(values["W"] * values["W"]))),
        }
        summary[key]["mass_delta"] = summary[key]["p"] * summary[key]["mean_delta"]
        summary[key]["mass_abs_delta"] = summary[key]["p"] * summary[key]["mean_abs_delta"]
        summary[key]["mass_sq_delta"] = summary[key]["p"] * summary[key]["rms_delta"] ** 2
    return summary


def _zero_like_summary(other_summary):
    if not other_summary:
        return defaultdict(float)
    return {field: 0.0 for field in next(iter(other_summary.values())).keys()}


def _fmt_key(key):
    info, action = key
    return f"{info}/a{action}"


def paired_compare(left_records, right_records):
    max_abs = {field: 0.0 for field in FLOAT_FIELDS}
    mismatch_count = abs(len(left_records) - len(right_records))
    first_mismatch = None
    for idx, (left, right) in enumerate(zip(left_records, right_records)):
        row_mismatch = False
        for field in ID_FIELDS:
            if left[field] != right[field]:
                row_mismatch = True
        for field in FLOAT_FIELDS:
            diff = abs(left[field] - right[field])
            max_abs[field] = max(max_abs[field], diff)
            if diff > 1e-12:
                row_mismatch = True
        if row_mismatch:
            mismatch_count += 1
            if first_mismatch is None:
                first_mismatch = (idx, left, right)
    return mismatch_count, max_abs, first_mismatch


def compare(label, history):
    print("\n" + "=" * 88)
    print(f"STATE: history={history}  ({label})")
    print("=" * 88)

    bots = {
        "OOS": run_bot(
            "OOS_frozen",
            lambda game: OOSBot(game, 0, num_simulations=N_ITERS, delta=0.5,
                                epsilon=0.4, gamma=0.01, seed=SEED),
            history,
        ),
        "v4": run_bot(
            "v4_frozen",
            lambda game: MixtureLOTRBot(game, 0, num_simulations=N_ITERS,
                                        schedule=step(0.5, 0), epsilon=0.4,
                                        gamma=0.01, seed=SEED),
            history,
        ),
    }

    oos_records = bots["OOS"]._regret_records
    mix_records = bots["v4"]._regret_records
    mismatch_count, max_abs, first_mismatch = paired_compare(oos_records, mix_records)
    summaries = {name: summarize(bot._regret_records) for name, bot in bots.items()}

    print(f"iters={N_ITERS} seed={SEED} full-tree zero-regret policy, no updates "
          f"align_deterministic_tau={ALIGN_DETERMINISTIC_TAU}")
    print(f"OOS records={len(oos_records)}  "
          f"table max abs before={bots['OOS']._frozen_before_max_abs:.3g} "
          f"after={bots['OOS']._frozen_after_max_abs:.3g}")
    print(f"v4  records={len(mix_records)}  "
          f"D={bots['v4']._D} depth_to_position={dict(bots['v4']._depth_to_position)}  "
          f"table max abs before={bots['v4']._frozen_before_max_abs:.3g} "
          f"after={bots['v4']._frozen_after_max_abs:.3g}")
    print(f"paired row mismatches={mismatch_count}")
    print("max abs paired field diffs: " + "  ".join(
        f"{field}={max_abs[field]:.3g}" for field in FLOAT_FIELDS
    ))

    if first_mismatch is not None:
        idx, left, right = first_mismatch
        print("first paired mismatch:")
        print(f"  row={idx}")
        print(f"  OOS info={left['info']} action={left['action']} "
              f"sampled={left['sampled_action']} c={left['c']:.6f} "
              f"x={left['x']:.6f} pi_opp={left['pi_opp']:.6f} "
              f"l={left['l']:.6f} W={left['W']:.6f} "
              f"dR={left['delta_R']:.6f}")
        print(f"  v4  info={right['info']} action={right['action']} "
              f"sampled={right['sampled_action']} c={right['c']:.6f} "
              f"x={right['x']:.6f} pi_opp={right['pi_opp']:.6f} "
              f"l={right['l']:.6f} W={right['W']:.6f} "
              f"dR={right['delta_R']:.6f}")

    keys = sorted(set(summaries["OOS"]) | set(summaries["v4"]))
    print()
    print(f"{'key':22s} {'n_oos':>7s} {'n_v4':>7s} {'p_oos':>7s} {'p_v4':>7s} "
          f"{'mean_dR_oos':>12s} {'mean_dR_v4':>12s} {'diff':>11s} "
          f"{'W_oos':>9s} {'W_v4':>9s} {'l_oos':>9s} {'l_v4':>9s} "
          f"{'pi_oos':>9s} {'pi_v4':>9s}")
    diffs = []
    for key in keys:
        left = summaries["OOS"].get(key)
        right = summaries["v4"].get(key)
        if left is None:
            left = _zero_like_summary(summaries["v4"])
        if right is None:
            right = _zero_like_summary(summaries["OOS"])
        diff = right["mean_delta"] - left["mean_delta"]
        diffs.append((abs(diff), key, left, right, diff))
        print(f"{_fmt_key(key):22s} {left['n']:7.0f} {right['n']:7.0f} "
              f"{left['p']:7.4f} {right['p']:7.4f} "
              f"{left['mean_delta']:12.5f} {right['mean_delta']:12.5f} {diff:11.5f} "
              f"{left['mean_W']:9.4f} {right['mean_W']:9.4f} "
              f"{left['mean_l']:9.4f} {right['mean_l']:9.4f} "
              f"{left['mean_pi_opp']:9.4f} {right['mean_pi_opp']:9.4f}")

    tv = 0.5 * sum(abs(summaries["v4"].get(key, {"p": 0.0})["p"] -
                       summaries["OOS"].get(key, {"p": 0.0})["p"])
                   for key in keys)
    total_mass = {
        name: sum(summary[key]["mass_delta"] for key in summary)
        for name, summary in summaries.items()
    }
    total_abs_mass = {
        name: sum(summary[key]["mass_abs_delta"] for key in summary)
        for name, summary in summaries.items()
    }
    total_sq_mass = {
        name: sum(summary[key]["mass_sq_delta"] for key in summary)
        for name, summary in summaries.items()
    }
    print()
    print(f"key-distribution TV distance: {tv:.5f}")
    print(f"E[delta_R per action-record]: OOS={total_mass['OOS']:+.5f} "
          f"v4={total_mass['v4']:+.5f} diff={total_mass['v4'] - total_mass['OOS']:+.5f}")
    print(f"E[|delta_R| per action-record]: OOS={total_abs_mass['OOS']:.5f} "
          f"v4={total_abs_mass['v4']:.5f} diff={total_abs_mass['v4'] - total_abs_mass['OOS']:+.5f}")
    print(f"E[delta_R^2 per action-record]: OOS={total_sq_mass['OOS']:.5f} "
          f"v4={total_sq_mass['v4']:.5f} diff={total_sq_mass['v4'] - total_sq_mass['OOS']:+.5f}")
    print("largest mean delta_R differences:")
    for _abs_diff, key, left, right, diff in sorted(diffs, reverse=True)[:6]:
        print(f"  {_fmt_key(key):22s} diff={diff:+.5f}  "
              f"OOS(dR={left['mean_delta']:+.5f}, c={left['mean_c']:.4f}, "
              f"x={left['mean_x']:.4f}, pi_opp={left['mean_pi_opp']:.4f}, "
              f"l={left['mean_l']:.4f})  "
              f"v4(dR={right['mean_delta']:+.5f}, c={right['mean_c']:.4f}, "
              f"x={right['mean_x']:.4f}, pi_opp={right['mean_pi_opp']:.4f}, "
              f"l={right['mean_l']:.4f})")


def main():
    states = [
        ("root_P0_acts (D=1)", [0, 1]),
        ("after_P0_pass (D=2)", [0, 1, 0]),
        ("after_pb (D=3)", [0, 1, 0, 1]),
    ]
    for label, history in states:
        compare(label, history)


if __name__ == "__main__":
    main()