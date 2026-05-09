"""Average-strategy delta parity diagnostic for OOS vs Mixture LOTR.

Logs every opponent-node average-strategy update row:

    info, action, policy[a], opp_reach, denominator, delta_avg

where

    delta_avg = opp_reach * policy[a] / denominator

OOS uses denominator = delta*s1 + (1-delta)*s2 at the opponent node.
Mixture LOTR uses denominator = l_here = dot(arm_probs, s) at the same
node. Regret and average-strategy updates are still applied, so this is
a learning-run diagnostic rather than a frozen-policy diagnostic.
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
FLOAT_FIELDS = ("policy", "opp_reach", "denom", "delta_avg")
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


def _record_avg_rows(bot, update_player, info_key, legal_actions,
                     sampled_idx, policy, opp_reach, denom):
    for action_idx, action in enumerate(legal_actions):
        delta_avg = opp_reach * policy[action_idx] / max(denom, 1e-30)
        bot._avg_records.append({
            "iter": bot._trace_iter,
            "update_player": update_player,
            "info": info_key,
            "action": int(action),
            "action_idx": action_idx,
            "sampled_action": int(legal_actions[sampled_idx]),
            "sampled_idx": int(sampled_idx),
            "policy": float(policy[action_idx]),
            "opp_reach": float(opp_reach),
            "denom": float(denom),
            "delta_avg": float(delta_avg),
        })


def _install_episode_counter(bot, episode_attr):
    bot._avg_records = []
    bot._trace_iter = -1
    orig_episode = getattr(bot, episode_attr)

    def episode_wrapper(state, update_player):
        bot._trace_iter += 1
        return orig_episode(state, update_player)

    setattr(bot, episode_attr, episode_wrapper)


def install_oos_avg_trace(bot):
    _install_episode_counter(bot, "_oos_episode")

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
            bot._infostates[info_key] = [
                np.zeros(num_actions, dtype=np.float64),
                np.zeros(num_actions, dtype=np.float64),
            ]
            action_idx = bot._rng.randint(num_actions)
            state.apply_action(legal_actions[action_idx])
            l_playout = (bot._delta * s1 + (1.0 - bot._delta) * s2) / num_actions
            x, l, u = bot._playout(state, update_player, l_playout)
            return (x / num_actions, l, u)

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
            if oos_mod._HAS_FAST_OPS:
                oos_mod.update_regrets(
                    bot._infostates[info_key][oos_mod.REGRET_INDEX],
                    sampled_idx, c, x, W, num_actions)
            else:
                for action_idx in range(num_actions):
                    if action_idx == sampled_idx:
                        bot._infostates[info_key][oos_mod.REGRET_INDEX][action_idx] += (c - x) * W
                    else:
                        bot._infostates[info_key][oos_mod.REGRET_INDEX][action_idx] += -x * W
        else:
            sample_prob = bot._delta * s1 + (1.0 - bot._delta) * s2
            _record_avg_rows(bot, update_player, info_key, legal_actions,
                             sampled_idx, policy, opp_reach, sample_prob)
            if oos_mod._HAS_FAST_OPS:
                oos_mod.update_avg_strategy(
                    bot._infostates[info_key][oos_mod.AVG_POLICY_INDEX],
                    policy, opp_reach, sample_prob, num_actions)
            else:
                for action_idx in range(num_actions):
                    bot._infostates[info_key][oos_mod.AVG_POLICY_INDEX][action_idx] += (
                        opp_reach * policy[action_idx] / max(sample_prob, 1e-30)
                    )

        return (x, l, u)

    bot._walk = walk_wrapper


def install_mixture_avg_trace(bot):
    _install_episode_counter(bot, "_episode")

    orig_sample_realized_action = bot._sample_realized_action

    def sample_realized_action_wrapper(target_idx, pi, tau, num_actions):
        if ALIGN_DETERMINISTIC_TAU:
            if tau >= 1.0:
                return target_idx, False
            if tau <= 0.0:
                return int(bot._rng.choice(num_actions, p=pi)), True
        return orig_sample_realized_action(target_idx, pi, tau, num_actions)

    bot._sample_realized_action = sample_realized_action_wrapper

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
            bot._infostates[info_key] = [
                np.zeros(num_actions, dtype=np.float64),
                np.zeros(num_actions, dtype=np.float64),
            ]
            action_idx = bot._rng.randint(num_actions)
            uniform_p = 1.0 / num_actions
            new_s = s * uniform_p
            state.apply_action(legal_actions[action_idx])
            x, l_out, u = bot._playout(state, update_player, new_s)
            return (x * uniform_p, l_out, u)

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

        l_here = bot._l_from_s(s)

        state.apply_action(action)
        suffix_x, l_out, u = bot._walk(state, update_player,
                                       new_my_reach, new_opp_reach,
                                       new_s, new_diverged)

        c = suffix_x
        x = suffix_x * policy[sampled_idx]

        if cur_player == update_player:
            W = u * opp_reach / max(l_out, 1e-30)
            if mix_mod._HAS_FAST_OPS:
                mix_mod.update_regrets(
                    bot._infostates[info_key][mix_mod.REGRET_INDEX],
                    sampled_idx, c, x, W, num_actions)
            else:
                for action_idx in range(num_actions):
                    if action_idx == sampled_idx:
                        bot._infostates[info_key][mix_mod.REGRET_INDEX][action_idx] += (c - x) * W
                    else:
                        bot._infostates[info_key][mix_mod.REGRET_INDEX][action_idx] += -x * W
        else:
            _record_avg_rows(bot, update_player, info_key, legal_actions,
                             sampled_idx, policy, opp_reach, l_here)
            if mix_mod._HAS_FAST_OPS:
                mix_mod.update_avg_strategy(
                    bot._infostates[info_key][mix_mod.AVG_POLICY_INDEX],
                    policy, opp_reach, l_here, num_actions)
            else:
                for action_idx in range(num_actions):
                    bot._infostates[info_key][mix_mod.AVG_POLICY_INDEX][action_idx] += (
                        opp_reach * policy[action_idx] / max(l_here, 1e-30)
                    )

        return (x, l_out, u)

    bot._walk = walk_wrapper


def run_bot(name, factory, history):
    game, state = make_state(history)
    bot = factory(game)
    if name.startswith("OOS"):
        install_oos_avg_trace(bot)
    else:
        install_mixture_avg_trace(bot)
    bot._num_simulations = N_ITERS
    bot.step_with_policy(state)
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
        delta = values["delta_avg"]
        summary[key] = {
            "n": len(rows),
            "p": len(rows) / total if total else 0.0,
            "mean_delta": float(delta.mean()),
            "rms_delta": float(np.sqrt(np.mean(delta * delta))),
            "mean_policy": float(values["policy"].mean()),
            "mean_opp_reach": float(values["opp_reach"].mean()),
            "mean_denom": float(values["denom"].mean()),
        }
        summary[key]["mass_delta"] = summary[key]["p"] * summary[key]["mean_delta"]
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
    print("\n" + "=" * 92)
    print(f"STATE: history={history}  ({label})")
    print("=" * 92)

    bots = {
        "OOS": run_bot(
            "OOS_avg",
            lambda game: OOSBot(game, 0, num_simulations=N_ITERS, delta=0.5,
                                epsilon=0.4, gamma=0.01, seed=SEED),
            history,
        ),
        "v4": run_bot(
            "v4_avg",
            lambda game: MixtureLOTRBot(game, 0, num_simulations=N_ITERS,
                                        schedule=step(0.5, 0), epsilon=0.4,
                                        gamma=0.01, seed=SEED),
            history,
        ),
    }

    oos_records = bots["OOS"]._avg_records
    mix_records = bots["v4"]._avg_records
    mismatch_count, max_abs, first_mismatch = paired_compare(oos_records, mix_records)
    summaries = {name: summarize(bot._avg_records) for name, bot in bots.items()}

    print(f"iters={N_ITERS} seed={SEED} learning run "
          f"align_deterministic_tau={ALIGN_DETERMINISTIC_TAU}")
    print(f"OOS avg-records={len(oos_records)}")
    print(f"v4  avg-records={len(mix_records)}  "
          f"D={bots['v4']._D} depth_to_position={dict(bots['v4']._depth_to_position)}")
    print(f"paired row mismatches={mismatch_count}")
    print("max abs paired field diffs: " + "  ".join(
        f"{field}={max_abs[field]:.3g}" for field in FLOAT_FIELDS
    ))

    if first_mismatch is not None:
        idx, left, right = first_mismatch
        print("first paired mismatch:")
        print(f"  row={idx}")
        print(f"  OOS info={left['info']} action={left['action']} "
              f"sampled={left['sampled_action']} policy={left['policy']:.6f} "
              f"opp={left['opp_reach']:.6f} denom={left['denom']:.6f} "
              f"davg={left['delta_avg']:.6f}")
        print(f"  v4  info={right['info']} action={right['action']} "
              f"sampled={right['sampled_action']} policy={right['policy']:.6f} "
              f"opp={right['opp_reach']:.6f} denom={right['denom']:.6f} "
              f"davg={right['delta_avg']:.6f}")

    keys = sorted(set(summaries["OOS"]) | set(summaries["v4"]))
    print()
    print(f"{'key':22s} {'n_oos':>7s} {'n_v4':>7s} {'p_oos':>7s} {'p_v4':>7s} "
          f"{'mean_avg_oos':>12s} {'mean_avg_v4':>12s} {'diff':>11s} "
          f"{'den_oos':>9s} {'den_v4':>9s} {'opp_oos':>9s} {'opp_v4':>9s}")
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
              f"{left['mean_denom']:9.4f} {right['mean_denom']:9.4f} "
              f"{left['mean_opp_reach']:9.4f} {right['mean_opp_reach']:9.4f}")

    tv = 0.5 * sum(abs(summaries["v4"].get(key, {"p": 0.0})["p"] -
                       summaries["OOS"].get(key, {"p": 0.0})["p"])
                   for key in keys)
    total_mass = {
        name: sum(summary[key]["mass_delta"] for key in summary)
        for name, summary in summaries.items()
    }
    total_sq_mass = {
        name: sum(summary[key]["mass_sq_delta"] for key in summary)
        for name, summary in summaries.items()
    }
    print()
    print(f"key-distribution TV distance: {tv:.5f}")
    print(f"E[delta_avg per action-record]: OOS={total_mass['OOS']:.5f} "
          f"v4={total_mass['v4']:.5f} diff={total_mass['v4'] - total_mass['OOS']:+.5f}")
    print(f"E[delta_avg^2 per action-record]: OOS={total_sq_mass['OOS']:.5f} "
          f"v4={total_sq_mass['v4']:.5f} diff={total_sq_mass['v4'] - total_sq_mass['OOS']:+.5f}")
    print("largest mean delta_avg differences:")
    for _abs_diff, key, left, right, diff in sorted(diffs, reverse=True)[:6]:
        print(f"  {_fmt_key(key):22s} diff={diff:+.5f}  "
              f"OOS(avg={left['mean_delta']:.5f}, denom={left['mean_denom']:.4f}, "
              f"opp={left['mean_opp_reach']:.4f})  "
              f"v4(avg={right['mean_delta']:.5f}, denom={right['mean_denom']:.4f}, "
              f"opp={right['mean_opp_reach']:.4f})")


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