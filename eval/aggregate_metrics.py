"""Extended aggregate exploitability with secondary metrics.

Wraps the core aggregate method (Section 4.2 of Lisý et al. 2015) and adds:
  1. Infoset coverage fraction (per player)
  2. Level-stratified weight profile (coverage + mean weight by IIG distance)
  3. Regret update magnitude by z* level (ISGT only, opt-in):
     a. active_regret_by_level: |Δregret| at I₀ only
     b. neighborhood_regret_by_level: total |Δregret| across upstream IIG neighborhood
"""

import numpy as np
import pyspiel

from collections import defaultdict

from open_spiel.python import policy as policy_lib
from open_spiel.python.bots.policy import PolicyBot

from src.games import load_game
from src.iig import IIG
from src.metrics import exploitability
from src.oos import OOSBot, AVG_POLICY_INDEX
from src.isgt import ISGTBot
from src.depth_delta import DepthDeltaBot


def aggregate_with_metrics(game, bot_factory, iig, num_matches=500,
                           seed=None, level_profile=False, num_sims=None):
    """Compute exploitability + secondary metrics via aggregate method.

    Args:
        game: OpenSpiel game instance.
        bot_factory: Callable(player_id) -> pyspiel.Bot.
        iig: Pre-built IIG instance for metric computation.
        num_matches: Number of matches to play vs random opponent.
        seed: Random seed.
        level_profile: If True, compute level-stratified weight profile.
        num_sims: Sims per move (needed for iteration-bucketed summaries).

    Returns:
        dict with keys:
          exploitability: float
          infoset_coverage: [float, float] per player
          level_profile: dict (if level_profile=True)
          regret_by_level: dict (if ISGT with regret tracking)
          neighborhood_regret_by_level: dict (if ISGT with regret tracking)
    """
    rng = np.random.RandomState(seed)

    # Global strategy accumulator
    global_strategy = [{}, {}]

    # Track active infosets encountered (for level profile)
    active_infosets = set()

    # Accumulate regret tracking data across matches (ISGT only)
    all_regret_by_level = defaultdict(list)
    all_neighborhood_regret_by_level = defaultdict(list)
    all_neighborhood_touched_by_level = defaultdict(list)
    all_neighborhood_size_by_level = defaultdict(list)
    all_sim_index_by_level = defaultdict(list)
    all_importance_weight_by_level = defaultdict(list)
    has_regret_data = False

    for match_idx in range(num_matches):
        search_player = match_idx % 2
        random_player = 1 - search_player

        search_bot = bot_factory(search_player)
        random_policy = policy_lib.UniformRandomPolicy(game)
        random_bot = PolicyBot(random_player, rng, random_policy)

        bots = [None, None]
        bots[search_player] = search_bot
        bots[random_player] = random_bot

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

                # Track active infosets for search bot decisions
                if player == search_player and level_profile:
                    info_key = state.information_state_string(player)
                    active_infosets.add((player, info_key))

                action = bots[player].step(state)
                for p, bot in enumerate(bots):
                    if p != player:
                        bot.inform_action(state, player, action)
                state.apply_action(action)

        # Merge strategy data
        _merge_bot_strategy(search_bot, search_player, global_strategy)

        # Collect regret tracking data from ISGT bots
        if isinstance(search_bot, ISGTBot) and search_bot._regret_tracking:
            has_regret_data = True
            for lvl, deltas in search_bot._regret_by_level.items():
                all_regret_by_level[lvl].extend(deltas)
            for lvl, deltas in search_bot._neighborhood_regret_by_level.items():
                all_neighborhood_regret_by_level[lvl].extend(deltas)
            for lvl, counts in search_bot._neighborhood_touched_by_level.items():
                all_neighborhood_touched_by_level[lvl].extend(counts)
            for lvl, sizes in search_bot._neighborhood_size_by_level.items():
                all_neighborhood_size_by_level[lvl].extend(sizes)
            for lvl, indices in search_bot._sim_index_by_level.items():
                all_sim_index_by_level[lvl].extend(indices)
            for lvl, weights in search_bot._importance_weight_by_level.items():
                all_importance_weight_by_level[lvl].extend(weights)

    # Primary metric
    from eval.aggregate_exploitability import AggregatePolicy
    agg_policy = AggregatePolicy(game, global_strategy)
    expl = exploitability(game, agg_policy)

    # Secondary metric 1: infoset coverage
    coverage = _compute_coverage(global_strategy, iig)

    result = {
        "exploitability": expl,
        "infoset_coverage": coverage,
    }

    # Secondary metric 2: level-stratified weight profile
    if level_profile:
        result["level_profile"] = _compute_level_profile(
            global_strategy, iig, active_infosets)

    # Secondary metric 3: regret by z* level
    if has_regret_data:
        result["regret_by_level"] = _summarize_regret_by_level(
            all_regret_by_level)
        result["neighborhood_regret_by_level"] = _summarize_regret_by_level(
            all_neighborhood_regret_by_level)
        result["neighborhood_touched_by_level"] = _summarize_touched_by_level(
            all_neighborhood_touched_by_level)
        result["neighborhood_size_by_level"] = _summarize_touched_by_level(
            all_neighborhood_size_by_level)
        result["importance_weight_by_level"] = _summarize_importance_weight_by_level(
            all_importance_weight_by_level)

        # Iteration-bucketed summaries (if sim indices available)
        if all_sim_index_by_level:
            result["bucketed_neighborhood_regret"] = (
                _summarize_bucketed_regret(
                    all_neighborhood_regret_by_level,
                    all_neighborhood_touched_by_level,
                    all_sim_index_by_level,
                    num_sims=num_sims))

    return result


def _merge_bot_strategy(bot, player_id, global_strategy):
    """Merge a bot's strategy data into the global accumulator."""
    if isinstance(bot, (OOSBot, ISGTBot, DepthDeltaBot)):
        for info_key, tables in bot._infostates.items():
            avg_strat = tables[AVG_POLICY_INDEX]
            if avg_strat.sum() == 0:
                continue
            for p in range(len(global_strategy)):
                if info_key not in global_strategy[p]:
                    global_strategy[p][info_key] = np.zeros_like(avg_strat)
                global_strategy[p][info_key] += avg_strat


def _compute_coverage(global_strategy, iig):
    """Compute per-player infoset coverage fraction."""
    coverage = []
    for p in range(2):
        player_iids = iig.infoset_ids_for_player(p)
        total = len(player_iids)
        if total == 0:
            coverage.append(0.0)
            continue
        covered = 0
        for iid in player_iids:
            info_key = iid[1]  # (player, info_state_string) → info_state_string
            if info_key in global_strategy[p]:
                data = global_strategy[p][info_key]
                if isinstance(data, np.ndarray) and data.sum() > 0:
                    covered += 1
        coverage.append(covered / total)
    return coverage


def _compute_level_profile(global_strategy, iig, active_infosets):
    """Compute level-stratified weight profile.

    For each active infoset, compute IIG levels.  Aggregate across all
    active infosets to get per-level coverage and mean weight.
    """
    # Collect all infosets organized by their minimum level across
    # all active infosets that reference them
    infoset_min_level = {}
    for active_iid in active_infosets:
        try:
            level_map = iig.levels(active_iid)
        except KeyError:
            continue
        for iid, lvl in level_map.items():
            if iid not in infoset_min_level:
                infoset_min_level[iid] = lvl
            else:
                infoset_min_level[iid] = min(infoset_min_level[iid], lvl)

    # Group by level
    level_infosets = defaultdict(set)
    for iid, lvl in infoset_min_level.items():
        level_infosets[lvl].add(iid)

    # Compute per-level metrics
    profile = {}
    for lvl in sorted(level_infosets.keys()):
        iids = level_infosets[lvl]
        n_infosets = len(iids)
        covered = 0
        total_weight = 0.0

        for iid in iids:
            player, info_key = iid
            if info_key in global_strategy[player]:
                data = global_strategy[player][info_key]
                if isinstance(data, np.ndarray):
                    w = float(data.sum())
                    if w > 0:
                        covered += 1
                        total_weight += w

        profile[str(lvl)] = {
            "coverage": covered / n_infosets if n_infosets > 0 else 0.0,
            "mean_weight": total_weight / n_infosets if n_infosets > 0 else 0.0,
            "n_infosets": n_infosets,
        }

    return profile


def _summarize_regret_by_level(regret_by_level):
    """Summarize raw regret deltas into per-level statistics."""
    summary = {}
    for lvl, deltas in sorted(regret_by_level.items()):
        arr = np.array(deltas)
        summary[str(lvl)] = {
            "mean_abs_delta": float(arr.mean()),
            "std_delta": float(arr.std()),
            "n_samples": len(deltas),
        }
    return summary


def _summarize_touched_by_level(touched_by_level):
    """Summarize per-iteration touched-infoset counts into per-level statistics."""
    summary = {}
    for lvl, counts in sorted(touched_by_level.items()):
        arr = np.array(counts, dtype=np.float64)
        summary[str(lvl)] = {
            "mean_touched": float(arr.mean()),
            "std_touched": float(arr.std()),
            "n_samples": len(counts),
        }
    return summary


def _summarize_importance_weight_by_level(iw_by_level):
    """Summarize realized importance weights (l) per z* level."""
    summary = {}
    for lvl, weights in sorted(iw_by_level.items()):
        arr = np.array(weights, dtype=np.float64)
        summary[str(lvl)] = {
            "mean_l": float(arr.mean()),
            "std_l": float(arr.std()),
            "min_l": float(arr.min()),
            "max_l": float(arr.max()),
            "median_l": float(np.median(arr)),
            "n_samples": len(weights),
        }
    return summary


def _summarize_bucketed_regret(nbr_regret_by_level, nbr_touched_by_level,
                               sim_index_by_level, num_sims):
    """Summarize neighborhood regret bucketed by iteration phase.

    Splits observations into 'early' (first 20% of sims within a step)
    and 'late' (last 20%), to detect proximity effects that appear only
    when the local tree is unsaturated.

    Returns dict keyed by level, each containing 'early', 'late', 'all'
    sub-dicts with per-node regret statistics.
    """
    if num_sims is None or num_sims <= 0:
        return {}

    early_cutoff = max(1, int(num_sims * 0.2))
    late_start = int(num_sims * 0.8)

    summary = {}
    for lvl in sorted(nbr_regret_by_level.keys()):
        deltas = nbr_regret_by_level[lvl]
        touched = nbr_touched_by_level.get(lvl, [0] * len(deltas))
        indices = sim_index_by_level.get(lvl, [])

        if len(indices) != len(deltas):
            # Fallback: no sim indices available for this level
            continue

        buckets = {}
        for bucket_name, cond_fn in [
            ("early", lambda s: s < early_cutoff),
            ("late", lambda s: s >= late_start),
            ("all", lambda s: True),
        ]:
            b_deltas = []
            b_touched = []
            for i, sim_idx in enumerate(indices):
                if cond_fn(sim_idx):
                    b_deltas.append(deltas[i])
                    b_touched.append(touched[i])

            if not b_deltas:
                buckets[bucket_name] = {
                    "mean_nbr_delta": 0.0,
                    "mean_touched": 0.0,
                    "per_node": 0.0,
                    "n_samples": 0,
                }
                continue

            arr_d = np.array(b_deltas)
            arr_t = np.array(b_touched, dtype=np.float64)
            mean_d = float(arr_d.mean())
            mean_t = float(arr_t.mean())
            per_node = mean_d / mean_t if mean_t > 0 else 0.0

            buckets[bucket_name] = {
                "mean_nbr_delta": mean_d,
                "std_nbr_delta": float(arr_d.std()),
                "mean_touched": mean_t,
                "per_node": per_node,
                "n_samples": len(b_deltas),
            }

        summary[str(lvl)] = buckets

    return summary
