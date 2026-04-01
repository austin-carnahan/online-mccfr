"""Shared display and flag-parsing utilities for algorithm output.

All MCCFR-family solvers in OpenSpiel share the same _infostates dict
structure (info_key -> [regrets, avg_strategy]), so a single set of
display functions works across outcome_sampling, OOS, ISGT, etc.

Default output: convergence table + game stats + per-player exploitability.

Optional flags:
    -s   Strategy table: average policy at every info set
    -r   Regret table: cumulative regrets at every info set
    -w   Visit weights: avg strategy accumulation per info set

Flags can be combined freely: -s, -s -r, -w, etc.
"""

import numpy as np
from open_spiel.python.algorithms import mccfr

from src.metrics import per_player_exploitability


# -- Flag parsing ----------------------------------------------------------

DISPLAY_FLAGS = {"-s", "-r", "-w"}


def parse_display_flags(args):
    """Split CLI args into positional args and a set of active display flags.

    Returns:
        (positional_args, flags) where flags is a set like {"-v", "-s"}.
    """
    flags = {a for a in args if a in DISPLAY_FLAGS}
    positional = [a for a in args if a not in DISPLAY_FLAGS]
    return positional, flags


# -- Game tree traversal (needed for readable action names) -----------------

def build_action_map(game):
    """Traverse the game tree to map info state key -> (player, action_ids, action_names).

    Uses state.action_to_string() which has full context, unlike the
    state-independent game.action_to_string() which returns generic IDs.
    """
    action_map = {}

    def visit(state):
        if state.is_terminal():
            return
        if state.is_chance_node():
            for action, _ in state.chance_outcomes():
                visit(state.child(action))
            return
        player = state.current_player()
        info_key = state.information_state_string(player)
        if info_key not in action_map:
            actions = list(state.legal_actions())
            names = [state.action_to_string(action) for action in actions]
            action_map[info_key] = (player, actions, names)
        for action in state.legal_actions():
            visit(state.child(action))

    visit(game.new_initial_state())
    return action_map


# -- Formatting helpers -----------------------------------------------------

def _format_info_key(key):
    """Flatten multi-line info state keys (e.g. goofspiel) to one line."""
    return key.replace('\n', ' | ').strip()


def _format_actions(probs, action_names):
    """Format action probabilities with readable names."""
    parts = []
    for i, p in enumerate(probs):
        name = action_names[i] if i < len(action_names) else f"a{i}"
        parts.append(f"{name}: {p:.3f}")
    return "  ".join(parts)


def _get_action_info(info_key, action_map, num_actions):
    """Look up action names for an info state, falling back to generic labels."""
    return action_map.get(
        info_key,
        (0, list(range(num_actions)), [f"a{i}" for i in range(num_actions)])
    )


# -- Convergence table (always printed) -------------------------------------

def print_results(algo_name, game_name, results):
    """Print the convergence table: iterations, exploitability, time."""
    print(f"\n{'=' * 55}")
    print(f"  {algo_name} on {game_name}")
    print(f"{'=' * 55}")
    print(f"  {'Iterations':>12s}  {'Exploitability':>16s}  {'Time (s)':>10s}")
    print(f"  {'-' * 12}  {'-' * 16}  {'-' * 10}")
    for itr, expl, elapsed in results:
        print(f"  {itr:>12,d}  {expl:>16.6f}  {elapsed:>10.2f}")
    print()


# -- Summary (always printed after convergence table) ----------------------

def print_summary(game, solver):
    """Game stats + per-player exploitability. Always shown."""
    policy = solver.average_policy()
    nash_conv, improvements = per_player_exploitability(game, policy)
    infostates = solver._infostates

    print(f"  {'—' * 50}")
    print(f"  Game stats")
    print(f"  {'—' * 50}")
    print(f"  Info sets visited:  {len(infostates)}")
    print(f"  Num players:        {game.num_players()}")
    print(f"  Max utility:        {game.max_utility()}")
    print()

    print(f"  {'—' * 50}")
    print(f"  Per-player exploitability")
    print(f"  {'—' * 50}")
    for p in range(game.num_players()):
        print(f"  Player {p}:  {improvements[p]:+.6f}")
    print(f"  NashConv:  {nash_conv:.6f}")
    print()


# -- -s: strategy table ----------------------------------------------------

def print_strategy_table(game, solver):
    """Average policy at every info set. One entry per info set visited."""
    infostates = solver._infostates
    action_map = build_action_map(game)

    print(f"  {'—' * 50}")
    print(f"  Average strategy")
    print(f"  {'—' * 50}")
    for info_key in sorted(infostates.keys()):
        avg_strat = infostates[info_key][mccfr.AVG_POLICY_INDEX]
        total = avg_strat.sum()
        probs = avg_strat / total if total > 0 else np.ones_like(avg_strat) / len(avg_strat)
        _, _, action_names = _get_action_info(info_key, action_map, len(probs))
        print(f"  {_format_info_key(info_key)}")
        print(f"    {_format_actions(probs, action_names)}")
    print()


# -- -r: regret table ------------------------------------------------------

def print_regret_table(game, solver):
    """Cumulative regrets per action at every info set. Shows what's driving strategy updates."""
    infostates = solver._infostates
    action_map = build_action_map(game)

    print(f"  {'—' * 50}")
    print(f"  Cumulative regrets")
    print(f"  {'—' * 50}")
    for info_key in sorted(infostates.keys()):
        regrets = infostates[info_key][mccfr.REGRET_INDEX]
        _, _, action_names = _get_action_info(info_key, action_map, len(regrets))
        parts = []
        for i, r in enumerate(regrets):
            name = action_names[i] if i < len(action_names) else f"a{i}"
            parts.append(f"{name}: {r:+.3f}")
        print(f"  {_format_info_key(info_key)}")
        print(f"    {'  '.join(parts)}")
    print()


# -- -w: visit weights -----------------------------------------------------

def print_visit_weights(solver):
    """Total avg strategy accumulation per info set. Proxy for sampling frequency."""
    infostates = solver._infostates

    print(f"  {'—' * 50}")
    print(f"  Visit weight (avg strategy accumulation)")
    print(f"  {'—' * 50}")
    for info_key in sorted(infostates.keys()):
        total = infostates[info_key][mccfr.AVG_POLICY_INDEX].sum()
        print(f"  {_format_info_key(info_key):<50s}  total: {total:.3f}")
    print()


# -- Dispatch ---------------------------------------------------------------

def print_output(game, solver, flags):
    """Print all requested sections based on flags set."""
    print_summary(game, solver)
    if "-s" in flags:
        print_strategy_table(game, solver)
    if "-r" in flags:
        print_regret_table(game, solver)
    if "-w" in flags:
        print_visit_weights(solver)
