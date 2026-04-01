"""Exploitability and convergence measurement utilities."""

from open_spiel.python.algorithms import exploitability as expl_lib


def exploitability(game, policy):
    """Compute exploitability of a policy in a game (sum of both players' best-response improvements)."""
    return expl_lib.exploitability(game, policy)


def per_player_exploitability(game, policy):
    """Compute per-player improvements (how much each player gains by switching to best response).

    Returns (nash_conv, player_improvements) where player_improvements is a numpy array.
    """
    result = expl_lib.nash_conv(game, policy, return_only_nash_conv=False)
    return result.nash_conv, result.player_improvements
