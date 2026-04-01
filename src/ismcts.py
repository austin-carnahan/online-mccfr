"""IS-MCTS bot wrapper using OpenSpiel's built-in ISMCTSBot.

Wraps the ISMCTSBot with a RandomRolloutEvaluator for use in our
online match framework. Provides a factory function that handles
all the setup boilerplate.
"""

import numpy as np
import pyspiel

from open_spiel.python.algorithms.ismcts import (
    ISMCTSBot,
    ISMCTSFinalPolicyType,
    ChildSelectionPolicy,
)
from open_spiel.python.algorithms.mcts import RandomRolloutEvaluator


def make_ismcts_bot(game, player_id, max_simulations=1000, uct_c=None,
                    random_state=None):
    """Create an IS-MCTS bot with sensible defaults.

    Args:
        game: OpenSpiel game instance.
        player_id: Which player this bot controls (0 or 1).
        max_simulations: Number of MCTS simulations per decision.
        uct_c: UCT exploration constant. Defaults to 2 * max_utility.
        random_state: numpy RandomState for reproducibility.

    Returns:
        An ISMCTSBot instance (extends pyspiel.Bot).
    """
    if uct_c is None:
        uct_c = 2.0 * game.max_utility()

    if random_state is None:
        random_state = np.random.RandomState()

    evaluator = RandomRolloutEvaluator(n_rollouts=1, random_state=random_state)

    bot = ISMCTSBot(
        game=game,
        evaluator=evaluator,
        uct_c=uct_c,
        max_simulations=max_simulations,
        random_state=random_state,
        final_policy_type=ISMCTSFinalPolicyType.NORMALIZED_VISITED_COUNT,
        child_selection_policy=ChildSelectionPolicy.UCT,
    )

    # Attach a custom resampler for games that don't implement
    # ResampleFromInfostate() natively (e.g. liars_dice).
    resampler = _make_resampler(game, random_state)
    if resampler is not None:
        bot.set_resampler(resampler)

    return bot


def _make_resampler(game, rng):
    """Return a resampler callback for games that need one, or None."""
    game_name = game.get_type().short_name
    if game_name == "liars_dice":
        return _liars_dice_resampler(game, rng)
    # turn_based_simultaneous_game wrapping goofspiel
    if game_name == "turn_based_simultaneous_game":
        params = game.get_parameters()
        if params.get("game", {}).get("name") == "goofspiel":
            return _goofspiel_resampler(game, rng)
    return None


def _liars_dice_resampler(game, rng):
    """Build a resampler for Liar's Dice.

    Liar's Dice has exactly 2 initial chance nodes (one die per player),
    then all actions are public bids. To resample from a player's
    information set: keep their own die, re-roll the opponent's die
    uniformly, then replay all public bid actions.
    """
    num_faces = len(game.new_initial_state().chance_outcomes())

    def resampler(state, player_id):
        history = state.history()
        own_die = history[player_id]           # 0 or 1 index
        opp_die = rng.randint(num_faces)       # uniform resample
        bids = history[2:]                     # all actions after the 2 dice

        new_state = game.new_initial_state()
        if player_id == 0:
            new_state.apply_action(own_die)
            new_state.apply_action(opp_die)
        else:
            new_state.apply_action(opp_die)
            new_state.apply_action(own_die)

        for action in bids:
            new_state.apply_action(action)

        return new_state

    return resampler


def _goofspiel_resampler(game, rng):
    """Build a resampler for Imperfect-Information Goofspiel.

    In II-Goofspiel (turn_based_simultaneous wrapper), the history is
    [p0_bid0, p1_bid0, p0_bid1, p1_bid1, ...].  Each player knows their
    own bids and the win/loss/tie result of each round, but not the
    opponent's specific bids.  To resample: enumerate opponent bid
    permutations consistent with the observed results and pick uniformly.
    """
    num_cards = game.get_parameters()["game"]["num_cards"]

    def resampler(state, player_id):
        history = state.history()
        if not history:
            return state.clone()

        full_rounds = len(history) // 2
        partial = len(history) % 2  # 1 if p0 has bid but p1 hasn't

        # Separate my bids and opponent bids for completed rounds
        my_bids = []
        opp_bids = []
        for r in range(full_rounds):
            p0_bid = history[2 * r]
            p1_bid = history[2 * r + 1]
            if player_id == 0:
                my_bids.append(p0_bid)
                opp_bids.append(p1_bid)
            else:
                my_bids.append(p1_bid)
                opp_bids.append(p0_bid)

        # Determine win/loss/tie constraints per round
        constraints = []
        for r in range(full_rounds):
            if my_bids[r] > opp_bids[r]:
                constraints.append("my_win")
            elif my_bids[r] < opp_bids[r]:
                constraints.append("opp_win")
            else:
                constraints.append("tie")

        # Enumerate valid opponent bid sequences via backtracking
        all_cards = set(range(num_cards))
        valid_seqs = []
        _find_valid_opp_seqs(all_cards, my_bids, constraints, 0, [], valid_seqs)

        if not valid_seqs:
            return state.clone()

        new_opp_bids = valid_seqs[rng.randint(len(valid_seqs))]

        # Replay the game with my bids + resampled opponent bids
        new_state = game.new_initial_state()
        for r in range(full_rounds):
            if player_id == 0:
                new_state.apply_action(my_bids[r])
                new_state.apply_action(new_opp_bids[r])
            else:
                new_state.apply_action(new_opp_bids[r])
                new_state.apply_action(my_bids[r])

        if partial:
            if player_id == 0:
                # I (p0) already bid; replay my pending bid
                new_state.apply_action(history[-1])
            else:
                # p0 bid is hidden from me (p1); resample it
                used_p0 = set(new_opp_bids)
                remaining_p0 = list(all_cards - used_p0)
                new_state.apply_action(remaining_p0[rng.randint(len(remaining_p0))])

        return new_state

    return resampler


def _find_valid_opp_seqs(available, my_bids, constraints, r, current, results):
    """Backtracking search for opponent bid sequences matching constraints."""
    if r == len(constraints):
        results.append(list(current))
        return

    my_bid = my_bids[r]
    constraint = constraints[r]

    for card in sorted(available):
        if constraint == "my_win" and card < my_bid:
            valid = True
        elif constraint == "opp_win" and card > my_bid:
            valid = True
        elif constraint == "tie" and card == my_bid:
            valid = True
        else:
            valid = False

        if valid:
            current.append(card)
            _find_valid_opp_seqs(available - {card}, my_bids, constraints,
                                r + 1, current, results)
            current.pop()
