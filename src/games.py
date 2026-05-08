"""Game loading helpers and configuration for OpenSpiel benchmarks."""

import pyspiel

# Game specs used across benchmarks.
# Each maps a short name to the pyspiel.load_game() call.
# Note: goofspiel is simultaneous, so we wrap it in turn_based_simultaneous_game.
GAME_SPECS = {
    "kuhn_poker": "kuhn_poker",
    "leduc_poker": "leduc_poker",
    "liars_dice": "liars_dice",
    "goofspiel": "turn_based_simultaneous_game(game=goofspiel(num_cards=4,imp_info=true,points_order=descending))",
}


def load_game(name: str) -> pyspiel.Game:
    """Load an OpenSpiel game by short name."""
    if name not in GAME_SPECS:
        raise ValueError(f"Unknown game '{name}'. Available: {list(GAME_SPECS)}")
    return pyspiel.load_game(GAME_SPECS[name])
