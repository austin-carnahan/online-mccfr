"""Compatibility aliases for the renamed LOTR implementation.

New code should import from ``src.lotr`` and use ``LOTRBot`` / ``LOTRPolicy``.
This module keeps older experiment scripts importable while we migrate them.
"""

from src import lotr as _lotr
from src.lotr import *  # noqa: F401,F403

MixtureLOTRBot = _lotr.LOTRBot
MixtureLOTRPolicy = _lotr.LOTRPolicy

__all__ = list(_lotr.__all__) + ["MixtureLOTRBot", "MixtureLOTRPolicy"]
