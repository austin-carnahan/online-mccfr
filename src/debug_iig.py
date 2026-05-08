"""Debug script: build and inspect the IIG for Kuhn Poker.

Prints summary, all infosets, all edges, and upstream levels
from a few chosen infosets to verify correctness.

Usage:
    python -m src.debug_iig
"""

import pyspiel
from src.iig import IIG


def main():
    game = pyspiel.load_game("kuhn_poker")
    iig = IIG(game)

    # ── Overview ─────────────────────────────────────────────────────────
    iig.print_summary()
    iig.print_infosets()
    iig.print_edges()

    # ── Upstream levels from a few infosets ───────────────────────────────

    # OpenSpiel Kuhn notation: "p" = pass/check, "b" = bet
    # Card encoding: 0 = Jack, 1 = Queen, 2 = King
    # Player 0 = first to act, Player 1 = second to act

    # P0 holds Queen (card=1), history "pb" (pass then bet by P1)
    # This is our worked example: P1|Q|"cb"
    target = (0, "1pb")
    print("=" * 60)
    print("Detailed check: P0|1pb  (our P1|Q|cb example)")
    print("=" * 60)
    iig.print_levels(target)

    # A P1 infoset: P1 holds Jack, facing a bet
    target2 = (1, "0b")
    print("=" * 60)
    print("P1|0b — upstream levels")
    print("=" * 60)
    iig.print_levels(target2)

    # A root infoset (should have no upstream besides itself)
    root = (0, "0")  # P0 holds Jack, no history
    print("=" * 60)
    print("P0|0 (root) — upstream levels")
    print("=" * 60)
    iig.print_levels(root)

    # ── Z-set intersection check ─────────────────────────────────────────
    # Verify that P1|0b (P2 holds J, bet branch) and P0|1pb (P1|Q|cb)
    # share NO terminal histories
    z_pb = iig.z_set((0, "1pb"))
    z_jb = iig.z_set((1, "0b"))
    z_kb = iig.z_set((1, "2b"))
    print("=" * 60)
    print("Z-set intersection checks (P0|1pb = our P1|Q|cb)")
    print("=" * 60)
    print(f"  Z(P0|1pb) ∩ Z(P1|0b) = {z_pb & z_jb}  (expect empty)")
    print(f"  Z(P0|1pb) ∩ Z(P1|2b) = {z_pb & z_kb}  (expect empty)")
    z_jp = iig.z_set((1, "0p"))
    z_kp = iig.z_set((1, "2p"))
    print(f"  Z(P0|1pb) ∩ Z(P1|0p) = {z_pb & z_jp}  (expect non-empty)")
    print(f"  Z(P0|1pb) ∩ Z(P1|2p) = {z_pb & z_kp}  (expect non-empty)")
    print()


if __name__ == "__main__":
    main()
