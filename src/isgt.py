"""Information Set Graph Targeting (ISGT) — novel algorithm.

Extends OOS with proximity-weighted targeting over the information set graph.
The IS graph has information sets as nodes and player actions as edges.
Instead of binary target/don't-target, ISGT creates a continuous gradient
of sampling weight that decays with graph distance from the current
information set.

Levels:
  0 — current information set (equivalent to OOS/IST)
  1 — opponent's previous information set
  2 — player's info set two actions ago
  N — expanding shells through the IS graph neighborhood
"""

# TODO: implement IS graph targeting with decay function on top of OOS
