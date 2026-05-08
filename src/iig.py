"""Infoset Intersection Graph (IIG) construction for OpenSpiel games.

Builds a directed graph over information sets where edge J → I means:
  1. There exists a terminal history passing through both J and I
  2. Along at least one such history, J is exactly one player decision before I

Provides directed upstream BFS levels: given an active infoset I₀,
Level 0 = {I₀}, Level k = infosets with edges into Level k−1 (not yet assigned).
Levels only look *backward* — downstream nodes are not assigned levels.

Usage:
    import pyspiel
    from src.iig import IIG

    game = pyspiel.load_game("kuhn_poker")
    iig = IIG(game)
    iig.print_summary()
    iig.print_levels(some_infoset_id)
"""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass


@dataclass(frozen=True)
class InfosetNode:
    """Metadata for a single information set."""
    player: int
    info_state_string: str
    num_actions: int = 0


class IIG:
    """Infoset Intersection Graph built from a full game-tree traversal.

    Infoset IDs are ``(player, info_state_string)`` tuples — stable and
    unique across the game tree.
    """

    def __init__(self, game):
        self._game = game

        # Phase 1 & 2: DFS to collect infosets and terminal decision sequences
        self._infosets: dict[tuple, InfosetNode] = {}
        self._terminal_seqs: list[list[tuple]] = []
        self._z_sets: dict[tuple, set[int]] = defaultdict(set)
        self._terminal_count = 0
        self._terminal_histories: list[tuple] = []

        root = game.new_initial_state()
        self._traverse(root, [])

        # Phase 3: build directed edges from consecutive pairs
        self._successors: dict[tuple, set[tuple]] = defaultdict(set)
        self._predecessors: dict[tuple, set[tuple]] = defaultdict(set)
        self._build_edges()

        # Phase 4: lazy BFS cache
        self._level_cache: dict[tuple, dict[tuple, int]] = {}

    # ── Phase 1: DFS traversal ───────────────────────────────────────────

    def _traverse(self, state, decision_seq: list[tuple]):
        """Recursive DFS collecting decision-node infoset sequences."""
        if state.is_terminal():
            tid = self._terminal_count
            self._terminal_count += 1
            self._terminal_seqs.append(list(decision_seq))
            self._terminal_histories.append(tuple(state.history()))
            for iid in decision_seq:
                self._z_sets[iid].add(tid)
            return

        if state.is_chance_node():
            for action, _ in state.chance_outcomes():
                child = state.clone()
                child.apply_action(action)
                self._traverse(child, decision_seq)
            return

        # Player decision node
        player = state.current_player()
        info_key = state.information_state_string(player)
        iid = (player, info_key)

        if iid not in self._infosets:
            self._infosets[iid] = InfosetNode(player, info_key,
                                              len(state.legal_actions()))

        decision_seq.append(iid)
        for action in state.legal_actions():
            child = state.clone()
            child.apply_action(action)
            self._traverse(child, decision_seq)
        decision_seq.pop()

    # ── Phase 3: edge construction ───────────────────────────────────────

    def _build_edges(self):
        """Build directed edges from consecutive infoset pairs in each
        terminal history's decision sequence."""
        for seq in self._terminal_seqs:
            for j_id, i_id in zip(seq, seq[1:]):
                if j_id != i_id:  # skip self-loops
                    self._successors[j_id].add(i_id)
                    self._predecessors[i_id].add(j_id)

    # ── Infoset queries ──────────────────────────────────────────────────

    @property
    def infosets(self) -> dict[tuple, InfosetNode]:
        return self._infosets

    @property
    def num_infosets(self) -> int:
        return len(self._infosets)

    def infoset_ids_for_player(self, player: int) -> list[tuple]:
        return [iid for iid, node in self._infosets.items()
                if node.player == player]

    def num_actions(self, infoset_id: tuple) -> int:
        """Number of legal actions at this infoset."""
        return self._infosets[infoset_id].num_actions

    # ── Terminal history queries ─────────────────────────────────────────

    @property
    def num_terminals(self) -> int:
        return self._terminal_count

    def z_set(self, infoset_id: tuple) -> set[int]:
        """Terminal history IDs passing through this infoset."""
        return self._z_sets.get(infoset_id, set())

    def infosets_for_action(self, history_prefix: tuple,
                            action: int) -> set[tuple]:
        """Infoset IDs on terminals reachable from history_prefix + (action,).

        Works for both chance and decision nodes. Used by ISGT to compute
        IIG relevance of each outcome/action.
        """
        prefix_len = len(history_prefix)
        result = set()
        for tid, hist in enumerate(self._terminal_histories):
            if (len(hist) > prefix_len and
                    hist[prefix_len] == action and
                    hist[:prefix_len] == history_prefix):
                result.update(self._terminal_seqs[tid])
        return result

    # ── Edge queries ─────────────────────────────────────────────────────

    @property
    def edges(self) -> set[tuple[tuple, tuple]]:
        """All directed (J → I) edges."""
        result = set()
        for j, targets in self._successors.items():
            for i in targets:
                result.add((j, i))
        return result

    @property
    def num_edges(self) -> int:
        return sum(len(v) for v in self._successors.values())

    def predecessors(self, infoset_id: tuple) -> set[tuple]:
        """Infosets with a directed edge INTO this one."""
        return self._predecessors.get(infoset_id, set())

    def successors(self, infoset_id: tuple) -> set[tuple]:
        """Infosets this one has directed edges TO."""
        return self._successors.get(infoset_id, set())

    # ── Directed upstream BFS levels ─────────────────────────────────────

    def levels(self, infoset_id: tuple) -> dict[tuple, int]:
        """Upstream BFS distances from infoset_id.

        Only follows predecessor edges (J → I means J is upstream of I).
        Returns a dict mapping each reachable upstream infoset to its level.
        Level 0 = infoset_id itself.

        Raises KeyError if infoset_id is not in the graph.
        """
        if infoset_id not in self._infosets:
            raise KeyError(f"Unknown infoset: {infoset_id}")
        if infoset_id not in self._level_cache:
            self._level_cache[infoset_id] = self._bfs_upstream(infoset_id)
        return self._level_cache[infoset_id]

    def level_sets(self, infoset_id: tuple) -> dict[int, list[tuple]]:
        """Inverted view: level number → list of infoset IDs at that level."""
        dist = self.levels(infoset_id)
        result: dict[int, list[tuple]] = defaultdict(list)
        for iid, d in dist.items():
            result[d].append(iid)
        # Sort each level for deterministic output
        for lvl in result:
            result[lvl].sort()
        return dict(sorted(result.items()))

    def _bfs_upstream(self, root_id: tuple) -> dict[tuple, int]:
        """BFS following only predecessor (upstream) edges."""
        dist = {root_id: 0}
        queue = deque([root_id])
        while queue:
            node = queue.popleft()
            for pred in self.predecessors(node):
                if pred not in dist:
                    dist[pred] = dist[node] + 1
                    queue.append(pred)
        return dist

    # ── Inspection / debugging ───────────────────────────────────────────

    def _fmt_id(self, iid: tuple) -> str:
        """Human-readable infoset label: P0|info_string"""
        return f"P{iid[0]}|{iid[1]}"

    def print_summary(self):
        """Print overview statistics."""
        print("=" * 60)
        print(f"IIG Summary: {self._game}")
        print("=" * 60)
        print(f"  Infosets:           {self.num_infosets}")
        print(f"  Terminal histories: {self.num_terminals}")
        print(f"  Directed IIG edges: {self.num_edges}")
        print()

        # Per-player breakdown
        players = sorted(set(n.player for n in self._infosets.values()))
        for p in players:
            ids = self.infoset_ids_for_player(p)
            print(f"  Player {p}: {len(ids)} infosets")
        print()

        # Z-set size distribution
        z_sizes = [len(self._z_sets[iid]) for iid in self._infosets]
        print(f"  |Z(I)| range: {min(z_sizes)}–{max(z_sizes)}, "
              f"mean: {sum(z_sizes)/len(z_sizes):.1f}")
        print()

    def print_infosets(self):
        """Print all infosets with their Z-set sizes."""
        print("Infosets:")
        for iid in sorted(self._infosets):
            node = self._infosets[iid]
            z_size = len(self._z_sets[iid])
            print(f"  {self._fmt_id(iid):30s}  |Z| = {z_size}")
        print()

    def print_edges(self):
        """Print all directed IIG edges."""
        print(f"Directed IIG edges ({self.num_edges}):")
        for j, i in sorted(self.edges):
            print(f"  {self._fmt_id(j)}  →  {self._fmt_id(i)}")
        print()

    def print_levels(self, infoset_id: tuple):
        """Print upstream BFS levels from a given infoset."""
        lsets = self.level_sets(infoset_id)
        print(f"Upstream levels from {self._fmt_id(infoset_id)}:")
        for lvl, ids in lsets.items():
            labels = [self._fmt_id(iid) for iid in ids]
            print(f"  Level {lvl}: {{{', '.join(labels)}}}")

        # Show which infosets are NOT reachable upstream
        all_ids = set(self._infosets)
        reached = set(self.levels(infoset_id))
        unreached = all_ids - reached
        if unreached:
            labels = sorted(self._fmt_id(iid) for iid in unreached)
            print(f"  Not upstream: {{{', '.join(labels)}}}")
        print()
