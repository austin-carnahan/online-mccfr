"""Information Set Graph Targeting (ISGT) — IIG-guided online MCCFR.

MCCFR with trajectory-level IIG targeting. Before each iteration, a
target terminal history z* is sampled from an IIG-weighted distribution:
histories passing through infosets close to the active decision point I₀
get higher weight.

Two bias modes control how z* guides the walk:

  "chance" — Chance nodes follow z* deterministically; decision nodes
             sample via ε-on-policy independently. Uses chance-action
             iterator (safe across decision deviations).
             sample_reach = P(z*) × Π(decision sample probs).

  "full"   — Anchor-split walk: deterministic from root to the anchor
             infoset (shallowest IIG-level-k node on z*'s path), then
             ε-on-policy for all nodes beyond the anchor. Combines
             proximity targeting with exploration in the suffix.

When no target is set (root convergence), all nodes sample from natural /
uniform distributions — equivalent to vanilla MCCFR.

A per-iteration δ parameter (0 ≤ δ ≤ 1) controls the fraction of
iterations that are targeted. With probability δ, z* guides the walk;
with probability 1-δ, an untargeted iteration uses independent chance
sampling. Both sample-reach probabilities (s1 targeted, s2 untargeted)
are tracked for correct importance weighting: l = δ·s1 + (1-δ)·s2.

Architecture:
  ISGTBot(pyspiel.Bot)
    ├── IIG precompute (from src.iig)
    ├── LevelWeightFn  — pluggable decay over IIG levels
    ├── _sample_target() — select target history from IIG-weighted distribution
    ├── _walk()          — recursive MCCFR traversal guided by target history
    └── ISGTPolicy       — adapter for exploitability computation
"""

from __future__ import annotations

import abc
from dataclasses import dataclass, field

import numpy as np
import pyspiel

from src.iig import IIG

# Index constants matching OpenSpiel's mccfr module
REGRET_INDEX = 0
AVG_POLICY_INDEX = 1


# ═════════════════════════════════════════════════════════════════════════
# Level weight functions
# ═════════════════════════════════════════════════════════════════════════

class LevelWeightFn(abc.ABC):
    """Interface for IIG level budget allocation.

    Given a level and the max level in the current neighborhood,
    returns an unnormalized weight that controls the **fraction of
    simulation budget allocated to that IIG level**:

        P(level=ℓ) ∝ f(ℓ)

    Within each level, terminals are sampled uniformly.

    e.g. LevelUniform → equal budget per level;
         LevelExponential(0.7) → 70% as much budget at each deeper level.
    """

    @abc.abstractmethod
    def weight(self, level: int, max_level: int) -> float:
        """Return the unnormalized weight for a given IIG level.

        Args:
            level: upstream BFS distance (0 = active infoset)
            max_level: maximum level in the current IIG neighborhood

        Returns:
            Non-negative weight. Higher = more sampling probability.
        """
        ...

    @abc.abstractmethod
    def name(self) -> str:
        """Short identifier for logging."""
        ...


class LevelExponential(LevelWeightFn):
    """f(ℓ) = alpha^ℓ.  Each deeper level gets alpha× the budget of the
    previous level. alpha=1 → level-uniform, alpha→0 → level-0 only."""

    def __init__(self, alpha: float = 0.5):
        self.alpha = alpha

    def weight(self, level: int, max_level: int) -> float:
        return self.alpha ** level

    def name(self) -> str:
        return f"level_exp(α={self.alpha})"


# Backward-compat alias
ExponentialDecay = LevelExponential


class LevelPolynomial(LevelWeightFn):
    """f(ℓ) = 1 / (1 + ℓ)^p.  Polynomial level-budget decay."""

    def __init__(self, p: float = 2.0):
        self.p = p

    def weight(self, level: int, max_level: int) -> float:
        return 1.0 / (1.0 + level) ** self.p

    def name(self) -> str:
        return f"level_poly(p={self.p})"


# Backward-compat alias
PolynomialDecay = LevelPolynomial


class LevelLinear(LevelWeightFn):
    """f(ℓ) = max(0, 1 - ℓ/D).  Linear level-budget decay, zero beyond D."""

    def __init__(self, D: int = 4):
        self.D = D

    def weight(self, level: int, max_level: int) -> float:
        return max(0.0, 1.0 - level / self.D)

    def name(self) -> str:
        return f"level_linear(D={self.D})"


# Backward-compat alias
LinearDecay = LevelLinear


class LevelStep(LevelWeightFn):
    """f(ℓ) = 1 if ℓ == 0, else floor.  Nearly all budget at level 0 (OOS-like)."""

    def __init__(self, floor: float = 0.01):
        self.floor = floor

    def weight(self, level: int, max_level: int) -> float:
        return 1.0 if level == 0 else self.floor

    def name(self) -> str:
        return f"level_step(floor={self.floor})"


# Backward-compat alias
StepFunction = LevelStep


class LevelUniform(LevelWeightFn):
    """f(ℓ) = 1.0 for all levels.  Equal budget per IIG level."""

    def weight(self, level: int, max_level: int) -> float:
        return 1.0

    def name(self) -> str:
        return "level_uniform"


# Backward-compat alias
ConstantWeight = LevelUniform


class TerminalBalancedWeight(LevelWeightFn):
    """Deprecated: use LevelUniform instead.

    Under level-budget semantics, this is equivalent to LevelUniform.
    Kept for backward compatibility with archived experiments.
    """

    def weight(self, level: int, max_level: int) -> float:
        return 1.0

    def name(self) -> str:
        return "terminal_balanced"


class SqrtBalancedWeight(LevelWeightFn):
    """Deprecated: use LevelUniform instead.

    Kept for backward compatibility with archived experiments.
    """

    def weight(self, level: int, max_level: int) -> float:
        return 1.0

    def name(self) -> str:
        return "sqrt_balanced"


# ═════════════════════════════════════════════════════════════════════════
# Sample metadata (for inspection / debugging)
# ═════════════════════════════════════════════════════════════════════════

@dataclass
class ISGTDebugInfo:
    """Metadata from one ISGT step, for inspection."""
    active_infoset: tuple = None
    num_levels: int = 0
    level_weights: dict = field(default_factory=dict)
    infoset_weights: dict = field(default_factory=dict)


# ═════════════════════════════════════════════════════════════════════════
# ISGTBot
# ═════════════════════════════════════════════════════════════════════════

class ISGTBot(pyspiel.Bot):
    """IIG-guided MCCFR bot with trajectory-level targeting.

    Before each iteration, a target terminal history z* is sampled
    from a distribution weighted by IIG proximity to the active
    infoset I₀.

    bias_mode controls how z* guides the walk:
      "chance" — only chance nodes follow z*; decisions sample ε-on-policy.
      "full"   — anchor-split: deterministic until z*'s IIG anchor node,
                 then ε-on-policy for the suffix.
    """

    def __init__(self, game, player_id, num_simulations=1000,
                 epsilon=0.6, gamma=0.01,
                 level_weight_fn=None, bias_mode="full", seed=None,
                 iig=None, delta=1.0, max_iig_depth=None):
        """
        Args:
            game: OpenSpiel game instance.
            player_id: Which player this bot controls.
            num_simulations: Iterations per decision point.
            epsilon: Exploration factor for update player actions.
            gamma: Regret matching floor.
            level_weight_fn: LevelWeightFn instance. Defaults to LevelExponential(0.5).
            bias_mode: "chance" (bias chance only) or "full" (bias all nodes).
            seed: Random seed.
            iig: Pre-built IIG instance (shared across bots to avoid redundant construction).
            delta: Targeting probability per iteration (0=pure MCCFR, 1=always targeted).
            max_iig_depth: Maximum IIG BFS depth for targeting. Levels beyond this\n                are treated as floor (zero targeting probability). None = unlimited.
        """
        pyspiel.Bot.__init__(self)
        self._game = game
        self._player_id = player_id
        self._num_simulations = num_simulations
        self._epsilon = epsilon
        self._gamma = gamma
        self._level_weight_fn = level_weight_fn or LevelExponential(0.5)
        self._bias_mode = bias_mode
        self._delta = delta
        self._max_iig_depth = max_iig_depth
        self._rng = np.random.RandomState(seed)
        self._num_players = game.num_players()

        # Use provided IIG or build one (expensive for large games)
        self._iig = iig if iig is not None else IIG(game)

        # Pre-initialize strategy tables from IIG (eliminates random playouts
        # on first visit — proper regret matching from iteration 1)
        self._infostates = {}
        for iid in self._iig.infosets:
            n_actions = self._iig.num_actions(iid)
            self._infostates[iid[1]] = [
                np.zeros(n_actions, dtype=np.float64),
                np.zeros(n_actions, dtype=np.float64),
            ]

        # Match state tracking
        self._match_history = []

        # Per-decision debug info (overwritten each step)
        self.last_debug = ISGTDebugInfo()

        # IIG weight cache: active_iid → {iid → weight}
        self._iig_weight_cache: dict[tuple, dict[tuple, float]] = {}

        # Target distribution cache: active_iid → (terminal_weights, total)
        self._target_dist_cache: dict[tuple, np.ndarray] = {}

        # Per-iteration targeting state (set before each walk)
        self._z_star_history = None         # full z* action tuple (always stored)
        self._z_star_chance_actions = []    # z*'s chance actions (always stored)
        self._z_star_chance_idx = 0         # index for chance action tracking
        self._is_targeted_iter = True       # whether current iteration follows z*
        self._last_z_star_tid = -1          # terminal ID of last sampled z*
        self._anchor_depth = 0              # game-tree depth of anchor infoset

        # Opt-in regret tracking: records per-iteration |Δregret| by z* level
        self._regret_tracking = False
        self._regret_by_level = {}  # level → list of |Δregret| at I₀
        self._neighborhood_regret_by_level = {}  # level → list of total |Δregret| across neighborhood
        self._neighborhood_touched_by_level = {}  # level → list of count of infosets with nonzero Δregret
        self._neighborhood_size_by_level = {}  # level → list of materialized neighborhood size
        self._sim_index_by_level = {}  # level → list of sim index within step()
        self._importance_weight_by_level = {}  # level → list of realized importance weight (l)

    # ── pyspiel.Bot interface ────────────────────────────────────────────

    def step(self, state):
        policy, action = self.step_with_policy(state)
        return action

    def step_with_policy(self, state):
        # Determine current active infoset
        info_key = state.information_state_string(self._player_id)
        active_iid = (self._player_id, info_key)

        # Precompute IIG weights and target distribution for this infoset
        iig_weights = self._get_iig_weights(active_iid)
        target_dist = self._get_target_dist(active_iid, iig_weights)

        # Populate debug info
        level_sets = self._iig.level_sets(active_iid)
        max_level = max(level_sets) if level_sets else 0
        self.last_debug = ISGTDebugInfo(
            active_infoset=active_iid,
            num_levels=max_level + 1,
            level_weights={
                lvl: self._level_weight_fn.weight(lvl, max_level)
                for lvl in range(max_level + 1)
            },
            infoset_weights=dict(iig_weights),
        )

        # Run ISGT iterations (alternating update player)
        for sim in range(self._num_simulations):
            update_player = sim % self._num_players
            root = self._game.new_initial_state()

            # Sample target terminal history from IIG-weighted distribution
            history, chance_actions, target_prob = self._sample_target(target_dist)

            # Store z* info (needed for s1 tracking in both scenarios)
            self._z_star_history = history
            self._z_star_chance_actions = chance_actions
            self._z_star_chance_idx = 0

            # Compute anchor depth for full-mode split walk
            if self._bias_mode == "full":
                self._anchor_depth = self._compute_anchor_depth(
                    active_iid, self._last_z_star_tid)

            # Iteration-level targeting decision (OOS-style δ coin flip)
            self._is_targeted_iter = (self._rng.random() < self._delta)

            # Snapshot regret before walk (if tracking enabled + targeted)
            tracking_this = (self._regret_tracking and
                             self._is_targeted_iter and
                             update_player == self._player_id)
            if tracking_this:
                regret_before = self._infostates[info_key][REGRET_INDEX].copy()
                # Snapshot neighborhood: all upstream infosets in _infostates
                neighborhood_snapshot = {}
                for iid in self._iig.levels(active_iid):
                    iid_key = iid[1]  # info_state_string
                    if iid_key in self._infostates:
                        neighborhood_snapshot[iid_key] = (
                            self._infostates[iid_key][REGRET_INDEX].copy())

            # s1=1.0: condition on the chosen z* (like OOS conditions on
            # match history). The z* selection probability is not part of
            # the walk's importance weight — it's external context.
            _x, _l, _u = self._walk(root, update_player,
                                    my_reach=1.0, opp_reach=1.0,
                                    s1=1.0, s2=1.0)

            # Record regret delta by z* level
            if tracking_this:
                regret_after = self._infostates[info_key][REGRET_INDEX]
                active_delta = float(np.abs(regret_after - regret_before).sum())
                # Neighborhood delta: total |Δregret| across all tracked infosets
                nbr_delta = 0.0
                nbr_touched = 0
                for iid_key, before_arr in neighborhood_snapshot.items():
                    if iid_key in self._infostates:
                        after_arr = self._infostates[iid_key][REGRET_INDEX]
                        d = float(np.abs(after_arr - before_arr).sum())
                        nbr_delta += d
                        if d > 0:
                            nbr_touched += 1
                z_level = self._get_z_star_level(active_iid, self._last_z_star_tid)
                if z_level not in self._regret_by_level:
                    self._regret_by_level[z_level] = []
                self._regret_by_level[z_level].append(active_delta)
                if z_level not in self._neighborhood_regret_by_level:
                    self._neighborhood_regret_by_level[z_level] = []
                self._neighborhood_regret_by_level[z_level].append(nbr_delta)
                if z_level not in self._neighborhood_touched_by_level:
                    self._neighborhood_touched_by_level[z_level] = []
                self._neighborhood_touched_by_level[z_level].append(nbr_touched)
                if z_level not in self._neighborhood_size_by_level:
                    self._neighborhood_size_by_level[z_level] = []
                self._neighborhood_size_by_level[z_level].append(
                    len(neighborhood_snapshot))
                if z_level not in self._sim_index_by_level:
                    self._sim_index_by_level[z_level] = []
                self._sim_index_by_level[z_level].append(sim)
                if z_level not in self._importance_weight_by_level:
                    self._importance_weight_by_level[z_level] = []
                self._importance_weight_by_level[z_level].append(float(_l))

        # Extract current policy at this info set
        legal_actions = state.legal_actions()
        policy = self._get_average_policy(info_key, len(legal_actions))

        action_probs = list(zip(legal_actions, policy))
        action = self._rng.choice(legal_actions, p=policy)

        self._match_history.append(action)
        return action_probs, action

    def inform_action(self, state, player_id, action):
        self._match_history.append(action)

    def restart(self):
        self._match_history = []

    def provides_policy(self):
        return True

    # ── IIG weight computation ───────────────────────────────────────────

    def _get_iig_weights(self, active_iid: tuple) -> dict[tuple, float]:
        """Get IIG proximity weights for all infosets relative to active_iid.

        Upstream infosets get weights from the level_weight_fn.
        Infosets not in the upstream neighborhood get floor weight
        to maintain full support.
        """
        if active_iid in self._iig_weight_cache:
            return self._iig_weight_cache[active_iid]

        level_map = self._iig.levels(active_iid)  # iid → level
        level_sets = self._iig.level_sets(active_iid)
        max_level = max(level_sets) if level_sets else 0

        # Assign weights to upstream infosets
        weights = {}
        for iid, level in level_map.items():
            weights[iid] = self._level_weight_fn.weight(level, max_level)

        # Floor weight for non-upstream infosets
        floor_w = self._level_weight_fn.weight(max_level + 1, max_level)
        floor_w = max(floor_w, 1e-6)

        for iid in self._iig.infosets:
            if iid not in weights:
                weights[iid] = floor_w

        self._iig_weight_cache[active_iid] = weights
        return weights

    def _get_target_dist(self, active_iid: tuple,
                         iig_weights: dict[tuple, float]) -> np.ndarray:
        """Two-stage level-first sampling distribution over terminals.

        Stage 1: Pick an IIG level with probability P(level=ℓ) ∝ f(ℓ).
        Stage 2: Pick a terminal uniformly among those assigned to that level.

        Each terminal is assigned to the level of its closest (minimum-
        distance) upstream infoset. Terminals with no upstream infosets
        are excluded — they have zero probability in the targeting
        distribution. (Full support is maintained by the (1-δ) untargeted
        iterations which sample ε-on-policy over the entire tree.)

        The decay function directly controls the level budget allocation:
            P(level=ℓ) ∝ f(ℓ)
            P(z | level=ℓ) = 1 / |T_ℓ|
        """
        if active_iid in self._target_dist_cache:
            return self._target_dist_cache[active_iid]

        # Upstream BFS: iid → level number
        level_map = self._iig.levels(active_iid)
        level_sets = self._iig.level_sets(active_iid)
        max_level = max(level_sets) if level_sets else 0

        # Apply max_iig_depth cap if set
        effective_max = max_level
        if self._max_iig_depth is not None:
            effective_max = min(max_level, self._max_iig_depth)

        num_t = self._iig.num_terminals

        # Assign each terminal to its closest upstream level (-1 = floor)
        terminal_level = np.full(num_t, -1, dtype=int)
        for tid in range(num_t):
            seq = self._iig._terminal_seqs[tid]
            if seq:
                best = None
                for iid in seq:
                    if iid in level_map:
                        lvl = level_map[iid]
                        if best is None or lvl < best:
                            best = lvl
                if best is not None:
                    terminal_level[tid] = best

        # Group terminals by level (exclude floor and beyond depth cap)
        level_groups: dict[int, list[int]] = {}
        for tid in range(num_t):
            lvl = int(terminal_level[tid])
            if lvl == -1 or lvl > effective_max:
                continue
            if lvl not in level_groups:
                level_groups[lvl] = []
            level_groups[lvl].append(tid)

        # Level-budget weights: P(level=ℓ) ∝ f(ℓ)
        # No terminal-count correction — f(ℓ) IS the level budget.
        level_weights: dict[int, float] = {}
        for lvl in level_groups:
            level_weights[lvl] = self._level_weight_fn.weight(lvl, effective_max)

        total_w = sum(level_weights.values())

        # Two-stage probability for each terminal
        weights = np.zeros(num_t, dtype=np.float64)
        for lvl, tids in level_groups.items():
            p_terminal = level_weights[lvl] / total_w / len(tids)
            for tid in tids:
                weights[tid] = p_terminal

        self._target_dist_cache[active_iid] = weights
        return weights

    def _sample_target(self, target_dist: np.ndarray):
        """Sample a target terminal history from the weighted distribution.

        Returns (history, chance_actions, target_prob) where history is
        the full z* action tuple, chance_actions is the extracted chance
        actions, and target_prob is the selection probability.
        """
        tid = self._rng.choice(len(target_dist), p=target_dist)
        self._last_z_star_tid = tid
        history = self._iig._terminal_histories[tid]
        chance_actions = self._extract_chance_actions(history)
        return history, chance_actions, target_dist[tid]

    def _extract_chance_actions(self, history):
        """Replay a terminal history and extract its chance node actions."""
        state = self._game.new_initial_state()
        chance_actions = []
        for action in history:
            if state.is_chance_node():
                chance_actions.append(action)
            state.apply_action(action)
        return chance_actions

    def _get_z_star_level(self, active_iid: tuple, tid: int) -> int:
        """Get the IIG level of terminal tid relative to active_iid.

        Returns the minimum IIG distance from any infoset on the terminal's
        decision sequence to the active infoset.  Returns -1 if no upstream
        infoset is found (floor group).
        """
        level_map = self._iig.levels(active_iid)
        seq = self._iig._terminal_seqs[tid]
        best = -1
        for iid in seq:
            if iid in level_map:
                lvl = level_map[iid]
                if best == -1 or lvl < best:
                    best = lvl
        return best

    def _compute_anchor_depth(self, active_iid: tuple, tid: int) -> int:
        """Compute game-tree depth of the anchor infoset on z*'s path.

        The anchor is the shallowest infoset at the minimum IIG level
        on z*'s decision sequence. Returns len(state.history()) at that
        infoset's decision point — the depth where the walk switches
        from deterministic to ε-on-policy.

        Returns 0 if no upstream infoset is found (should not happen
        since floor terminals are excluded from the targeting distribution,
        but handles the edge case defensively with a fully ε-on-policy walk).
        """
        level_map = self._iig.levels(active_iid)
        seq = self._iig._terminal_seqs[tid]
        history = self._iig._terminal_histories[tid]

        # Find the minimum IIG level on this terminal's path
        best_level = None
        for iid in seq:
            if iid in level_map:
                lvl = level_map[iid]
                if best_level is None or lvl < best_level:
                    best_level = lvl

        if best_level is None:
            return 0  # no anchor — fully ε-on-policy

        # Find the first (shallowest) infoset at best_level
        anchor_iid = None
        for iid in seq:
            if iid in level_map and level_map[iid] == best_level:
                anchor_iid = iid
                break

        # Replay z*'s history to find game-tree depth of anchor infoset
        state = self._game.new_initial_state()
        for action in history:
            if not state.is_chance_node() and not state.is_terminal():
                player = state.current_player()
                info_key = state.information_state_string(player)
                iid = (player, info_key)
                if iid == anchor_iid:
                    return len(state.history())
            state.apply_action(action)

        return len(history)  # fallback

    # ── Core algorithm ───────────────────────────────────────────────────

    def _walk(self, state, update_player,
              my_reach, opp_reach, s1, s2):
        """Recursive ISGT walk from root to terminal.

        MCCFR with trajectory-level targeting and dual sample-reach
        tracking (s1 for targeted scenario, s2 for untargeted). At
        terminals, l = δ·s1 + (1-δ)·s2 gives the combined sampling
        probability for importance weighting.

        Args:
            state: Current OpenSpiel state.
            update_player: Player whose regrets are updated.
            my_reach: Update player's strategy reach.
            opp_reach: Opponent/chance reach (uses natural chance probs).
            s1: Targeted scenario sample reach probability.
            s2: Untargeted scenario sample reach probability.

        Returns:
            (x, l, u): suffix reach (natural probs), sample prob, utility.
        """
        # Terminal
        if state.is_terminal():
            l = self._delta * s1 + (1.0 - self._delta) * s2
            return (1.0, l, state.player_return(update_player))

        # Chance node
        if state.is_chance_node():
            return self._handle_chance(state, update_player,
                                       my_reach, opp_reach, s1, s2)

        # Decision node
        cur_player = state.current_player()
        info_key = state.information_state_string(cur_player)
        legal_actions = state.legal_actions()
        num_actions = len(legal_actions)

        # Look up or create info state
        in_tree = info_key in self._infostates
        if not in_tree:
            self._infostates[info_key] = [
                np.zeros(num_actions, dtype=np.float64),
                np.zeros(num_actions, dtype=np.float64),
            ]
            action_idx = self._rng.randint(num_actions)
            state.apply_action(legal_actions[action_idx])
            playout_s = (self._delta * s1 + (1.0 - self._delta) * s2) / num_actions
            x, l, u = self._playout(state, update_player,
                                     playout_s)
            return (x / num_actions, l, u)

        # Policy via regret matching (always needed for reach tracking)
        policy = self._regret_matching(self._infostates[info_key][REGRET_INDEX])

        # Compute ε-on-policy sampling distribution
        if cur_player == update_player:
            uniform = np.ones(num_actions, dtype=np.float64) / num_actions
            sample_policy = self._epsilon * uniform + (1.0 - self._epsilon) * policy
        else:
            sample_policy = policy.copy()

        # Action selection depends on bias mode, targeting, and anchor depth
        depth = len(state.history())
        in_prefix = (self._bias_mode == "full" and depth < self._anchor_depth)

        if self._is_targeted_iter and in_prefix:
            # Full mode targeted, before anchor: deterministically follow z*
            target_action = self._z_star_history[depth]
            sampled_idx = legal_actions.index(target_action)
            action = target_action
            new_s1 = s1   # deterministic in targeted scenario
            new_s2 = s2 * sample_policy[sampled_idx]
        elif in_prefix:
            # Full mode untargeted, before anchor: ε-on-policy + s1 tracking
            sampled_idx = self._rng.choice(num_actions, p=sample_policy)
            action = legal_actions[sampled_idx]
            new_s2 = s2 * sample_policy[sampled_idx]
            if action == self._z_star_history[depth]:
                new_s1 = s1
            else:
                new_s1 = 0.0
        else:
            # After anchor (full) or chance mode: ε-on-policy, identical s1/s2
            sampled_idx = self._rng.choice(num_actions, p=sample_policy)
            action = legal_actions[sampled_idx]
            new_s1 = s1 * sample_policy[sampled_idx]
            new_s2 = s2 * sample_policy[sampled_idx]

        # Player reach updates (use strategy probs, not sample probs)
        if cur_player == update_player:
            new_my_reach = my_reach * policy[sampled_idx]
            new_opp_reach = opp_reach
        else:
            new_my_reach = my_reach
            new_opp_reach = opp_reach * policy[sampled_idx]

        # Recurse
        state.apply_action(action)
        x, l, u = self._walk(state, update_player,
                             new_my_reach, new_opp_reach,
                             new_s1, new_s2)

        # Regret / average strategy updates
        c = x
        x = x * policy[sampled_idx]

        if cur_player == update_player:
            W = u * opp_reach / max(l, 1e-30)
            for a_idx in range(num_actions):
                if a_idx == sampled_idx:
                    self._infostates[info_key][REGRET_INDEX][a_idx] += (c - x) * W
                else:
                    self._infostates[info_key][REGRET_INDEX][a_idx] += -x * W
        else:
            combined_s = self._delta * s1 + (1.0 - self._delta) * s2
            for a_idx in range(num_actions):
                self._infostates[info_key][AVG_POLICY_INDEX][a_idx] += (
                    opp_reach * policy[a_idx] / max(combined_s, 1e-30)
                )

        return (x, l, u)

    def _handle_chance(self, state, update_player,
                       my_reach, opp_reach, s1, s2):
        """Handle chance nodes with trajectory targeting and s1/s2 tracking.

        In full mode, chance nodes before the anchor depth follow z*
        deterministically (targeted) or track s1 against z* (untargeted).
        After the anchor, both scenarios sample from the natural
        distribution with identical s1/s2 evolution.

        In chance mode, z*'s chance actions are always available for
        deterministic targeting (targeted) and s1 tracking (untargeted).
        """
        outcomes = state.chance_outcomes()
        chance_probs = dict(outcomes)
        depth = len(state.history())
        in_prefix = (self._bias_mode == "full" and depth < self._anchor_depth)

        # Get z*'s chance action at this node (for deterministic follow / s1 tracking)
        z_star_action = None
        if in_prefix:
            # Full mode before anchor: use depth indexing into z*
            if depth < len(self._z_star_history):
                z_star_action = self._z_star_history[depth]
        elif self._bias_mode == "chance":
            # Chance mode: always use chance action iterator
            if self._z_star_chance_idx < len(self._z_star_chance_actions):
                z_star_action = self._z_star_chance_actions[self._z_star_chance_idx]
                self._z_star_chance_idx += 1
        # else: full mode past anchor — no z* tracking, sample naturally

        if self._is_targeted_iter and z_star_action is not None:
            # Targeted + have z* reference: follow deterministically
            action = z_star_action
            rho = chance_probs[action]
            new_s1 = s1        # deterministic in targeted scenario
            new_s2 = s2 * rho  # natural prob under untargeted
        elif z_star_action is not None:
            # Untargeted but have z* reference: sample natural, track s1
            actions, probs = zip(*outcomes)
            idx = self._rng.choice(len(actions), p=probs)
            action = actions[idx]
            rho = probs[idx]
            if action == z_star_action:
                new_s1 = s1    # matches z*'s action
            else:
                new_s1 = 0.0   # diverged from z*
            new_s2 = s2 * rho
        else:
            # No z* reference (full past anchor): both scenarios same
            actions, probs = zip(*outcomes)
            idx = self._rng.choice(len(actions), p=probs)
            action = actions[idx]
            rho = probs[idx]
            new_s1 = s1 * rho
            new_s2 = s2 * rho

        state.apply_action(action)
        x, l, u = self._walk(state, update_player,
                             my_reach, rho * opp_reach,
                             new_s1, new_s2)
        return (rho * x, l, u)

    def _playout(self, state, update_player, sample_reach):
        """Random playout to terminal (outside the tree)."""
        x = 1.0
        while not state.is_terminal():
            if state.is_chance_node():
                actions, probs = zip(*state.chance_outcomes())
                idx = self._rng.choice(len(actions), p=probs)
                x *= probs[idx]
                sample_reach *= probs[idx]
                state.apply_action(actions[idx])
            else:
                legal = state.legal_actions()
                n = len(legal)
                action = legal[self._rng.randint(n)]
                x *= 1.0 / n
                sample_reach *= 1.0 / n
                state.apply_action(action)
        return (x, sample_reach, state.player_return(update_player))

    # ── Strategy computation ─────────────────────────────────────────────

    def _regret_matching(self, regrets):
        positive = np.maximum(regrets, 0)
        total = positive.sum()
        n = len(regrets)
        if total > 0:
            policy = positive / total
        else:
            policy = np.ones(n, dtype=np.float64) / n
        if self._gamma > 0:
            policy = self._gamma / n + (1.0 - self._gamma) * policy
        return policy

    def _get_average_policy(self, info_key, num_actions):
        if info_key not in self._infostates:
            return np.ones(num_actions, dtype=np.float64) / num_actions
        avg = self._infostates[info_key][AVG_POLICY_INDEX]
        total = avg.sum()
        if total > 0:
            return avg / total
        return np.ones(num_actions, dtype=np.float64) / num_actions

    # ── Exploitability adapter ───────────────────────────────────────────

    def average_policy(self):
        return ISGTPolicy(self._game, self._infostates)

    # ── Debugging ────────────────────────────────────────────────────────

    def print_debug(self):
        """Print debug info from the last step() call."""
        d = self.last_debug
        print(f"ISGT Debug — {self._level_weight_fn.name()} [{self._bias_mode}]")
        print(f"  Active infoset: {self._iig._fmt_id(d.active_infoset)}")
        print(f"  IIG levels: {d.num_levels}")
        print(f"  Level weights:")
        for lvl, w in sorted(d.level_weights.items()):
            print(f"    Level {lvl}: {w:.4f}")
        print(f"  Top infoset weights:")
        top = sorted(d.infoset_weights.items(), key=lambda x: -x[1])[:10]
        for iid, w in top:
            print(f"    {self._iig._fmt_id(iid)}: {w:.4f}")
        print()


class ISGTPolicy:
    """Adapter to make ISGT info states look like an OpenSpiel Policy."""

    def __init__(self, game, infostates):
        self._game = game
        self._infostates = infostates

    def action_probabilities(self, state, player_id=None):
        if player_id is None:
            player_id = state.current_player()
        info_key = state.information_state_string(player_id)
        legal_actions = state.legal_actions()
        n = len(legal_actions)
        if info_key in self._infostates:
            avg = self._infostates[info_key][AVG_POLICY_INDEX]
            total = avg.sum()
            if total > 0:
                probs = avg / total
            else:
                probs = np.ones(n, dtype=np.float64) / n
        else:
            probs = np.ones(n, dtype=np.float64) / n
        return dict(zip(legal_actions, probs))
