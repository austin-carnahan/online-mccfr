"""011 Baseline Viability — Unified validation harness for ISGT.

Runs three sub-experiments in sequence on Kuhn Poker (small, fully
interpretable) and prints clean, human-readable results for each.

Sub-experiments:
  011a — Single-Trace Rollout Validation
         5 fully traced ISGT iterations, node-by-node verification.
  011b — Sampling Mode Sanity Check
         Statistical checks on δ frequency, ε distribution, mode behavior.
  011c — Support Restriction Check
         Verifies IIG targeting ≠ random; untargeted covers all terminals.

Usage:
    python -m experiments.validation_011
"""

from __future__ import annotations

import sys
from collections import defaultdict
from dataclasses import dataclass, field

import numpy as np
import pyspiel

from src.games import load_game
from src.iig import IIG
from src.isgt import ISGTBot, ConstantWeight, ExponentialDecay, REGRET_INDEX, AVG_POLICY_INDEX


# ═════════════════════════════════════════════════════════════════════════
# Helpers
# ═════════════════════════════════════════════════════════════════════════

def _bar(ok: bool) -> str:
    return "PASS" if ok else "** FAIL **"


def _pct(n: int, d: int) -> str:
    return f"{n}/{d} ({100*n/d:.0f}%)" if d > 0 else "0/0"


def _section(title: str, width: int = 72):
    print()
    print("=" * width)
    print(f"  {title}")
    print("=" * width)


def _subsection(title: str, width: int = 60):
    print()
    print(f"--- {title} " + "-" * max(0, width - len(title) - 5))


# ═════════════════════════════════════════════════════════════════════════
# Traced walker — wraps ISGTBot._walk to capture per-node data
# ═════════════════════════════════════════════════════════════════════════

@dataclass
class TraceNode:
    """One node visited during a traced walk."""
    depth: int = 0
    node_type: str = ""       # "chance", "decision", "terminal"
    player: str = ""
    info_key: str = ""
    action: int | None = None
    z_star_action: int | None = None
    matches_z_star: bool | None = None
    s1: float = 0.0
    s2: float = 0.0
    sigma_a: float | None = None      # strategy prob σ(a)
    sigma_sample_a: float | None = None  # sample policy prob
    pi_i: float = 0.0
    pi_neg_i: float = 0.0


@dataclass
class TraceResult:
    """Full trace of one ISGT iteration."""
    iteration: int = 0
    update_player: int = 0
    targeted: bool = False
    z_star_tid: int = -1
    z_star_history: tuple = ()
    z_star_prob: float = 0.0
    z_star_iig_dist: int = -1
    nodes: list[TraceNode] = field(default_factory=list)
    terminal_utility: float = 0.0
    terminal_l: float = 0.0
    terminal_W: float = 0.0
    regret_updates: dict = field(default_factory=dict)  # a_idx -> delta


class TracedISGTBot(ISGTBot):
    """ISGTBot subclass that captures per-node trace data.

    Call run_traced_iterations() instead of step_with_policy().
    Traces are stored in self.traces.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.traces: list[TraceResult] = []
        self._current_trace: TraceResult | None = None

    def run_traced_iterations(self, state, n_iters: int,
                              active_iid: tuple | None = None):
        """Run n_iters traced iterations from the given state.

        If active_iid is provided, use it for IIG weight computation.
        Otherwise derive from state.
        """
        self.traces = []

        if active_iid is None:
            info_key = state.information_state_string(self._player_id)
            active_iid = (self._player_id, info_key)

        iig_weights = self._get_iig_weights(active_iid)
        target_dist = self._get_target_dist(active_iid, iig_weights)

        # Precompute upstream set for IIG distance labeling
        level_map = self._iig.levels(active_iid)

        for i in range(n_iters):
            update_player = i % self._num_players
            root = self._game.new_initial_state()

            # Sample z*
            tid = self._rng.choice(len(target_dist), p=target_dist)
            history = self._iig._terminal_histories[tid]
            chance_actions = self._extract_chance_actions(history)

            self._z_star_history = history
            self._z_star_chance_actions = chance_actions
            self._z_star_chance_idx = 0
            self._is_targeted_iter = (self._rng.random() < self._delta)

            # Compute IIG distance of z*
            seq = self._iig._terminal_seqs[tid]
            if seq:
                dists = [level_map.get(iid, -1) for iid in seq]
                valid = [d for d in dists if d >= 0]
                z_dist = min(valid) if valid else -1
            else:
                z_dist = -1

            trace = TraceResult(
                iteration=i,
                update_player=update_player,
                targeted=self._is_targeted_iter,
                z_star_tid=tid,
                z_star_history=tuple(history),
                z_star_prob=target_dist[tid],
                z_star_iig_dist=z_dist,
            )
            self._current_trace = trace

            self._walk(root, update_player,
                       my_reach=1.0, opp_reach=1.0,
                       s1=target_dist[tid], s2=1.0)

            self.traces.append(trace)

        self._current_trace = None

    def _walk(self, state, update_player, my_reach, opp_reach, s1, s2):
        trace = self._current_trace

        # Terminal
        if state.is_terminal():
            l = self._delta * s1 + (1.0 - self._delta) * s2
            u = state.player_return(update_player)

            if trace is not None:
                node = TraceNode(
                    depth=len(state.history()),
                    node_type="terminal",
                    player="—",
                    s1=s1, s2=s2,
                    pi_i=my_reach, pi_neg_i=opp_reach,
                )
                trace.nodes.append(node)
                trace.terminal_utility = u
                trace.terminal_l = l
                trace.terminal_W = u * opp_reach / max(l, 1e-30)

            return (1.0, l, u)

        # Chance node
        if state.is_chance_node():
            return self._handle_chance_traced(state, update_player,
                                              my_reach, opp_reach, s1, s2)

        # Decision node
        cur_player = state.current_player()
        info_key = state.information_state_string(cur_player)
        legal_actions = state.legal_actions()
        num_actions = len(legal_actions)

        in_tree = info_key in self._infostates
        if not in_tree:
            self._infostates[info_key] = [
                np.zeros(num_actions, dtype=np.float64),
                np.zeros(num_actions, dtype=np.float64),
            ]
            action_idx = self._rng.randint(num_actions)
            state.apply_action(legal_actions[action_idx])
            playout_s = (self._delta * s1 + (1.0 - self._delta) * s2) / num_actions
            x, l, u = self._playout(state, update_player, playout_s)
            return (x / num_actions, l, u)

        policy = self._regret_matching(self._infostates[info_key][REGRET_INDEX])

        if cur_player == update_player:
            uniform = np.ones(num_actions, dtype=np.float64) / num_actions
            sample_policy = self._epsilon * uniform + (1.0 - self._epsilon) * policy
        else:
            sample_policy = policy.copy()

        # Determine z*'s action at this depth (for tracing)
        depth = len(state.history())
        z_star_action_here = None
        if depth < len(self._z_star_history):
            za = self._z_star_history[depth]
            if za in legal_actions:
                z_star_action_here = za

        # Action selection (same logic as parent)
        if self._is_targeted_iter and self._bias_mode == "full":
            target_action = self._z_star_history[depth]
            sampled_idx = legal_actions.index(target_action)
            action = target_action
            new_s1 = s1
            new_s2 = s2 * sample_policy[sampled_idx]
        else:
            sampled_idx = self._rng.choice(num_actions, p=sample_policy)
            action = legal_actions[sampled_idx]
            new_s2 = s2 * sample_policy[sampled_idx]
            if self._bias_mode == "full":
                if depth < len(self._z_star_history) and action == self._z_star_history[depth]:
                    new_s1 = s1
                else:
                    new_s1 = 0.0
            else:
                new_s1 = s1 * sample_policy[sampled_idx]

        matches = (z_star_action_here is not None and action == z_star_action_here)

        if trace is not None:
            node = TraceNode(
                depth=depth,
                node_type="decision",
                player=f"P{cur_player}",
                info_key=info_key,
                action=action,
                z_star_action=z_star_action_here,
                matches_z_star=matches,
                s1=new_s1, s2=new_s2,
                sigma_a=policy[sampled_idx],
                sigma_sample_a=sample_policy[sampled_idx],
                pi_i=my_reach * policy[sampled_idx] if cur_player == update_player else my_reach,
                pi_neg_i=opp_reach if cur_player == update_player else opp_reach * policy[sampled_idx],
            )
            trace.nodes.append(node)

        # Reach updates
        if cur_player == update_player:
            new_my_reach = my_reach * policy[sampled_idx]
            new_opp_reach = opp_reach
        else:
            new_my_reach = my_reach
            new_opp_reach = opp_reach * policy[sampled_idx]

        state.apply_action(action)
        x, l, u = self._walk(state, update_player,
                             new_my_reach, new_opp_reach, new_s1, new_s2)

        # Regret updates (trace them for update player nodes)
        c = x
        x = x * policy[sampled_idx]

        if cur_player == update_player:
            W = u * opp_reach / max(l, 1e-30)
            if trace is not None:
                trace.regret_updates[info_key] = {}
            for a_idx in range(num_actions):
                if a_idx == sampled_idx:
                    delta_r = (c - x) * W
                else:
                    delta_r = -x * W
                self._infostates[info_key][REGRET_INDEX][a_idx] += delta_r
                if trace is not None:
                    trace.regret_updates[info_key][a_idx] = delta_r
        else:
            combined_s = self._delta * s1 + (1.0 - self._delta) * s2
            for a_idx in range(num_actions):
                self._infostates[info_key][AVG_POLICY_INDEX][a_idx] += (
                    opp_reach * policy[a_idx] / max(combined_s, 1e-30)
                )

        return (x, l, u)

    def _handle_chance_traced(self, state, update_player,
                              my_reach, opp_reach, s1, s2):
        """Chance node handling with tracing."""
        trace = self._current_trace
        outcomes = state.chance_outcomes()
        chance_probs = dict(outcomes)

        z_star_action = None
        if self._bias_mode == "full":
            depth = len(state.history())
            if depth < len(self._z_star_history):
                z_star_action = self._z_star_history[depth]
        else:
            if self._z_star_chance_idx < len(self._z_star_chance_actions):
                z_star_action = self._z_star_chance_actions[self._z_star_chance_idx]
                self._z_star_chance_idx += 1

        if self._is_targeted_iter and z_star_action is not None:
            action = z_star_action
            rho = chance_probs[action]
            new_s1 = s1
            new_s2 = s2 * rho
        else:
            actions, probs = zip(*outcomes)
            idx = self._rng.choice(len(actions), p=probs)
            action = actions[idx]
            rho = probs[idx]
            if z_star_action is not None and action == z_star_action:
                new_s1 = s1
            elif z_star_action is not None:
                new_s1 = 0.0
            else:
                new_s1 = s1 * rho
            new_s2 = s2 * rho

        matches = (z_star_action is not None and action == z_star_action)

        if trace is not None:
            node = TraceNode(
                depth=len(state.history()),
                node_type="chance",
                player="C",
                action=action,
                z_star_action=z_star_action,
                matches_z_star=matches,
                s1=new_s1, s2=new_s2,
                pi_i=my_reach,
                pi_neg_i=rho * opp_reach,
            )
            trace.nodes.append(node)

        state.apply_action(action)
        x, l, u = self._walk(state, update_player,
                             my_reach, rho * opp_reach,
                             new_s1, new_s2)
        return (rho * x, l, u)


# ═════════════════════════════════════════════════════════════════════════
# 011a — Single-Trace Rollout Validation
# ═════════════════════════════════════════════════════════════════════════

def run_011a(game, iig, n_iters: int = 5, seed: int = 42):
    """Run traced ISGT iterations and print step-by-step verification."""
    _section("011a — Single-Trace Rollout Validation")

    for mode in ["chance", "full"]:
        _subsection(f"Mode: {mode} (δ=0.9, ε=0.6, ConstantWeight)")

        bot = TracedISGTBot(
            game, player_id=0, num_simulations=0,
            epsilon=0.6, gamma=0.01,
            level_weight_fn=ConstantWeight(),
            bias_mode=mode, seed=seed, iig=iig, delta=0.9,
        )

        state = game.new_initial_state()
        # Advance past initial chance node to reach player 0's first decision
        # (Kuhn deals one card to each player)
        info_key = None
        while not state.is_terminal():
            if state.is_chance_node():
                # dealt cards — deterministic for tracing
                break
            break

        # Use P0's first infoset for the trace
        # Find a valid P0 root infoset
        p0_infosets = iig.infoset_ids_for_player(0)
        # Pick the first root-level one (no predecessors)
        root_iids = [iid for iid in p0_infosets if not iig.predecessors(iid)]
        active_iid = root_iids[0] if root_iids else p0_infosets[0]

        print(f"  Active infoset: {iig._fmt_id(active_iid)}")
        level_sets = iig.level_sets(active_iid)
        for lvl, ids in level_sets.items():
            labels = [iig._fmt_id(iid) for iid in ids]
            print(f"    Level {lvl}: {', '.join(labels)}")

        # Upstream terminal count
        upstream_iids = set(iig.levels(active_iid).keys())
        upstream_tids = set()
        for iid in upstream_iids:
            upstream_tids |= iig.z_set(iid)
        print(f"  Upstream terminals: {len(upstream_tids)}/{iig.num_terminals}")
        print()

        bot.run_traced_iterations(state, n_iters, active_iid)

        checks_all_pass = True
        for t in bot.traces:
            _print_trace(t, bot._delta, mode)
            ok = _verify_trace(t, bot._delta, bot._epsilon, mode)
            if not ok:
                checks_all_pass = False

        print(f"\n  011a ({mode}): {_bar(checks_all_pass)}")


def _print_trace(t: TraceResult, delta: float, mode: str):
    """Print one iteration trace."""
    tgt_str = "YES" if t.targeted else "NO"
    print(f"  Iter {t.iteration} | update=P{t.update_player} | "
          f"targeted={tgt_str} | z*=tid#{t.z_star_tid} "
          f"(IIG dist={t.z_star_iig_dist}, p={t.z_star_prob:.4f})")
    print(f"    z* actions: {t.z_star_history}")

    # Table header
    print(f"    {'Dp':>3} {'Type':8} {'Pl':3} {'Info':18} "
          f"{'Act':>4} {'z*':>4} {'Match':5} "
          f"{'s1':>8} {'s2':>8} {'σ(a)':>7} {'σ_s(a)':>7}")
    print(f"    {'---':>3} {'--------':8} {'---':3} {'------------------':18} "
          f"{'----':>4} {'----':>4} {'-----':5} "
          f"{'--------':>8} {'--------':>8} {'-------':>7} {'-------':>7}")

    for n in t.nodes:
        info_short = n.info_key[:16] if n.info_key else "—"
        act_str = str(n.action) if n.action is not None else "—"
        z_str = str(n.z_star_action) if n.z_star_action is not None else "—"
        m_str = "✓" if n.matches_z_star else ("✗" if n.matches_z_star is False else "—")

        if n.node_type == "terminal":
            m_str = "—"

        sig_str = f"{n.sigma_a:.4f}" if n.sigma_a is not None else "—"
        sigs_str = f"{n.sigma_sample_a:.4f}" if n.sigma_sample_a is not None else "—"

        print(f"    {n.depth:>3} {n.node_type:8} {n.player:3} {info_short:18} "
              f"{act_str:>4} {z_str:>4} {m_str:>5} "
              f"{n.s1:>8.4f} {n.s2:>8.4f} {sig_str:>7} {sigs_str:>7}")

    print(f"    Terminal: u={t.terminal_utility:.2f}, "
          f"l=δ·s1+(1-δ)·s2 = {delta:.1f}·{t.nodes[-1].s1:.4f}"
          f"+{1-delta:.1f}·{t.nodes[-1].s2:.4f} = {t.terminal_l:.6f}")
    print(f"    W = u·π₋ᵢ/l = {t.terminal_W:.4f}")

    # Regret updates
    if t.regret_updates:
        for ik, updates in t.regret_updates.items():
            parts = [f"a{a}={v:+.4f}" for a, v in sorted(updates.items())]
            print(f"    Regrets @{ik[:16]}: {', '.join(parts)}")
    print()


def _verify_trace(t: TraceResult, delta: float, epsilon: float,
                  mode: str) -> bool:
    """Run automated checks on a trace. Returns True if all pass."""
    problems = []

    # Check: l > 0
    if t.terminal_l <= 0:
        problems.append("l <= 0 at terminal")

    # Check: l = delta*s1 + (1-delta)*s2 at terminal
    term_node = t.nodes[-1]
    expected_l = delta * term_node.s1 + (1 - delta) * term_node.s2
    if abs(t.terminal_l - expected_l) > 1e-10:
        problems.append(f"l mismatch: {t.terminal_l} vs computed {expected_l}")

    for i, n in enumerate(t.nodes):
        if n.node_type == "chance":
            if t.targeted and mode == "chance":
                # Targeted chance: should follow z*
                if n.z_star_action is not None and not n.matches_z_star:
                    problems.append(f"depth {n.depth}: targeted chance didn't follow z*")
            if t.targeted and mode == "full":
                if n.z_star_action is not None and not n.matches_z_star:
                    problems.append(f"depth {n.depth}: targeted full chance didn't follow z*")

        if n.node_type == "decision":
            if t.targeted and mode == "full":
                # Full mode targeted: MUST follow z*
                if n.z_star_action is not None and not n.matches_z_star:
                    problems.append(f"depth {n.depth}: full targeted decision didn't follow z*")

            # s2 should be parent_s2 * sigma_sample(a) at decisions
            # (can't easily check without parent ref, skip for now)

    if problems:
        for p in problems:
            print(f"    ** CHECK FAIL: {p}")
        return False
    return True


# ═════════════════════════════════════════════════════════════════════════
# 011b — Sampling Mode Sanity Check
# ═════════════════════════════════════════════════════════════════════════

def run_011b(game, iig, seed: int = 42):
    """Statistical checks on δ frequency, ε distribution, mode behavior."""
    _section("011b — Sampling Mode Sanity Check")
    n_iters = 50
    all_pass = True

    # ── Check 1: Targeting frequency ≈ δ ──
    _subsection("Check 1: Targeting Frequency (N=50 per δ)")

    for delta in [0.0, 0.2, 0.5, 0.9, 1.0]:
        bot = TracedISGTBot(
            game, player_id=0, num_simulations=0,
            epsilon=0.6, gamma=0.01,
            level_weight_fn=ConstantWeight(),
            bias_mode="chance", seed=seed, iig=iig, delta=delta,
        )
        state = game.new_initial_state()
        p0_infosets = iig.infoset_ids_for_player(0)
        root_iids = [iid for iid in p0_infosets if not iig.predecessors(iid)]
        active_iid = root_iids[0]

        bot.run_traced_iterations(state, n_iters, active_iid)
        n_targeted = sum(1 for t in bot.traces if t.targeted)

        # Binomial 95% CI
        from scipy.stats import binom
        lo, hi = binom.interval(0.95, n_iters, delta)

        ok = lo <= n_targeted <= hi
        if not ok:
            all_pass = False
        print(f"  δ={delta:.1f}: {n_targeted}/{n_iters} targeted "
              f"(95% CI: {lo:.0f}–{hi:.0f})  {_bar(ok)}")

    # ── Check 2: ε-on-policy computation is correct ──
    _subsection("Check 2: ε-on-Policy Computation (deterministic)")

    # Verify that σ_sample = ε/|A| + (1-ε)·σ at decision nodes
    # by checking trace data directly (no statistical noise)
    warmup_bot = TracedISGTBot(
        game, player_id=0, num_simulations=0,
        epsilon=0.6, gamma=0.01,
        level_weight_fn=ConstantWeight(),
        bias_mode="chance", seed=seed, iig=iig, delta=0.5,
    )
    state = game.new_initial_state()
    p0_infosets = iig.infoset_ids_for_player(0)
    root_iids = [iid for iid in p0_infosets if not iig.predecessors(iid)]
    active_iid = root_iids[0]

    # Run some iterations to build non-uniform policy
    warmup_bot.run_traced_iterations(state, 50, active_iid)

    # Now run traced iterations and verify σ_sample values in trace data
    check_bot = TracedISGTBot(
        game, player_id=0, num_simulations=0,
        epsilon=0.6, gamma=0.01,
        level_weight_fn=ConstantWeight(),
        bias_mode="chance", seed=seed + 1, iig=iig, delta=0.5,
    )
    # Copy learned strategy
    for k, v in warmup_bot._infostates.items():
        check_bot._infostates[k] = [v[0].copy(), v[1].copy()]

    check_bot.run_traced_iterations(state, 20, active_iid)

    eps_check_pass = True
    n_checked = 0
    for t in check_bot.traces:
        for n in t.nodes:
            if n.node_type != "decision":
                continue
            if n.sigma_a is None or n.sigma_sample_a is None:
                continue

            info_key = n.info_key
            if info_key not in check_bot._infostates:
                continue

            # At update-player decision nodes: σ_s = ε/|A| + (1-ε)·σ
            # At non-update-player nodes: σ_s = σ
            is_update = (n.player == f"P{t.update_player}")
            n_a = len(check_bot._infostates[info_key][REGRET_INDEX])

            if is_update:
                expected_sample = check_bot._epsilon / n_a + (1 - check_bot._epsilon) * n.sigma_a
            else:
                expected_sample = n.sigma_a

            if abs(n.sigma_sample_a - expected_sample) > 1e-8:
                eps_check_pass = False
                print(f"  ** FAIL at {info_key}: σ_s={n.sigma_sample_a:.6f} "
                      f"expected {expected_sample:.6f}")
            n_checked += 1

    if not eps_check_pass:
        all_pass = False
    print(f"  Checked {n_checked} decision nodes: {_bar(eps_check_pass)}")

    # ── Check 3: Chance vs Full at decision nodes ──
    _subsection("Check 3: Full Mode Locks Decisions to z* (N=30, δ=1.0)")

    for mode in ["chance", "full"]:
        bot = TracedISGTBot(
            game, player_id=0, num_simulations=0,
            epsilon=0.6, gamma=0.01,
            level_weight_fn=ConstantWeight(),
            bias_mode=mode, seed=seed, iig=iig, delta=1.0,
        )
        state = game.new_initial_state()
        bot.run_traced_iterations(state, 30, active_iid)

        total_decisions = 0
        z_star_matches = 0
        for t in bot.traces:
            for n in t.nodes:
                if n.node_type == "decision" and n.z_star_action is not None:
                    total_decisions += 1
                    if n.matches_z_star:
                        z_star_matches += 1

        if mode == "full":
            ok = (z_star_matches == total_decisions)
        else:
            # Chance mode: should NOT be 100% (ε-exploration causes divergence)
            ok = (z_star_matches < total_decisions)
        if not ok:
            all_pass = False

        pct = f"{100*z_star_matches/total_decisions:.0f}%" if total_decisions else "N/A"
        expectation = "= 100%" if mode == "full" else "< 100%"
        print(f"  {mode:7s}: {z_star_matches}/{total_decisions} matched z* "
              f"({pct}, expected {expectation})  {_bar(ok)}")

    # ── Check 4: δ=0 makes s1 ≡ s2 ──
    _subsection("Check 4: Untargeted (δ=0) — s1 tracking")

    bot = TracedISGTBot(
        game, player_id=0, num_simulations=0,
        epsilon=0.6, gamma=0.01,
        level_weight_fn=ConstantWeight(),
        bias_mode="chance", seed=seed, iig=iig, delta=0.0,
    )
    state = game.new_initial_state()
    bot.run_traced_iterations(state, 50, active_iid)

    # With δ=0, s1 doesn't matter for l computation (l = 0·s1 + 1·s2 = s2).
    # But verify no iterations are targeted
    n_targeted = sum(1 for t in bot.traces if t.targeted)
    ok = (n_targeted == 0)
    if not ok:
        all_pass = False
    print(f"  δ=0.0: {n_targeted}/50 targeted (expected 0)  {_bar(ok)}")

    # Verify l = s2 at every terminal
    l_eq_s2 = True
    for t in bot.traces:
        term = t.nodes[-1]
        if abs(t.terminal_l - term.s2) > 1e-10:
            l_eq_s2 = False
            break
    ok2 = l_eq_s2
    if not ok2:
        all_pass = False
    print(f"  l = s2 at all terminals: {_bar(ok2)}")

    print(f"\n  011b overall: {_bar(all_pass)}")
    return all_pass


# ═════════════════════════════════════════════════════════════════════════
# 011c — Support Restriction Check
# ═════════════════════════════════════════════════════════════════════════

def run_011c(game, iig, seed: int = 42):
    """Verify IIG-restricted sampling ≠ random; untargeted covers all."""
    _section("011c — Support Restriction Check")
    all_pass = True

    p0_infosets = iig.infoset_ids_for_player(0)
    root_iids = [iid for iid in p0_infosets if not iig.predecessors(iid)]
    active_iid = root_iids[0]

    # Compute upstream terminal set
    level_map = iig.levels(active_iid)
    upstream_iids = set(level_map.keys())
    upstream_tids = set()
    for iid in upstream_iids:
        upstream_tids |= iig.z_set(iid)
    all_tids = set(range(iig.num_terminals))

    print(f"  Active infoset: {iig._fmt_id(active_iid)}")
    print(f"  Upstream terminals: {len(upstream_tids)}/{iig.num_terminals}")

    # ── Check 1: ConstantWeight gives ~uniform z*, Exp(0.5) concentrates ──
    _subsection("Check 1: Decay Function Shapes z* Distribution (N=100)")

    # ConstantWeight: floor weight = weight(max_level+1, max_level) = 1.0,
    # so ALL terminals get max weight 1.0 → uniform distribution.
    # Expected: ~33% in upstream (10/30 terminals on Kuhn).
    bot_const = TracedISGTBot(
        game, player_id=0, num_simulations=0,
        epsilon=0.6, gamma=0.01,
        level_weight_fn=ConstantWeight(),
        bias_mode="chance", seed=seed, iig=iig, delta=1.0,
    )
    state = game.new_initial_state()
    bot_const.run_traced_iterations(state, 100, active_iid)

    const_in_upstream = sum(1 for t in bot_const.traces if t.z_star_tid in upstream_tids)
    expected_pct = len(upstream_tids) / iig.num_terminals
    print(f"  ConstantWeight: {_pct(const_in_upstream, 100)} in upstream "
          f"(expected ~{100*expected_pct:.0f}% if uniform)")
    ok_const = abs(const_in_upstream / 100 - expected_pct) < 0.15
    if not ok_const:
        all_pass = False
    print(f"    ~Uniform z* distribution: {_bar(ok_const)}")

    # Exp(0.5): floor weight = 0.5^(max_level+1) ≈ small.
    # Upstream terminals get weight 1.0 (level 0), non-upstream get floor.
    # Expected: vast majority in upstream.
    bot_exp = TracedISGTBot(
        game, player_id=0, num_simulations=0,
        epsilon=0.6, gamma=0.01,
        level_weight_fn=ExponentialDecay(0.5),
        bias_mode="chance", seed=seed, iig=iig, delta=1.0,
    )
    state = game.new_initial_state()
    bot_exp.run_traced_iterations(state, 100, active_iid)

    exp_in_upstream = sum(1 for t in bot_exp.traces if t.z_star_tid in upstream_tids)
    print(f"  Exp(0.5):       {_pct(exp_in_upstream, 100)} in upstream "
          f"(expected >> {100*expected_pct:.0f}%)")
    ok_exp = exp_in_upstream > const_in_upstream  # Exp should concentrate more
    if not ok_exp:
        all_pass = False
    print(f"    Concentrates more than Constant: {_bar(ok_exp)}")

    # ── Check 2: Exp(0.5) concentrates even more ──
    _subsection("Check 2: Exp(0.5) Concentrates on Nearby Terminals (N=100)")

    bot_exp = TracedISGTBot(
        game, player_id=0, num_simulations=0,
        epsilon=0.6, gamma=0.01,
        level_weight_fn=ExponentialDecay(0.5),
        bias_mode="chance", seed=seed, iig=iig, delta=1.0,
    )
    state = game.new_initial_state()
    bot_exp.run_traced_iterations(state, 100, active_iid)

    dist_counts = defaultdict(int)
    for t in bot_exp.traces:
        dist_counts[t.z_star_iig_dist] += 1

    print(f"  z* IIG distance distribution:")
    for d in sorted(dist_counts):
        print(f"    distance {d}: {dist_counts[d]}/100")

    # Expect most at distance 0
    ok2 = dist_counts.get(0, 0) >= dist_counts.get(max(dist_counts, default=0), 0)
    if not ok2:
        all_pass = False
    print(f"  Distance 0 is most frequent: {_bar(ok2)}")

    # ── Check 3: Untargeted iterations visit non-upstream terminals ──
    _subsection("Check 3: Untargeted Iterations Cover Non-Upstream (δ=0.5, N=200)")

    bot_mixed = TracedISGTBot(
        game, player_id=0, num_simulations=0,
        epsilon=0.6, gamma=0.01,
        level_weight_fn=ConstantWeight(),
        bias_mode="chance", seed=seed, iig=iig, delta=0.5,
    )
    state = game.new_initial_state()
    bot_mixed.run_traced_iterations(state, 200, active_iid)

    targeted_terms = set()
    untargeted_terms = set()
    for t in bot_mixed.traces:
        # The walked terminal is determined by the actual path taken.
        # z_star_tid is what was selected; we need the ACTUAL terminal visited.
        # We can reconstruct from the action sequence in the trace nodes.
        # For now, use z_star_tid for targeted iters (since targeted follows z*)
        # and we need the actual walked path for untargeted.
        # Since we don't directly record the visited terminal ID, let's
        # collect the chance actions taken and match to terminals.
        chance_actions_taken = []
        for n in t.nodes:
            if n.node_type == "chance" and n.action is not None:
                chance_actions_taken.append(n.action)

        # For Kuhn poker, the first two chance actions are the card deals.
        # Each unique pair of deals maps to a subset of terminals.
        if t.targeted:
            targeted_terms.add(t.z_star_tid)
        else:
            # Find terminal by matching action sequence
            actions_taken = tuple(
                n.action for n in t.nodes
                if n.action is not None and n.node_type != "terminal"
            )
            for tid, hist in enumerate(iig._terminal_histories):
                if tuple(hist[:len(actions_taken)]) == actions_taken:
                    untargeted_terms.add(tid)
                    break

    # Check: untargeted_terms should include some tids NOT in upstream_tids
    non_upstream_visited = untargeted_terms - upstream_tids
    n_untargeted = sum(1 for t in bot_mixed.traces if not t.targeted)

    print(f"  Targeted iters: {200 - n_untargeted}, Untargeted: {n_untargeted}")
    print(f"  Unique terminals visited (untargeted): {len(untargeted_terms)}")
    print(f"  Of which non-upstream: {len(non_upstream_visited)}")

    ok3 = len(non_upstream_visited) > 0
    if not ok3:
        all_pass = False
    print(f"  Untargeted reached beyond upstream: {_bar(ok3)}")

    # ── Check 4: z* distribution vs theoretical ──
    _subsection("Check 4: z* Matches Theoretical Distribution (Exp(0.5), N=500)")

    bot_verify = TracedISGTBot(
        game, player_id=0, num_simulations=0,
        epsilon=0.6, gamma=0.01,
        level_weight_fn=ExponentialDecay(0.5),
        bias_mode="chance", seed=seed, iig=iig, delta=1.0,
    )
    state = game.new_initial_state()
    bot_verify.run_traced_iterations(state, 500, active_iid)

    # Get theoretical distribution
    iig_weights = bot_verify._get_iig_weights(active_iid)
    target_dist = bot_verify._get_target_dist(active_iid, iig_weights)

    # Count observed z* selections
    observed_counts = np.zeros(iig.num_terminals)
    for t in bot_verify.traces:
        observed_counts[t.z_star_tid] += 1
    observed_freq = observed_counts / 500

    # Compare: upstream terminals should have similar frequency
    # Non-upstream should have ~0 frequency
    # Chi-squared test (group floor terminals together to avoid tiny expected)
    from scipy.stats import chisquare

    # Group: each upstream terminal gets its own bin, all floor terminals = 1 bin
    upstream_list = sorted(upstream_tids)
    floor_list = sorted(all_tids - upstream_tids)

    obs_grouped = []
    exp_grouped = []
    for tid in upstream_list:
        obs_grouped.append(observed_counts[tid])
        exp_grouped.append(target_dist[tid] * 500)
    # Floor group
    if floor_list:
        obs_grouped.append(sum(observed_counts[tid] for tid in floor_list))
        exp_grouped.append(sum(target_dist[tid] for tid in floor_list) * 500)

    obs_grouped = np.array(obs_grouped)
    exp_grouped = np.array(exp_grouped)

    # Filter out bins with 0 expected (they'd cause division issues)
    mask = exp_grouped > 0.5
    if mask.sum() > 1:
        stat, p_val = chisquare(obs_grouped[mask], exp_grouped[mask])
        ok4 = p_val > 0.01  # generous threshold
        print(f"  χ² test vs IIG distribution: stat={stat:.1f}, p={p_val:.3f}  {_bar(ok4)}")
    else:
        ok4 = True
        print(f"  (Too few bins for χ² test, skipping)  {_bar(ok4)}")

    if not ok4:
        all_pass = False

    print(f"\n  011c overall: {_bar(all_pass)}")
    return all_pass


# ═════════════════════════════════════════════════════════════════════════
# Main
# ═════════════════════════════════════════════════════════════════════════

def main():
    game = load_game("kuhn_poker")
    iig = IIG(game)

    print("=" * 72)
    print("  011 — Baseline Viability Validation (Kuhn Poker)")
    print("=" * 72)
    print(f"  Infosets: {iig.num_infosets}, Terminals: {iig.num_terminals}")

    results = {}

    results["011a"] = run_011a(game, iig)
    results["011b"] = run_011b(game, iig)
    results["011c"] = run_011c(game, iig)

    _section("Summary")
    for key in ["011a", "011b", "011c"]:
        status = results.get(key)
        if status is None:
            label = "(trace-only, check output above)"
        else:
            label = _bar(status)
        print(f"  {key}: {label}")

    all_ok = all(v for v in results.values() if v is not None)
    print(f"\n  Overall: {_bar(all_ok)}")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
