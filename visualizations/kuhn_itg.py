"""Kuhn Poker — Infoset Transition Graph (ITG), P1 holds Queen.

Simplified slice: Player 1's private card is fixed to Q.
Player 2 can hold J or K.

Nodes = infosets  (player | private card | public action history).
Directed edges = player actions connecting one infoset to the next.
Terminal sinks show game-ending outcomes.

Usage:
    python kuhn_itg.py
"""

import os
import graphviz

# ── Palette ──────────────────────────────────────────────────────────────────

P1_FILL    = '#dbeafe'
P1_BORDER  = '#2563eb'
P2_FILL    = '#fee2e2'
P2_BORDER  = '#dc2626'
TERM_FILL  = '#f3f4f6'
TERM_BORDER = '#9ca3af'

CHECK_CLR = '#16a34a'   # green
BET_CLR   = '#ea580c'   # orange
CALL_CLR  = '#2563eb'   # blue
FOLD_CLR  = '#9ca3af'   # gray

FONT = 'Helvetica'


# ── Graph construction ───────────────────────────────────────────────────────

def build():
    g = graphviz.Digraph('kuhn_itg', format='png', engine='dot')

    g.attr(
        rankdir='LR',
        fontname=FONT,
        ranksep='1.4',
        nodesep='0.7',
        dpi='200',
        label=('Kuhn Poker — Infoset Transition Graph (ITG)\n'
               'Player 1 holds Q  ·  Blue = P1  ·  Red = P2'),
        labelloc='t',
        labelfontsize='14',
        fontsize='14',
        splines='spline',
        bgcolor='white',
    )

    g.attr('node', fontname=FONT, fontsize='11')
    g.attr('edge', fontname=FONT, fontsize='9', arrowsize='0.7',
           penwidth='1.3')

    # ── Helpers ──────────────────────────────────────────────────────────
    def info_node(nid, player, card, history):
        hist = f'"{history}"' if history else '""'
        label = f'P{player} | {card} | {hist}'
        fill = P1_FILL if player == 1 else P2_FILL
        border = P1_BORDER if player == 1 else P2_BORDER
        g.node(nid, label, shape='box', style='filled,rounded',
               fillcolor=fill, color=border, penwidth='2',
               width='1.3', height='0.45')

    def term_node(nid, label):
        g.node(nid, label, shape='diamond', style='filled',
               fillcolor=TERM_FILL, color=TERM_BORDER,
               fontsize='9', width='0.5', height='0.5', penwidth='1.5')

    # ── Stage 1: P1 initial (history "") ─────────────────────────────────
    with g.subgraph() as s:
        s.attr(rank='same')
        info_node('P1_Q', 1, 'Q', '')

    # ── Stage 2: P2 after check (history "c") ────────────────────────────
    with g.subgraph() as s:
        s.attr(rank='same')
        info_node('P2_J_c', 2, 'J', 'c')
        info_node('P2_K_c', 2, 'K', 'c')
    g.edge('P2_J_c', 'P2_K_c', style='invis')

    # ── Stage 3: P2 after bet (history "b") ──────────────────────────────
    with g.subgraph() as s:
        s.attr(rank='same')
        info_node('P2_J_b', 2, 'J', 'b')
        info_node('P2_K_b', 2, 'K', 'b')
    g.edge('P2_J_b', 'P2_K_b', style='invis')

    # ── Stage 4: P1 after check-bet (history "cb") ──────────────────────
    with g.subgraph() as s:
        s.attr(rank='same')
        info_node('P1_Q_cb', 1, 'Q', 'cb')

    # ── Terminals ────────────────────────────────────────────────────────
    # From Stage 2 (check → check = showdown)
    with g.subgraph() as s:
        s.attr(rank='same')
        term_node('T_Jc_cc', 'showdown')
        term_node('T_Kc_cc', 'showdown')

    # From Stage 3 (bet → call/fold)
    with g.subgraph() as s:
        s.attr(rank='same')
        term_node('T_Jb_call', 'showdown')
        term_node('T_Jb_fold', 'fold')
        term_node('T_Kb_call', 'showdown')
        term_node('T_Kb_fold', 'fold')

    # From Stage 4 (cb → call/fold)
    with g.subgraph() as s:
        s.attr(rank='same')
        term_node('T_cb_call', 'showdown')
        term_node('T_cb_fold', 'fold')

    # ── Edges: Stage 1 → Stages 2 & 3 ───────────────────────────────────
    # check → P2 infosets where P2 holds J or K
    g.edge('P1_Q', 'P2_J_c', label='check', color=CHECK_CLR,
           fontcolor=CHECK_CLR)
    g.edge('P1_Q', 'P2_K_c', label='check', color=CHECK_CLR,
           fontcolor=CHECK_CLR)

    # bet → P2 infosets where P2 holds J or K
    g.edge('P1_Q', 'P2_J_b', label='bet', color=BET_CLR,
           fontcolor=BET_CLR)
    g.edge('P1_Q', 'P2_K_b', label='bet', color=BET_CLR,
           fontcolor=BET_CLR)

    # ── Edges: Stage 2 → check (terminal) / bet (Stage 4) ───────────────
    g.edge('P2_J_c', 'T_Jc_cc', label='check', color=CHECK_CLR,
           fontcolor=CHECK_CLR)
    g.edge('P2_K_c', 'T_Kc_cc', label='check', color=CHECK_CLR,
           fontcolor=CHECK_CLR)

    g.edge('P2_J_c', 'P1_Q_cb', label='bet', color=BET_CLR,
           fontcolor=BET_CLR)
    g.edge('P2_K_c', 'P1_Q_cb', label='bet', color=BET_CLR,
           fontcolor=BET_CLR)

    # ── Edges: Stage 3 → call (terminal) / fold (terminal) ──────────────
    g.edge('P2_J_b', 'T_Jb_call', label='call', color=CALL_CLR,
           fontcolor=CALL_CLR)
    g.edge('P2_J_b', 'T_Jb_fold', label='fold', color=FOLD_CLR,
           fontcolor=FOLD_CLR, style='dashed')

    g.edge('P2_K_b', 'T_Kb_call', label='call', color=CALL_CLR,
           fontcolor=CALL_CLR)
    g.edge('P2_K_b', 'T_Kb_fold', label='fold', color=FOLD_CLR,
           fontcolor=FOLD_CLR, style='dashed')

    # ── Edges: Stage 4 → call (terminal) / fold (terminal) ──────────────
    g.edge('P1_Q_cb', 'T_cb_call', label='call', color=CALL_CLR,
           fontcolor=CALL_CLR)
    g.edge('P1_Q_cb', 'T_cb_fold', label='fold', color=FOLD_CLR,
           fontcolor=FOLD_CLR, style='dashed')

    return g


def main():
    output_dir = os.path.join(os.path.dirname(__file__), 'output')
    os.makedirs(output_dir, exist_ok=True)

    g = build()
    out_path = os.path.join(output_dir, 'kuhn_itg')
    g.render(out_path, cleanup=True)
    print(f'Saved: {out_path}.png')


if __name__ == '__main__':
    main()
