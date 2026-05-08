"""Kuhn Poker — Infoset Intersection Graph (IIG), P1 holds Queen.

Directed upstream adjacency induced by shared terminal histories.
Edge J → I means J occurs exactly one player decision before I
along at least one terminal history passing through both.

Active infoset I₀ = P1 | Q | "cb".
Levels are upstream BFS layers from I₀.

Usage:
    python kuhn_iig.py
"""

import os
import graphviz

# ── Palette ──────────────────────────────────────────────────────────────────

FONT = 'Helvetica'

# Level-dependent fills (P1 = blue tones, P2 = red tones)
# Level 0 (active): saturated / bold
P1_FILL_L0 = '#60a5fa'   # strong blue
P1_BDR_L0  = '#1d4ed8'
P2_FILL_L0 = '#f87171'   # strong red
P2_BDR_L0  = '#b91c1c'

# Level 1: medium
P1_FILL_L1 = '#93c5fd'
P1_BDR_L1  = '#2563eb'
P2_FILL_L1 = '#fca5a5'
P2_BDR_L1  = '#dc2626'

# Level 2: light
P1_FILL_L2 = '#dbeafe'
P1_BDR_L2  = '#3b82f6'
P2_FILL_L2 = '#fee2e2'
P2_BDR_L2  = '#ef4444'

# Level 3: very light (downstream from L2)
P1_FILL_L3 = '#eff6ff'
P1_BDR_L3  = '#93c5fd'
P2_FILL_L3 = '#fff1f2'
P2_BDR_L3  = '#fca5a5'

EDGE_CLR   = '#7c3aed'   # purple — distinct from ITG green/orange


# ── Graph construction ───────────────────────────────────────────────────────

def build():
    g = graphviz.Digraph('kuhn_iig', format='png', engine='dot')

    g.attr(
        rankdir='LR',
        fontname=FONT,
        ranksep='1.6',
        nodesep='0.8',
        dpi='200',
        label=('Kuhn Poker — Infoset Intersection Graph (IIG)\n'
               'P1 = Q slice  ·  I\u2080 = P1 | Q | "cb"  ·  '
               'Edges = shared-terminal-history adjacency'),
        labelloc='t',
        labelfontsize='14',
        fontsize='14',
        splines='spline',
        bgcolor='white',
    )

    g.attr('node', fontname=FONT, fontsize='11')
    g.attr('edge', fontname=FONT, fontsize='9', arrowsize='0.7',
           penwidth='1.3', color=EDGE_CLR)

    # ── Node helper ──────────────────────────────────────────────────────

    def info_node(nid, player, card, history, level):
        hist = f'"{history}"' if history else '""'
        label = f'P{player} | {card} | {hist}'

        if level == 0:
            fill = P1_FILL_L0 if player == 1 else P2_FILL_L0
            border = P1_BDR_L0 if player == 1 else P2_BDR_L0
            pw = '3'
        elif level == 1:
            fill = P1_FILL_L1 if player == 1 else P2_FILL_L1
            border = P1_BDR_L1 if player == 1 else P2_BDR_L1
            pw = '2'
        elif level == 2:
            fill = P1_FILL_L2 if player == 1 else P2_FILL_L2
            border = P1_BDR_L2 if player == 1 else P2_BDR_L2
            pw = '2'
        elif level == 3:
            fill = P1_FILL_L3 if player == 1 else P2_FILL_L3
            border = P1_BDR_L3 if player == 1 else P2_BDR_L3
            pw = '1.5'

        g.node(nid, label, shape='box', style='filled,rounded',
               fillcolor=fill, color=border, penwidth=pw,
               width='1.4', height='0.5')

    # ── Nodes by IIG level ───────────────────────────────────────────────

    # Level 2 (leftmost column)
    with g.subgraph() as s:
        s.attr(rank='same')
        info_node('P1_Q', 1, 'Q', '', level=2)

    # Level 1
    with g.subgraph() as s:
        s.attr(rank='same')
        info_node('P2_J_c', 2, 'J', 'c', level=1)
        info_node('P2_K_c', 2, 'K', 'c', level=1)
    g.edge('P2_J_c', 'P2_K_c', style='invis')

    # Level 0 (active infoset)
    with g.subgraph() as s:
        s.attr(rank='same')
        info_node('P1_Q_cb', 1, 'Q', 'cb', level=0)

    # Level 3 (downstream from Level 2)
    with g.subgraph() as s:
        s.attr(rank='same')
        info_node('P2_J_b', 2, 'J', 'b', level=3)
        info_node('P2_K_b', 2, 'K', 'b', level=3)
    g.edge('P2_J_b', 'P2_K_b', style='invis')

    # ── Directed IIG edges (upstream: J → I) ────────────────────────────

    # Level 2 → Level 1
    g.edge('P1_Q', 'P2_J_c')
    g.edge('P1_Q', 'P2_K_c')

    # Level 1 → Level 0
    g.edge('P2_J_c', 'P1_Q_cb')
    g.edge('P2_K_c', 'P1_Q_cb')

    # Level 2 → Level 3 (downstream — don't pass through I₀ but
    # are reachable in the IIG via shared terminal histories)
    g.edge('P1_Q', 'P2_J_b', color='#c4b5fd')
    g.edge('P1_Q', 'P2_K_b', color='#c4b5fd')

    # ── Level legend ─────────────────────────────────────────────────────

    with g.subgraph(name='cluster_legend') as leg:
        leg.attr(label='IIG Levels', fontsize='10', fontname=FONT,
                 style='rounded', color='#e5e7eb', bgcolor='#fafafa')
        leg.node('leg0', 'Level 0  (active)', shape='box',
                 style='filled,rounded', fillcolor=P1_FILL_L0,
                 color=P1_BDR_L0, fontsize='9', fontname=FONT,
                 width='1.2', height='0.3', penwidth='2')
        leg.node('leg1', 'Level 1', shape='box',
                 style='filled,rounded', fillcolor=P2_FILL_L1,
                 color=P2_BDR_L1, fontsize='9', fontname=FONT,
                 width='1.2', height='0.3', penwidth='1.5')
        leg.node('leg2', 'Level 2', shape='box',
                 style='filled,rounded', fillcolor=P1_FILL_L2,
                 color=P1_BDR_L2, fontsize='9', fontname=FONT,
                 width='1.2', height='0.3', penwidth='1.5')
        leg.node('leg3', 'Level 3', shape='box',
                 style='filled,rounded', fillcolor=P2_FILL_L3,
                 color=P2_BDR_L3, fontsize='9', fontname=FONT,
                 width='1.2', height='0.3', penwidth='1')
        leg.edge('leg0', 'leg1', style='invis')
        leg.edge('leg1', 'leg2', style='invis')
        leg.edge('leg2', 'leg3', style='invis')

    return g


def main():
    output_dir = os.path.join(os.path.dirname(__file__), 'output')
    os.makedirs(output_dir, exist_ok=True)

    g = build()
    out_path = os.path.join(output_dir, 'kuhn_iig')
    g.render(out_path, cleanup=True)
    print(f'Saved: {out_path}.png')


if __name__ == '__main__':
    main()
