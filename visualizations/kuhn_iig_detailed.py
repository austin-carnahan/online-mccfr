"""Kuhn Poker — Detailed IIG with terminal histories (P1 = Q slice).

Combines the IIG level coloring with the actual terminal histories
threading through each infoset, showing *why* each IIG edge exists.

Paths that pass through I₀ = P1|Q|"cb" are drawn solid and saturated.
Paths that don't pass through I₀ are drawn in a lighter style.

IIG levels use undirected graph distance from I₀:
  Level 0: I₀ itself
  Level 1: P2|J|"c", P2|K|"c"  (upstream, 1 hop)
  Level 2: P1|Q|""             (upstream, 2 hops)
  Level 3: P2|J|"b", P2|K|"b"  (downstream from L2, 3 hops from I₀)

10 terminal histories total for P1=Q:
  Through I₀:  (Q,J) cb-call, (Q,J) cb-fold, (Q,K) cb-call, (Q,K) cb-fold
  Not through I₀: (Q,J) cc, (Q,K) cc, (Q,J) b-call, (Q,J) b-fold,
                   (Q,K) b-call, (Q,K) b-fold

Usage:
    python kuhn_iig_detailed.py
"""

import os
import graphviz

FONT = 'Helvetica'

# ── Level fills (P1 = blue, P2 = red) ────────────────────────────────────────

STYLES = {
    # (player, level) → (fill, border)
    (1, 0): ('#60a5fa', '#1d4ed8'),   # L0 — bold blue
    (2, 0): ('#f87171', '#b91c1c'),
    (1, 1): ('#93c5fd', '#2563eb'),   # L1
    (2, 1): ('#fca5a5', '#dc2626'),
    (1, 2): ('#dbeafe', '#3b82f6'),   # L2
    (2, 2): ('#fee2e2', '#ef4444'),
    (1, 3): ('#eff6ff', '#93c5fd'),   # L3 — very light
    (2, 3): ('#fff1f2', '#fca5a5'),
}

# Edge colours
ACTIVE_EDGE  = '#7c3aed'   # purple — paths through I₀
FAR_EDGE     = '#c4b5fd'   # light purple — paths not through I₀ but in IIG

TERM_FILL    = '#f9fafb'
TERM_BORDER  = '#9ca3af'

# Terminal highlighting: through I₀ vs not
TERM_ACTIVE_FILL = '#ede9fe'   # faint purple
TERM_ACTIVE_BDR  = '#7c3aed'


def build():
    g = graphviz.Digraph('kuhn_iig_detail', format='png', engine='dot')

    g.attr(
        rankdir='LR',
        fontname=FONT,
        ranksep='1.3',
        nodesep='0.55',
        dpi='200',
        label=(
            'Kuhn Poker — IIG with Terminal Histories  (P1 = Q)\n'
            'I\u2080 = P1 | Q | "cb"  ·  '
            'Solid purple = paths through I\u2080  ·  '
            'Light purple = other sampled paths'
        ),
        labelloc='t',
        labelfontsize='13',
        fontsize='13',
        splines='spline',
        bgcolor='white',
    )

    g.attr('node', fontname=FONT, fontsize='10')
    g.attr('edge', fontname=FONT, fontsize='9', arrowsize='0.6')

    # ── Helpers ──────────────────────────────────────────────────────────

    def info_node(nid, player, card, history, level):
        hist = f'"{history}"' if history else '""'
        label = f'P{player} | {card} | {hist}'
        fill, border = STYLES[(player, level)]
        pw = '3' if level == 0 else '2' if level <= 2 else '1.5'
        g.node(nid, label, shape='box', style='filled,rounded',
               fillcolor=fill, color=border, penwidth=pw,
               width='1.3', height='0.45')

    def term(nid, label, active=False):
        fill = TERM_ACTIVE_FILL if active else TERM_FILL
        bdr = TERM_ACTIVE_BDR if active else TERM_BORDER
        pw = '1.8' if active else '1.2'
        g.node(nid, label, shape='diamond', style='filled',
               fillcolor=fill, color=bdr, penwidth=pw,
               fontsize='8', width='0.45', height='0.45')

    def active_edge(src, dst, label=''):
        g.edge(src, dst, label=label, color=ACTIVE_EDGE,
               fontcolor=ACTIVE_EDGE, penwidth='1.4')

    def far_edge(src, dst, label=''):
        g.edge(src, dst, label=label, color=FAR_EDGE,
               fontcolor='#a78bfa', penwidth='1.0')

    # ── Column 1: Level 2 — P1 root ─────────────────────────────────────

    with g.subgraph() as s:
        s.attr(rank='same')
        info_node('P1_Q', 1, 'Q', '', level=2)

    # ── Column 2: Level 1 + Level 3 ──────────────────────────────────────

    with g.subgraph() as s:
        s.attr(rank='same')
        info_node('P2_J_c', 2, 'J', 'c', level=1)
        info_node('P2_K_c', 2, 'K', 'c', level=1)
        info_node('P2_J_b', 2, 'J', 'b', level=3)
        info_node('P2_K_b', 2, 'K', 'b', level=3)

    # vertical ordering hint
    for a, b in [('P2_J_c', 'P2_K_c'), ('P2_K_c', 'P2_J_b'),
                 ('P2_J_b', 'P2_K_b')]:
        g.edge(a, b, style='invis')

    # ── Column 3: Level 0 + some terminals ───────────────────────────────

    with g.subgraph() as s:
        s.attr(rank='same')
        info_node('P1_Q_cb', 1, 'Q', 'cb', level=0)
        # Terminals from the "cc" path (not through I₀)
        term('T_Jcc', 'cc\nshowdown', active=False)
        term('T_Kcc', 'cc\nshowdown', active=False)
        # Terminals from "b" path (not through I₀)
        term('T_Jbc', 'bc\nshowdown', active=False)
        term('T_Jbf', 'bf\nfold', active=False)
        term('T_Kbc', 'bc\nshowdown', active=False)
        term('T_Kbf', 'bf\nfold', active=False)

    for a, b in [('P1_Q_cb', 'T_Jcc'), ('T_Jcc', 'T_Kcc'),
                 ('T_Kcc', 'T_Jbc'), ('T_Jbc', 'T_Jbf'),
                 ('T_Jbf', 'T_Kbc'), ('T_Kbc', 'T_Kbf')]:
        g.edge(a, b, style='invis')

    # ── Column 4: Terminals from I₀ ─────────────────────────────────────

    with g.subgraph() as s:
        s.attr(rank='same')
        term('T_cb_Jcall',  'cbc\nshowdown', active=True)
        term('T_cb_Jfold',  'cbf\nfold', active=True)
        term('T_cb_Kcall',  'cbc\nshowdown', active=True)
        term('T_cb_Kfold',  'cbf\nfold', active=True)

    for a, b in [('T_cb_Jcall', 'T_cb_Jfold'),
                 ('T_cb_Jfold', 'T_cb_Kcall'),
                 ('T_cb_Kcall', 'T_cb_Kfold')]:
        g.edge(a, b, style='invis')

    # ══════════════════════════════════════════════════════════════════════
    # PATHS THROUGH I₀  (solid purple)
    # ══════════════════════════════════════════════════════════════════════

    # (Q,J) c → b → call     P1_Q → P2_J_c → P1_Q_cb → showdown
    active_edge('P1_Q',    'P2_J_c',     label='check')
    active_edge('P2_J_c',  'P1_Q_cb',    label='bet')
    active_edge('P1_Q_cb', 'T_cb_Jcall', label='call')

    # (Q,J) c → b → fold
    active_edge('P1_Q_cb', 'T_cb_Jfold', label='fold')

    # (Q,K) c → b → call     P1_Q → P2_K_c → P1_Q_cb → showdown
    active_edge('P1_Q',    'P2_K_c',     label='check')
    active_edge('P2_K_c',  'P1_Q_cb',    label='bet')
    active_edge('P1_Q_cb', 'T_cb_Kcall', label='call')

    # (Q,K) c → b → fold
    active_edge('P1_Q_cb', 'T_cb_Kfold', label='fold')

    # ══════════════════════════════════════════════════════════════════════
    # PATHS NOT THROUGH I₀  (dashed gray)
    # ══════════════════════════════════════════════════════════════════════

    # (Q,J) c → c → showdown
    far_edge('P2_J_c', 'T_Jcc', label='check')

    # (Q,K) c → c → showdown
    far_edge('P2_K_c', 'T_Kcc', label='check')

    # (Q,J) b → call
    far_edge('P1_Q',   'P2_J_b', label='bet')
    far_edge('P2_J_b', 'T_Jbc',  label='call')

    # (Q,J) b → fold
    far_edge('P2_J_b', 'T_Jbf', label='fold')

    # (Q,K) b → call
    far_edge('P1_Q',   'P2_K_b', label='bet')
    far_edge('P2_K_b', 'T_Kbc',  label='call')

    # (Q,K) b → fold
    far_edge('P2_K_b', 'T_Kbf', label='fold')

    # ── Legend ───────────────────────────────────────────────────────────

    with g.subgraph(name='cluster_legend') as leg:
        leg.attr(label='Legend', fontsize='10', fontname=FONT,
                 style='rounded', color='#e5e7eb', bgcolor='#fafafa')

        leg.node('L0', 'Level 0 (I\u2080)', shape='box',
                 style='filled,rounded', fillcolor='#60a5fa',
                 color='#1d4ed8', fontsize='9', fontname=FONT,
                 width='1.1', height='0.28', penwidth='2')
        leg.node('L1', 'Level 1', shape='box',
                 style='filled,rounded', fillcolor='#fca5a5',
                 color='#dc2626', fontsize='9', fontname=FONT,
                 width='1.1', height='0.28', penwidth='1.5')
        leg.node('L2', 'Level 2', shape='box',
                 style='filled,rounded', fillcolor='#dbeafe',
                 color='#3b82f6', fontsize='9', fontname=FONT,
                 width='1.1', height='0.28', penwidth='1.5')
        leg.node('L3', 'Level 3', shape='box',
                 style='filled,rounded', fillcolor='#fff1f2',
                 color='#fca5a5', fontsize='9', fontname=FONT,
                 width='1.1', height='0.28', penwidth='1')

        leg.edge('L0', 'L1', style='invis')
        leg.edge('L1', 'L2', style='invis')
        leg.edge('L2', 'L3', style='invis')

    return g


def main():
    out_dir = os.path.join(os.path.dirname(__file__), 'output')
    os.makedirs(out_dir, exist_ok=True)
    g = build()
    path = g.render(filename='kuhn_iig_detailed', directory=out_dir,
                    cleanup=True)
    print(f'Saved: {path}')


if __name__ == '__main__':
    main()
