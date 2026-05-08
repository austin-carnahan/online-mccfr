"""Kuhn Poker game tree with information set overlay.

Generates the complete Kuhn Poker game tree (6 card deals × 5 terminals each),
with dashed connections between nodes sharing the same information set.

Node labels show the acting player's private card — nodes in the same infoset
share the same label, visually reinforcing that the player can't distinguish them.

Colors:  Blue = Player 0,  Red = Player 1,  Green = Chance,  Gray = Terminal.
Terminal values are Player 0 payoffs (zero-sum: P1 = −P0).

Usage:
    python -m visualizations.kuhn_game_tree
"""

import os
import graphviz

# ── Styling ──────────────────────────────────────────────────────────────────

P0_FILL    = '#dbeafe'   # light blue
P0_BORDER  = '#2563eb'   # blue
P1_FILL    = '#fee2e2'   # light red
P1_BORDER  = '#dc2626'   # red
CHANCE_FILL   = '#d1fae5'
CHANCE_BORDER = '#059669'
TERM_FILL   = '#f3f4f6'
TERM_BORDER = '#6b7280'

EDGE_COLOR = '#374151'   # dark gray for tree edges

# ── Game logic ───────────────────────────────────────────────────────────────

RANK = {'J': 0, 'Q': 1, 'K': 2}

# Ordered so P0-card pairs are adjacent: (JQ,JK), (QJ,QK), (KJ,KQ)
DEALS = [('J', 'Q'), ('J', 'K'),
         ('Q', 'J'), ('Q', 'K'),
         ('K', 'J'), ('K', 'Q')]


def p0_payoff(p0_card, p1_card, actions):
    """Player 0 payoff at a terminal node."""
    p0_wins = RANK[p0_card] > RANK[p1_card]
    if actions == 'pp':   return +1 if p0_wins else -1   # showdown, pot=2
    if actions == 'pbp':  return -1                       # P0 folds
    if actions == 'pbb':  return +2 if p0_wins else -2   # showdown, pot=4
    if actions == 'bp':   return +1                       # P1 folds
    if actions == 'bb':   return +2 if p0_wins else -2   # showdown, pot=4


# ── Graph construction ───────────────────────────────────────────────────────

def build():
    g = graphviz.Digraph('kuhn_game_tree', format='png', engine='dot')

    g.attr(
        rankdir='TB',
        fontname='Helvetica',
        ranksep='0.55',
        nodesep='0.12',
        dpi='200',
        label=('Kuhn Poker — Game Tree with Information Sets\n'
               'Node labels = acting player\'s private card  |  '
               'Dashed lines = information sets  |  '
               'Terminal values = P0 payoff'),
        labelloc='t',
        labelfontsize='12',
    )
    g.attr('node', fontname='Helvetica', fontsize='9',
           fixedsize='true', width='0.45', height='0.45')
    g.attr('edge', fontname='Helvetica', fontsize='8', color=EDGE_COLOR)

    # ── Chance node ──────────────────────────────────────────────────────
    g.node('C', 'C', shape='diamond', style='filled',
           fillcolor=CHANCE_FILL, color=CHANCE_BORDER,
           penwidth='2', width='0.55', height='0.55', fontsize='11')

    # ── Build subtree for each deal ──────────────────────────────────────
    for p0, p1 in DEALS:
        d = f'{p0}{p1}'

        # P0 first decision
        g.node(d, p0, shape='circle', style='filled',
               fillcolor=P0_FILL, color=P0_BORDER, penwidth='1.5')
        g.edge('C', d, label=f' {p0},{p1} ', fontsize='7')

        # ── P0 passes → P1 decision ─────────────────────────────────
        g.node(f'{d}_p', p1, shape='circle', style='filled',
               fillcolor=P1_FILL, color=P1_BORDER, penwidth='1.5')
        g.edge(d, f'{d}_p', label='p')

        # ── P0 bets → P1 decision ───────────────────────────────────
        g.node(f'{d}_b', p1, shape='circle', style='filled',
               fillcolor=P1_FILL, color=P1_BORDER, penwidth='1.5')
        g.edge(d, f'{d}_b', label='b')

        # ── pass-pass → terminal (showdown) ─────────────────────────
        pay = p0_payoff(p0, p1, 'pp')
        g.node(f'{d}_pp', f'{pay:+d}', shape='rect', style='filled,rounded',
               fillcolor=TERM_FILL, color=TERM_BORDER,
               width='0.35', height='0.28', fontsize='8')
        g.edge(f'{d}_p', f'{d}_pp', label='p')

        # ── pass → bet → P0 second decision ─────────────────────────
        g.node(f'{d}_pb', p0, shape='circle', style='filled',
               fillcolor=P0_FILL, color=P0_BORDER, penwidth='1.5')
        g.edge(f'{d}_p', f'{d}_pb', label='b')

        # ── pass-bet-pass → terminal (P0 folds) ─────────────────────
        pay = p0_payoff(p0, p1, 'pbp')
        g.node(f'{d}_pbp', f'{pay:+d}', shape='rect', style='filled,rounded',
               fillcolor=TERM_FILL, color=TERM_BORDER,
               width='0.35', height='0.28', fontsize='8')
        g.edge(f'{d}_pb', f'{d}_pbp', label='p')

        # ── pass-bet-bet → terminal (showdown) ──────────────────────
        pay = p0_payoff(p0, p1, 'pbb')
        g.node(f'{d}_pbb', f'{pay:+d}', shape='rect', style='filled,rounded',
               fillcolor=TERM_FILL, color=TERM_BORDER,
               width='0.35', height='0.28', fontsize='8')
        g.edge(f'{d}_pb', f'{d}_pbb', label='b')

        # ── bet-pass → terminal (P1 folds) ──────────────────────────
        pay = p0_payoff(p0, p1, 'bp')
        g.node(f'{d}_bp', f'{pay:+d}', shape='rect', style='filled,rounded',
               fillcolor=TERM_FILL, color=TERM_BORDER,
               width='0.35', height='0.28', fontsize='8')
        g.edge(f'{d}_b', f'{d}_bp', label='p')

        # ── bet-bet → terminal (showdown) ────────────────────────────
        pay = p0_payoff(p0, p1, 'bb')
        g.node(f'{d}_bb', f'{pay:+d}', shape='rect', style='filled,rounded',
               fillcolor=TERM_FILL, color=TERM_BORDER,
               width='0.35', height='0.28', fontsize='8')
        g.edge(f'{d}_b', f'{d}_bb', label='b')

    # ── Information set connections (dashed lines) ───────────────────────
    #
    # Each infoset has exactly 2 game nodes (one per possible opponent card).
    # P0 infosets are grouped by P0's card; P1 by P1's card.

    infosets = [
        # P0 first decision — grouped by P0's card
        ('P0: J',     'JQ',    'JK',    P0_BORDER),
        ('P0: Q',     'QJ',    'QK',    P0_BORDER),
        ('P0: K',     'KJ',    'KQ',    P0_BORDER),

        # P1 after pass — grouped by P1's card
        ('P1: J, p',  'QJ_p',  'KJ_p',  P1_BORDER),
        ('P1: Q, p',  'JQ_p',  'KQ_p',  P1_BORDER),
        ('P1: K, p',  'JK_p',  'QK_p',  P1_BORDER),

        # P1 after bet — grouped by P1's card
        ('P1: J, b',  'QJ_b',  'KJ_b',  P1_BORDER),
        ('P1: Q, b',  'JQ_b',  'KQ_b',  P1_BORDER),
        ('P1: K, b',  'JK_b',  'QK_b',  P1_BORDER),

        # P0 after pass-bet — grouped by P0's card
        ('P0: J, pb', 'JQ_pb', 'JK_pb', P0_BORDER),
        ('P0: Q, pb', 'QJ_pb', 'QK_pb', P0_BORDER),
        ('P0: K, pb', 'KJ_pb', 'KQ_pb', P0_BORDER),
    ]

    for label, n1, n2, color in infosets:
        g.edge(n1, n2,
               style='dashed', color=color, penwidth='1.5',
               label=f'  {label}  ', fontcolor=color, fontsize='7',
               constraint='false', dir='none')

    return g


def main():
    output_dir = os.path.join(os.path.dirname(__file__), 'output')
    os.makedirs(output_dir, exist_ok=True)

    g = build()
    out_path = os.path.join(output_dir, 'kuhn_game_tree')
    g.render(out_path, cleanup=True)
    print(f'Saved: {out_path}.png')


if __name__ == '__main__':
    main()
