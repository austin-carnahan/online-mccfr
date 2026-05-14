"""IIG Reachability Test — Terminal coverage via upstream BFS.

For each game and each infoset I₀, computes:
  - How many infosets are upstream-reachable from I₀
  - How many terminals pass through at least one upstream-reachable infoset
  - Coverage = |reachable terminals| / |total terminals|

If coverage < 100%, shaped decay functions systematically underweight
terminals that are off the upstream BFS map — potentially explaining
why ConstantWeight (which gives all infosets equal weight) outperforms
shaped decays on wide games.

Usage:
    python -m experiments.iig_reachability
"""

import pyspiel
from src.games import GAME_SPECS
from src.iig import IIG

GAMES = GAME_SPECS


def analyze_game(name, game_str):
    game = pyspiel.load_game(game_str)
    iig = IIG(game)

    num_infosets = iig.num_infosets
    num_terminals = iig.num_terminals

    print(f"\n{'='*70}")
    print(f"{name}")
    print(f"{'='*70}")
    print(f"  Infosets: {num_infosets}")
    print(f"  Terminals: {num_terminals}")
    print(f"  IIG edges: {iig.num_edges}")

    # For each infoset, compute upstream reachability
    coverages = []
    upstream_sizes = []

    for iid in sorted(iig.infosets):
        # Get all upstream-reachable infosets (including I₀ itself)
        upstream_map = iig.levels(iid)
        upstream_iids = set(upstream_map.keys())

        # Collect all terminals passing through any upstream infoset
        reachable_tids = set()
        for uid in upstream_iids:
            reachable_tids.update(iig.z_set(uid))

        coverage = len(reachable_tids) / num_terminals if num_terminals > 0 else 0.0
        coverages.append(coverage)
        upstream_sizes.append(len(upstream_iids))

    # Summary statistics
    min_cov = min(coverages)
    max_cov = max(coverages)
    avg_cov = sum(coverages) / len(coverages)
    full_coverage_count = sum(1 for c in coverages if c >= 1.0 - 1e-9)

    min_up = min(upstream_sizes)
    max_up = max(upstream_sizes)
    avg_up = sum(upstream_sizes) / len(upstream_sizes)

    print(f"\n  Upstream BFS size (infosets reached):")
    print(f"    min: {min_up}  max: {max_up}  avg: {avg_up:.1f}  "
          f"(of {num_infosets} total)")

    print(f"\n  Terminal coverage via upstream BFS:")
    print(f"    min: {min_cov:.1%}  max: {max_cov:.1%}  avg: {avg_cov:.1%}")
    print(f"    Full coverage (100%): {full_coverage_count}/{num_infosets} infosets")

    if min_cov < 1.0 - 1e-9:
        print(f"\n  ⚠ NOT ALL TERMINALS REACHABLE via upstream BFS!")
        print(f"    Shaped decays will underweight {1-min_cov:.1%} of terminals")
        print(f"    at the worst-case infoset.")

        # Show the worst-case infosets
        print(f"\n  Bottom 5 infosets by coverage:")
        indexed = [(iid, coverages[i], upstream_sizes[i])
                   for i, iid in enumerate(sorted(iig.infosets))]
        indexed.sort(key=lambda x: x[1])
        for iid, cov, ups in indexed[:5]:
            z_size = len(iig.z_set(iid))
            print(f"    {iig._fmt_id(iid):35s}  coverage={cov:.1%}  "
                  f"upstream={ups}  |Z(I)|={z_size}")
    else:
        print(f"\n  ✓ All terminals reachable from every infoset via upstream BFS.")

    # Level depth distribution
    max_levels = []
    for iid in iig.infosets:
        level_sets = iig.level_sets(iid)
        max_levels.append(max(level_sets) if level_sets else 0)

    print(f"\n  Max upstream BFS depth:")
    print(f"    min: {min(max_levels)}  max: {max(max_levels)}  "
          f"avg: {sum(max_levels)/len(max_levels):.1f}")

    return {
        "name": name,
        "num_infosets": num_infosets,
        "num_terminals": num_terminals,
        "min_coverage": min_cov,
        "max_coverage": max_cov,
        "avg_coverage": avg_cov,
        "full_coverage_count": full_coverage_count,
        "min_upstream": min_up,
        "max_upstream": max_up,
    }


def main():
    print("IIG Reachability Test")
    print("=" * 70)
    print("For each infoset, how many terminals are reachable via upstream BFS?")
    print("If < 100%, shaped decay functions underweight unreachable terminals.")

    results = {}
    for name, game_str in GAMES.items():
        results[name] = analyze_game(name, game_str)

    # Cross-game summary
    print(f"\n\n{'='*70}")
    print("CROSS-GAME SUMMARY")
    print(f"{'='*70}")
    print(f"{'Game':<20} {'Infosets':>8} {'Terminals':>10} "
          f"{'Min Cov':>8} {'Avg Cov':>8} {'Full':>8}")
    print("-" * 70)
    for name in GAMES:
        r = results[name]
        full_str = f"{r['full_coverage_count']}/{r['num_infosets']}"
        print(f"{name:<20} {r['num_infosets']:>8} {r['num_terminals']:>10} "
              f"{r['min_coverage']:>7.1%} {r['avg_coverage']:>7.1%} "
              f"{full_str:>8}")


if __name__ == "__main__":
    main()
