"""Quick comparison of archive 010 (precomp OOS) vs archive 008 (original OOS)."""
import json

# New results (with precomp OOS, 1.5x matches)
with open("results/isgt_delta_ablation_full/ablation_summary.json") as f:
    new = json.load(f)

# Old results (archive 008, original OOS)
with open("archive/008_isgt_delta_ablation_full/isgt_delta_ablation_full/ablation_summary.json") as f:
    old = json.load(f)

games = ["kuhn_poker", "leduc_poker", "goofspiel", "liars_dice"]

print("=" * 80)
print("OOS PRECOMP vs ORIGINAL OOS at sims=1000")
print("=" * 80)
for g in games:
    old_oos = [x for x in old[g].get("oos", []) if x["sims_per_move"] == 1000]
    new_oos = [x for x in new[g].get("oos_precomp", []) if x["sims_per_move"] == 1000]
    old_v = old_oos[0]["exploitability"] if old_oos else None
    new_v = new_oos[0]["exploitability"] if new_oos else None
    if old_v and new_v:
        diff_pct = (new_v - old_v) / old_v * 100
        print(f"  {g:15s}  OOS={old_v:.4f}  OOS_precomp={new_v:.4f}  diff={diff_pct:+.1f}%")

print()
print("=" * 80)
print("FULL RANKINGS at sims=1000 (new run, 1.5x matches, precomp OOS)")
print("=" * 80)
for g in games:
    print(f"\n--- {g} ---")
    configs = []
    for config, entries in new[g].items():
        for e in entries:
            if e["sims_per_move"] == 1000:
                configs.append((e["exploitability"], config))
    configs.sort()
    for rank, (expl, config) in enumerate(configs, 1):
        marker = " ***" if config == "oos_precomp" else ""
        print(f"  {rank:2d}. {config:30s} {expl:.4f}{marker}")

# Also show all budgets for OOS comparison
print()
print("=" * 80)
print("OOS PRECOMP vs ORIGINAL OOS at ALL budgets")
print("=" * 80)
for g in games:
    print(f"\n--- {g} ---")
    for sims in [100, 250, 500, 1000]:
        old_oos = [x for x in old[g].get("oos", []) if x["sims_per_move"] == sims]
        new_oos = [x for x in new[g].get("oos_precomp", []) if x["sims_per_move"] == sims]
        old_v = old_oos[0]["exploitability"] if old_oos else None
        new_v = new_oos[0]["exploitability"] if new_oos else None
        if old_v and new_v:
            diff_pct = (new_v - old_v) / old_v * 100
            print(f"  sims={sims:>5d}  OOS={old_v:.4f}  OOS_precomp={new_v:.4f}  diff={diff_pct:+.1f}%")
