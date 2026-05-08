"""Quick smoke test for src.fast_ops Cython module."""
import numpy as np
from src.fast_ops import (
    FastRNG, regret_match_sample, regret_match_sample_eps,
    update_regrets, update_avg_strategy,
)


def test_regret_match_sample():
    rng = FastRNG(42)
    regrets = np.array([1.0, -2.0, 3.0, 0.5])
    policy, idx = regret_match_sample(regrets, 0.01, rng)
    assert abs(policy.sum() - 1.0) < 1e-10, f"Policy not normalized: {policy.sum()}"
    assert all(p > 0 for p in policy), "Gamma floor violated"
    assert 0 <= idx < 4
    print(f"  regret_match_sample: policy={policy}, idx={idx} ✓")


def test_regret_match_sample_eps():
    rng = FastRNG(42)
    regrets = np.array([1.0, -2.0, 3.0, 0.5])
    policy, sample_pol, idx = regret_match_sample_eps(regrets, 0.01, 0.4, rng)
    assert abs(policy.sum() - 1.0) < 1e-10
    assert abs(sample_pol.sum() - 1.0) < 1e-10
    # ε-blend should be more uniform than pure policy
    assert sample_pol.min() > policy.min()
    assert 0 <= idx < 4
    print(f"  regret_match_sample_eps: sample_pol={sample_pol}, idx={idx} ✓")


def test_update_regrets():
    regret_table = np.zeros(4)
    update_regrets(regret_table, 1, 0.8, 0.32, 2.5, 4)
    # sampled_idx=1: += (0.8 - 0.32) * 2.5 = 1.2
    # others: += -0.32 * 2.5 = -0.8
    assert abs(regret_table[1] - 1.2) < 1e-10, f"Got {regret_table[1]}"
    assert abs(regret_table[0] - (-0.8)) < 1e-10, f"Got {regret_table[0]}"
    assert abs(regret_table[2] - (-0.8)) < 1e-10
    assert abs(regret_table[3] - (-0.8)) < 1e-10
    print(f"  update_regrets: {regret_table} ✓")


def test_update_avg_strategy():
    avg = np.zeros(3)
    pol = np.array([0.5, 0.3, 0.2])
    update_avg_strategy(avg, pol, 0.6, 0.4, 3)
    # scale = 0.6 / 0.4 = 1.5
    assert abs(avg[0] - 0.75) < 1e-10, f"Got {avg[0]}"
    assert abs(avg[1] - 0.45) < 1e-10
    assert abs(avg[2] - 0.30) < 1e-10
    print(f"  update_avg_strategy: {avg} ✓")


def test_fastrng_choice():
    rng = FastRNG(123)
    # Int form
    counts = [0] * 3
    for _ in range(3000):
        counts[rng.choice(3)] += 1
    assert all(c > 800 for c in counts), f"Uniform choice broken: {counts}"

    # Sequence form
    items = [10, 20, 30]
    probs = np.array([0.1, 0.2, 0.7])
    counts = {10: 0, 20: 0, 30: 0}
    for _ in range(3000):
        counts[rng.choice(items, p=probs)] += 1
    assert counts[30] > counts[20] > counts[10], f"Weighted choice broken: {counts}"
    print(f"  FastRNG.choice: uniform={[c//30 for c in [counts[10],counts[20],counts[30]]]}% ✓")


def test_fastrng_randint():
    rng = FastRNG(7)
    counts = [0] * 5
    for _ in range(5000):
        counts[rng.randint(5)] += 1
    assert all(c > 800 for c in counts), f"randint broken: {counts}"
    print(f"  FastRNG.randint: {counts} ✓")


if __name__ == "__main__":
    print("Testing fast_ops...")
    test_regret_match_sample()
    test_regret_match_sample_eps()
    test_update_regrets()
    test_update_avg_strategy()
    test_fastrng_choice()
    test_fastrng_randint()
    print("\nAll fast_ops tests passed!")
