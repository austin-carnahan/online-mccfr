"""Tests for the offline OS-MCCFR outcome sampling wrapper."""

from src.outcome_sampling import run_os_baseline, make_checkpoints


def test_make_checkpoints():
    """Checkpoints are log-spaced and include the final iteration."""
    cps = make_checkpoints(10_000)
    assert cps == [10, 100, 1000, 10_000]


def test_make_checkpoints_small():
    """Small iteration count still includes the final iteration."""
    cps = make_checkpoints(50)
    assert cps == [10, 50]


def test_run_returns_results():
    """run_os_baseline returns results and a solver object."""
    results, solver = run_os_baseline("leduc_poker", 100, checkpoints=[50, 100])
    assert len(results) == 2
    assert results[0][0] == 50   # iteration number
    assert results[1][0] == 100


def test_solver_has_average_policy():
    """The returned solver exposes an average policy."""
    _, solver = run_os_baseline("leduc_poker", 10, checkpoints=[10])
    policy = solver.average_policy()
    assert policy is not None
