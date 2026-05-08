# cython: boundscheck=False, wraparound=False, cdivision=True
"""Fast inner-loop operations for MCCFR bots (OOS, Depth-Delta).

Provides C-level implementations of:
  1. Regret matching + action sampling (fused)
  2. ε-on-policy blend + sampling (fused)
  3. Regret/avg-strategy updates
  4. Fast PCG RNG (no Python overhead per sample)

All functions operate on raw C arrays or memoryviews to avoid NumPy
allocation overhead on tiny (2-10 element) arrays.
"""

from libc.stdlib cimport malloc, free
from libc.string cimport memset, memcpy
from libc.math cimport fmax

import numpy as np
cimport numpy as np

np.import_array()

# ═══════════════════════════════════════════════════════════════════════════
# PCG-XSH-RR (32-bit output, 64-bit state) — fast, high-quality RNG
# ═══════════════════════════════════════════════════════════════════════════

cdef unsigned long long PCG_MULTIPLIER = 6364136223846793005ULL
cdef unsigned long long PCG_INCREMENT = 1442695040888963407ULL

cdef struct PCGState:
    unsigned long long state

cdef inline void pcg_seed(PCGState* rng, unsigned long long seed) noexcept nogil:
    rng.state = 0
    pcg_next(rng)
    rng.state += seed
    pcg_next(rng)

cdef inline unsigned int pcg_next(PCGState* rng) noexcept nogil:
    cdef unsigned long long old = rng.state
    rng.state = old * PCG_MULTIPLIER + PCG_INCREMENT
    cdef unsigned int xorshifted = <unsigned int>(((old >> 18) ^ old) >> 27)
    cdef unsigned int rot = <unsigned int>(old >> 59)
    return (xorshifted >> rot) | (xorshifted << ((<unsigned int>32 - rot) & 31))

cdef inline double pcg_uniform(PCGState* rng) noexcept nogil:
    """Uniform [0, 1) double from PCG."""
    return <double>pcg_next(rng) / 4294967296.0


# ═══════════════════════════════════════════════════════════════════════════
# Scratch buffer (thread-local would be needed for parallelism, but we're
# single-threaded in the simulation loop)
# ═══════════════════════════════════════════════════════════════════════════

DEF MAX_ACTIONS = 32  # generous upper bound for any game we test

cdef double[MAX_ACTIONS] _scratch_policy
cdef double[MAX_ACTIONS] _scratch_sample


# ═══════════════════════════════════════════════════════════════════════════
# Core operations
# ═══════════════════════════════════════════════════════════════════════════

cdef class FastRNG:
    """Cython-level RNG wrapper exposed to Python."""
    cdef PCGState _state

    def __init__(self, unsigned long long seed=42):
        pcg_seed(&self._state, seed)

    cdef inline unsigned int _next(self) noexcept nogil:
        return pcg_next(&self._state)

    cdef inline double _uniform(self) noexcept nogil:
        return pcg_uniform(&self._state)

    cdef inline int _choice_cdf(self, double* probs, int n) noexcept nogil:
        """Sample index from probability distribution using CDF inversion."""
        cdef double r = pcg_uniform(&self._state)
        cdef double cumul = 0.0
        cdef int i
        for i in range(n - 1):
            cumul += probs[i]
            if r < cumul:
                return i
        return n - 1

    def randint(self, int n):
        """Python-accessible: random int in [0, n)."""
        return <int>(pcg_next(&self._state) % <unsigned int>n)

    def uniform(self):
        """Python-accessible: uniform [0, 1)."""
        return pcg_uniform(&self._state)

    def random(self):
        """Alias for uniform() — matches numpy interface."""
        return pcg_uniform(&self._state)

    def choice(self, a, p=None):
        """Python-accessible: numpy-compatible choice.

        Args:
            a: If int, sample from [0, a). If sequence, pick an element.
            p: Optional probability array. If None, uniform.

        Returns:
            An index (if a is int) or an element (if a is sequence).
        """
        cdef int n
        cdef int idx
        cdef int i
        cdef double r, cumul
        cdef double[::1] pv

        if isinstance(a, (int, np.integer)):
            n = <int>a
            if p is None:
                return <int>(pcg_next(&self._state) % <unsigned int>n)
            # Sample index from probability distribution
            pv = np.ascontiguousarray(p, dtype=np.float64)
            r = pcg_uniform(&self._state)
            cumul = 0.0
            for idx in range(n - 1):
                cumul += pv[idx]
                if r < cumul:
                    return idx
            return n - 1
        else:
            # a is a sequence — pick an element
            n = len(a)
            if p is None:
                idx = <int>(pcg_next(&self._state) % <unsigned int>n)
            else:
                pv = np.ascontiguousarray(p, dtype=np.float64)
                r = pcg_uniform(&self._state)
                cumul = 0.0
                idx = n - 1
                for i in range(n - 1):
                    cumul += pv[i]
                    if r < cumul:
                        idx = i
                        break
            return a[idx]


def regret_match_sample(double[::1] regrets, double gamma, FastRNG rng):
    """Regret matching → policy, then sample an action index.

    Fuses regret matching + sampling into one call with no intermediate
    Python array allocation.

    Args:
        regrets: Cumulative regret array (contiguous double memoryview).
        gamma: Floor exploration parameter.
        rng: FastRNG instance.

    Returns:
        (policy_array, sampled_idx): numpy policy and chosen index.
    """
    cdef int n = regrets.shape[0]
    cdef int i
    cdef double total = 0.0
    cdef double inv_n = 1.0 / n
    cdef double g = gamma

    # Regret matching: positive regrets → proportional
    for i in range(n):
        _scratch_policy[i] = fmax(regrets[i], 0.0)
        total += _scratch_policy[i]

    if total > 0.0:
        for i in range(n):
            _scratch_policy[i] /= total
    else:
        for i in range(n):
            _scratch_policy[i] = inv_n

    # Apply gamma floor
    if g > 0.0:
        for i in range(n):
            _scratch_policy[i] = g * inv_n + (1.0 - g) * _scratch_policy[i]

    # Sample
    cdef int idx = rng._choice_cdf(_scratch_policy, n)

    # Copy to numpy for caller (they need the policy for reach updates)
    cdef np.ndarray[double, ndim=1] policy = np.empty(n, dtype=np.float64)
    cdef double* pdata = <double*>policy.data
    for i in range(n):
        pdata[i] = _scratch_policy[i]

    return policy, idx


def regret_match_sample_eps(double[::1] regrets, double gamma,
                            double epsilon, FastRNG rng):
    """Regret matching → ε-on-policy blend → sample.

    For the update player: sample_policy = ε·uniform + (1-ε)·policy.
    Returns (policy, sample_policy, sampled_idx).

    Args:
        regrets: Cumulative regret array.
        gamma: Floor exploration parameter.
        epsilon: Exploration blend weight.
        rng: FastRNG instance.

    Returns:
        (policy_array, sample_policy_array, sampled_idx)
    """
    cdef int n = regrets.shape[0]
    cdef int i
    cdef double total = 0.0
    cdef double inv_n = 1.0 / n
    cdef double g = gamma

    # Regret matching
    for i in range(n):
        _scratch_policy[i] = fmax(regrets[i], 0.0)
        total += _scratch_policy[i]

    if total > 0.0:
        for i in range(n):
            _scratch_policy[i] /= total
    else:
        for i in range(n):
            _scratch_policy[i] = inv_n

    # Gamma floor
    if g > 0.0:
        for i in range(n):
            _scratch_policy[i] = g * inv_n + (1.0 - g) * _scratch_policy[i]

    # ε-on-policy blend into _scratch_sample
    for i in range(n):
        _scratch_sample[i] = epsilon * inv_n + (1.0 - epsilon) * _scratch_policy[i]

    # Sample from blended distribution
    cdef int idx = rng._choice_cdf(_scratch_sample, n)

    # Copy to numpy
    cdef np.ndarray[double, ndim=1] policy = np.empty(n, dtype=np.float64)
    cdef np.ndarray[double, ndim=1] sample_pol = np.empty(n, dtype=np.float64)
    cdef double* pdata = <double*>policy.data
    cdef double* sdata = <double*>sample_pol.data
    for i in range(n):
        pdata[i] = _scratch_policy[i]
        sdata[i] = _scratch_sample[i]

    return policy, sample_pol, idx


def update_regrets(double[::1] regret_table, int sampled_idx,
                   double c, double x, double W, int num_actions):
    """In-place regret update for the update player.

    For sampled action: regret += (c - x) * W
    For others:         regret += -x * W

    Args:
        regret_table: The info state's regret array (modified in-place).
        sampled_idx: Index of the sampled action.
        c: Suffix reach at child (before multiplying by policy[sampled]).
        x: Suffix reach at current node (c * policy[sampled]).
        W: u * opp_reach / l.
        num_actions: Number of legal actions.
    """
    cdef int i
    cdef double neg_xW = -x * W
    cdef double bonus = (c - x) * W
    for i in range(num_actions):
        if i == sampled_idx:
            regret_table[i] += bonus
        else:
            regret_table[i] += neg_xW


def update_avg_strategy(double[::1] avg_table, double[::1] policy,
                        double opp_reach, double sample_prob, int num_actions):
    """In-place average strategy update for the opponent's node.

    avg[a] += opp_reach * policy[a] / sample_prob

    Args:
        avg_table: The info state's avg strategy array (modified in-place).
        policy: Current regret-matched policy.
        opp_reach: Opponent reach probability.
        sample_prob: Sample probability (l for DD, δ*s1+(1-δ)*s2 for OOS).
        num_actions: Number of legal actions.
    """
    cdef int i
    cdef double scale = opp_reach / fmax(sample_prob, 1e-30)
    for i in range(num_actions):
        avg_table[i] += policy[i] * scale
