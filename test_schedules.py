
from src.depth_lotr import max_backoff, step, uniform, near_active, far_backoff
import numpy as np

# OOS parity: max_backoff == step(rho, depth=0)
for rho in [0.1, 0.5, 0.9]:
    for D in [1, 2, 3, 5, 8]:
        mb = max_backoff(rho).tau_array(D)
        sp = step(rho, depth=0).tau_array(D)
        assert np.allclose(mb, sp), f'mismatch at rho={rho} D={D}: {mb} vs {sp}'
print('OOS parity OK')

# Show shapes at D=5 for rho=0.5
print()
print('D=5, rho=0.5 tau arrays (per-depth target-action probability):')
for name, sched in [('uniform', uniform(0.5)), ('near_active', near_active(0.5)),
                     ('far_backoff', far_backoff(0.5)), ('max_backoff', max_backoff(0.5))]:
    tau = sched.tau_array(5)
    one_minus_tau = 1.0 - tau
    print(f'  {name}: tau={tau.round(4)}')
    print(f'         1-tau (per-pos diverge): {one_minus_tau.round(4)}')

# Also # Also # Also # Also # off=k | diverged) under each
print()
print('print('o=0.p: marginal P(backoff=k | diverged):')
for name, sched in [('for name, sciforfor name, sched in [('for name, sciforfor name, sched in [('for name, sciforff',for name, sched i), ('max_backoff', max_backoff(0.5))]:
    tau = sc    tau = sc    tau = sc  d     tau = sc    tau = sc  .0
    for d in range(5):
        p_at_d[d] = s        p_at_d[d] = s        p_at_d[d] = s 
                             
    cond = p_at_d / max(p_diverged, 1e-12)
    cond_    cond_    d[::-1]
    print(f'  {name}: P(div)={p_diverged:.4f}  P(k|div) for k=1..5: {cond_backoff.round(4)}')
