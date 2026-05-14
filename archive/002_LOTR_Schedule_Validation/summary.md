# LOTR schedule divergence probe

## Config

- games: ['kuhn_poker', 'leduc_poker', 'goofspiel', 'liars_dice']
- schedules: ['max_backoff_r05', 'far_backoff_r05', 'local_uniform_r05', 'near_active_r05']
- rho: 0.5
- states_per_game: 5
- sims_per_state: 100
- num_seeds: 3
- seed_offset: 0
- seeds: [0, 1, 2]
- epsilon: 0.4
- gamma: 0.01
- elapsed_s: 0.398203

## Empirical Coin Divergence

| game | schedule | states | iters | coin diverged | no divergence | mean D | D range |
|---|---|---:|---:|---:|---:|---:|---|
| kuhn_poker | max_backoff_r05 | 5 | 500 | 0.466 | 0.534 | 1.4 | 1..2 |
| kuhn_poker | far_backoff_r05 | 5 | 500 | 0.462 | 0.538 | 1.4 | 1..2 |
| kuhn_poker | local_uniform_r05 | 5 | 500 | 0.506 | 0.494 | 1.4 | 1..2 |
| kuhn_poker | near_active_r05 | 5 | 500 | 0.504 | 0.496 | 1.4 | 1..2 |
| leduc_poker | max_backoff_r05 | 5 | 500 | 0.496 | 0.504 | 1.8 | 1..3 |
| leduc_poker | far_backoff_r05 | 5 | 500 | 0.5 | 0.5 | 1.8 | 1..3 |
| leduc_poker | local_uniform_r05 | 5 | 500 | 0.486 | 0.514 | 1.8 | 1..3 |
| leduc_poker | near_active_r05 | 5 | 500 | 0.49 | 0.51 | 1.8 | 1..3 |
| goofspiel | max_backoff_r05 | 5 | 500 | 0.488 | 0.512 | 3 | 1..5 |
| goofspiel | far_backoff_r05 | 5 | 500 | 0.484 | 0.516 | 3 | 1..5 |
| goofspiel | local_uniform_r05 | 5 | 500 | 0.464 | 0.536 | 3 | 1..5 |
| goofspiel | near_active_r05 | 5 | 500 | 0.514 | 0.486 | 3 | 1..5 |
| liars_dice | max_backoff_r05 | 5 | 500 | 0.51 | 0.49 | 1.8 | 1..3 |
| liars_dice | far_backoff_r05 | 5 | 500 | 0.496 | 0.504 | 1.8 | 1..3 |
| liars_dice | local_uniform_r05 | 5 | 500 | 0.466 | 0.534 | 1.8 | 1..3 |
| liars_dice | near_active_r05 | 5 | 500 | 0.462 | 0.538 | 1.8 | 1..3 |

## Plots

- `plots/kuhn_poker_divergence_hist.png`: stacked by backoff distance from active state, colored by node type.
- `plots/leduc_poker_divergence_hist.png`: stacked by backoff distance from active state, colored by node type.
- `plots/goofspiel_divergence_hist.png`: stacked by backoff distance from active state, colored by node type.
- `plots/liars_dice_divergence_hist.png`: stacked by backoff distance from active state, colored by node type.
