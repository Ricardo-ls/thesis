# Figure 3 Spread Definition

- direct verdict: valid raw-spread replacement for the canonical_v1 evidence layer
- what is averaged: raw per-case `masked_ADE` values
- grouping keys: `condition`, `method`
- N: deterministic methods use trajectories; DDPM methods use trajectory-seed cases
- spread type: std
- spread population: raw per-case values from `per_case_results_seed_level.csv`
- does spread use already-averaged summaries: NO
- source file: `outputs/stage3/canonical_v1/raw/per_case_results_seed_level.csv`
- note: this replacement uses raw rows and does not reuse the old random-span summary std
