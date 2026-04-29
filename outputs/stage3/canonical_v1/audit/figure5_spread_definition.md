# Figure 5 Spread Definition

- direct verdict: still limited by historical alpha-sweep raw availability
- source used: historical `outputs/stage3/refinement/alpha_sweep/alpha_sweep_metrics.csv`
- what is averaged: per cell `masked_ADE` rows from the historical alpha sweep
- grouping keys: `alpha`
- N per alpha: number of historical rows available for that alpha
- spread type: std over historical row values
- does spread use raw per-trajectory rows: NO
- does spread use already-aggregated rows: YES
- trajectory variability separation: unavailable from historical raw because only condition/coarse/alpha summary rows exist
- DDPM seed variability separation: unavailable from historical raw because seed-level alpha sweep outputs were not saved
- degradation variability separation: mixed into the historical alpha rows
- coarse method variability separation: mixed into the historical alpha rows
- interpretation rule: treat this figure as a historical alpha diagnostic, not as a cleanly separated variance decomposition
