# Amendment 001: Mean-form likelihood normalization for numerical stability

Date: 2026-05-15

Amends: `docs/stage4/E2_DPS_hypothesis.md`

## 1. Reason

The original sum-form likelihood caused gradient-scale explosion in the normalized residual latent space. E2-DPS NaN diagnosis showed that zeta=0 no-guidance reverse sampling remained finite, but guided runs exploded in early reverse steps.

Observed numerical evidence:

- first non-finite stage: `likelihood_loss`
- max `grad_norm`: `5.026e17`
- max `update_to_x_ratio`: `24775.3`
- diagnosis: likelihood-guidance scale mismatch / hyperparameter instability

This indicates that the failure was not caused by the DDPM base reverse step, checkpoint corruption, or the A operator. The instability arose when the observation likelihood gradient was applied to the normalized relative-displacement latent.

## 2. Method Change

Original likelihood:

```text
L_sum = sum_t (1 / sigma_obs,t^2) ||A(x)_t - y_t||^2
```

Amended likelihood:

```text
L_mean = mean_t [ (1 / sigma_obs,t^2) ||A(x)_t - y_t||^2 ]
```

The DPS guidance step continues to use the gradient of the observation likelihood with respect to the latent reverse state. Only the frame aggregation normalization changes from sum-form to mean-form.

## 3. What Is Unchanged

- zeta remains `{0.3, 1.0, 3.0}`
- ζ remains `{0.3, 1.0, 3.0}`
- sigma_0 remains `0.05`
- σ₀ remains `0.05`
- kappa remains `1.0`
- κ remains `1.0`
- the six-condition protocol remains unchanged
- `bias_medium` remains a negative control
- `jump_medium` remains a known limitation
- no clipping is introduced
- no adaptive guidance is introduced
- no robust likelihood is introduced
- no bias estimation is introduced
- no estimated confidence is introduced
- PASS / Partial-PASS / NO-PASS criteria remain unchanged

## 4. Expected Effect

This amendment only rescales the likelihood gradient to prevent numerical overflow. It does not change the scientific hypothesis, the observation model family, the confidence definition, the baseline set, the six-condition protocol, or the PASS criteria.

The expected numerical effect is lower guidance update magnitude, especially in early reverse steps where the sum-form likelihood previously produced update-to-state ratios orders of magnitude above one.

## 5. Audit Requirement

Before rerunning full Stage 8A, a small re-audit must show finite outputs under mean-form likelihood for at least:

- conditions: `drift_medium`, `burst_medium`, `gaussian_medium`, `bias_medium`
- zeta: `0.3`, `1.0`, `3.0`
- ζ: `0.3`, `1.0`, `3.0`
- seed: `42`
- trajectories: first 3 trajectories per condition

The re-audit must record:

- finite / non-finite status
- first non-finite reverse step, if any
- mean-form likelihood
- gradient norm
- update-to-state ratio
- maximum absolute state magnitude after guidance
- ADE if finite
- acceleration RMS if finite

The re-audit is not a formal Stage 8A result. It is a numerical stability gate. Full Stage 8A may only be rerun after this amendment is recorded and the small re-audit supports finite mean-form execution.

