# Amendment 002: Canonical norm-based DPS guidance

Date: 2026-05-15

Amends:

- `docs/stage4/E2_DPS_hypothesis.md`
- `docs/stage4/E2_DPS_hypothesis_amendment_001.md`

## 1. Motivation

The original squared likelihood guidance and Amendment 001 mean-squared likelihood remained numerically unstable in the normalized relative-displacement latent space.

Observed diagnostics:

- zeta=0 no-guidance reverse sampling stayed finite.
- Guided runs exploded in early reverse steps.
- Original sum-form guidance reached max `update_to_x_ratio = 24775.3`.
- Amendment 001 mean-form guidance still failed:
  - zeta=0.3: 11/12 finite
  - zeta=1.0: 0/12 finite
  - zeta=3.0: 0/12 finite
  - max `update_to_x_ratio = 1497.91`

This shows that the issue is likelihood-guidance scale mismatch, not the DDPM base reverse step, checkpoint, A operator, or confidence cache.

## 2. Canonical DPS Correction

DPS canonical implementation uses the gradient of a measurement residual L2 norm, rather than a squared residual sum. Amendment 002 corrects the E2-DPS guidance objective to this norm-based form.

This is an implementation correction to align the experiment with canonical DPS guidance while retaining the pre-registered confidence-weighted observation model.

## 3. Method Change

The heteroscedastic confidence weighting is retained:

```text
sigma_obs,t^2 = sigma_0^2 * (1 + kappa * (1 - c_t))
```

Let:

```text
sigma_obs,t = sqrt(sigma_obs,t^2)
weighted_residual_t = (A(x_hat_0)_t - y_t) / sigma_obs,t
```

Amended guidance objective:

```text
L_norm = ||weighted_residual||_2
```

Gradient:

```text
grad = nabla L_norm
```

Update:

```text
x_{t-1} = ddpm_step(x_t) - zeta * grad
```

## 4. Heteroscedastic Extension Statement

DPS canonical form is originally defined for isotropic measurement noise. This amendment adapts the norm form to heteroscedastic per-frame `sigma_obs` by placing the scaling inside the norm.

This directly encodes the high-confidence anchoring hypothesis into the gradient direction:

- high-confidence frames have smaller `sigma_obs` and therefore stronger observation anchoring;
- low-confidence frames have larger `sigma_obs` and therefore weaker observation anchoring.

## 5. What Is Unchanged

- zeta remains `{0.3, 1.0, 3.0}`
- ζ remains `{0.3, 1.0, 3.0}`
- sigma_0 remains `0.05`
- σ₀ remains `0.05`
- kappa remains `1.0`
- κ remains `1.0`
- oracle confidence remains unchanged
- six-condition protocol remains unchanged
- `bias_medium` remains a negative control
- `jump_medium` remains a known limitation
- output metrics remain unchanged
- PASS / Partial-PASS / NO-PASS criteria remain unchanged
- no trust-region is introduced
- no gradient clipping is introduced
- no timestep guidance schedule is introduced
- no anchor modification is introduced
- no robust likelihood is introduced
- no Huber / L1 / Mahalanobis likelihood is introduced
- no bias estimation is introduced
- no estimated confidence is introduced

## 6. Scope

This is an implementation correction to align E2-DPS with canonical norm-based DPS guidance under heteroscedastic confidence weighting.

It is not a new hypothesis and not a new success criterion. The re-audit under this amendment is a numerical stability check only. It must not be reported as a formal E2-DPS PASS / Partial-PASS / NO-PASS result.

