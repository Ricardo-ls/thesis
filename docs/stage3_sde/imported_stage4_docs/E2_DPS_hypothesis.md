# E2-DPS Hypothesis Pre-Registration

Date locked: 2026-05-15

## 1. Scope Statement

E2-DPS tests whether deterministic posterior-style absolute-space anchoring can be improved by DPS-style observation guidance while keeping the Stage 3 trajectory prior fixed.

This experiment is run under the unified six-condition Stage 4 protocol:

- gaussian_medium
- drift_medium
- burst_medium
- bias_medium
- jump_medium
- combined_medium

The official input terminology is:

Stage 3 protocol-validated per-frame conditional outputs.

Data provenance may be reported as:

- gaussian/drift/burst/bias: archived_original
- jump/combined: matched_protocol_recomputed

All formal ADE tables, mean ADE calculations, and PASS / Partial-PASS / NO-PASS decisions must use all six conditions.

E2-DPS uses a fixed relative-displacement diffusion prior. It does not train a new prior, does not retrain Stage 3 conditional residual DDPM, and does not modify Stage 3 artifacts.

## 2. Scientific Question

Can DPS-style absolute-space observation guidance repair the E1 and E2-Min failure modes that scalar residual gating and deterministic L2 anchoring did not fully solve?

The core question is not whether E2-DPS improves every degradation type. The core question is whether posterior guidance can reduce drift/burst high-confidence over-correction while preserving useful low-confidence correction under the fixed Stage 3 prior.

## 3. Hypothesis

E2-DPS should improve high-confidence no-harm under drift_medium and/or burst_medium because it directly anchors the sampled trajectory to reliable absolute observations instead of only scaling the Stage 3 residual.

Core hypothesis targets:

- drift_medium high-confidence no-harm
- burst_medium high-confidence no-harm

Non-core conditions:

- bias_medium is a structural null / negative control because the Stage 3 conditional residual signal shows very weak systematic bias-correcting direction.
- jump_medium is a known limitation under L2 likelihood because sparse outliers are structurally hostile to quadratic observation losses.

Bias and jump remain in all six-condition tables and global robustness reporting, but they are not E2-DPS core PASS criteria.

## 4. PASS Criteria

Let E2-DPS be evaluated against Formal E1 and E2-Min under the full six-condition protocol.

### Global Criteria

G1. Six-condition mean ADE improves over Formal E1.

G2. No severe regression in any of the six conditions:

ADE_E2-DPS / ADE_Formal_E1 < 1.10

G3. Low-confidence correction preservation passes where Stage 3 conditional residual provides useful low-confidence improvement.

### Core Hypothesis Criteria

H1. drift_medium high-confidence no-harm passes:

ADE_E2-DPS_high <= 1.05 * ADE_noisy_high

H2. burst_medium high-confidence no-harm passes:

ADE_E2-DPS_high <= 1.05 * ADE_noisy_high

H3. drift_medium overall ADE improves over Formal E1.

H4. burst_medium overall ADE improves over Formal E1.

### Decision Rule

PASS:

- G1, G2, and G3 all pass; and
- H1 and H2 both pass; and
- H3 and H4 both pass.

Partial-PASS:

- G1, G2, and G3 all pass; and
- H3 and H4 both pass; and
- H1 OR H2 passes.

NO-PASS:

- Any required global criterion fails; or
- drift/burst overall ADE does not improve; or
- neither H1 nor H2 passes.

Important: bias_medium and jump_medium do not define E2-DPS core PASS criteria, but they must still be reported in all six-condition global tables.

## 5. Negative Controls

### bias_medium

bias_medium is a structural null / negative control. The prior-signal sanity check found only a very weak or absent systematic bias-correcting residual direction:

- Stage 3 conditional residual barely improves ADE.
- Absolute offset improvement is extremely small.
- Frame-level and trajectory-level residual/oracle correction cosine alignment are weak.

Therefore, failure to repair bias absolute offset is not fatal to the E2-DPS core hypothesis. It should be reported as evidence that stronger absolute-position likelihood or explicit bias modeling is required.

### jump_medium

jump_medium is a known limitation for L2 likelihood. Sparse outliers can dominate quadratic observation losses and produce misleading guidance. Jump remains in the six-condition global robustness table, but it is not a core hypothesis target.

## 6. Known Limitations

- The diffusion prior is fixed and relative-displacement based; it is not retrained for E2-DPS.
- E2-DPS may still fail when the prior lacks a useful bias-correcting direction.
- L2 observation likelihood is not robust to sparse outliers, so jump_medium can regress.
- High-confidence no-harm can fail if guidance does not sufficiently protect reliable observation frames.
- Improvements in mean ADE do not alone prove hypothesis success; drift/burst high-confidence behavior must be inspected.

## 7. Anticipated Critiques and Responses

Critique: If bias fails, E2-DPS failed.

Response: bias_medium is pre-registered as a structural null / negative control. Bias is included in six-condition global reporting, but it is not a core E2-DPS PASS criterion.

Critique: If jump regresses, E2-DPS failed.

Response: jump_medium is a known limitation of L2 likelihood under sparse outliers. It remains in global robustness reporting, but it is not a core hypothesis criterion.

Critique: Mean ADE improvement is not enough.

Response: Correct. The decision rule requires global ADE robustness and drift/burst high-confidence hypothesis checks.

Critique: Hyperparameters could be expanded after seeing failures.

Response: Hyperparameters are locked before E2-DPS execution. If Stage 8A shows no signal, the protocol requires pausing and auditing first, then proceeding to Stage 8B as a full negative-result evaluation if no implementation issue is found.

## 8. Fixed Hyperparameters

Observation likelihood scale:

- sigma_0 = 0.05

Guidance strength scale:

- kappa = 1.0

DPS guidance sweep:

- zeta in {0.3, 1.0, 3.0}

Locked values:

- ζ = {0.3, 1.0, 3.0}
- σ₀ = 0.05
- κ = 1.0

No additional zeta, sigma_0, or kappa values may be added after observing Stage 8A results.

## 9. Two-Stage Execution Protocol

### Stage 8A: Small Protocol Check

Stage 8A is a small implementation and signal check. It must:

- use the fixed relative-displacement diffusion prior;
- use the locked hyperparameters above;
- evaluate the pre-specified degradation conditions needed to verify implementation and initial signal;
- verify shapes, ADE computation, confidence bins, and high-confidence no-harm metrics;
- avoid expanding the hyperparameter grid.

Stage 8A no-signal fallback:

pause and audit first; if no implementation issue is found, proceed to Stage 8B as full negative-result evaluation without expanding hyperparameters.

### Stage 8B: Full Six-Condition Formal Evaluation

Stage 8B is the formal E2-DPS evaluation. It must:

- use all six degradation conditions;
- report all six-condition ADE tables;
- report mean ADE over all six conditions;
- report drift/burst high-confidence no-harm;
- report low-confidence correction preservation;
- report bias and jump as negative-control / known-limitation diagnostics;
- apply the PASS / Partial-PASS / NO-PASS decision rule exactly as locked above.

## 10. Carry-Over Diagnostics

E2-DPS must carry forward the following diagnostics from E1 and E2-Min:

- overall ADE by condition;
- RMSE by condition;
- smoothness / acceleration RMS;
- confidence-bin ADE for high, mid, and low confidence frames;
- high-confidence no-harm ratio;
- low-confidence correction preservation;
- noisy_reversion_gap;
- residual or motion usage where applicable;
- bias absolute offset error;
- representative figures for drift, burst, bias, jump, and combined.

Carry-over diagnostics are explanatory and must not be used to retroactively tune E2-DPS.

## 11. Date Locked

Date locked = 2026-05-15

This document must be treated as the pre-registration record for E2-DPS. Any later changes must be recorded as post-registration amendments and must not be presented as pre-registered decisions.

## 12. What Is NOT in E2-DPS

E2-DPS does not include:

- retraining the diffusion prior;
- retraining the Stage 3 conditional residual DDPM;
- using SDEdit as the starting point;
- using unconditional prior samples as the baseline;
- changing Stage 3 saved outputs;
- trajectory-level method tuning;
- expanding hyperparameters after seeing failures;
- treating bias_medium or jump_medium as core PASS criteria;
- replacing the unified six-condition protocol with the old artifact-limited protocol.

