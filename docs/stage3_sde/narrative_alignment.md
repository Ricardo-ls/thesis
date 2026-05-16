# Stage 3 SDE Narrative Alignment

## Scientific Framing

The project is currently organized around Stage 3 unconditional SDEdit / prior-guided refinement.

## Evidence Roles

- Mainline: frozen holdout-1000 confidence-aware SDEdit and mechanism hard-threshold fusion evidence.
- Supplementary ablation: multi-t candidate-source tests, gain bottleneck diagnostics, and consistency extraction.
- Negative diagnostic: DPS, E2-Min posterior anchoring, bias sanity, and legacy/old SDEdit limitation audits.
- Future sensor interface: observation-initialized or sensor-adapter work that is not part of the current mainline.

## Current Interpretation

Raw SDEdit has limited and condition-dependent benefit. Confidence-aware stabilization is the current stable SDE component. Multi-t and DPS artifacts should not be interpreted as mainline unless a later manifest revision says so.
