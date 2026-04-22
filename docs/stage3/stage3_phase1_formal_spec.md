# Stage 3 Phase 1 Formal Specification

## Positioning

Stage 3 Phase 1 is the first operational benchmark layer after the completed Stage 2 prior study.

At this stage, the repository does not treat Stage 3 as a learning-model competition. The immediate goal is to establish a narrow, stable, and interpretable benchmark for indoor trajectory completion under missing observations.

Stage 2 remains background work and is not expanded here.

## Problem Definition

The Phase 1 task is:

- input: degraded coarse absolute trajectory in `(x, y)`
- degradation type: one contiguous missing segment
- output: completed absolute trajectory in `(x, y)`
- target: clean indoor trajectory

This phase is intentionally defined as a reconstruction benchmark rather than a generic denoising problem and not as a diffusion-model design exercise.

## Scope Boundary

The benchmark is restricted to the smallest defensible mainline:

1. build clean trajectory windows
2. generate one contiguous missing span per window
3. run simple non-learning baselines
4. evaluate reconstruction quality
5. evaluate minimal geometry feasibility

This boundary is deliberate. The purpose is to create a reliable reference layer before adding any stronger modeling assumptions.

## Current Data Interface

The data interface for Phase 1 is simple and fixed:

- clean trajectory: absolute coordinates with shape `[N, T, 2]`
- degraded trajectory: absolute coordinates with missing entries
- observation mask: binary visibility indicator over time
- occupancy map: minimal free-space / occupied-space grid

The missing pattern is limited to a single contiguous missing span.

The occupancy representation is limited to a binary map:

- `0`: free space
- `1`: occupied space

No semantic room structure is assumed at this stage.

## Baseline Layer

Phase 1 compares only simple baselines:

- linear interpolation
- Savitzky-Golay smoothing after interpolation
- constant-velocity Kalman filtering

The repository uses these methods as benchmark references, not as final solutions.

The comparison logic is:

- first establish a transparent benchmark;
- then evaluate whether later learning-based or prior-based methods produce meaningful improvement over it.

## Evaluation Layer

Phase 1 keeps only the most necessary metrics.

Reconstruction metrics:

- ADE
- FDE
- RMSE

Geometry feasibility metrics:

- off-map ratio
- wall-crossing count

The geometry evaluation is intentionally lightweight. It is meant to expose obvious feasibility failures without introducing heavy geometric machinery.

## Explicit Exclusions

The following items are outside the Phase 1 boundary:

- prior integration
- q20 or multi-prior comparison
- diffusion backbone upgrades
- cross-attention or guidance modules
- occupancy encoders or geometry-aware training
- room graph or door semantics
- multi-dataset comparison
- complex smoothing or tracking systems beyond the minimal Kalman baseline

These exclusions are not rejected permanently. They are postponed until the benchmark reference layer is stable.

## Engineering Principle

The implementation standard for Phase 1 is:

- minimal
- readable
- reproducible
- easy to debug
- easy to extend later without rewriting the benchmark logic

This means the repository should prefer explicit scripts, clear I/O files, and narrow command-line tools over early abstraction.

## Repository Mapping

The current Phase 1 scaffold is organized as follows:

- `tools/stage3/data/`: clean-window building, missing-span generation, occupancy-map building
- `tools/stage3/baselines/`: simple completion baselines
- `tools/stage3/eval/`: reconstruction and geometry metrics
- `utils/stage3/`: shared path and I/O helpers
- `docs/stage3/`: Stage 3 planning and specification notes
- `outputs/stage3/`: generated local outputs

## Interpretation Rule

Any later Stage 3 extension should be read relative to this benchmark-first baseline.

If a future method is introduced, it should answer two questions clearly:

1. does it improve reconstruction quality over the simple baselines?
2. does it reduce geometry violations under the same evaluation setup?

If those two questions are not answered, the extension should not replace the Phase 1 benchmark as the repository mainline.
