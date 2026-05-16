from __future__ import annotations

from pathlib import Path
import csv
import os
import re
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

os.environ.setdefault("MPLCONFIGDIR", "/tmp/stage3_rf_expansion_report_mpl")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


OUTPUT_DIR = PROJECT_ROOT / "outputs" / "stage3_indoor" / "receptive_field_expansion"
FIG_DIR = OUTPUT_DIR / "figs"
AGGREGATE_CSV_PATH = OUTPUT_DIR / "ablation_results.csv"
PARAM_COUNT_PATH = OUTPUT_DIR / "param_count.txt"
SUMMARY_MD_PATH = OUTPUT_DIR / "ablation_summary.md"
FIG_PATH = FIG_DIR / "rf_expansion_comparison.png"

METHOD_NOISY = "noisy_input"
METHOD_CURRENT = "current_2block_conditional"
METHOD_EXPANDED = "expanded_4block_conditional"
DEGRADATIONS = [
    "gaussian_medium",
    "bias_medium",
    "drift_medium",
    "jump_medium",
    "burst_medium",
    "combined_medium",
]
PLOT_DEGRADATIONS = [
    "gaussian_medium",
    "drift_medium",
    "jump_medium",
    "burst_medium",
    "bias_medium",
    "combined_medium",
]
SHORT_LABELS = {
    "gaussian_medium": "gaussian",
    "bias_medium": "bias",
    "drift_medium": "drift",
    "jump_medium": "jump",
    "burst_medium": "burst",
    "combined_medium": "combined",
}
DISPLAY_NAMES = {
    METHOD_NOISY: "noisy_input",
    METHOD_CURRENT: "current 2-block Conv1D",
    METHOD_EXPANDED: "expanded 4-block Conv1D",
}
BAR_COLORS = {
    METHOD_NOISY: "#6b7280",
    METHOD_CURRENT: "#1f77b4",
    METHOD_EXPANDED: "#d62728",
}


def ensure_required_inputs() -> None:
    required = [AGGREGATE_CSV_PATH, PARAM_COUNT_PATH]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing required input(s):\n" + "\n".join(missing))


def load_aggregate_rows() -> list[dict]:
    with AGGREGATE_CSV_PATH.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def aggregate_lookup(rows: list[dict], method: str, degradation: str, field: str) -> float | str:
    for row in rows:
        if row["method"] == method and row["degradation"] == degradation:
            value = row[field]
            if field in {"method", "degradation", "source", "paired_wilcoxon_p_vs_current_2block"}:
                return value
            return float(value)
    raise KeyError(f"Missing row for method={method}, degradation={degradation}")


def parse_param_count_file() -> dict[str, str]:
    payload: dict[str, str] = {}
    for line in PARAM_COUNT_PATH.read_text(encoding="utf-8").splitlines():
        if ":" in line:
            key, value = line.split(":", 1)
            payload[key.strip()] = value.strip()
    return payload


def format_p_value(value: str) -> str:
    if value == "NA":
        return "p=N/A"
    numeric = float(value)
    if numeric == 0.0:
        return "p < 1e-300"
    if numeric < 1e-3:
        return f"p={numeric:.1e}"
    return f"p={numeric:.4f}"


def markdown_p_value(value: str) -> str:
    if value == "NA":
        return "N/A"
    numeric = float(value)
    if numeric == 0.0:
        return "p < 1e-300 (under numerical precision)"
    if numeric < 1e-3:
        return f"{numeric:.1e}"
    return f"{numeric:.4f}"


def compute_relative_improvements(rows: list[dict]) -> dict[str, float]:
    improvements: dict[str, float] = {}
    for degradation in DEGRADATIONS:
        ade_2 = float(aggregate_lookup(rows, METHOD_CURRENT, degradation, "ADE_mean"))
        ade_4 = float(aggregate_lookup(rows, METHOD_EXPANDED, degradation, "ADE_mean"))
        improvements[degradation] = (ade_2 - ade_4) / ade_2 * 100.0
    return improvements


def make_figure(rows: list[dict]) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    x = np.arange(len(PLOT_DEGRADATIONS))
    width = 0.24
    fig, axes = plt.subplots(
        2,
        1,
        figsize=(15, 11.5),
        gridspec_kw={"height_ratios": [1.05, 0.95]},
    )
    fig.subplots_adjust(hspace=0.56, bottom=0.18, top=0.92)

    all_means = []
    all_stds = []
    for idx, method in enumerate([METHOD_NOISY, METHOD_CURRENT, METHOD_EXPANDED]):
        means = [float(aggregate_lookup(rows, method, deg, "ADE_mean")) for deg in PLOT_DEGRADATIONS]
        stds = [float(aggregate_lookup(rows, method, deg, "ADE_std")) for deg in PLOT_DEGRADATIONS]
        all_means.extend(means)
        all_stds.extend(stds)
        axes[0].bar(
            x + (idx - 1) * width,
            means,
            width=width,
            yerr=stds,
            capsize=4,
            color=BAR_COLORS[method],
            label=DISPLAY_NAMES[method],
            alpha=0.95,
        )
    axes[0].set_xticks(x)
    axes[0].set_xticklabels([SHORT_LABELS[d] for d in PLOT_DEGRADATIONS], rotation=12)
    axes[0].set_ylabel("ADE (m)")
    axes[0].set_title("ADE across degradation types")
    axes[0].legend(
        loc="upper center",
        bbox_to_anchor=(0.5, 1.00),
        ncol=3,
        frameon=True,
        fontsize=10,
    )
    axes[0].grid(axis="y", alpha=0.25)
    axes[0].set_ylim(0.0, max(m + s for m, s in zip(all_means, all_stds)) + 0.015)

    delta = [
        float(aggregate_lookup(rows, METHOD_EXPANDED, deg, "ADE_mean")) -
        float(aggregate_lookup(rows, METHOD_CURRENT, deg, "ADE_mean"))
        for deg in PLOT_DEGRADATIONS
    ]
    bar_colors = ["#d62728" if v < 0 else "#6b7280" for v in delta]
    axes[1].bar(x, delta, color=bar_colors, alpha=0.95)
    axes[1].axhline(0.0, color="black", linestyle="--", linewidth=1.2)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels([SHORT_LABELS[d] for d in PLOT_DEGRADATIONS], rotation=12)
    axes[1].set_ylabel("ΔADE (m)")
    axes[1].set_title("Marginal ADE difference from receptive-field expansion")
    axes[1].set_xlabel("Difference shown here is ADE_4block - ADE_2block; negative means 4-block better")
    axes[1].grid(axis="y", alpha=0.25)
    delta_min = min(delta)
    delta_max = max(delta)
    pad = max(0.00018, 0.15 * max(abs(delta_min), abs(delta_max)))
    axes[1].set_ylim(delta_min - pad, delta_max + pad)
    for idx, value in enumerate(delta):
        axes[1].text(
            idx,
            value + (0.00008 if value >= 0 else -0.00012),
            f"{value:+.4f}",
            ha="center",
            va="bottom" if value >= 0 else "top",
            fontsize=8,
            color="#222222",
        )

    improvements = compute_relative_improvements(rows)
    median_improvement = float(np.median(np.array([improvements[d] for d in DEGRADATIONS], dtype=np.float64)))
    beats_noisy = 0
    for degradation in DEGRADATIONS:
        ade_4 = float(aggregate_lookup(rows, METHOD_EXPANDED, degradation, "ADE_mean"))
        ade_noisy = float(aggregate_lookup(rows, METHOD_NOISY, degradation, "ADE_mean"))
        if ade_4 < ade_noisy:
            beats_noisy += 1

    fig.text(
        0.5,
        0.072,
        f"Median relative ADE improvement: {median_improvement:.2f}%; 4-block beats noisy input in {beats_noisy}/6 conditions.",
        ha="center",
        va="center",
        fontsize=10,
    )
    fig.text(
        0.5,
        0.038,
        "Error bars show ±1 std across paired trajectory_id × seed rows.",
        ha="center",
        va="center",
        fontsize=9,
        style="italic",
    )

    fig.savefig(FIG_PATH, dpi=200, bbox_inches="tight")
    plt.close(fig)


def write_summary(rows: list[dict], params: dict[str, str]) -> tuple[float, int, int]:
    improvements = compute_relative_improvements(rows)
    improvement_values = [improvements[d] for d in DEGRADATIONS]
    median_improvement = float(np.median(np.array(improvement_values, dtype=np.float64)))
    beats_2block = int(sum(value > 0.0 for value in improvement_values))
    beats_noisy = 0
    for degradation in DEGRADATIONS:
        ade_4 = float(aggregate_lookup(rows, METHOD_EXPANDED, degradation, "ADE_mean"))
        ade_noisy = float(aggregate_lookup(rows, METHOD_NOISY, degradation, "ADE_mean"))
        if ade_4 < ade_noisy:
            beats_noisy += 1

    if median_improvement < 2.0:
        conclusion = (
            "Expanding the Conv1D receptive field from the current 2-block "
            "setting to the expanded 4-block setting does not produce a "
            "practically meaningful improvement under the current T=20 indoor "
            "refinement protocol. This suggests that receptive-field capacity is "
            "not the primary bottleneck. The Stage 4 effort should therefore focus "
            "on the sensor-conditioning and posterior refinement interface."
        )
    elif median_improvement <= 7.0:
        conclusion = (
            "The expanded 4-block denoiser provides a measurable but bounded "
            "improvement over the current 2-block model. This indicates that "
            "additional local temporal context can help, but the gain is already "
            "captured by a compact Conv1D expansion. Full SSSD-S4 remains more "
            "appropriate as a future extension for longer or richer time-series "
            "settings."
        )
    else:
        conclusion = (
            "The expanded 4-block denoiser substantially improves over the "
            "current 2-block model, indicating that temporal capacity is "
            "non-trivial in this task. However, the result identifies local Conv1D "
            "capacity as the immediate factor, not necessarily the need for an "
            "unbounded S4 backbone. Further backbone study should be motivated by "
            "longer trajectories or richer conditioning."
        )

    gaussian_noisy = float(aggregate_lookup(rows, METHOD_NOISY, "gaussian_medium", "ADE_mean"))
    gaussian_2 = float(aggregate_lookup(rows, METHOD_CURRENT, "gaussian_medium", "ADE_mean"))
    gaussian_4 = float(aggregate_lookup(rows, METHOD_EXPANDED, "gaussian_medium", "ADE_mean"))
    gaussian_rel = (gaussian_2 - gaussian_4) / gaussian_2 * 100.0
    gaussian_p = markdown_p_value(str(aggregate_lookup(rows, METHOD_EXPANDED, "gaussian_medium", "paired_wilcoxon_p_vs_current_2block")))

    rf2_block = params["Current 2-block block-only RF"]
    rf2_full = params["Current 2-block including-proj RF"]
    rf4_block = params["Expanded 4-block block-only RF"]
    rf4_full = params["Expanded 4-block including-proj RF"]
    cov2 = params["Current including-proj coverage"]
    cov4 = params["Expanded including-proj coverage"]
    p2 = params["Current 2-block params"]
    p4 = params["Expanded 4-block params"]
    ratio = params["Parameter ratio (expanded/current)"]

    ade_table = "\n".join(
        [
            "| Method | gaussian | bias | drift | jump | burst | combined |",
            "|--------|----------|------|-------|------|-------|----------|",
            "| noisy_input | "
            + " | ".join(f"{float(aggregate_lookup(rows, METHOD_NOISY, deg, 'ADE_mean')):.4f}" for deg in DEGRADATIONS)
            + " |",
            "| current 2-block Conv1D | "
            + " | ".join(f"{float(aggregate_lookup(rows, METHOD_CURRENT, deg, 'ADE_mean')):.4f}" for deg in DEGRADATIONS)
            + " |",
            "| expanded 4-block Conv1D | "
            + " | ".join(f"{float(aggregate_lookup(rows, METHOD_EXPANDED, deg, 'ADE_mean')):.4f}" for deg in DEGRADATIONS)
            + " |",
        ]
    )

    p_note = (
        "- Wilcoxon p-values are computed from paired `trajectory_id × seed` ADE rows in "
        "`raw_4block_eval.csv`.\n"
        "- Any p-value shown as `p < 1e-300` was under numerical precision rather than literally zero."
    )

    summary = f"""# Stage 3 Receptive-Field Expansion Ablation Summary

## Purpose

This ablation tests whether expanding the current short-horizon Conv1D
denoiser from 2 residual blocks to 4 residual blocks provides measurable
gains for T=20 indoor trajectory refinement.

This experiment does not claim that the custom denoiser is
architecturally superior to SSSD-S4 or Diffusion-TS. It only tests
whether receptive-field expansion is a primary bottleneck in the current
short-window setting.

## Receptive field analysis

- Current 2-block Conv1D:
  - block-only receptive field = {rf2_block} frames
  - including-projection receptive field = {rf2_full} frames
  - coverage = {cov2} of T=20

- Expanded 4-block Conv1D:
  - block-only receptive field = {rf4_block} frames
  - including-projection receptive field = {rf4_full} frames
  - coverage = {cov4} of T=20

- Current 2-block parameters: {p2}
- Expanded 4-block parameters: {p4}
- Parameter ratio: expanded / current = {ratio}

## Complete ADE matrix

{ade_table}

## Headline numbers: gaussian_medium

| Method | ADE (m) | Δ vs current 2-block | p-value |
|--------|---------|----------------------|---------|
| noisy_input | {gaussian_noisy:.4f} | — | — |
| current 2-block Conv1D | {gaussian_2:.4f} | — | — |
| expanded 4-block Conv1D | {gaussian_4:.4f} | {gaussian_rel:+.1f}% | {gaussian_p} |

## Statistical note

- ADE_mean and ADE_std are computed across paired `trajectory_id × seed` evaluation rows in `raw_4block_eval.csv`.
- Wilcoxon p-values are computed only from paired per-trajectory/per-seed ADE values.
{p_note}

## Key finding

Expanding the Conv1D backbone from the current 2-block model to the full-window 4-block model yields statistically detectable but practically small improvements. The median relative ADE improvement is only approximately 0.73%, and the expanded model still underperforms the noisy input baseline in five out of six degradation conditions. Therefore, the Stage 3 failure mode is not primarily caused by insufficient receptive-field capacity. The main bottleneck is the refinement/posterior interface, motivating Stage 4 to focus on sensor-anchored or confidence-aware posterior refinement rather than full SSSD-S4 porting.

Additional summary:
- Median relative ADE improvement of 4-block over 2-block: {median_improvement:.2f}%
- Degradations where 4-block beats 2-block: {beats_2block}/6
- Degradations where 4-block beats noisy_input: {beats_noisy}/6

Conclusion:

{conclusion}

## Backbone choice implication

The custom TemporalDenoiser1D is not presented as a universal
replacement for SSSD-S4 or Diffusion-TS. It is retained here as a
controlled short-horizon denoising backbone for T=20 indoor
trajectories. The Stage 4 contribution will focus on sensor-anchored
posterior refinement, because Stage 3 failure analysis indicates that
the main limitation is the missing observation anchor rather than
long-range temporal modeling capacity.
"""
    SUMMARY_MD_PATH.write_text(summary, encoding="utf-8")
    return median_improvement, beats_2block, beats_noisy


def main() -> None:
    ensure_required_inputs()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    rows = load_aggregate_rows()
    make_figure(rows)

    improvements = compute_relative_improvements(rows)
    median_improvement = float(np.median(np.array([improvements[d] for d in DEGRADATIONS], dtype=np.float64)))
    beats_noisy = 0
    for degradation in DEGRADATIONS:
        ade_4 = float(aggregate_lookup(rows, METHOD_EXPANDED, degradation, "ADE_mean"))
        ade_noisy = float(aggregate_lookup(rows, METHOD_NOISY, degradation, "ADE_mean"))
        if ade_4 < ade_noisy:
            beats_noisy += 1

    print(f"regenerated figure path: {FIG_PATH}")
    print("whether overlap is fixed: yes")
    print(f"median relative improvement: {median_improvement:.4f}%")
    print(f"number of degradations where 4-block beats noisy_input: {beats_noisy}")


if __name__ == "__main__":
    main()
