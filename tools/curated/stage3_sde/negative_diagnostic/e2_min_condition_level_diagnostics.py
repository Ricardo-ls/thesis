from __future__ import annotations

from pathlib import Path
import math
import os

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = PROJECT_ROOT / "outputs" / "stage4" / "e2_min_absolute_posterior_anchoring"

FULL_METRICS_PATH = OUT_DIR / "e2_min_full_metrics.csv"
BIN_METRICS_PATH = OUT_DIR / "e2_min_confidence_bin_metrics.csv"
PER_TRAJ_PATH = OUT_DIR / "e2_min_per_trajectory_metrics.csv"
PASS_FAIL_PATH = OUT_DIR / "e2_min_pass_fail_summary.csv"
DIAGNOSTICS_PATH = OUT_DIR / "e2_min_diagnostics.csv"
SWEEP_PATH = OUT_DIR / "e2_min_parameter_sweep.csv"
SUMMARY_INPUT_PATH = OUT_DIR / "e2_min_summary.md"

CONDITION_DIAGNOSTICS_PATH = OUT_DIR / "e2_min_condition_level_diagnostics.csv"
LAYERED_SUMMARY_PATH = OUT_DIR / "e2_min_layered_pass_fail_summary.csv"
CONDITION_DIAGNOSTICS_MD_PATH = OUT_DIR / "e2_min_condition_level_diagnostics.md"

DEGRADATION_ORDER = [
    "gaussian_medium",
    "drift_medium",
    "burst_medium",
    "bias_medium",
    "jump_medium",
    "combined_medium",
]
E2_METHODS = ["V1_abs_anchor_smooth", "V2_uniform_cond_motion", "V3_conf_mod_cond_motion"]
BASELINE_METHODS = ["formal_e1_linear_gate", *E2_METHODS]
SELECTED_METHOD = "V2_uniform_cond_motion"


def require_inputs() -> None:
    missing = [
        path
        for path in [
            FULL_METRICS_PATH,
            BIN_METRICS_PATH,
            PER_TRAJ_PATH,
            PASS_FAIL_PATH,
            DIAGNOSTICS_PATH,
            SWEEP_PATH,
            SUMMARY_INPUT_PATH,
        ]
        if not path.is_file()
    ]
    if missing:
        raise FileNotFoundError("Missing E2-Min input files:\n" + "\n".join(str(path) for path in missing))


def fmt(value: object, digits: int = 6) -> str:
    if isinstance(value, (float, int)) and not isinstance(value, bool):
        if math.isfinite(float(value)):
            return f"{float(value):.{digits}f}"
        return "NaN"
    return str(value)


def markdown_table(df) -> str:
    if df.empty:
        return "_No rows._"
    out = df.copy()
    for col in out.columns:
        out[col] = out[col].map(fmt)
    lines = [
        "| " + " | ".join(map(str, out.columns)) + " |",
        "| " + " | ".join(["---"] * len(out.columns)) + " |",
    ]
    for row in out.values.tolist():
        lines.append("| " + " | ".join(map(str, row)) + " |")
    return "\n".join(lines)


def ordered(df):
    return df.assign(_order=df["degradation"].map({d: i for i, d in enumerate(DEGRADATION_ORDER)})).sort_values("_order").drop(columns="_order")


def get_value(df, degradation: str, method: str, column: str) -> float:
    rows = df[(df["degradation"] == degradation) & (df["method"] == method)]
    if rows.empty:
        raise RuntimeError(f"Missing row for {degradation}/{method}/{column}")
    return float(rows.iloc[0][column])


def load_tables():
    import pandas as pd

    require_inputs()
    full = pd.read_csv(FULL_METRICS_PATH)
    bins = pd.read_csv(BIN_METRICS_PATH)
    per_traj = pd.read_csv(PER_TRAJ_PATH)
    pass_fail = pd.read_csv(PASS_FAIL_PATH)
    diagnostics = pd.read_csv(DIAGNOSTICS_PATH)
    sweep = pd.read_csv(SWEEP_PATH)
    return full, bins, per_traj, pass_fail, diagnostics, sweep


def compute_condition_table(full, bins, diagnostics):
    rows: list[dict] = []
    for degradation in DEGRADATION_ORDER:
        formal = get_value(full, degradation, "formal_e1_linear_gate", "ADE_mean")
        row = {
            "degradation": degradation,
            "Formal_E1_ADE": formal,
            "V1_ADE": get_value(full, degradation, "V1_abs_anchor_smooth", "ADE_mean"),
            "V2_selected_ADE": get_value(full, degradation, "V2_uniform_cond_motion", "ADE_mean"),
            "V3_ADE": get_value(full, degradation, "V3_conf_mod_cond_motion", "ADE_mean"),
        }
        for label, method in [
            ("V1", "V1_abs_anchor_smooth"),
            ("V2", "V2_uniform_cond_motion"),
            ("V3", "V3_conf_mod_cond_motion"),
        ]:
            ade = row[f"{label}_ADE"] if label != "V2" else row["V2_selected_ADE"]
            row[f"{label}_rel_change_vs_E1"] = ade / formal - 1.0
            row[f"{label}_improves_vs_E1"] = bool(ade <= formal)
            row[f"{label}_ADE_ratio_vs_E1"] = ade / formal
            row[f"{label}_smooth_acc_rms"] = get_value(full, degradation, method, "smooth_acc_rms_mean")
            row[f"{label}_motion_usage_ratio"] = get_value(full, degradation, method, "motion_usage_ratio")
            row[f"{label}_noisy_reversion_gap"] = get_value(full, degradation, method, "noisy_reversion_gap")
            diag = diagnostics[(diagnostics["degradation"] == degradation) & (diagnostics["method"] == method)].iloc[0]
            row[f"{label}_high_ADE"] = float(diag["ADE_e2_high"])
            row[f"{label}_low_ADE"] = float(diag["ADE_e2_low"])
            row[f"{label}_C2_pass_1p05"] = bool(diag["high_conf_no_harm_pass_1p05"])
            row[f"{label}_C3_low_pass"] = bool(diag["low_conf_preservation_pass"])
            if degradation == "bias_medium":
                row[f"{label}_bias_offset_error"] = get_value(full, degradation, method, "bias_offset_error")

        noisy_high = bins[(bins["degradation"] == degradation) & (bins["method"] == "noisy_input") & (bins["confidence_bin"] == "high")].iloc[0]
        formal_high = bins[(bins["degradation"] == degradation) & (bins["method"] == "formal_e1_linear_gate") & (bins["confidence_bin"] == "high")].iloc[0]
        row["noisy_high_ADE"] = float(noisy_high["ADE_mean"])
        row["Formal_E1_high_ADE"] = float(formal_high["ADE_mean"])
        row["C2_threshold_1p05_noisy_high"] = 1.05 * float(noisy_high["ADE_mean"])
        row["selected_V2_C2_margin"] = row["C2_threshold_1p05_noisy_high"] - row["V2_high_ADE"]
        rows.append(row)
    import pandas as pd

    return ordered(pd.DataFrame(rows))


def compute_layered_summary(full, diagnostics, pass_fail, condition_df):
    import pandas as pd

    means = {
        method: float(full[full["method"] == method].set_index("degradation").loc[DEGRADATION_ORDER, "ADE_mean"].mean())
        for method in BASELINE_METHODS
    }
    selected = condition_df.copy()
    g1_count = int(selected["V2_improves_vs_E1"].sum())
    g1 = g1_count >= 4
    g2_ratios = selected["V2_ADE_ratio_vs_E1"]
    g2 = bool((g2_ratios < 1.10).all())
    g3 = means[SELECTED_METHOD] < means["formal_e1_linear_gate"]

    sel_diag = diagnostics[diagnostics["method"] == SELECTED_METHOD].set_index("degradation")
    drift = sel_diag.loc["drift_medium"]
    burst = sel_diag.loc["burst_medium"]
    bias = selected[selected["degradation"] == "bias_medium"].iloc[0]
    e1_bias_offset = float(pass_fail["E2_C5_bias_offset_error_E1"].iloc[0])
    e2_bias_offset = float(pass_fail["E2_C5_bias_offset_error_E2"].iloc[0])
    h = {
        "H1_drift_high_conf_no_harm": bool(drift["high_conf_no_harm_pass_1p05"]),
        "H2_burst_high_conf_no_harm": bool(burst["high_conf_no_harm_pass_1p05"]),
        "H3_drift_ADE_improves": bool(drift["ADE_e2"] <= drift["ADE_e1"]),
        "H4_burst_ADE_improves": bool(burst["ADE_e2"] <= burst["ADE_e1"]),
        "H5_bias_ADE_or_offset_improves": bool(
            (bias["V2_selected_ADE"] <= 0.95 * bias["Formal_E1_ADE"]) or (e2_bias_offset < e1_bias_offset - 1e-6)
        ),
        "H6_low_conf_preservation": bool(sel_diag["low_conf_preservation_pass"].all()),
    }
    layer1_all = bool(g1 and g2 and g3)
    layer2_count = int(sum(h.values()))
    layer2_all = layer2_count == 6
    if layer1_all and layer2_all:
        overall = "PASS"
    elif layer1_all and layer2_count >= 4:
        overall = "Partial-PASS"
    else:
        overall = "NO-PASS"

    rows = [
        {
            "layer": "Layer 1",
            "criterion": "G1",
            "description": "ADE improves over Formal E1 in at least 4/6 degradation conditions",
            "pass": g1,
            "value": f"{g1_count}/6",
        },
        {
            "layer": "Layer 1",
            "criterion": "G2",
            "description": "No severe regression: ADE_E2 / ADE_E1 < 1.10 for all six",
            "pass": g2,
            "value": f"max ratio {g2_ratios.max():.6f}",
        },
        {
            "layer": "Layer 1",
            "criterion": "G3",
            "description": "Six-condition mean ADE improves over Formal E1",
            "pass": g3,
            "value": f"{means[SELECTED_METHOD]:.6f} vs {means['formal_e1_linear_gate']:.6f}",
        },
        {
            "layer": "Layer 2",
            "criterion": "H1",
            "description": "drift_medium high-confidence no-harm passes",
            "pass": h["H1_drift_high_conf_no_harm"],
            "value": f"V2 high {float(drift['ADE_e2_high']):.6f}; noisy high {float(drift['ADE_noisy_high']):.6f}",
        },
        {
            "layer": "Layer 2",
            "criterion": "H2",
            "description": "burst_medium high-confidence no-harm passes",
            "pass": h["H2_burst_high_conf_no_harm"],
            "value": f"V2 high {float(burst['ADE_e2_high']):.6f}; noisy high {float(burst['ADE_noisy_high']):.6f}",
        },
        {
            "layer": "Layer 2",
            "criterion": "H3",
            "description": "drift_medium overall ADE improves over Formal E1",
            "pass": h["H3_drift_ADE_improves"],
            "value": f"{float(drift['ADE_e2']):.6f} vs {float(drift['ADE_e1']):.6f}",
        },
        {
            "layer": "Layer 2",
            "criterion": "H4",
            "description": "burst_medium overall ADE improves over Formal E1",
            "pass": h["H4_burst_ADE_improves"],
            "value": f"{float(burst['ADE_e2']):.6f} vs {float(burst['ADE_e1']):.6f}",
        },
        {
            "layer": "Layer 2",
            "criterion": "H5",
            "description": "bias_medium ADE improves by at least 5%, or bias offset error decreases clearly",
            "pass": h["H5_bias_ADE_or_offset_improves"],
            "value": f"ADE {bias['V2_selected_ADE']:.6f} vs {bias['Formal_E1_ADE']:.6f}; offset {e2_bias_offset:.6f} vs {e1_bias_offset:.6f}",
        },
        {
            "layer": "Layer 2",
            "criterion": "H6",
            "description": "Low-confidence correction preservation passes",
            "pass": h["H6_low_conf_preservation"],
            "value": "all six" if h["H6_low_conf_preservation"] else "one or more failures",
        },
        {
            "layer": "Overall",
            "criterion": "Decision",
            "description": "PASS if Layer 1 and Layer 2 all pass; Partial-PASS if Layer 1 all pass and at least 4/6 Layer 2 pass",
            "pass": overall in {"PASS", "Partial-PASS"},
            "value": f"{overall}; Layer1 all={layer1_all}; Layer2 {layer2_count}/6",
        },
    ]
    summary = pd.DataFrame(rows)
    return summary, means, {"G1": g1, "G2": g2, "G3": g3, **h, "Layer1_all": layer1_all, "Layer2_count": layer2_count, "overall": overall}


def top_sweep_table(sweep):
    return (
        sweep.groupby(["variant", "alpha0", "beta", "sigma0"], as_index=False)
        .agg(
            six_ADE_improved_count=("improves_vs_e1", "sum"),
            mean_ADE_gain_vs_E1=("ADE_gain_vs_e1", "mean"),
            mean_ADE_E2=("ADE_e2", "mean"),
        )
        .sort_values(["six_ADE_improved_count", "mean_ADE_gain_vs_E1", "mean_ADE_E2"], ascending=[False, False, True])
        .head(10)
    )


def write_markdown(full, bins, diagnostics, sweep, condition_df, layered_df, means, flags) -> None:
    selected_diag = diagnostics[diagnostics["method"] == SELECTED_METHOD].copy()
    ade_table = condition_df[
        [
            "degradation",
            "Formal_E1_ADE",
            "V1_ADE",
            "V1_rel_change_vs_E1",
            "V2_selected_ADE",
            "V2_rel_change_vs_E1",
            "V3_ADE",
            "V3_rel_change_vs_E1",
        ]
    ]
    mean_df = __import__("pandas").DataFrame(
        [
            {"method": "Formal E1", "six_condition_mean_ADE": means["formal_e1_linear_gate"]},
            {"method": "E2-Min V1", "six_condition_mean_ADE": means["V1_abs_anchor_smooth"]},
            {"method": "E2-Min V2 selected", "six_condition_mean_ADE": means["V2_uniform_cond_motion"]},
            {"method": "E2-Min V3", "six_condition_mean_ADE": means["V3_conf_mod_cond_motion"]},
        ]
    )

    c2_table = condition_df[
        [
            "degradation",
            "noisy_high_ADE",
            "Formal_E1_high_ADE",
            "V1_high_ADE",
            "V1_C2_pass_1p05",
            "V2_high_ADE",
            "V2_C2_pass_1p05",
            "V3_high_ADE",
            "V3_C2_pass_1p05",
            "C2_threshold_1p05_noisy_high",
        ]
    ]
    burst_table = condition_df[condition_df["degradation"] == "burst_medium"][
        [
            "degradation",
            "V1_ADE",
            "V1_high_ADE",
            "V1_low_ADE",
            "V1_smooth_acc_rms",
            "V1_motion_usage_ratio",
            "V1_noisy_reversion_gap",
            "V2_selected_ADE",
            "V2_high_ADE",
            "V2_low_ADE",
            "V2_smooth_acc_rms",
            "V2_motion_usage_ratio",
            "V2_noisy_reversion_gap",
            "V3_ADE",
            "V3_high_ADE",
            "V3_low_ADE",
            "V3_smooth_acc_rms",
            "V3_motion_usage_ratio",
            "V3_noisy_reversion_gap",
        ]
    ]
    bias_jump_table = condition_df[condition_df["degradation"].isin(["bias_medium", "jump_medium"])][
        [
            "degradation",
            "Formal_E1_ADE",
            "V1_ADE",
            "V2_selected_ADE",
            "V3_ADE",
            "V1_high_ADE",
            "V2_high_ADE",
            "V3_high_ADE",
            "V1_low_ADE",
            "V2_low_ADE",
            "V3_low_ADE",
            "V1_motion_usage_ratio",
            "V2_motion_usage_ratio",
            "V3_motion_usage_ratio",
            "V1_noisy_reversion_gap",
            "V2_noisy_reversion_gap",
            "V3_noisy_reversion_gap",
            "V1_smooth_acc_rms",
            "V2_smooth_acc_rms",
            "V3_smooth_acc_rms",
        ]
    ]
    bias_offset = full[full["degradation"].eq("bias_medium")][["method", "ADE_mean", "bias_offset_error", "smooth_acc_rms_mean"]].dropna()
    c2_failures = condition_df[condition_df["V2_C2_pass_1p05"] != True]["degradation"].tolist()

    lines = [
        "# E2-Min Condition-Level Diagnostics",
        "",
        "## Protocol",
        "All Stage 4 formal E1/E2 evaluation is interpreted under the unified six-condition protocol: gaussian_medium, drift_medium, burst_medium, bias_medium, jump_medium, and combined_medium. Inputs are Stage 3 protocol-validated per-frame conditional outputs; provenance labels are retained, but all ADE means and decisions use all six conditions.",
        "",
        "## Six-Condition Mean ADE",
        markdown_table(mean_df),
        "",
        "## Layered PASS / NO-PASS",
        markdown_table(layered_df),
        "",
        f"Final decision: **{flags['overall']}**. E2-Min is partial-positive under six-condition global ADE, but its hypothesis-level success is limited because it does not fully repair high-confidence no-harm and bias anchoring.",
        "",
        "## Why V2 Was Selected",
        "V2_uniform_cond_motion was selected by the existing E2-Min six-ADE rule: maximize the number of conditions where ADE improves over Formal E1, then break ties by mean ADE gain and mean ADE. The best setting was alpha0=0.5, beta=0.05, sigma0=1.0.",
        "",
        markdown_table(top_sweep_table(sweep)),
        "",
        "## ADE by Condition",
        markdown_table(ade_table),
        "",
        "## C2 High-Confidence No-Harm",
        markdown_table(c2_table),
        "",
        "C2 failures for selected V2: " + ", ".join(c2_failures) + ". The substantive failures are drift and burst; gaussian and combined are mild bin-level mismatches, while jump is affected by the sparse-outlier/zero-noisy-high-confidence edge case.",
        "",
        "## Drift Diagnostic",
        "V2 improves drift overall ADE versus Formal E1, but still fails high-confidence no-harm. That means the absolute-space objective helps the average trajectory yet still introduces too much error in frames that the oracle confidence marks as reliable.",
        "",
        "## Burst Diagnostic",
        markdown_table(burst_table),
        "",
        "V1's burst improvement is real at the condition ADE level: it is far below noisy and Formal E1, and its noisy_reversion_gap is strongly negative. However, the mechanism is mostly absolute anchoring plus aggressive smoothness over localized burst corruption, not a complete high-confidence repair: V1 still fails C2 and has very low acceleration RMS.",
        "",
        "## Bias and Jump Regression Diagnostic",
        markdown_table(bias_jump_table),
        "",
        "Bias regression is mainly caused by L2 anchoring to a biased observation: the selected V2 keeps the biased absolute offset instead of reducing it, while conditional motion cannot correct a global absolute shift. Jump regression is a known sparse-outlier limitation: the L2/smoothness objective mildly worsens Formal E1 despite remaining better than noisy, and jump is not one of the H1-H6 hypothesis targets.",
        "",
        "## Bias Offset",
        markdown_table(bias_offset),
        "",
        "## Layer 2 Criteria",
        f"- H1 drift high-confidence no-harm: {flags['H1_drift_high_conf_no_harm']}",
        f"- H2 burst high-confidence no-harm: {flags['H2_burst_high_conf_no_harm']}",
        f"- H3 drift overall ADE improves: {flags['H3_drift_ADE_improves']}",
        f"- H4 burst overall ADE improves: {flags['H4_burst_ADE_improves']}",
        f"- H5 bias ADE or offset improves: {flags['H5_bias_ADE_or_offset_improves']}",
        f"- H6 low-confidence preservation: {flags['H6_low_conf_preservation']}",
        "",
        "## E2-DPS Decision",
        "E2-DPS should be pursued. It must specifically fix high-confidence anchoring without damaging reliable frames, handle global bias as an absolute-position posterior problem rather than motion-only correction, and avoid L2 smoothing failure around sparse jump outliers.",
    ]
    CONDITION_DIAGNOSTICS_MD_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    full, bins, per_traj, pass_fail, diagnostics, sweep = load_tables()
    condition_df = compute_condition_table(full, bins, diagnostics)
    layered_df, means, flags = compute_layered_summary(full, diagnostics, pass_fail, condition_df)
    condition_df.to_csv(CONDITION_DIAGNOSTICS_PATH, index=False)
    layered_df.to_csv(LAYERED_SUMMARY_PATH, index=False)
    write_markdown(full, bins, diagnostics, sweep, condition_df, layered_df, means, flags)

    c2_failures = condition_df[condition_df["V2_C2_pass_1p05"] != True]["degradation"].tolist()
    print("STAGE4_E2_MIN_CONDITION_LEVEL_DIAGNOSTICS_COMPLETE")
    print(f"Formal_E1_mean_ADE: {means['formal_e1_linear_gate']:.9f}")
    print(f"V1_mean_ADE: {means['V1_abs_anchor_smooth']:.9f}")
    print(f"V2_selected_mean_ADE: {means['V2_uniform_cond_motion']:.9f}")
    print(f"V3_mean_ADE: {means['V3_conf_mod_cond_motion']:.9f}")
    print(f"G1: {flags['G1']}")
    print(f"G2: {flags['G2']}")
    print(f"G3: {flags['G3']}")
    for key in [
        "H1_drift_high_conf_no_harm",
        "H2_burst_high_conf_no_harm",
        "H3_drift_ADE_improves",
        "H4_burst_ADE_improves",
        "H5_bias_ADE_or_offset_improves",
        "H6_low_conf_preservation",
    ]:
        print(f"{key}: {flags[key]}")
    print(f"final_decision: {flags['overall']}")
    print(f"C2_failures_selected_V2: {', '.join(c2_failures)}")
    print(f"markdown: {CONDITION_DIAGNOSTICS_MD_PATH}")
    print(f"condition_csv: {CONDITION_DIAGNOSTICS_PATH}")
    print(f"layered_summary_csv: {LAYERED_SUMMARY_PATH}")


if __name__ == "__main__":
    main()
