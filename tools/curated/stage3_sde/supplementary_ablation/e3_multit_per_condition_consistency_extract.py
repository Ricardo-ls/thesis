from __future__ import annotations

import math
import sys
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUT_ROOT = PROJECT_ROOT / "outputs" / "stage4" / "e3_multit_candidate_source_ablation"
OUT_DIR = OUT_ROOT / "per_condition_consistency"

FULL_METRICS = OUT_ROOT / "e3_multit_candidate_source_full_metrics.csv"
BIN_METRICS = OUT_ROOT / "e3_multit_candidate_source_bin_metrics.csv"
CONDITION_SUMMARY = OUT_ROOT / "e3_multit_candidate_source_condition_summary.csv"
PASS_FAIL_SUMMARY = OUT_ROOT / "e3_multit_candidate_source_pass_fail_summary.csv"
DECISION_MD = OUT_ROOT / "e3_multit_candidate_source_decision_summary.md"

ADE_TABLE_PATH = OUT_DIR / "per_condition_ade_table.csv"
HIGH_CONF_PATH = OUT_DIR / "per_condition_high_conf_table.csv"
LOW_CONF_PATH = OUT_DIR / "per_condition_low_conf_table.csv"
MOTION_PATH = OUT_DIR / "per_condition_motion_usage_table.csv"
SUMMARY_MD_PATH = OUT_DIR / "per_condition_consistency_summary.md"

CONDITIONS = [
    "gaussian_medium",
    "drift_medium",
    "burst_medium",
    "bias_medium",
    "jump_medium",
    "combined_medium",
]

CANONICAL_VARIANTS = {
    "t1": "V1_single_t1",
    "t2": "V2_single_t2",
    "t3": "V3_single_t3",
    "multit": "V4_multit_confidence_bin",
}


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def require_file(path: Path) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"Required input not found: {path}")


def infer_variant_mapping(variants: list[str]) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for variant in variants:
        lower = variant.lower()
        if ("single" in lower or "v1" in lower) and "t1" in lower:
            mapping["t1"] = variant
        elif ("single" in lower or "v2" in lower) and "t2" in lower:
            mapping["t2"] = variant
        elif ("single" in lower or "v3" in lower) and "t3" in lower:
            mapping["t3"] = variant
        elif "multi" in lower or "bin" in lower or "v4" in lower:
            mapping["multit"] = variant
    missing = [key for key in CANONICAL_VARIANTS if key not in mapping]
    if missing:
        raise RuntimeError(f"Could not infer variant mapping for keys {missing}; variants={variants}")
    return mapping


def fnum(value: object, digits: int = 9) -> str:
    try:
        fval = float(value)
    except Exception:
        return str(value)
    if math.isnan(fval):
        return "nan"
    return f"{fval:.{digits}f}"


def markdown_table(df: pd.DataFrame, columns: list[str], digits: int = 9) -> str:
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for _, row in df[columns].iterrows():
        vals = []
        for col in columns:
            value = row[col]
            if isinstance(value, float) or hasattr(value, "dtype"):
                vals.append(fnum(value, digits))
            else:
                vals.append(str(value))
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


def build_tables(full_df: pd.DataFrame, variant_map: dict[str, str]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    full_df = full_df.copy()
    reverse_map = {value: key for key, value in variant_map.items()}
    full_df["variant_short"] = full_df["variant"].map(reverse_map)
    if full_df["variant_short"].isna().any():
        unknown = sorted(full_df.loc[full_df["variant_short"].isna(), "variant"].unique())
        raise RuntimeError(f"Unknown variants in full metrics: {unknown}")

    ade_pivot = (
        full_df.pivot(index="condition", columns="variant_short", values="ADE")
        .reindex(CONDITIONS)
        .reset_index()
    )
    ade_pivot = ade_pivot.rename(
        columns={
            "t1": "ADE_t1",
            "t2": "ADE_t2",
            "t3": "ADE_t3",
            "multit": "ADE_multit_bin",
        }
    )
    ordered_cols = ["condition", "ADE_t1", "ADE_t2", "ADE_t3", "ADE_multit_bin"]
    ade_pivot = ade_pivot[ordered_cols]
    best_variants = []
    for _, row in ade_pivot.iterrows():
        candidates = {
            "t1": row["ADE_t1"],
            "t2": row["ADE_t2"],
            "t3": row["ADE_t3"],
            "multit_bin": row["ADE_multit_bin"],
        }
        best_variants.append(min(candidates, key=candidates.get))
    ade_pivot["best_variant"] = best_variants
    ade_pivot["t3_minus_t2"] = ade_pivot["ADE_t3"] - ade_pivot["ADE_t2"]
    ade_pivot["t3_minus_t1"] = ade_pivot["ADE_t3"] - ade_pivot["ADE_t1"]
    ade_pivot["multit_minus_t3"] = ade_pivot["ADE_multit_bin"] - ade_pivot["ADE_t3"]
    ade_pivot["t3_improves_over_t1"] = ade_pivot["t3_minus_t1"] < 0
    ade_pivot["t3_improves_over_t2"] = ade_pivot["t3_minus_t2"] < 0
    ade_pivot["multit_improves_over_all_single_t"] = ade_pivot["ADE_multit_bin"] < ade_pivot[
        ["ADE_t1", "ADE_t2", "ADE_t3"]
    ].min(axis=1)

    high_rows = []
    low_rows = []
    motion_rows = []
    for _, row in full_df.iterrows():
        high_rows.append(
            {
                "condition": row["condition"],
                "variant": row["variant_short"],
                "high_conf_ADE_y": row.get("high_conf_ADE_y", math.nan),
                "high_conf_ADE_fused": row.get("high_conf_ADE_fused", math.nan),
                "high_conf_ADE_ratio": row.get("high_conf_ADE_ratio", math.nan),
                "high_conf_no_harm_status": row.get("high_conf_no_harm_pass", ""),
            }
        )
        low_rows.append(
            {
                "condition": row["condition"],
                "variant": row["variant_short"],
                "low_conf_ADE_y": row.get("low_conf_ADE_y", math.nan),
                "low_conf_ADE_fused": row.get("low_conf_ADE_fused", math.nan),
                "low_conf_improvement": row.get("low_conf_improvement", math.nan),
                "low_conf_pass_status": row.get("low_conf_pass", ""),
            }
        )
        motion = float(row.get("motion_usage_ratio", math.nan))
        motion_rows.append(
            {
                "condition": row["condition"],
                "variant": row["variant_short"],
                "motion_usage_ratio": motion,
                "within_0p2_0p7": bool(0.2 <= motion <= 0.7),
            }
        )
    high_df = pd.DataFrame(high_rows).sort_values(["condition", "variant"], key=lambda s: s.map({c: i for i, c in enumerate(CONDITIONS)}).fillna(s))
    low_df = pd.DataFrame(low_rows).sort_values(["condition", "variant"], key=lambda s: s.map({c: i for i, c in enumerate(CONDITIONS)}).fillna(s))
    motion_df = pd.DataFrame(motion_rows).sort_values(["condition", "variant"], key=lambda s: s.map({c: i for i, c in enumerate(CONDITIONS)}).fillna(s))
    return ade_pivot, high_df, low_df, motion_df


def classify_pattern(ade_df: pd.DataFrame) -> tuple[str, str, str]:
    gaussian = ade_df[ade_df["condition"] == "gaussian_medium"].iloc[0]
    gaussian_t3_minus_t2 = float(gaussian["t3_minus_t2"])
    t3_best_count = int((ade_df["best_variant"] == "t3").sum())
    multit_best_count = int((ade_df["best_variant"] == "multit_bin").sum())
    t3_worse_t1_count = int((ade_df["t3_minus_t1"] > 0).sum())
    t3_worse_t2_count = int((ade_df["t3_minus_t2"] > 0).sum())

    approx_tol = 1e-4
    clearly_better_tol = -1e-4
    if multit_best_count >= 4:
        pattern = "Pattern 4"
        evidence = (
            f"multi-t-bin is best in {multit_best_count}/6 conditions, so it beats single-t variants in most conditions."
        )
        caution = "Multi-t-bin should be reconsidered as final candidate source."
    elif gaussian_t3_minus_t2 < clearly_better_tol:
        pattern = "Pattern 2"
        evidence = (
            "gaussian_medium has t3 clearly better than t2 "
            f"(t3_minus_t2={gaussian_t3_minus_t2:.9f}); t3 is best in {t3_best_count}/6 conditions."
        )
        caution = (
            "Protocol reconciliation is needed before final lock because this conflicts with the earlier Stage 3 "
            "vanilla gaussian t2 sweet spot."
        )
    elif abs(gaussian_t3_minus_t2) <= approx_tol or gaussian_t3_minus_t2 > 0:
        structural_t3_wins = int(
            ade_df[ade_df["condition"].isin(["drift_medium", "burst_medium", "bias_medium", "combined_medium"])]
            ["t3_improves_over_t1"]
            .sum()
        )
        pattern = "Pattern 1"
        evidence = (
            "gaussian_medium has t2 approximately equal to or better than t3 "
            f"(t3_minus_t2={gaussian_t3_minus_t2:.9f}), while t3 improves over t1 in "
            f"{structural_t3_wins}/4 checked structural conditions."
        )
        caution = (
            "Stage 3 vanilla t2 sweet spot and Stage 4 fused t3 mean-best are compatible if confidence fusion protects "
            "high-confidence frames and lets stronger t3 correction act mainly in lower-confidence regions."
        )
    elif t3_worse_t1_count >= 2 or t3_worse_t2_count >= 2:
        pattern = "Pattern 3"
        evidence = (
            f"t3 wins the mean but worsens {t3_worse_t1_count} conditions versus t1 and "
            f"{t3_worse_t2_count} conditions versus t2."
        )
        caution = "t3 may be mean-best but not robustness-best; final lock should be reconsidered."
    else:
        pattern = "Pattern 2"
        evidence = (
            "gaussian_medium does not preserve the earlier t2 sweet spot under the fused protocol, and no alternate "
            "multi-t robustness pattern dominates."
        )
        caution = "Protocol reconciliation is needed before final lock."
    return pattern, evidence, caution


def write_summary(
    variant_map: dict[str, str],
    ade_df: pd.DataFrame,
    high_df: pd.DataFrame,
    low_df: pd.DataFrame,
    motion_df: pd.DataFrame,
    pattern: str,
    evidence: str,
    caution: str,
) -> None:
    diff_cols = [
        "condition",
        "t3_minus_t2",
        "t3_minus_t1",
        "multit_minus_t3",
        "best_variant",
        "t3_improves_over_t1",
        "t3_improves_over_t2",
        "multit_improves_over_all_single_t",
    ]
    high_compact = high_df.pivot(index="condition", columns="variant", values="high_conf_no_harm_status").reindex(CONDITIONS).reset_index()
    low_compact = low_df.pivot(index="condition", columns="variant", values="low_conf_pass_status").reindex(CONDITIONS).reset_index()
    motion_compact = motion_df.pivot(index="condition", columns="variant", values="motion_usage_ratio").reindex(CONDITIONS).reset_index()

    lines = [
        "# E3 Multi-t Per-Condition Consistency Extraction",
        "",
        "## A. Purpose And Constraints",
        "",
        "This is a read-only extraction from existing Stage 4 E3 multi-t candidate-source ablation outputs. It does not rerun SDEdit, generate candidates, train, use DPS, modify checkpoints, or change tau/gamma/t_start/confidence thresholds.",
        "",
        "## Variant Mapping",
        "",
        *[f"- `{short}` -> `{actual}`" for short, actual in variant_map.items()],
        "",
        "## B. 4 x 6 ADE Table",
        "",
        markdown_table(ade_df, ["condition", "ADE_t1", "ADE_t2", "ADE_t3", "ADE_multit_bin", "best_variant"], 9),
        "",
        "## C. Difference Table",
        "",
        markdown_table(ade_df, diff_cols, 9),
        "",
        "## D. High-Confidence No-Harm Table",
        "",
        markdown_table(high_compact, list(high_compact.columns), 9),
        "",
        "## E. Low-Confidence Improvement Table",
        "",
        markdown_table(low_compact, list(low_compact.columns), 9),
        "",
        "## F. Motion Usage Sanity Table",
        "",
        markdown_table(motion_compact, list(motion_compact.columns), 9),
        "",
        "All motion_usage_ratio entries are also written with `[0.2, 0.7]` sanity flags in the CSV output.",
        "",
        "## G. Pattern Classification",
        "",
        f"- Classification: `{pattern}`",
        f"- Evidence: {evidence}",
        "",
        "## H. Final Caution",
        "",
        caution,
        "",
        "This extraction does not make the final scientific decision alone; it supplies the per-condition evidence needed before locking the final t_start/source rule.",
        "",
        "## Output Files",
        "",
        f"- `{rel(ADE_TABLE_PATH)}`",
        f"- `{rel(HIGH_CONF_PATH)}`",
        f"- `{rel(LOW_CONF_PATH)}`",
        f"- `{rel(MOTION_PATH)}`",
        f"- `{rel(SUMMARY_MD_PATH)}`",
    ]
    SUMMARY_MD_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    for path in [FULL_METRICS, BIN_METRICS, CONDITION_SUMMARY, PASS_FAIL_SUMMARY, DECISION_MD]:
        require_file(path)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    full_df = pd.read_csv(FULL_METRICS)
    _ = pd.read_csv(BIN_METRICS)
    variants = sorted(full_df["variant"].unique())
    variant_map = infer_variant_mapping(variants)
    print("VARIANT_MAPPING=" + ", ".join(f"{short}:{actual}" for short, actual in variant_map.items()))

    ade_df, high_df, low_df, motion_df = build_tables(full_df, variant_map)
    pattern, evidence, caution = classify_pattern(ade_df)

    ade_df.to_csv(ADE_TABLE_PATH, index=False)
    high_df.to_csv(HIGH_CONF_PATH, index=False)
    low_df.to_csv(LOW_CONF_PATH, index=False)
    motion_df.to_csv(MOTION_PATH, index=False)
    write_summary(variant_map, ade_df, high_df, low_df, motion_df, pattern, evidence, caution)

    diff_cols = ["condition", "t3_minus_t2", "t3_minus_t1", "multit_minus_t3", "best_variant"]
    print(f"COMMAND=.venv/bin/python {rel(Path(__file__))}")
    print("\nPER_CONDITION_ADE_TABLE")
    print(ade_df[["condition", "ADE_t1", "ADE_t2", "ADE_t3", "ADE_multit_bin", "best_variant"]].to_string(index=False))
    print("\nDIFFERENCE_TABLE")
    print(ade_df[diff_cols].to_string(index=False))
    print(f"\nPATTERN_CLASSIFICATION={pattern}")
    print(f"PATTERN_EVIDENCE={evidence}")
    print(f"FINAL_CAUTION={caution}")
    print(f"SUMMARY_PATH={rel(SUMMARY_MD_PATH)}")
    print(f"[FILE] {rel(ADE_TABLE_PATH)} written")
    print(f"[FILE] {rel(HIGH_CONF_PATH)} written")
    print(f"[FILE] {rel(LOW_CONF_PATH)} written")
    print(f"[FILE] {rel(MOTION_PATH)} written")
    print(f"[FILE] {rel(SUMMARY_MD_PATH)} written")


if __name__ == "__main__":
    main()
