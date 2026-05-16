from __future__ import annotations

import math
import os
from pathlib import Path
import sys

os.environ.setdefault("MPLCONFIGDIR", "/tmp/e3_gain_bottleneck_mpl")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd


CONDITIONS = [
    "gaussian_medium",
    "drift_medium",
    "burst_medium",
    "bias_medium",
    "jump_medium",
    "combined_medium",
]
T_STARTS = [1, 2, 3]
EPS = 1e-8

E3_DIR = PROJECT_ROOT / "outputs" / "stage4" / "e3_holdout1000_confidence_aware_sdedit"
ARRAY_DIR = E3_DIR / "arrays"
GLOBAL_METRICS_PATH = E3_DIR / "e3_holdout1000_global_t1_metrics.csv"
BEST_METRICS_PATH = E3_DIR / "e3_holdout1000_per_condition_best_metrics.csv"
TREND_PATH = E3_DIR / "e3_holdout1000_trend_stability_summary.csv"
FROZEN_DOC_PATH = PROJECT_ROOT / "docs" / "stage4" / "e3_frozen_holdout1000_baseline.md"

OUT_DIR = PROJECT_ROOT / "outputs" / "stage4" / "e3_gain_bottleneck_minimal_diagnosis"
ORACLE_PATH = OUT_DIR / "e3_oracle_upper_bound_minimal.csv"
DIRECTION_PATH = OUT_DIR / "e3_correction_direction_minimal.csv"
DECISION_PATH = OUT_DIR / "e3_gain_bottleneck_decision_matrix.md"
SUMMARY_PATH = OUT_DIR / "e3_gain_bottleneck_minimal_summary.md"


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def require_inputs() -> None:
    missing = [path for path in [E3_DIR, ARRAY_DIR, GLOBAL_METRICS_PATH, BEST_METRICS_PATH, TREND_PATH, FROZEN_DOC_PATH] if not path.exists()]
    for condition in CONDITIONS:
        for suffix in [
            "clean",
            "degraded",
            "confidence",
            "sdedit_t1",
            "sdedit_t2",
            "sdedit_t3",
            "fused_t1_tau07_gamma2",
            "fused_best_tau07_gamma2",
        ]:
            path = ARRAY_DIR / f"{condition}_{suffix}.npy"
            if not path.is_file():
                missing.append(path)
    if missing:
        raise FileNotFoundError("Missing required frozen E3 inputs:\n" + "\n".join(str(path) for path in missing))


def load_condition(condition: str) -> dict[str, np.ndarray]:
    arrays = {
        "clean": np.load(ARRAY_DIR / f"{condition}_clean.npy").astype(np.float32),
        "degraded": np.load(ARRAY_DIR / f"{condition}_degraded.npy").astype(np.float32),
        "confidence": np.load(ARRAY_DIR / f"{condition}_confidence.npy").astype(np.float32),
        "sdedit_t1": np.load(ARRAY_DIR / f"{condition}_sdedit_t1.npy").astype(np.float32),
        "sdedit_t2": np.load(ARRAY_DIR / f"{condition}_sdedit_t2.npy").astype(np.float32),
        "sdedit_t3": np.load(ARRAY_DIR / f"{condition}_sdedit_t3.npy").astype(np.float32),
        "fused_t1": np.load(ARRAY_DIR / f"{condition}_fused_t1_tau07_gamma2.npy").astype(np.float32),
        "fused_best": np.load(ARRAY_DIR / f"{condition}_fused_best_tau07_gamma2.npy").astype(np.float32),
    }
    clean_shape = arrays["clean"].shape
    if clean_shape != (1000, 20, 2):
        raise ValueError(f"{condition}: unexpected clean shape {clean_shape}")
    for name, arr in arrays.items():
        expected = clean_shape[:2] if name == "confidence" else clean_shape
        if arr.shape != expected:
            raise ValueError(f"{condition}: {name} shape {arr.shape} != {expected}")
        if not np.all(np.isfinite(arr)):
            raise ValueError(f"{condition}: {name} contains non-finite values")
    return arrays


def frame_error(pred: np.ndarray, clean: np.ndarray) -> np.ndarray:
    return np.linalg.norm(pred - clean, axis=-1).astype(np.float32)


def ade(pred: np.ndarray, clean: np.ndarray) -> float:
    return float(frame_error(pred, clean).mean())


def rmse(pred: np.ndarray, clean: np.ndarray) -> float:
    err = frame_error(pred, clean)
    return float(np.sqrt(np.mean(err**2)))


def bin_masks(confidence: np.ndarray) -> dict[str, np.ndarray]:
    return {
        "high": confidence > 0.7,
        "mid": (confidence >= 0.3) & (confidence <= 0.7),
        "low": confidence < 0.3,
    }


def masked_ade(pred: np.ndarray, clean: np.ndarray, mask: np.ndarray) -> float:
    err = frame_error(pred, clean)
    return float(err[mask].mean()) if int(mask.sum()) else math.nan


def select_framewise(candidates: list[np.ndarray], clean: np.ndarray) -> np.ndarray:
    stack = np.stack(candidates, axis=0)
    errors = np.stack([frame_error(candidate, clean) for candidate in candidates], axis=0)
    best_idx = np.argmin(errors, axis=0)
    selected = np.take_along_axis(stack, best_idx[None, :, :, None], axis=0)[0]
    return selected.astype(np.float32)


def select_trajectory_level(candidates: list[np.ndarray], clean: np.ndarray) -> np.ndarray:
    stack = np.stack(candidates, axis=0)
    per_candidate_ade = np.stack([frame_error(candidate, clean).mean(axis=1) for candidate in candidates], axis=0)
    best_idx = np.argmin(per_candidate_ade, axis=0)
    traj_idx = np.arange(clean.shape[0])
    return stack[best_idx, traj_idx].astype(np.float32)


def oracle_predictions(arrays: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    clean = arrays["clean"]
    degraded = arrays["degraded"]
    confidence = arrays["confidence"]
    sdedit_candidates = [arrays["sdedit_t1"], arrays["sdedit_t2"], arrays["sdedit_t3"]]
    all_candidates = [degraded] + sdedit_candidates
    framewise = select_framewise(all_candidates, clean)
    best_sdedit_non_high = select_framewise(sdedit_candidates, clean)
    confidence_constrained = np.where((confidence > 0.7)[..., None], degraded, best_sdedit_non_high).astype(np.float32)
    traj_best = select_trajectory_level(all_candidates, clean)
    return {
        "oracle_A_framewise_best": framewise,
        "oracle_B_confidence_constrained_framewise": confidence_constrained,
        "oracle_C_trajectory_best_tstart": traj_best,
    }


def oracle_rows(condition: str, arrays: dict[str, np.ndarray]) -> list[dict]:
    clean = arrays["clean"]
    degraded = arrays["degraded"]
    confidence = arrays["confidence"]
    fused_t1 = arrays["fused_t1"]
    fused_best = arrays["fused_best"]
    masks = bin_masks(confidence)
    ade_noisy = ade(degraded, clean)
    ade_fused_t1 = ade(fused_t1, clean)
    ade_fused_best = ade(fused_best, clean)
    rows = []
    for oracle_name, pred in oracle_predictions(arrays).items():
        oracle_ade = ade(pred, clean)
        gap_t1 = ade_fused_t1 - oracle_ade
        gap_best = ade_fused_best - oracle_ade
        denom_remaining = max(ade_noisy - oracle_ade, EPS)
        rows.append(
            {
                "condition": condition,
                "oracle_variant": oracle_name,
                "ADE_noisy": ade_noisy,
                "ADE_fused_t1_tau07_gamma2": ade_fused_t1,
                "ADE_fused_best_tau07_gamma2": ade_fused_best,
                "ADE_oracle": oracle_ade,
                "RMSE_oracle": rmse(pred, clean),
                "high_conf_ADE_oracle": masked_ade(pred, clean, masks["high"]),
                "mid_conf_ADE_oracle": masked_ade(pred, clean, masks["mid"]),
                "low_conf_ADE_oracle": masked_ade(pred, clean, masks["low"]),
                "improvement_vs_noisy": ade_noisy - oracle_ade,
                "gap_vs_fused_t1": gap_t1,
                "gap_vs_fused_best": gap_best,
                "oracle_gap_ratio_t1": gap_t1 / max(ade_fused_t1, EPS),
                "oracle_gap_ratio_best": gap_best / max(ade_fused_best, EPS),
                "remaining_gain_vs_noisy_t1": gap_t1 / denom_remaining,
            }
        )
    return rows


def direction_rows(condition: str, arrays: dict[str, np.ndarray]) -> list[dict]:
    clean = arrays["clean"]
    degraded = arrays["degraded"]
    confidence = arrays["confidence"]
    oracle = clean - degraded
    oracle_norm = np.linalg.norm(oracle, axis=-1)
    valid_oracle = oracle_norm > EPS
    rows = []
    for t_start in T_STARTS:
        pred = arrays[f"sdedit_t{t_start}"]
        r = pred - degraded
        r_norm = np.linalg.norm(r, axis=-1)
        dot = np.sum(r * oracle, axis=-1)
        cosine = dot / (r_norm * oracle_norm + EPS)
        magnitude_ratio = r_norm / (oracle_norm + EPS)
        for bin_name, mask in bin_masks(confidence).items():
            use = mask & valid_oracle
            cos_vals = cosine[use]
            mag_vals = magnitude_ratio[use]
            rows.append(
                {
                    "condition": condition,
                    "t_start": t_start,
                    "confidence_bin": bin_name,
                    "sample_count": int(use.sum()),
                    "cosine_mean": float(np.mean(cos_vals)) if cos_vals.size else math.nan,
                    "cosine_median": float(np.median(cos_vals)) if cos_vals.size else math.nan,
                    "fraction_cosine_gt_0": float(np.mean(cos_vals > 0.0)) if cos_vals.size else math.nan,
                    "fraction_cosine_gt_0p5": float(np.mean(cos_vals > 0.5)) if cos_vals.size else math.nan,
                    "fraction_cosine_lt_0": float(np.mean(cos_vals < 0.0)) if cos_vals.size else math.nan,
                    "magnitude_ratio_mean": float(np.mean(mag_vals)) if mag_vals.size else math.nan,
                    "magnitude_ratio_median": float(np.median(mag_vals)) if mag_vals.size else math.nan,
                }
            )
    return rows


def classify_case(gap_ratio: float, cosine: float, mag: float, neg_frac: float) -> str:
    if gap_ratio < 0.05:
        return "Case D"
    if 0.05 <= gap_ratio <= 0.10:
        return "Case E"
    if cosine > 0.5 and mag < 0.75:
        return "Case A"
    if cosine > 0.5 and 0.75 <= mag <= 1.25:
        return "Case B"
    if cosine < 0.3 or neg_frac > 0.35:
        return "Case C"
    return "Case E"


def case_recommendation(case: str) -> str:
    return {
        "Case A": "selection has space and correction direction is good but underused; test adaptive gamma < 1 only on a new independent hold-out",
        "Case B": "SDEdit candidates are useful and selection is bottleneck; test adaptive t_start / confidence-bin selection only on a new independent hold-out",
        "Case C": "oracle gains are not learnable from confidence alone; do not amplify without prior/model or confidence improvement",
        "Case D": "current fusion is near the SDEdit-candidate ceiling; stop tuning fusion and consider reporting stable limited improvement or a separate SDEdit-initialized DPS audit",
        "Case E": "boundary case; only a small adaptive test is justified if direction diagnostics are strong, and it must use a new independent hold-out",
    }[case]


def markdown_table(df: pd.DataFrame, cols: list[str]) -> str:
    show = df[cols].copy()
    for col in show.columns:
        show[col] = show[col].map(
            lambda x: f"{float(x):.6f}" if isinstance(x, (float, np.floating)) and np.isfinite(x) else str(x)
        )
    lines = [
        "| " + " | ".join(show.columns) + " |",
        "| " + " | ".join(["---"] * len(show.columns)) + " |",
    ]
    for values in show.values.tolist():
        lines.append("| " + " | ".join(str(value) for value in values) + " |")
    return "\n".join(lines)


def build_decision_df(oracle_df: pd.DataFrame, direction_df: pd.DataFrame) -> pd.DataFrame:
    b = oracle_df[oracle_df["oracle_variant"] == "oracle_B_confidence_constrained_framewise"].copy()
    rows = []
    for condition in CONDITIONS:
        gap_row = b[b["condition"] == condition].iloc[0]
        low = direction_df[(direction_df["condition"] == condition) & (direction_df["confidence_bin"] == "low")].copy()
        low["score"] = low["cosine_mean"].fillna(-999.0)
        best_low = low.sort_values(["score", "fraction_cosine_gt_0p5"], ascending=[False, False]).iloc[0]
        mid = direction_df[(direction_df["condition"] == condition) & (direction_df["confidence_bin"] == "mid")].copy()
        mid["score"] = mid["cosine_mean"].fillna(-999.0)
        best_mid = mid.sort_values(["score", "fraction_cosine_gt_0p5"], ascending=[False, False]).iloc[0]
        gap_ratio = float(gap_row["oracle_gap_ratio_t1"])
        cosine = float(best_low["cosine_mean"])
        mag = float(best_low["magnitude_ratio_median"])
        neg_frac = float(best_low["fraction_cosine_lt_0"])
        case = classify_case(gap_ratio, cosine, mag, neg_frac)
        rows.append(
            {
                "condition": condition,
                "confidence_oracle_gap_ratio_t1": gap_ratio,
                "confidence_oracle_gap_ratio_best": float(gap_row["oracle_gap_ratio_best"]),
                "best_low_t_start_by_cosine": int(best_low["t_start"]),
                "low_cosine_mean": cosine,
                "low_fraction_cosine_gt_0p5": float(best_low["fraction_cosine_gt_0p5"]),
                "low_fraction_cosine_lt_0": neg_frac,
                "low_magnitude_ratio_median": mag,
                "best_mid_t_start_by_cosine": int(best_mid["t_start"]),
                "mid_cosine_mean": float(best_mid["cosine_mean"]),
                "decision_case": case,
                "recommendation": case_recommendation(case),
            }
        )
    return pd.DataFrame(rows)


def write_decision_matrix(decision_df: pd.DataFrame, oracle_df: pd.DataFrame, direction_df: pd.DataFrame) -> None:
    cols = [
        "condition",
        "confidence_oracle_gap_ratio_t1",
        "best_low_t_start_by_cosine",
        "low_cosine_mean",
        "low_magnitude_ratio_median",
        "low_fraction_cosine_lt_0",
        "decision_case",
        "recommendation",
    ]
    overall_case = decision_df["decision_case"].mode().iloc[0]
    lines = [
        "# E3 Gain Bottleneck Decision Matrix",
        "",
        "Oracle diagnostics are analysis-only and are not method performance.",
        "",
        "Decision uses the confidence-constrained frame-wise oracle gap vs `fused_t1_tau07_gamma2`, combined with the best low-confidence direction alignment across t_start 1/2/3.",
        "",
        markdown_table(decision_df, cols),
        "",
        "## Overall",
        "",
        f"- modal case: `{overall_case}`",
        f"- recommendation: {case_recommendation(overall_case)}",
        "",
        "Case thresholds:",
        "",
        "- Case A: oracle gap > 10%, low-conf cosine > 0.5, median magnitude ratio < 0.75",
        "- Case B: oracle gap > 10%, low-conf cosine > 0.5, median magnitude ratio in [0.75, 1.25]",
        "- Case C: oracle gap > 10%, low-conf cosine < 0.3 or negative-direction fraction > 0.35",
        "- Case D: oracle gap < 5%",
        "- Case E: oracle gap 5-10% or mixed direction/magnitude evidence",
    ]
    DECISION_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[FILE] {rel(DECISION_PATH)} written")


def write_summary(oracle_df: pd.DataFrame, direction_df: pd.DataFrame, decision_df: pd.DataFrame) -> None:
    b = oracle_df[oracle_df["oracle_variant"] == "oracle_B_confidence_constrained_framewise"]
    a = oracle_df[oracle_df["oracle_variant"] == "oracle_A_framewise_best"]
    c = oracle_df[oracle_df["oracle_variant"] == "oracle_C_trajectory_best_tstart"]

    def mean_col(df: pd.DataFrame, col: str) -> float:
        return float(df[col].mean())

    low = direction_df[direction_df["confidence_bin"] == "low"]
    mid = direction_df[direction_df["confidence_bin"] == "mid"]
    best_low_overall = (
        low.groupby("t_start", as_index=False)
        .agg(cosine_mean=("cosine_mean", "mean"), magnitude_ratio_median=("magnitude_ratio_median", "median"))
        .sort_values("cosine_mean", ascending=False)
        .iloc[0]
    )
    best_mid_overall = (
        mid.groupby("t_start", as_index=False)
        .agg(cosine_mean=("cosine_mean", "mean"), magnitude_ratio_median=("magnitude_ratio_median", "median"))
        .sort_values("cosine_mean", ascending=False)
        .iloc[0]
    )
    overall_case = decision_df["decision_case"].mode().iloc[0]

    lines = [
        "# E3 Gain Bottleneck Minimal Diagnosis",
        "",
        "Current 1000 hold-out results were frozen: `yes`.",
        "",
        "This diagnosis reads only existing arrays. It does not run new SDEdit sampling, train models, modify checkpoints, tune tau/gamma/t_start, or create a new formal method.",
        "",
        "## Oracle Upper Bound",
        "",
        f"- Frame-wise best oracle mean gap ratio vs fused_t1: `{mean_col(a, 'oracle_gap_ratio_t1'):.6f}`",
        f"- Confidence-constrained oracle mean gap ratio vs fused_t1: `{mean_col(b, 'oracle_gap_ratio_t1'):.6f}`",
        f"- Confidence-constrained oracle mean gap ratio vs fused_best: `{mean_col(b, 'oracle_gap_ratio_best'):.6f}`",
        f"- Trajectory-level best oracle mean gap ratio vs fused_t1: `{mean_col(c, 'oracle_gap_ratio_t1'):.6f}`",
        "",
        "Interpretation:",
        "",
    ]
    conf_gap = mean_col(b, "oracle_gap_ratio_t1")
    if conf_gap > 0.10:
        lines.append("- Confidence-constrained oracle is substantially better than current fused_t1; selection/fusion has clear improvement space.")
    elif conf_gap < 0.05:
        lines.append("- Confidence-constrained oracle is close to current fused_t1; current fusion is near the SDEdit-candidate ceiling.")
    else:
        lines.append("- Confidence-constrained oracle shows a boundary-size remaining gap.")

    lines += [
        "",
        "## Correction Direction And Magnitude",
        "",
        f"- Best overall low-confidence t_start by cosine: `t{int(best_low_overall['t_start'])}`",
        f"- Best low-confidence cosine mean: `{float(best_low_overall['cosine_mean']):.6f}`",
        f"- Best low-confidence median magnitude ratio: `{float(best_low_overall['magnitude_ratio_median']):.6f}`",
        f"- Best overall mid-confidence t_start by cosine: `t{int(best_mid_overall['t_start'])}`",
        f"- Best mid-confidence cosine mean: `{float(best_mid_overall['cosine_mean']):.6f}`",
        "",
    ]
    if float(best_low_overall["cosine_mean"]) > 0.5:
        lines.append("- Low-confidence SDEdit correction direction is reliably positive on average.")
    elif float(best_low_overall["cosine_mean"]) < 0.3:
        lines.append("- Low-confidence SDEdit correction direction is weak; amplifying SDEdit would be risky.")
    else:
        lines.append("- Low-confidence SDEdit correction direction is mixed.")

    mag = float(best_low_overall["magnitude_ratio_median"])
    if mag < 0.75:
        lines.append("- Low-confidence correction magnitude is generally too weak relative to oracle correction.")
    elif mag <= 1.25:
        lines.append("- Low-confidence correction magnitude is broadly adequate.")
    else:
        lines.append("- Low-confidence correction magnitude is often too strong.")

    lines += [
        "",
        "## Decision Cases",
        "",
        markdown_table(
            decision_df,
            [
                "condition",
                "confidence_oracle_gap_ratio_t1",
                "best_low_t_start_by_cosine",
                "low_cosine_mean",
                "low_magnitude_ratio_median",
                "decision_case",
            ],
        ),
        "",
        f"Overall modal case: `{overall_case}`.",
        f"Recommended next step: {case_recommendation(overall_case)}.",
        "",
        "If any adaptive gamma or adaptive t_start rule is tested next, it must be validated on a new independent hold-out set, not seed 13000-13999.",
        "",
        "## Output Files",
        "",
        f"- `{rel(ORACLE_PATH)}`",
        f"- `{rel(DIRECTION_PATH)}`",
        f"- `{rel(DECISION_PATH)}`",
    ]
    SUMMARY_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[FILE] {rel(SUMMARY_PATH)} written")


def main() -> None:
    require_inputs()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    oracle_all: list[dict] = []
    direction_all: list[dict] = []
    for condition in CONDITIONS:
        arrays = load_condition(condition)
        oracle_all.extend(oracle_rows(condition, arrays))
        direction_all.extend(direction_rows(condition, arrays))
        print(f"[DONE] diagnosis condition={condition}")

    oracle_df = pd.DataFrame(oracle_all)
    direction_df = pd.DataFrame(direction_all)
    oracle_df.to_csv(ORACLE_PATH, index=False)
    print(f"[FILE] {rel(ORACLE_PATH)} written")
    direction_df.to_csv(DIRECTION_PATH, index=False)
    print(f"[FILE] {rel(DIRECTION_PATH)} written")

    decision_df = build_decision_df(oracle_df, direction_df)
    write_decision_matrix(decision_df, oracle_df, direction_df)
    write_summary(oracle_df, direction_df, decision_df)

    b = oracle_df[oracle_df["oracle_variant"] == "oracle_B_confidence_constrained_framewise"]
    low = direction_df[direction_df["confidence_bin"] == "low"]
    low_by_t = low.groupby("t_start", as_index=False)["cosine_mean"].mean().sort_values("cosine_mean", ascending=False)
    print("E3_GAIN_BOTTLENECK_MINIMAL_DIAGNOSIS_COMPLETE")
    print(f"frozen_baseline_doc={rel(FROZEN_DOC_PATH)}")
    print(f"confidence_constrained_oracle_gap_ratio_t1_mean={float(b['oracle_gap_ratio_t1'].mean()):.6f}")
    print(f"confidence_constrained_oracle_gap_ratio_best_mean={float(b['oracle_gap_ratio_best'].mean()):.6f}")
    print(f"best_low_direction_t_start=t{int(low_by_t.iloc[0]['t_start'])}")
    print("decision_cases=" + ",".join(decision_df["decision_case"].tolist()))
    print(f"output_dir={rel(OUT_DIR)}")


if __name__ == "__main__":
    main()
