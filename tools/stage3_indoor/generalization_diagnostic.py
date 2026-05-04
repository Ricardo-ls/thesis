from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import csv
import json
import os

os.environ.setdefault("MPLCONFIGDIR", "/tmp/stage3_indoor_generalization_mpl")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.colors import TwoSlopeNorm

from diffusion.ddpm_utils import DDPMForwardProcess
from models.temporal_denoiser import TemporalDenoiser1D
from models.temporal_denoiser_conditional import ConditionalTemporalDenoiser1D
from tools.stage3_indoor.sdedit_gaussian_full import run_sdedit
from tools.stage3_indoor.train_cond_residual_gaussian import (
    load_state_dict_flexible,
    sample_conditional_residual,
)

try:
    from scipy.stats import wilcoxon

    SCIPY_AVAILABLE = True
except Exception:
    wilcoxon = None
    SCIPY_AVAILABLE = False


EXPECTED_CWD = "/Users/shangshanchong/Desktop/pytorch_env_check/trajectory_ddpm_mvp/SingularTrajectory"

CLEAN_PATH = PROJECT_ROOT / "data" / "stage3_indoor" / "clean_trajs.npy"
COND_CKPT_PATH = PROJECT_ROOT / "outputs" / "stage3_indoor" / "conditional_residual_ddpm_gaussian" / "seed42" / "best_ema_model.pt"
UNCOND_CKPT_PATH = PROJECT_ROOT / "outputs" / "stage3_indoor" / "ddpm_indoor_v2" / "seed42" / "best_ema_model.pt"
NORM_PATH = PROJECT_ROOT / "data" / "stage3_indoor" / "rel_norm_params_v2.npz"
COND_MODEL_DEF_PATH = PROJECT_ROOT / "models" / "temporal_denoiser_conditional.py"

OUTPUT_DIR = PROJECT_ROOT / "outputs" / "stage3_indoor" / "conditional_residual_ddpm_gaussian" / "seed42"
SUMMARY_PATH = OUTPUT_DIR / "generalization_summary.csv"
PER_TRAJ_PATH = OUTPUT_DIR / "generalization_per_traj.csv"
CONCLUSION_PATH = OUTPUT_DIR / "generalization_conclusion.json"
FIG_PATH = OUTPUT_DIR / "generalization_diagnostic.png"
DEGRADED_PATHS = {
    "gaussian_medium": OUTPUT_DIR / "generalization_degraded_gaussian.npy",
    "drift_medium": OUTPUT_DIR / "generalization_degraded_drift.npy",
    "jump_medium": OUTPUT_DIR / "generalization_degraded_jump.npy",
    "burst_medium": OUTPUT_DIR / "generalization_degraded_burst.npy",
    "bias_medium": OUTPUT_DIR / "generalization_degraded_bias.npy",
    "combined_medium": OUTPUT_DIR / "generalization_degraded_combined.npy",
}

TIMESTEPS = 100
N = 200
SAMPLE_SEEDS = [42, 43, 44, 45, 46]
DEGRADATION_ORDER = [
    "gaussian_medium",
    "drift_medium",
    "jump_medium",
    "burst_medium",
    "bias_medium",
    "combined_medium",
]
METHOD_ORDER = [
    "noisy_input",
    "uncond_sdedit_t2",
    "cond_residual_t20",
]
DEGRADATION_GROUPS = {
    "gaussian_medium": "relative_observable",
    "drift_medium": "relative_observable",
    "jump_medium": "relative_observable",
    "burst_medium": "relative_observable",
    "bias_medium": "relative_unobservable",
    "combined_medium": "partially_observable",
}
RELATIVE_OBSERVABLE = [
    "gaussian_medium",
    "drift_medium",
    "jump_medium",
    "burst_medium",
]
PARTIALLY_OBSERVABLE = ["combined_medium"]
RELATIVE_UNOBSERVABLE = ["bias_medium"]


def save_numpy(path: Path, array: np.ndarray) -> None:
    np.save(path, array.astype(np.float32))
    print(f"Saved: {path}")


def save_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved: {path}")


def save_json(path: Path, payload: dict) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f"Saved: {path}")


def format_pvalue_table(value: float) -> str:
    if np.isnan(value):
        return "nan"
    return f"{value:.4e}"


def format_pvalue_short(value: float) -> str:
    if np.isnan(value):
        return "nan"
    return f"{value:.4g}"


def safe_wilcoxon(delta: np.ndarray) -> float:
    if not SCIPY_AVAILABLE:
        return float("nan")
    if np.allclose(delta, 0.0):
        return 1.0
    try:
        return float(wilcoxon(delta, alternative="less").pvalue)
    except Exception:
        return float("nan")


def ensure_required_inputs() -> None:
    required_paths = [
        CLEAN_PATH,
        COND_CKPT_PATH,
        UNCOND_CKPT_PATH,
        NORM_PATH,
        COND_MODEL_DEF_PATH,
    ]
    for path in required_paths:
        if not path.is_file():
            raise FileNotFoundError(f"Missing required input: {path}")


def check_working_directory() -> None:
    print("=== Working Directory Check ===")
    print(f"cwd: {os.getcwd()}")
    if os.getcwd() != EXPECTED_CWD:
        raise RuntimeError(
            f"Current working directory must be {EXPECTED_CWD}, got {os.getcwd()}"
        )


def compute_metrics(pred: np.ndarray, clean: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    errors = np.linalg.norm(pred - clean, axis=-1)
    ade = errors.mean(axis=1).astype(np.float32)
    acc = pred[:, 2:, :] - 2.0 * pred[:, 1:-1, :] + pred[:, :-2, :]
    smooth = np.linalg.norm(acc, axis=-1).mean(axis=1).astype(np.float32)
    return ade, smooth


def generate_gaussian(clean: np.ndarray) -> np.ndarray:
    degraded = np.empty_like(clean, dtype=np.float32)
    for idx in range(clean.shape[0]):
        rng = np.random.default_rng(42 + idx)
        noise = rng.normal(0.0, 0.05, size=(20, 2)).astype(np.float32)
        degraded[idx] = clean[idx] + noise
    return degraded


def generate_drift(clean: np.ndarray) -> np.ndarray:
    degraded = np.empty_like(clean, dtype=np.float32)
    for idx in range(clean.shape[0]):
        rng = np.random.default_rng(42 + idx)
        drift_steps = np.zeros((20, 2), dtype=np.float32)
        drift_steps[1:] = rng.normal(0.0, 0.005, size=(19, 2)).astype(np.float32)
        drift = np.cumsum(drift_steps, axis=0)
        degraded[idx] = clean[idx] + drift
    return degraded


def generate_jump(clean: np.ndarray) -> np.ndarray:
    degraded = np.empty_like(clean, dtype=np.float32)
    for idx in range(clean.shape[0]):
        rng = np.random.default_rng(42 + idx)
        jump_count = int(rng.choice([2, 3, 4]))
        jump_indices = np.sort(rng.choice(np.arange(20), size=jump_count, replace=False))
        offsets = np.zeros((20, 2), dtype=np.float32)
        current_offset = np.zeros(2, dtype=np.float32)
        jump_ptr = 0
        for frame_idx in range(20):
            if jump_ptr < jump_count and frame_idx == jump_indices[jump_ptr]:
                magnitude = float(rng.uniform(0.2, 0.5))
                angle = float(rng.uniform(0.0, 2.0 * np.pi))
                current_offset = np.array(
                    [magnitude * np.cos(angle), magnitude * np.sin(angle)],
                    dtype=np.float32,
                )
                jump_ptr += 1
            offsets[frame_idx] = current_offset
        degraded[idx] = clean[idx] + offsets
    return degraded


def generate_burst(clean: np.ndarray) -> np.ndarray:
    degraded = np.empty_like(clean, dtype=np.float32)
    for idx in range(clean.shape[0]):
        rng = np.random.default_rng(42 + idx)
        burst_len = int(rng.choice([3, 4, 5]))
        burst_start = int(rng.integers(0, 20 - burst_len + 1))
        noise = rng.normal(0.0, 0.01, size=(20, 2)).astype(np.float32)
        noise[burst_start : burst_start + burst_len] = rng.normal(
            0.0,
            0.25,
            size=(burst_len, 2),
        ).astype(np.float32)
        degraded[idx] = clean[idx] + noise
    return degraded


def generate_bias(clean: np.ndarray) -> np.ndarray:
    degraded = np.empty_like(clean, dtype=np.float32)
    for idx in range(clean.shape[0]):
        rng = np.random.default_rng(42 + idx)
        bias = rng.normal(0.0, 0.15, size=(2,)).astype(np.float32)
        degraded[idx] = clean[idx] + bias[None, :]
    return degraded


def generate_combined(clean: np.ndarray) -> np.ndarray:
    degraded = np.empty_like(clean, dtype=np.float32)
    for idx in range(clean.shape[0]):
        base = clean[idx].astype(np.float32).copy()

        rng_gaussian = np.random.default_rng(42 + idx)
        gaussian_noise = rng_gaussian.normal(0.0, 0.05, size=(20, 2)).astype(np.float32)
        base = base + gaussian_noise

        rng_bias = np.random.default_rng(10042 + idx)
        bias = rng_bias.normal(0.0, 0.15, size=(2,)).astype(np.float32)
        base = base + bias[None, :]

        rng_drift = np.random.default_rng(20042 + idx)
        drift_steps = np.zeros((20, 2), dtype=np.float32)
        drift_steps[1:] = rng_drift.normal(0.0, 0.005, size=(19, 2)).astype(np.float32)
        drift = np.cumsum(drift_steps, axis=0)
        degraded[idx] = base + drift
    return degraded


def make_degraded_map(clean: np.ndarray) -> dict[str, np.ndarray]:
    return {
        "gaussian_medium": generate_gaussian(clean),
        "drift_medium": generate_drift(clean),
        "jump_medium": generate_jump(clean),
        "burst_medium": generate_burst(clean),
        "bias_medium": generate_bias(clean),
        "combined_medium": generate_combined(clean),
    }


def run_uncond_sdedit_t2_average(
    degraded_abs: np.ndarray,
    model: TemporalDenoiser1D,
    diffusion: DDPMForwardProcess,
    rel_mean: np.ndarray,
    rel_std: np.ndarray,
    device: torch.device,
) -> np.ndarray:
    preds = []
    for sample_seed in SAMPLE_SEEDS:
        preds.append(
            run_sdedit(
                degraded_abs=degraded_abs,
                model=model,
                diffusion=diffusion,
                rel_mean=rel_mean,
                rel_std=rel_std,
                t_start=2,
                sdedit_seed=sample_seed,
                device=device,
            )
        )
    return np.mean(np.stack(preds, axis=0), axis=0).astype(np.float32)


def run_cond_residual_t20_average(
    degraded_abs: np.ndarray,
    model: ConditionalTemporalDenoiser1D,
    diffusion: DDPMForwardProcess,
    rel_mean: np.ndarray,
    rel_std: np.ndarray,
    device: torch.device,
) -> np.ndarray:
    preds = []
    for sample_seed in SAMPLE_SEEDS:
        preds.append(
            sample_conditional_residual(
                degraded_abs=degraded_abs,
                model=model,
                diffusion=diffusion,
                rel_mean=rel_mean,
                rel_std=rel_std,
                start_t=20,
                sample_seed=sample_seed,
                device=device,
            )
        )
    return np.mean(np.stack(preds, axis=0), axis=0).astype(np.float32)


def determine_generalization_level(n_improved: int) -> str:
    if n_improved == 4:
        return "strong"
    if n_improved >= 2:
        return "partial"
    if n_improved == 1:
        return "weak"
    return "none"


def main() -> None:
    check_working_directory()
    ensure_required_inputs()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    clean_all = np.load(CLEAN_PATH).astype(np.float32)
    clean = clean_all[:N]
    if clean.shape != (200, 20, 2):
        raise ValueError(f"Expected clean shape (200, 20, 2), got {clean.shape}")

    norm = np.load(NORM_PATH)
    if "rel_mean" not in norm or "rel_std" not in norm:
        raise KeyError("rel_norm_params_v2.npz must contain rel_mean and rel_std")
    rel_mean = norm["rel_mean"].astype(np.float32)
    rel_std = norm["rel_std"].astype(np.float32)
    if rel_mean.shape != (2,) or rel_std.shape != (2,):
        raise ValueError(f"Expected rel_mean/rel_std shape (2,), got {rel_mean.shape} and {rel_std.shape}")

    device = torch.device("cpu")
    diffusion = DDPMForwardProcess(timesteps=TIMESTEPS, device=device)

    cond_model = ConditionalTemporalDenoiser1D(max_timesteps=TIMESTEPS, in_channels=4, hidden_dim=128).to(device)
    load_state_dict_flexible(cond_model, COND_CKPT_PATH, device)
    cond_model.eval()

    uncond_model = TemporalDenoiser1D(max_timesteps=TIMESTEPS, in_channels=2, hidden_dim=128).to(device)
    load_state_dict_flexible(uncond_model, UNCOND_CKPT_PATH, device)
    uncond_model.eval()

    degraded_map = make_degraded_map(clean)
    for degradation in DEGRADATION_ORDER:
        save_numpy(DEGRADED_PATHS[degradation], degraded_map[degradation])

    if not SCIPY_AVAILABLE:
        print("WARNING: scipy not available; Wilcoxon tests skipped")

    print("=== Step 3g-B: Generalization Diagnostic ===")
    print("Model: gaussian-only conditional residual DDPM")
    print("N = 200 trajectories per degradation")
    print("Methods: noisy_input, uncond_sdedit_t2, cond_residual_t20")
    print("")
    print(
        f"{'degradation':18s}  "
        f"{'method':20s}  "
        f"{'ADE_mean':>10s}  "
        f"{'ADE_std':>10s}  "
        f"{'smooth':>10s}  "
        f"{'delta_ADE':>11s}  "
        f"{'improved':>10s}  "
        f"{'p_noisy':>12s}"
    )

    summary_rows: list[dict] = []
    per_traj_rows: list[dict] = []
    results: dict[str, dict[str, dict[str, np.ndarray | float]]] = {}
    preds_for_plot: dict[str, dict[str, np.ndarray]] = {}

    for degradation in DEGRADATION_ORDER:
        group = DEGRADATION_GROUPS[degradation]
        degraded = degraded_map[degradation]
        preds = {
            "noisy_input": degraded,
            "uncond_sdedit_t2": run_uncond_sdedit_t2_average(
                degraded_abs=degraded,
                model=uncond_model,
                diffusion=diffusion,
                rel_mean=rel_mean,
                rel_std=rel_std,
                device=device,
            ),
            "cond_residual_t20": run_cond_residual_t20_average(
                degraded_abs=degraded,
                model=cond_model,
                diffusion=diffusion,
                rel_mean=rel_mean,
                rel_std=rel_std,
                device=device,
            ),
        }
        preds_for_plot[degradation] = preds

        metrics: dict[str, dict[str, np.ndarray]] = {}
        for method in METHOD_ORDER:
            ade, smooth = compute_metrics(preds[method], clean)
            metrics[method] = {
                "ADE": ade,
                "smooth": smooth,
            }

        results[degradation] = {}
        noisy_ade = metrics["noisy_input"]["ADE"]
        noisy_smooth = metrics["noisy_input"]["smooth"]

        for method in METHOD_ORDER:
            ade = metrics[method]["ADE"]
            smooth = metrics[method]["smooth"]
            delta_ade = ade - noisy_ade
            delta_smooth = smooth - noisy_smooth

            if method == "noisy_input":
                delta_ade_mean = 0.0
                improved_fraction = 0.0
                p_noisy = float("nan")
            else:
                delta_ade_mean = float(np.mean(delta_ade))
                improved_fraction = float(np.mean(delta_ade < 0))
                p_noisy = safe_wilcoxon(delta_ade)

            ade_mean = float(np.mean(ade))
            ade_std = float(np.std(ade, ddof=0))
            smooth_mean = float(np.mean(smooth))

            results[degradation][method] = {
                "ADE_mean": ade_mean,
                "ADE_std": ade_std,
                "smooth_mean": smooth_mean,
                "delta_ADE_vs_noisy_mean": delta_ade_mean,
                "improved_fraction": improved_fraction,
                "wilcoxon_p_vs_noisy": p_noisy,
                "ADE": ade,
                "smooth": smooth,
            }

            summary_rows.append(
                {
                    "degradation": degradation,
                    "degradation_group": group,
                    "method": method,
                    "N": N,
                    "ADE_mean": ade_mean,
                    "ADE_std": ade_std,
                    "smooth_mean": smooth_mean,
                    "delta_ADE_vs_noisy_mean": delta_ade_mean,
                    "improved_fraction": improved_fraction,
                    "wilcoxon_p_vs_noisy": p_noisy,
                }
            )

            for traj_idx in range(N):
                per_traj_rows.append(
                    {
                        "degradation": degradation,
                        "degradation_group": group,
                        "traj_idx": traj_idx,
                        "method": method,
                        "ADE": float(ade[traj_idx]),
                        "smooth": float(smooth[traj_idx]),
                        "noisy_ADE": float(noisy_ade[traj_idx]),
                        "noisy_smooth": float(noisy_smooth[traj_idx]),
                        "delta_ADE_vs_noisy": float(0.0 if method == "noisy_input" else delta_ade[traj_idx]),
                        "delta_smooth_vs_noisy": float(0.0 if method == "noisy_input" else delta_smooth[traj_idx]),
                        "improved_ADE_vs_noisy": int(False if method == "noisy_input" else delta_ade[traj_idx] < 0),
                        "improved_smooth_vs_noisy": int(False if method == "noisy_input" else delta_smooth[traj_idx] < 0),
                    }
                )

            print(
                f"{degradation:18s}  "
                f"{method:20s}  "
                f"{ade_mean:10.4f}  "
                f"{ade_std:10.4f}  "
                f"{smooth_mean:10.4f}  "
                f"{delta_ade_mean:+11.4f}  "
                f"{improved_fraction * 100:9.1f}%  "
                f"{format_pvalue_table(p_noisy):>12s}"
            )

    save_csv(
        SUMMARY_PATH,
        summary_rows,
        [
            "degradation",
            "degradation_group",
            "method",
            "N",
            "ADE_mean",
            "ADE_std",
            "smooth_mean",
            "delta_ADE_vs_noisy_mean",
            "improved_fraction",
            "wilcoxon_p_vs_noisy",
        ],
    )
    save_csv(
        PER_TRAJ_PATH,
        per_traj_rows,
        [
            "degradation",
            "degradation_group",
            "traj_idx",
            "method",
            "ADE",
            "smooth",
            "noisy_ADE",
            "noisy_smooth",
            "delta_ADE_vs_noisy",
            "delta_smooth_vs_noisy",
            "improved_ADE_vs_noisy",
            "improved_smooth_vs_noisy",
        ],
    )

    print("=== 分组诊断 ===")
    print("")
    print("--- relative-observable 退化（预期可能有改善）---")
    supported_map = {}
    for degradation in RELATIVE_OBSERVABLE:
        noisy_ade = float(results[degradation]["noisy_input"]["ADE_mean"])
        cond_ade = float(results[degradation]["cond_residual_t20"]["ADE_mean"])
        delta = float(results[degradation]["cond_residual_t20"]["delta_ADE_vs_noisy_mean"])
        p_value = float(results[degradation]["cond_residual_t20"]["wilcoxon_p_vs_noisy"])
        supported = cond_ade < noisy_ade and delta < 0 and not np.isnan(p_value) and p_value < 0.05
        supported_map[degradation] = supported

        if supported:
            print(
                f"  {degradation}: ✅ cond_residual_t20 有统计支持的 ADE 改善 "
                f"{-delta:.4f} m ({-delta / noisy_ade * 100:.1f}%)"
            )
        elif cond_ade < noisy_ade and delta < 0 and (np.isnan(p_value) or p_value >= 0.05):
            print(
                f"  {degradation}: ⚠️ cond_residual_t20 均值改善但统计支持不足 "
                f"delta={delta:+.4f}, p={format_pvalue_short(p_value)}"
            )
        else:
            print(
                f"  {degradation}: ❌ cond_residual_t20 未改善 ADE "
                f"(delta={delta:+.4f}, p={format_pvalue_short(p_value)})"
            )

    print("")
    print("--- partially-observable 退化（combined = gaussian + bias + drift）---")
    combined_cond_ade = float(results["combined_medium"]["cond_residual_t20"]["ADE_mean"])
    combined_noisy_ade = float(results["combined_medium"]["noisy_input"]["ADE_mean"])
    combined_delta = float(results["combined_medium"]["cond_residual_t20"]["delta_ADE_vs_noisy_mean"])
    combined_p = float(results["combined_medium"]["cond_residual_t20"]["wilcoxon_p_vs_noisy"])
    print(
        f"  combined_medium: cond_residual_t20 ADE={combined_cond_ade:.4f}, "
        f"noisy={combined_noisy_ade:.4f}, delta={combined_delta:+.4f}, p={format_pvalue_short(combined_p)}"
    )
    if combined_delta < 0 and not np.isnan(combined_p) and combined_p < 0.05:
        print("         → 部分改善，符合 partially-observable 预期：模型可能修正 gaussian/drift 成分，但不能解释为修正了 bias。")
        combined_observation = "partial_observable_improvement"
    elif abs(combined_delta) < 0.005:
        print("         → 几乎无变化，说明 bias 或混合误差可能限制了 relative-residual 模型。")
        combined_observation = "no_effect"
    elif combined_delta > 0:
        print("         → 恶化，说明 gaussian-only 模型对 mixed corruption 产生干扰。")
        combined_observation = "interference"
    else:
        print("         → 均值略有改善但统计支持不足，需要 mixed training 验证。")
        combined_observation = "no_effect"

    print("")
    print("--- relative-unobservable 退化（bias 在 rel 空间不可观测）---")
    bias_cond_ade = float(results["bias_medium"]["cond_residual_t20"]["ADE_mean"])
    bias_noisy_ade = float(results["bias_medium"]["noisy_input"]["ADE_mean"])
    bias_delta = float(results["bias_medium"]["cond_residual_t20"]["delta_ADE_vs_noisy_mean"])
    bias_p = float(results["bias_medium"]["cond_residual_t20"]["wilcoxon_p_vs_noisy"])
    print(
        f"  bias_medium: cond_residual_t20 ADE={bias_cond_ade:.4f}, "
        f"noisy={bias_noisy_ade:.4f}, delta={bias_delta:+.4f}, p={format_pvalue_short(bias_p)}"
    )
    if abs(bias_delta) < 0.005:
        print("         → 几乎无变化，符合预期：constant bias 在 relative displacement 空间不可观测。")
        bias_observation = "unobservable"
    elif bias_delta > 0:
        print("         → 恶化，说明模型对不可观测 bias 产生了不必要的 relative correction。")
        bias_observation = "interference"
    else:
        print("         → 意外改善，需要进一步检查是否由非 bias 误差、起点处理或随机效应造成。")
        bias_observation = "unexpected_improvement"

    print("")
    print("=== 泛化结论 ===")
    n_improved = int(sum(supported_map.values()))
    generalization_level = determine_generalization_level(n_improved)
    if n_improved == 4:
        print("✅ 强泛化：gaussian-only 训练的模型在所有 4 种 relative-observable 退化上都有统计支持的改善")
        print("→ 模型学到了通用 correction prior，不只是 gaussian-specific")
    elif n_improved >= 2:
        print(f"⚠️ 部分泛化：{n_improved}/4 种 relative-observable 退化有统计支持的改善")
        print("→ 模型有一定泛化能力，但 mixed relative-degradation training 可能进一步提升")
    elif n_improved == 1:
        print("⚠️ 弱泛化：仅少数 relative-observable 退化显示统计支持的改善")
        print("→ 当前 gaussian-only 模型更接近 degradation-specific，需要 mixed training")
    else:
        print("❌ 无明显泛化：在 4 种 relative-observable 退化上均未观察到统计支持的改善")
        print("→ 当前 gaussian-only 训练未显示通用 correction prior")

    print("")
    print("Representation limitation:")
    print("  bias_medium is relative-unobservable under the current relative-residual formulation.")
    print("  combined_medium is partially observable because gaussian/drift components affect relative displacement but bias does not.")

    per_degradation_delta = {
        degradation: float(results[degradation]["cond_residual_t20"]["delta_ADE_vs_noisy_mean"])
        for degradation in DEGRADATION_ORDER
    }
    per_degradation_p_value = {
        degradation: float(results[degradation]["cond_residual_t20"]["wilcoxon_p_vs_noisy"])
        for degradation in DEGRADATION_ORDER
    }

    interpretation = (
        f"The gaussian-only conditional residual DDPM shows {generalization_level} generalization: "
        f"{n_improved}/4 relative-observable degradations have paired statistical support for ADE improvement. "
        f"Bias behavior should be read as a representation limitation because constant bias is cancelled in relative displacement, "
        f"while any combined_medium gain can only be attributed to the observable gaussian/drift components rather than bias correction."
    )
    conclusion = {
        "N": 200,
        "methods": ["noisy_input", "uncond_sdedit_t2", "cond_residual_t20"],
        "relative_observable": ["gaussian_medium", "drift_medium", "jump_medium", "burst_medium"],
        "partially_observable": ["combined_medium"],
        "relative_unobservable": ["bias_medium"],
        "n_relative_observable_improved": n_improved,
        "generalization_level": generalization_level,
        "per_degradation_delta": per_degradation_delta,
        "per_degradation_p_value": per_degradation_p_value,
        "bias_observation": bias_observation,
        "combined_observation": combined_observation,
        "interpretation": interpretation,
    }
    save_json(CONCLUSION_PATH, conclusion)

    short_deg_labels = ["gaussian", "drift", "jump", "burst", "bias", "combined"]
    x = np.arange(len(DEGRADATION_ORDER))
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    axes[0, 0].axvspan(-0.5, 3.5, color="#d9edf7", alpha=0.35)
    axes[0, 0].axvspan(3.5, 4.5, color="#f2dede", alpha=0.35)
    axes[0, 0].axvspan(4.5, 5.5, color="#fcf8e3", alpha=0.35)
    width = 0.24
    colors = {
        "noisy_input": "#9e9e9e",
        "uncond_sdedit_t2": "#4c78a8",
        "cond_residual_t20": "#d62728",
    }
    for idx, method in enumerate(METHOD_ORDER):
        ade_means = [float(results[deg][method]["ADE_mean"]) for deg in DEGRADATION_ORDER]
        axes[0, 0].bar(x + (idx - 1) * width, ade_means, width=width, color=colors[method], label=method)
    axes[0, 0].set_xticks(x)
    axes[0, 0].set_xticklabels(short_deg_labels, rotation=20)
    axes[0, 0].set_title("ADE across degradation types")
    axes[0, 0].set_ylabel("ADE_mean")
    axes[0, 0].legend(fontsize=8)

    heatmap_methods = ["uncond_sdedit_t2", "cond_residual_t20"]
    heatmap_data = np.array(
        [
            [float(results[deg][method]["delta_ADE_vs_noisy_mean"]) for deg in DEGRADATION_ORDER]
            for method in heatmap_methods
        ],
        dtype=np.float32,
    )
    max_abs = float(np.max(np.abs(heatmap_data)))
    norm = TwoSlopeNorm(vmin=-max_abs, vcenter=0.0, vmax=max_abs) if max_abs > 0 else None
    im = axes[0, 1].imshow(heatmap_data, cmap="coolwarm", norm=norm, aspect="auto")
    axes[0, 1].set_xticks(np.arange(len(DEGRADATION_ORDER)))
    axes[0, 1].set_xticklabels(short_deg_labels, rotation=20)
    axes[0, 1].set_yticks(np.arange(len(heatmap_methods)))
    axes[0, 1].set_yticklabels(heatmap_methods)
    axes[0, 1].set_title("ΔADE vs noisy_input")
    for row_idx in range(heatmap_data.shape[0]):
        for col_idx in range(heatmap_data.shape[1]):
            axes[0, 1].text(
                col_idx,
                row_idx,
                f"{heatmap_data[row_idx, col_idx]:.4f}",
                ha="center",
                va="center",
                fontsize=8,
                color="black",
            )
    fig.colorbar(im, ax=axes[0, 1], fraction=0.046, pad=0.04)

    for idx, method in enumerate(heatmap_methods):
        improved_vals = [float(results[deg][method]["improved_fraction"]) for deg in DEGRADATION_ORDER]
        axes[0, 2].bar(x + (idx - 0.5) * 0.32, improved_vals, width=0.32, color=colors[method], label=method)
    axes[0, 2].axhline(0.5, color="black", linestyle="--", linewidth=1.0)
    axes[0, 2].set_xticks(x)
    axes[0, 2].set_xticklabels(short_deg_labels, rotation=20)
    axes[0, 2].set_ylim(0.0, 1.0)
    axes[0, 2].set_title("Improved fraction")
    axes[0, 2].legend(fontsize=8)

    for ax, degradation, title_prefix in [
        (axes[1, 0], "gaussian_medium", "gaussian_medium, idx=0"),
        (axes[1, 1], "burst_medium", "burst_medium, idx=0"),
        (axes[1, 2], "bias_medium", "bias_medium, idx=0"),
    ]:
        idx0 = 0
        clean_traj = clean[idx0]
        degraded_traj = degraded_map[degradation][idx0]
        cond_traj = preds_for_plot[degradation]["cond_residual_t20"][idx0]
        noisy_ade = float(results[degradation]["noisy_input"]["ADE"][idx0])
        cond_ade = float(results[degradation]["cond_residual_t20"]["ADE"][idx0])
        ax.plot(clean_traj[:, 0], clean_traj[:, 1], color="#1f77b4", linewidth=2.0, label="clean")
        ax.plot(degraded_traj[:, 0], degraded_traj[:, 1], color="#7f7f7f", linewidth=1.7, linestyle="--", label="degraded")
        ax.plot(cond_traj[:, 0], cond_traj[:, 1], color="#d62728", linewidth=1.8, label="cond_residual_t20")
        ax.set_title(f"{title_prefix}\nnoisy ADE={noisy_ade:.4f}, cond ADE={cond_ade:.4f}")
        ax.set_xlim(-0.5, 3.5)
        ax.set_ylim(-0.5, 3.5)
        ax.set_aspect("equal", adjustable="box")
        ax.legend(fontsize=8)

    plt.tight_layout()
    fig.savefig(FIG_PATH, dpi=180)
    plt.close(fig)
    print(f"Saved: {FIG_PATH}")


if __name__ == "__main__":
    main()
