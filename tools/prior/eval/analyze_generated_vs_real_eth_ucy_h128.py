from pathlib import Path
import argparse

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[3]

from utils.prior.ablation_paths import (
    get_eval_ratios_by_name,
    get_paths_by_name,
    get_train_record_by_name,
    resolve_variant_or_objective,
    to_abs_path,
)

MOVING_THRESHOLD_QUANTILE = 0.10
MOVING_POSITIVE_EPS = 1e-8
EPS = 1e-8


def load_data(real_path, gen_path, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)

    if not real_path.exists():
        raise FileNotFoundError(f"找不到真实数据: {real_path}")
    if not gen_path.exists():
        raise FileNotFoundError(f"找不到生成数据: {gen_path}")

    real = np.load(real_path).astype(np.float32)
    gen = np.load(gen_path).astype(np.float32)

    if real.ndim != 3:
        raise ValueError(f"真实数据维度不对: {real.shape}")
    if gen.ndim != 3:
        raise ValueError(f"生成数据维度不对: {gen.shape}")

    if gen.shape[1] == 2 and gen.shape[2] == 19:
        gen = np.transpose(gen, (0, 2, 1))
    elif gen.shape[1] == 19 and gen.shape[2] == 2:
        pass
    else:
        raise ValueError(f"无法识别生成数据 shape: {gen.shape}")

    if real.shape[1] != 19 or real.shape[2] != 2:
        raise ValueError(f"真实数据 shape 异常，期望 [N,19,2]，实际为: {real.shape}")

    return real, gen


def compute_step_norms(rel):
    return np.linalg.norm(rel, axis=-1)


def compute_global_moving_threshold(real_rel, q=MOVING_THRESHOLD_QUANTILE, positive_eps=MOVING_POSITIVE_EPS):
    real_step_norm = compute_step_norms(real_rel).reshape(-1)
    positive_step_norm = real_step_norm[real_step_norm > positive_eps]

    if positive_step_norm.size == 0:
        raise ValueError(f"真实数据中没有大于 {positive_eps} 的 step_norm，无法计算 moving threshold")

    tau = np.quantile(positive_step_norm, q)
    print(f"Positive real step_norm count (> {positive_eps}): {positive_step_norm.size} / {real_step_norm.size}")
    return float(tau)


def compute_metrics(rel, moving_threshold, eps=EPS):
    step_norm = compute_step_norms(rel)
    total_length = step_norm.sum(axis=1)
    avg_speed = step_norm.mean(axis=1)
    endpoint_vec = rel.sum(axis=1)
    endpoint_displacement = np.linalg.norm(endpoint_vec, axis=1)
    moving_ratio_global = (step_norm > moving_threshold).mean(axis=1)
    propulsion_ratio = endpoint_displacement / (total_length + eps)

    if rel.shape[1] >= 2:
        acc = rel[:, 1:, :] - rel[:, :-1, :]
        acc_norm = np.linalg.norm(acc, axis=-1)
        acc_rms = np.sqrt(np.mean(acc_norm ** 2, axis=1))
    else:
        acc_rms = np.zeros(rel.shape[0], dtype=np.float32)

    return {
        "step_norm_all": step_norm.reshape(-1),
        "avg_speed": avg_speed,
        "total_length": total_length,
        "endpoint_displacement": endpoint_displacement,
        "moving_ratio_global": moving_ratio_global,
        "propulsion_ratio": propulsion_ratio,
        "acc_rms": acc_rms,
        "endpoint_vec": endpoint_vec,
    }


def summarize_one_metric(real_values, gen_values, metric_name):
    real_values = np.asarray(real_values).reshape(-1)
    gen_values = np.asarray(gen_values).reshape(-1)

    real_mean = float(np.mean(real_values))
    gen_mean = float(np.mean(gen_values))
    real_std = float(np.std(real_values))
    gen_std = float(np.std(gen_values))
    real_median = float(np.median(real_values))
    gen_median = float(np.median(gen_values))
    real_q10 = float(np.quantile(real_values, 0.10))
    gen_q10 = float(np.quantile(gen_values, 0.10))
    real_q90 = float(np.quantile(real_values, 0.90))
    gen_q90 = float(np.quantile(gen_values, 0.90))

    return {
        "metric": metric_name,
        "real_mean": real_mean,
        "gen_mean": gen_mean,
        "real_std": real_std,
        "gen_std": gen_std,
        "real_median": real_median,
        "gen_median": gen_median,
        "real_q10": real_q10,
        "gen_q10": gen_q10,
        "real_q90": real_q90,
        "gen_q90": gen_q90,
        "mean_ratio_gen_over_real": gen_mean / (real_mean + EPS),
        "std_ratio_gen_over_real": gen_std / (real_std + EPS),
    }


def build_summary_table(real_metrics, gen_metrics):
    metric_order = [
        "step_norm_all",
        "avg_speed",
        "total_length",
        "endpoint_displacement",
        "moving_ratio_global",
        "propulsion_ratio",
        "acc_rms",
    ]
    rows = []
    for name in metric_order:
        rows.append(summarize_one_metric(real_metrics[name], gen_metrics[name], name))
    return pd.DataFrame(rows)


def plot_hist(real_values, gen_values, title, xlabel, save_path, bins=50):
    plt.figure(figsize=(6, 4))
    plt.hist(real_values, bins=bins, alpha=0.6, label="Real", density=True)
    plt.hist(gen_values, bins=bins, alpha=0.6, label="Generated", density=True)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("Density")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()


def plot_endpoint_scatter(real_endpoint_vec, gen_endpoint_vec, save_path):
    plt.figure(figsize=(6, 6))
    plt.scatter(real_endpoint_vec[:, 0], real_endpoint_vec[:, 1], s=10, alpha=0.4, label="Real")
    plt.scatter(gen_endpoint_vec[:, 0], gen_endpoint_vec[:, 1], s=10, alpha=0.5, label="Generated")
    plt.title("Endpoint Scatter")
    plt.xlabel("Endpoint X")
    plt.ylabel("Endpoint Y")
    plt.legend()
    plt.axis("equal")
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()


def plot_speed_vs_length(real_avg_speed, real_total_length, gen_avg_speed, gen_total_length, save_path):
    plt.figure(figsize=(6, 5))
    plt.scatter(real_avg_speed, real_total_length, s=10, alpha=0.4, label="Real")
    plt.scatter(gen_avg_speed, gen_total_length, s=10, alpha=0.5, label="Generated")
    plt.title("Avg Speed vs Total Length")
    plt.xlabel("Average Speed")
    plt.ylabel("Total Length")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", type=str, default="motion_balanced")
    parser.add_argument("--train_seed", type=int, default=42)
    parser.add_argument("--train_epochs", type=int, default=100)
    parser.add_argument("--num_generate", type=int, default=512)
    parser.add_argument("--generated_rel_path", type=str, default=None)
    parser.add_argument("--reference_tag", type=str, default="reference_seed42")
    parser.add_argument("--save_manifest", action="store_true")
    args = parser.parse_args()

    resolved_variant = resolve_variant_or_objective(args.variant)
    cfg = get_paths_by_name(args.variant)
    train_record = get_train_record_by_name(args.variant)
    eval_ratios = get_eval_ratios_by_name(args.variant)
    run_tag = f"seed{args.train_seed}-{args.train_epochs}epoch"

    real_path = to_abs_path(cfg["rel_path"])
    if args.generated_rel_path is not None:
        gen_path = to_abs_path(args.generated_rel_path)
    else:
        gen_path = (
            PROJECT_ROOT
            / "outputs"
            / "prior"
            / "sample"
            / cfg["sample_tag"]
            / run_tag
            / args.reference_tag
            / "generated_rel_samples.npy"
        )
    out_dir = (
        PROJECT_ROOT
        / "outputs"
        / "prior"
        / "eval"
        / cfg["eval_tag"]
        / run_tag
        / args.reference_tag
    )

    real, gen = load_data(real_path, gen_path, out_dir)

    print(f"input_name       = {args.variant}")
    print(f"resolved_variant = {resolved_variant}")
    print(f"real shape = {real.shape}")
    print(f"gen  shape = {gen.shape}")
    print(f"real_path = {real_path}")
    print(f"gen_path  = {gen_path}")
    print(f"out_dir   = {out_dir}")
    print(f"train_seed = {args.train_seed}")
    print(f"train_epochs = {args.train_epochs}")
    print(f"run_tag = {run_tag}")
    print(f"reference_tag = {args.reference_tag}")
    print(f"train_record = {train_record}")
    print(f"eval_ratios  = {eval_ratios}")

    moving_threshold = compute_global_moving_threshold(real, q=MOVING_THRESHOLD_QUANTILE, positive_eps=MOVING_POSITIVE_EPS)
    print(f"\nGlobal moving threshold (positive-only real step_norm q{int(MOVING_THRESHOLD_QUANTILE * 100)}): {moving_threshold:.6f}")

    real_metrics = compute_metrics(real, moving_threshold)
    gen_metrics = compute_metrics(gen, moving_threshold)
    summary_df = build_summary_table(real_metrics, gen_metrics)

    print("\n===== Summary Table =====")
    print(summary_df.to_string(index=False))

    summary_df.to_csv(out_dir / "summary_metrics.csv", index=False, float_format="%.6f")
    if args.save_manifest:
        with open(out_dir / "manifest.json", "w", encoding="utf-8") as f:
            import json

            json.dump(
                {
                    "input_name": args.variant,
                    "resolved_variant": resolved_variant,
                    "generated_rel_path": str(gen_path),
                    "rel_path": str(real_path),
                    "train_seed": args.train_seed,
                    "train_epochs": args.train_epochs,
                    "run_tag": run_tag,
                    "reference_tag": args.reference_tag,
                },
                f,
                indent=2,
                ensure_ascii=False,
            )
    with open(out_dir / "analysis_config.txt", "w", encoding="utf-8") as f:
        f.write(f"INPUT_NAME={args.variant}\n")
        f.write(f"RESOLVED_VARIANT={resolved_variant}\n")
        f.write(f"REAL_PATH={real_path}\n")
        f.write(f"GEN_PATH={gen_path}\n")
        f.write("MOVING_THRESHOLD_MODE=positive_only_quantile\n")
        f.write(f"MOVING_THRESHOLD_QUANTILE={MOVING_THRESHOLD_QUANTILE}\n")
        f.write(f"MOVING_POSITIVE_EPS={MOVING_POSITIVE_EPS}\n")
        f.write(f"GLOBAL_MOVING_THRESHOLD={moving_threshold:.6f}\n")

    plot_hist(real_metrics["step_norm_all"], gen_metrics["step_norm_all"], "Step Norm Distribution", "Step Norm", out_dir / "hist_step_norm.png")
    plot_hist(real_metrics["avg_speed"], gen_metrics["avg_speed"], "Average Speed Distribution", "Average Speed", out_dir / "hist_avg_speed.png")
    plot_hist(real_metrics["total_length"], gen_metrics["total_length"], "Total Length Distribution", "Total Length", out_dir / "hist_total_length.png")
    plot_hist(real_metrics["endpoint_displacement"], gen_metrics["endpoint_displacement"], "Endpoint Displacement Distribution", "Endpoint Displacement", out_dir / "hist_endpoint_displacement.png")
    plot_hist(real_metrics["moving_ratio_global"], gen_metrics["moving_ratio_global"], "Moving Ratio Global Distribution", "Moving Ratio", out_dir / "hist_moving_ratio_global.png")
    plot_hist(real_metrics["propulsion_ratio"], gen_metrics["propulsion_ratio"], "Propulsion Ratio Distribution", "Propulsion Ratio", out_dir / "hist_propulsion_ratio.png")
    plot_hist(real_metrics["acc_rms"], gen_metrics["acc_rms"], "Acceleration RMS Distribution", "Acc RMS", out_dir / "hist_acc_rms.png")

    plot_endpoint_scatter(real_metrics["endpoint_vec"], gen_metrics["endpoint_vec"], out_dir / "scatter_endpoint.png")
    plot_speed_vs_length(real_metrics["avg_speed"], real_metrics["total_length"], gen_metrics["avg_speed"], gen_metrics["total_length"], out_dir / "scatter_speed_vs_length.png")

    print(f"\nSaved analysis to: {out_dir}")


if __name__ == "__main__":
    main()
