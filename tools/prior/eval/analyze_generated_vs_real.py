from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# =============================
# 配置区
# =============================
PROJECT_ROOT = Path(__file__).resolve().parents[3]

REAL_PATH = PROJECT_ROOT / "datasets" / "processed" / "data_eth_20_rel_q20.npy"
GEN_PATH = PROJECT_ROOT / "outputs" / "prior" / "sample" / "ddpm_minimal_q20" / "reverse_sampling_check_512" / "generated_rel_samples.npy"
OUT_DIR = PROJECT_ROOT / "outputs" / "prior" / "eval" / "ddpm_minimal_q20" / "distribution_analysis_512_v3metrics"

# moving_ratio_global 的阈值：
# 在“真实数据中所有正的 step_norm”上取 q10
MOVING_THRESHOLD_QUANTILE = 0.10
MOVING_POSITIVE_EPS = 1e-8

EPS = 1e-8


# =============================
# 数据读取
# =============================
def load_data():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    if not REAL_PATH.exists():
        raise FileNotFoundError(f"找不到真实数据: {REAL_PATH}")
    if not GEN_PATH.exists():
        raise FileNotFoundError(f"找不到生成数据: {GEN_PATH}")

    real = np.load(REAL_PATH).astype(np.float32)   # 期望 [N, 19, 2]
    gen = np.load(GEN_PATH).astype(np.float32)     # 可能是 [N, 2, 19] 或 [N, 19, 2]

    if real.ndim != 3:
        raise ValueError(f"真实数据维度不对: {real.shape}")
    if gen.ndim != 3:
        raise ValueError(f"生成数据维度不对: {gen.shape}")

    # 自动统一 gen 为 [N, 19, 2]
    if gen.shape[1] == 2 and gen.shape[2] == 19:
        gen = np.transpose(gen, (0, 2, 1))
    elif gen.shape[1] == 19 and gen.shape[2] == 2:
        pass
    else:
        raise ValueError(f"无法识别生成数据 shape: {gen.shape}")

    # 校验 real
    if real.shape[1] != 19 or real.shape[2] != 2:
        raise ValueError(f"真实数据 shape 异常，期望 [N,19,2]，实际为: {real.shape}")

    return real, gen


# =============================
# 指标计算
# =============================
def compute_step_norms(rel):
    """
    rel: [N, T, 2]
    return: [N, T]
    """
    return np.linalg.norm(rel, axis=-1)


def compute_global_moving_threshold(
    real_rel,
    q=MOVING_THRESHOLD_QUANTILE,
    positive_eps=MOVING_POSITIVE_EPS,
):
    """
    只在真实数据中“正的 step_norm”上取分位点，
    避免全局阈值被大量 0 压成 0。
    """
    real_step_norm = compute_step_norms(real_rel).reshape(-1)
    positive_step_norm = real_step_norm[real_step_norm > positive_eps]

    if positive_step_norm.size == 0:
        raise ValueError(
            f"真实数据中没有大于 {positive_eps} 的 step_norm，无法计算 moving threshold"
        )

    tau = np.quantile(positive_step_norm, q)

    print(
        f"Positive real step_norm count (> {positive_eps}): "
        f"{positive_step_norm.size} / {real_step_norm.size}"
    )

    return float(tau)


def compute_metrics(rel, moving_threshold, eps=EPS):
    """
    rel: [N, T, 2]
    """
    step_norm = compute_step_norms(rel)                              # [N, T]
    total_length = step_norm.sum(axis=1)                            # [N]
    avg_speed = step_norm.mean(axis=1)                              # [N]
    endpoint_vec = rel.sum(axis=1)                                  # [N, 2]
    endpoint_displacement = np.linalg.norm(endpoint_vec, axis=1)    # [N]

    # 新 moving_ratio：基于全局固定阈值
    moving_ratio_global = (step_norm > moving_threshold).mean(axis=1)

    # 新增 propulsion_ratio：有效推进比
    propulsion_ratio = endpoint_displacement / (total_length + eps)

    # 可选补充：加速度变化强度（平滑性粗代理）
    if rel.shape[1] >= 2:
        acc = rel[:, 1:, :] - rel[:, :-1, :]                        # [N, T-1, 2]
        acc_norm = np.linalg.norm(acc, axis=-1)                     # [N, T-1]
        acc_rms = np.sqrt(np.mean(acc_norm ** 2, axis=1))           # [N]
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
        rows.append(
            summarize_one_metric(
                real_metrics[name],
                gen_metrics[name],
                name,
            )
        )
    return pd.DataFrame(rows)


# =============================
# 画图
# =============================
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
    plt.scatter(real_endpoint_vec[:, 0], real_endpoint_vec[:, 1], s=10, alpha=0.5, label="Real")
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
    plt.scatter(real_avg_speed, real_total_length, s=10, alpha=0.5, label="Real")
    plt.scatter(gen_avg_speed, gen_total_length, s=10, alpha=0.5, label="Generated")
    plt.title("Avg Speed vs Total Length")
    plt.xlabel("Average Speed")
    plt.ylabel("Total Length")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()


# =============================
# 主函数
# =============================
def main():
    real, gen = load_data()

    print(f"real shape = {real.shape}")
    print(f"gen  shape = {gen.shape}")

    moving_threshold = compute_global_moving_threshold(
        real,
        q=MOVING_THRESHOLD_QUANTILE,
        positive_eps=MOVING_POSITIVE_EPS,
    )

    print(
        f"\nGlobal moving threshold "
        f"(positive-only real step_norm q{int(MOVING_THRESHOLD_QUANTILE * 100)}): "
        f"{moving_threshold:.6f}"
    )

    real_metrics = compute_metrics(real, moving_threshold)
    gen_metrics = compute_metrics(gen, moving_threshold)

    summary_df = build_summary_table(real_metrics, gen_metrics)

    print("\n===== Summary Table =====")
    print(summary_df.to_string(index=False))

    summary_csv_path = OUT_DIR / "summary_metrics.csv"
    summary_df.to_csv(summary_csv_path, index=False, float_format="%.6f")

    config_txt_path = OUT_DIR / "analysis_config.txt"
    with open(config_txt_path, "w", encoding="utf-8") as f:
        f.write(f"REAL_PATH={REAL_PATH}\n")
        f.write(f"GEN_PATH={GEN_PATH}\n")
        f.write("MOVING_THRESHOLD_MODE=positive_only_quantile\n")
        f.write(f"MOVING_THRESHOLD_QUANTILE={MOVING_THRESHOLD_QUANTILE}\n")
        f.write(f"MOVING_POSITIVE_EPS={MOVING_POSITIVE_EPS}\n")
        f.write(f"GLOBAL_MOVING_THRESHOLD={moving_threshold:.6f}\n")

    # 直方图
    plot_hist(
        real_metrics["step_norm_all"],
        gen_metrics["step_norm_all"],
        title="Step Norm Distribution",
        xlabel="Step Norm",
        save_path=OUT_DIR / "hist_step_norm.png",
    )

    plot_hist(
        real_metrics["avg_speed"],
        gen_metrics["avg_speed"],
        title="Average Speed Distribution",
        xlabel="Average Speed",
        save_path=OUT_DIR / "hist_avg_speed.png",
    )

    plot_hist(
        real_metrics["total_length"],
        gen_metrics["total_length"],
        title="Total Length Distribution",
        xlabel="Total Length",
        save_path=OUT_DIR / "hist_total_length.png",
    )

    plot_hist(
        real_metrics["endpoint_displacement"],
        gen_metrics["endpoint_displacement"],
        title="Endpoint Displacement Distribution",
        xlabel="Endpoint Displacement",
        save_path=OUT_DIR / "hist_endpoint_displacement.png",
    )

    plot_hist(
        real_metrics["moving_ratio_global"],
        gen_metrics["moving_ratio_global"],
        title="Moving Ratio (Global Threshold) Distribution",
        xlabel="Moving Ratio Global",
        save_path=OUT_DIR / "hist_moving_ratio_global.png",
    )

    plot_hist(
        real_metrics["propulsion_ratio"],
        gen_metrics["propulsion_ratio"],
        title="Propulsion Ratio Distribution",
        xlabel="Propulsion Ratio",
        save_path=OUT_DIR / "hist_propulsion_ratio.png",
    )

    plot_hist(
        real_metrics["acc_rms"],
        gen_metrics["acc_rms"],
        title="Acceleration RMS Distribution",
        xlabel="Acc RMS",
        save_path=OUT_DIR / "hist_acc_rms.png",
    )

    # 散点图
    plot_endpoint_scatter(
        real_metrics["endpoint_vec"],
        gen_metrics["endpoint_vec"],
        save_path=OUT_DIR / "scatter_endpoint.png",
    )

    plot_speed_vs_length(
        real_metrics["avg_speed"],
        real_metrics["total_length"],
        gen_metrics["avg_speed"],
        gen_metrics["total_length"],
        save_path=OUT_DIR / "scatter_speed_vs_length.png",
    )

    print(f"\n已保存统计表: {summary_csv_path}")
    print(f"已保存配置: {config_txt_path}")

    print("\n已保存图像:")
    print(OUT_DIR / "hist_step_norm.png")
    print(OUT_DIR / "hist_avg_speed.png")
    print(OUT_DIR / "hist_total_length.png")
    print(OUT_DIR / "hist_endpoint_displacement.png")
    print(OUT_DIR / "hist_moving_ratio_global.png")
    print(OUT_DIR / "hist_propulsion_ratio.png")
    print(OUT_DIR / "hist_acc_rms.png")
    print(OUT_DIR / "scatter_endpoint.png")
    print(OUT_DIR / "scatter_speed_vs_length.png")

    print("\n分析完成。")


if __name__ == "__main__":
    main()