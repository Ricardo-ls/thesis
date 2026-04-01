from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = PROJECT_ROOT / "datasets" / "processed"
OUT_DIR = PROJECT_ROOT / "outputs" / "prior" / "data" / "eth_ucy_data_analysis_q20"

ABS_PATH = DATA_DIR / "data_eth_ucy_20.npy"
REL_PATH = DATA_DIR / "data_eth_ucy_20_rel.npy"

FILTER_QUANTILE = 0.20
EPS = 1e-8


def compute_step_norm(rel):
    # rel: [N, 19, 2]
    return np.linalg.norm(rel, axis=-1)  # [N, 19]


def compute_metrics(rel):
    """
    rel: [N, 19, 2]
    """
    step_norm = compute_step_norm(rel)                      # [N, 19]
    avg_speed = step_norm.mean(axis=1)                      # [N]

    # 这里仍保留你前期一直在用的“旧 moving_ratio”，
    # 因为这一阶段主要目的是复现 baseline 的过滤逻辑
    moving_threshold_each = avg_speed[:, None] * 0.1
    moving_ratio = (step_norm > moving_threshold_each).mean(axis=1)

    if rel.shape[1] >= 2:
        acc = rel[:, 1:, :] - rel[:, :-1, :]                # [N, 18, 2]
        acc_norm = np.linalg.norm(acc, axis=-1)             # [N, 18]
        acc_rms = np.sqrt(np.mean(acc_norm ** 2, axis=1))   # [N]
    else:
        acc_rms = np.zeros(rel.shape[0], dtype=np.float32)

    return {
        "step_norm": step_norm,
        "avg_speed": avg_speed,
        "moving_ratio": moving_ratio,
        "acc_rms": acc_rms,
    }


def summarize_metrics(metrics):
    return {
        "sample_count": int(len(metrics["avg_speed"])),
        "avg_speed_mean": float(np.mean(metrics["avg_speed"])),
        "avg_speed_median": float(np.median(metrics["avg_speed"])),
        "moving_ratio_mean": float(np.mean(metrics["moving_ratio"])),
        "moving_ratio_median": float(np.median(metrics["moving_ratio"])),
        "acc_rms_mean": float(np.mean(metrics["acc_rms"])),
        "acc_rms_median": float(np.median(metrics["acc_rms"])),
    }


def plot_hist_all_step_speed(step_norm, save_path):
    plt.figure(figsize=(6, 4))
    plt.hist(step_norm.reshape(-1), bins=80, alpha=0.8)
    plt.title("All Step-wise Speed Distribution (ETH+UCY)")
    plt.xlabel("Step Norm")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()


def plot_compare_avg_speed(before_avg, after_avg, threshold, save_path):
    plt.figure(figsize=(6, 4))
    plt.hist(before_avg, bins=80, alpha=0.6, label="Before")
    plt.hist(after_avg, bins=80, alpha=0.6, label="After q20")
    plt.axvline(threshold, linestyle="--", label=f"threshold={threshold:.6f}")
    plt.title("Avg Speed Before vs After q20 Filtering (ETH+UCY)")
    plt.xlabel("Average Speed")
    plt.ylabel("Count")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    if not ABS_PATH.exists():
        raise FileNotFoundError(f"找不到绝对坐标数据: {ABS_PATH}")
    if not REL_PATH.exists():
        raise FileNotFoundError(f"找不到相对位移数据: {REL_PATH}")

    abs_data = np.load(ABS_PATH).astype(np.float32)   # [N, 20, 2]
    rel_data = np.load(REL_PATH).astype(np.float32)   # [N, 19, 2]

    print("=" * 60)
    print("ETH+UCY q20 filtering analysis")
    print(f"ABS_PATH = {ABS_PATH}")
    print(f"REL_PATH = {REL_PATH}")
    print(f"abs_data shape = {abs_data.shape}")
    print(f"rel_data shape = {rel_data.shape}")

    metrics_before = compute_metrics(rel_data)
    summary_before = summarize_metrics(metrics_before)

    threshold = float(np.quantile(metrics_before["avg_speed"], FILTER_QUANTILE))
    keep_mask = metrics_before["avg_speed"] >= threshold

    abs_q20 = abs_data[keep_mask]
    rel_q20 = rel_data[keep_mask]

    metrics_after = compute_metrics(rel_q20)
    summary_after = summarize_metrics(metrics_after)

    kept_ratio = float(np.mean(keep_mask))

    abs_q20_path = DATA_DIR / "data_eth_ucy_20_q20.npy"
    rel_q20_path = DATA_DIR / "data_eth_ucy_20_rel_q20.npy"

    np.save(abs_q20_path, abs_q20)
    np.save(rel_q20_path, rel_q20)

    # 保存 summary 表
    summary_df = pd.DataFrame(
        [
            {
                "metric": "sample_count",
                "before": summary_before["sample_count"],
                "after_q20": summary_after["sample_count"],
            },
            {
                "metric": "avg_speed_mean",
                "before": summary_before["avg_speed_mean"],
                "after_q20": summary_after["avg_speed_mean"],
            },
            {
                "metric": "avg_speed_median",
                "before": summary_before["avg_speed_median"],
                "after_q20": summary_after["avg_speed_median"],
            },
            {
                "metric": "moving_ratio_mean",
                "before": summary_before["moving_ratio_mean"],
                "after_q20": summary_after["moving_ratio_mean"],
            },
            {
                "metric": "moving_ratio_median",
                "before": summary_before["moving_ratio_median"],
                "after_q20": summary_after["moving_ratio_median"],
            },
            {
                "metric": "acc_rms_mean",
                "before": summary_before["acc_rms_mean"],
                "after_q20": summary_after["acc_rms_mean"],
            },
            {
                "metric": "acc_rms_median",
                "before": summary_before["acc_rms_median"],
                "after_q20": summary_after["acc_rms_median"],
            },
        ]
    )

    summary_csv_path = OUT_DIR / "eth_ucy_q20_summary.csv"
    summary_df.to_csv(summary_csv_path, index=False, float_format="%.6f")

    # 画图
    plot_hist_all_step_speed(
        metrics_before["step_norm"],
        OUT_DIR / "hist_all_step_speed_eth_ucy.png",
    )

    plot_compare_avg_speed(
        metrics_before["avg_speed"],
        metrics_after["avg_speed"],
        threshold,
        OUT_DIR / "compare_avg_speed_eth_ucy_q20.png",
    )

    # 终端输出
    print("\n===== q20 Filtering Summary =====")
    print(f"threshold (avg_speed q20) = {threshold:.8f}")
    print(f"before = {summary_before['sample_count']}")
    print(f"after  = {summary_after['sample_count']}")
    print(f"kept_ratio = {kept_ratio:.6f}")

    print("\n===== Motion Statistics =====")
    print(f"avg_speed_mean    : {summary_before['avg_speed_mean']:.8f} -> {summary_after['avg_speed_mean']:.8f}")
    print(f"avg_speed_median  : {summary_before['avg_speed_median']:.8f} -> {summary_after['avg_speed_median']:.8f}")
    print(f"moving_ratio_mean : {summary_before['moving_ratio_mean']:.8f} -> {summary_after['moving_ratio_mean']:.8f}")
    print(f"moving_ratio_median: {summary_before['moving_ratio_median']:.8f} -> {summary_after['moving_ratio_median']:.8f}")
    print(f"acc_rms_mean      : {summary_before['acc_rms_mean']:.8f} -> {summary_after['acc_rms_mean']:.8f}")
    print(f"acc_rms_median    : {summary_before['acc_rms_median']:.8f} -> {summary_after['acc_rms_median']:.8f}")

    print("\n===== Saved Files =====")
    print(f"abs_q20_path = {abs_q20_path}")
    print(f"rel_q20_path = {rel_q20_path}")
    print(f"summary_csv  = {summary_csv_path}")
    print(f"plot         = {OUT_DIR / 'hist_all_step_speed_eth_ucy.png'}")
    print(f"plot         = {OUT_DIR / 'compare_avg_speed_eth_ucy_q20.png'}")
    print("=" * 60)


if __name__ == "__main__":
    main()