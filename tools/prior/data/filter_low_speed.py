# tools/filter_low_speed.py
#按 avg_speed 的分位数做过滤
#默认保留 avg_speed >= q20
#保存过滤后的 rel 和 abs
#保存 keep_mask
#保存过滤前后的统计和对比图



import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from utils.prior.motion_utils import (
    compute_motion_stats,
    save_per_sample_csv,
    save_summary_txt,
    select_by_quantile,
    validate_abs_shape,
)


def plot_compare_hist(before, after, title, xlabel, out_path, bins=50):
    plt.figure(figsize=(7, 5))
    plt.hist(before, bins=bins, alpha=0.6, label="before")
    plt.hist(after, bins=bins, alpha=0.6, label="after")
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("Count")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_trajectories(trajs, title, out_path, max_plot=40):
    plt.figure(figsize=(7, 7))
    plot_count = min(len(trajs), max_plot)

    for i in range(plot_count):
        tr = trajs[i]
        plt.plot(tr[:, 0], tr[:, 1], alpha=0.7, linewidth=1.0)

    plt.title(title)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.axis("equal")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rel_path", type=str, required=True)
    parser.add_argument("--abs_path", type=str, default=None)
    parser.add_argument("--out_dir", type=str, required=True)

    parser.add_argument(
        "--metric",
        type=str,
        default="avg_speed",
        choices=["avg_speed", "path_length", "moving_ratio"],
        help="按哪个统计量做过滤，第一版推荐 avg_speed"
    )
    parser.add_argument(
        "--quantile",
        type=float,
        default=20.0,
        help="例如 20 表示过滤最慢的约 20 percent"
    )
    parser.add_argument(
        "--moving_eps",
        type=float,
        default=None,
        help="moving_ratio 的步级阈值，默认自动计算"
    )
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    rel = np.load(args.rel_path)
    abs_data = None
    if args.abs_path is not None:
        abs_data = np.load(args.abs_path)
        validate_abs_shape(abs_data, rel)

    # 原始数据统计
    stats_before = compute_motion_stats(rel, moving_eps=args.moving_eps)

    # 依据某个指标分位数过滤
    metric_values = stats_before[args.metric]
    threshold, keep_mask = select_by_quantile(metric_values, args.quantile)

    rel_filtered = rel[keep_mask]

    # after 统计沿用 before 的 moving_eps，保证 before/after 可比
    moving_eps = float(stats_before["moving_eps"][0])
    stats_after = compute_motion_stats(rel_filtered, moving_eps=moving_eps)

    # 保存过滤结果
    np.save(os.path.join(args.out_dir, "keep_mask.npy"), keep_mask)
    np.save(os.path.join(args.out_dir, "filtered_rel.npy"), rel_filtered)

    if abs_data is not None:
        abs_filtered = abs_data[keep_mask]
        np.save(os.path.join(args.out_dir, "filtered_abs.npy"), abs_filtered)
    else:
        abs_filtered = None

    # 保存统计
    save_per_sample_csv(
        os.path.join(args.out_dir, "per_sample_stats_before.csv"),
        stats_before
    )
    save_per_sample_csv(
        os.path.join(args.out_dir, "per_sample_stats_after.csv"),
        stats_after
    )
    save_summary_txt(
        os.path.join(args.out_dir, "summary_before.txt"),
        stats_before,
        rel.shape
    )
    save_summary_txt(
        os.path.join(args.out_dir, "summary_after.txt"),
        stats_after,
        rel_filtered.shape
    )

    # 保存过滤报告
    with open(os.path.join(args.out_dir, "filter_report.txt"), "w", encoding="utf-8") as f:
        f.write("=== Low-Speed Filtering Report ===\n")
        f.write(f"metric = {args.metric}\n")
        f.write(f"quantile = {args.quantile}\n")
        f.write(f"threshold = {threshold:.8f}\n")
        f.write(f"moving_eps = {moving_eps:.8f}\n")
        f.write(f"before_count = {len(rel)}\n")
        f.write(f"after_count = {len(rel_filtered)}\n")
        f.write(f"kept_ratio = {len(rel_filtered) / len(rel):.6f}\n")

    # 对比图
    plot_compare_hist(
        stats_before["avg_speed"],
        stats_after["avg_speed"],
        title="Avg Speed Before vs After Filtering",
        xlabel="avg_speed",
        out_path=os.path.join(args.out_dir, "compare_avg_speed.png"),
        bins=50
    )

    plot_compare_hist(
        stats_before["speed"].reshape(-1),
        stats_after["speed"].reshape(-1),
        title="All Step Speeds Before vs After Filtering",
        xlabel="step speed",
        out_path=os.path.join(args.out_dir, "compare_step_speed.png"),
        bins=60
    )

    plot_compare_hist(
        stats_before["acc_rms"],
        stats_after["acc_rms"],
        title="Acc RMS Before vs After Filtering",
        xlabel="acc_rms",
        out_path=os.path.join(args.out_dir, "compare_acc_rms.png"),
        bins=50
    )

    if abs_data is not None and abs_filtered is not None and len(abs_filtered) > 0:
        rng = np.random.default_rng(42)
        rand_before = rng.choice(len(abs_data), size=min(40, len(abs_data)), replace=False)
        rand_after = rng.choice(len(abs_filtered), size=min(40, len(abs_filtered)), replace=False)

        plot_trajectories(
            abs_data[rand_before],
            title="Random Trajectories Before Filtering",
            out_path=os.path.join(args.out_dir, "traj_before_random.png"),
        )

        plot_trajectories(
            abs_filtered[rand_after],
            title="Random Trajectories After Filtering",
            out_path=os.path.join(args.out_dir, "traj_after_random.png"),
        )

    print("=" * 60)
    print("过滤完成")
    print(f"metric     = {args.metric}")
    print(f"quantile   = {args.quantile}")
    print(f"threshold  = {threshold:.8f}")
    print(f"before N   = {len(rel)}")
    print(f"after N    = {len(rel_filtered)}")
    print(f"keep ratio = {len(rel_filtered) / len(rel):.6f}")
    print(f"输出目录: {args.out_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
