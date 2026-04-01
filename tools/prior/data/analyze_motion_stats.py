# tools/analyze_motion_stats.py

import argparse
import os

import matplotlib
matplotlib.use("Agg")  # 服务器/终端环境也能保存图
import matplotlib.pyplot as plt
import numpy as np

from utils.prior.motion_utils import (
    compute_motion_stats,
    save_per_sample_csv,
    save_summary_txt,
    validate_abs_shape,
)


def plot_hist(data, title, xlabel, out_path, bins=50):
    plt.figure(figsize=(7, 5))
    plt.hist(data, bins=bins)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_scatter(x, y, title, xlabel, ylabel, out_path):
    plt.figure(figsize=(7, 5))
    plt.scatter(x, y, alpha=0.6, s=12)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
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
    parser.add_argument(
        "--rel_path",
        type=str,
        required=True,
        help="相对位移 npy 路径，例如 datasets/processed/data_eth_20_rel.npy"
    )
    parser.add_argument(
        "--abs_path",
        type=str,
        default=None,
        help="绝对轨迹 npy 路径，例如 datasets/processed/data_eth_20.npy"
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        required=True,
        help="输出目录，例如 outputs/eth_stats"
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
    stats = compute_motion_stats(rel, moving_eps=args.moving_eps)

    save_per_sample_csv(
        os.path.join(args.out_dir, "per_sample_stats.csv"),
        stats
    )
    save_summary_txt(
        os.path.join(args.out_dir, "summary.txt"),
        stats,
        rel.shape
    )

    # 1) 所有 step speed 的总体分布
    plot_hist(
        stats["speed"].reshape(-1),
        title="Histogram of All Step Speeds",
        xlabel="step speed",
        out_path=os.path.join(args.out_dir, "hist_all_step_speed.png"),
        bins=60
    )

    # 2) 每条轨迹 avg_speed 分布
    plot_hist(
        stats["avg_speed"],
        title="Histogram of Avg Speed Per Trajectory",
        xlabel="avg_speed",
        out_path=os.path.join(args.out_dir, "hist_avg_speed.png"),
        bins=50
    )

    # 3) 平滑性 acc_rms 分布
    plot_hist(
        stats["acc_rms"],
        title="Histogram of Acceleration RMS",
        xlabel="acc_rms",
        out_path=os.path.join(args.out_dir, "hist_acc_rms.png"),
        bins=50
    )

    # 4) avg_speed vs acc_rms
    plot_scatter(
        stats["avg_speed"],
        stats["acc_rms"],
        title="Avg Speed vs Acc RMS",
        xlabel="avg_speed",
        ylabel="acc_rms",
        out_path=os.path.join(args.out_dir, "scatter_avg_speed_vs_acc_rms.png")
    )

    # 如果有绝对轨迹数据，再画轨迹图
    if args.abs_path is not None:
        abs_data = np.load(args.abs_path)
        validate_abs_shape(abs_data, rel)

        order = np.argsort(stats["avg_speed"])
        slow_idx = order[: min(40, len(order))]
        fast_idx = order[-min(40, len(order)):]

        rng = np.random.default_rng(42)
        rand_idx = rng.choice(len(abs_data), size=min(40, len(abs_data)), replace=False)

        plot_trajectories(
            abs_data[rand_idx],
            title="Random Absolute Trajectories",
            out_path=os.path.join(args.out_dir, "traj_random.png"),
        )

        plot_trajectories(
            abs_data[slow_idx],
            title="Slowest Trajectories by Avg Speed",
            out_path=os.path.join(args.out_dir, "traj_slowest.png"),
        )

        plot_trajectories(
            abs_data[fast_idx],
            title="Fastest Trajectories by Avg Speed",
            out_path=os.path.join(args.out_dir, "traj_fastest.png"),
        )

    print("=" * 60)
    print("统计完成")
    print(f"输入 rel: {args.rel_path}")
    if args.abs_path:
        print(f"输入 abs: {args.abs_path}")
    print(f"输出目录: {args.out_dir}")
    print(f"样本数 N = {rel.shape[0]}")
    print(f"时间步 T = {rel.shape[1]}")
    print(f"moving_eps = {float(stats['moving_eps'][0]):.8f}")
    print("=" * 60)


if __name__ == "__main__":
    main()