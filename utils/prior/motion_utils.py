# tools/motion_utils.py

import csv
import os
from typing import Dict, Tuple

import numpy as np


def validate_rel_shape(rel: np.ndarray) -> None:
    if rel.ndim != 3 or rel.shape[-1] != 2:
        raise ValueError(
            f"rel 数据形状应为 (N, T, 2)，当前收到: {rel.shape}"
        )


def validate_abs_shape(abs_data: np.ndarray, rel: np.ndarray) -> None:
    if abs_data.ndim != 3 or abs_data.shape[-1] != 2:
        raise ValueError(
            f"abs 数据形状应为 (N, T+1, 2)，当前收到: {abs_data.shape}"
        )
    if abs_data.shape[0] != rel.shape[0]:
        raise ValueError(
            f"abs 和 rel 的样本数不一致: {abs_data.shape[0]} vs {rel.shape[0]}"
        )
    if abs_data.shape[1] != rel.shape[1] + 1:
        raise ValueError(
            f"abs 时间长度应比 rel 多 1: {abs_data.shape[1]} vs {rel.shape[1]}"
        )


def compute_motion_stats(
    rel: np.ndarray,
    moving_eps: float = None
) -> Dict[str, np.ndarray]:
    """
    输入:
        rel: shape = (N, T, 2)，这里 T=19
    输出:
        一个字典，包含每条样本的各种统计量
    """
    validate_rel_shape(rel)

    # 每一步位移大小: shape = (N, T)
    speed = np.linalg.norm(rel, axis=2)

    # 每条样本的速度类统计
    avg_speed = speed.mean(axis=1)
    max_speed = speed.max(axis=1)
    speed_std = speed.std(axis=1)
    path_length = speed.sum(axis=1)

    # 自动给 moving_ratio 选一个很小的“接近零运动”阈值
    # 这里使用所有正 speed 的 5% 分位数作为辅助阈值
    positive_speed = speed[speed > 0]
    if moving_eps is None:
        if positive_speed.size == 0:
            moving_eps = 1e-8
        else:
            moving_eps = float(np.percentile(positive_speed, 5))

    moving_ratio = (speed > moving_eps).mean(axis=1)

    # 二阶差分，近似加速度变化: shape = (N, T-1, 2)
    acc_proxy = np.diff(rel, axis=1)

    # 每一步“加速度变化量”的大小
    acc_mag = np.linalg.norm(acc_proxy, axis=2)

    # 平滑性统计
    acc_rms = np.sqrt((acc_mag ** 2).mean(axis=1))
    acc_mean = acc_mag.mean(axis=1)

    return {
        "speed": speed,                  # (N, T)
        "avg_speed": avg_speed,          # (N,)
        "max_speed": max_speed,          # (N,)
        "speed_std": speed_std,          # (N,)
        "path_length": path_length,      # (N,)
        "moving_ratio": moving_ratio,    # (N,)
        "acc_mag": acc_mag,              # (N, T-1)
        "acc_rms": acc_rms,              # (N,)
        "acc_mean": acc_mean,            # (N,)
        "moving_eps": np.array([moving_eps], dtype=np.float64),
    }


def vector_summary(x: np.ndarray) -> Dict[str, float]:
    x = np.asarray(x)
    return {
        "min": float(np.min(x)),
        "p05": float(np.percentile(x, 5)),
        "p10": float(np.percentile(x, 10)),
        "p20": float(np.percentile(x, 20)),
        "p25": float(np.percentile(x, 25)),
        "p30": float(np.percentile(x, 30)),
        "median": float(np.median(x)),
        "mean": float(np.mean(x)),
        "p70": float(np.percentile(x, 70)),
        "p75": float(np.percentile(x, 75)),
        "p80": float(np.percentile(x, 80)),
        "p90": float(np.percentile(x, 90)),
        "p95": float(np.percentile(x, 95)),
        "max": float(np.max(x)),
    }


def save_per_sample_csv(
    out_csv_path: str,
    stats: Dict[str, np.ndarray]
) -> None:
    """
    只保存每条样本的一维统计量，便于后面手动检查。
    """
    fields = [
        "sample_idx",
        "avg_speed",
        "max_speed",
        "speed_std",
        "path_length",
        "moving_ratio",
        "acc_rms",
        "acc_mean",
    ]

    n = len(stats["avg_speed"])
    os.makedirs(os.path.dirname(out_csv_path), exist_ok=True)

    with open(out_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(fields)

        for i in range(n):
            writer.writerow([
                i,
                float(stats["avg_speed"][i]),
                float(stats["max_speed"][i]),
                float(stats["speed_std"][i]),
                float(stats["path_length"][i]),
                float(stats["moving_ratio"][i]),
                float(stats["acc_rms"][i]),
                float(stats["acc_mean"][i]),
            ])


def save_summary_txt(
    out_txt_path: str,
    stats: Dict[str, np.ndarray],
    rel_shape: Tuple[int, ...]
) -> None:
    os.makedirs(os.path.dirname(out_txt_path), exist_ok=True)

    avg_speed_summary = vector_summary(stats["avg_speed"])
    acc_rms_summary = vector_summary(stats["acc_rms"])
    moving_ratio_summary = vector_summary(stats["moving_ratio"])
    moving_eps = float(stats["moving_eps"][0])

    lines = []
    lines.append("=== Motion Statistics Summary ===")
    lines.append(f"rel_shape = {rel_shape}")
    lines.append(f"moving_eps = {moving_eps:.8f}")
    lines.append("")

    lines.append("[avg_speed]")
    for k, v in avg_speed_summary.items():
        lines.append(f"{k}: {v:.8f}")
    lines.append("")

    lines.append("[acc_rms]")
    for k, v in acc_rms_summary.items():
        lines.append(f"{k}: {v:.8f}")
    lines.append("")

    lines.append("[moving_ratio]")
    for k, v in moving_ratio_summary.items():
        lines.append(f"{k}: {v:.8f}")
    lines.append("")

    with open(out_txt_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def select_by_quantile(
    values: np.ndarray,
    quantile: float
) -> Tuple[float, np.ndarray]:
    """
    quantile=20 表示取 20% 分位点为阈值，
    保留 values >= threshold 的样本，相当于过滤最慢的约 20%
    """
    if not (0 <= quantile <= 100):
        raise ValueError(f"quantile 应在 [0, 100]，当前为 {quantile}")

    threshold = float(np.percentile(values, quantile))
    keep_mask = values >= threshold
    return threshold, keep_mask