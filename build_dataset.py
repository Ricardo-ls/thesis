from pathlib import Path
import numpy as np
import pandas as pd


def load_eth_file(file_path: str) -> pd.DataFrame:
    """
    读取 ETH/UCY txt 文件
    格式：frame_id, ped_id, x, y
    """
    df = pd.read_csv(
        file_path,
        delim_whitespace=True,
        header=None,
        names=["frame_id", "ped_id", "x", "y"]
    )
    return df


def build_trajectory_windows(df: pd.DataFrame, window_size: int = 20):
    """
    按 ped_id 分组，按 frame_id 排序，切固定长度窗口
    输出 list，每个元素 shape = (window_size, 2)
    """
    samples = []

    # 按行人 ID 分组
    for ped_id, group in df.groupby("ped_id"):
        group = group.sort_values("frame_id").reset_index(drop=True)

        coords = group[["x", "y"]].values
        frames = group["frame_id"].values

        # 滑窗切片
        for start in range(0, len(group) - window_size + 1):
            end = start + window_size
            window_coords = coords[start:end]
            window_frames = frames[start:end]

            # 检查帧是否连续（ETH 常见间隔是 10）
            frame_diffs = np.diff(window_frames)
            if not np.all(frame_diffs == frame_diffs[0]):
                continue

            samples.append(window_coords)

    return np.array(samples, dtype=np.float32)


def main():
    file_path = "datasets/raw/all_data/biwi_eth.txt"
    output_path = "data_eth_20.npy"
    window_size = 20

    print(f"读取文件: {file_path}")
    df = load_eth_file(file_path)

    print("\n前5行数据：")
    print(df.head())

    print("\n总行数：", len(df))
    print("不同行人数量：", df['ped_id'].nunique())

    samples = build_trajectory_windows(df, window_size=window_size)

    print("\n生成样本 shape:", samples.shape)

    np.save(output_path, samples)
    print(f"已保存到: {output_path}")

    # 打印第一条样本看看
    if len(samples) > 0:
        print("\n第一条样本：")
        print(samples[0])


if __name__ == "__main__":
    main()