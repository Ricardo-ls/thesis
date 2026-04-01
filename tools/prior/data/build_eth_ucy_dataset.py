from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[3]
RAW_DIR = PROJECT_ROOT / "datasets" / "raw" / "all_data"
OUT_DIR = PROJECT_ROOT / "datasets" / "processed"

WINDOW_SIZE = 20
STRIDE = 1

# 当前主实验：ETH + UCY（先不加 hotel，保持“ETH baseline + UCY 扩展”更清晰）
SCENE_FILES = [
    "biwi_eth.txt",
    "students001.txt",
    "students003.txt",
    "uni_examples.txt",
    "crowds_zara01.txt",
    "crowds_zara02.txt",
    "crowds_zara03.txt",
]

ABS_SAVE_PATH = OUT_DIR / "data_eth_ucy_20.npy"
REL_SAVE_PATH = OUT_DIR / "data_eth_ucy_20_rel.npy"
META_SAVE_PATH = OUT_DIR / "data_eth_ucy_20_meta.csv"
SUMMARY_SAVE_PATH = OUT_DIR / "data_eth_ucy_20_summary.csv"


def load_scene_file(file_path: Path) -> pd.DataFrame:
    """
    读取单个轨迹 txt 文件，格式为：
    frame_id  ped_id  x  y
    分隔符为空白符。
    """
    df = pd.read_csv(
        file_path,
        sep=r"\s+",
        header=None,
        names=["frame_id", "ped_id", "x", "y"],
        engine="python",
    )

    # 转成数值，避免隐式字符串问题
    for col in ["frame_id", "ped_id", "x", "y"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna().copy()
    df = df.sort_values(["ped_id", "frame_id"]).reset_index(drop=True)
    return df


def build_windows_from_scene(
    df: pd.DataFrame,
    scene_name: str,
    window_size: int = WINDOW_SIZE,
    stride: int = STRIDE,
):
    """
    对单个 scene：
    - 按 ped_id 分组
    - 按 frame_id 排序
    - 做固定长度滑窗
    """
    windows = []
    meta_rows = []

    ped_groups = df.groupby("ped_id", sort=False)

    total_tracks = 0
    usable_tracks = 0

    for ped_id, ped_df in ped_groups:
        total_tracks += 1

        ped_df = ped_df.sort_values("frame_id").reset_index(drop=True)
        n = len(ped_df)

        if n < window_size:
            continue

        usable_tracks += 1

        coords = ped_df[["x", "y"]].to_numpy(dtype=np.float32)
        frames = ped_df["frame_id"].to_numpy()

        for start in range(0, n - window_size + 1, stride):
            end = start + window_size
            window = coords[start:end]  # [20, 2]
            windows.append(window)

            meta_rows.append(
                {
                    "scene": scene_name,
                    "ped_id": float(ped_id),
                    "start_idx": int(start),
                    "end_idx": int(end - 1),
                    "start_frame": float(frames[start]),
                    "end_frame": float(frames[end - 1]),
                    "window_size": int(window_size),
                }
            )

    return windows, meta_rows, total_tracks, usable_tracks


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    all_windows = []
    all_meta = []
    summary_rows = []

    print("=" * 60)
    print("Building ETH+UCY fixed-window trajectory dataset")
    print(f"RAW_DIR      = {RAW_DIR}")
    print(f"OUT_DIR      = {OUT_DIR}")
    print(f"WINDOW_SIZE  = {WINDOW_SIZE}")
    print(f"STRIDE       = {STRIDE}")
    print("SCENE_FILES  =")
    for name in SCENE_FILES:
        print(f"  - {name}")
    print("=" * 60)

    for scene_file in SCENE_FILES:
        scene_path = RAW_DIR / scene_file
        if not scene_path.exists():
            raise FileNotFoundError(f"找不到文件: {scene_path}")

        df = load_scene_file(scene_path)

        windows, meta_rows, total_tracks, usable_tracks = build_windows_from_scene(
            df=df,
            scene_name=scene_file.replace(".txt", ""),
            window_size=WINDOW_SIZE,
            stride=STRIDE,
        )

        scene_window_count = len(windows)

        print(
            f"[{scene_file}] rows={len(df):6d} | "
            f"tracks={total_tracks:4d} | usable_tracks={usable_tracks:4d} | "
            f"windows={scene_window_count:6d}"
        )

        all_windows.extend(windows)
        all_meta.extend(meta_rows)

        summary_rows.append(
            {
                "scene": scene_file.replace(".txt", ""),
                "num_rows": int(len(df)),
                "num_tracks": int(total_tracks),
                "num_usable_tracks": int(usable_tracks),
                "num_windows": int(scene_window_count),
            }
        )

    if len(all_windows) == 0:
        raise RuntimeError("没有构建出任何窗口，请检查数据读取或 window_size 设置。")

    abs_data = np.stack(all_windows, axis=0).astype(np.float32)   # [N, 20, 2]
    rel_data = np.diff(abs_data, axis=1).astype(np.float32)       # [N, 19, 2]

    meta_df = pd.DataFrame(all_meta)
    summary_df = pd.DataFrame(summary_rows)

    np.save(ABS_SAVE_PATH, abs_data)
    np.save(REL_SAVE_PATH, rel_data)
    meta_df.to_csv(META_SAVE_PATH, index=False)
    summary_df.to_csv(SUMMARY_SAVE_PATH, index=False)

    print("\n" + "=" * 60)
    print("Build finished.")
    print(f"ABS_SAVE_PATH  = {ABS_SAVE_PATH}")
    print(f"REL_SAVE_PATH  = {REL_SAVE_PATH}")
    print(f"META_SAVE_PATH = {META_SAVE_PATH}")
    print(f"SUMMARY_PATH   = {SUMMARY_SAVE_PATH}")
    print(f"abs_data shape = {abs_data.shape}")
    print(f"rel_data shape = {rel_data.shape}")
    print("=" * 60)


if __name__ == "__main__":
    main()