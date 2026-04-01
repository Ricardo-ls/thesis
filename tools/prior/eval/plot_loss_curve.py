from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
PROJECT_ROOT = Path(__file__).resolve().parents[3]


def find_column(columns, candidates):
    for c in candidates:
        if c in columns:
            return c
    return None


def main():
    csv_path = PROJECT_ROOT / "outputs" / "ddpm_minimal_q20" / "loss_history.csv"
    save_path = PROJECT_ROOT / "outputs" / "prior" / "eval" / "ddpm_minimal_q20" / "loss_curve.png"

    if not csv_path.exists():
        raise FileNotFoundError(f"找不到文件: {csv_path}")

    df = pd.read_csv(csv_path)

    print("读取到的列名：", list(df.columns))
    print(df.head())

    # 尝试自动识别常见列名
    epoch_col = find_column(df.columns, ["epoch", "Epoch", "epochs"])
    train_col = find_column(df.columns, ["train_loss", "loss_train", "train", "Train Loss"])
    val_col = find_column(df.columns, ["val_loss", "valid_loss", "loss_val", "val", "Val Loss"])

    if epoch_col is None:
        # 如果没有 epoch 列，就默认按行号生成
        df["epoch_auto"] = range(1, len(df) + 1)
        epoch_col = "epoch_auto"

    if train_col is None:
        raise ValueError(f"没有识别到训练损失列，请检查 CSV 列名：{list(df.columns)}")

    if val_col is None:
        raise ValueError(f"没有识别到验证损失列，请检查 CSV 列名：{list(df.columns)}")

    best_idx = df[val_col].idxmin()
    best_epoch = int(df.loc[best_idx, epoch_col])
    best_val = float(df.loc[best_idx, val_col])

    plt.figure(figsize=(8, 5))
    plt.plot(df[epoch_col], df[train_col], label="train_loss")
    plt.plot(df[epoch_col], df[val_col], label="val_loss")
    plt.scatter([best_epoch], [best_val], label=f"best val @ epoch {best_epoch}")

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("DDPM Training / Validation Loss Curve")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=200)
    plt.close()

    print(f"\n最优验证损失: {best_val:.6f}")
    print(f"最优 epoch: {best_epoch}")
    print(f"图片已保存到: {save_path}")


if __name__ == "__main__":
    main()