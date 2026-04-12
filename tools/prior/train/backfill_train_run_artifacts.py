from pathlib import Path
import csv
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.prior.train.train_ddpm_eth_ucy_h128 import save_loss_curve, save_run_note


def parse_run_dir_name(run_dir: Path):
    seed_part, epoch_part = run_dir.name.split("-", 1)
    random_seed = int(seed_part.replace("seed", ""))
    epochs = int(epoch_part.replace("epoch", ""))
    return random_seed, epochs


def load_history(csv_path: Path):
    with csv_path.open("r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    return [
        {
            "epoch": int(row["epoch"]),
            "train_loss": float(row["train_loss"]),
            "val_loss": float(row["val_loss"]),
        }
        for row in rows
    ]


def main():
    train_root = PROJECT_ROOT / "outputs" / "prior" / "train"
    for variant_dir in sorted(train_root.glob("ddpm_eth_ucy_*_h128")):
        variant = variant_dir.name.replace("ddpm_eth_ucy_", "").replace("_h128", "")
        for run_dir in sorted(
            path
            for path in variant_dir.iterdir()
            if path.is_dir()
            and path.name.startswith("seed")
            and path.name.endswith("epoch")
            and "-" in path.name
        ):
            csv_path = run_dir / "loss_history.csv"
            if not csv_path.exists():
                continue

            history = load_history(csv_path)
            save_loss_curve(history, run_dir)

            random_seed, epochs = parse_run_dir_name(run_dir)
            note_path = run_dir / f"RUN_NOTE_{variant}_ep{epochs}_seed{random_seed}.md"
            if not note_path.exists():
                class Args:
                    pass

                args = Args()
                args.epochs = epochs
                args.batch_size = 128
                args.timesteps = 100
                args.hidden_dim = 128
                args.random_seed = random_seed
                args.train_ratio = 0.8
                args.lr = 1e-3

                best_row = min(history, key=lambda row: row["val_loss"])
                save_run_note(
                    output_dir=run_dir,
                    variant=variant,
                    args=args,
                    best_epoch=best_row["epoch"],
                    best_val_loss=best_row["val_loss"],
                )


if __name__ == "__main__":
    main()
