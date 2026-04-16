from __future__ import annotations

from pathlib import Path
import csv
import re


def _extract_from_run_note(run_note_path: Path):
    if not run_note_path.exists():
        return None

    best_epoch = None
    best_val_loss = None
    text = run_note_path.read_text(encoding="utf-8", errors="ignore")

    epoch_match = re.search(r"best_epoch:\s*(\d+)", text)
    loss_match = re.search(r"best_val_loss:\s*([0-9]*\.?[0-9]+)", text)
    if epoch_match:
        best_epoch = int(epoch_match.group(1))
    if loss_match:
        best_val_loss = float(loss_match.group(1))

    if best_epoch is None and best_val_loss is None:
        return None
    return {
        "run_note_path": str(run_note_path),
        "current_run_best_epoch": best_epoch,
        "current_run_best_val_loss": best_val_loss,
    }


def _extract_from_loss_history(loss_history_path: Path):
    if not loss_history_path.exists():
        return None

    rows = []
    with loss_history_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if "epoch" not in row or "val_loss" not in row:
                continue
            rows.append((int(row["epoch"]), float(row["val_loss"])))

    if not rows:
        return None

    best_epoch, best_val_loss = min(rows, key=lambda item: item[1])
    return {
        "loss_history_path": str(loss_history_path),
        "current_run_best_epoch": best_epoch,
        "current_run_best_val_loss": best_val_loss,
    }


def resolve_current_run_metadata(variant: str, train_seed: int, train_epochs: int, train_root: Path):
    variant = variant.lower()
    run_tag = f"seed{train_seed}-{train_epochs}epoch"
    train_tag = f"ddpm_eth_ucy_{variant}_h128"
    run_dir = train_root / train_tag / run_tag
    ckpt_path = run_dir / "best_model.pt"

    metadata = {
        "variant": variant,
        "train_seed": train_seed,
        "train_epochs": train_epochs,
        "run_tag": run_tag,
        "train_tag": train_tag,
        "run_dir": str(run_dir),
        "ckpt_path": str(ckpt_path),
        "current_run_best_epoch": None,
        "current_run_best_val_loss": None,
        "run_note_path": None,
        "loss_history_path": None,
    }

    if not run_dir.exists():
        return metadata

    run_notes = sorted(run_dir.glob("RUN_NOTE_*.md"))
    for run_note_path in run_notes:
        parsed = _extract_from_run_note(run_note_path)
        if parsed is not None:
            metadata.update(parsed)
            break

    if metadata["current_run_best_epoch"] is None or metadata["current_run_best_val_loss"] is None:
        parsed = _extract_from_loss_history(run_dir / "loss_history.csv")
        if parsed is not None:
            metadata.update(parsed)

    return metadata
