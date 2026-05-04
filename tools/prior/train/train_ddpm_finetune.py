from __future__ import annotations

from pathlib import Path
import csv
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from diffusion.ddpm_utils import DDPMForwardProcess
from models.temporal_denoiser import TemporalDenoiser1D


CHECKPOINT_PATH = PROJECT_ROOT / "outputs" / "prior" / "train" / "ddpm_eth_ucy_none_h128" / "seed42-100epoch" / "best_model.pt"
TRAIN_PATH = PROJECT_ROOT / "data" / "simulated" / "train_trajs.npy"
VAL_PATH = PROJECT_ROOT / "data" / "simulated" / "val_trajs.npy"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "prior" / "train" / "ddpm_finetuned_h128" / "seed42-100epoch"
BEST_MODEL_PATH = OUTPUT_DIR / "best_model.pt"
LOSS_CURVE_PATH = OUTPUT_DIR / "loss_curve.csv"

HIDDEN_DIM = 128
EPOCHS = 100
BATCH_SIZE = 128
TIMESTEPS = 100
RANDOM_SEED = 42
LEARNING_RATE = 1e-4
EVAL_SEED = 12345


def set_train_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def load_checkpoint_state(path: Path):
    if not path.exists():
        raise FileNotFoundError(
            f"Required ETH/UCY checkpoint does not exist: {path}. "
            "Stopping instead of auto-searching."
        )
    checkpoint = torch.load(path, map_location="cpu")
    if isinstance(checkpoint, dict):
        if "model_state_dict" in checkpoint:
            return checkpoint["model_state_dict"], checkpoint
        if "model" in checkpoint:
            return checkpoint["model"], checkpoint
    if isinstance(checkpoint, dict):
        return checkpoint, checkpoint
    raise RuntimeError(f"Unsupported checkpoint format at {path}: {type(checkpoint).__name__}")


def load_absolute_data(path: Path, expected_shape: tuple[int, int, int]):
    if not path.exists():
        raise FileNotFoundError(f"Required data file does not exist: {path}")
    arr = np.load(path, allow_pickle=False).astype(np.float32)
    if tuple(arr.shape) != expected_shape:
        raise RuntimeError(f"Data shape mismatch for {path}: expected {expected_shape}, got {arr.shape}")
    return arr


def to_relative(trajs_abs: np.ndarray) -> np.ndarray:
    rel = trajs_abs[:, 1:, :] - trajs_abs[:, :-1, :]
    return rel.astype(np.float32)


def build_dataloaders(train_rel: np.ndarray, val_rel: np.ndarray):
    train_ds = TensorDataset(torch.from_numpy(train_rel))
    val_ds = TensorDataset(torch.from_numpy(val_rel))
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)
    return train_loader, val_loader


def build_model(device: str):
    return TemporalDenoiser1D(
        max_timesteps=TIMESTEPS,
        in_channels=2,
        hidden_dim=HIDDEN_DIM,
    ).to(device)


def run_one_epoch(model, diffusion, loader, optimizer, device: str, train: bool):
    if train:
        model.train()
    else:
        model.eval()

    total_loss = 0.0
    total_count = 0

    for (batch_rel,) in loader:
        x0 = batch_rel.permute(0, 2, 1).to(device)
        t = diffusion.sample_timesteps(batch_size=x0.shape[0])
        xt, noise = diffusion.q_sample(x0, t)

        if train:
            optimizer.zero_grad()

        with torch.set_grad_enabled(train):
            pred_noise = model(xt, t)
            loss = F.mse_loss(pred_noise, noise)
            if train:
                loss.backward()
                optimizer.step()

        batch_size = x0.shape[0]
        total_loss += float(loss.item()) * batch_size
        total_count += batch_size

    return float(total_loss / max(1, total_count))


def evaluate_with_fixed_seed(model, diffusion, loader, device: str, eval_seed: int):
    torch.manual_seed(eval_seed)
    np.random.seed(eval_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(eval_seed)
    model.eval()

    total_loss = 0.0
    total_count = 0
    with torch.no_grad():
        for (batch_rel,) in loader:
            x0 = batch_rel.permute(0, 2, 1).to(device)
            t = diffusion.sample_timesteps(batch_size=x0.shape[0])
            xt, noise = diffusion.q_sample(x0, t)
            pred_noise = model(xt, t)
            loss = F.mse_loss(pred_noise, noise)

            batch_size = x0.shape[0]
            total_loss += float(loss.item()) * batch_size
            total_count += batch_size
    return float(total_loss / max(1, total_count))


def save_loss_curve_csv(history: list[dict], path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["epoch", "train_loss", "val_loss"])
        writer.writeheader()
        writer.writerows(history)
    print(f"Saved: {path}")


def save_best_model(path: Path, state_dict: dict, best_epoch: int, best_val_loss: float):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": state_dict,
            "epoch": int(best_epoch),
            "best_val_loss": float(best_val_loss),
            "config": {
                "hidden_dim": HIDDEN_DIM,
                "epochs": EPOCHS,
                "batch_size": BATCH_SIZE,
                "timesteps": TIMESTEPS,
                "random_seed": RANDOM_SEED,
                "learning_rate": LEARNING_RATE,
                "source_checkpoint": str(CHECKPOINT_PATH),
                "train_data": str(TRAIN_PATH),
                "val_data": str(VAL_PATH),
            },
        },
        path,
    )
    print(f"Saved: {path}")


def main():
    set_train_seed(RANDOM_SEED)
    device = resolve_device()

    train_abs = load_absolute_data(TRAIN_PATH, (20000, 20, 2))
    val_abs = load_absolute_data(VAL_PATH, (2000, 20, 2))
    train_rel = to_relative(train_abs)
    val_rel = to_relative(val_abs)

    if tuple(train_rel.shape) != (20000, 19, 2):
        raise RuntimeError(f"train_rel shape mismatch: expected (20000, 19, 2), got {train_rel.shape}")
    if tuple(val_rel.shape) != (2000, 19, 2):
        raise RuntimeError(f"val_rel shape mismatch: expected (2000, 19, 2), got {val_rel.shape}")

    train_loader, val_loader = build_dataloaders(train_rel, val_rel)
    example_batch_shape = next(iter(train_loader))[0].permute(0, 2, 1).shape

    state_dict, checkpoint_meta = load_checkpoint_state(CHECKPOINT_PATH)
    model = build_model(device)
    model.load_state_dict(state_dict)

    diffusion = DDPMForwardProcess(timesteps=TIMESTEPS, device=device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    history = []
    best_state = None
    best_val_loss = float("inf")
    best_epoch = -1

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"device            : {device}")
    print(f"train_abs shape   : {train_abs.shape}")
    print(f"val_abs shape     : {val_abs.shape}")
    print(f"checkpoint        : {CHECKPOINT_PATH}")
    print(f"output_dir        : {OUTPUT_DIR}")

    for epoch in range(1, EPOCHS + 1):
        train_loss = run_one_epoch(model, diffusion, train_loader, optimizer, device=device, train=True)
        val_loss = run_one_epoch(model, diffusion, val_loader, optimizer, device=device, train=False)

        history.append(
            {
                "epoch": int(epoch),
                "train_loss": float(train_loss),
                "val_loss": float(val_loss),
            }
        )
        print(f"Epoch [{epoch:03d}/{EPOCHS:03d}] train_loss = {train_loss:.6f} | val_loss = {val_loss:.6f}")

        if val_loss < best_val_loss:
            best_val_loss = float(val_loss)
            best_epoch = int(epoch)
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if best_state is None:
        raise RuntimeError("No best model state was captured during fine-tuning.")

    save_best_model(BEST_MODEL_PATH, best_state, best_epoch=best_epoch, best_val_loss=best_val_loss)
    save_loss_curve_csv(history, LOSS_CURVE_PATH)

    eth_model = build_model(device)
    eth_model.load_state_dict(state_dict)
    finetuned_model = build_model(device)
    finetuned_model.load_state_dict(best_state)

    eth_val_loss = evaluate_with_fixed_seed(eth_model, diffusion, val_loader, device=device, eval_seed=EVAL_SEED)
    finetuned_val_loss = evaluate_with_fixed_seed(
        finetuned_model,
        diffusion,
        val_loader,
        device=device,
        eval_seed=EVAL_SEED,
    )

    delta = float(eth_val_loss - finetuned_val_loss)
    pct = float(delta / eth_val_loss * 100.0) if eth_val_loss != 0.0 else 0.0

    print("=== 指令3 验证输出 ===")
    print(f"train_rel shape      : {train_rel.shape}")
    print(f"val_rel shape        : {val_rel.shape}")
    print(f"model input shape    : {tuple(example_batch_shape)}")
    print(f"ETH prior val_loss   (室内val集): {eth_val_loss:.6f}")
    print(f"Fine-tuned val_loss  (室内val集): {finetuned_val_loss:.6f}")
    print(f"Fine-tune 改善       : {delta:.6f}  ({pct:.1f}%)")
    print(f"Fine-tuned checkpoint: {BEST_MODEL_PATH}")
    print("=== 等待人工确认 ===")


if __name__ == "__main__":
    main()
