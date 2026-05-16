from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import argparse
import csv
import copy
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from diffusion.ddpm_utils import DDPMForwardProcess
from models.temporal_denoiser_conditional_4block import TemporalDenoiserConditional4Block
from tools.stage3_indoor.train_cond_residual_gaussian import (
    BATCH_SIZE,
    EARLY_STOP_PATIENCE,
    EMA_DECAY,
    LEARNING_RATE,
    MAX_EPOCHS,
    MIN_EPOCHS,
    NORM_PATH,
    OUTPUT_NORM_PATH as _REFERENCE_OUTPUT_NORM_PATH,
    SEED,
    TIMESTEPS,
    TRAIN_PATH,
    VAL_PATH,
    apply_augmentations,
    build_epoch_training_arrays,
    ensure_required_inputs,
    generate_gaussian_degraded,
    normalize_rel,
    save_csv,
    save_npz,
    save_torch,
    set_all_seeds,
    to_rel,
    validate_shapes,
)


OUTPUT_DIR = PROJECT_ROOT / "outputs" / "stage3_indoor" / "receptive_field_expansion"
FINAL_MODEL_PATH = OUTPUT_DIR / "ckpt_4block_final.pt"
BEST_MODEL_PATH = OUTPUT_DIR / "ckpt_4block_best.pt"
FINAL_EMA_PATH = OUTPUT_DIR / "ckpt_4block_final_ema.pt"
BEST_EMA_PATH = OUTPUT_DIR / "ckpt_4block_best_ema.pt"
TRAIN_LOG_PATH = OUTPUT_DIR / "train_4block_log.csv"
OUTPUT_NORM_PATH = OUTPUT_DIR / _REFERENCE_OUTPUT_NORM_PATH.name


def count_parameters(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def write_epoch_log_row(path: Path, row: dict) -> None:
    fieldnames = ["epoch", "train_loss", "val_loss", "elapsed_seconds", "is_best"]
    file_exists = path.is_file()
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def print_startup_info(
    *,
    train_samples: int,
    val_samples: int,
    train_batches: int,
    val_batches: int,
    device: torch.device,
    model_params: int,
) -> None:
    print(f"training data path: {TRAIN_PATH}", flush=True)
    print(f"validation data path: {VAL_PATH}", flush=True)
    print(f"number of train samples: {train_samples}", flush=True)
    print(f"number of val samples: {val_samples}", flush=True)
    print(f"batch size: {BATCH_SIZE}", flush=True)
    print(f"number of train batches per epoch: {train_batches}", flush=True)
    print(f"number of val batches per epoch: {val_batches}", flush=True)
    print(f"total epochs: {MAX_EPOCHS}", flush=True)
    print(f"diffusion steps: {TIMESTEPS}", flush=True)
    print(f"device: {device}", flush=True)
    print(f"model parameter count: {model_params}", flush=True)
    print(f"output directory: {OUTPUT_DIR}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ensure_required_inputs()
    set_all_seeds(SEED)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    train_trajs = np.load(TRAIN_PATH).astype(np.float32)
    val_trajs = np.load(VAL_PATH).astype(np.float32)
    norm = np.load(NORM_PATH)
    rel_mean = norm["rel_mean"].astype(np.float32)
    rel_std = norm["rel_std"].astype(np.float32)

    # Reuse the same reference validation helper with a dummy clean array,
    # since Phase 2 does not run the post-training evaluation section.
    validate_shapes(train_trajs, val_trajs, val_trajs, rel_mean, rel_std)

    aug_train = apply_augmentations(train_trajs)
    if aug_train.shape != (60000, 20, 2):
        raise ValueError(f"Expected aug_train shape (60000, 20, 2), got {aug_train.shape}")

    first_epoch_residual_norm, _, _ = build_epoch_training_arrays(
        aug_train,
        epoch=1,
        rel_mean=rel_mean,
        rel_std=rel_std,
    )
    residual_norm_std = float(first_epoch_residual_norm.std())
    residual_norm_mean = float(first_epoch_residual_norm.mean())

    print("=== Stage 3 Receptive-Field Expansion: 4-block Conditional Residual DDPM ===")
    print(f"train augmented shape: {aug_train.shape}")
    print(f"rel_mean: ({rel_mean[0]:.6f}, {rel_mean[1]:.6f})")
    print(f"rel_std:  ({rel_std[0]:.6f}, {rel_std[1]:.6f})")
    print(f"residual_norm std: {residual_norm_std:.4f}")
    print(f"residual_norm mean: {residual_norm_mean:.4f}")
    if residual_norm_std < 0.1:
        print("WARNING: residual signal too weak, DDPM may not learn effectively")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    diffusion = DDPMForwardProcess(timesteps=TIMESTEPS, device=device)

    model = TemporalDenoiserConditional4Block(
        max_timesteps=TIMESTEPS,
        in_channels=4,
        hidden_dim=128,
    ).to(device)
    ema_model = copy.deepcopy(model).to(device)
    ema_model.eval()
    for p in ema_model.parameters():
        p.requires_grad_(False)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    val_degraded = generate_gaussian_degraded(val_trajs, seed_base=42, epoch=None)
    val_clean_rel_norm = normalize_rel(to_rel(val_trajs), rel_mean, rel_std)
    val_degraded_rel_norm = normalize_rel(to_rel(val_degraded), rel_mean, rel_std)
    val_residual_norm = (val_clean_rel_norm - val_degraded_rel_norm).astype(np.float32)

    val_residual_tensor = torch.from_numpy(val_residual_norm.transpose(0, 2, 1))
    val_cond_tensor = torch.from_numpy(val_degraded_rel_norm.transpose(0, 2, 1))
    val_rng = np.random.default_rng(SEED + 1000)
    val_t_all = torch.from_numpy(
        val_rng.integers(0, TIMESTEPS, size=(val_residual_tensor.shape[0],), dtype=np.int64)
    ).to(device=device, dtype=torch.long)
    val_noise_all = torch.from_numpy(
        val_rng.normal(0.0, 1.0, size=(val_residual_tensor.shape[0], 2, 19)).astype(np.float32)
    ).to(device=device, dtype=torch.float32)

    train_samples = aug_train.shape[0]
    val_samples = val_residual_tensor.shape[0]
    train_batches_per_epoch = (train_samples + BATCH_SIZE - 1) // BATCH_SIZE
    val_batches_per_epoch = (val_samples + BATCH_SIZE - 1) // BATCH_SIZE
    model_params = count_parameters(model)

    print_startup_info(
        train_samples=train_samples,
        val_samples=val_samples,
        train_batches=train_batches_per_epoch,
        val_batches=val_batches_per_epoch,
        device=device,
        model_params=model_params,
    )

    if args.dry_run:
        dry_residual_norm, dry_degraded_rel_norm, _ = build_epoch_training_arrays(
            aug_train,
            epoch=1,
            rel_mean=rel_mean,
            rel_std=rel_std,
        )
        dry_residual_tensor = torch.from_numpy(dry_residual_norm.transpose(0, 2, 1))
        dry_cond_tensor = torch.from_numpy(dry_degraded_rel_norm.transpose(0, 2, 1))
        dry_train_loader = DataLoader(
            TensorDataset(dry_residual_tensor, dry_cond_tensor),
            batch_size=BATCH_SIZE,
            shuffle=True,
            num_workers=0,
            generator=torch.Generator().manual_seed(SEED + 1),
        )

        print("Epoch [1/60] started", flush=True)
        model.train()
        batch_residual, batch_cond = next(iter(dry_train_loader))
        x0 = batch_residual.to(device=device, dtype=torch.float32)
        x_cond = batch_cond.to(device=device, dtype=torch.float32)
        t = diffusion.sample_timesteps(batch_size=x0.shape[0])
        x_t, noise = diffusion.q_sample(x0, t)
        optimizer.zero_grad()
        eps_pred = model(x_t, x_cond, t)
        train_loss = F.mse_loss(eps_pred, noise)
        train_loss.backward()
        optimizer.step()
        print(f"Epoch 1/60 batch 1/{train_batches_per_epoch} loss={train_loss.item():.6f}", flush=True)

        ema_model.eval()
        with torch.no_grad():
            x0_val = val_residual_tensor[:BATCH_SIZE].to(device=device, dtype=torch.float32)
            x_cond_val = val_cond_tensor[:BATCH_SIZE].to(device=device, dtype=torch.float32)
            t_val = val_t_all[: x0_val.shape[0]]
            noise_val = val_noise_all[: x0_val.shape[0]]
            x_t_val, _ = diffusion.q_sample(x0_val, t_val, noise=noise_val)
            eps_pred_val = ema_model(x_t_val, x_cond_val, t_val)
            val_loss = F.mse_loss(eps_pred_val, noise_val)

        print(
            f"epoch=1, train_loss={train_loss.item():.6f}, val_loss={val_loss.item():.6f}, elapsed_seconds=0.00, is_best=False",
            flush=True,
        )
        print("dry-run complete: one train batch and one val batch executed; no checkpoints saved", flush=True)
        return

    history: list[dict] = []
    best_val_loss = float("inf")
    best_epoch = -1
    epochs_since_improve = 0
    ema_saved = True

    for epoch in range(1, MAX_EPOCHS + 1):
        epoch_start_time = time.time()
        print(f"Epoch [{epoch}/{MAX_EPOCHS}] started", flush=True)
        residual_norm, degraded_rel_norm, _ = build_epoch_training_arrays(
            aug_train,
            epoch=epoch,
            rel_mean=rel_mean,
            rel_std=rel_std,
        )
        residual_tensor = torch.from_numpy(residual_norm.transpose(0, 2, 1))
        cond_tensor = torch.from_numpy(degraded_rel_norm.transpose(0, 2, 1))
        train_loader = DataLoader(
            TensorDataset(residual_tensor, cond_tensor),
            batch_size=BATCH_SIZE,
            shuffle=True,
            num_workers=0,
            generator=torch.Generator().manual_seed(SEED + epoch),
        )

        model.train()
        train_total = 0.0
        train_count = 0
        for batch_idx, (batch_residual, batch_cond) in enumerate(train_loader, start=1):
            x0 = batch_residual.to(device=device, dtype=torch.float32)
            x_cond = batch_cond.to(device=device, dtype=torch.float32)
            t = diffusion.sample_timesteps(batch_size=x0.shape[0])
            x_t, noise = diffusion.q_sample(x0, t)

            optimizer.zero_grad()
            eps_pred = model(x_t, x_cond, t)
            loss = F.mse_loss(eps_pred, noise)
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                for ema_p, model_p in zip(ema_model.parameters(), model.parameters()):
                    ema_p.data.mul_(EMA_DECAY).add_(model_p.data, alpha=1.0 - EMA_DECAY)

            train_total += loss.item() * x0.shape[0]
            train_count += x0.shape[0]
            if batch_idx % 50 == 0 or batch_idx == len(train_loader):
                print(
                    f"Epoch {epoch}/{MAX_EPOCHS} batch {batch_idx}/{len(train_loader)} loss={loss.item():.6f}",
                    flush=True,
                )

        train_loss = train_total / train_count

        ema_model.eval()
        val_total = 0.0
        val_count = 0
        with torch.no_grad():
            for start in range(0, val_residual_tensor.shape[0], BATCH_SIZE):
                end = min(start + BATCH_SIZE, val_residual_tensor.shape[0])
                x0 = val_residual_tensor[start:end].to(device=device, dtype=torch.float32)
                x_cond = val_cond_tensor[start:end].to(device=device, dtype=torch.float32)
                t = val_t_all[start:end]
                noise = val_noise_all[start:end]
                x_t, _ = diffusion.q_sample(x0, t, noise=noise)
                eps_pred = ema_model(x_t, x_cond, t)
                loss = F.mse_loss(eps_pred, noise)
                val_total += loss.item() * x0.shape[0]
                val_count += x0.shape[0]
        val_loss = val_total / val_count

        is_best = val_loss < best_val_loss
        elapsed_seconds = time.time() - epoch_start_time
        epoch_row = {
            "epoch": epoch,
            "train_loss": f"{train_loss:.8f}",
            "val_loss": f"{val_loss:.8f}",
            "elapsed_seconds": f"{elapsed_seconds:.2f}",
            "is_best": is_best,
        }
        history.append(epoch_row)
        write_epoch_log_row(TRAIN_LOG_PATH, epoch_row)

        if is_best:
            best_val_loss = val_loss
            best_epoch = epoch
            epochs_since_improve = 0
            save_torch(BEST_MODEL_PATH, model.state_dict())
            save_torch(BEST_EMA_PATH, ema_model.state_dict())
        else:
            epochs_since_improve += 1

        print(
            f"epoch={epoch:03d}, train_loss={train_loss:.6f}, val_loss={val_loss:.6f}, elapsed_seconds={elapsed_seconds:.2f}, is_best={is_best}",
            flush=True,
        )

        if epoch >= MIN_EPOCHS and epochs_since_improve >= EARLY_STOP_PATIENCE:
            break

    save_torch(FINAL_MODEL_PATH, model.state_dict())
    save_torch(FINAL_EMA_PATH, ema_model.state_dict())
    save_npz(OUTPUT_NORM_PATH, rel_mean=rel_mean, rel_std=rel_std)

    print(f"best val_loss: {best_val_loss:.8f}", flush=True)
    print(f"best epoch: {best_epoch}", flush=True)
    print(f"final checkpoint path: {FINAL_MODEL_PATH}", flush=True)
    print(f"best checkpoint path: {BEST_MODEL_PATH}", flush=True)
    print(f"EMA checkpoints saved: {ema_saved}", flush=True)
    if ema_saved:
        print(f"final EMA checkpoint path: {FINAL_EMA_PATH}", flush=True)
        print(f"best EMA checkpoint path: {BEST_EMA_PATH}", flush=True)

    print("PHASE 2 COMPLETE.", flush=True)
    print("Generated:", flush=True)
    print("- outputs/stage3_indoor/receptive_field_expansion/ckpt_4block_final.pt", flush=True)
    print("- outputs/stage3_indoor/receptive_field_expansion/ckpt_4block_best.pt", flush=True)
    print("- outputs/stage3_indoor/receptive_field_expansion/train_4block_log.csv", flush=True)
    print(f"best val_loss: {best_val_loss:.8f}", flush=True)
    print(f"best epoch: {best_epoch}", flush=True)
    print(f"final checkpoint path: {FINAL_MODEL_PATH}", flush=True)
    print(f"best checkpoint path: {BEST_MODEL_PATH}", flush=True)
    if ema_saved:
        print(f"final EMA checkpoint path: {FINAL_EMA_PATH}", flush=True)
        print(f"best EMA checkpoint path: {BEST_EMA_PATH}", flush=True)


if __name__ == "__main__":
    main()
