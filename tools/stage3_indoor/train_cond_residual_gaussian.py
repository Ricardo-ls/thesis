from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import copy
import csv
import json
import random

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from diffusion.ddpm_utils import DDPMForwardProcess
from models.temporal_denoiser import TemporalDenoiser1D
from models.temporal_denoiser_conditional import ConditionalTemporalDenoiser1D
from tools.stage3.canonical_v1.utils import CanonicalConfig, run_kalman

try:
    from scipy.stats import wilcoxon

    SCIPY_AVAILABLE = True
except Exception:
    wilcoxon = None
    SCIPY_AVAILABLE = False


DATA_DIR = PROJECT_ROOT / "data" / "stage3_indoor"
TRAIN_PATH = DATA_DIR / "train_trajs.npy"
VAL_PATH = DATA_DIR / "val_trajs.npy"
CLEAN_PATH = DATA_DIR / "clean_trajs.npy"
NORM_PATH = DATA_DIR / "rel_norm_params_v2.npz"
UNCOND_CKPT_PATH = PROJECT_ROOT / "outputs" / "stage3_indoor" / "ddpm_indoor_v2" / "seed42" / "best_ema_model.pt"

OUTPUT_DIR = PROJECT_ROOT / "outputs" / "stage3_indoor" / "conditional_residual_ddpm_gaussian" / "seed42"
BEST_MODEL_PATH = OUTPUT_DIR / "best_model.pt"
BEST_EMA_PATH = OUTPUT_DIR / "best_ema_model.pt"
FINAL_MODEL_PATH = OUTPUT_DIR / "final_model.pt"
FINAL_EMA_PATH = OUTPUT_DIR / "final_ema_model.pt"
LOSS_CURVE_PATH = OUTPUT_DIR / "loss_curve.csv"
OUTPUT_NORM_PATH = OUTPUT_DIR / "rel_norm_params_v2.npz"
EVAL_DEGRADED_PATH = OUTPUT_DIR / "eval_degraded_gaussian.npy"
SUMMARY_PATH = OUTPUT_DIR / "cond_residual_gaussian_summary.csv"
PER_TRAJ_PATH = OUTPUT_DIR / "cond_residual_gaussian_per_traj.csv"
CONCLUSION_PATH = OUTPUT_DIR / "cond_residual_gaussian_conclusion.json"
FIG_PATH = OUTPUT_DIR / "cond_residual_gaussian_eval.png"

CENTER = np.array([1.5, 1.5], dtype=np.float32)
SEED = 42
TIMESTEPS = 100
MAX_EPOCHS = 60
MIN_EPOCHS = 20
EARLY_STOP_PATIENCE = 10
BATCH_SIZE = 256
LEARNING_RATE = 1e-3
EMA_DECAY = 0.999
TRAIN_NOISE_STD = 0.05
N_EVAL = 200
SAMPLE_SEEDS = [42, 43, 44, 45, 46]
METHODS = [
    "noisy_input",
    "kalman_cv",
    "unconditional_sdedit_t2",
    "cond_residual_full",
    "cond_residual_t20",
    "cond_residual_t10",
]


def set_all_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def ensure_required_inputs() -> None:
    required_paths = [
        TRAIN_PATH,
        VAL_PATH,
        CLEAN_PATH,
        NORM_PATH,
        UNCOND_CKPT_PATH,
    ]
    for path in required_paths:
        if not path.is_file():
            raise FileNotFoundError(f"Missing required input: {path}")


def save_torch(path: Path, payload) -> None:
    torch.save(payload, path)
    print(f"Saved: {path}")


def save_numpy(path: Path, array: np.ndarray) -> None:
    np.save(path, array.astype(np.float32))
    print(f"Saved: {path}")


def save_npz(path: Path, **arrays) -> None:
    np.savez(path, **arrays)
    print(f"Saved: {path}")


def save_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved: {path}")


def save_json(path: Path, payload: dict) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f"Saved: {path}")


def apply_augmentations(trajs: np.ndarray) -> np.ndarray:
    x = trajs[..., 0]
    y = trajs[..., 1]
    augments = [
        np.stack([x, y], axis=-1),
        np.stack([3.0 - x, y], axis=-1),
        np.stack([x, 3.0 - y], axis=-1),
        np.stack([3.0 - x, 3.0 - y], axis=-1),
        np.stack([CENTER[0] + (y - CENTER[1]), CENTER[1] - (x - CENTER[0])], axis=-1),
        np.stack([CENTER[0] - (y - CENTER[1]), CENTER[1] + (x - CENTER[0])], axis=-1),
    ]
    return np.concatenate(augments, axis=0).astype(np.float32)


def generate_gaussian_degraded(clean: np.ndarray, seed_base: int, epoch: int | None = None) -> np.ndarray:
    degraded = np.empty_like(clean, dtype=np.float32)
    for idx in range(clean.shape[0]):
        if epoch is None:
            seed = seed_base + idx
        else:
            seed = 100000 * epoch + idx
        rng = np.random.default_rng(seed)
        noise = rng.normal(0.0, TRAIN_NOISE_STD, size=(20, 2)).astype(np.float32)
        degraded[idx] = clean[idx] + noise
    return degraded


def to_rel(trajs: np.ndarray) -> np.ndarray:
    return (trajs[:, 1:, :] - trajs[:, :-1, :]).astype(np.float32)


def normalize_rel(rel: np.ndarray, rel_mean: np.ndarray, rel_std: np.ndarray) -> np.ndarray:
    return ((rel - rel_mean[None, None, :]) / rel_std[None, None, :]).astype(np.float32)


def reconstruct_from_rel(start_points: np.ndarray, rel: np.ndarray) -> np.ndarray:
    abs_hat = np.zeros((rel.shape[0], 20, 2), dtype=np.float32)
    abs_hat[:, 0, :] = start_points.astype(np.float32)
    abs_hat[:, 1:, :] = start_points[:, None, :] + np.cumsum(rel, axis=1)
    return abs_hat


def load_state_dict_flexible(model: torch.nn.Module, checkpoint_path: Path, device: torch.device) -> None:
    state = torch.load(checkpoint_path, map_location=device)
    if isinstance(state, dict) and "model_state_dict" in state:
        model.load_state_dict(state["model_state_dict"])
    elif isinstance(state, dict) and "model" in state:
        model.load_state_dict(state["model"])
    else:
        model.load_state_dict(state)


def kalman_wrapper(degraded_abs: np.ndarray) -> np.ndarray:
    obs_mask = np.ones((degraded_abs.shape[0], degraded_abs.shape[1]), dtype=np.uint8)
    config = CanonicalConfig()
    return run_kalman(
        degraded_abs.astype(np.float32),
        obs_mask,
        dt=config.kalman_dt,
        process_var=config.kalman_process_var,
        measure_var=config.kalman_measure_var,
    ).astype(np.float32)


def reverse_step(
    x_t: torch.Tensor,
    eps_pred: torch.Tensor,
    t_idx: int,
    diffusion: DDPMForwardProcess,
) -> torch.Tensor:
    alpha_t = diffusion.alphas[t_idx]
    alpha_bar_t = diffusion.alpha_bars[t_idx]
    beta_t = diffusion.betas[t_idx]
    mean = (1.0 / torch.sqrt(alpha_t)) * (
        x_t - beta_t / torch.sqrt(1.0 - alpha_bar_t) * eps_pred
    )
    if t_idx > 0:
        z = torch.randn_like(x_t)
        return mean + torch.sqrt(beta_t) * z
    return mean


def run_unconditional_sdedit_t2(
    degraded_abs: np.ndarray,
    model: TemporalDenoiser1D,
    diffusion: DDPMForwardProcess,
    rel_mean: np.ndarray,
    rel_std: np.ndarray,
    sdedit_seed: int,
    device: torch.device,
) -> np.ndarray:
    torch.manual_seed(sdedit_seed)
    np.random.seed(sdedit_seed)

    degraded_rel = to_rel(degraded_abs)
    rel_norm = normalize_rel(degraded_rel, rel_mean, rel_std)
    x0 = torch.from_numpy(rel_norm.transpose(0, 2, 1)).to(device=device, dtype=torch.float32)
    t = torch.full((x0.shape[0],), 2, device=device, dtype=torch.long)
    x_t, _ = diffusion.q_sample(x0, t)

    with torch.no_grad():
        for t_idx in reversed(range(3)):
            t_cur = torch.full((x_t.shape[0],), t_idx, device=device, dtype=torch.long)
            eps_pred = model(x_t, t_cur)
            x_t = reverse_step(x_t, eps_pred, t_idx, diffusion)

    rel_hat_norm = x_t.permute(0, 2, 1).cpu().numpy().astype(np.float32)
    rel_hat = (rel_hat_norm * rel_std[None, None, :] + rel_mean[None, None, :]).astype(np.float32)
    return reconstruct_from_rel(degraded_abs[:, 0, :], rel_hat)


def sample_conditional_residual(
    degraded_abs: np.ndarray,
    model: ConditionalTemporalDenoiser1D,
    diffusion: DDPMForwardProcess,
    rel_mean: np.ndarray,
    rel_std: np.ndarray,
    start_t: int,
    sample_seed: int,
    device: torch.device,
) -> np.ndarray:
    torch.manual_seed(sample_seed)
    np.random.seed(sample_seed)

    degraded_rel = to_rel(degraded_abs)
    degraded_rel_norm = normalize_rel(degraded_rel, rel_mean, rel_std)
    x_cond = torch.from_numpy(degraded_rel_norm.transpose(0, 2, 1)).to(device=device, dtype=torch.float32)

    if start_t == TIMESTEPS - 1:
        x_t = torch.randn((degraded_abs.shape[0], 2, 19), device=device, dtype=torch.float32)
    else:
        eps = torch.randn((degraded_abs.shape[0], 2, 19), device=device, dtype=torch.float32)
        alpha_bar_t = diffusion.alpha_bars[start_t]
        x_t = torch.sqrt(1.0 - alpha_bar_t) * eps

    with torch.no_grad():
        for t_idx in reversed(range(start_t + 1)):
            t_cur = torch.full((x_t.shape[0],), t_idx, device=device, dtype=torch.long)
            eps_pred = model(x_t, x_cond, t_cur)
            x_t = reverse_step(x_t, eps_pred, t_idx, diffusion)

    residual_hat_norm = x_t.permute(0, 2, 1).cpu().numpy().astype(np.float32)
    final_rel_norm = degraded_rel_norm + residual_hat_norm
    final_rel = (final_rel_norm * rel_std[None, None, :] + rel_mean[None, None, :]).astype(np.float32)
    return reconstruct_from_rel(degraded_abs[:, 0, :], final_rel)


def compute_per_traj_metrics(pred: np.ndarray, clean: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    errors = np.linalg.norm(pred - clean, axis=-1)
    ade = errors.mean(axis=1).astype(np.float32)
    rmse = np.sqrt(np.mean(errors**2, axis=1)).astype(np.float32)
    acc = pred[:, 2:, :] - 2.0 * pred[:, 1:-1, :] + pred[:, :-2, :]
    smooth = np.linalg.norm(acc, axis=-1).mean(axis=1).astype(np.float32)
    return ade, rmse, smooth


def safe_wilcoxon(delta: np.ndarray) -> float:
    if not SCIPY_AVAILABLE:
        return float("nan")
    if np.allclose(delta, 0.0):
        return 1.0
    try:
        return float(wilcoxon(delta, alternative="less").pvalue)
    except Exception:
        return float("nan")


def format_pvalue(value: float) -> str:
    if np.isnan(value):
        return "nan"
    return f"{value:.4g}"


def build_epoch_training_arrays(
    clean_trajs: np.ndarray,
    epoch: int,
    rel_mean: np.ndarray,
    rel_std: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    degraded = generate_gaussian_degraded(clean_trajs, seed_base=0, epoch=epoch)
    clean_rel_norm = normalize_rel(to_rel(clean_trajs), rel_mean, rel_std)
    degraded_rel_norm = normalize_rel(to_rel(degraded), rel_mean, rel_std)
    residual_norm = (clean_rel_norm - degraded_rel_norm).astype(np.float32)
    return residual_norm, degraded_rel_norm, degraded


def validate_shapes(train_trajs: np.ndarray, val_trajs: np.ndarray, clean_trajs: np.ndarray, rel_mean: np.ndarray, rel_std: np.ndarray) -> None:
    if train_trajs.shape != (10000, 20, 2):
        raise ValueError(f"Expected train_trajs shape (10000, 20, 2), got {train_trajs.shape}")
    if val_trajs.shape != (2000, 20, 2):
        raise ValueError(f"Expected val_trajs shape (2000, 20, 2), got {val_trajs.shape}")
    if clean_trajs.shape[1:] != (20, 2):
        raise ValueError(f"Expected clean_trajs shape (*, 20, 2), got {clean_trajs.shape}")
    if rel_mean.shape != (2,) or rel_std.shape != (2,):
        raise ValueError(f"Expected rel_mean/rel_std shape (2,), got {rel_mean.shape} and {rel_std.shape}")


def main() -> None:
    ensure_required_inputs()
    set_all_seeds(SEED)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    train_trajs = np.load(TRAIN_PATH).astype(np.float32)
    val_trajs = np.load(VAL_PATH).astype(np.float32)
    clean_all = np.load(CLEAN_PATH).astype(np.float32)
    norm = np.load(NORM_PATH)
    rel_mean = norm["rel_mean"].astype(np.float32)
    rel_std = norm["rel_std"].astype(np.float32)
    validate_shapes(train_trajs, val_trajs, clean_all, rel_mean, rel_std)

    aug_train = apply_augmentations(train_trajs)
    if aug_train.shape != (60000, 20, 2):
        raise ValueError(f"Expected aug_train shape (60000, 20, 2), got {aug_train.shape}")

    first_epoch_residual_norm, _, _ = build_epoch_training_arrays(aug_train, epoch=1, rel_mean=rel_mean, rel_std=rel_std)
    residual_norm_std = float(first_epoch_residual_norm.std())
    residual_norm_mean = float(first_epoch_residual_norm.mean())

    print("=== Step 3g-A: Conditional Residual DDPM Gaussian-only Scout ===")
    print(f"train augmented shape: {aug_train.shape}")
    print(f"rel_mean: ({rel_mean[0]:.6f}, {rel_mean[1]:.6f})")
    print(f"rel_std:  ({rel_std[0]:.6f}, {rel_std[1]:.6f})")
    print(f"residual_norm std: {residual_norm_std:.4f}  预期 0.3~1.5")
    print(f"residual_norm mean: {residual_norm_mean:.4f}  预期接近 0")
    if residual_norm_std < 0.1:
        print("⚠️ WARNING: residual signal too weak, DDPM may not learn effectively")
        print("Consider using separate normalization for residual")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    diffusion = DDPMForwardProcess(timesteps=TIMESTEPS, device=device)

    model = ConditionalTemporalDenoiser1D(max_timesteps=TIMESTEPS, in_channels=4, hidden_dim=128).to(device)
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
    val_t_all = torch.from_numpy(val_rng.integers(0, TIMESTEPS, size=(val_residual_tensor.shape[0],), dtype=np.int64)).to(
        device=device,
        dtype=torch.long,
    )
    val_noise_all = torch.from_numpy(
        val_rng.normal(0.0, 1.0, size=(val_residual_tensor.shape[0], 2, 19)).astype(np.float32)
    ).to(device=device, dtype=torch.float32)

    history: list[dict] = []
    best_val_loss = float("inf")
    best_epoch = -1
    epochs_since_improve = 0

    for epoch in range(1, MAX_EPOCHS + 1):
        residual_norm, degraded_rel_norm, _ = build_epoch_training_arrays(aug_train, epoch=epoch, rel_mean=rel_mean, rel_std=rel_std)
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
        for batch_residual, batch_cond in train_loader:
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
        val_loss_ema = val_total / val_count

        history.append(
            {
                "epoch": epoch,
                "train_loss": f"{train_loss:.8f}",
                "val_loss_ema": f"{val_loss_ema:.8f}",
            }
        )

        if val_loss_ema < best_val_loss:
            best_val_loss = val_loss_ema
            best_epoch = epoch
            epochs_since_improve = 0
            save_torch(BEST_MODEL_PATH, model.state_dict())
            save_torch(BEST_EMA_PATH, ema_model.state_dict())
        else:
            epochs_since_improve += 1

        print(f"Epoch {epoch:03d}/{MAX_EPOCHS}  train_loss={train_loss:.6f}  val_loss_ema={val_loss_ema:.6f}")

        if epoch >= MIN_EPOCHS and epochs_since_improve >= EARLY_STOP_PATIENCE:
            break

    save_torch(FINAL_MODEL_PATH, model.state_dict())
    save_torch(FINAL_EMA_PATH, ema_model.state_dict())
    save_csv(LOSS_CURVE_PATH, history, ["epoch", "train_loss", "val_loss_ema"])
    save_npz(OUTPUT_NORM_PATH, rel_mean=rel_mean, rel_std=rel_std)

    best_cond_model = ConditionalTemporalDenoiser1D(max_timesteps=TIMESTEPS, in_channels=4, hidden_dim=128).to(device)
    load_state_dict_flexible(best_cond_model, BEST_EMA_PATH, device)
    best_cond_model.eval()

    uncond_model = TemporalDenoiser1D(max_timesteps=TIMESTEPS, in_channels=2, hidden_dim=128).to(device)
    load_state_dict_flexible(uncond_model, UNCOND_CKPT_PATH, device)
    uncond_model.eval()

    clean = clean_all[:N_EVAL]
    if clean.shape != (200, 20, 2):
        raise ValueError(f"Expected clean shape (200, 20, 2), got {clean.shape}")
    eval_degraded = generate_gaussian_degraded(clean, seed_base=42, epoch=None)
    save_numpy(EVAL_DEGRADED_PATH, eval_degraded)

    if not SCIPY_AVAILABLE:
        print("WARNING: scipy not available; Wilcoxon tests skipped")

    method_preds: dict[str, np.ndarray] = {"noisy_input": eval_degraded}
    method_preds["kalman_cv"] = kalman_wrapper(eval_degraded)

    uncond_preds = []
    for sample_seed in SAMPLE_SEEDS:
        uncond_preds.append(
            run_unconditional_sdedit_t2(
                degraded_abs=eval_degraded,
                model=uncond_model,
                diffusion=diffusion,
                rel_mean=rel_mean,
                rel_std=rel_std,
                sdedit_seed=sample_seed,
                device=device,
            )
        )
    method_preds["unconditional_sdedit_t2"] = np.mean(np.stack(uncond_preds, axis=0), axis=0).astype(np.float32)

    cond_start_map = {
        "cond_residual_full": 99,
        "cond_residual_t20": 20,
        "cond_residual_t10": 10,
    }
    for method, start_t in cond_start_map.items():
        preds = []
        for sample_seed in SAMPLE_SEEDS:
            preds.append(
                sample_conditional_residual(
                    degraded_abs=eval_degraded,
                    model=best_cond_model,
                    diffusion=diffusion,
                    rel_mean=rel_mean,
                    rel_std=rel_std,
                    start_t=start_t,
                    sample_seed=sample_seed,
                    device=device,
                )
            )
        method_preds[method] = np.mean(np.stack(preds, axis=0), axis=0).astype(np.float32)

    metrics: dict[str, dict[str, np.ndarray]] = {}
    for method in METHODS:
        ade, rmse, smooth = compute_per_traj_metrics(method_preds[method], clean)
        metrics[method] = {
            "ADE": ade,
            "RMSE": rmse,
            "smooth": smooth,
        }

    noisy_ade = metrics["noisy_input"]["ADE"]
    uncond_ade = metrics["unconditional_sdedit_t2"]["ADE"]

    summary_rows: list[dict] = []
    per_traj_rows: list[dict] = []
    conditional_methods = ["cond_residual_full", "cond_residual_t20", "cond_residual_t10"]

    best_cond_method = None
    best_cond_ade_mean = float("inf")

    for method in METHODS:
        ade = metrics[method]["ADE"]
        rmse = metrics[method]["RMSE"]
        smooth = metrics[method]["smooth"]

        delta_noisy = ade - noisy_ade
        delta_uncond = ade - uncond_ade

        if method == "noisy_input":
            delta_noisy_mean = 0.0
            improved_noisy = 0.0
            p_noisy = float("nan")
            delta_uncond_mean = float(np.mean(delta_uncond))
            improved_uncond = float(np.mean(delta_uncond < 0))
            p_uncond = safe_wilcoxon(delta_uncond)
        elif method == "unconditional_sdedit_t2":
            delta_noisy_mean = float(np.mean(delta_noisy))
            improved_noisy = float(np.mean(delta_noisy < 0))
            p_noisy = safe_wilcoxon(delta_noisy)
            delta_uncond_mean = 0.0
            improved_uncond = 0.0
            p_uncond = float("nan")
        else:
            delta_noisy_mean = float(np.mean(delta_noisy))
            improved_noisy = float(np.mean(delta_noisy < 0))
            p_noisy = safe_wilcoxon(delta_noisy)
            delta_uncond_mean = float(np.mean(delta_uncond))
            improved_uncond = float(np.mean(delta_uncond < 0))
            p_uncond = safe_wilcoxon(delta_uncond)

        ade_mean = float(np.mean(ade))
        if method in conditional_methods and ade_mean < best_cond_ade_mean:
            best_cond_ade_mean = ade_mean
            best_cond_method = method

        summary_rows.append(
            {
                "method": method,
                "N": N_EVAL,
                "ADE_mean": ade_mean,
                "ADE_std": float(np.std(ade, ddof=0)),
                "RMSE_mean": float(np.mean(rmse)),
                "RMSE_std": float(np.std(rmse, ddof=0)),
                "smooth_mean": float(np.mean(smooth)),
                "smooth_std": float(np.std(smooth, ddof=0)),
                "delta_ADE_vs_noisy_mean": delta_noisy_mean,
                "improved_fraction_vs_noisy": improved_noisy,
                "wilcoxon_p_vs_noisy": p_noisy,
                "delta_ADE_vs_uncond_mean": delta_uncond_mean,
                "improved_fraction_vs_uncond": improved_uncond,
                "wilcoxon_p_vs_uncond": p_uncond,
            }
        )

        for traj_idx in range(N_EVAL):
            per_traj_rows.append(
                {
                    "traj_idx": traj_idx,
                    "method": method,
                    "ADE": float(ade[traj_idx]),
                    "RMSE": float(rmse[traj_idx]),
                    "smooth": float(smooth[traj_idx]),
                    "noisy_ADE": float(noisy_ade[traj_idx]),
                    "uncond_ADE": float(uncond_ade[traj_idx]),
                    "delta_ADE_vs_noisy": float(delta_noisy[traj_idx]),
                    "delta_ADE_vs_uncond": float(delta_uncond[traj_idx]),
                    "improved_vs_noisy": int(delta_noisy[traj_idx] < 0),
                    "improved_vs_uncond": int(delta_uncond[traj_idx] < 0),
                }
            )

    save_csv(
        SUMMARY_PATH,
        summary_rows,
        [
            "method",
            "N",
            "ADE_mean",
            "ADE_std",
            "RMSE_mean",
            "RMSE_std",
            "smooth_mean",
            "smooth_std",
            "delta_ADE_vs_noisy_mean",
            "improved_fraction_vs_noisy",
            "wilcoxon_p_vs_noisy",
            "delta_ADE_vs_uncond_mean",
            "improved_fraction_vs_uncond",
            "wilcoxon_p_vs_uncond",
        ],
    )
    save_csv(
        PER_TRAJ_PATH,
        per_traj_rows,
        [
            "traj_idx",
            "method",
            "ADE",
            "RMSE",
            "smooth",
            "noisy_ADE",
            "uncond_ADE",
            "delta_ADE_vs_noisy",
            "delta_ADE_vs_uncond",
            "improved_vs_noisy",
            "improved_vs_uncond",
        ],
    )

    summary_map = {row["method"]: row for row in summary_rows}
    noisy_ade_mean = float(summary_map["noisy_input"]["ADE_mean"])
    uncond_ade_mean = float(summary_map["unconditional_sdedit_t2"]["ADE_mean"])
    best_cond_row = summary_map[best_cond_method]
    best_cond_delta_vs_noisy = float(best_cond_row["delta_ADE_vs_noisy_mean"])
    best_cond_delta_vs_uncond = float(best_cond_row["delta_ADE_vs_uncond_mean"])
    best_cond_p_vs_noisy = float(best_cond_row["wilcoxon_p_vs_noisy"])
    best_cond_p_vs_uncond = float(best_cond_row["wilcoxon_p_vs_uncond"])

    conditional_beats_unconditional = (
        best_cond_ade_mean < uncond_ade_mean
        and best_cond_delta_vs_uncond < 0
        and not np.isnan(best_cond_p_vs_uncond)
        and best_cond_p_vs_uncond < 0.05
    )
    conditional_beats_noisy = (
        best_cond_ade_mean < noisy_ade_mean
        and best_cond_delta_vs_noisy < 0
        and not np.isnan(best_cond_p_vs_noisy)
        and best_cond_p_vs_noisy < 0.05
    )

    if conditional_beats_unconditional:
        interpretation = "conditional_improves_over_unconditional"
    elif conditional_beats_noisy:
        interpretation = "conditional_improves_only_vs_noisy"
    else:
        interpretation = "conditional_no_clear_gain"

    conclusion = {
        "best_epoch": best_epoch,
        "best_val_loss": float(best_val_loss),
        "noisy_ADE_mean": noisy_ade_mean,
        "uncond_sdedit_t2_ADE_mean": uncond_ade_mean,
        "best_cond_method": best_cond_method,
        "best_cond_ADE_mean": best_cond_ade_mean,
        "best_cond_delta_vs_noisy": best_cond_delta_vs_noisy,
        "best_cond_delta_vs_uncond": best_cond_delta_vs_uncond,
        "best_cond_p_vs_noisy": best_cond_p_vs_noisy,
        "best_cond_p_vs_uncond": best_cond_p_vs_uncond,
        "conditional_beats_noisy": conditional_beats_noisy,
        "conditional_beats_unconditional": conditional_beats_unconditional,
        "interpretation": interpretation,
    }
    save_json(CONCLUSION_PATH, conclusion)

    history_epochs = np.array([int(row["epoch"]) for row in history], dtype=np.int32)
    history_train = np.array([float(row["train_loss"]) for row in history], dtype=np.float32)
    history_val = np.array([float(row["val_loss_ema"]) for row in history], dtype=np.float32)

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    ade_means = [summary_map[m]["ADE_mean"] for m in METHODS]
    axes[0, 0].bar(range(len(METHODS)), ade_means, color=["#8c8c8c", "#4c78a8", "#72b7b2", "#f58518", "#e45756", "#54a24b"])
    axes[0, 0].set_xticks(range(len(METHODS)))
    axes[0, 0].set_xticklabels(METHODS, rotation=25, ha="right")
    axes[0, 0].set_title("ADE Mean")
    axes[0, 0].set_ylabel("ADE")

    box_data = [metrics[m]["ADE"] - uncond_ade for m in conditional_methods]
    axes[0, 1].boxplot(box_data, labels=conditional_methods, showfliers=False)
    axes[0, 1].axhline(0.0, color="black", linewidth=1.0, linestyle="--")
    axes[0, 1].set_title("delta_ADE_vs_uncond")
    axes[0, 1].tick_params(axis="x", rotation=20)

    best_cond_ade = metrics[best_cond_method]["ADE"]
    axes[0, 2].scatter(uncond_ade, best_cond_ade, s=16, alpha=0.7)
    min_lim = float(min(uncond_ade.min(), best_cond_ade.min()))
    max_lim = float(max(uncond_ade.max(), best_cond_ade.max()))
    axes[0, 2].plot([min_lim, max_lim], [min_lim, max_lim], linestyle="--", color="black")
    axes[0, 2].set_xlabel("unconditional ADE")
    axes[0, 2].set_ylabel("best conditional ADE")
    axes[0, 2].set_title("Per-trajectory ADE")

    smooth_means = [summary_map[m]["smooth_mean"] for m in METHODS]
    axes[1, 0].bar(range(len(METHODS)), smooth_means, color=["#8c8c8c", "#4c78a8", "#72b7b2", "#f58518", "#e45756", "#54a24b"])
    axes[1, 0].set_xticks(range(len(METHODS)))
    axes[1, 0].set_xticklabels(METHODS, rotation=25, ha="right")
    axes[1, 0].set_title("Smooth Mean")
    axes[1, 0].set_ylabel("smooth")

    ax_traj = axes[1, 1]
    colors = {
        "clean": "#1f77b4",
        "degraded": "#7f7f7f",
        "unconditional_sdedit_t2": "#2ca02c",
        "best_cond": "#d62728",
    }
    for idx in [0, 1]:
        alpha = 1.0 if idx == 0 else 0.65
        suffix = "idx0" if idx == 0 else "idx1"
        ax_traj.plot(clean[idx, :, 0], clean[idx, :, 1], color=colors["clean"], linewidth=2.0, alpha=alpha, label=f"clean_{suffix}")
        ax_traj.plot(eval_degraded[idx, :, 0], eval_degraded[idx, :, 1], color=colors["degraded"], linewidth=1.5, alpha=alpha, linestyle="--", label=f"degraded_{suffix}")
        ax_traj.plot(method_preds["unconditional_sdedit_t2"][idx, :, 0], method_preds["unconditional_sdedit_t2"][idx, :, 1], color=colors["unconditional_sdedit_t2"], linewidth=1.8, alpha=alpha, label=f"uncond_{suffix}")
        ax_traj.plot(method_preds[best_cond_method][idx, :, 0], method_preds[best_cond_method][idx, :, 1], color=colors["best_cond"], linewidth=1.8, alpha=alpha, label=f"best_cond_{suffix}")
    ax_traj.set_title("Trajectory Examples idx=0,1")
    ax_traj.set_aspect("equal", adjustable="box")
    ax_traj.legend(fontsize=7, ncol=2)

    axes[1, 2].plot(history_epochs, history_train, label="train_loss")
    axes[1, 2].plot(history_epochs, history_val, label="val_loss_ema")
    axes[1, 2].axvline(best_epoch, color="black", linestyle="--", linewidth=1.0)
    axes[1, 2].set_title("Loss Curve")
    axes[1, 2].set_xlabel("epoch")
    axes[1, 2].legend()

    plt.tight_layout()
    fig.savefig(FIG_PATH, dpi=180)
    plt.close(fig)
    print(f"Saved: {FIG_PATH}")

    print(f"best epoch: {best_epoch}")
    print(f"best val_loss: {best_val_loss:.6f}")
    print("")
    print("=== Evaluation: gaussian_medium, N=200 ===")
    print(
        f"{'method':28s}  "
        f"{'ADE_mean':>10s}  "
        f"{'ADE_std':>10s}  "
        f"{'RMSE_mean':>10s}  "
        f"{'smooth':>10s}  "
        f"{'dADE_noisy':>12s}  "
        f"{'imp_noisy':>10s}  "
        f"{'p_noisy':>12s}  "
        f"{'dADE_uncond':>12s}  "
        f"{'imp_uncond':>11s}  "
        f"{'p_uncond':>12s}"
    )
    for method in METHODS:
        row = summary_map[method]
        print(
            f"{method:28s}  "
            f"{row['ADE_mean']:10.4f}  "
            f"{row['ADE_std']:10.4f}  "
            f"{row['RMSE_mean']:10.4f}  "
            f"{row['smooth_mean']:10.4f}  "
            f"{row['delta_ADE_vs_noisy_mean']:12.4f}  "
            f"{row['improved_fraction_vs_noisy']:10.3f}  "
            f"{format_pvalue(float(row['wilcoxon_p_vs_noisy'])):>12s}  "
            f"{row['delta_ADE_vs_uncond_mean']:12.4f}  "
            f"{row['improved_fraction_vs_uncond']:11.3f}  "
            f"{format_pvalue(float(row['wilcoxon_p_vs_uncond'])):>12s}"
        )

    print("")
    print("=== 关键结论 ===")
    print(f"noisy_input ADE: {noisy_ade_mean:.4f}")
    print(f"unconditional_sdedit_t2 ADE: {uncond_ade_mean:.4f}")
    print(f"best conditional method: {best_cond_method}")
    print(f"best conditional ADE: {best_cond_ade_mean:.4f}")
    print(f"delta vs unconditional: {best_cond_delta_vs_uncond:.4f}")
    print(f"Wilcoxon p vs unconditional: {format_pvalue(best_cond_p_vs_uncond)}")
    if conditional_beats_unconditional:
        print("✅ Conditional residual DDPM beats unconditional SDEdit-t2 with paired statistical support.")
    elif conditional_beats_noisy:
        print("⚠️ Conditional residual DDPM beats noisy_input but not unconditional SDEdit-t2.")
    else:
        print("❌ Conditional residual DDPM does not show clear gain.")


if __name__ == "__main__":
    main()
