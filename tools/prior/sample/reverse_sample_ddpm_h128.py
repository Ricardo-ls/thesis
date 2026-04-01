from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn


# ============================================================
# 1) 模型导入：这里要和你训练时使用的模型类保持一致
#    优先尝试从独立模型文件导入；如果没有，就从训练脚本导入
# ============================================================
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[3]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.temporal_denoiser import TemporalDenoiser1D


# ============================================================
# 2) 一些工具函数
# ============================================================
def extract(a, t, x_shape):
    """
    从 shape=[T] 的系数张量 a 中，按 batch 的 t 取值提取，
    并 reshape 成可广播到 x_shape 的形式。
    """
    out = a.gather(0, t)
    return out.view(t.shape[0], *([1] * (len(x_shape) - 1)))


def rel_to_abs(rel):
    """
    rel: numpy array, shape = [N, 2, 19]
    返回: abs_traj, shape = [N, 20, 2]
    起点固定在原点 (0, 0)
    """
    rel = np.transpose(rel, (0, 2, 1))   # [N, 19, 2]
    cumsum = np.cumsum(rel, axis=1)      # [N, 19, 2]
    origin = np.zeros((rel.shape[0], 1, 2), dtype=rel.dtype)
    abs_traj = np.concatenate([origin, cumsum], axis=1)  # [N, 20, 2]
    return abs_traj


# ============================================================
# 3) DDPM 采样器
# ============================================================
class DDPMSampler:
    def __init__(self, timesteps=100, beta_start=1e-4, beta_end=0.02, device="cpu"):
        self.device = device
        self.timesteps = timesteps

        self.betas = torch.linspace(beta_start, beta_end, timesteps, device=device)
        self.alphas = 1.0 - self.betas
        self.alpha_bars = torch.cumprod(self.alphas, dim=0)

        self.alpha_bars_prev = torch.cat(
            [torch.tensor([1.0], device=device), self.alpha_bars[:-1]], dim=0
        )

        self.sqrt_alpha_bars = torch.sqrt(self.alpha_bars)
        self.sqrt_one_minus_alpha_bars = torch.sqrt(1.0 - self.alpha_bars)
        self.sqrt_recip_alphas = torch.sqrt(1.0 / self.alphas)

        # DDPM posterior variance
        self.posterior_variance = (
            self.betas * (1.0 - self.alpha_bars_prev) / (1.0 - self.alpha_bars)
        )
        self.posterior_variance = torch.clamp(self.posterior_variance, min=1e-20)

    def q_sample(self, x0, t, noise=None):
        """
        前向加噪:
        x_t = sqrt(alpha_bar_t) * x0 + sqrt(1 - alpha_bar_t) * noise
        """
        if noise is None:
            noise = torch.randn_like(x0)

        sqrt_ab = extract(self.sqrt_alpha_bars, t, x0.shape)
        sqrt_1mab = extract(self.sqrt_one_minus_alpha_bars, t, x0.shape)
        return sqrt_ab * x0 + sqrt_1mab * noise

    def predict_x0_from_eps(self, xt, t, eps_pred):
        """
        从模型预测噪声 eps_pred 反推 x0
        """
        sqrt_ab = extract(self.sqrt_alpha_bars, t, xt.shape)
        sqrt_1mab = extract(self.sqrt_one_minus_alpha_bars, t, xt.shape)
        x0_pred = (xt - sqrt_1mab * eps_pred) / torch.clamp(sqrt_ab, min=1e-8)
        return x0_pred

    @torch.no_grad()
    def p_sample(self, model, xt, t_scalar):
        """
        单步反向采样: x_t -> x_{t-1}
        """
        b = xt.shape[0]
        t = torch.full((b,), t_scalar, device=self.device, dtype=torch.long)

        eps_pred = model(xt, t)

        beta_t = extract(self.betas, t, xt.shape)
        sqrt_1mab_t = extract(self.sqrt_one_minus_alpha_bars, t, xt.shape)
        sqrt_recip_alpha_t = extract(self.sqrt_recip_alphas, t, xt.shape)

        # DDPM 反向均值
        model_mean = sqrt_recip_alpha_t * (
            xt - (beta_t / torch.clamp(sqrt_1mab_t, min=1e-8)) * eps_pred
        )

        if t_scalar == 0:
            return model_mean

        posterior_var_t = extract(self.posterior_variance, t, xt.shape)
        noise = torch.randn_like(xt)
        return model_mean + torch.sqrt(posterior_var_t) * noise

    @torch.no_grad()
    def sample(self, model, num_samples, channels=2, seq_len=19, return_history=False):
        """
        从纯噪声开始完整 reverse sampling
        """
        xt = torch.randn(num_samples, channels, seq_len, device=self.device)

        history = {}
        record_steps = {self.timesteps - 1, 75, 50, 25, 0}

        for t in reversed(range(self.timesteps)):
            xt = self.p_sample(model, xt, t)
            if return_history and t in record_steps:
                history[t] = xt.detach().cpu().numpy()

        if return_history:
            return xt, history
        return xt


# ============================================================
# 4) 画图
# ============================================================
def plot_real_vs_generated(real_abs, gen_abs, save_path):
    """
    real_abs: [N, 20, 2]
    gen_abs : [N, 20, 2]
    """
    n_show = min(len(real_abs), len(gen_abs), 16)

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    for i in range(n_show):
        traj = real_abs[i]
        plt.plot(traj[:, 0], traj[:, 1], alpha=0.8)
        plt.scatter(traj[0, 0], traj[0, 1], s=18)
        plt.scatter(traj[-1, 0], traj[-1, 1], s=18, marker="x")
    plt.title("Real trajectories (origin aligned)")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.axis("equal")
    plt.grid(True)

    plt.subplot(1, 2, 2)
    for i in range(n_show):
        traj = gen_abs[i]
        plt.plot(traj[:, 0], traj[:, 1], alpha=0.8)
        plt.scatter(traj[0, 0], traj[0, 1], s=18)
        plt.scatter(traj[-1, 0], traj[-1, 1], s=18, marker="x")
    plt.title("Generated trajectories")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.axis("equal")
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()


def plot_denoise_check(x0_abs, xt_abs, x0_pred_abs, save_path, t_vis):
    plt.figure(figsize=(15, 4))

    plt.subplot(1, 3, 1)
    plt.plot(x0_abs[:, 0], x0_abs[:, 1], marker="o")
    plt.title("Real x0")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.axis("equal")
    plt.grid(True)

    plt.subplot(1, 3, 2)
    plt.plot(xt_abs[:, 0], xt_abs[:, 1], marker="o")
    plt.title(f"Noisy xt (t={t_vis})")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.axis("equal")
    plt.grid(True)

    plt.subplot(1, 3, 3)
    plt.plot(x0_pred_abs[:, 0], x0_pred_abs[:, 1], marker="o")
    plt.title("Predicted x0 from xt")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.axis("equal")
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()


# ============================================================
# 5) 主程序
# ============================================================
def main():
    # -----------------------------
    # 路径与超参数
    # -----------------------------
    data_path = PROJECT_ROOT / "datasets" / "processed" / "data_eth_20_rel_q20.npy"
    ckpt_path = PROJECT_ROOT / "outputs" / "prior" / "train" / "ddpm_minimal_q20_h128" / "best_model.pt"
    out_dir = PROJECT_ROOT / "outputs" / "prior" / "sample" / "ddpm_minimal_q20_h128" / "reverse_sampling_check_512"
    out_dir.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 这些要与训练时保持一致
    timesteps = 100
    hidden_dim = 128
    seq_len = 19
    channels = 2
    num_generate = 512

    print(f"device      = {device}")
    print(f"data_path    = {data_path}")
    print(f"ckpt_path    = {ckpt_path}")
    print(f"output_dir   = {out_dir}")

    if not data_path.exists():
        raise FileNotFoundError(f"找不到数据文件: {data_path}")
    if not ckpt_path.exists():
        raise FileNotFoundError(f"找不到模型文件: {ckpt_path}")

    # -----------------------------
    # 读取真实数据
    # 原始 shape: [N, 19, 2]
    # 转成      : [N, 2, 19]
    # -----------------------------
    real_rel_np = np.load(data_path).astype(np.float32)
    print("real_rel_np shape =", real_rel_np.shape)

    real_rel_torch = torch.from_numpy(real_rel_np).permute(0, 2, 1).contiguous()
    real_abs_np = rel_to_abs(real_rel_torch.numpy())

    # -----------------------------
    # 构建模型
    # 这里默认模型初始化参数为 in_channels / hidden_dim
    # 如果你训练时类的参数名不同，只改这里一处
    # -----------------------------
    model = TemporalDenoiser1D(
        max_timesteps=timesteps,
        in_channels=channels,
        hidden_dim=hidden_dim
    ).to(device)

    # -----------------------------
    # 加载 checkpoint
    # 兼容几种常见保存格式
    # -----------------------------
    ckpt = torch.load(ckpt_path, map_location=device)

    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        state_dict = ckpt["model_state_dict"]
    elif isinstance(ckpt, dict) and "state_dict" in ckpt:
        state_dict = ckpt["state_dict"]
    elif isinstance(ckpt, dict):
        state_dict = ckpt
    else:
        raise ValueError("无法识别 checkpoint 格式，请检查 best_model.pt 的保存方式。")

    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    print("missing keys   =", missing)
    print("unexpected keys=", unexpected)

    model.eval()

    # -----------------------------
    # 构建采样器
    # -----------------------------
    sampler = DDPMSampler(
        timesteps=timesteps,
        beta_start=1e-4,
        beta_end=0.02,
        device=device
    )

    # ============================================================
    # A. 最小 reverse sampling
    # ============================================================
    print("\n[1] 开始 reverse sampling ...")
    gen_rel_torch, history = sampler.sample(
        model=model,
        num_samples=num_generate,
        channels=channels,
        seq_len=seq_len,
        return_history=True
    )

    gen_rel_np = gen_rel_torch.detach().cpu().numpy()   # [N, 2, 19]
    gen_abs_np = rel_to_abs(gen_rel_np)

    np.save(out_dir / "generated_rel_samples.npy", gen_rel_np)
    np.save(out_dir / "generated_abs_samples.npy", gen_abs_np)

    plot_real_vs_generated(
        real_abs=real_abs_np[:num_generate],
        gen_abs=gen_abs_np,
        save_path=out_dir / "real_vs_generated.png"
    )

    print("已保存:")
    print(out_dir / "generated_rel_samples.npy")
    print(out_dir / "generated_abs_samples.npy")
    print(out_dir / "real_vs_generated.png")

    # ============================================================
    # B. 单样本去噪 sanity check
    # ============================================================
    print("\n[2] 开始单样本去噪检查 ...")
    idx = 0
    t_vis = 80

    x0 = real_rel_torch[idx:idx+1].to(device)  # [1, 2, 19]
    t_batch = torch.tensor([t_vis], device=device, dtype=torch.long)
    noise = torch.randn_like(x0)

    xt = sampler.q_sample(x0, t_batch, noise=noise)

    with torch.no_grad():
        eps_pred = model(xt, t_batch)
        x0_pred = sampler.predict_x0_from_eps(xt, t_batch, eps_pred)

    x0_np = x0.detach().cpu().numpy()
    xt_np = xt.detach().cpu().numpy()
    x0_pred_np = x0_pred.detach().cpu().numpy()

    x0_abs = rel_to_abs(x0_np)[0]
    xt_abs = rel_to_abs(xt_np)[0]
    x0_pred_abs = rel_to_abs(x0_pred_np)[0]

    plot_denoise_check(
        x0_abs=x0_abs,
        xt_abs=xt_abs,
        x0_pred_abs=x0_pred_abs,
        save_path=out_dir / "denoise_check.png",
        t_vis=t_vis
    )

    print("已保存:")
    print(out_dir / "denoise_check.png")

    # ============================================================
    # C. 打印一点简单统计
    # ============================================================
    real_step_norm = np.linalg.norm(real_rel_np.reshape(-1, 2), axis=1)
    gen_step_norm = np.linalg.norm(np.transpose(gen_rel_np, (0, 2, 1)).reshape(-1, 2), axis=1)

    print("\n[3] 简单运动统计比较")
    print(f"real step norm mean = {real_step_norm.mean():.6f}")
    print(f"real step norm std  = {real_step_norm.std():.6f}")
    print(f"gen  step norm mean = {gen_step_norm.mean():.6f}")
    print(f"gen  step norm std  = {gen_step_norm.std():.6f}")

    print("\n完成。")


if __name__ == "__main__":
    main()