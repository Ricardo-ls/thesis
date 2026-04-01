from pathlib import Path

import torch

from models.temporal_denoiser import TemporalDenoiser1D
from tools.prior.sample.reverse_sample_ddpm import DDPMSampler, rel_to_abs

project_root = Path.cwd()
device = "cuda" if torch.cuda.is_available() else "cpu"

ckpt_path = project_root / "outputs" / "prior" / "train" / "ddpm_minimal_q20" / "best_model.pt"
print("ckpt_path =", ckpt_path)
print("exists    =", ckpt_path.exists())

timesteps = 100
hidden_dim = 64
channels = 2
seq_len = 19
num_generate = 4

model = TemporalDenoiser1D(
    max_timesteps=timesteps,
    in_channels=channels,
    hidden_dim=hidden_dim
).to(device)

ckpt = torch.load(ckpt_path, map_location=device)

if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
    state_dict = ckpt["model_state_dict"]
elif isinstance(ckpt, dict) and "state_dict" in ckpt:
    state_dict = ckpt["state_dict"]
elif isinstance(ckpt, dict):
    state_dict = ckpt
else:
    raise ValueError("无法识别 checkpoint 格式。")

missing, unexpected = model.load_state_dict(state_dict, strict=False)
print("missing keys   =", missing)
print("unexpected keys=", unexpected)

model.eval()

sampler = DDPMSampler(
    timesteps=timesteps,
    beta_start=1e-4,
    beta_end=0.02,
    device=device
)

with torch.no_grad():
    gen_rel = sampler.sample(
        model=model,
        num_samples=num_generate,
        channels=channels,
        seq_len=seq_len,
        return_history=False
    )

gen_rel_np = gen_rel.detach().cpu().numpy()
gen_abs_np = rel_to_abs(gen_rel_np)

print("gen_rel shape =", gen_rel_np.shape)
print("gen_abs shape =", gen_abs_np.shape)
print("sample mean   =", float(gen_rel_np.mean()))
print("sample std    =", float(gen_rel_np.std()))