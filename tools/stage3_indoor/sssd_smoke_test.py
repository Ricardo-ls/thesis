from __future__ import annotations

from pathlib import Path
import re
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch

from models.sssd_denoiser import SSSDDenoiser, count_parameters


OUTPUT_DIR = PROJECT_ROOT / "outputs" / "stage3_indoor" / "sssd_ablation"
PARAM_REPORT_PATH = OUTPUT_DIR / "param_count.txt"
STAGE3_COND_SCRIPT = PROJECT_ROOT / "tools" / "stage3_indoor" / "train_cond_residual_gaussian.py"


def _parse_stage3_conditional_config(script_path: Path) -> tuple[int, int, int] | None:
    if not script_path.is_file():
        return None

    text = script_path.read_text(encoding="utf-8")

    timestep_match = re.search(r"^TIMESTEPS\s*=\s*(\d+)\s*$", text, flags=re.MULTILINE)
    model_match = re.search(
        r"ConditionalTemporalDenoiser1D\s*\(\s*max_timesteps\s*=\s*TIMESTEPS\s*,\s*in_channels\s*=\s*(\d+)\s*,\s*hidden_dim\s*=\s*(\d+)\s*\)",
        text,
        flags=re.MULTILINE | re.DOTALL,
    )
    if timestep_match is None or model_match is None:
        return None

    timesteps = int(timestep_match.group(1))
    in_channels = int(model_match.group(1))
    hidden_dim = int(model_match.group(2))
    return timesteps, in_channels, hidden_dim


def _try_count_custom_ddpm_params() -> tuple[int | None, str]:
    config = _parse_stage3_conditional_config(STAGE3_COND_SCRIPT)
    if config is None:
        return None, "Custom DDPM params: NOT FOUND automatically. Please provide class/config path."

    try:
        from models.temporal_denoiser_conditional import ConditionalTemporalDenoiser1D
    except Exception:
        return None, "Custom DDPM params: NOT FOUND automatically. Please provide class/config path."

    timesteps, in_channels, hidden_dim = config
    model = ConditionalTemporalDenoiser1D(
        max_timesteps=timesteps,
        in_channels=in_channels,
        hidden_dim=hidden_dim,
    )
    return count_parameters(model), (
        "Custom DDPM params: auto-detected from "
        f"{STAGE3_COND_SCRIPT.relative_to(PROJECT_ROOT)}"
    )


def _format_ratio_warning(sssd_params: int, custom_params: int | None) -> tuple[str | None, str | None]:
    if custom_params is None or custom_params <= 0:
        return None, None

    ratio = sssd_params / custom_params
    ratio_line = f"Parameter ratio (SSSD/custom): {ratio:.4f}x"
    if 0.5 <= ratio <= 3.0:
        return ratio_line, None

    warning = (
        "WARNING: SSSD parameter ratio is outside the target range "
        f"[0.5x, 3.0x]. Actual ratio: {ratio:.4f}x"
    )
    return ratio_line, warning


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    model = SSSDDenoiser(
        in_channels=2,
        cond_channels=2,
        hidden_dim=64,
        out_channels=2,
        n_layers=4,
        timestep_dim=128,
        dropout=0.0,
    )
    x = torch.randn(4, 2, 20)
    cond = torch.randn(4, 2, 20)
    t = torch.randint(low=0, high=100, size=(4,))

    y = model(x, t, cond)
    assert y.shape == (4, 2, 20), f"Expected output shape (4, 2, 20), got {tuple(y.shape)}"

    loss = y.pow(2).mean()
    loss.backward()

    sssd_params = count_parameters(model)
    custom_params, custom_msg = _try_count_custom_ddpm_params()
    ratio_line, ratio_warning = _format_ratio_warning(sssd_params, custom_params)

    print(f"SSSD output shape: {tuple(y.shape)}")
    print(f"SSSD params: {sssd_params}")
    print("backward: ok")

    if custom_params is None:
        print("Custom DDPM params: NOT FOUND automatically. Please provide class/config path.")
    else:
        print(f"Custom DDPM params: {custom_params}")
        print(custom_msg)

    print(
        "Temporal block: "
        + ("official S4Layer" if model.uses_official_s4 else "fallback SSSD-compatible block")
    )

    if ratio_line is not None:
        print(ratio_line)
    if ratio_warning is not None:
        print(ratio_warning)

    report_lines = [
        f"SSSD params: {sssd_params}",
        (
            f"Custom DDPM params: {custom_params}"
            if custom_params is not None
            else "Custom DDPM params: NOT FOUND"
        ),
        (
            "Temporal block used: official S4Layer"
            if model.uses_official_s4
            else "Temporal block used: fallback SSSD-compatible block"
        ),
    ]
    if ratio_line is not None:
        report_lines.append(ratio_line)
    if ratio_warning is not None:
        report_lines.append(ratio_warning)
    else:
        report_lines.append("Parameter ratio warning: none")

    PARAM_REPORT_PATH.write_text("\n".join(report_lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
