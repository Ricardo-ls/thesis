from __future__ import annotations

from pathlib import Path
import re
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
import torch.nn as nn

from models.temporal_denoiser_conditional import ConditionalTemporalDenoiser1D
from models.temporal_denoiser_conditional_4block import TemporalDenoiserConditional4Block


TRAIN_SCRIPT_PATH = PROJECT_ROOT / "tools" / "stage3_indoor" / "train_cond_residual_gaussian.py"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "stage3_indoor" / "receptive_field_expansion"
PARAM_COUNT_PATH = OUTPUT_DIR / "param_count.txt"
SEQUENCE_LENGTH = 20


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def parse_stage3_main_model_config(script_path: Path) -> dict[str, int]:
    if not script_path.is_file():
        raise FileNotFoundError(f"Missing required training script: {script_path}")

    text = script_path.read_text(encoding="utf-8")

    timestep_match = re.search(r"^TIMESTEPS\s*=\s*(\d+)\s*$", text, flags=re.MULTILINE)
    model_match = re.search(
        r"ConditionalTemporalDenoiser1D\s*\(\s*max_timesteps\s*=\s*TIMESTEPS\s*,\s*in_channels\s*=\s*(\d+)\s*,\s*hidden_dim\s*=\s*(\d+)\s*\)",
        text,
        flags=re.MULTILINE | re.DOTALL,
    )
    if timestep_match is None or model_match is None:
        raise RuntimeError(
            "Unable to parse ConditionalTemporalDenoiser1D config from "
            f"{script_path}"
        )

    return {
        "max_timesteps": int(timestep_match.group(1)),
        "in_channels": int(model_match.group(1)),
        "hidden_dim": int(model_match.group(2)),
    }


def get_block_names(model: nn.Module) -> list[str]:
    block_names = [name for name, module in model.named_children() if name.startswith("block") and isinstance(module, nn.Sequential)]
    if not block_names:
        raise RuntimeError(f"No residual blocks found in model {type(model).__name__}")
    return sorted(block_names)


def get_block_conv_layers(model: nn.Module) -> list[nn.Conv1d]:
    convs: list[nn.Conv1d] = []
    for block_name in get_block_names(model):
        block = getattr(model, block_name)
        convs.extend(module for module in block if isinstance(module, nn.Conv1d))
    if not convs:
        raise RuntimeError(f"No Conv1d layers found inside residual blocks of {type(model).__name__}")
    return convs


def get_projection_conv_layers(model: nn.Module) -> list[nn.Conv1d]:
    convs: list[nn.Conv1d] = []
    if isinstance(getattr(model, "input_proj", None), nn.Conv1d):
        convs.append(model.input_proj)
    if isinstance(getattr(model, "output_proj", None), nn.Conv1d):
        convs.append(model.output_proj)
    if len(convs) != 2:
        raise RuntimeError(f"Expected input_proj and output_proj Conv1d layers in {type(model).__name__}")
    return convs


def receptive_field_from_convs(convs: list[nn.Conv1d]) -> int:
    rf = 1
    for conv in convs:
        kernel_size = conv.kernel_size[0]
        dilation = conv.dilation[0]
        rf += (kernel_size - 1) * dilation
    return rf


def coverage_percent(rf: int, seq_len: int) -> float:
    return 100.0 * rf / float(seq_len)


def kernel_summary(convs: list[nn.Conv1d]) -> str:
    return ", ".join(
        f"k={conv.kernel_size[0]}, d={conv.dilation[0]}"
        for conv in convs
    )


def main() -> None:
    config = parse_stage3_main_model_config(TRAIN_SCRIPT_PATH)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    current_model = ConditionalTemporalDenoiser1D(**config)
    expanded_model = TemporalDenoiserConditional4Block(**config)

    current_block_convs = get_block_conv_layers(current_model)
    expanded_block_convs = get_block_conv_layers(expanded_model)
    current_proj_convs = get_projection_conv_layers(current_model)
    expanded_proj_convs = get_projection_conv_layers(expanded_model)

    current_block_rf = receptive_field_from_convs(current_block_convs)
    current_full_rf = receptive_field_from_convs(current_proj_convs + current_block_convs)
    expanded_block_rf = receptive_field_from_convs(expanded_block_convs)
    expanded_full_rf = receptive_field_from_convs(expanded_proj_convs + expanded_block_convs)

    current_params = count_parameters(current_model)
    expanded_params = count_parameters(expanded_model)
    ratio = expanded_params / current_params

    x = torch.randn(4, 2, SEQUENCE_LENGTH)
    cond = torch.randn(4, 2, SEQUENCE_LENGTH)
    t = torch.randint(low=0, high=config["max_timesteps"], size=(4,))

    y2 = current_model(x, cond, t)
    y4 = expanded_model(x, cond, t)
    assert y2.shape == (4, 2, SEQUENCE_LENGTH), f"Expected current model output shape (4, 2, {SEQUENCE_LENGTH}), got {tuple(y2.shape)}"
    assert y4.shape == (4, 2, SEQUENCE_LENGTH), f"Expected expanded model output shape (4, 2, {SEQUENCE_LENGTH}), got {tuple(y4.shape)}"

    loss = y4.pow(2).mean()
    loss.backward()

    current_coverage = coverage_percent(current_full_rf, SEQUENCE_LENGTH)
    expanded_coverage = coverage_percent(expanded_full_rf, SEQUENCE_LENGTH)

    print(f"Current 2-block block-only RF: {current_block_rf}")
    print(f"Current 2-block including-proj RF: {current_full_rf}")
    print(f"Expanded 4-block block-only RF: {expanded_block_rf}")
    print(f"Expanded 4-block including-proj RF: {expanded_full_rf}")
    print(f"Sequence length T: {SEQUENCE_LENGTH}")
    print(f"Current including-proj coverage: {current_coverage:.1f}%")
    print(f"Expanded including-proj coverage: {expanded_coverage:.1f}%")
    print(f"Current 2-block params: {current_params}")
    print(f"Expanded 4-block params: {expanded_params}")
    print(f"Ratio: {ratio:.4f}x")
    print(f"Expanded 4-block output shape: {tuple(y4.shape)}")
    print("Expanded 4-block backward: ok")

    report_lines = [
        f"Current 2-block params: {current_params}",
        f"Expanded 4-block params: {expanded_params}",
        f"Parameter ratio (expanded/current): {ratio:.4f}x",
        f"Current 2-block block-only RF: {current_block_rf}",
        f"Current 2-block including-proj RF: {current_full_rf}",
        f"Expanded 4-block block-only RF: {expanded_block_rf}",
        f"Expanded 4-block including-proj RF: {expanded_full_rf}",
        f"Sequence length T: {SEQUENCE_LENGTH}",
        f"Current including-proj coverage: {current_coverage:.1f}%",
        f"Expanded including-proj coverage: {expanded_coverage:.1f}%",
        (
            "Model config used: "
            f"max_timesteps={config['max_timesteps']}, "
            f"in_channels={config['in_channels']}, "
            f"hidden_dim={config['hidden_dim']}"
        ),
        f"Current block convs: {kernel_summary(current_block_convs)}",
        f"Expanded block convs: {kernel_summary(expanded_block_convs)}",
        f"Projection convs: {kernel_summary(current_proj_convs)}",
    ]
    PARAM_COUNT_PATH.write_text("\n".join(report_lines) + "\n", encoding="utf-8")

    print("PHASE 1 COMPLETE.")
    print("Generated:")
    print("- models/temporal_denoiser_conditional_4block.py")
    print("- tools/stage3_indoor/compute_rf_expansion.py")
    print("- outputs/stage3_indoor/receptive_field_expansion/param_count.txt")
    print(f"Current 2-block params: {current_params}")
    print(f"Expanded 4-block params: {expanded_params}")
    print(f"Parameter ratio (expanded/current): {ratio:.4f}x")
    print(f"Current 2-block block-only RF: {current_block_rf}")
    print(f"Current 2-block including-proj RF: {current_full_rf}")
    print(f"Expanded 4-block block-only RF: {expanded_block_rf}")
    print(f"Expanded 4-block including-proj RF: {expanded_full_rf}")
    print(f"Current including-proj coverage: {current_coverage:.1f}%")
    print(f"Expanded including-proj coverage: {expanded_coverage:.1f}%")


if __name__ == "__main__":
    main()
