from pathlib import Path


PROCESSED_DIR = Path("datasets/processed")
TRAIN_OUT_DIR = Path("outputs/prior/train")
SAMPLE_OUT_DIR = Path("outputs/prior/sample")
EVAL_OUT_DIR = Path("outputs/prior/eval")


def get_eth_ucy_variant_paths(variant: str):
    variant = variant.lower()

    if variant == "none":
        return {
            "variant": "none",
            "abs_path": str(PROCESSED_DIR / "data_eth_ucy_20.npy"),
            "rel_path": str(PROCESSED_DIR / "data_eth_ucy_20_rel.npy"),
            "train_tag": "ddpm_eth_ucy_none_h128",
            "sample_tag": "ddpm_eth_ucy_none_h128",
            "eval_tag": "ddpm_eth_ucy_none_h128",
        }

    if variant in {"q10", "q20", "q30"}:
        return {
            "variant": variant,
            "abs_path": str(PROCESSED_DIR / f"data_eth_ucy_20_{variant}.npy"),
            "rel_path": str(PROCESSED_DIR / f"data_eth_ucy_20_rel_{variant}.npy"),
            "train_tag": f"ddpm_eth_ucy_{variant}_h128",
            "sample_tag": f"ddpm_eth_ucy_{variant}_h128",
            "eval_tag": f"ddpm_eth_ucy_{variant}_h128",
        }

    raise ValueError(f"Unsupported variant: {variant}")