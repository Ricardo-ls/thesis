from pathlib import Path


PROCESSED_DIR = Path("datasets/processed")
TRAIN_OUT_DIR = Path("outputs/prior/train")
SAMPLE_OUT_DIR = Path("outputs/prior/sample")
EVAL_OUT_DIR = Path("outputs/prior/eval")


OFFICIAL_STAGE2_PROTOCOL = {
    "dataset": "ETH+UCY",
    "model": "h128",
    "batch_size": 128,
    "timesteps": 100,
    "random_seed": 42,
    "max_epoch": 100,
    "checkpoint_selection": "best_validation_checkpoint",
    "reverse_samples": 512,
    "eval_metrics": [
        "step_norm_all",
        "avg_speed",
        "total_length",
        "endpoint_displacement",
        "moving_ratio_global",
        "propulsion_ratio",
        "acc_rms",
    ],
}


OFFICIAL_STAGE2_TRAIN_RESULTS = {
    "none": {
        "samples": 36073,
        "best_val_loss": 0.087299,
        "best_epoch": 68,
    },
    "q10": {
        "samples": 32465,
        "best_val_loss": 0.093522,
        "best_epoch": 99,
    },
    "q20": {
        "samples": 28858,
        "best_val_loss": 0.099376,
        "best_epoch": 70,
    },
    "q30": {
        "samples": 25251,
        "best_val_loss": 0.102625,
        "best_epoch": 72,
    },
}


OFFICIAL_STAGE2_EVAL_RATIOS = {
    "none": {
        "step_norm_all": 0.920140,
        "endpoint_displacement": 0.811718,
        "moving_ratio_global": 1.103156,
        "propulsion_ratio": 0.855215,
        "acc_rms": 1.452409,
    },
    "q10": {
        "step_norm_all": 0.912383,
        "endpoint_displacement": 0.797510,
        "moving_ratio_global": 1.079710,
        "propulsion_ratio": 0.855401,
        "acc_rms": 1.376507,
    },
    "q20": {
        "step_norm_all": 0.864282,
        "endpoint_displacement": 0.738620,
        "moving_ratio_global": 1.031362,
        "propulsion_ratio": 0.826810,
        "acc_rms": 1.372372,
    },
    "q30": {
        "step_norm_all": 0.948828,
        "endpoint_displacement": 0.805835,
        "moving_ratio_global": 0.964964,
        "propulsion_ratio": 0.817302,
        "acc_rms": 1.555117,
    },
}


OFFICIAL_STAGE2_RECOMMENDATION = {
    "optimization_best": "none",
    "motion_balanced": "q20",
    "filtered_variants_best": "q20",
    "weak_filter_reference": "q10",
    "strong_filter_reference": "q30",
}


OFFICIAL_STAGE2_NARRATIVE = {
    "none": "optimization-best baseline under the unified 100-epoch protocol",
    "q10": "filtering too weak; gains are limited",
    "q20": "most balanced motion-focused prior among filtered variants",
    "q30": "filtering too strong; some dynamic ratios improve but propulsion worsens and train loss is the highest",
}


def get_eth_ucy_variant_paths(variant: str, seed: int = 42):
    variant = variant.lower()
    seed_tag = f"seed{seed}"

    if variant == "none":
        train_tag = "ddpm_eth_ucy_none_h128"
        return {
            "variant": "none",
            "abs_path": str(PROCESSED_DIR / "data_eth_ucy_20.npy"),
            "rel_path": str(PROCESSED_DIR / "data_eth_ucy_20_rel.npy"),
            "train_tag": train_tag,
            "sample_tag": train_tag,
            "eval_tag": train_tag,
            "train_dir": str(TRAIN_OUT_DIR / train_tag / seed_tag),
            "sample_dir": str(SAMPLE_OUT_DIR / train_tag / seed_tag),
            "eval_dir": str(EVAL_OUT_DIR / train_tag / seed_tag),
            "ckpt_path": str(TRAIN_OUT_DIR / train_tag / seed_tag / "best_model.pt"),
        }

    if variant in {"q10", "q20", "q30"}:
        train_tag = f"ddpm_eth_ucy_{variant}_h128"
        return {
            "variant": variant,
            "abs_path": str(PROCESSED_DIR / f"data_eth_ucy_20_{variant}.npy"),
            "rel_path": str(PROCESSED_DIR / f"data_eth_ucy_20_rel_{variant}.npy"),
            "train_tag": train_tag,
            "sample_tag": train_tag,
            "eval_tag": train_tag,
            "train_dir": str(TRAIN_OUT_DIR / train_tag / seed_tag),
            "sample_dir": str(SAMPLE_OUT_DIR / train_tag / seed_tag),
            "eval_dir": str(EVAL_OUT_DIR / train_tag / seed_tag),
            "ckpt_path": str(TRAIN_OUT_DIR / train_tag / seed_tag / "best_model.pt"),
        }

    raise ValueError(f"Unsupported variant: {variant}")


def list_supported_variants():
    return ["none", "q10", "q20", "q30"]


def get_stage2_train_record(variant: str):
    variant = variant.lower()
    if variant not in OFFICIAL_STAGE2_TRAIN_RESULTS:
        raise ValueError(
            f"Unsupported variant: {variant}. "
            f"Supported variants: {list_supported_variants()}"
        )
    return OFFICIAL_STAGE2_TRAIN_RESULTS[variant]


def get_stage2_eval_ratios(variant: str):
    variant = variant.lower()
    if variant not in OFFICIAL_STAGE2_EVAL_RATIOS:
        raise ValueError(
            f"Unsupported variant: {variant}. "
            f"Supported variants: {list_supported_variants()}"
        )
    return OFFICIAL_STAGE2_EVAL_RATIOS[variant]


def get_stage2_narrative(variant: str):
    variant = variant.lower()
    if variant not in OFFICIAL_STAGE2_NARRATIVE:
        raise ValueError(
            f"Unsupported variant: {variant}. "
            f"Supported variants: {list_supported_variants()}"
        )
    return OFFICIAL_STAGE2_NARRATIVE[variant]


def get_recommended_prior(objective: str = "optimization_best"):
    if objective not in OFFICIAL_STAGE2_RECOMMENDATION:
        raise ValueError(
            f"Unsupported objective: {objective}. "
            f"Supported objectives: {list(OFFICIAL_STAGE2_RECOMMENDATION.keys())}"
        )
    return OFFICIAL_STAGE2_RECOMMENDATION[objective]


def get_recommended_prior_paths(objective: str = "optimization_best"):
    variant = get_recommended_prior(objective)
    return get_eth_ucy_variant_paths(variant)


def list_supported_objectives():
    return list(OFFICIAL_STAGE2_RECOMMENDATION.keys())


def resolve_variant_or_objective(name: str):
    name = name.lower()

    if name in list_supported_variants():
        return name

    if name in OFFICIAL_STAGE2_RECOMMENDATION:
        return OFFICIAL_STAGE2_RECOMMENDATION[name]

    raise ValueError(
        f"Unsupported name: {name}. "
        f"Supported variants: {list_supported_variants()}, "
        f"supported objectives: {list_supported_objectives()}"
    )


def get_paths_by_name(name: str):
    variant = resolve_variant_or_objective(name)
    return get_eth_ucy_variant_paths(variant)


def get_train_record_by_name(name: str):
    variant = resolve_variant_or_objective(name)
    return get_stage2_train_record(variant)


def get_eval_ratios_by_name(name: str):
    variant = resolve_variant_or_objective(name)
    return get_stage2_eval_ratios(variant)


def get_narrative_by_name(name: str):
    variant = resolve_variant_or_objective(name)
    return get_stage2_narrative(variant)


def to_abs_path(path_like):
    path = Path(path_like)
    return path if path.is_absolute() else Path(__file__).resolve().parents[2] / path
