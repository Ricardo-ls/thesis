from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.stage3.refinement.refiners import REFINER_NAMES, run_refiner
from tools.stage3.refinement.ddpm_refiner import ddpm_prior_masked_replace_v1
from utils.stage3.controlled_benchmark import (
    DEGRADATION_NAMES,
    METHODS,
    DEFAULT_SEED,
    DEFAULT_SPAN_MODE,
    DEFAULT_SPAN_RATIO,
    clean_path,
    ensure_controlled_dirs,
    experiment_tag,
    mask_path,
    reconstruction_path,
)


REFINEMENT_ROOT_DIR = PROJECT_ROOT / "outputs" / "stage3" / "refinement"
REFINED_DIR = REFINEMENT_ROOT_DIR / "refined"
REFINEMENT_EVAL_DIR = REFINEMENT_ROOT_DIR / "eval"
REFINEMENT_FIGURE_DIR = REFINEMENT_ROOT_DIR / "figures"
REFINEMENT_METADATA_DIR = REFINEMENT_ROOT_DIR / "metadata"


def ensure_refinement_dirs():
    for path in [
        REFINEMENT_ROOT_DIR,
        REFINED_DIR,
        REFINEMENT_EVAL_DIR,
        REFINEMENT_FIGURE_DIR,
        REFINEMENT_METADATA_DIR,
    ]:
        path.mkdir(parents=True, exist_ok=True)


def refined_path(degradation: str, coarse_method: str, refiner_name: str):
    return REFINED_DIR / f"refined_{degradation}_{coarse_method}_{refiner_name}.npy"


def refinement_metadata_path(degradation: str, coarse_method: str, refiner_name: str):
    return REFINEMENT_METADATA_DIR / f"refined_{degradation}_{coarse_method}_{refiner_name}.json"


def load_array(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"Required file not found: {path}")
    return np.load(path, allow_pickle=False)


def main():
    parser = argparse.ArgumentParser(description="Run the Stage 3 refinement interface.")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--span_ratio", type=float, default=DEFAULT_SPAN_RATIO)
    parser.add_argument("--span_mode", type=str, default=DEFAULT_SPAN_MODE, choices=["fixed", "random"])
    args = parser.parse_args()

    ensure_controlled_dirs()
    ensure_refinement_dirs()
    tag = experiment_tag(args.span_ratio, args.span_mode, args.seed)

    clean = load_array(clean_path()).astype(np.float32)
    obs_mask = load_array(mask_path(tag)).astype(np.uint8)
    if clean.ndim != 3 or clean.shape[-1] != 2:
        raise ValueError(f"Expected clean trajectory shape [N, T, 2], got {clean.shape}")
    if obs_mask.shape != clean.shape[:2]:
        raise ValueError(f"Expected obs_mask shape {clean.shape[:2]}, got {obs_mask.shape}")

    print("=" * 60)
    print("Running Stage 3 refinement interface")
    print(f"experiment_tag = {tag}")
    print(f"clean_path     = {clean_path()}")
    print(f"clean_shape    = {clean.shape}")

    for degradation in DEGRADATION_NAMES:
        for coarse_method in METHODS:
            coarse_path = reconstruction_path(degradation, coarse_method, tag)
            coarse = load_array(coarse_path).astype(np.float32)
            if coarse.shape != clean.shape:
                raise ValueError(
                    f"Coarse shape mismatch for {degradation}/{coarse_method}: {coarse.shape} vs {clean.shape}"
                )
            ddpm_candidate_cache = None
            ddpm_metadata_cache = None
            for refiner_name in REFINER_NAMES:
                if refiner_name == "ddpm_prior_masked_replace_v1" and ddpm_candidate_cache is not None:
                    refined, metadata = ddpm_prior_masked_replace_v1(
                        coarse,
                        obs_mask=obs_mask,
                        ddpm_candidate=ddpm_candidate_cache,
                        base_metadata=ddpm_metadata_cache,
                    )
                else:
                    refined, metadata = run_refiner(refiner_name, coarse, obs_mask=obs_mask)
                    if refiner_name == "ddpm_prior_interface_v0":
                        ddpm_candidate_cache = refined.copy()
                        ddpm_metadata_cache = dict(metadata)
                out_path = refined_path(degradation, coarse_method, refiner_name)
                np.save(out_path, refined)
                metadata_path = refinement_metadata_path(degradation, coarse_method, refiner_name)
                metadata_record = {
                    "degradation": degradation,
                    "coarse_method": coarse_method,
                    "refiner": refiner_name,
                    "input_path": str(coarse_path),
                    "output_path": str(out_path),
                    **metadata,
                }
                metadata_path.write_text(json.dumps(metadata_record, indent=2), encoding="utf-8")
                print(f"[done] {degradation:20s} {coarse_method:30s} {refiner_name:22s} -> {out_path}")

    print("=" * 60)


if __name__ == "__main__":
    main()
