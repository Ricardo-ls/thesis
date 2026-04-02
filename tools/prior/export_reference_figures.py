from pathlib import Path
import argparse
import shutil
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.prior.ablation_paths import get_paths_by_name, resolve_variant_or_objective, to_abs_path


def copy_public_assets(source_dir: Path, target_dir: Path):
    if not source_dir.exists():
        raise FileNotFoundError(f"Source directory not found: {source_dir}")

    target_dir.mkdir(parents=True, exist_ok=True)

    for png_path in sorted(source_dir.glob("*.png")):
        shutil.copy2(png_path, target_dir / png_path.name)

    manifest_path = source_dir / "manifest.json"
    if manifest_path.exists():
        shutil.copy2(manifest_path, target_dir / "manifest.json")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", type=str, default="motion_balanced")
    parser.add_argument("--reference_tag", type=str, default="reference_seed42")
    parser.add_argument(
        "--include",
        type=str,
        default="both",
        choices=["sample", "eval", "both"],
    )
    args = parser.parse_args()

    resolved_variant = resolve_variant_or_objective(args.variant)
    cfg = get_paths_by_name(args.variant)

    sample_source = to_abs_path(cfg["sample_dir"]) / args.reference_tag
    eval_source = to_abs_path(cfg["eval_dir"]) / args.reference_tag
    target_root = PROJECT_ROOT / "docs" / "assets" / "stage2" / resolved_variant / args.reference_tag

    if args.include in {"sample", "both"}:
        copy_public_assets(sample_source, target_root / "sample")
    if args.include in {"eval", "both"}:
        copy_public_assets(eval_source, target_root / "eval")

    print(f"resolved_variant = {resolved_variant}")
    print(f"sample_source    = {sample_source}")
    print(f"eval_source      = {eval_source}")
    print(f"target_root      = {target_root}")


if __name__ == "__main__":
    main()
