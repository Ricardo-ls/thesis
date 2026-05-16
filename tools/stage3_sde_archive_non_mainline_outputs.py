from __future__ import annotations

import csv
import hashlib
import shutil
from datetime import datetime
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUTS = PROJECT_ROOT / "outputs"
ARCHIVE_ROOT = OUTPUTS / "curated" / "stage3_sde" / "legacy_or_deprecated"
TOP_LEVEL_ARCHIVE = ARCHIVE_ROOT / "original_outputs_tree"
INDOOR_ARCHIVE = ARCHIVE_ROOT / "stage3_indoor_non_mainline"
LOG_PATH = ARCHIVE_ROOT / "non_mainline_archive_move_log.csv"
README_PATH = ARCHIVE_ROOT / "README.md"
OUTPUTS_README = OUTPUTS / "README.md"


TOP_LEVEL_NON_MAINLINE = [
    " readme",
    "ddpm_eth_ucy_q20_h128",
    "ppt_refine_nature_skill",
    "prior",
    "stage3",
    "stage3_backups",
    "stage3_sim",
    "stage4",
]

STAGE3_INDOOR_KEEP = {"ddpm_indoor_v2"}


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def file_count_and_size(path: Path) -> tuple[int, int]:
    if not path.exists():
        return 0, 0
    if path.is_file():
        return 1, path.stat().st_size
    count = 0
    total = 0
    for child in path.rglob("*"):
        if child.is_file():
            count += 1
            total += child.stat().st_size
    return count, total


def hash_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def sample_hashes(path: Path, limit: int = 25) -> dict[str, str]:
    if not path.exists():
        return {}
    files = [path] if path.is_file() else sorted(p for p in path.rglob("*") if p.is_file())
    return {rel(p): hash_file(p) for p in files[:limit]}


def verify_sample_hashes(sample: dict[str, str], old_prefix: Path, new_prefix: Path) -> bool:
    for old_rel, old_hash in sample.items():
        old_abs = PROJECT_ROOT / old_rel
        try:
            suffix = old_abs.relative_to(old_prefix)
        except ValueError:
            suffix = old_abs.name
        new_abs = new_prefix / suffix
        if not new_abs.is_file() or hash_file(new_abs) != old_hash:
            return False
    return True


def safe_move(src: Path, dst: Path, category: str, notes: str) -> dict:
    row = {
        "source": rel(src),
        "destination": rel(dst),
        "category": category,
        "status": "",
        "file_count": 0,
        "size_bytes": 0,
        "sample_hash_verified": "",
        "notes": notes,
    }
    if not src.exists():
        row["status"] = "source_missing"
        return row

    count, size = file_count_and_size(src)
    row["file_count"] = count
    row["size_bytes"] = size
    sample = sample_hashes(src)

    if dst.exists():
        row["status"] = "destination_exists_skipped"
        row["notes"] += "; destination already exists, not overwriting"
        return row

    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(src), str(dst))
    row["status"] = "moved"
    row["sample_hash_verified"] = str(verify_sample_hashes(sample, src, dst))
    return row


def write_csv(rows: list[dict]) -> None:
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with LOG_PATH.open("w", newline="", encoding="utf-8") as f:
        fieldnames = ["source", "destination", "category", "status", "file_count", "size_bytes", "sample_hash_verified", "notes"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"[FILE] {rel(LOG_PATH)} written")


def write_readmes(rows: list[dict]) -> None:
    moved = [r for r in rows if r["status"] == "moved"]
    skipped = [r for r in rows if r["status"] != "moved"]
    lines = [
        "# Stage 3 SDE Legacy / Non-Mainline Archive",
        "",
        f"Generated: `{datetime.now().isoformat(timespec='seconds')}`",
        "",
        "This directory stores outputs that are not part of the current Stage 3 SDE mainline.",
        "",
        "Current mainline remains:",
        "",
        "- `docs/stage3_sde/mainline_manifest.md`",
        "- `outputs/curated/stage3_sde/mainline/`",
        "- `tools/curated/stage3_sde/mainline/`",
        "- `outputs/stage3_indoor/ddpm_indoor_v2/seed42/best_ema_model.pt`",
        "- `outputs/stage3_indoor/ddpm_indoor_v2/seed42/rel_norm_params_v2.npz`",
        "",
        "Do not use archived outputs as mainline evidence unless explicitly instructed.",
        "",
        "## Moved Non-Mainline Directories",
        "",
        "| source | destination | files | size_bytes | notes |",
        "| --- | --- | ---: | ---: | --- |",
    ]
    for row in moved:
        lines.append(f"| `{row['source']}` | `{row['destination']}` | {row['file_count']} | {row['size_bytes']} | {row['notes']} |")
    if skipped:
        lines.extend(["", "## Skipped / Already Missing", ""])
        for row in skipped:
            lines.append(f"- `{row['source']}`: {row['status']} ({row['notes']})")
    lines.extend(["", f"Move log: `{rel(LOG_PATH)}`"])
    README_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[FILE] {rel(README_PATH)} written")

    output_lines = [
        "# Outputs Directory",
        "",
        "This outputs directory has been cleaned around the current Stage 3 SDE mainline.",
        "",
        "Use these paths first:",
        "",
        "- `curated/stage3_sde/mainline/` for current mainline evidence.",
        "- `../docs/stage3_sde/mainline_manifest.md` for the source-of-truth manifest.",
        "- `stage3_indoor/ddpm_indoor_v2/` only for the retained indoor-v2 unconditional prior dependency.",
        "",
        "Non-mainline historical outputs were moved to:",
        "",
        "- `curated/stage3_sde/legacy_or_deprecated/original_outputs_tree/`",
        "- `curated/stage3_sde/legacy_or_deprecated/stage3_indoor_non_mainline/`",
        "",
        "Do not treat `prior`, old `stage3`, DPS, conditional-residual, or PPT-processing outputs as current mainline.",
    ]
    OUTPUTS_README.write_text("\n".join(output_lines) + "\n", encoding="utf-8")
    print(f"[FILE] {rel(OUTPUTS_README)} written")


def main() -> None:
    rows: list[dict] = []
    for name in TOP_LEVEL_NON_MAINLINE:
        src = OUTPUTS / name
        dst = TOP_LEVEL_ARCHIVE / name.strip().replace(" ", "_") if name == " readme" else TOP_LEVEL_ARCHIVE / name
        rows.append(safe_move(src, dst, "top_level_non_mainline_output", "not current Stage 3 SDE mainline"))

    stage3_indoor = OUTPUTS / "stage3_indoor"
    if stage3_indoor.exists():
        for child in sorted(stage3_indoor.iterdir()):
            if child.name in STAGE3_INDOOR_KEEP:
                continue
            rows.append(
                safe_move(
                    child,
                    INDOOR_ARCHIVE / child.name,
                    "stage3_indoor_non_mainline",
                    "stage3_indoor residue; retained ddpm_indoor_v2 in place as current prior dependency",
                )
            )

    write_csv(rows)
    write_readmes(rows)
    print("STAGE3_SDE_NON_MAINLINE_ARCHIVE_COMPLETE")
    print(f"MOVED_COUNT={sum(1 for r in rows if r['status'] == 'moved')}")
    print(f"SKIPPED_COUNT={sum(1 for r in rows if r['status'] != 'moved')}")
    print(f"ARCHIVE_ROOT={rel(ARCHIVE_ROOT)}")
    print(f"README={rel(README_PATH)}")


if __name__ == "__main__":
    main()
