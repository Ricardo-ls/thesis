from __future__ import annotations

import csv
import hashlib
import os
import re
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CURATED_ROOT = PROJECT_ROOT / "outputs" / "curated" / "stage3_sde"
DOC_DIR = PROJECT_ROOT / "docs" / "stage3_sde"

APPLY_LOG = CURATED_ROOT / "reorganization_apply_log.csv"
MANUAL_REVIEW = CURATED_ROOT / "reorganization_manual_review.csv"
VALIDATION_SUMMARY = CURATED_ROOT / "reorganization_validation_summary.md"
VALIDATION_ERRORS = CURATED_ROOT / "reorganization_validation_errors.csv"

SCAN_ROOTS = [
    PROJECT_ROOT / "outputs" / "stage3",
    PROJECT_ROOT / "outputs" / "stage4",
    PROJECT_ROOT / "outputs" / "stage3_indoor",
    PROJECT_ROOT / "docs" / "stage4",
    PROJECT_ROOT / "tools" / "stage3",
    PROJECT_ROOT / "tools" / "stage4",
]


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_csv(path: Path) -> list[dict]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing CSV: {path}")
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict]) -> None:
    keys = ["check", "path", "status", "message"]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)
    print(f"[FILE] {rel(path)} written")


def allowed_cache(path: Path) -> bool:
    s = rel(path).lower()
    return "/__pycache__/" in s or s.endswith(".pyc") or s.endswith(".ds_store") or ".matplotlib_cache" in s


def validate_moved_files(log_rows: list[dict]) -> tuple[list[dict], int]:
    errors: list[dict] = []
    checked = 0
    for row in log_rows:
        if row["status"] not in {"moved", "already_moved_or_duplicate", "already_moved_source_missing"}:
            continue
        src = PROJECT_ROOT / row["original_path"]
        dst = PROJECT_ROOT / row["proposed_new_path"]
        checked += 1
        if not dst.is_file():
            errors.append({"check": "destination_exists", "path": row["proposed_new_path"], "status": "FAIL", "message": "destination missing"})
            continue
        dest_hash = sha256(dst)
        if row["source_hash"] and dest_hash != row["source_hash"]:
            errors.append(
                {
                    "check": "hash_match",
                    "path": row["proposed_new_path"],
                    "status": "FAIL",
                    "message": f"hash mismatch: expected {row['source_hash']} got {dest_hash}",
                }
            )
        if src.exists():
            errors.append({"check": "old_source_absent", "path": row["original_path"], "status": "FAIL", "message": "old source still exists"})
    return errors, checked


def validate_remaining_old_files(manual_rows: list[dict]) -> tuple[list[dict], int]:
    errors: list[dict] = []
    manual_paths = {row["original_path"] for row in manual_rows}
    remaining_count = 0
    for root in SCAN_ROOTS:
        if not root.exists():
            continue
        for path in root.rglob("*"):
            if not path.is_file():
                continue
            remaining_count += 1
            r = rel(path)
            if r in manual_paths or allowed_cache(path):
                continue
            errors.append(
                {
                    "check": "remaining_old_file",
                    "path": r,
                    "status": "FAIL",
                    "message": "remaining old file is not marked manual review/cache",
                }
            )
    return errors, remaining_count


def validate_manifest() -> list[dict]:
    errors: list[dict] = []
    manifest = DOC_DIR / "mainline_manifest.md"
    if not manifest.is_file():
        return [{"check": "manifest_exists", "path": rel(manifest), "status": "FAIL", "message": "mainline manifest missing"}]
    text = manifest.read_text(encoding="utf-8")
    if "outputs/curated/stage3_sde/mainline" not in text:
        errors.append(
            {
                "check": "manifest_curated_paths",
                "path": rel(manifest),
                "status": "FAIL",
                "message": "manifest does not reference curated mainline paths",
            }
        )
    if "Do not treat Stage 4/DPS/multi-t/sensor-adapter outputs as mainline" not in text:
        errors.append(
            {
                "check": "manifest_warning",
                "path": rel(manifest),
                "status": "FAIL",
                "message": "manifest missing non-mainline warning",
            }
        )
    return errors


def validate_markdown_links() -> list[dict]:
    errors: list[dict] = []
    link_re = re.compile(r"\[[^\]]+\]\(([^)]+)\)")
    for path in DOC_DIR.glob("*.md"):
        text = path.read_text(encoding="utf-8")
        for match in link_re.finditer(text):
            target = match.group(1).strip()
            if not target or target.startswith(("http://", "https://", "mailto:", "#")):
                continue
            target = target.split("#", 1)[0]
            target_path = (path.parent / target) if not target.startswith("/") else Path(target)
            if not target_path.exists():
                errors.append(
                    {
                        "check": "markdown_link",
                        "path": rel(path),
                        "status": "FAIL",
                        "message": f"broken link target: {target}",
                    }
                )
    return errors


def main() -> None:
    log_rows = read_csv(APPLY_LOG)
    manual_rows = read_csv(MANUAL_REVIEW)
    errors: list[dict] = []

    moved_errors, moved_checked = validate_moved_files(log_rows)
    remaining_errors, remaining_count = validate_remaining_old_files(manual_rows)
    errors.extend(moved_errors)
    errors.extend(remaining_errors)
    errors.extend(validate_manifest())
    errors.extend(validate_markdown_links())

    write_csv(VALIDATION_ERRORS, errors)
    passed = len(errors) == 0
    moved_count = sum(row["status"] == "moved" for row in log_rows)
    duplicate_count = sum(row["status"] == "already_moved_or_duplicate" for row in log_rows)
    conflict_count = sum(row["status"].startswith("conflict") for row in log_rows)
    skipped_count = sum(row["status"].startswith("skipped") for row in log_rows)

    summary = [
        "# Stage 3 SDE Reorganization Validation Summary",
        "",
        f"- validation passed: `{passed}`",
        f"- moved files checked: `{moved_checked}`",
        f"- moved status count: `{moved_count}`",
        f"- duplicate/already moved count: `{duplicate_count}`",
        f"- conflict count in apply log: `{conflict_count}`",
        f"- skipped count in apply log: `{skipped_count}`",
        f"- remaining old files scanned: `{remaining_count}`",
        f"- validation error count: `{len(errors)}`",
        f"- validation errors CSV: `{rel(VALIDATION_ERRORS)}`",
        "",
        "## Manifest",
        "",
        f"- manifest path: `{rel(DOC_DIR / 'mainline_manifest.md')}`",
        "- future Codex work should follow the manifest and not treat Stage 4/DPS/multi-t/sensor-adapter outputs as mainline unless explicitly instructed.",
    ]
    if errors:
        summary.extend(["", "## Errors", ""])
        summary.extend(f"- `{row['check']}` `{row['path']}`: {row['message']}" for row in errors[:200])
    VALIDATION_SUMMARY.write_text("\n".join(summary) + "\n", encoding="utf-8")
    print(f"[FILE] {rel(VALIDATION_SUMMARY)} written")
    print("STAGE3_SDE_REORGANIZATION_VALIDATION_COMPLETE")
    print(f"VALIDATION_PASSED={passed}")
    print(f"VALIDATION_ERROR_COUNT={len(errors)}")
    print(f"MOVED_FILES_CHECKED={moved_checked}")
    print(f"REMAINING_OLD_FILES_SCANNED={remaining_count}")


if __name__ == "__main__":
    main()
