from __future__ import annotations

import csv
import hashlib
import os
import shutil
import subprocess
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CURATED_ROOT = PROJECT_ROOT / "outputs" / "curated" / "stage3_sde"
MOVE_PLAN = CURATED_ROOT / "reorganization_move_plan.csv"
APPLY_LOG = CURATED_ROOT / "reorganization_apply_log.csv"
APPLY_SUMMARY = CURATED_ROOT / "reorganization_apply_summary.md"
CONFLICTS = CURATED_ROOT / "reorganization_conflicts.csv"
MANUAL_REVIEW = CURATED_ROOT / "reorganization_manual_review.csv"

CATEGORY_UNKNOWN = "UNKNOWN_NEEDS_MANUAL_REVIEW"


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


def read_plan() -> list[dict]:
    if not MOVE_PLAN.is_file():
        raise FileNotFoundError(f"Missing move plan: {MOVE_PLAN}")
    with MOVE_PLAN.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        keys: list[str] = []
        for row in rows:
            for key in row:
                if key not in keys:
                    keys.append(key)
        fieldnames = keys
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"[FILE] {rel(path)} written")


def is_git_tracked(path: Path) -> bool:
    result = subprocess.run(
        ["git", "ls-files", "--error-unmatch", rel(path)],
        cwd=PROJECT_ROOT,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    return result.returncode == 0


def git_mv(src: Path, dst: Path) -> bool:
    result = subprocess.run(
        ["git", "mv", rel(src), rel(dst)],
        cwd=PROJECT_ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )
    return result.returncode == 0


def safe_move(row: dict) -> dict:
    src = PROJECT_ROOT / row["original_path"]
    dst = PROJECT_ROOT / row["proposed_new_path"]
    log = {
        **row,
        "source_hash": "",
        "destination_hash": "",
        "status": "",
        "move_method": "",
        "message": "",
    }

    if not src.exists():
        if dst.exists():
            log["destination_hash"] = sha256(dst)
            log["status"] = "already_moved_source_missing"
            log["message"] = "source missing but destination exists"
        else:
            log["status"] = "skipped_source_missing"
            log["message"] = "source missing and destination missing"
        return log
    if src.is_dir():
        log["status"] = "skipped_source_is_directory"
        return log

    source_hash = sha256(src)
    log["source_hash"] = source_hash
    dst.parent.mkdir(parents=True, exist_ok=True)

    if dst.exists():
        dest_hash = sha256(dst)
        log["destination_hash"] = dest_hash
        if dest_hash == source_hash:
            src.unlink()
            log["status"] = "already_moved_or_duplicate"
            log["message"] = "destination already has identical content; source duplicate removed"
        else:
            log["status"] = "conflict_destination_exists_different_hash"
            log["message"] = "destination exists with different hash; skipped"
        return log

    tracked = is_git_tracked(src)
    if tracked:
        moved = git_mv(src, dst)
        if moved:
            log["move_method"] = "git mv"
        else:
            shutil.move(str(src), str(dst))
            log["move_method"] = "shutil.move_after_git_mv_failed"
    else:
        shutil.move(str(src), str(dst))
        log["move_method"] = "shutil.move"

    if not dst.exists():
        log["status"] = "conflict_destination_missing_after_move"
        return log
    dest_hash = sha256(dst)
    log["destination_hash"] = dest_hash
    if dest_hash == source_hash:
        log["status"] = "moved"
        log["message"] = "hash verified"
    else:
        log["status"] = "conflict_hash_mismatch_after_move"
        log["message"] = "destination hash does not match source hash"
    return log


def update_new_docs(log_rows: list[dict]) -> tuple[int, int]:
    replacements = {
        row["original_path"]: row["proposed_new_path"]
        for row in log_rows
        if row.get("status") in {"moved", "already_moved_or_duplicate", "already_moved_source_missing"}
    }
    changed_files = 0
    replacement_count = 0
    # Keep imported historical docs byte-stable; update only the newly created
    # top-level stage3_sde manifest/inventory docs.
    for path in (PROJECT_ROOT / "docs" / "stage3_sde").glob("*.md"):
        text = path.read_text(encoding="utf-8")
        new_text = text
        for old, new in replacements.items():
            count = new_text.count(old)
            if count:
                new_text = new_text.replace(old, new)
                replacement_count += count
        if new_text != text:
            path.write_text(new_text, encoding="utf-8")
            changed_files += 1
    return changed_files, replacement_count


def remove_empty_old_dirs() -> int:
    roots = [
        PROJECT_ROOT / "outputs" / "stage3",
        PROJECT_ROOT / "outputs" / "stage4",
        PROJECT_ROOT / "outputs" / "stage3_indoor",
        PROJECT_ROOT / "docs" / "stage4",
        PROJECT_ROOT / "tools" / "stage3",
        PROJECT_ROOT / "tools" / "stage4",
    ]
    removed = 0
    for root in roots:
        if not root.exists():
            continue
        for dirpath, dirnames, filenames in os.walk(root, topdown=False):
            path = Path(dirpath)
            try:
                if not any(path.iterdir()):
                    path.rmdir()
                    removed += 1
            except OSError:
                pass
    return removed


def main() -> None:
    rows = read_plan()
    eligible = [
        row
        for row in rows
        if row["should_move"] == "yes" and row["confidence"] == "high" and row["category"] != CATEGORY_UNKNOWN
    ]
    manual = [
        row
        for row in rows
        if not (row["should_move"] == "yes" and row["confidence"] == "high" and row["category"] != CATEGORY_UNKNOWN)
    ]

    log_rows = [safe_move(row) for row in eligible]
    docs_changed, replacements = update_new_docs(log_rows)
    empty_dirs_removed = remove_empty_old_dirs()

    conflict_rows = [row for row in log_rows if row["status"].startswith("conflict")]
    skipped_rows = [row for row in log_rows if row["status"].startswith("skipped")]

    write_csv(APPLY_LOG, log_rows)
    write_csv(CONFLICTS, conflict_rows)
    write_csv(MANUAL_REVIEW, manual)

    moved_count = sum(row["status"] == "moved" for row in log_rows)
    duplicate_deleted_count = sum(row["status"] == "already_moved_or_duplicate" for row in log_rows)
    skipped_count = len(skipped_rows)
    conflict_count = len(conflict_rows)
    manual_count = len(manual)

    summary = [
        "# Stage 3 SDE Reorganization Apply Summary",
        "",
        f"- moved file count: `{moved_count}`",
        f"- deleted original duplicate count: `{duplicate_deleted_count}`",
        f"- skipped file count: `{skipped_count}`",
        f"- conflict count: `{conflict_count}`",
        f"- manual review count: `{manual_count}`",
        f"- docs files updated: `{docs_changed}`",
        f"- reference replacements applied in docs/stage3_sde: `{replacements}`",
        f"- empty old directories removed: `{empty_dirs_removed}`",
        "",
        "## Logs",
        "",
        f"- apply log: `{rel(APPLY_LOG)}`",
        f"- conflicts: `{rel(CONFLICTS)}`",
        f"- manual review: `{rel(MANUAL_REVIEW)}`",
        "",
        "Scientific code logic, model code, raw datasets, and checkpoints were not modified.",
    ]
    APPLY_SUMMARY.write_text("\n".join(summary) + "\n", encoding="utf-8")
    print(f"[FILE] {rel(APPLY_SUMMARY)} written")
    print("STAGE3_SDE_REORGANIZATION_APPLY_COMPLETE")
    print(f"MOVED_COUNT={moved_count}")
    print(f"DELETED_ORIGINAL_DUPLICATE_COUNT={duplicate_deleted_count}")
    print(f"SKIPPED_COUNT={skipped_count}")
    print(f"CONFLICT_COUNT={conflict_count}")
    print(f"MANUAL_REVIEW_COUNT={manual_count}")
    print(f"DOCS_CHANGED={docs_changed}")


if __name__ == "__main__":
    main()
