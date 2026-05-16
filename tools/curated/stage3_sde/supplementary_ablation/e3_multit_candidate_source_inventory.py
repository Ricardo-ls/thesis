from __future__ import annotations

import csv
import os
import re
import sys
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

CONDITIONS = [
    "gaussian_medium",
    "drift_medium",
    "burst_medium",
    "bias_medium",
    "jump_medium",
    "combined_medium",
]
T_STARTS = [1, 2, 3]
EXPECTED_CANDIDATE_SHAPE = (1000, 20, 2)
EXPECTED_CONFIDENCE_SHAPE = (1000, 20)

EXPECTED_DATASET = PROJECT_ROOT / "data" / "stage4" / "e3_holdout_1000" / "clean_trajs.npy"
CACHE_ROOT = PROJECT_ROOT / "outputs" / "stage4" / "e3_holdout1000_confidence_aware_sdedit"
OUT_DIR = PROJECT_ROOT / "outputs" / "stage4" / "e3_multit_candidate_source_ablation"
RAW_LIST_PATH = OUT_DIR / "cache_inventory_raw_file_list.txt"
CSV_PATH = OUT_DIR / "cache_inventory.csv"
MD_PATH = OUT_DIR / "cache_inventory.md"


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def condition_in_path(path: Path) -> str | None:
    text = str(path).lower()
    matches = [condition for condition in CONDITIONS if condition.lower() in text]
    return matches[0] if len(matches) == 1 else None


def infer_t_start(path: Path) -> int | None:
    text = path.name.lower()
    patterns = [
        r"(?:^|[_\-/])sdedit[_\-/]?t([123])(?:[_\-.]|$)",
        r"(?:^|[_\-/])t_start[_\-/]?([123])(?:[_\-.]|$)",
        r"(?:^|[_\-/])t([123])(?:[_\-.]|$)",
    ]
    matches: list[int] = []
    for pattern in patterns:
        for match in re.finditer(pattern, text):
            value = int(match.group(1))
            if value in T_STARTS:
                matches.append(value)
    unique = sorted(set(matches))
    return unique[0] if len(unique) == 1 else None


def file_stats(path: Path) -> dict:
    stat = path.stat()
    row = {
        "path": rel(path),
        "file_size_bytes": stat.st_size,
        "modified_time": datetime.fromtimestamp(stat.st_mtime).isoformat(timespec="seconds"),
        "shape": "",
        "dtype": "",
        "finite_ok": "",
        "numeric_min": "",
        "numeric_max": "",
        "numeric_mean": "",
        "load_error": "",
    }
    try:
        import numpy as np

        arr = np.load(path, allow_pickle=False)
        row["shape"] = str(tuple(arr.shape))
        row["dtype"] = str(arr.dtype)
        if np.issubdtype(arr.dtype, np.number):
            finite = bool(np.isfinite(arr).all())
            row["finite_ok"] = finite
            row["numeric_min"] = float(np.nanmin(arr))
            row["numeric_max"] = float(np.nanmax(arr))
            row["numeric_mean"] = float(np.nanmean(arr))
        else:
            row["finite_ok"] = "non_numeric"
    except Exception as exc:
        row["load_error"] = repr(exc)
        row["finite_ok"] = False
    return row


def companion_kind(path: Path) -> str | None:
    name = path.name.lower()
    if "clean" in name:
        return "clean"
    if "degraded" in name or "noisy" in name:
        return "degraded"
    if "confidence" in name or re.search(r"(?:^|_)c(?:[_\.]|$)", name):
        return "confidence"
    if "fused" in name:
        return "fused"
    return None


def bool_text(value: object) -> str:
    if value is True:
        return "yes"
    if value is False:
        return "no"
    return str(value)


def markdown_table(rows: list[dict], columns: list[str]) -> str:
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for row in rows:
        vals = []
        for col in columns:
            value = row.get(col, "")
            if isinstance(value, float):
                vals.append(f"{value:.6g}")
            else:
                vals.append(str(value))
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


def main() -> None:
    if not CACHE_ROOT.is_dir():
        raise FileNotFoundError(f"Missing expected cache root: {CACHE_ROOT}")
    if not EXPECTED_DATASET.is_file():
        raise FileNotFoundError(f"Missing expected dataset: {EXPECTED_DATASET}")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    files = sorted(path for path in CACHE_ROOT.rglob("*") if path.is_file() and path.suffix.lower() in {".npy", ".npz"})
    RAW_LIST_PATH.write_text("\n".join(rel(path) for path in files) + "\n", encoding="utf-8")
    print(f"[FILE] {rel(RAW_LIST_PATH)} written")

    raw_rows: list[dict] = []
    candidate_matches: dict[tuple[str, int], list[dict]] = {(c, t): [] for c in CONDITIONS for t in T_STARTS}
    companion_matches: dict[tuple[str, str], list[dict]] = {}

    for path in files:
        stats = file_stats(path)
        condition = condition_in_path(path)
        t_start = infer_t_start(path)
        kind = companion_kind(path)
        is_candidate = condition is not None and t_start is not None and "sdedit" in path.name.lower()
        row = {
            **stats,
            "inferred_condition": condition or "",
            "inferred_t_start": t_start if t_start is not None else "",
            "inferred_kind": "sdedit_candidate" if is_candidate else kind or "",
            "is_candidate_match": is_candidate,
        }
        raw_rows.append(row)
        if is_candidate and condition is not None and t_start is not None:
            candidate_matches[(condition, t_start)].append(row)
        if condition is not None and kind is not None:
            companion_matches.setdefault((condition, kind), []).append(row)

    completeness_rows: list[dict] = []
    missing: list[str] = []
    ambiguous: list[str] = []
    invalid: list[str] = []
    for condition in CONDITIONS:
        for t_start in T_STARTS:
            matches = candidate_matches[(condition, t_start)]
            found = len(matches) > 0
            amb = len(matches) > 1
            selected = matches[0] if len(matches) == 1 else None
            shape_ok = selected is not None and selected["shape"] == str(EXPECTED_CANDIDATE_SHAPE)
            finite_ok = selected is not None and selected["finite_ok"] is True
            notes = []
            if not found:
                missing.append(f"{condition} × t{t_start}")
                notes.append("missing")
            if amb:
                ambiguous.append(f"{condition} × t{t_start}")
                notes.append("multiple matches: " + "; ".join(match["path"] for match in matches))
            if selected is not None and not shape_ok:
                invalid.append(f"{condition} × t{t_start}: shape {selected['shape']}")
                notes.append(f"shape invalid: {selected['shape']}")
            if selected is not None and not finite_ok:
                invalid.append(f"{condition} × t{t_start}: finite_ok={selected['finite_ok']}")
                notes.append(f"finite invalid: {selected['finite_ok']}")
            completeness_rows.append(
                {
                    "condition": condition,
                    "t_start": t_start,
                    "found": found,
                    "path": selected["path"] if selected is not None else "; ".join(match["path"] for match in matches),
                    "shape": selected["shape"] if selected is not None else "",
                    "dtype": selected["dtype"] if selected is not None else "",
                    "shape_ok": shape_ok,
                    "finite_ok": finite_ok,
                    "ambiguous": amb,
                    "notes": " | ".join(notes),
                }
            )

    companion_rows: list[dict] = []
    for condition in CONDITIONS:
        for kind in ["clean", "degraded", "confidence", "fused"]:
            matches = companion_matches.get((condition, kind), [])
            expected_shape = EXPECTED_CONFIDENCE_SHAPE if kind == "confidence" else EXPECTED_CANDIDATE_SHAPE
            if len(matches) == 1:
                match = matches[0]
                shape_ok = match["shape"] == str(expected_shape)
                finite_ok = match["finite_ok"] is True
                path = match["path"]
                notes = ""
            elif len(matches) > 1:
                shape_ok = False
                finite_ok = False
                path = "; ".join(match["path"] for match in matches)
                notes = "ambiguous companion matches"
            else:
                shape_ok = False
                finite_ok = False
                path = ""
                notes = "missing companion"
            companion_rows.append(
                {
                    "condition": condition,
                    "kind": kind,
                    "found": len(matches) > 0,
                    "path": path,
                    "shape_ok": shape_ok,
                    "finite_ok": finite_ok,
                    "ambiguous": len(matches) > 1,
                    "notes": notes,
                }
            )

    with CSV_PATH.open("w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "condition",
            "t_start",
            "found",
            "path",
            "shape",
            "dtype",
            "shape_ok",
            "finite_ok",
            "ambiguous",
            "notes",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(completeness_rows)
    print(f"[FILE] {rel(CSV_PATH)} written")

    all_complete = not missing and not ambiguous and not invalid
    all_shape_ok = all(row["shape_ok"] for row in completeness_rows)
    all_finite_ok = all(row["finite_ok"] for row in completeness_rows)
    any_ambiguous = any(row["ambiguous"] for row in completeness_rows)
    any_missing = bool(missing)
    any_invalid = bool(invalid)

    matrix_rows = []
    for condition in CONDITIONS:
        row = {"condition": condition, "notes": ""}
        notes = []
        for t in T_STARTS:
            item = next(r for r in completeness_rows if r["condition"] == condition and r["t_start"] == t)
            if item["found"] and item["shape_ok"] and item["finite_ok"] and not item["ambiguous"]:
                row[f"t{t}"] = "OK"
            elif not item["found"]:
                row[f"t{t}"] = "MISSING"
                notes.append(f"t{t} missing")
            elif item["ambiguous"]:
                row[f"t{t}"] = "AMBIGUOUS"
                notes.append(f"t{t} ambiguous")
            else:
                row[f"t{t}"] = "INVALID"
                notes.append(f"t{t} invalid")
        row["notes"] = "; ".join(notes)
        matrix_rows.append(row)

    if all_complete:
        next_action = "READY_FOR_EVAL"
        next_detail = "All t1/t2/t3 candidates are present and valid."
    elif ambiguous or invalid:
        next_action = "NEED_MANUAL_RESOLUTION"
        items = ambiguous + invalid
        next_detail = "Manual resolution needed for: " + "; ".join(items)
    else:
        next_action = "NEED_GENERATE_MISSING"
        next_detail = "Missing candidates: " + "; ".join(missing)

    detailed_candidate_rows = [row for row in raw_rows if row["is_candidate_match"]]
    lines = [
        "# E3 Multi-t Candidate Source Cache Inventory",
        "",
        "Inventory-only pass. No SDEdit candidates were generated, no evaluation was run, and no existing E3 outputs were modified.",
        "",
        "## A. Summary",
        "",
        f"- Python executable used: `{sys.executable}`",
        f"- Expected dataset: `{rel(EXPECTED_DATASET)}`",
        f"- Expected cache root: `{rel(CACHE_ROOT)}`",
        f"- Raw .npy/.npz files scanned: `{len(files)}`",
        f"- Complete 6 conditions x 3 t_start candidates: `{all_complete}`",
        f"- All candidates shape [1000,20,2]: `{all_shape_ok}`",
        f"- Any missing candidate: `{any_missing}`",
        f"- Any ambiguous candidate match: `{any_ambiguous}`",
        f"- Any invalid candidate: `{any_invalid}`",
        f"- Any candidate NaN/Inf: `{not all_finite_ok}`",
        "",
        "## B. Completeness Matrix",
        "",
        markdown_table(matrix_rows, ["condition", "t1", "t2", "t3", "notes"]),
        "",
        "## C. Detailed File Table",
        "",
        markdown_table(
            detailed_candidate_rows,
            [
                "path",
                "inferred_condition",
                "inferred_t_start",
                "shape",
                "dtype",
                "finite_ok",
                "numeric_min",
                "numeric_max",
                "numeric_mean",
                "file_size_bytes",
                "modified_time",
            ],
        ),
        "",
        "## Companion Array Inventory",
        "",
        markdown_table(companion_rows, ["condition", "kind", "found", "path", "shape_ok", "finite_ok", "ambiguous", "notes"]),
        "",
        "## D. Required Next Action",
        "",
        next_action,
        "",
        next_detail,
    ]
    MD_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[FILE] {rel(MD_PATH)} written")
    print(f"NEXT_ACTION={next_action}")


if __name__ == "__main__":
    main()
