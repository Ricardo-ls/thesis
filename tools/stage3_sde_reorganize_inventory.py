from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DOC_DIR = PROJECT_ROOT / "docs" / "stage3_sde"
CURATED_ROOT = PROJECT_ROOT / "outputs" / "curated" / "stage3_sde"
TOOLS_CURATED_ROOT = PROJECT_ROOT / "tools" / "curated" / "stage3_sde"

MOVE_PLAN = CURATED_ROOT / "reorganization_move_plan.csv"
PATH_MAPPING = CURATED_ROOT / "reorganization_path_mapping.json"
DRY_RUN_SUMMARY = CURATED_ROOT / "reorganization_dry_run_summary.md"

CATEGORY_MAINLINE = "MAINLINE_STAGE3_SDE"
CATEGORY_SUPP = "SUPPLEMENTARY_ABLATION"
CATEGORY_NEG = "NEGATIVE_DIAGNOSTIC"
CATEGORY_FUTURE = "FUTURE_SENSOR_INTERFACE"
CATEGORY_LEGACY = "LEGACY_OR_DEPRECATED"
CATEGORY_UNKNOWN = "UNKNOWN_NEEDS_MANUAL_REVIEW"

SCAN_ROOTS = [
    PROJECT_ROOT / "outputs" / "stage3",
    PROJECT_ROOT / "outputs" / "stage4",
    PROJECT_ROOT / "outputs" / "stage3_indoor",
    PROJECT_ROOT / "docs" / "stage4",
    PROJECT_ROOT / "tools" / "stage3",
    PROJECT_ROOT / "tools" / "stage4",
]

CURATED_STRUCTURE_DIRS = [
    DOC_DIR,
    CURATED_ROOT / "mainline" / "figures",
    CURATED_ROOT / "mainline" / "metrics",
    CURATED_ROOT / "mainline" / "arrays",
    CURATED_ROOT / "mainline" / "summaries",
    CURATED_ROOT / "supplementary_ablation" / "multit_candidate_source",
    CURATED_ROOT / "supplementary_ablation" / "gaussian_protocol_reconciliation",
    CURATED_ROOT / "supplementary_ablation" / "other",
    CURATED_ROOT / "negative_diagnostic" / "dps",
    CURATED_ROOT / "negative_diagnostic" / "e2_min",
    CURATED_ROOT / "negative_diagnostic" / "bias_sanity",
    CURATED_ROOT / "negative_diagnostic" / "old_sdedit_diagnosis",
    CURATED_ROOT / "negative_diagnostic" / "other",
    CURATED_ROOT / "future_sensor_interface",
    CURATED_ROOT / "legacy_or_deprecated",
    CURATED_ROOT / "unknown_manual_review",
    TOOLS_CURATED_ROOT / "mainline",
    TOOLS_CURATED_ROOT / "supplementary_ablation",
    TOOLS_CURATED_ROOT / "negative_diagnostic",
    TOOLS_CURATED_ROOT / "future_sensor_interface",
    TOOLS_CURATED_ROOT / "legacy_or_deprecated",
]


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def item_type(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix in {".png", ".jpg", ".jpeg", ".svg", ".pdf"}:
        return "figure"
    if suffix == ".csv":
        return "metric_csv"
    if suffix in {".npy", ".npz"}:
        return "array"
    if suffix == ".json":
        return "json"
    if suffix in {".md", ".markdown"}:
        return "markdown"
    if suffix == ".py":
        return "script"
    return "other"


def is_checkpoint_or_raw(path: Path) -> bool:
    suffix = path.suffix.lower()
    if suffix in {".pt", ".pth", ".ckpt", ".safetensors"}:
        return True
    text = rel(path).lower()
    if text.startswith("data/"):
        return True
    return False


def scan_files() -> list[Path]:
    files: list[Path] = []
    for root in SCAN_ROOTS:
        if root.exists():
            files.extend(path for path in root.rglob("*") if path.is_file())
    return sorted(set(files))


def classify(path: Path) -> tuple[str, str, str, bool]:
    r = rel(path)
    s = r.lower()
    typ = item_type(path)

    if "/__pycache__/" in s or s.endswith(".pyc") or ".matplotlib_cache" in s or s.endswith(".ds_store"):
        return CATEGORY_LEGACY, "low", "cache/system file; not moved", False

    if is_checkpoint_or_raw(path):
        if "ddpm_indoor_v2" in s or "best_ema_model" in s:
            return CATEGORY_MAINLINE, "medium", "checkpoint/raw-like artifact preserved in place", False
        return CATEGORY_UNKNOWN, "low", "checkpoint or raw-like artifact; manual review before moving", False

    # Current Stage 3 SDE mainline evidence.
    if "outputs/stage4/e3_holdout1000_confidence_aware_sdedit" in s:
        return CATEGORY_MAINLINE, "high", "current frozen holdout-1000 confidence-aware SDEdit mainline output", True
    if "outputs/stage4/stage3_sdedit_tstart123_confidence_hardfusion" in s:
        return CATEGORY_MAINLINE, "high", "mechanism cache for vanilla SDEdit plus confidence-aware hard-threshold fusion", True
    if "docs/stage4/e3_frozen_holdout1000_baseline.md" in s:
        return CATEGORY_MAINLINE, "high", "frozen E3/SDE baseline document", True
    if "tools/stage4/e3_holdout1000_confidence_aware_sdedit.py" in s:
        return CATEGORY_MAINLINE, "high", "mainline holdout-1000 confidence-aware SDEdit script", True
    if "tools/stage4/stage3_sdedit_tstart123_confidence_hardfusion.py" in s:
        return CATEGORY_MAINLINE, "high", "mainline mechanism cache/fusion script", True

    # Supplementary ablations and post-hoc SDE source analyses.
    if "e3_multit_candidate_source" in s or "e3_multit_per_condition_consistency" in s:
        return CATEGORY_SUPP, "high", "multi-t candidate-source ablation / consistency extraction", True
    if "e3_gain_bottleneck" in s:
        return CATEGORY_SUPP, "high", "gain bottleneck oracle/correction-direction diagnostic", True
    if "stage3_sde_legacy_audit" in s:
        return CATEGORY_NEG, "high", "legacy SDE/SDEdit audit supporting limitation analysis", True

    # Negative diagnostics: DPS, E2-Min, bias sanity, old SDEdit diagnosis.
    if "dps" in s:
        return CATEGORY_NEG, "high", "DPS audit or rejected DPS alternative", True
    if "e2_min" in s:
        return CATEGORY_NEG, "high", "E2-Min deterministic posterior anchoring diagnostic", True
    if "e2_prior_signal_sanity_bias" in s or "bias_prior_signal" in s:
        return CATEGORY_NEG, "high", "bias prior-signal sanity diagnostic", True
    if "stage3_sdedit_mechanism_diagnosis" in s or "sdedit_diagnostic" in s:
        return CATEGORY_NEG, "high", "old vanilla SDEdit mechanism/limitation diagnosis", True
    if "tools/stage4/stage3_sdedit_mechanism_diagnosis.py" in s:
        return CATEGORY_NEG, "high", "old SDEdit diagnosis script", True

    # Future interface / adapter references.
    if "sensor" in s or "adapter" in s:
        return CATEGORY_FUTURE, "medium", "future sensor-interface/adaptor reference", False
    if "docs/stage4/e3_observation_initialized_prior_interface_audit.md" in s:
        return CATEGORY_FUTURE, "high", "future observation-initialized prior interface audit", True

    # Conditional residual / old Stage 4 path is not the current SDE mainline.
    if "e1_" in s or "reconstructed_stage3_cond_outputs" in s or "conditional_residual" in s:
        return CATEGORY_LEGACY, "high", "conditional-residual Stage 4 branch superseded by SDE mainline", True
    if "docs/stage4/e2_dps_hypothesis" in s:
        return CATEGORY_NEG, "high", "DPS pre-registration/amendment for rejected diagnostic branch", True
    if "tools/stage4/rebuild_e1_confidence_cache.py" in s or "tools/stage4/reconstruct_missing_stage3_cond_outputs.py" in s:
        return CATEGORY_LEGACY, "high", "conditional-residual support script; not current SDE mainline", True

    # Stage 3 indoor reports and old prior diagnostics are useful but not unambiguously current.
    if "outputs/stage3_indoor/report" in s or "outputs/stage3_indoor/ddpm_prior" in s:
        return CATEGORY_LEGACY, "medium", "old indoor prior/report artifact; preserved unless manually reviewed", False
    if "outputs/stage3_indoor/receptive_field_expansion" in s or "sssd_ablation" in s:
        return CATEGORY_LEGACY, "medium", "conditional architecture ablation; not current SDE mainline", False

    if typ == "script" and ("tools/stage3/" in s or "tools/stage4/" in s):
        return CATEGORY_UNKNOWN, "low", "script not matched to current SDE cleanup rules", False

    return CATEGORY_UNKNOWN, "low", "no high-confidence cleanup rule matched", False


def destination_for(path: Path, category: str) -> Path:
    r = rel(path)
    typ = item_type(path)
    name = path.name
    parent_tag = path.parent.name
    parts = Path(r).parts
    source_tag = parent_tag
    if len(parts) >= 3 and parts[0] == "outputs" and parts[1] in {"stage3", "stage4", "stage3_indoor"}:
        source_tag = parts[2]
    elif len(parts) >= 3 and parts[0] == "docs":
        source_tag = parts[1]
    elif len(parts) >= 3 and parts[0] == "tools":
        source_tag = parts[1]
    if typ == "script":
        if category == CATEGORY_MAINLINE:
            return TOOLS_CURATED_ROOT / "mainline" / name
        if category == CATEGORY_SUPP:
            return TOOLS_CURATED_ROOT / "supplementary_ablation" / name
        if category == CATEGORY_NEG:
            return TOOLS_CURATED_ROOT / "negative_diagnostic" / name
        if category == CATEGORY_FUTURE:
            return TOOLS_CURATED_ROOT / "future_sensor_interface" / name
        if category == CATEGORY_LEGACY:
            return TOOLS_CURATED_ROOT / "legacy_or_deprecated" / name
        return TOOLS_CURATED_ROOT / "legacy_or_deprecated" / name

    if r.startswith("docs/stage4/"):
        return DOC_DIR / "imported_stage4_docs" / name

    if category == CATEGORY_MAINLINE:
        base = CURATED_ROOT / "mainline"
        if typ == "figure":
            return base / "figures" / source_tag / name
        if typ == "metric_csv":
            return base / "metrics" / source_tag / name
        if typ == "array":
            return base / "arrays" / source_tag / name
        if typ in {"markdown", "json"}:
            return base / "summaries" / source_tag / name
        return base / "summaries" / source_tag / name

    if category == CATEGORY_SUPP:
        if "e3_multit_candidate_source" in r.lower() or "e3_multit_per_condition" in r.lower():
            return CURATED_ROOT / "supplementary_ablation" / "multit_candidate_source" / source_tag / parent_tag / name
        return CURATED_ROOT / "supplementary_ablation" / "other" / source_tag / parent_tag / name

    if category == CATEGORY_NEG:
        sl = r.lower()
        if "dps" in sl:
            sub = "dps"
        elif "e2_min" in sl:
            sub = "e2_min"
        elif "bias" in sl:
            sub = "bias_sanity"
        elif "sdedit" in sl or "sde_legacy" in sl:
            sub = "old_sdedit_diagnosis"
        else:
            sub = "other"
        return CURATED_ROOT / "negative_diagnostic" / sub / source_tag / parent_tag / name

    if category == CATEGORY_FUTURE:
        return CURATED_ROOT / "future_sensor_interface" / source_tag / parent_tag / name
    if category == CATEGORY_LEGACY:
        return CURATED_ROOT / "legacy_or_deprecated" / source_tag / parent_tag / name
    return CURATED_ROOT / "unknown_manual_review" / source_tag / parent_tag / name


def build_rows() -> list[dict]:
    rows = []
    for path in scan_files():
        category, confidence, notes, eligible = classify(path)
        move = eligible and confidence == "high" and category != CATEGORY_UNKNOWN and not is_checkpoint_or_raw(path)
        dest = destination_for(path, category)
        rows.append(
            {
                "original_path": rel(path),
                "proposed_new_path": rel(dest),
                "category": category,
                "item_type": item_type(path),
                "should_move": "yes" if move else "no",
                "should_delete_original_after_move": "yes" if move else "no",
                "requires_reference_update": "yes" if move else "no",
                "confidence": confidence,
                "notes": notes,
            }
        )
    return rows


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "original_path",
        "proposed_new_path",
        "category",
        "item_type",
        "should_move",
        "should_delete_original_after_move",
        "requires_reference_update",
        "confidence",
        "notes",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"[FILE] {rel(path)} written")


def write_path_mapping(rows: list[dict]) -> None:
    mapping = {
        row["original_path"]: {
            "proposed_new_path": row["proposed_new_path"],
            "category": row["category"],
            "should_move": row["should_move"],
            "confidence": row["confidence"],
        }
        for row in rows
    }
    PATH_MAPPING.write_text(json.dumps(mapping, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"[FILE] {rel(PATH_MAPPING)} written")


def list_by_category(rows: list[dict], category: str, move_only: bool = False) -> list[str]:
    selected = [row for row in rows if row["category"] == category and (not move_only or row["should_move"] == "yes")]
    return [row["original_path"] for row in selected]


def write_docs(rows: list[dict]) -> None:
    DOC_DIR.mkdir(parents=True, exist_ok=True)
    for path in CURATED_STRUCTURE_DIRS:
        path.mkdir(parents=True, exist_ok=True)

    counts_move = Counter(row["category"] for row in rows if row["should_move"] == "yes")
    counts_all = Counter(row["category"] for row in rows)
    unknown = [row for row in rows if row["category"] == CATEGORY_UNKNOWN]
    mainline = [row for row in rows if row["category"] == CATEGORY_MAINLINE and row["should_move"] == "yes"]
    supp = [row for row in rows if row["category"] == CATEGORY_SUPP and row["should_move"] == "yes"]
    neg = [row for row in rows if row["category"] == CATEGORY_NEG and row["should_move"] == "yes"]
    future = [row for row in rows if row["category"] == CATEGORY_FUTURE]
    legacy = [row for row in rows if row["category"] == CATEGORY_LEGACY]

    output_inventory = [
        "# Stage 3 SDE Output Inventory",
        "",
        "Generated by `tools/stage3_sde_reorganize_inventory.py` as a dry-run inventory. No experiments were rerun.",
        "",
        "## Category Counts",
        "",
        "| category | all_files | proposed_move_high_confidence |",
        "| --- | ---: | ---: |",
    ]
    for category in [CATEGORY_MAINLINE, CATEGORY_SUPP, CATEGORY_NEG, CATEGORY_FUTURE, CATEGORY_LEGACY, CATEGORY_UNKNOWN]:
        output_inventory.append(f"| {category} | {counts_all[category]} | {counts_move[category]} |")
    output_inventory.extend(
        [
            "",
            "## Move Plan",
            "",
            f"- CSV: `{rel(MOVE_PLAN)}`",
            f"- JSON mapping: `{rel(PATH_MAPPING)}`",
            f"- Dry-run summary: `{rel(DRY_RUN_SUMMARY)}`",
        ]
    )
    (DOC_DIR / "output_inventory.md").write_text("\n".join(output_inventory) + "\n", encoding="utf-8")

    manifest_lines = [
        "# Stage 3 SDE Mainline Manifest",
        "",
        "Future Codex work should treat this file as the source of truth for the current Stage 3 SDE mainline.",
        "",
        "## Current Mainline",
        "",
        "The current mainline is unconditional SDEdit / prior-guided refinement:",
        "",
        "`degraded trajectory -> unconditional DDPM prior -> SDEdit / prior-guided refinement -> vanilla capability and limitation analysis -> confidence-aware stabilization`",
        "",
        "This is not the full Stage 4 real-sensor system.",
        "",
        "## Core Hypothesis",
        "",
        "A pre-trained unconditional trajectory DDPM can serve as a domain-knowledge prior to pull degraded trajectories toward physically plausible and smoother human trajectories, but raw SDEdit is not reliable as a standalone final estimator and requires confidence-aware stabilization.",
        "",
        "## Curated Mainline Paths",
        "",
    ]
    for row in mainline:
        manifest_lines.append(f"- `{row['proposed_new_path']}` <- `{row['original_path']}`")
    manifest_lines.extend(
        [
            "",
            "## Non-Mainline Warning",
            "",
            "Do not treat Stage 4/DPS/multi-t/sensor-adapter outputs as mainline unless explicitly instructed.",
        ]
    )
    (DOC_DIR / "mainline_manifest.md").write_text("\n".join(manifest_lines) + "\n", encoding="utf-8")

    narrative = [
        "# Stage 3 SDE Narrative Alignment",
        "",
        "## Scientific Framing",
        "",
        "The project is currently organized around Stage 3 unconditional SDEdit / prior-guided refinement.",
        "",
        "## Evidence Roles",
        "",
        "- Mainline: frozen holdout-1000 confidence-aware SDEdit and mechanism hard-threshold fusion evidence.",
        "- Supplementary ablation: multi-t candidate-source tests, gain bottleneck diagnostics, and consistency extraction.",
        "- Negative diagnostic: DPS, E2-Min posterior anchoring, bias sanity, and legacy/old SDEdit limitation audits.",
        "- Future sensor interface: observation-initialized or sensor-adapter work that is not part of the current mainline.",
        "",
        "## Current Interpretation",
        "",
        "Raw SDEdit has limited and condition-dependent benefit. Confidence-aware stabilization is the current stable SDE component. Multi-t and DPS artifacts should not be interpreted as mainline unless a later manifest revision says so.",
    ]
    (DOC_DIR / "narrative_alignment.md").write_text("\n".join(narrative) + "\n", encoding="utf-8")

    deprecated = [
        "# Deprecated And Non-Mainline Outputs",
        "",
        "The following files are categorized as legacy/deprecated or negative diagnostic, not current Stage 3 SDE mainline.",
        "",
        "## Negative Diagnostic",
        "",
        *[f"- `{row['original_path']}` -> `{row['proposed_new_path']}`" for row in neg],
        "",
        "## Legacy Or Deprecated",
        "",
        *[f"- `{row['original_path']}` -> `{row['proposed_new_path']}`" for row in legacy if row["should_move"] == "yes"],
        "",
        "## Unknown Manual Review",
        "",
        *[f"- `{row['original_path']}` ({row['notes']})" for row in unknown[:250]],
    ]
    (DOC_DIR / "deprecated_outputs.md").write_text("\n".join(deprecated) + "\n", encoding="utf-8")

    path_mapping_md = [
        "# Stage 3 SDE Path Mapping",
        "",
        f"Machine-readable mapping: `{rel(PATH_MAPPING)}`",
        "",
        "| category | original_path | proposed_new_path | move | confidence |",
        "| --- | --- | --- | --- | --- |",
    ]
    for row in rows:
        if row["should_move"] == "yes":
            path_mapping_md.append(
                f"| {row['category']} | `{row['original_path']}` | `{row['proposed_new_path']}` | {row['should_move']} | {row['confidence']} |"
            )
    (DOC_DIR / "path_mapping.md").write_text("\n".join(path_mapping_md) + "\n", encoding="utf-8")

    for name in [
        "output_inventory.md",
        "mainline_manifest.md",
        "narrative_alignment.md",
        "deprecated_outputs.md",
        "path_mapping.md",
    ]:
        print(f"[FILE] {rel(DOC_DIR / name)} written")

    dry = [
        "# Stage 3 SDE Reorganization Dry-Run Summary",
        "",
        f"Generated: `{datetime.now().isoformat(timespec='seconds')}`",
        "",
        "No files were moved by this dry-run.",
        "",
        "## Proposed Move Counts",
        "",
        "| category | files_proposed_to_move | total_inventory_files |",
        "| --- | ---: | ---: |",
    ]
    for category in [CATEGORY_MAINLINE, CATEGORY_SUPP, CATEGORY_NEG, CATEGORY_FUTURE, CATEGORY_LEGACY, CATEGORY_UNKNOWN]:
        dry.append(f"| {category} | {counts_move[category]} | {counts_all[category]} |")
    dry.extend(
        [
            "",
            f"- UNKNOWN files blocked as manual review: `{len(unknown)}`",
            "",
            "## MAINLINE_STAGE3_SDE Paths",
            "",
            *[f"- `{row['original_path']}` -> `{row['proposed_new_path']}`" for row in mainline],
            "",
            "## SUPPLEMENTARY_ABLATION Paths",
            "",
            *[f"- `{row['original_path']}` -> `{row['proposed_new_path']}`" for row in supp],
            "",
            "## NEGATIVE_DIAGNOSTIC Paths",
            "",
            *[f"- `{row['original_path']}` -> `{row['proposed_new_path']}`" for row in neg],
            "",
            "## FUTURE_SENSOR_INTERFACE Paths",
            "",
            *[f"- `{row['original_path']}` -> `{row['proposed_new_path']}`" for row in future],
            "",
            "## LEGACY_OR_DEPRECATED Paths",
            "",
            *[f"- `{row['original_path']}` -> `{row['proposed_new_path']}`" for row in legacy if row["should_move"] == "yes"],
            "",
            "## UNKNOWN_NEEDS_MANUAL_REVIEW Paths",
            "",
            *[f"- `{row['original_path']}` ({row['notes']})" for row in unknown[:500]],
            "",
            "## Reference Update Warnings",
            "",
            "- Rows with `requires_reference_update=yes` will be string-replaced in newly created `docs/stage3_sde/*.md` by the apply script.",
            "- Existing scientific scripts are not automatically rewritten unless the path update is obviously safe.",
        ]
    )
    DRY_RUN_SUMMARY.write_text("\n".join(dry) + "\n", encoding="utf-8")
    print(f"[FILE] {rel(DRY_RUN_SUMMARY)} written")


def main() -> None:
    rows = build_rows()
    for path in CURATED_STRUCTURE_DIRS:
        path.mkdir(parents=True, exist_ok=True)
    write_csv(MOVE_PLAN, rows)
    write_path_mapping(rows)
    write_docs(rows)
    print("STAGE3_SDE_REORGANIZATION_INVENTORY_COMPLETE")
    print(f"MOVE_PLAN={rel(MOVE_PLAN)}")
    print(f"PATH_MAPPING={rel(PATH_MAPPING)}")
    print(f"DRY_RUN_SUMMARY={rel(DRY_RUN_SUMMARY)}")


if __name__ == "__main__":
    main()
