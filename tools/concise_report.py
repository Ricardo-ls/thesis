from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import re
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
from PIL import Image
from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.shared import Inches, Pt
from docx.text.paragraph import Paragraph


INPUT_DOC = PROJECT_ROOT / "stage3_report_complete.docx"
OUTPUT_DOC = PROJECT_ROOT / "stage3_report_v3_concise.docx"
DATA_DIR = PROJECT_ROOT / "outputs" / "stage3_indoor" / "conditional_residual_ddpm_gaussian" / "seed42"
CSV_PATH = DATA_DIR / "generalization_summary.csv"
FIG_GAUSSIAN = DATA_DIR / "cond_residual_gaussian_eval.png"
FIG_GENERALIZATION = DATA_DIR / "generalization_diagnostic.png"

METHODS = ["noisy_input", "uncond_sdedit_t2", "cond_residual_t20"]
DEGRADATIONS = [
    "gaussian_medium",
    "drift_medium",
    "jump_medium",
    "burst_medium",
    "bias_medium",
    "combined_medium",
]
RESULT_MAP = {
    ("gaussian_medium", "noisy_input"): "—",
    ("gaussian_medium", "uncond_sdedit_t2"): "sig. impr.",
    ("gaussian_medium", "cond_residual_t20"): "sig. impr.",
    ("drift_medium", "noisy_input"): "—",
    ("drift_medium", "uncond_sdedit_t2"): "worsened",
    ("drift_medium", "cond_residual_t20"): "over-corr.",
    ("jump_medium", "noisy_input"): "—",
    ("jump_medium", "uncond_sdedit_t2"): "small impr.",
    ("jump_medium", "cond_residual_t20"): "sig. impr.",
    ("burst_medium", "noisy_input"): "—",
    ("burst_medium", "uncond_sdedit_t2"): "worsened",
    ("burst_medium", "cond_residual_t20"): "worsened",
    ("bias_medium", "noisy_input"): "—",
    ("bias_medium", "uncond_sdedit_t2"): "no change",
    ("bias_medium", "cond_residual_t20"): "no change",
    ("combined_medium", "noisy_input"): "—",
    ("combined_medium", "uncond_sdedit_t2"): "no change",
    ("combined_medium", "cond_residual_t20"): "no change",
}


def ensure_inputs() -> None:
    for path in [INPUT_DOC, CSV_PATH, FIG_GAUSSIAN, FIG_GENERALIZATION]:
        if not path.is_file():
            raise FileNotFoundError(f"Missing required input: {path}")


def word_count_doc(doc: Document) -> int:
    pattern = re.compile(r"\b\w+[\w.-]*\b")
    count = sum(len(pattern.findall(p.text)) for p in doc.paragraphs)
    for table in doc.tables:
        for row in table.rows:
            for cell in row.cells:
                count += len(pattern.findall(cell.text))
    return count


def fmt4(value: float) -> str:
    return f"{value:.4f}"


def fmt_delta(value: float) -> str:
    return f"{value:+.4f}"


def fmt_percent(value: float) -> str:
    return f"{value * 100:.1f}%"


def fmt_p_main(value: float) -> str:
    if pd.isna(value):
        return "—"
    if value < 1e-3:
        return f"{value:.1e}"
    if value >= 0.995:
        return "1.0"
    return f"{value:.2f}"


def fmt_p_appendix(value: float) -> str:
    if pd.isna(value):
        return "—"
    if value < 1e-3 or value >= 1e4:
        return f"{value:.1e}"
    return f"{value:.4f}"


def get_row(df: pd.DataFrame, degradation: str, method: str) -> pd.Series:
    subset = df[(df["degradation"] == degradation) & (df["method"] == method)]
    if subset.empty:
        raise ValueError(f"Missing CSV row for {degradation} / {method}")
    return subset.iloc[0]


def copy_table_after(paragraph: Paragraph, source_table) -> None:
    paragraph._p.addnext(deepcopy(source_table._tbl))


def add_caption(doc: Document, text: str) -> None:
    p = doc.add_paragraph()
    run = p.add_run(text)
    run.italic = True
    run.font.size = Pt(9)
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER


def add_figure_block(doc: Document, image_path: Path, caption: str) -> None:
    with Image.open(image_path):
        pass
    doc.add_paragraph("")
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.add_run().add_picture(str(image_path), width=Inches(6.5))
    add_caption(doc, caption)
    doc.add_paragraph("")


def add_simple_table(doc: Document, headers: list[str], rows: list[list[str]]) -> None:
    table = doc.add_table(rows=1, cols=len(headers))
    table.style = "Table Grid"
    for idx, header in enumerate(headers):
        table.rows[0].cells[idx].text = header
    for values in rows:
        cells = table.add_row().cells
        for idx, value in enumerate(values):
            cells[idx].text = value


def build_main_summary_rows(df: pd.DataFrame) -> list[list[str]]:
    outcome = {
        "gaussian_medium": "sig. improvement",
        "drift_medium": "over-correction",
        "jump_medium": "sig. improvement",
        "burst_medium": "worsened",
        "bias_medium": "no change",
        "combined_medium": "no change",
    }
    labels = {
        "gaussian_medium": "gaussian",
        "drift_medium": "drift",
        "jump_medium": "jump",
        "burst_medium": "burst",
        "bias_medium": "bias",
        "combined_medium": "combined",
    }
    rows = []
    for degradation in DEGRADATIONS:
        noisy = get_row(df, degradation, "noisy_input")
        cond = get_row(df, degradation, "cond_residual_t20")
        rows.append(
            [
                labels[degradation],
                fmt4(float(noisy["ADE_mean"])),
                fmt4(float(cond["ADE_mean"])),
                fmt_delta(float(cond["delta_ADE_vs_noisy_mean"])),
                fmt_p_main(float(cond["wilcoxon_p_vs_noisy"])),
                outcome[degradation],
            ]
        )
    return rows


def build_appendix_rows(df: pd.DataFrame) -> list[list[str]]:
    rows = []
    for degradation in DEGRADATIONS:
        for method in METHODS:
            record = get_row(df, degradation, method)
            if method == "noisy_input":
                delta, imp, pval = "—", "—", "—"
            else:
                delta = fmt_delta(float(record["delta_ADE_vs_noisy_mean"]))
                imp = fmt_percent(float(record["improved_fraction"]))
                pval = fmt_p_appendix(float(record["wilcoxon_p_vs_noisy"]))
            rows.append(
                [
                    degradation,
                    method,
                    fmt4(float(record["ADE_mean"])),
                    fmt4(float(record["ADE_std"])),
                    fmt4(float(record["smooth_mean"])),
                    delta,
                    imp,
                    pval,
                    RESULT_MAP[(degradation, method)],
                ]
            )
    return rows


def add_bullets(doc: Document, items: list[str]) -> None:
    for item in items:
        doc.add_paragraph(item, style="List Bullet")


def main() -> None:
    ensure_inputs()
    df = pd.read_csv(CSV_PATH)
    print("generalization_summary.csv columns:")
    print(list(df.columns))
    print(df.head(3).to_string(index=False))

    required = [
        "degradation",
        "method",
        "ADE_mean",
        "ADE_std",
        "smooth_mean",
        "delta_ADE_vs_noisy_mean",
        "improved_fraction",
        "wilcoxon_p_vs_noisy",
    ]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"CSV missing required columns: {missing}. Actual columns: {list(df.columns)}")

    source_doc = Document(str(INPUT_DOC))
    old_word_count = word_count_doc(source_doc)
    if not source_doc.tables:
        raise RuntimeError("Source report has no tables; cannot preserve Table 1")
    source_table1 = source_doc.tables[0]

    doc = Document()
    section = doc.sections[0]
    section.left_margin = Inches(0.55)
    section.right_margin = Inches(0.55)
    section.top_margin = Inches(0.6)
    section.bottom_margin = Inches(0.6)

    title = doc.add_heading(
        "Stage 3: Controlled Evaluation of Conditional DDPM Refinement\nfor Indoor Sensor-Degraded Trajectories",
        level=0,
    )
    title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    subtitle = doc.add_paragraph("Stage 3 Experiment Report — May 2026")
    subtitle.alignment = WD_ALIGN_PARAGRAPH.CENTER

    doc.add_heading("1. Objective and Hypotheses", level=1)
    doc.add_paragraph(
        "Objective: Evaluate whether a DDPM trained on indoor simulated trajectories can refine sensor-degraded trajectories under controlled corruption conditions."
    )
    doc.add_paragraph(
        "H1 (primary): Conditional residual DDPM, where the degraded trajectory is provided as a condition and the model learns a displacement correction, improves trajectory reconstruction beyond both noisy input and unconditional SDEdit."
    )
    doc.add_paragraph(
        "H2 (secondary): The effectiveness is bounded by whether corruption is observable in relative displacement; constant absolute bias should remain uncorrectable."
    )
    doc.add_paragraph(
        "This work shows (i) unconditional DDPM-SDEdit yields only weak refinement (2.8% ADE under Gaussian noise); (ii) conditional residual DDPM substantially improves Gaussian refinement (11.0%, p = 5.3×10⁻¹¹) and significantly outperforms unconditional SDEdit; (iii) the method generalizes partially to jump corruption but fails on burst and drift, and is structurally unable to correct constant bias."
    )

    doc.add_heading("2. Method", level=1)
    doc.add_paragraph(
        "Stage 2 established the trajectory DDPM pipeline (relative displacement representation, 1D temporal denoiser, training/sampling procedures) on public ETH/UCY data. Stage 3 retrains a separate model from scratch on indoor simulated trajectories, focused on degradation refinement rather than open-ended generation."
    )
    doc.add_paragraph(
        "DDPM choice: a learned probabilistic prior captures the multi-modal distribution of indoor motion (walks, turns, pauses) that hand-crafted filters cannot. Conditional residual choice: unconditional SDEdit gave only 2.8% improvement; concatenating the degraded trajectory as a condition channel and learning the residual correction shrinks the search space and aligns the task with sensor refinement."
    )
    doc.add_paragraph(
        "Baselines: noisy_input (raw degraded), kalman_cv (constant-velocity RTS, fixed parameters, not optimized), and unconditional SDEdit (t=2)."
    )

    doc.add_heading("3. Experimental Protocol", level=1)
    doc.add_paragraph(
        "Data: 2,000 indoor simulated trajectories in a 3m × 3m room, 20 frames at 3 Hz. Five behavior types (goal-directed, multi-goal, pacing, stationary, boundary-walk). Training set 10,000 expanded to 60,000 via 6-fold geometric augmentation."
    )
    doc.add_paragraph("Degradations (six types, per-frame):")
    add_bullets(
        doc,
        [
            "gaussian_medium: N(0, 0.05²)",
            "bias_medium: constant N(0, 0.15²) per axis",
            "drift_medium: cumulative random walk σ_step=0.005",
            "jump_medium: 2–4 piecewise offsets, U(0.2, 0.5)m",
            "burst_medium: 3–5 frame burst, σ=0.25 (vs 0.01 elsewhere)",
            "combined_medium: gaussian + bias + drift",
        ],
    )
    doc.add_paragraph(
        "Model: 1D conditional temporal denoiser, hidden_dim=128, T=100, input channels = 4 (noisy state + degraded condition). Trained 60 epochs on Gaussian degradation only, with EMA. (Training details: Appendix A.)"
    )
    doc.add_paragraph(
        "Evaluation: N=200 trajectories. DDPM methods average over 5 inference seeds before computing per-trajectory ADE, RMSE, smooth (acceleration magnitude). Statistical test: paired Wilcoxon signed-rank (one-sided). Standard deviations are across 200 trajectories. Geometry constraints are not used during training, inference, or evaluation."
    )

    doc.add_heading("4. Results", level=1)
    doc.add_heading("4.1 Gaussian-medium deep comparison", level=2)
    anchor = doc.add_paragraph("Table 1. Gaussian-medium reconstruction metrics.")
    copy_table_after(anchor, source_table1)
    doc.add_paragraph(
        "Conditional residual DDPM achieves 0.0557m ADE, an absolute gain of 6.9 mm (11.0% relative) over noisy_input (p = 5.3×10⁻¹¹), and outperforms unconditional SDEdit by 5.1 mm (p = 6.7×10⁻⁸). The conditional model has larger trajectory-level spread (std=0.0162 vs 0.0084), reflecting larger per-trajectory corrections rather than instability (improved fraction 73.0%). Kalman worsens ADE in this protocol due to mismatch between the constant-velocity assumption and short indoor trajectories."
    )

    doc.add_heading("4.2 Generalization summary", level=2)
    add_simple_table(
        doc,
        ["Degradation", "noisy ADE", "cond ADE", "ΔADE", "p", "Outcome"],
        build_main_summary_rows(df),
    )
    doc.add_paragraph(
        "The model generalizes to jump but fails on drift and burst. Drift and burst failures are training-related; bias and combined failures are representational. Full per-method statistics are in Appendix B."
    )

    doc.add_heading("4.3 Conditioning effect", level=2)
    doc.add_paragraph(
        "The 4× gap between unconditional (2.8%, 1.8 mm) and conditional (11.0%, 6.9 mm) under identical architecture indicates the inference interface, not the prior, is the primary performance lever."
    )

    doc.add_heading("5. Figures", level=1)
    add_figure_block(
        doc,
        FIG_GAUSSIAN,
        "Figure 1. Gaussian-medium refinement examples (clean / degraded / uncond / cond).",
    )
    add_figure_block(
        doc,
        FIG_GENERALIZATION,
        "Figure 2. Generalization diagnostic: per-degradation ADE, ΔADE heatmap, and example trajectories (burst showing smoothing without accuracy gain; bias showing preserved offset).",
    )

    doc.add_heading("6. Discussion", level=1)
    doc.add_paragraph(
        "What worked. Conditional residual DDPM significantly improves Gaussian (p = 5.3×10⁻¹¹) and jump (p = 2.4×10⁻¹⁴) refinement. H1 is supported for relative-observable corruption."
    )
    doc.add_paragraph(
        "What failed. Drift over-corrects because noisy ADE is already small (0.0176m); the model trained on 5cm Gaussian noise applies similar-magnitude corrections to a near-clean signal. Burst fails because Gaussian-only training does not recognize local strong interference and uniformly smooths both burst and clean frames. Bias fails because constant offset cancels in relative displacement (visible only in absolute coordinates) — the model literally cannot see it."
    )
    doc.add_paragraph(
        "What it means. H2 is fully supported: the bound on DDPM refinement is representational, not capacity-related. The 4× gap between unconditional and conditional shows the inference interface dominates over prior quality."
    )

    doc.add_heading("7. Limitations and Next Step", level=1)
    doc.add_heading("7.1 Limitations", level=2)
    add_bullets(
        doc,
        [
            "Simulated data, not real sensor measurements",
            "Single training seed (42)",
            "Kalman parameters not optimized",
            "Gaussian-only training for the conditional model",
            "Evaluation on N=200 trajectory subset",
        ],
    )
    doc.add_heading("7.2 Next hypotheses", level=2)
    doc.add_paragraph(
        "H3 (method fix): Mixed relative-degradation training (gaussian + drift + jump + burst) should improve drift and burst while preserving Gaussian and jump gains."
    )
    doc.add_paragraph(
        "H4 (interface fix): An auxiliary head predicting absolute start-point correction (clean_start − degraded_start) should enable bias correction without changing the core DDPM."
    )

    doc.add_heading("Appendix A: Training details", level=1)
    add_bullets(
        doc,
        [
            "Adam, lr=1e-3, no scheduler",
            "60 epochs, batch 256",
            "EMA decay 0.999",
            "Degradation seed: 100000 × epoch + index",
            "Validation seed: 42 + index",
            "Best checkpoint by EMA val_loss",
        ],
    )

    doc.add_heading("Appendix B: Full Method × Degradation Matrix", level=1)
    add_simple_table(
        doc,
        ["Degradation", "Method", "ADE", "std", "Smooth", "ΔADE", "Imp%", "p val", "Result"],
        build_appendix_rows(df),
    )

    new_word_count = word_count_doc(doc)
    doc.save(str(OUTPUT_DOC))
    print(f"Saved: {OUTPUT_DOC}")
    print("=== 精简版报告生成完成 ===")
    print("输入: stage3_report_complete.docx")
    print("输出: stage3_report_v3_concise.docx")
    print("")
    print("结构: 7 sections + 2 appendices")
    print("图片: 全部 6.5 inches 宽度")
    print("Table 2: 正文精简版 (6×6) + Appendix 完整版 (18×9)")
    print("")
    print("文字长度变化:")
    print(f"  原: ~{old_word_count} words")
    print(f"  新: ~{new_word_count} words")
    print(f"  压缩: {(1 - new_word_count / old_word_count) * 100:.1f}%")


if __name__ == "__main__":
    main()
