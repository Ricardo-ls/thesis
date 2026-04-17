from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from matplotlib.ticker import FormatStrFormatter, MultipleLocator
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[3]
TRAIN_ROOT = PROJECT_ROOT / "outputs" / "prior" / "train"
EVAL_ROOT = PROJECT_ROOT / "outputs" / "prior" / "eval"
SAMPLE_ROOT = PROJECT_ROOT / "outputs" / "prior" / "sample"
REPORT_ROOT = PROJECT_ROOT / "thesis" / "report"

VARIANTS = ["none", "q10", "q20", "q30"]
SEEDS = [2, 3, 4, 12, 13, 14, 22, 23, 24, 32, 33, 34, 42, 43, 44]
EPOCHS = 100
RUN_TAG_SUFFIX = f"{EPOCHS}epoch"
REFERENCE_TAG = "reference_seed42"
SAMPLE_SEED = 42
VIS_SEED = 42

VARIANT_COLORS = {
    "none": "#1F77FF",
    "q10": "#FF7F0E",
    "q20": "#00B050",
    "q30": "#D62728",
}
SEED_BACKGROUND_COLOR = "#BFBFBF"
REAL_COLOR = "#4A4A4A"
STD_ALPHA = 0.30

TITLE_SIZE = 16
SUBPLOT_TITLE_SIZE = 13
AXIS_LABEL_SIZE = 12
TICK_LABEL_SIZE = 10
LEGEND_SIZE = 10
TABLE_TEXT_SIZE = 10

MAIN_MEAN_LINEWIDTH = 2.8
SECONDARY_MEAN_LINEWIDTH = 2.4
FAINT_SEED_LINEWIDTH = 0.9
BOXPLOT_EDGE_LINEWIDTH = 1.8

TABLE_HEADER_FILL = "#EEEEEE"
GRID_COLOR = "#AFAFAF"
GRID_ALPHA = 0.18
GRID_LINESTYLE = "--"

MOTION_METRICS = [
    "step_norm_all_ratio",
    "endpoint_displacement_ratio",
    "moving_ratio_global_ratio",
    "acc_rms_ratio",
]

MOTION_METRIC_LABELS = {
    "step_norm_all_ratio": "step_norm_all_ratio",
    "endpoint_displacement_ratio": "endpoint_displacement_ratio",
    "moving_ratio_global_ratio": "moving_ratio_global_ratio",
    "acc_rms_ratio": "acc_rms_ratio",
}

COMBINED_PDF_NAME = "stage2_formal_results_report_combined.pdf"


@dataclass
class RunRecord:
    variant: str
    seed: int
    run_tag: str
    train_dir: Path
    eval_dir: Path
    sample_dir: Path
    loss_history: pd.DataFrame
    summary_df: pd.DataFrame
    eval_manifest: dict
    sample_manifest: dict
    best_val_loss: float
    best_epoch: int
    ratios: dict[str, float]


def configure_matplotlib() -> None:
    matplotlib.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "DejaVu Sans"],
            "font.size": TICK_LABEL_SIZE,
            "axes.titlesize": SUBPLOT_TITLE_SIZE,
            "axes.labelsize": AXIS_LABEL_SIZE,
            "xtick.labelsize": TICK_LABEL_SIZE,
            "ytick.labelsize": TICK_LABEL_SIZE,
            "legend.fontsize": LEGEND_SIZE,
            "figure.titlesize": TITLE_SIZE,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def ensure_output_dir() -> None:
    REPORT_ROOT.mkdir(parents=True, exist_ok=True)


def run_tag(seed: int) -> str:
    return f"seed{seed}-{RUN_TAG_SUFFIX}"


def variant_train_dir(variant: str, seed: int) -> Path:
    return TRAIN_ROOT / f"ddpm_eth_ucy_{variant}_h128" / run_tag(seed)


def variant_eval_dir(variant: str, seed: int) -> Path:
    return EVAL_ROOT / f"ddpm_eth_ucy_{variant}_h128" / run_tag(seed) / REFERENCE_TAG


def variant_sample_dir(variant: str, seed: int) -> Path:
    return SAMPLE_ROOT / f"ddpm_eth_ucy_{variant}_h128" / run_tag(seed) / REFERENCE_TAG


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_loss_history(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    expected = {"epoch", "train_loss", "val_loss"}
    missing = expected.difference(df.columns)
    if missing:
        raise ValueError(f"Missing columns {sorted(missing)} in {path}")
    return df.sort_values("epoch").reset_index(drop=True)


def load_summary_metrics(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    expected = {"metric", "mean_ratio_gen_over_real"}
    missing = expected.difference(df.columns)
    if missing:
        raise ValueError(f"Missing columns {sorted(missing)} in {path}")
    return df


def ratio_from_summary(summary_df: pd.DataFrame, metric_name: str) -> float:
    matched = summary_df.loc[summary_df["metric"] == metric_name, "mean_ratio_gen_over_real"]
    if matched.empty:
        raise KeyError(f"Metric {metric_name} not found")
    return float(matched.iloc[0])


def parse_record(variant: str, seed: int) -> RunRecord:
    train_dir = variant_train_dir(variant, seed)
    eval_dir = variant_eval_dir(variant, seed)
    sample_dir = variant_sample_dir(variant, seed)

    loss_path = train_dir / "loss_history.csv"
    summary_path = eval_dir / "summary_metrics.csv"
    eval_manifest_path = eval_dir / "manifest.json"
    sample_manifest_path = sample_dir / "manifest.json"

    for path in [loss_path, summary_path, eval_manifest_path, sample_manifest_path]:
        if not path.exists():
            raise FileNotFoundError(f"Required file missing: {path}")

    loss_history = load_loss_history(loss_path)
    summary_df = load_summary_metrics(summary_path)
    best_idx = int(loss_history["val_loss"].astype(float).idxmin())
    best_val_loss = float(loss_history.loc[best_idx, "val_loss"])
    best_epoch = int(loss_history.loc[best_idx, "epoch"])
    ratios = {
        "step_norm_all_ratio": ratio_from_summary(summary_df, "step_norm_all"),
        "endpoint_displacement_ratio": ratio_from_summary(summary_df, "endpoint_displacement"),
        "moving_ratio_global_ratio": ratio_from_summary(summary_df, "moving_ratio_global"),
        "acc_rms_ratio": ratio_from_summary(summary_df, "acc_rms"),
    }

    return RunRecord(
        variant=variant,
        seed=seed,
        run_tag=run_tag(seed),
        train_dir=train_dir,
        eval_dir=eval_dir,
        sample_dir=sample_dir,
        loss_history=loss_history,
        summary_df=summary_df,
        eval_manifest=load_json(eval_manifest_path),
        sample_manifest=load_json(sample_manifest_path),
        best_val_loss=best_val_loss,
        best_epoch=best_epoch,
        ratios=ratios,
    )


def collect_records() -> list[RunRecord]:
    records = []
    for variant in VARIANTS:
        for seed in SEEDS:
            records.append(parse_record(variant, seed))
    return records


def build_tables(records: list[RunRecord]) -> tuple[pd.DataFrame, pd.DataFrame]:
    per_run_rows = []
    for record in records:
        row = {
            "variant": record.variant,
            "seed": record.seed,
            "best_val_loss": record.best_val_loss,
            "best_epoch": record.best_epoch,
        }
        row.update(record.ratios)
        per_run_rows.append(row)
    per_run_df = pd.DataFrame(per_run_rows)

    per_variant_rows = []
    for variant in VARIANTS:
        subset = per_run_df.loc[per_run_df["variant"] == variant].copy()
        per_variant_rows.append(
            {
                "variant": variant,
                "mean_best_val_loss": subset["best_val_loss"].mean(),
                "std_best_val_loss": subset["best_val_loss"].std(ddof=0),
                "min_best_val_loss": subset["best_val_loss"].min(),
                "max_best_val_loss": subset["best_val_loss"].max(),
                "mean_best_epoch": subset["best_epoch"].mean(),
                "std_best_epoch": subset["best_epoch"].std(ddof=0),
                "step_norm_all_ratio": subset["step_norm_all_ratio"].mean(),
                "endpoint_displacement_ratio": subset["endpoint_displacement_ratio"].mean(),
                "moving_ratio_global_ratio": subset["moving_ratio_global_ratio"].mean(),
                "acc_rms_ratio": subset["acc_rms_ratio"].mean(),
            }
        )
    per_variant_df = pd.DataFrame(per_variant_rows)
    return per_run_df, per_variant_df


def style_axes(ax: plt.Axes) -> None:
    ax.grid(False)
    ax.yaxis.grid(True, linestyle=GRID_LINESTYLE, linewidth=0.8, color=GRID_COLOR, alpha=GRID_ALPHA)
    ax.xaxis.grid(False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def save_figure(fig: plt.Figure, stem: str, pdf_pages: PdfPages) -> None:
    png_path = REPORT_ROOT / f"{stem}.png"
    pdf_path = REPORT_ROOT / f"{stem}.pdf"
    fig.savefig(png_path, dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(pdf_path, dpi=300, bbox_inches="tight", facecolor="white")
    pdf_pages.savefig(fig, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_figure1(records: list[RunRecord], per_variant_df: pd.DataFrame) -> plt.Figure:
    fig, axes = plt.subplots(2, 2, figsize=(13, 10), sharex=True, sharey=True)
    axes = axes.flatten()

    cropped_histories = []
    for record in records:
        cropped = record.loss_history.loc[record.loss_history["epoch"] >= 10].copy()
        cropped_histories.append(cropped)

    all_val_losses = np.concatenate([cropped["val_loss"].to_numpy(dtype=float) for cropped in cropped_histories])
    y_min = float(np.min(all_val_losses))
    y_max = float(np.max(all_val_losses))
    y_pad = max((y_max - y_min) * 0.08, 0.0010)

    for ax, variant in zip(axes, VARIANTS):
        subset = [record for record in records if record.variant == variant]
        cropped_subset = [record.loss_history.loc[record.loss_history["epoch"] >= 10].copy() for record in subset]
        epochs = cropped_subset[0]["epoch"].to_numpy(dtype=float)
        val_matrix = np.stack([cropped["val_loss"].to_numpy(dtype=float) for cropped in cropped_subset], axis=0)
        mean_curve = val_matrix.mean(axis=0)
        std_curve = val_matrix.std(axis=0, ddof=0)
        mean_best_epoch = float(per_variant_df.loc[per_variant_df["variant"] == variant, "mean_best_epoch"].iloc[0])
        color = VARIANT_COLORS[variant]

        for cropped in cropped_subset:
            ax.plot(
                cropped["epoch"],
                cropped["val_loss"],
                color=SEED_BACKGROUND_COLOR,
                linewidth=FAINT_SEED_LINEWIDTH,
                alpha=0.32,
                linestyle=(0, (2.2, 2.2)),
                zorder=1,
            )

        ax.fill_between(
            epochs,
            mean_curve - std_curve,
            mean_curve + std_curve,
            color=color,
            alpha=STD_ALPHA,
            zorder=2,
        )
        ax.plot(epochs, mean_curve - std_curve, color=color, linewidth=1.0, alpha=0.45, zorder=2)
        ax.plot(epochs, mean_curve + std_curve, color=color, linewidth=1.0, alpha=0.45, zorder=2)
        ax.plot(epochs, mean_curve, color=color, linewidth=MAIN_MEAN_LINEWIDTH, zorder=3)
        ax.axvline(
            mean_best_epoch,
            color=color,
            linewidth=2.0,
            linestyle=(0, (4, 3)),
            alpha=0.95,
            zorder=4,
        )
        ax.set_title(variant)
        ax.set_xlim(10, EPOCHS)
        ax.set_ylim(y_min - y_pad, y_max + y_pad)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Validation loss")
        ax.yaxis.set_major_locator(MultipleLocator(0.0025))
        ax.yaxis.set_major_formatter(FormatStrFormatter("%.4f"))
        ax.text(
            0.98,
            0.96,
            "seeds = 15\nepochs = 100",
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=9,
            color="#555555",
        )
        style_axes(ax)
        ax.yaxis.grid(True, linestyle=GRID_LINESTYLE, linewidth=0.9, color=GRID_COLOR, alpha=0.28)

    fig.suptitle("Validation loss curves by variant", y=0.98)
    fig.tight_layout(rect=[0.03, 0.03, 0.99, 0.96])
    return fig


def plot_figure2(per_run_df: pd.DataFrame, per_variant_df: pd.DataFrame) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(12, 8))
    positions = np.arange(1, len(VARIANTS) + 1)
    rng = np.random.default_rng(42)
    box_data = [per_run_df.loc[per_run_df["variant"] == variant, "best_val_loss"].to_numpy(dtype=float) for variant in VARIANTS]

    bp = ax.boxplot(
        box_data,
        positions=positions,
        widths=0.55,
        patch_artist=True,
        showfliers=False,
        medianprops={"color": "#222222", "linewidth": 1.6},
        whiskerprops={"color": "#555555", "linewidth": 1.3},
        capprops={"color": "#555555", "linewidth": 1.3},
    )

    for patch, variant in zip(bp["boxes"], VARIANTS):
        color = VARIANT_COLORS[variant]
        patch.set(facecolor="white", edgecolor=color, linewidth=BOXPLOT_EDGE_LINEWIDTH)

    for pos, variant in zip(positions, VARIANTS):
        values = per_run_df.loc[per_run_df["variant"] == variant, "best_val_loss"].to_numpy(dtype=float)
        jitter = rng.uniform(-0.11, 0.11, size=len(values))
        ax.scatter(
            np.full_like(values, pos, dtype=float) + jitter,
            values,
            s=36,
            color=VARIANT_COLORS[variant],
            alpha=0.75,
            edgecolors="white",
            linewidths=0.4,
            zorder=3,
        )
        row = per_variant_df.loc[per_variant_df["variant"] == variant].iloc[0]
        ax.errorbar(
            pos,
            row["mean_best_val_loss"],
            yerr=row["std_best_val_loss"],
            color="black",
            linewidth=1.2,
            capsize=4,
            zorder=4,
        )
        ax.scatter(
            [pos],
            [row["mean_best_val_loss"]],
            marker="D",
            s=58,
            color="black",
            zorder=5,
        )

    ax.set_xticks(positions, VARIANTS)
    ax.set_xlabel("Variant")
    ax.set_ylabel("Best validation loss")
    ax.set_title("Best validation loss summary")
    ax.text(0.5, 1.02, "n = 15 seeds per variant", transform=ax.transAxes, ha="center", va="bottom", fontsize=10, color="#555555")
    style_axes(ax)
    fig.tight_layout()
    return fig


def draw_table_figure(
    dataframe: pd.DataFrame,
    title: str,
    footnote: str | None = None,
    figsize: tuple[float, float] = (12, 3.8),
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=figsize)
    ax.axis("off")
    ax.set_title(title, fontsize=TITLE_SIZE, pad=14)

    display_df = dataframe.copy()
    col_widths = [0.16] + [(0.84 / max(len(display_df.columns) - 1, 1))] * (len(display_df.columns) - 1)

    table = ax.table(
        cellText=display_df.values,
        colLabels=display_df.columns,
        loc="center",
        cellLoc="center",
        colLoc="center",
        colWidths=col_widths,
    )
    table.auto_set_font_size(False)
    table.set_fontsize(TABLE_TEXT_SIZE)
    table.scale(1, 1.6)

    for (row, col), cell in table.get_celld().items():
        cell.set_linewidth(0.8)
        cell.set_edgecolor("#C8C8C8")
        if row == 0:
            cell.set_facecolor(TABLE_HEADER_FILL)
            cell.set_text_props(weight="bold")
        else:
            cell.set_facecolor("white")
        if col == 0 and row > 0:
            variant_name = str(display_df.iloc[row - 1, 0])
            cell.get_text().set_color(VARIANT_COLORS[variant_name])
            cell.get_text().set_weight("bold")

    if footnote:
        fig.text(0.5, 0.05, footnote, ha="center", va="center", fontsize=10, color="#555555")

    fig.tight_layout(rect=[0.02, 0.08 if footnote else 0.02, 0.98, 0.95])
    return fig


def plot_figure3(per_variant_df: pd.DataFrame) -> plt.Figure:
    heatmap_values = per_variant_df[MOTION_METRICS].to_numpy(dtype=float)
    vmin = float(np.min(heatmap_values))
    vmax = float(np.max(heatmap_values))
    span = max(abs(vmin - 1.0), abs(vmax - 1.0))
    norm = TwoSlopeNorm(vmin=1.0 - span, vcenter=1.0, vmax=1.0 + span)
    cmap = LinearSegmentedColormap.from_list("ratio_balance", ["#6FA8DC", "#F7F7F7", "#F4A261"])

    fig, ax = plt.subplots(figsize=(12, 8))
    im = ax.imshow(heatmap_values, cmap=cmap, norm=norm, aspect="auto")
    ax.set_xticks(np.arange(len(MOTION_METRICS)), [MOTION_METRIC_LABELS[m] for m in MOTION_METRICS])
    ax.set_yticks(np.arange(len(VARIANTS)), VARIANTS)
    ax.set_title("Motion realism heatmap")
    ax.tick_params(axis="x", rotation=15)

    for row in range(len(VARIANTS)):
        for col in range(len(MOTION_METRICS)):
            value = heatmap_values[row, col]
            ax.text(col, row, f"{value:.3f}", ha="center", va="center", fontsize=10, color="#222222")

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Generated / real ratio")
    cbar.set_ticks([1.0 - span, 1.0, 1.0 + span])
    cbar.set_ticklabels([f"{1.0 - span:.2f}", "1.0 (reference)", f"{1.0 + span:.2f}"])

    for spine in ax.spines.values():
        spine.set_visible(False)

    fig.text(0.5, 0.04, "Ratios are generated / real, reference value = 1.0", ha="center", fontsize=10, color="#555555")
    fig.tight_layout(rect=[0.03, 0.07, 0.95, 0.97])
    return fig


def compute_step_norms(rel: np.ndarray) -> np.ndarray:
    return np.linalg.norm(rel, axis=-1)


def compute_global_moving_threshold(real_rel: np.ndarray, quantile: float = 0.10, positive_eps: float = 1e-8) -> float:
    real_step_norm = compute_step_norms(real_rel).reshape(-1)
    positive = real_step_norm[real_step_norm > positive_eps]
    if positive.size == 0:
        raise ValueError("Real data has no positive step norms")
    return float(np.quantile(positive, quantile))


def compute_distribution_metrics(rel: np.ndarray, moving_threshold: float, eps: float = 1e-8) -> dict[str, np.ndarray]:
    step_norm = compute_step_norms(rel)
    total_length = step_norm.sum(axis=1)
    endpoint_vec = rel.sum(axis=1)
    endpoint_displacement = np.linalg.norm(endpoint_vec, axis=1)
    moving_ratio_global = (step_norm > moving_threshold).mean(axis=1)
    if rel.shape[1] >= 2:
        acc = rel[:, 1:, :] - rel[:, :-1, :]
        acc_norm = np.linalg.norm(acc, axis=-1)
        acc_rms = np.sqrt(np.mean(acc_norm ** 2, axis=1))
    else:
        acc_rms = np.zeros(rel.shape[0], dtype=np.float32)
    return {
        "step_norm_all": step_norm.reshape(-1),
        "endpoint_displacement": endpoint_displacement,
        "moving_ratio_global": moving_ratio_global,
        "acc_rms": acc_rms,
        "total_length": total_length + eps,
    }


def load_real_and_generated_arrays(records: list[RunRecord]) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    real_path = Path(records[0].eval_manifest["real_data_path"])
    real_rel = np.load(real_path).astype(np.float32)

    generated_by_variant: dict[str, list[np.ndarray]] = {variant: [] for variant in VARIANTS}
    for record in records:
        generated_path = Path(record.sample_manifest["generated_rel_path"])
        gen = np.load(generated_path).astype(np.float32)
        if gen.ndim != 3:
            raise ValueError(f"Unexpected generated array shape: {generated_path} -> {gen.shape}")
        if gen.shape[1:] == (2, 19):
            gen = np.transpose(gen, (0, 2, 1))
        if gen.shape[1:] != (19, 2):
            raise ValueError(f"Unexpected generated array layout: {generated_path} -> {gen.shape}")
        generated_by_variant[record.variant].append(gen)

    pooled_generated = {variant: np.concatenate(chunks, axis=0) for variant, chunks in generated_by_variant.items()}
    return real_rel, pooled_generated


def histogram_limits(arrays: list[np.ndarray], pad_ratio: float = 0.03) -> tuple[float, float]:
    merged = np.concatenate(arrays).astype(float)
    low = float(np.min(merged))
    high = float(np.max(merged))
    if math.isclose(low, high):
        return low - 0.5, high + 0.5
    pad = (high - low) * pad_ratio
    return low - pad, high + pad


def plot_overlay_histogram(
    ax: plt.Axes,
    real_values: np.ndarray,
    gen_values: np.ndarray,
    gen_label: str,
    gen_color: str,
    title: str,
    xlabel: str,
    bins: int,
    xlim: tuple[float, float],
) -> None:
    ax.hist(
        real_values,
        bins=bins,
        density=True,
        color=REAL_COLOR,
        alpha=0.35,
        label="real",
        range=xlim,
    )
    ax.hist(
        gen_values,
        bins=bins,
        density=True,
        color=gen_color,
        alpha=0.35,
        label=gen_label,
        range=xlim,
    )
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Density")
    ax.set_xlim(*xlim)
    style_axes(ax)


def plot_figure4(records: list[RunRecord]) -> plt.Figure:
    real_rel, pooled_generated = load_real_and_generated_arrays(records)
    moving_threshold = compute_global_moving_threshold(real_rel)
    real_metrics = compute_distribution_metrics(real_rel, moving_threshold)
    none_metrics = compute_distribution_metrics(pooled_generated["none"], moving_threshold)
    q20_metrics = compute_distribution_metrics(pooled_generated["q20"], moving_threshold)

    endpoint_xlim = histogram_limits(
        [
            real_metrics["endpoint_displacement"],
            none_metrics["endpoint_displacement"],
            q20_metrics["endpoint_displacement"],
        ]
    )
    acc_xlim = histogram_limits([real_metrics["acc_rms"], none_metrics["acc_rms"], q20_metrics["acc_rms"]])

    fig, axes = plt.subplots(2, 2, figsize=(13, 10), sharey="row")

    plot_overlay_histogram(
        axes[0, 0],
        real_metrics["endpoint_displacement"],
        none_metrics["endpoint_displacement"],
        "none",
        VARIANT_COLORS["none"],
        "Endpoint displacement: real vs none",
        "Endpoint displacement",
        bins=50,
        xlim=endpoint_xlim,
    )
    plot_overlay_histogram(
        axes[0, 1],
        real_metrics["endpoint_displacement"],
        q20_metrics["endpoint_displacement"],
        "q20",
        VARIANT_COLORS["q20"],
        "Endpoint displacement: real vs q20",
        "Endpoint displacement",
        bins=50,
        xlim=endpoint_xlim,
    )
    plot_overlay_histogram(
        axes[1, 0],
        real_metrics["acc_rms"],
        none_metrics["acc_rms"],
        "none",
        VARIANT_COLORS["none"],
        "Acc RMS: real vs none",
        "Acc RMS",
        bins=50,
        xlim=acc_xlim,
    )
    plot_overlay_histogram(
        axes[1, 1],
        real_metrics["acc_rms"],
        q20_metrics["acc_rms"],
        "q20",
        VARIANT_COLORS["q20"],
        "Acc RMS: real vs q20",
        "Acc RMS",
        bins=50,
        xlim=acc_xlim,
    )

    for ax in axes.flat:
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles, labels, frameon=False, loc="upper right")

    fig.tight_layout()
    return fig


def format_float_columns(df: pd.DataFrame, digits: int) -> pd.DataFrame:
    formatted = df.copy()
    for col in formatted.columns:
        if col == "variant":
            continue
        formatted[col] = formatted[col].map(lambda v: f"{float(v):.{digits}f}")
    return formatted


def create_manifest(records: list[RunRecord]) -> str:
    lines = [
        "Stage 2 formal results report manifest",
        "",
        "Files:",
        "fig1_val_loss_2x2_multiseed.png",
        "fig1_val_loss_2x2_multiseed.pdf",
        "fig2_best_val_loss_summary.png",
        "fig2_best_val_loss_summary.pdf",
        "table1_optimization_summary.png",
        "table1_optimization_summary.pdf",
        "fig3_motion_realism_heatmap.png",
        "fig3_motion_realism_heatmap.pdf",
        "fig4_pairwise_metric_distributions_none_q20.png",
        "fig4_pairwise_metric_distributions_none_q20.pdf",
        "table2_motion_metric_summary.png",
        "table2_motion_metric_summary.pdf",
        COMBINED_PDF_NAME,
        "manifest.txt",
        "short_notes.md",
        "",
        "Data source paths:",
        f"train_root: {TRAIN_ROOT}",
        f"eval_root: {EVAL_ROOT}",
        f"sample_root: {SAMPLE_ROOT}",
        f"real_data_path: {records[0].eval_manifest['real_data_path']}",
        "",
        "Protocol:",
        f"variants: {', '.join(VARIANTS)}",
        f"seeds: {', '.join(str(seed) for seed in SEEDS)}",
        f"epochs: {EPOCHS}",
        "checkpoint: best_model.pt",
        f"sample_seed: {SAMPLE_SEED}",
        f"vis_seed: {VIS_SEED}",
    ]
    return "\n".join(lines) + "\n"


def create_short_notes() -> str:
    return "\n".join(
        [
            "# Short notes",
            "",
            "- `Figure 1` shows the 15-seed validation-loss trajectories for each variant with the variant mean curve, variability band, and mean best epoch.",
            "- `Figure 2` shows the distribution of best validation loss across 15 seeds for the four variants.",
            "- `Table 1` lists the core optimization statistics for best validation loss and best epoch under the shared 100-epoch screening protocol.",
            "- `Figure 3` shows the four retained motion realism ratios as a variant-by-metric heatmap centered at the 1.0 reference line.",
            "- `Figure 4` shows pooled distribution comparisons for endpoint displacement and acc_rms between real data and the none / q20 variants.",
            "- `Table 2` lists the four retained motion ratios for the four variants under the shared evaluation protocol.",
        ]
    ) + "\n"


def main() -> None:
    configure_matplotlib()
    ensure_output_dir()

    records = collect_records()
    per_run_df, per_variant_df = build_tables(records)

    table1_df = per_variant_df[
        [
            "variant",
            "mean_best_val_loss",
            "std_best_val_loss",
            "min_best_val_loss",
            "max_best_val_loss",
            "mean_best_epoch",
            "std_best_epoch",
        ]
    ]
    table2_df = per_variant_df[["variant"] + MOTION_METRICS]

    with PdfPages(REPORT_ROOT / COMBINED_PDF_NAME) as combined_pdf:
        fig1 = plot_figure1(records, per_variant_df)
        save_figure(fig1, "fig1_val_loss_2x2_multiseed", combined_pdf)

        fig2 = plot_figure2(per_run_df, per_variant_df)
        save_figure(fig2, "fig2_best_val_loss_summary", combined_pdf)

        table1_fig = draw_table_figure(
            format_float_columns(table1_df, 4),
            "Optimization summary",
            figsize=(12, 3.8),
        )
        save_figure(table1_fig, "table1_optimization_summary", combined_pdf)

        fig3 = plot_figure3(per_variant_df)
        save_figure(fig3, "fig3_motion_realism_heatmap", combined_pdf)

        fig4 = plot_figure4(records)
        save_figure(fig4, "fig4_pairwise_metric_distributions_none_q20", combined_pdf)

        table2_fig = draw_table_figure(
            format_float_columns(table2_df, 3),
            "Motion metric summary table",
            footnote="Reference value for all ratios is 1.0",
            figsize=(12, 3.8),
        )
        save_figure(table2_fig, "table2_motion_metric_summary", combined_pdf)

    (REPORT_ROOT / "manifest.txt").write_text(create_manifest(records), encoding="utf-8")
    (REPORT_ROOT / "short_notes.md").write_text(create_short_notes(), encoding="utf-8")


if __name__ == "__main__":
    main()
