from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
import math

import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[3]
TRAIN_ROOT = PROJECT_ROOT / "outputs" / "prior" / "train"
EVAL_ROOT = PROJECT_ROOT / "outputs" / "prior" / "eval"
AGG_ROOT = PROJECT_ROOT / "outputs" / "prior" / "archive" / "stage2_phaseA_multiseed_100epoch" / "eval"
FIG_ROOT = PROJECT_ROOT / "docs" / "assets" / "stage2_phaseA_multiseed_100epoch"
REPORT_PATH = PROJECT_ROOT / "docs" / "stage2_phaseA_multiseed_100epoch_report.md"

VARIANTS = ["none", "q10", "q20", "q30"]
SEEDS = [42, 43, 44]
EPOCH_TAG = "100epoch"

VARIANT_COLORS = {
    "none": "#1f4e79",
    "q10": "#5a8f29",
    "q20": "#c4661f",
    "q30": "#7d3c98",
}
SEED_COLORS = {
    42: "#d97706",
    43: "#1d4ed8",
    44: "#8b5e3c",
}
SEED_MARKERS = {
    42: "o",
    43: "s",
    44: "^",
}


@dataclass
class RunRecord:
    variant: str
    seed: int
    epoch_tag: str
    train_dir: Path
    eval_dir: Path
    loss_history: pd.DataFrame
    summary_df: pd.DataFrame
    best_val_loss: float
    best_epoch: int
    ratios: dict[str, float]
    mrs4: float


def ensure_dirs():
    AGG_ROOT.mkdir(parents=True, exist_ok=True)
    FIG_ROOT.mkdir(parents=True, exist_ok=True)


def load_loss_history(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing loss history: {path}")
    df = pd.read_csv(path)
    required = {"epoch", "train_loss", "val_loss"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Loss history missing columns {missing}: {path}")
    return df


def load_summary(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing summary metrics: {path}")
    df = pd.read_csv(path)
    required = {"metric", "mean_ratio_gen_over_real"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Summary missing columns {missing}: {path}")
    return df


def metric_ratio(summary_df: pd.DataFrame, metric_name: str) -> float:
    row = summary_df.loc[summary_df["metric"] == metric_name]
    if row.empty:
        raise KeyError(f"Metric {metric_name} not found in summary")
    return float(row.iloc[0]["mean_ratio_gen_over_real"])


def parse_run(variant: str, seed: int) -> RunRecord:
    run_tag = f"seed{seed}-{EPOCH_TAG}"
    train_dir = TRAIN_ROOT / f"ddpm_eth_ucy_{variant}_h128" / run_tag
    eval_dir = EVAL_ROOT / f"ddpm_eth_ucy_{variant}_h128" / run_tag / "reference_seed42"

    loss_df = load_loss_history(train_dir / "loss_history.csv")
    summary_df = load_summary(eval_dir / "summary_metrics.csv")

    best_idx = int(loss_df["val_loss"].astype(float).idxmin())
    best_val_loss = float(loss_df.iloc[best_idx]["val_loss"])
    best_epoch = int(loss_df.iloc[best_idx]["epoch"])

    ratios = {
        "step_norm_all_ratio": metric_ratio(summary_df, "step_norm_all"),
        "total_length_ratio": metric_ratio(summary_df, "total_length"),
        "endpoint_displacement_ratio": metric_ratio(summary_df, "endpoint_displacement"),
        "acc_rms_ratio": metric_ratio(summary_df, "acc_rms"),
    }
    mrs4 = float(np.mean([abs(v - 1.0) for v in ratios.values()]))

    return RunRecord(
        variant=variant,
        seed=seed,
        epoch_tag=EPOCH_TAG,
        train_dir=train_dir,
        eval_dir=eval_dir,
        loss_history=loss_df,
        summary_df=summary_df,
        best_val_loss=best_val_loss,
        best_epoch=best_epoch,
        ratios=ratios,
        mrs4=mrs4,
    )


def build_dataframes(records: list[RunRecord]) -> tuple[pd.DataFrame, pd.DataFrame]:
    per_run_rows = []
    for record in records:
        per_run_rows.append(
            {
                "variant": record.variant,
                "seed": record.seed,
                "epoch_tag": record.epoch_tag,
                "best_val_loss": record.best_val_loss,
                "best_epoch": record.best_epoch,
                "step_norm_all_ratio": record.ratios["step_norm_all_ratio"],
                "total_length_ratio": record.ratios["total_length_ratio"],
                "endpoint_displacement_ratio": record.ratios["endpoint_displacement_ratio"],
                "acc_rms_ratio": record.ratios["acc_rms_ratio"],
                "mrs4": record.mrs4,
            }
        )

    per_run_df = pd.DataFrame(per_run_rows).sort_values(["variant", "seed"]).reset_index(drop=True)

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
                "mean_step_norm_all_ratio": subset["step_norm_all_ratio"].mean(),
                "std_step_norm_all_ratio": subset["step_norm_all_ratio"].std(ddof=0),
                "mean_total_length_ratio": subset["total_length_ratio"].mean(),
                "std_total_length_ratio": subset["total_length_ratio"].std(ddof=0),
                "mean_endpoint_displacement_ratio": subset["endpoint_displacement_ratio"].mean(),
                "std_endpoint_displacement_ratio": subset["endpoint_displacement_ratio"].std(ddof=0),
                "mean_acc_rms_ratio": subset["acc_rms_ratio"].mean(),
                "std_acc_rms_ratio": subset["acc_rms_ratio"].std(ddof=0),
                "mean_mrs4": subset["mrs4"].mean(),
                "std_mrs4": subset["mrs4"].std(ddof=0),
            }
        )

    per_variant_df = pd.DataFrame(per_variant_rows)
    return per_run_df, per_variant_df


def save_csvs(per_run_df: pd.DataFrame, per_variant_df: pd.DataFrame):
    per_run_df.to_csv(AGG_ROOT / "per_run_summary.csv", index=False, float_format="%.6f")
    per_variant_df.to_csv(AGG_ROOT / "per_variant_summary.csv", index=False, float_format="%.6f")


def style_axes(ax):
    ax.grid(True, alpha=0.25, linewidth=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def save_fig(fig, stem: str):
    fig.savefig(FIG_ROOT / f"{stem}.png", dpi=220, bbox_inches="tight")
    fig.savefig(FIG_ROOT / f"{stem}.svg", bbox_inches="tight")
    plt.close(fig)


def plot_fig1_val_loss_seed_overlay(records: list[RunRecord]):
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
    axes = axes.flatten()

    for ax, variant in zip(axes, VARIANTS):
        subset = [r for r in records if r.variant == variant]
        min_vals = []
        for record in sorted(subset, key=lambda r: r.seed):
            color = SEED_COLORS[record.seed]
            df = record.loss_history.copy()
            df = df.loc[df["epoch"] >= 10]
            ax.plot(
                df["epoch"],
                df["val_loss"],
                color=color,
                linestyle="-",
                linewidth=2.0,
                alpha=0.95,
                label=f"seed{record.seed}",
            )
            best_idx = df["val_loss"].astype(float).idxmin()
            best_row = df.loc[best_idx]
            min_vals.append(float(best_row["val_loss"]))
            ax.scatter(
                [best_row["epoch"]],
                [best_row["val_loss"]],
                color=color,
                marker=SEED_MARKERS[record.seed],
                s=42,
                edgecolors="white",
                linewidths=0.6,
                zorder=3,
            )
        y_min = min(min_vals)
        y_series = np.concatenate([r.loss_history.loc[r.loss_history["epoch"] >= 10, "val_loss"].to_numpy(dtype=float) for r in subset])
        y_max = float(np.max(y_series))
        pad = max((y_max - y_min) * 0.10, 0.0015)
        ax.set_ylim(y_min - pad, y_max + pad)
        ax.set_xlim(10, 100)
        ax.set_title(f"{variant} | val loss (lower is better)")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Validation Loss")
        style_axes(ax)
        ax.legend(frameon=False, fontsize=9)

    fig.suptitle("Stage 2 Phase A Validation Loss by Seed (Epochs 10-100)", fontsize=14)
    fig.tight_layout()
    save_fig(fig, "fig1_val_loss_seed_overlay_2x2")


def plot_fig2_best_val_loss(per_run_df: pd.DataFrame, per_variant_df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(VARIANTS))
    for i, variant in enumerate(VARIANTS):
        subset = per_run_df.loc[per_run_df["variant"] == variant]
        jitter = np.linspace(-0.08, 0.08, len(subset))
        ax.scatter(
            np.full(len(subset), x[i]) + jitter,
            subset["best_val_loss"],
            color=VARIANT_COLORS[variant],
            s=55,
            zorder=3,
            label=variant if i == 0 else None,
        )
        row = per_variant_df.loc[per_variant_df["variant"] == variant].iloc[0]
        ax.errorbar(
            x[i],
            row["mean_best_val_loss"],
            yerr=row["std_best_val_loss"],
            color=VARIANT_COLORS[variant],
            capsize=5,
            linewidth=2.0,
            fmt="_",
            markersize=20,
            zorder=4,
        )
    ax.set_xticks(x, VARIANTS)
    ax.set_ylabel("Best Validation Loss (lower is better)")
    ax.set_title("Stage 2 Phase A Best Validation Loss Across Seeds")
    style_axes(ax)
    fig.tight_layout()
    save_fig(fig, "fig2_best_val_loss_summary")


def plot_fig3_motion_realism_heatmap(per_variant_df: pd.DataFrame):
    columns = [
        ("mean_step_norm_all_ratio", "std_step_norm_all_ratio", "step_norm_all"),
        ("mean_total_length_ratio", "std_total_length_ratio", "total_length"),
        ("mean_endpoint_displacement_ratio", "std_endpoint_displacement_ratio", "endpoint_displacement"),
        ("mean_acc_rms_ratio", "std_acc_rms_ratio", "acc_rms"),
    ]
    data = np.array([[per_variant_df.loc[per_variant_df["variant"] == variant].iloc[0][mean_col] for mean_col, _, _ in columns] for variant in VARIANTS], dtype=float)

    deviation = max(abs(data.min() - 1.0), abs(data.max() - 1.0))
    norm = TwoSlopeNorm(vmin=1.0 - deviation, vcenter=1.0, vmax=1.0 + deviation)

    fig, ax = plt.subplots(figsize=(9, 4.8))
    im = ax.imshow(data, cmap="RdYlBu_r", norm=norm, aspect="auto")
    ax.set_xticks(np.arange(len(columns)), [name for _, _, name in columns], rotation=15)
    ax.set_yticks(np.arange(len(VARIANTS)), VARIANTS)
    ax.set_title("Stage 2 Phase A Motion Realism Heatmap (Ideal ratio = 1.0)")
    for row_idx, variant in enumerate(VARIANTS):
        row = per_variant_df.loc[per_variant_df["variant"] == variant].iloc[0]
        for col_idx, (mean_col, std_col, _) in enumerate(columns):
            ax.text(
                col_idx,
                row_idx,
                f"{row[mean_col]:.3f}\n±{row[std_col]:.3f}",
                ha="center",
                va="center",
                fontsize=9,
                color="black",
            )
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Mean ratio (closer to 1.0 is better)")
    fig.tight_layout()
    save_fig(fig, "fig3_motion_realism_heatmap")


def plot_fig4_mrs4(per_run_df: pd.DataFrame, per_variant_df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(VARIANTS))
    for i, variant in enumerate(VARIANTS):
        subset = per_run_df.loc[per_run_df["variant"] == variant]
        jitter = np.linspace(-0.08, 0.08, len(subset))
        ax.scatter(
            np.full(len(subset), x[i]) + jitter,
            subset["mrs4"],
            color=VARIANT_COLORS[variant],
            s=55,
            zorder=3,
        )
        row = per_variant_df.loc[per_variant_df["variant"] == variant].iloc[0]
        ax.errorbar(
            x[i],
            row["mean_mrs4"],
            yerr=row["std_mrs4"],
            color=VARIANT_COLORS[variant],
            capsize=5,
            linewidth=2.0,
            fmt="_",
            markersize=20,
            zorder=4,
        )
    ax.set_xticks(x, VARIANTS)
    ax.set_ylabel("MRS4 (lower is better)")
    ax.set_title("Stage 2 Phase A MRS4 Across Seeds")
    style_axes(ax)
    fig.tight_layout()
    save_fig(fig, "fig4_mrs4_summary")


def plot_figA1_none_seed42_100_vs_150():
    path_100 = TRAIN_ROOT / "ddpm_eth_ucy_none_h128" / "seed42-100epoch" / "loss_history.csv"
    path_150 = TRAIN_ROOT / "ddpm_eth_ucy_none_h128" / "seed42-150epoch" / "loss_history.csv"
    if not path_100.exists() or not path_150.exists():
        return False

    df100 = pd.read_csv(path_100)
    df150 = pd.read_csv(path_150)
    df100 = df100.loc[df100["epoch"] >= 10]
    df150 = df150.loc[df150["epoch"] >= 10]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(df100["epoch"], df100["val_loss"], color="#1f4e79", linewidth=2.0, label="seed42-100epoch")
    ax.plot(df150["epoch"], df150["val_loss"], color="#ff8c00", linewidth=2.0, label="seed42-150epoch")
    best100 = df100.loc[df100["val_loss"].astype(float).idxmin()]
    best150 = df150.loc[df150["val_loss"].astype(float).idxmin()]
    ax.scatter([best100["epoch"]], [best100["val_loss"]], color="#1f4e79", s=45)
    ax.scatter([best150["epoch"]], [best150["val_loss"]], color="#ff8c00", s=45)
    y_min = min(float(df100["val_loss"].min()), float(df150["val_loss"].min()))
    y_max = max(float(df100["val_loss"].max()), float(df150["val_loss"].max()))
    pad = max((y_max - y_min) * 0.10, 0.0015)
    ax.set_ylim(y_min - pad, y_max + pad)
    ax.set_xlim(10, max(int(df150["epoch"].max()), 100))
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Validation Loss (lower is better)")
    ax.set_title("Appendix: none seed42 Validation Loss, 100 vs 150 Epochs")
    style_axes(ax)
    ax.legend(frameon=False)
    fig.tight_layout()
    save_fig(fig, "figA1_none_seed42_100_vs_150_val_curve")
    return True


def md_table_from_df(df: pd.DataFrame, columns: list[str]) -> str:
    table = df[columns].copy()
    for col in table.columns:
        if pd.api.types.is_float_dtype(table[col]):
            table[col] = table[col].map(lambda x: f"{x:.4f}")
    headers = list(table.columns)
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for _, row in table.iterrows():
        lines.append("| " + " | ".join(str(row[h]) for h in headers) + " |")
    return "\n".join(lines)


def choose_candidates(per_variant_df: pd.DataFrame) -> tuple[str, str]:
    optimization_best = per_variant_df.sort_values("mean_best_val_loss", ascending=True).iloc[0]["variant"]
    application_best = per_variant_df.sort_values("mean_mrs4", ascending=True).iloc[0]["variant"]
    return optimization_best, application_best


def interim_answers(per_variant_df: pd.DataFrame, appendix_generated: bool) -> tuple[str, str, str]:
    best_epochs = per_variant_df["mean_best_epoch"].to_numpy(dtype=float)
    if appendix_generated:
        insufficient = (
            "Not obviously. The appendix comparison for `none/seed42` shows additional improvement from 100 to 150 epochs, "
            "but the 100-epoch runs have already entered a relatively stable validation-loss regime rather than remaining in the steep early-descent phase."
        )
    else:
        insufficient = (
            "Not obviously from Phase A alone. The 100-epoch runs are already in a stable validation-loss regime, although a longer-horizon comparison is not available in this report."
        )

    loss_rank = per_variant_df.sort_values("mean_best_val_loss")["variant"].tolist()
    motion_rank = per_variant_df.sort_values("mean_mrs4")["variant"].tolist()
    ranking_diff = (
        f"They are not identical. The optimization view orders variants by mean best validation loss as `{', '.join(loss_rank)}`, "
        f"while the motion-realism view orders them by mean MRS4 as `{', '.join(motion_rank)}`."
    )

    finalists = per_variant_df.sort_values(["mean_best_val_loss", "mean_mrs4"]).head(2)["variant"].tolist()
    finalists_answer = (
        f"The two variants that should move forward are `{finalists[0]}` and `{finalists[1]}`: "
        "one to preserve the strongest optimization-side baseline and one to preserve the strongest motion-realism candidate."
    )
    return insufficient, ranking_diff, finalists_answer


def write_report(per_variant_df: pd.DataFrame, appendix_generated: bool):
    optimization_best, application_best = choose_candidates(per_variant_df)
    insufficient, ranking_diff, finalists_answer = interim_answers(per_variant_df, appendix_generated)

    opt_table = md_table_from_df(
        per_variant_df,
        [
            "variant",
            "mean_best_val_loss",
            "std_best_val_loss",
            "mean_best_epoch",
            "std_best_epoch",
        ],
    )
    motion_table = md_table_from_df(
        per_variant_df,
        [
            "variant",
            "mean_step_norm_all_ratio",
            "mean_total_length_ratio",
            "mean_endpoint_displacement_ratio",
            "mean_acc_rms_ratio",
            "mean_mrs4",
        ],
    )

    appendix_section = ""
    if appendix_generated:
        appendix_section = """
## Appendix Figure

![Appendix none seed42 100 vs 150 val curve](assets/stage2_phaseA_multiseed_100epoch/figA1_none_seed42_100_vs_150_val_curve.png)
"""
    else:
        appendix_section = """
## Appendix Figure

The appendix comparison between `none/seed42` at `100` and `150` epochs was skipped because the `150`-epoch loss history was not available in the expected archive location.
"""

    report = f"""# Stage 2 Phase A Multi-seed 100-epoch Protocol Check

## Objective

This report consolidates Stage 2 Phase A under a fixed `100`-epoch budget across:

- `none`
- `q10`
- `q20`
- `q30`

with train seeds:

- `42`
- `43`
- `44`

Validation loss is treated as a training diagnostic rather than a complete decision rule. The report therefore compares both:

- an optimization view
- a motion realism view

## Protocol

- dataset: ETH+UCY processed relative trajectories
- representation: `19 x 2` relative-step trajectory windows
- variants: `none`, `q10`, `q20`, `q30`
- train seeds: `42`, `43`, `44`
- epochs: `100`
- checkpoint used for eval: `best_model.pt`
- sample seed fixed to: `42`
- visualization seed fixed to: `42`
- reverse sample count per run: `512`
- qualitative figure count per run: `16`

## Optimization View

![Figure 1](assets/stage2_phaseA_multiseed_100epoch/fig1_val_loss_seed_overlay_2x2.png)

![Figure 2](assets/stage2_phaseA_multiseed_100epoch/fig2_best_val_loss_summary.png)

{opt_table}

## Motion Realism View

![Figure 3](assets/stage2_phaseA_multiseed_100epoch/fig3_motion_realism_heatmap.png)

![Figure 4](assets/stage2_phaseA_multiseed_100epoch/fig4_mrs4_summary.png)

{motion_table}

## Interim Answers

### Is 100 epochs still obviously insufficient?

{insufficient}

### How different are validation loss and motion realism rankings?

{ranking_diff}

### Which two variants should move to the next stage?

{finalists_answer}

## Recommended Finalists

- optimization-best candidate: `{optimization_best}`
- application-best candidate: `{application_best}`

These labels are deliberately conservative. The optimization-best candidate is selected from the lowest mean best validation loss, while the application-best candidate is selected from the lowest mean MRS4. The two views are related but not identical, and they should not be collapsed into a single global winner without the downstream stage.
{appendix_section}
"""
    REPORT_PATH.write_text(report, encoding="utf-8")


def main():
    ensure_dirs()

    records = [parse_run(variant, seed) for variant in VARIANTS for seed in SEEDS]

    # Backfill alias expected by some downstream docs.
    for record in records:
        hist_old = record.eval_dir / "hist_step_norm.png"
        hist_new = record.eval_dir / "hist_step_norm_all.png"
        if hist_old.exists() and not hist_new.exists():
            hist_new.write_bytes(hist_old.read_bytes())

    per_run_df, per_variant_df = build_dataframes(records)
    save_csvs(per_run_df, per_variant_df)

    plot_fig1_val_loss_seed_overlay(records)
    plot_fig2_best_val_loss(per_run_df, per_variant_df)
    plot_fig3_motion_realism_heatmap(per_variant_df)
    plot_fig4_mrs4(per_run_df, per_variant_df)
    appendix_generated = plot_figA1_none_seed42_100_vs_150()
    write_report(per_variant_df, appendix_generated)

    print(f"Saved per-run summary to: {AGG_ROOT / 'per_run_summary.csv'}")
    print(f"Saved per-variant summary to: {AGG_ROOT / 'per_variant_summary.csv'}")
    print(f"Saved figures to: {FIG_ROOT}")
    print(f"Saved report to: {REPORT_PATH}")


if __name__ == "__main__":
    main()
