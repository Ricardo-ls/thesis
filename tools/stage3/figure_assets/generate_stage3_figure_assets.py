from __future__ import annotations

import os
from pathlib import Path
import sys

os.environ.setdefault("MPLCONFIGDIR", "/tmp/stage3_mplconfig")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.stage3.refinement.run_refinement_interface import load_array
from utils.stage3.controlled_benchmark import DEGRADATION_LABELS, DEGRADATION_NAMES, METHOD_LABELS, METHODS

DOCS_ASSET_ROOT = PROJECT_ROOT / "docs" / "assets" / "stage3"
FIG_ROOT = DOCS_ASSET_ROOT

PHASE1_ROOT = PROJECT_ROOT / "outputs" / "stage3" / "phase1" / "canonical_room3"
CONTROLLED_ROOT = PROJECT_ROOT / "outputs" / "stage3" / "controlled_benchmark"
REFINEMENT_ROOT = PROJECT_ROOT / "outputs" / "stage3" / "refinement"
METHOD_NAME_MAP = {
    "linear_interp": "Linear",
    "savgol_w5_p2": "Savitzky-Golay",
    "kalman_cv_dt1.0_q1e-3_r1e-2": "Kalman",
    "identity_refiner": "Identity",
    "light_savgol_refiner": "Light SG",
    "ddpm_prior_interface_v0": "DDPM v0",
    "ddpm_prior_masked_replace_v1": "DDPM v1",
    "ddpm_prior_masked_blend_v2_alpha0.25": "DDPM v2 alpha=0.25",
}


def ensure_dirs():
    DOCS_ASSET_ROOT.mkdir(parents=True, exist_ok=True)


def save_figure(fig: plt.Figure, path: Path):
    fig.tight_layout()
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def copy_to_docs(path: Path):
    return path


def add_manifest_record(records: list[dict], **kwargs):
    records.append(kwargs)


def pretty_method(name: str) -> str:
    return METHOD_NAME_MAP.get(name, name)


def draw_step_box(ax, x0, y0, w, h, title, subtitle="", face="#F5F5F5", edge="#444444"):
    rect = plt.Rectangle((x0, y0), w, h, facecolor=face, edgecolor=edge, linewidth=1.5)
    ax.add_patch(rect)
    ax.text(x0 + w / 2, y0 + h * 0.62, title, ha="center", va="center", fontsize=11, weight="bold")
    if subtitle:
        ax.text(x0 + w / 2, y0 + h * 0.30, subtitle, ha="center", va="center", fontsize=9)


def generate_overall_stage3_objective(path: Path):
    fig, ax = plt.subplots(figsize=(13, 3.8))
    ax.set_axis_off()
    xs = [0.02, 0.22, 0.42, 0.62, 0.82]
    draw_step_box(ax, xs[0], 0.25, 0.14, 0.5, "Degraded / incomplete", "indoor trajectory")
    draw_step_box(ax, xs[1], 0.25, 0.14, 0.5, "Coarse reconstruction", "baseline recovery")
    draw_step_box(ax, xs[2], 0.25, 0.14, 0.5, "Prior / diffusion", "refinement interface")
    draw_step_box(ax, xs[3], 0.25, 0.14, 0.5, "Refined trajectory", "coarse-to-refined output")
    draw_step_box(ax, xs[4], 0.25, 0.14, 0.5, "Evaluation", "compare with clean target")
    for idx in range(len(xs) - 1):
        ax.annotate("", xy=(xs[idx + 1] - 0.01, 0.5), xytext=(xs[idx] + 0.14, 0.5), arrowprops=dict(arrowstyle="->", lw=2))
    ax.text(
        0.5,
        0.92,
        "Stage 3 solves missing indoor trajectory reconstruction with coarse-to-refined refinement",
        ha="center",
        va="center",
        fontsize=13,
        weight="bold",
    )
    ax.text(
        0.5,
        0.08,
        "This is not generic trajectory prediction: the target is a clean trajectory window with a missing contiguous span.",
        ha="center",
        va="center",
        fontsize=10,
    )
    save_figure(fig, path)


def generate_missing_reconstruction_task(path: Path):
    clean_npz = np.load(PHASE1_ROOT / "data" / "clean_windows_room3.npz", allow_pickle=False)
    miss_npz = np.load(PHASE1_ROOT / "data" / "experiments" / "span20_fixed_seed42" / "missing_span_windows.npz", allow_pickle=False)
    clean = clean_npz["traj_abs"].astype(np.float32)
    observed = miss_npz["traj_obs"].astype(np.float32)
    mask = miss_npz["obs_mask"].astype(bool)
    recon = np.load(PHASE1_ROOT / "baselines" / "span20_fixed_seed42" / "linear_interp" / "results.npz", allow_pickle=False)["traj_hat"].astype(np.float32)

    sample_idx = 0
    gt = clean[sample_idx]
    obs = observed[sample_idx]
    pred = recon[sample_idx]
    obs_mask = mask[sample_idx]

    fig, axes = plt.subplots(1, 3, figsize=(12, 4), sharex=True, sharey=True)
    titles = ["Clean trajectory", "Missing contiguous segment", "Reconstructed trajectory"]
    for ax, title in zip(axes, titles):
        ax.set_title(title)
        ax.set_xlim(0.0, 3.0)
        ax.set_ylim(0.0, 3.0)
        ax.set_aspect("equal", adjustable="box")
        ax.grid(alpha=0.25)
        ax.set_xlabel("x")
        ax.set_ylabel("y")

    axes[0].plot(gt[:, 0], gt[:, 1], color="#1F1F1F", lw=2.5)
    axes[1].plot(gt[:, 0], gt[:, 1], color="#BBBBBB", lw=1.5, linestyle="--")
    axes[1].plot(obs[obs_mask, 0], obs[obs_mask, 1], color="#D62728", lw=2.5)
    axes[2].plot(gt[:, 0], gt[:, 1], color="#BBBBBB", lw=1.5, linestyle="--", label="Clean target")
    axes[2].plot(pred[:, 0], pred[:, 1], color="#1F77B4", lw=2.5, label="Reconstruction")
    axes[2].legend(frameon=False, loc="best")
    fig.suptitle("Phase 1 task: remove one contiguous segment and reconstruct it against the clean target")
    save_figure(fig, path)


def generate_refinement_interface(path: Path):
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.set_axis_off()
    ys = [0.72, 0.42, 0.12]
    labels = [
        ("v0", "coarse abs -> relative displacement -> DDPM candidate -> whole trajectory modified"),
        ("v1", "observed points = coarse; missing points = DDPM candidate"),
        ("v2", "observed points = coarse; missing points = (1 - alpha) * coarse + alpha * DDPM candidate"),
    ]
    colors = ["#EAEAEA", "#E8F4FA", "#FBEEDB"]
    for (version, text), y, color in zip(labels, ys, colors):
        draw_step_box(ax, 0.03, y, 0.10, 0.16, version, face=color)
        draw_step_box(ax, 0.18, y, 0.24, 0.16, "Input", "coarse absolute trajectory", face="#F8F8F8")
        draw_step_box(ax, 0.47, y, 0.22, 0.16, "DDPM interface", face="#F8F8F8")
        draw_step_box(ax, 0.74, y, 0.23, 0.16, "Output rule", text, face="#F8F8F8")
        ax.annotate("", xy=(0.18, y + 0.08), xytext=(0.13, y + 0.08), arrowprops=dict(arrowstyle="->", lw=1.8))
        ax.annotate("", xy=(0.47, y + 0.08), xytext=(0.42, y + 0.08), arrowprops=dict(arrowstyle="->", lw=1.8))
        ax.annotate("", xy=(0.74, y + 0.08), xytext=(0.69, y + 0.08), arrowprops=dict(arrowstyle="->", lw=1.8))
    ax.text(0.5, 0.95, "DDPM refinement interfaces: v0 integration, v1 observed-point protection, v2 masked blending", ha="center", fontsize=13, weight="bold")
    save_figure(fig, path)


def generate_full_vs_masked_comparison(path: Path):
    src = PHASE1_ROOT / "random_span_statistics" / "figures" / "full_vs_masked_comparison.png"
    if src.exists():
        shutil.copy2(src, path)
        return True, [str(src)]
    return False, [str(src)]


def generate_random_span_masked_ade(path: Path):
    df = pd.read_csv(PHASE1_ROOT / "random_span_statistics" / "metrics_summary_mean_std.csv")
    sub = df[df["metric"] == "masked_ADE"].copy()
    order = ["linear_interp", "savgol_w5_p2", "kalman_cv_dt1.0_q1e-3_r1e-2"]
    sub["method"] = pd.Categorical(sub["method"], order, ordered=True)
    sub = sub.sort_values("method")

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    ax.bar([pretty_method(x) for x in sub["method"]], sub["mean"], yerr=sub["std"], color=["#4C78A8", "#72B7B2", "#595959"], capsize=5)
    ax.set_ylabel("masked_ADE")
    ax.set_title("Random-span reliability: missing-segment reconstruction quality")
    ax.grid(axis="y", alpha=0.25)
    save_figure(fig, path)
    return [str(PHASE1_ROOT / "random_span_statistics" / "metrics_summary_mean_std.csv")]


def generate_controlled_degradation_examples(path: Path):
    clean = np.load(CONTROLLED_ROOT / "degradation" / "clean.npy", allow_pickle=False).astype(np.float32)
    mask = np.load(CONTROLLED_ROOT / "degradation" / "mask_span20_fixed_seed42.npy", allow_pickle=False).astype(bool)
    sample_idx = 0
    gt = clean[sample_idx]
    obs_mask = mask[sample_idx]

    plt.rcParams.update(
        {
            "font.size": 16,
            "axes.titlesize": 18,
            "axes.labelsize": 17,
            "xtick.labelsize": 14,
            "ytick.labelsize": 14,
            "legend.fontsize": 14,
        }
    )

    panel_titles = {
        "missing_only": "Missing only",
        "missing_noise": "Missing + noise",
        "missing_drift": "Missing + drift",
        "missing_noise_drift": "Missing + noise + drift",
    }
    clean_color = "#4B5563"
    degraded_color = "#C62828"
    missing_color = "#1565C0"

    x_min = float(gt[:, 0].min() - 0.18)
    x_max = float(gt[:, 0].max() + 0.18)
    y_min = float(gt[:, 1].min() - 0.18)
    y_max = float(gt[:, 1].max() + 0.18)

    fig, axes = plt.subplots(2, 2, figsize=(12.2, 8.9), sharex=True, sharey=True)
    fig.patch.set_facecolor("white")
    axes = axes.ravel()
    for ax, degradation in zip(axes, DEGRADATION_NAMES):
        degraded = np.load(CONTROLLED_ROOT / "degradation" / f"degraded_{degradation}_span20_fixed_seed42.npy", allow_pickle=False).astype(np.float32)
        deg = degraded[sample_idx]
        ax.set_facecolor("#FCFCFB")
        ax.plot(
            gt[:, 0],
            gt[:, 1],
            color="white",
            lw=6.0,
            linestyle=(0, (10, 7)),
            zorder=0,
        )
        ax.plot(
            gt[:, 0],
            gt[:, 1],
            color=clean_color,
            lw=4.0,
            linestyle=(0, (10, 7)),
            dash_capstyle="butt",
            zorder=1.2,
            label="Clean target",
        )
        ax.plot(
            deg[:, 0],
            deg[:, 1],
            color=degraded_color,
            lw=3.8,
            solid_capstyle="round",
            zorder=2,
            label="Degraded/coarse input",
        )
        missing_pts = np.where(~obs_mask)[0]
        if len(missing_pts) > 0:
            ax.scatter(
                gt[missing_pts, 0],
                gt[missing_pts, 1],
                color=missing_color,
                s=88,
                marker="o",
                edgecolors="white",
                linewidths=1.6,
                zorder=3.2,
                label="Missing segment",
            )
        ax.set_title(panel_titles[degradation], pad=8, weight="bold")
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_aspect("equal", adjustable="box")
        ax.grid(color="#CBD5E1", alpha=0.28, linewidth=0.85)
        for spine in ax.spines.values():
            spine.set_linewidth(1.5)
            spine.set_color("#374151")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.suptitle("Controlled benchmark degradation settings", fontsize=22, weight="bold", y=0.975)
    fig.text(
        0.5,
        0.935,
        "Clean target vs. degraded input, with the missing segment highlighted",
        ha="center",
        va="center",
        fontsize=13,
        color="#4B5563",
    )
    fig.legend(
        handles[:3],
        labels[:3],
        frameon=False,
        loc="upper center",
        ncol=3,
        bbox_to_anchor=(0.5, 0.905),
        handlelength=3.0,
        columnspacing=1.5,
        handletextpad=0.5,
    )
    fig.subplots_adjust(left=0.08, right=0.98, bottom=0.08, top=0.82, wspace=0.11, hspace=0.28)
    fig.savefig(path, dpi=260, bbox_inches="tight")
    plt.close(fig)
    return [
        str(CONTROLLED_ROOT / "degradation" / "clean.npy"),
        str(CONTROLLED_ROOT / "degradation" / "mask_span20_fixed_seed42.npy"),
    ] + [str(CONTROLLED_ROOT / "degradation" / f"degraded_{d}_span20_fixed_seed42.npy") for d in DEGRADATION_NAMES]


def generate_controlled_metric_summary(path: Path):
    csv_path = CONTROLLED_ROOT / "eval" / "metrics_summary.csv"
    if not csv_path.exists():
        return False, [str(csv_path)], ["controlled benchmark metrics summary CSV missing"]
    df = pd.read_csv(csv_path)
    order = METHODS
    colors = {"linear_interp": "#4C78A8", "savgol_w5_p2": "#72B7B2", "kalman_cv_dt1.0_q1e-3_r1e-2": "#595959"}
    x = np.arange(len(DEGRADATION_NAMES))
    width = 0.22

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharex=True)
    for ax, metric in zip(axes, ["masked_ADE", "ADE"]):
        for offset, method in zip([-width, 0.0, width], order):
            vals = [df[(df["degradation"] == d) & (df["method"] == method)][metric].iloc[0] for d in DEGRADATION_NAMES]
            ax.bar(x + offset, vals, width=width, label=pretty_method(method), color=colors[method])
        ax.set_xticks(x)
        ax.set_xticklabels([DEGRADATION_LABELS[d] for d in DEGRADATION_NAMES], rotation=15)
        ax.set_ylabel(metric)
        ax.set_title(f"Controlled benchmark: {metric}")
        ax.grid(axis="y", alpha=0.25)
    axes[1].legend(frameon=False, loc="upper right")
    fig.suptitle("Controlled coarse reconstruction under four degradation types")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    todos = ["input_degraded is not present in metrics_summary.csv, so it is omitted from this plot."]
    return True, [str(csv_path)], todos


def generate_alpha_sweep_mean_masked_ade(path: Path):
    csv_path = REFINEMENT_ROOT / "alpha_sweep" / "alpha_sweep_summary.csv"
    df = pd.read_csv(csv_path)
    plot_df = df.groupby(["coarse_method", "alpha"], as_index=False).agg(
        mean_masked_ADE=("masked_ADE", "mean"),
        min_masked_ADE=("masked_ADE", "min"),
        max_masked_ADE=("masked_ADE", "max"),
    )

    plt.rcParams.update(
        {
            "font.size": 16,
            "axes.titlesize": 20,
            "axes.labelsize": 17,
            "xtick.labelsize": 15,
            "ytick.labelsize": 15,
            "legend.fontsize": 14,
        }
    )

    style_map = {
        "linear_interp": {"color": "#1D4ED8", "marker": "o", "label": "Linear"},
        "savgol_w5_p2": {"color": "#047857", "marker": "s", "label": "Savitzky-Golay"},
        "kalman_cv_dt1.0_q1e-3_r1e-2": {"color": "#B45309", "marker": "^", "label": "Kalman"},
    }

    alpha_values = sorted(plot_df["alpha"].unique())
    alpha_to_pos = {alpha: idx for idx, alpha in enumerate(alpha_values)}

    fig, ax = plt.subplots(figsize=(9.2, 5.8))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("#FCFCFB")
    shared_best_done = False
    for method in METHODS:
        method_df = plot_df[plot_df["coarse_method"] == method].sort_values("alpha")
        style = style_map[method]
        lower = method_df["mean_masked_ADE"] - method_df["min_masked_ADE"]
        upper = method_df["max_masked_ADE"] - method_df["mean_masked_ADE"]
        x_pos = np.array([alpha_to_pos[a] for a in method_df["alpha"]], dtype=float)
        ax.fill_between(
            x_pos,
            method_df["min_masked_ADE"].to_numpy(),
            method_df["max_masked_ADE"].to_numpy(),
            color=style["color"],
            alpha=0.08,
            zorder=1,
        )
        ax.errorbar(
            x_pos,
            method_df["mean_masked_ADE"],
            yerr=np.vstack([lower.to_numpy(), upper.to_numpy()]),
            color=style["color"],
            marker=style["marker"],
            markersize=10.5,
            linewidth=3.8,
            elinewidth=1.5,
            capsize=4,
            alpha=0.98,
            label=style["label"],
            zorder=2,
        )

        best_row = method_df.sort_values(["mean_masked_ADE", "alpha"]).iloc[0]
        best_x = alpha_to_pos[float(best_row["alpha"])]
        y_offset = -18 if method != "kalman_cv_dt1.0_q1e-3_r1e-2" else 10
        ax.scatter(
            [best_x],
            [best_row["mean_masked_ADE"]],
            s=170,
            marker=style["marker"],
            color=style["color"],
            edgecolors="white",
            linewidths=1.4,
            zorder=3,
        )
        if method in {"linear_interp", "savgol_w5_p2"}:
            if not shared_best_done:
                shared_best_done = True
                ax.annotate(
                    "Linear + Savitzky-Golay:\nbest alpha = 0.00",
                    xy=(best_x, best_row["mean_masked_ADE"]),
                    xytext=(16, -28),
                    textcoords="offset points",
                    color="#1E3A8A",
                    fontsize=12.5,
                    weight="bold",
                )
        else:
            ax.annotate(
                f"Kalman:\nbest alpha = {best_row['alpha']:.2f}",
                xy=(best_x, best_row["mean_masked_ADE"]),
                xytext=(10, y_offset),
                textcoords="offset points",
                color=style["color"],
                fontsize=12.5,
                weight="bold",
            )

    ax.set_title("Alpha sensitivity: missing-segment reconstruction quality", pad=12, weight="bold")
    ax.set_xlabel("alpha")
    ax.set_ylabel("mean masked_ADE")
    ax.set_xticks(np.arange(len(alpha_values)))
    ax.set_xticklabels([f"{alpha:.2f}" for alpha in alpha_values])
    ax.tick_params(axis="x", labelrotation=90)
    ax.grid(axis="y", color="#CBD5E1", alpha=0.4, linewidth=0.9)
    ax.grid(axis="x", visible=False)
    for spine in ax.spines.values():
        spine.set_linewidth(1.5)
        spine.set_color("#111827")
    ax.legend(frameon=False, loc="upper right")
    ax.text(0.99, 0.03, "Lower is better", transform=ax.transAxes, ha="right", va="bottom", fontsize=13, color="#374151")
    ax.text(
        0.01,
        0.97,
        "Mean across four degradation settings",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=12.5,
        color="#4B5563",
    )
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return [str(csv_path)]


def generate_refinement_comparison(path: Path):
    df = pd.read_csv(REFINEMENT_ROOT / "eval" / "refinement_metrics.csv")
    order = [
        "identity_refiner",
        "light_savgol_refiner",
        "ddpm_prior_interface_v0",
        "ddpm_prior_masked_replace_v1",
        "ddpm_prior_masked_blend_v2_alpha0.25",
    ]
    grouped = df.groupby("refiner", as_index=False)[["masked_ADE", "ADE"]].mean()
    grouped["refiner"] = pd.Categorical(grouped["refiner"], order, ordered=True)
    grouped = grouped.sort_values("refiner")
    x = np.arange(len(grouped))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 4.8))
    ax.bar(x - width / 2, grouped["masked_ADE"], width=width, color="#4C78A8", label="masked_ADE")
    ax.bar(x + width / 2, grouped["ADE"], width=width, color="#B0B0B0", label="ADE")
    ax.set_xticks(x)
    ax.set_xticklabels([pretty_method(r) for r in grouped["refiner"]], rotation=15)
    ax.set_ylabel("Metric value")
    ax.set_title("Refinement progression: interface integration vs missing-segment reconstruction quality")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False)
    save_figure(fig, path)
    return [str(REFINEMENT_ROOT / "eval" / "refinement_metrics.csv")]


def generate_alpha_sweep_masked_ade(path: Path):
    return generate_alpha_sweep_mean_masked_ade(path)


def generate_alpha_sweep_improvement(path: Path):
    df = pd.read_csv(REFINEMENT_ROOT / "alpha_sweep" / "alpha_sweep_summary.csv")
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    for method in METHODS:
        sub = df[df["coarse_method"] == method].sort_values("alpha").copy()
        base = float(sub[sub["alpha"] == 0.0]["masked_ADE"].iloc[0])
        sub["delta_vs_alpha0"] = base - sub["masked_ADE"]
        ax.plot(sub["alpha"], sub["delta_vs_alpha0"], marker="o", lw=2, label=pretty_method(method))
    ax.axhline(0.0, color="#444444", lw=1.2, linestyle="--")
    ax.set_xlabel("alpha")
    ax.set_ylabel("masked_ADE improvement vs alpha=0.00")
    ax.set_title("Alpha sensitivity relative to identity blending")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    save_figure(fig, path)
    return [str(REFINEMENT_ROOT / "alpha_sweep" / "alpha_sweep_summary.csv")]


def build_manifest(records: list[dict], path: Path, todos: list[str], missing: list[str]):
    lines = [
        "# Figure Manifest",
        "",
        "## Figures for first full review",
        "",
        "### 1. Problem and protocol",
        "",
        "- `overall_stage3_objective.png`",
        "- `missing_reconstruction_task.png`",
        "- `refinement_interface_v0_v1_v2.png`",
        "",
        "### 2. Benchmark evidence",
        "",
        "- `full_vs_masked_comparison.png`",
        "- `random_span_masked_ADE_mean_std.png`",
        "",
        "### 3. Controlled coarse reconstruction",
        "",
        "- `controlled_degradation_examples.png`",
        "- `controlled_benchmark_metric_summary.png`",
        "",
        "### 4. DDPM prior refinement",
        "",
        "- `refinement_v0_v1_v2_comparison.png`",
        "- `alpha_sweep_masked_ADE.png`",
        "- `alpha_sweep_improvement_masked_ADE.png`",
        "",
        "| Figure filename | Type | Source files used | Purpose | Status | Short interpretation |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for rec in records:
        lines.append(
            f"| {rec['filename']} | {rec['type']} | {rec['sources']} | {rec['purpose']} | {rec['status']} | {rec['interpretation']} |"
        )
    lines.extend(["", "## Missing data sources", ""])
    if missing:
        lines.extend([f"- {item}" for item in missing])
    else:
        lines.append("- None")
    lines.extend(["", "## TODO", ""])
    if todos:
        lines.extend([f"- {item}" for item in todos])
    else:
        lines.append("- None")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    ensure_dirs()
    records: list[dict] = []
    todos: list[str] = []
    missing_sources: list[str] = []
    generated_paths: list[Path] = []
    copied_paths: list[Path] = []

    conceptual_specs = [
        ("overall_stage3_objective.png", generate_overall_stage3_objective, "conceptual", "Show the Stage 3 reconstruction/refinement objective.", "Stage 3 is missing indoor trajectory reconstruction, not generic forecasting."),
        ("missing_reconstruction_task.png", generate_missing_reconstruction_task, "conceptual", "Show the one contiguous missing-segment task definition.", "Missing-segment reconstruction quality should be read against the clean target."),
        ("refinement_interface_v0_v1_v2.png", generate_refinement_interface, "conceptual", "Explain v0/v1/v2 DDPM refinement interfaces.", "v0 changes the whole trajectory; v1/v2 protect observed points."),
    ]
    for filename, fn, fig_type, purpose, interp in conceptual_specs:
        out = FIG_ROOT / filename
        fn(out)
        generated_paths.append(out)
        copied_paths.append(copy_to_docs(out))
        add_manifest_record(records, filename=filename, type=fig_type, sources="programmatic schematic", purpose=purpose, status="generated", interpretation=interp)

    # A reused existing figure
    out = FIG_ROOT / "full_vs_masked_comparison.png"
    ok, srcs = generate_full_vs_masked_comparison(out)
    if ok:
        generated_paths.append(out)
        copied_paths.append(copy_to_docs(out))
        status = "copied"
        interp = "Full-trajectory consistency and missing-segment reconstruction quality can rank methods differently."
    else:
        missing_sources.extend(srcs)
        status = "missing source"
        interp = "Existing random-span comparison figure not found."
    add_manifest_record(records, filename=out.name, type="data-result", sources=", ".join(srcs), purpose="Show full vs masked metric ranking differences.", status=status, interpretation=interp)

    data_specs = []
    # custom generated data-result figures
    out = FIG_ROOT / "random_span_masked_ADE_mean_std.png"
    srcs = generate_random_span_masked_ade(out)
    generated_paths.append(out)
    copied_paths.append(copy_to_docs(out))
    add_manifest_record(records, filename=out.name, type="data-result", sources=", ".join(srcs), purpose="Show mean ± std masked_ADE over random span positions.", status="generated", interpretation="Masked metrics are the direct view of missing-segment reconstruction quality.")

    out = FIG_ROOT / "controlled_degradation_examples.png"
    srcs = generate_controlled_degradation_examples(out)
    generated_paths.append(out)
    copied_paths.append(copy_to_docs(out))
    add_manifest_record(records, filename=out.name, type="data-result", sources=", ".join(srcs), purpose="Show the four controlled degradation settings.", status="generated", interpretation="The controlled benchmark stresses reconstruction under missingness, noise, drift, and combined degradation.")

    out = FIG_ROOT / "controlled_benchmark_metric_summary.png"
    ok, srcs, local_todos = generate_controlled_metric_summary(out)
    todos.extend(local_todos)
    if ok:
        generated_paths.append(out)
        copied_paths.append(copy_to_docs(out))
        status = "generated"
        interp = "Baseline behavior changes across degradation types; masked_ADE emphasizes the missing segment."
    else:
        missing_sources.extend(srcs)
        status = "missing source"
        interp = "Controlled metric summary could not be generated from existing outputs."
    add_manifest_record(records, filename=out.name, type="data-result", sources=", ".join(srcs), purpose="Summarize controlled coarse reconstruction metrics.", status=status, interpretation=interp)

    out = FIG_ROOT / "refinement_v0_v1_v2_comparison.png"
    srcs = generate_refinement_comparison(out)
    generated_paths.append(out)
    copied_paths.append(copy_to_docs(out))
    add_manifest_record(records, filename=out.name, type="data-result", sources=", ".join(srcs), purpose="Compare identity, Light SG, DDPM v0, DDPM v1, and DDPM v2 alpha=0.25.", status="generated", interpretation="v0 proves integration, v1 protects observed points, and v2 blends the DDPM candidate into the missing span.")

    out = FIG_ROOT / "alpha_sweep_masked_ADE.png"
    srcs = generate_alpha_sweep_masked_ade(out)
    generated_paths.append(out)
    copied_paths.append(copy_to_docs(out))
    add_manifest_record(records, filename=out.name, type="data-result", sources=", ".join(srcs), purpose="Show alpha sensitivity by coarse method.", status="generated", interpretation="Linear and Savitzky-Golay prefer alpha=0.00, while Kalman benefits only from very small alpha.")

    out = FIG_ROOT / "alpha_sweep_mean_masked_ADE.png"
    srcs = generate_alpha_sweep_mean_masked_ade(out)
    generated_paths.append(out)
    copied_paths.append(copy_to_docs(out))
    add_manifest_record(records, filename=out.name, type="data-result", sources=", ".join(srcs), purpose="Show mean masked_ADE by alpha for the three coarse methods.", status="generated", interpretation="Linear and Savitzky-Golay are best at alpha=0.00, while Kalman is best near alpha=0.10.")

    out = FIG_ROOT / "alpha_sweep_improvement_masked_ADE.png"
    srcs = generate_alpha_sweep_improvement(out)
    generated_paths.append(out)
    copied_paths.append(copy_to_docs(out))
    add_manifest_record(records, filename=out.name, type="data-result", sources=", ".join(srcs), purpose="Show masked_ADE improvement relative to alpha=0.00.", status="generated", interpretation="Large alpha usually hurts, which indicates the unconditional DDPM prior is not a reliable direct refiner.")

    build_manifest(records, FIG_ROOT / "figure_manifest.md", todos, missing_sources)

    print("=" * 72)
    print("Generated docs/assets/stage3 figure paths")
    for path in generated_paths:
        print(path)
    print(DOCS_ASSET_ROOT / "figure_manifest.md")
    print("-" * 72)
    print("Missing data sources")
    if missing_sources:
        for item in missing_sources:
            print(item)
    else:
        print("None")
    print("-" * 72)
    print("TODOs")
    if todos:
        for item in todos:
            print(item)
    else:
        print("None")
    print("=" * 72)


if __name__ == "__main__":
    main()
