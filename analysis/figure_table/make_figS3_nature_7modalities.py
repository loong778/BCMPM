import argparse
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch
from scipy.stats import gaussian_kde, mannwhitneyu


ORDER = [
    "MPM-Only",
    "MPM-Clinical",
    "Clinical-Only",
    "H&E-Only",
    "H&E-Clinical",
    "H&E-MPM",
    "BiCA",
]


COHORTS = ["Train", "Validation", "Test"]
COHORT_LABELS = {
    "Train": "Training cohort",
    "Validation": "Internal validation cohort",
    "Test": "External test cohort",
}

COLOR_NON_RECUR = "#1f77b4"
COLOR_RECUR = "#d62728"


def modality_dir_name(modality):
    return modality.lower().replace("-", "_")


def resolve_result_root(user_root=None):
    candidates = []
    if user_root:
        candidates.append(Path(user_root))
    candidates.extend(
        [
            Path('./xxx/xx'),
            # Path('./xxx/xx') / "results",
            # Path("results"),
            # Path('./xxx/xx') / "results",
        ]
    )
    for root in candidates:
        root = root.resolve()
        if all((root / modality_dir_name(modality) / "results.npz").exists() for modality in ORDER):
            return root
    searched = "\n".join(str(path.resolve()) for path in candidates)
    raise FileNotFoundError(f"Could not find all non-BCMPM modality result files. Searched:\n{searched}")


def load_results(result_root):
    results = {}
    for modality in ORDER:
        npz_path = result_root / modality_dir_name(modality) / "results.npz"
        data = np.load(npz_path, allow_pickle=True)
        results[modality] = {
            key: data[key].item() if data[key].shape == () else data[key]
            for key in data.files
        }
    return results


def configure_style():
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 8,
            "axes.labelsize": 8,
            "axes.titlesize": 9,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 6.5,
            "axes.linewidth": 0.8,
            "xtick.major.width": 0.8,
            "ytick.major.width": 0.8,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def x_limits_for_modality(result):
    values = np.concatenate([np.asarray(result["risk"][cohort], dtype=float) for cohort in COHORTS])
    data_min = float(np.nanmin(values))
    data_max = float(np.nanmax(values))
    data_range = max(data_max - data_min, 1e-8)
    return data_min - 0.12 * data_range, data_max + 0.12 * data_range


def p_value_text(p_value):
    return "p < 0.0001" if p_value < 0.0001 else f"p = {p_value:.4f}"


def draw_raincloud_panel(ax, result, modality, panel_letter, rng):
    row_gap = 2.0
    kde_max_height = 0.8
    box_vert_offset = 0.35
    scatter_vert_offset = 0.75
    jitter_strength = 0.05
    y_baselines = np.arange(len(COHORTS))[::-1] * row_gap
    x_min, x_max = x_limits_for_modality(result)
    x_grid = np.linspace(x_min, x_max, 250)
    rows = []

    for idx, cohort in enumerate(COHORTS):
        y_base = y_baselines[idx]
        risk = np.asarray(result["risk"][cohort], dtype=float)
        event = np.asarray(result["event"][cohort], dtype=int)
        risk_non = risk[event == 0]
        risk_rec = risk[event == 1]

        for values, color in [(risk_non, COLOR_NON_RECUR), (risk_rec, COLOR_RECUR)]:
            if len(values) > 1 and np.nanstd(values) > 0:
                density = gaussian_kde(values)(x_grid)
                density = density / max(float(density.max()), 1e-8) * kde_max_height
                ax.fill_between(x_grid, y_base, y_base + density, color=color, alpha=0.7, linewidth=0)
                ax.plot(x_grid, y_base + density, color=color, linewidth=1.0)

        box_y_center = y_base - box_vert_offset
        box = ax.boxplot(
            [risk_non, risk_rec],
            positions=[box_y_center - 0.15, box_y_center + 0.15],
            vert=False,
            widths=0.25,
            patch_artist=True,
            showfliers=False,
            medianprops={"color": "black", "linewidth": 1.0},
            whiskerprops={"linewidth": 0.8, "color": "#666666"},
            capprops={"linewidth": 0.8, "color": "#666666"},
        )
        for patch, color in zip(box["boxes"], [COLOR_NON_RECUR, COLOR_RECUR]):
            patch.set_facecolor(color)
            patch.set_alpha(0.85)
            patch.set_edgecolor("#666666")
            patch.set_linewidth(0.55)

        scatter_y_center = y_base - scatter_vert_offset
        ax.scatter(
            risk_non,
            rng.normal(scatter_y_center, jitter_strength, size=len(risk_non)),
            s=3.2,
            color=COLOR_NON_RECUR,
            alpha=0.45,
            linewidths=0,
            zorder=1,
        )
        ax.scatter(
            risk_rec,
            rng.normal(scatter_y_center, jitter_strength, size=len(risk_rec)),
            s=3.2,
            color=COLOR_RECUR,
            alpha=0.45,
            linewidths=0,
            zorder=1,
        )

        if len(risk_non) > 0 and len(risk_rec) > 0:
            p_value = float(mannwhitneyu(risk_non, risk_rec, alternative="two-sided").pvalue)
            ax.text(
                x_max - 0.015 * (x_max - x_min),
                y_base + kde_max_height * 0.62,
                p_value_text(p_value),
                fontsize=6.4,
                ha="right",
                va="center",
            )
        else:
            p_value = np.nan

        rows.append(
            {
                "Panel": panel_letter,
                "Modality": modality,
                "Dataset": cohort,
                "Non_recurrence_n": int(len(risk_non)),
                "Recurrence_n": int(len(risk_rec)),
                "Mann_Whitney_p": p_value,
                "Non_recurrence_median": float(np.nanmedian(risk_non)) if len(risk_non) else np.nan,
                "Recurrence_median": float(np.nanmedian(risk_rec)) if len(risk_rec) else np.nan,
            }
        )

    ax.set_yticks(y_baselines)
    ax.set_yticklabels([COHORT_LABELS[cohort] for cohort in COHORTS], fontweight="bold")
    ax.set_xlabel(modality, fontweight="bold", labelpad=2)
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(-scatter_vert_offset - 0.30, y_baselines[0] + kde_max_height + 0.20)
    ax.tick_params(axis="both", labelsize=7)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(
        handles=[
            Patch(facecolor=COLOR_NON_RECUR, label="Non-recurrence"),
            Patch(facecolor=COLOR_RECUR, label="Recurrence"),
        ],
        bbox_to_anchor=(1.01, 1.0),
        loc="upper left",
        fontsize=6.2,
        frameon=False,
        title="Group",
        title_fontsize=6.6,
        handlelength=1.0,
        borderaxespad=0,
    )
    ax.text(
        -0.14,
        1.07,
        panel_letter,
        transform=ax.transAxes,
        fontsize=14,
        fontweight="bold",
        va="top",
        ha="left",
    )
    return rows


def make_figs3(results, output_dir):
    fig, axes = plt.subplots(4, 2, figsize=(13.2, 13.2))
    fig.subplots_adjust(left=0.095, right=0.925, top=0.975, bottom=0.055, wspace=0.52, hspace=0.45)
    rng = np.random.default_rng(42)
    summary_rows = []

    for ax, modality, panel_letter in zip(axes.flat, ORDER, list("abcdefg")):
        summary_rows.extend(draw_raincloud_panel(ax, results[modality], modality, panel_letter, rng))

    fig.delaxes(axes.flat[len(ORDER)])

    output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(summary_rows).to_csv(output_dir / "FigS3_raincloud_statistics.csv", index=False, float_format="%.5g")

    png_path = output_dir / "FigS3_raincloud_7modalities_600dpi.png"
    svg_path = output_dir / "FigS3_raincloud_7modalities_600dpi.svg"
    fig.savefig(png_path, dpi=600, bbox_inches="tight")
    fig.savefig(svg_path, bbox_inches="tight")
    plt.close(fig)
    return png_path, svg_path


def main():
    parser = argparse.ArgumentParser(
        description="Generate supplementary FigS3 raincloud plots for 7 non-BCMPM modalities."
    )
    parser.add_argument("--result-root", default=None, help="Root containing modality result folders with results.npz.")
    parser.add_argument("--output-dir", default="./xxx/xx/FigS3_Nature")
    args = parser.parse_args()

    configure_style()
    result_root = resolve_result_root(args.result_root)
    results = load_results(result_root)
    png_path, svg_path = make_figs3(results, Path(args.output_dir).resolve())

    print(f"Result root: {result_root}")
    print(f"Saved: {png_path}")
    print(f"Saved: {svg_path}")


if __name__ == "__main__":
    main()
