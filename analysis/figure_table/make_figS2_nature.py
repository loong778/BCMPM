import argparse
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ORDER = [
    "MPM-Only",
    "MPM-Clinical",
    "Clinical-Only",
    "H&E-Only",
    "H&E-Clinical",
    "H&E-MPM",
    "BiCA",
    "BCMPM",
]


COHORTS = ["Train", "Validation", "Test"]
COHORT_TITLE = {
    "Train": "training cohort",
    "Validation": "Internal validation cohort",
    "Test": "External test cohort",
}

TIME_POINTS = [36, 60]
TIME_LABEL = {36: "3 years", 60: "5 years"}

COLORS = {
    "MPM-Only": "#8D8D8D",
    "MPM-Clinical": "#74ADD1",
    "Clinical-Only": "#E45756",
    "H&E-Only": "#F1A340",
    "H&E-Clinical": "#F2C94C",
    "H&E-MPM": "#4C78A8",
    "BiCA": "#9C6BC8",
    "BCMPM": "#7B1FA2",
}


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
        if all((root / modality_dir_name(mod) / "results.npz").exists() for mod in ORDER):
            return root
    searched = "\n".join(str(p.resolve()) for p in candidates)
    raise FileNotFoundError(f"Could not find all 9 modality result files. Searched:\n{searched}")


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
            "axes.labelsize": 8.5,
            "axes.titlesize": 8.8,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 5.8,
            "axes.linewidth": 0.8,
            "xtick.major.width": 0.8,
            "ytick.major.width": 0.8,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def format_auc_label(modality, auc_val, ci_low, ci_high):
    if np.isfinite(ci_low) and np.isfinite(ci_high):
        return f"{modality} AUC={auc_val:.3f} [{ci_low:.3f}-{ci_high:.3f}]"
    return f"{modality} AUC={auc_val:.3f}"


def draw_roc_panel(ax, results, time_point, cohort, panel_letter):
    title = f"Time-dependent ROC curves at {TIME_LABEL[time_point]} ({COHORT_TITLE[cohort]})"

    for modality in ORDER:
        fpr, tpr, auc_val, ci_low, ci_high = results[modality]["td_roc_raw"][time_point][cohort]
        lw = 1.65 if modality == "BCMPM" else 1.25
        zorder = 5 if modality == "BCMPM" else 3
        ax.step(
            fpr,
            tpr,
            where="post",
            color=COLORS[modality],
            lw=lw,
            label=format_auc_label(modality, float(auc_val), float(ci_low), float(ci_high)),
            zorder=zorder,
        )

    ax.plot([0, 1], [0, 1], color="#4D4D4D", lw=0.85, ls="--", zorder=1)
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.05)
    ax.set_xlabel("False positive rate", fontweight="bold")
    ax.set_ylabel("True positive rate", fontweight="bold")
    ax.set_title(title, fontsize=8.8, fontweight="bold", pad=8)
    ax.tick_params(axis="both", length=3.5)
    ax.legend(
        loc="lower right",
        ncol=1,
        frameon=True,
        framealpha=0.88,
        facecolor="white",
        edgecolor="#DDDDDD",
        handlelength=1.7,
        labelspacing=0.22,
        borderpad=0.35,
    )
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.text(
        -0.105,
        1.075,
        panel_letter,
        transform=ax.transAxes,
        fontsize=14,
        fontweight="bold",
        va="top",
        ha="left",
    )


def make_figs2(results, output_dir):
    fig, axes = plt.subplots(2, 3, figsize=(14.6, 9.0))
    fig.subplots_adjust(left=0.060, right=0.985, top=0.955, bottom=0.075, wspace=0.34, hspace=0.42)

    panel_letters = list("abcdef")
    summary_rows = []
    idx = 0
    for row_idx, time_point in enumerate(TIME_POINTS):
        for col_idx, cohort in enumerate(COHORTS):
            draw_roc_panel(axes[row_idx, col_idx], results, time_point, cohort, panel_letters[idx])
            for modality in ORDER:
                _, _, auc_val, ci_low, ci_high = results[modality]["td_roc_raw"][time_point][cohort]
                summary_rows.append(
                    {
                        "Panel": panel_letters[idx],
                        "Time_months": time_point,
                        "Time_label": TIME_LABEL[time_point],
                        "Dataset": cohort,
                        "Modality": modality,
                        "AUC": float(auc_val),
                        "95% CI Low": float(ci_low),
                        "95% CI High": float(ci_high),
                    }
                )
            idx += 1

    output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(summary_rows).to_csv(output_dir / "FigS2_3y_5y_tdroc_summary.csv", index=False, float_format="%.4f")

    png_path = output_dir / "FigS2_3y_5y_ROC_9modalities_600dpi.png"
    svg_path = output_dir / "FigS2_3y_5y_ROC_9modalities_600dpi.svg"
    fig.savefig(png_path, dpi=600, bbox_inches="tight")
    fig.savefig(svg_path, bbox_inches="tight")
    plt.close(fig)
    return png_path, svg_path


def main():
    parser = argparse.ArgumentParser(
        description="Generate supplementary FigS2: 3-year and 5-year time-dependent ROC curves for 9 modalities."
    )
    parser.add_argument("--result-root", default=None, help="Root containing modality result folders with results.npz.")
    parser.add_argument("--output-dir", default="./xxx/xx/FigS2_Nature")
    args = parser.parse_args()

    configure_style()
    result_root = resolve_result_root(args.result_root)
    results = load_results(result_root)
    png_path, svg_path = make_figs2(results, Path(args.output_dir).resolve())

    print(f"Result root: {result_root}")
    print(f"Saved: {png_path}")
    print(f"Saved: {svg_path}")


if __name__ == "__main__":
    main()
