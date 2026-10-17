# ================== filename: make_figS19_20_nature.py ==================
# Generate FigS19-20: cohort-level composite heatmaps based on compare_all_modalities.py.
#
# FigS19: Validation cohort
# FigS20: Test cohort
#
# Layout per figure:
#   Row 1: 4 single-model confusion-matrix heatmaps
#   Row 2: 3 BCMPM-vs-ablation cross-classification heatmaps

import os

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix

from utils import BASE_RESULT_DIR, load_modality_result, time_dependent_roc_ipcw


TIME_CUT = 60
# OUTPUT_DIR = os.path.join(BASE_RESULT_DIR, "FigS19_20_confusion_heatmaps")
# Example output path; generated files are written here.
OUTPUT_DIR = './xxx/xx'
TARGET_MODS = [
    "BCMPM",
    "Clinical-Only",
    "MPM-Only",
    "H&E-Only",
]

COMPARE_PAIRS = [
    ("BCMPM", "Clinical-Only"),
    ("BCMPM", "MPM-Only"),
    ("BCMPM", "H&E-Only"),
]

COHORT_TO_FIG = {
    "Validation": "FigS19",
    "Test": "FigS20",
}


def set_publication_style() -> None:
    """Use the same Figure 5 heatmap style from compare_all_modalities.py."""
    plt.rcParams["font.family"] = "Arial"
    plt.rcParams["font.size"] = 12
    plt.rcParams["axes.labelweight"] = "bold"
    plt.rcParams["axes.titleweight"] = "bold"
    plt.rcParams["pdf.fonttype"] = 42
    plt.rcParams["ps.fonttype"] = 42
    sns.set_style("white")


def compute_training_cutoff(result: dict) -> float:
    """Training-set 60-month Youden cutoff, matching compare_all_modalities.py."""
    train_risk = result["risk"]["Train"]
    train_time = result["time"]["Train"]
    train_event = result["event"]["Train"]
    _, train_fpr, train_tpr, train_thresh = time_dependent_roc_ipcw(
        train_risk,
        train_time,
        train_event,
        TIME_CUT,
    )
    return float(train_thresh[np.argmax(train_tpr - train_fpr)])


def load_target_results() -> dict:
    all_results = {}
    for mod in TARGET_MODS:
        result = load_modality_result(mod, BASE_RESULT_DIR)
        result["best_threshold"] = compute_training_cutoff(result)
        all_results[mod] = result
        print(f"Loaded {mod}; training cutoff = {result['best_threshold']:.4f}")
    return all_results


def get_confusion_category(true_label, pred_high) -> str:
    if true_label == 1 and pred_high == 1:
        return "TP"
    if true_label == 0 and pred_high == 0:
        return "TN"
    if true_label == 0 and pred_high == 1:
        return "FP"
    if true_label == 1 and pred_high == 0:
        return "FN"
    raise ValueError(f"Unexpected label/prediction pair: {true_label}, {pred_high}")


def add_panel_label(ax, label: str, x: float = -0.17, y: float = 1.08) -> None:
    ax.text(
        x,
        y,
        label,
        transform=ax.transAxes,
        fontsize=22,
        fontweight="bold",
        va="bottom",
        ha="right",
    )


def draw_single_model_heatmap(ax, result: dict, mod: str, cohort: str) -> None:
    risk = result["risk"][cohort]
    event = result["event"][cohort]
    cutoff = result["best_threshold"]
    pred_high = risk >= cutoff

    cm = confusion_matrix(event, pred_high, labels=[0, 1])
    percent = cm / cm.sum() * 100

    annot = np.empty_like(cm, dtype="U24")
    for i in range(2):
        for j in range(2):
            annot[i, j] = f"{percent[i, j]:.1f}%\n({cm[i, j]})"

    sns.heatmap(
        percent,
        annot=annot,
        fmt="",
        cmap="Blues",
        cbar=True,
        ax=ax,
        xticklabels=["Low risk", "High risk"],
        yticklabels=["No recurrence", "Recurrence"],
        linewidths=1.5,
        linecolor="black",
        annot_kws={"size": 10},
        cbar_kws={"label": "Percent", "shrink": 0.95, "pad": 0.03},
    )

    ax.set_title(f"{mod} in the {cohort} cohorts", fontsize=10, fontweight="bold", pad=10)
    ax.set_xlabel("Predicted Label", fontsize=9, fontweight="bold")
    ax.set_ylabel("True Label", fontsize=9, fontweight="bold")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=0, fontsize=8)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=8)

    cbar = ax.collections[0].colorbar
    cbar.set_label("Percent", fontsize=9, fontweight="bold")
    cbar.ax.tick_params(labelsize=8)


def draw_cross_classification_heatmap(
    ax,
    all_results: dict,
    mod_base: str,
    mod_comp: str,
    cohort: str,
) -> None:
    risk_base = all_results[mod_base]["risk"][cohort]
    risk_comp = all_results[mod_comp]["risk"][cohort]
    cutoff_base = all_results[mod_base]["best_threshold"]
    cutoff_comp = all_results[mod_comp]["best_threshold"]
    event = all_results[mod_base]["event"][cohort]
    total_n = len(event)

    pred_base_high = risk_base >= cutoff_base
    pred_comp_high = risk_comp >= cutoff_comp

    cat_base = [get_confusion_category(e, p) for e, p in zip(event, pred_base_high)]
    cat_comp = [get_confusion_category(e, p) for e, p in zip(event, pred_comp_high)]

    row_order = ["FP", "TN", "FN", "TP"]
    col_order = ["TP", "FN", "TN", "FP"]

    cross_tab = np.zeros((len(row_order), len(col_order)), dtype=int)
    row_lookup = {label: idx for idx, label in enumerate(row_order)}
    col_lookup = {label: idx for idx, label in enumerate(col_order)}
    for row_label, col_label in zip(cat_base, cat_comp):
        cross_tab[row_lookup[row_label], col_lookup[col_label]] += 1

    percent_matrix = cross_tab / total_n * 100
    annot_matrix = np.empty_like(cross_tab, dtype="U24")
    for i in range(len(row_order)):
        for j in range(len(col_order)):
            annot_matrix[i, j] = f"{percent_matrix[i, j]:.1f}%\n({cross_tab[i, j]})"

    sns.heatmap(
        percent_matrix,
        annot=annot_matrix,
        fmt="",
        cmap="Blues",
        cbar=True,
        ax=ax,
        xticklabels=col_order,
        yticklabels=row_order,
        linewidths=1.5,
        linecolor="black",
        annot_kws={"size": 10},
        cbar_kws={"label": "Percent", "shrink": 0.82, "pad": 0.03},
    )

    ax.set_title(
        f"{mod_base} vs. {mod_comp} risk classification\n"
        f"comparison analysis in the {cohort} cohorts",
        fontsize=10,
        fontweight="bold",
        pad=10,
    )
    ax.set_ylabel(mod_base, fontsize=9, fontweight="bold", labelpad=8)
    ax.set_xlabel(mod_comp, fontsize=9, fontweight="bold", labelpad=8)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=0, fontsize=8)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=8)

    cbar = ax.collections[0].colorbar
    cbar.set_label("Percent", fontsize=9, fontweight="bold")
    cbar.ax.tick_params(labelsize=8)


def draw_cohort_figure(all_results: dict, cohort: str, fig_name: str) -> None:
    fig = plt.figure(figsize=(18.0, 8.3))
    outer_gs = fig.add_gridspec(
        nrows=2,
        ncols=1,
        height_ratios=[1.0, 1.15],
        hspace=0.38,
    )
    top_gs = outer_gs[0].subgridspec(1, 4, wspace=0.48)
    bottom_gs = outer_gs[1].subgridspec(1, 3, wspace=0.42)

    panel_labels = list("abcdefg")

    for idx, mod in enumerate(TARGET_MODS):
        ax = fig.add_subplot(top_gs[0, idx])
        draw_single_model_heatmap(ax, all_results[mod], mod, cohort)
        add_panel_label(ax, panel_labels[idx])

    for idx, (mod_base, mod_comp) in enumerate(COMPARE_PAIRS):
        ax = fig.add_subplot(bottom_gs[0, idx])
        draw_cross_classification_heatmap(ax, all_results, mod_base, mod_comp, cohort)
        add_panel_label(ax, panel_labels[4 + idx], x=-0.15, y=1.06)

    fig.subplots_adjust(left=0.055, right=0.985, top=0.94, bottom=0.075)

    png_path = os.path.join(OUTPUT_DIR, f"{fig_name}_{cohort}_confusion_heatmaps.png")
    svg_path = os.path.join(OUTPUT_DIR, f"{fig_name}_{cohort}_confusion_heatmaps.svg")
    pdf_path = os.path.join(OUTPUT_DIR, f"{fig_name}_{cohort}_confusion_heatmaps.pdf")

    fig.savefig(png_path, dpi=600, bbox_inches="tight")
    fig.savefig(svg_path, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved {fig_name}: {png_path}")


def main() -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    set_publication_style()

    print("=" * 72)
    print("Generating FigS19-20 composite confusion heatmaps")
    print("=" * 72)

    all_results = load_target_results()
    for cohort, fig_name in COHORT_TO_FIG.items():
        draw_cohort_figure(all_results, cohort, fig_name)

    print(f"\nDone. Outputs saved in: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
