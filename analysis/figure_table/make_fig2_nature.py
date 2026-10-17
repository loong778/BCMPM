import argparse
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch
from scipy.stats import gaussian_kde, mannwhitneyu
from sklearn.metrics import auc, roc_curve



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
COHORT_TITLES = {
    "Train": "Training cohort",
    "Validation": "Internal validation cohort",
    "Test": "External test cohort",
}
TIME_MONTHS = [12, 24, 36, 48, 60]

COLORS = {
    "MPM-Only": "#8D8D8D",
    "MPM-Clinical": "#74ADD1",
    "Clinical-Only": "#4C78A8",
    "H&E-Only": "#F1A340",
    "H&E-Clinical": "#66C2A5",
    "H&E-MPM": "#5E8C61",
    "BiCA": "#8C6BB1",
    "BCMPM": "#7B1FA2",
}

FOREST_ORDER = [
    "MPM-Only",
    "MPM-Clinical",
    "Clinical-Only",
    "H&E-Only",
    "H&E-Clinical",
    "H&E-MPM",
    "BiCA",
]


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
        if all((root / modality_dir_name(m) / "results.npz").exists() for m in ORDER):
            return root
    searched = "\n".join(str(p.resolve()) for p in candidates)
    raise FileNotFoundError(f"Could not find all 9 modality result files. Searched:\n{searched}")


def load_results(result_root):
    results = {}
    for modality in ORDER:
        npz_path = result_root / modality_dir_name(modality) / "results.npz"
        data = np.load(npz_path, allow_pickle=True)
        results[modality] = {
            key: data[key].item() if data[key].ndim == 0 else data[key]
            for key in data.files
        }
    return results


def cindex_ci(result, cohort):
    mean = float(result["cindex"][cohort])
    boot = np.asarray(result["bootstrap_cindex"][cohort], dtype=float).ravel()
    if boot.size == 3 and boot[1] <= boot[0] <= boot[2]:
        return mean, float(boot[1]), float(boot[2])
    if boot.size > 3:
        return mean, float(np.nanpercentile(boot, 2.5)), float(np.nanpercentile(boot, 97.5))
    return mean, np.nan, np.nan


def km_censor_survival_at(time, event, query_times):
    time = np.asarray(time, dtype=float)
    censor_event = 1 - np.asarray(event, dtype=int)
    query_times = np.asarray(query_times, dtype=float)
    order = np.argsort(time)
    unique_times = np.unique(time[order])
    surv = 1.0
    surv_by_time = []
    for t in unique_times:
        at_risk = np.sum(time >= t)
        n_censor = np.sum((time == t) & (censor_event == 1))
        if at_risk > 0:
            surv *= 1.0 - n_censor / at_risk
        surv_by_time.append((t, surv))

    out = np.ones_like(query_times, dtype=float)
    times = np.array([x[0] for x in surv_by_time], dtype=float)
    survs = np.array([x[1] for x in surv_by_time], dtype=float)
    idx = np.searchsorted(times, query_times, side="right") - 1
    valid = idx >= 0
    out[valid] = survs[idx[valid]]
    return np.clip(out, 1e-8, 1.0)


def time_dependent_auc_ipcw(risk, time, event, t):
    risk = np.asarray(risk, dtype=float).reshape(-1)
    time = np.asarray(time, dtype=float).reshape(-1)
    event = np.asarray(event, dtype=int).reshape(-1)
    y_true = ((time <= t) & (event == 1)).astype(int)
    if y_true.min() == y_true.max():
        return np.nan

    g_i = km_censor_survival_at(time, event, np.minimum(time, t))
    g_t = float(km_censor_survival_at(time, event, np.array([t]))[0])
    weights = np.where(y_true == 1, 1.0 / g_i, 1.0 / max(g_t, 1e-8))
    fpr, tpr, _ = roc_curve(y_true, risk, sample_weight=weights)
    return float(auc(fpr, tpr))


def harrell_cindex_vectorized(risk, time, event):
    risk = np.asarray(risk, dtype=float).reshape(-1)
    time = np.asarray(time, dtype=float).reshape(-1)
    event = np.asarray(event, dtype=bool).reshape(-1)
    comparable = (time[:, None] < time[None, :]) & event[:, None]
    n_comp = int(np.sum(comparable))
    if n_comp == 0:
        return np.nan
    r_i = risk[:, None]
    r_j = risk[None, :]
    concordant = np.sum((r_i > r_j) & comparable)
    ties = np.sum((r_i == r_j) & comparable)
    return float((concordant + 0.5 * ties) / n_comp)


def paired_delta_cindex(results, ref_mod="BCMPM", n_boot=2000, seed=42):
    rng = np.random.default_rng(seed)
    ref = results[ref_mod]
    time = np.asarray(ref["time"]["Test"], dtype=float)
    event = np.asarray(ref["event"]["Test"], dtype=int)
    ref_risk = np.asarray(ref["risk"]["Test"], dtype=float)
    n = len(time)
    rows = []

    for modality in FOREST_ORDER:
        risk = np.asarray(results[modality]["risk"]["Test"], dtype=float)
        observed = harrell_cindex_vectorized(ref_risk, time, event) - harrell_cindex_vectorized(risk, time, event)
        deltas = []
        for _ in range(n_boot):
            idx = rng.choice(n, n, replace=True)
            d = harrell_cindex_vectorized(ref_risk[idx], time[idx], event[idx]) - harrell_cindex_vectorized(
                risk[idx], time[idx], event[idx]
            )
            if np.isfinite(d):
                deltas.append(d)
        deltas = np.asarray(deltas, dtype=float)
        rows.append(
            {
                "Comparison": f"vs. {modality}",
                "Delta_Cindex": observed,
                "CI_low": float(np.percentile(deltas, 2.5)),
                "CI_high": float(np.percentile(deltas, 97.5)),
                "P_value": float(np.mean(deltas <= 0)),
                "N_boot": int(deltas.size),
            }
        )
    return pd.DataFrame(rows)



def configure_style():
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 10,
            "axes.labelsize": 11,
            "axes.titlesize": 12,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 9,
            "axes.linewidth": 1.0,
            "xtick.major.width": 1.0,
            "ytick.major.width": 1.0,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

def add_aligned_panel_labels(fig, axes, labels, x_offset=0.035, y_offset=0.025):
    positions = [ax.get_position() for ax in axes]
    label_y = max(pos.y1 for pos in positions) + y_offset
    for pos, label in zip(positions, labels):
        fig.text(
            pos.x0 - x_offset,
            label_y,
            label,
            fontsize=14,
            fontweight="bold",
            va="top",
            ha="left",
        )


def add_aligned_panel_titles(fig, axes, titles, y_offset=0.026):
    positions = [ax.get_position() for ax in axes]
    title_y = max(pos.y1 for pos in positions) + y_offset
    for pos, title in zip(positions, titles):
        fig.text(
            (pos.x0 + pos.x1) / 2,
            title_y,
            title,
            fontsize=11,
            fontweight="bold",
            va="top",
            ha="center",
        )


def draw_cindex_bar(ax, results, cohort, label):
    x = np.arange(len(ORDER))
    for i, modality in enumerate(ORDER):
        mean, low, high = cindex_ci(results[modality], cohort)
        yerr = [[max(mean - low, 0) if np.isfinite(low) else 0], [max(high - mean, 0) if np.isfinite(high) else 0]]
        ax.bar(
            i,
            mean,
            width=0.68,
            color=COLORS[modality],
            edgecolor="#2F2F2F" if modality == "BCMPM" else "none",
            linewidth=0.55 if modality == "BCMPM" else 0,
            yerr=yerr,
            error_kw={"elinewidth": 0.65, "ecolor": "#333333", "capsize": 1.8, "capthick": 0.65},
        )
        ax.text(
            i,
            min(high if np.isfinite(high) else mean, 0.96) + 0.022,
            f"{mean:.3f}",
            ha="center",
            va="bottom",
            fontsize=7.5 if modality == "BCMPM" else 7.0,
            fontweight="bold",
            color=COLORS[modality] if modality == "BCMPM" else "black",
        )
    ax.set_title(COHORT_TITLES[cohort], fontweight="bold", pad=4)
    ax.set_ylabel("C-index")
    ax.set_ylim(0, 1.0)
    ax.set_xticks(x)
    ax.set_xticklabels(ORDER, rotation=45, ha="right", rotation_mode="anchor")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def draw_raincloud(ax, results):
    result = results["BCMPM"]
    non_color = "#2C7FB8"
    rec_color = "#E45756"
    baselines = np.arange(len(COHORTS))[::-1] * 1.62
    all_risk = np.concatenate([np.asarray(result["risk"][c], dtype=float) for c in COHORTS])
    pad = 0.12 * (all_risk.max() - all_risk.min())
    xmin, xmax = all_risk.min() - pad, all_risk.max() + pad
    xgrid = np.linspace(xmin, xmax, 300)

    rng = np.random.default_rng(42)
    for i, cohort in enumerate(COHORTS):
        y0 = baselines[i]
        risk = np.asarray(result["risk"][cohort], dtype=float)
        event = np.asarray(result["event"][cohort], dtype=int)
        groups = [(risk[event == 0], non_color), (risk[event == 1], rec_color)]
        for values, color in groups:
            if len(values) > 1 and np.std(values) > 0:
                dens = gaussian_kde(values)(xgrid)
                dens = dens / dens.max() * 0.55
                ax.fill_between(xgrid, y0, y0 + dens, color=color, alpha=0.78, linewidth=0)
                ax.plot(xgrid, y0 + dens, color=color, lw=0.55)

        box_y = [y0 - 0.28, y0 - 0.08]
        box = ax.boxplot(
            [groups[0][0], groups[1][0]],
            positions=box_y,
            vert=False,
            widths=0.18,
            patch_artist=True,
            showfliers=False,
            medianprops={"color": "black", "linewidth": 0.65},
            whiskerprops={"linewidth": 0.55, "color": "#666666"},
            capprops={"linewidth": 0.55, "color": "#666666"},
        )
        for patch, (_, color) in zip(box["boxes"], groups):
            patch.set_facecolor(color)
            patch.set_alpha(0.9)
            patch.set_edgecolor("#666666")
            patch.set_linewidth(0.45)

        for values, color in groups:
            y = rng.normal(y0 - 0.58, 0.035, len(values))
            ax.scatter(values, y, s=2.4, color=color, alpha=0.42, linewidths=0)

        if len(groups[0][0]) > 0 and len(groups[1][0]) > 0:
            p = mannwhitneyu(groups[0][0], groups[1][0], alternative="two-sided").pvalue
            p_text = "p < 0.0001" if p < 0.0001 else f"p = {p:.4f}"
            ax.text(xmax, y0 + 0.22, p_text, ha="right", va="center", fontsize=7.2)

    ax.set_xlabel("BCMPM risk score")
    ax.set_yticks(baselines)
    ax.set_yticklabels([COHORT_TITLES[c] for c in COHORTS])
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(-0.88, baselines[0] + 0.75)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(
        handles=[Patch(facecolor=non_color, label="Non-recurrence"), Patch(facecolor=rec_color, label="Recurrence")],
        title="Group",
        frameon=True,
        loc="upper left",
        bbox_to_anchor=(0.76, 1.03),
        fontsize=6.5,
        title_fontsize=6.8,
        # See README.md for project setup and required files.
        handlelength=1.0,
        handleheight=0.8,

        # See README.md for project setup and required files.
        labelspacing=0.20,
        handletextpad=0.45,
        borderpad=0.40,

        # See README.md for project setup and required files.
        borderaxespad=0.2,

        framealpha=0.95,
    )


def draw_auc_panel(ax, results):
    rows = []
    for modality in ORDER:
        risk = np.asarray(results[modality]["risk"]["Test"], dtype=float)
        time = np.asarray(results[modality]["time"]["Test"], dtype=float)
        event = np.asarray(results[modality]["event"]["Test"], dtype=int)
        aucs = []
        for month in TIME_MONTHS:
            stored = results[modality].get("td_roc_raw", {}).get(month, {}).get("Test")
            auc_val = float(stored[2]) if stored is not None else time_dependent_auc_ipcw(risk, time, event, month)
            aucs.append(auc_val)
            rows.append({"Modality": modality, "Time_month": month, "AUC": auc_val})
        ax.plot(
            TIME_MONTHS,
            aucs,
            color=COLORS[modality],
            marker="o",
            markersize=3.0 if modality == "BCMPM" else 2.4,
            linewidth=1.45 if modality == "BCMPM" else 0.95,
            label=modality,
            zorder=4 if modality == "BCMPM" else 2,
        )
        for x, y in zip(TIME_MONTHS, aucs):
            ax.text(
                x,
                y + 0.010,
                f"{y:.3f}",
                color=COLORS[modality],
                ha="center",
                va="bottom",
                fontsize=7.0 if modality == "BCMPM" else 6.4,
                fontweight="bold" if modality == "BCMPM" else "normal",
                zorder=5 if modality == "BCMPM" else 3,
            )
    ax.set_xlabel("Time (months)")
    ax.set_ylabel("AUC")
    ax.set_xticks(TIME_MONTHS)
    ax.set_ylim(0.45, 1.08)
    ax.set_yticks(np.arange(0.5, 1.01, 0.1))
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    leg = ax.legend(
        frameon=True,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.965),
        ncol=3,
        fontsize=7.0,
        handlelength=1.2,
        columnspacing=0.9,
        labelspacing=0.28,
        borderaxespad=0.25,
        handletextpad=0.35,
    )
    leg.get_frame().set_facecolor("white")
    leg.get_frame().set_alpha(0.82)
    leg.get_frame().set_edgecolor("none")
    return pd.DataFrame(rows)


def p_value_text(p):
    if p < 0.001:
        return "<0.001"
    return f"{p:.3f}"


def draw_forest(ax, delta_df):
    y = np.arange(len(delta_df))[::-1]
    color = COLORS["BCMPM"]
    for yi, (_, row) in zip(y, delta_df.iterrows()):
        x = row["Delta_Cindex"]
        xerr = [[x - row["CI_low"]], [row["CI_high"] - x]]
        ax.errorbar(
            x,
            yi,
            xerr=xerr,
            fmt="o",
            color=color,
            markersize=3.5,
            capsize=2.2,
            elinewidth=0.85,
            markeredgecolor="white",
            markeredgewidth=0.45,
        )

    ax.axvline(0, color="#333333", lw=0.7)
    ax.set_yticks(y)
    ax.set_yticklabels([])
    ax.set_xlabel("Delta C-index")
    ax.set_xlim(-0.55, 1.20)
    ax.set_xticks(np.arange(0.0, 0.6, 0.2))
    ax.set_ylim(-0.7, len(delta_df) + 0.55)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)

    header_y = len(delta_df) + 0.12
    left_col_x = -0.705
    ax.text(left_col_x, header_y, "Compared with BCMPM", ha="left", va="bottom", fontsize=7.7, fontweight="bold")
    ax.text(0.56, header_y, "Delta C-index (95% CI)", ha="left", va="bottom", fontsize=7.7, fontweight="bold")
    ax.text(1.24, header_y, "P value", ha="left", va="bottom", fontsize=7.7, fontweight="bold")
    for yi, (_, row) in zip(y, delta_df.iterrows()):
        ax.text(left_col_x, yi, row["Comparison"], ha="left", va="center", fontsize=7.8)
        delta_text = f"{row['Delta_Cindex']:.3f} ({row['CI_low']:.3f}-{row['CI_high']:.3f})"
        ax.text(0.56, yi, delta_text, ha="left", va="center", fontsize=7.8)
        ax.text(1.24, yi, p_value_text(row["P_value"]), ha="left", va="center", fontsize=7.8)


def build_figure(results, output_dir, n_boot=2000):
    configure_style()
    output_dir.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(13.4, 7.8), constrained_layout=False)
    gs = fig.add_gridspec(
        nrows=2,
        ncols=6,
        height_ratios=[1.0, 1.42],
        left=0.12,
        right=0.965,
        top=0.965,
        bottom=0.095,
        hspace=0.48,
        wspace=0.86,
    )

    axes_top = [fig.add_subplot(gs[0, 0:2]), fig.add_subplot(gs[0, 2:4]), fig.add_subplot(gs[0, 4:6])]
    for ax, cohort, label in zip(axes_top, COHORTS, ["a", "b", "c"]):
        draw_cindex_bar(ax, results, cohort, label)

    ax_d = fig.add_subplot(gs[1, 0:2])
    ax_e = fig.add_subplot(gs[1, 2:4])
    ax_f = fig.add_subplot(gs[1, 4:6])
    draw_raincloud(ax_d, results)
    auc_df = draw_auc_panel(ax_e, results)
    delta_df = paired_delta_cindex(results, n_boot=n_boot)
    draw_forest(ax_f, delta_df)
    add_aligned_panel_labels(fig, axes_top, ["a", "b", "c"])
    add_aligned_panel_labels(fig, [ax_d, ax_e, ax_f], ["d", "e", "f"])
    add_aligned_panel_titles(
        fig,
        [ax_d, ax_e, ax_f],
        [
            "BCMPM risk score distribution",
            "Time-dependent AUC (external test cohort)",
            "Statistical comparison (external test cohort)",
        ],
    )

    for ax in fig.axes:
        ax.grid(False)

    png_path = output_dir / "Fig2_a-f_Nature_600dpi.png"
    svg_path = output_dir / "Fig2_a-f_Nature.svg"
    fig.savefig(png_path, dpi=600)
    fig.savefig(svg_path)
    plt.close(fig)

    cindex_rows = []
    for modality in ORDER:
        for cohort in COHORTS:
            mean, low, high = cindex_ci(results[modality], cohort)
            cindex_rows.append(
                {
                    "Modality": modality,
                    "Cohort": COHORT_TITLES[cohort],
                    "C_index": mean,
                    "CI_low": low,
                    "CI_high": high,
                }
            )
    pd.DataFrame(cindex_rows).to_csv(output_dir / "Fig2a-c_cindex_summary.csv", index=False, float_format="%.4f")
    auc_df.to_csv(output_dir / "Fig2e_external_time_dependent_auc.csv", index=False, float_format="%.4f")
    delta_df.to_csv(output_dir / "Fig2f_delta_cindex_paired_bootstrap.csv", index=False, float_format="%.4f")
    return png_path, svg_path, auc_df, delta_df


def main():
    parser = argparse.ArgumentParser(description="Generate publication-ready Fig2a-f for the  BCMPM modalities.")
    parser.add_argument("--result-root", default=None, help="Directory containing modality subfolders with results.npz.")
    parser.add_argument("--output-dir", default="./xxx/xx/Fig2_Nature09171", help="Directory for Fig2 outputs.")
    parser.add_argument("--n-boot", type=int, default=2000, help="Paired bootstrap iterations for panel f.")
    args = parser.parse_args()

    result_root = resolve_result_root(args.result_root)
    results = load_results(result_root)
    output_dir = Path(args.output_dir).resolve()
    png_path, svg_path, _, delta_df = build_figure(results, output_dir, n_boot=args.n_boot)

    print(f"Result root: {result_root}")
    print(f"Saved PNG: {png_path}")
    print(f"Saved SVG: {svg_path}")
    print("Panel f paired bootstrap summary:")
    print(delta_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))


if __name__ == "__main__":
    main()



