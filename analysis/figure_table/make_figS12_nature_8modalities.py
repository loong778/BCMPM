"""Generate FigS12 KM panels for 8 modalities.

This script is self-contained and does not import make_figS12_nature.py.

Default output is one 8x4 figure. Use --layout 4x4 to save two 4x4 figures,
or --layout both to save both formats.
"""

import argparse
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import MultipleLocator
from scipy.interpolate import interp1d
from scipy.stats import chi2
from sklearn.metrics import auc, roc_curve


ORDER = [
    "MPM-Only",
    "MPM-Clinical",
    "Clinical-Only",
    "H&E-Only",
    "H&E-Clinical",
    "H&E-MPM",
    # "H&E-MPM-TACS",
    "BiCA",
    "BCMPM",
]



def modality_dir_name(modality):
    return modality.lower().replace("-", "_")


DATASETS = ["Train", "Validation", "Test"]
DATASET_TITLES = [
    "Training cohort (binary)",
    "Internal validation cohort (binary)",
    "External test cohort (binary)",
    "Total (quartile)",
]
COLOR_LOW = "#1f77b4"
COLOR_HIGH = "#d62728"
COLOR_QUARTILES = ["#2ca02c", "#ffcc00", "#ff7f0e", "#d62728"]
QUARTILE_LABELS = ["Q1 (Lowest)", "Q2", "Q3", "Q4 (Highest)"]
TARGET_TICKS = [0, 30, 60, 90]
X_MAX = 90


def configure_style():
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 10,
            "axes.linewidth": 1.0,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


def km_survival_curve(time, event):
    time = np.asarray(time, dtype=float)
    event = np.asarray(event, dtype=int)
    valid = np.isfinite(time) & np.isfinite(event)
    time, event = time[valid], event[valid]
    event_times = np.sort(np.unique(time[event == 1]))

    x = [0.0]
    y = [1.0]
    surv = 1.0
    for t in event_times:
        at_risk = np.sum(time >= t)
        n_event = np.sum((time == t) & (event == 1))
        if at_risk > 0:
            surv *= 1.0 - n_event / at_risk
        x.append(float(t))
        y.append(float(surv))
    return np.asarray(x), np.asarray(y)


def km_censor_survival_at(time, event, query_times):
    censor_event = (np.asarray(event, dtype=int) == 0).astype(int)
    x, y = km_survival_curve(time, censor_event)
    if len(x) == 1:
        return np.ones_like(query_times, dtype=float)
    fn = interp1d(x, y, kind="previous", bounds_error=False, fill_value=(1.0, y[-1]))
    return np.asarray(fn(query_times), dtype=float)


def time_dependent_roc_ipcw(risk, time, event, t):
    risk = np.asarray(risk, dtype=float)
    time = np.asarray(time, dtype=float)
    event = np.asarray(event, dtype=int)
    g = km_censor_survival_at(time, event, np.minimum(time, t))
    g_t = float(km_censor_survival_at(time, event, np.asarray([t], dtype=float))[0])
    y_true = (time <= t) & (event == 1)
    weights = np.where(y_true, 1.0 / np.maximum(g, 1e-8), 1.0 / np.maximum(g_t, 1e-8))
    fpr, tpr, thresholds = roc_curve(y_true, risk, sample_weight=weights)
    auc_val = auc(fpr, tpr)
    return auc_val, fpr, tpr, thresholds


def training_ipcw_youden_cutoff(result, time_cut=60):
    risk = result["risk"]["Train"]
    time = result["time"]["Train"]
    event = result["event"]["Train"]
    _, fpr, tpr, thresholds = time_dependent_roc_ipcw(risk, time, event, time_cut)
    return float(thresholds[np.argmax(tpr - fpr)])


def logrank_pvalue(time, event, groups):
    time = np.asarray(time, dtype=float)
    event = np.asarray(event, dtype=int)
    groups = np.asarray(groups)
    valid = np.isfinite(time) & np.isfinite(event) & pd.notna(groups)
    time, event, groups = time[valid], event[valid], groups[valid]
    group_values = np.asarray(pd.unique(groups))
    k = len(group_values)
    if k < 2 or np.sum(event == 1) == 0:
        return np.nan

    group_index = {value: idx for idx, value in enumerate(group_values)}
    g = np.asarray([group_index[value] for value in groups], dtype=int)
    obs = np.zeros(k, dtype=float)
    exp = np.zeros(k, dtype=float)
    var = np.zeros((k, k), dtype=float)

    for t in np.sort(np.unique(time[event == 1])):
        at_risk = time >= t
        events_at_t = (time == t) & (event == 1)
        n = float(np.sum(at_risk))
        d = float(np.sum(events_at_t))
        if n <= 1 or d <= 0:
            continue
        n_g = np.asarray([np.sum(at_risk & (g == i)) for i in range(k)], dtype=float)
        d_g = np.asarray([np.sum(events_at_t & (g == i)) for i in range(k)], dtype=float)
        obs += d_g
        exp += d * n_g / n
        factor = d * (n - d) / (n * n * (n - 1.0))
        var += factor * (np.diag(n_g * (n - n_g)) - np.outer(n_g, n_g) + np.diag(n_g * n_g))

    oe = obs[:-1] - exp[:-1]
    cov = var[:-1, :-1]
    try:
        stat = float(oe @ np.linalg.pinv(cov) @ oe)
    except Exception:
        return np.nan
    df = int(np.linalg.matrix_rank(cov))
    if df < 1 or not np.isfinite(stat):
        return np.nan
    return float(chi2.sf(stat, df))


def cox_hr_binary_efron(time, event, high_risk, max_iter=100, tol=1e-9):
    time = np.asarray(time, dtype=float)
    event = np.asarray(event, dtype=int)
    x = np.asarray(high_risk, dtype=float)
    valid = np.isfinite(time) & np.isfinite(event) & np.isfinite(x)
    time, event, x = time[valid], event[valid], x[valid]
    if len(np.unique(x)) < 2 or np.sum(event == 1) == 0:
        return np.nan, np.nan, np.nan

    beta = 0.0
    info = np.nan
    event_times = np.unique(time[event == 1])
    for _ in range(max_iter):
        eta = np.clip(beta * x, -50, 50)
        exp_eta = np.exp(eta)
        score = 0.0
        info = 0.0

        for t in event_times:
            event_mask = (time == t) & (event == 1)
            risk_mask = time >= t
            d = int(np.sum(event_mask))
            if d == 0:
                continue

            x_events = x[event_mask]
            e_events = exp_eta[event_mask]
            s0 = float(np.sum(exp_eta[risk_mask]))
            s1 = float(np.sum(exp_eta[risk_mask] * x[risk_mask]))
            s2 = float(np.sum(exp_eta[risk_mask] * x[risk_mask] * x[risk_mask]))
            e0 = float(np.sum(e_events))
            e1 = float(np.sum(e_events * x_events))
            e2 = float(np.sum(e_events * x_events * x_events))

            score += float(np.sum(x_events))
            for j in range(d):
                frac = j / d
                denom = s0 - frac * e0
                if denom <= 0:
                    continue
                mean = (s1 - frac * e1) / denom
                second = (s2 - frac * e2) / denom
                score -= mean
                info += second - mean * mean

        if info <= 0 or not np.isfinite(info):
            return np.nan, np.nan, np.nan
        step = score / info
        beta += step
        if abs(step) < tol:
            break

    se = np.sqrt(1.0 / info)
    hr = np.exp(beta)
    low = np.exp(beta - 1.96 * se)
    high = np.exp(beta + 1.96 * se)
    return float(hr), float(low), float(high)


def p_text(p):
    if not np.isfinite(p):
        return "P=NA"
    return "P<0.0001" if p < 0.0001 else f"P={p:.4f}"


def plot_km(ax, time, event, mask, color, label):
    if np.sum(mask) == 0:
        return
    x, y = km_survival_curve(np.asarray(time)[mask], np.asarray(event)[mask])
    ax.step(x, y, where="post", color=color, linewidth=2.0, label=label)


def setup_km_axis(ax, col_idx):
    ax.axvline(x=60, color="#2ca02c", linestyle="--", linewidth=1.5)
    ax.set_xlabel("Time (month)", fontsize=10.5, labelpad=8)
    ax.set_ylabel("DFS (%)" if col_idx == 0 else "", fontsize=10.5, labelpad=8)
    ax.set_xlim(0, X_MAX)
    ax.set_ylim(0, 1.05)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, pos: f"{int(y * 100)}"))
    ax.xaxis.set_major_locator(MultipleLocator(30))
    ax.grid(False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(direction="out", top=False, right=False, labelsize=10, width=1.0)


def add_binary_at_risk_table(ax, time, high_mask, low_mask):
    x_pos = np.asarray(TARGET_TICKS, dtype=float) / X_MAX
    for xi, t in zip(x_pos, TARGET_TICKS):
        nh = int(np.sum(np.asarray(time)[high_mask] >= t))
        nl = int(np.sum(np.asarray(time)[low_mask] >= t))
        ax.text(xi, -0.13, str(nh), ha="center", transform=ax.transAxes,
                color=COLOR_HIGH, fontsize=8.5, weight="bold")
        ax.text(xi, -0.22, str(nl), ha="center", transform=ax.transAxes,
                color=COLOR_LOW, fontsize=8.5, weight="bold")


def add_quartile_at_risk_table(ax, total_time, quartile_masks):
    x_pos = np.asarray(TARGET_TICKS, dtype=float) / X_MAX
    y_list = [-0.13, -0.17, -0.21, -0.25]
    for idx_q, mask in enumerate(quartile_masks):
        sub_t = total_time[mask]
        for xi, t in zip(x_pos, TARGET_TICKS):
            ax.text(xi, y_list[idx_q], str(int(np.sum(sub_t >= t))), ha="center",
                    transform=ax.transAxes, color=COLOR_QUARTILES[idx_q], fontsize=8, weight="bold")


def draw_modality_row(axes, modality, result, cutoff):
    total_risk = np.concatenate([result["risk"][d] for d in DATASETS])
    total_time = np.concatenate([result["time"][d] for d in DATASETS])
    total_event = np.concatenate([result["event"][d] for d in DATASETS])

    q25, q50, q75 = np.percentile(total_risk, [25, 50, 75])
    quartile_masks = [
        total_risk < q25,
        (total_risk >= q25) & (total_risk < q50),
        (total_risk >= q50) & (total_risk < q75),
        total_risk >= q75,
    ]
    quartile_groups = np.zeros(len(total_risk), dtype=int)
    for idx, mask in enumerate(quartile_masks):
        quartile_groups[mask] = idx
    p_total = logrank_pvalue(total_time, total_event, quartile_groups)

    for col_idx, (dset, title) in enumerate(zip(DATASETS, DATASET_TITLES[:3])):
        ax = axes[col_idx]
        risk = np.asarray(result["risk"][dset], dtype=float)
        time = np.asarray(result["time"][dset], dtype=float)
        event = np.asarray(result["event"][dset], dtype=int)

        high_mask_plot = risk > cutoff
        low_mask_plot = risk <= cutoff
        n_high = int(np.sum(high_mask_plot))
        n_low = int(np.sum(low_mask_plot))
        plot_km(ax, time, event, high_mask_plot, COLOR_HIGH, f"High risk (n={n_high})")
        plot_km(ax, time, event, low_mask_plot, COLOR_LOW, f"Low risk (n={n_low})")

        auc_5year = result["td_roc_raw"][60][dset][2]
        high_for_hr = (risk >= cutoff).astype(int)
        hr, hr_low, hr_high = cox_hr_binary_efron(time, event, high_for_hr)
        p_val = logrank_pvalue(time, event, high_for_hr)

        ax.set_title(f"{title}\nAUC={auc_5year:.3f}", fontsize=12, fontweight="bold", pad=10)
        if np.isfinite(hr):
            stats_text = f"{p_text(p_val)}\nHR={hr:.2f}\n({hr_low:.2f}-{hr_high:.2f})"
        else:
            stats_text = f"{p_text(p_val)}\nHR=NA"
        ax.text(
            0.03, 0.28, stats_text, transform=ax.transAxes,
            bbox=dict(facecolor="white", alpha=0.9, edgecolor="gray", pad=2.5),
            fontsize=9, verticalalignment="top", linespacing=1.2,
        )
        setup_km_axis(ax, col_idx)
        ax.legend(loc="lower left", fontsize=9, framealpha=0.9, edgecolor="gray", borderpad=0.6)
        add_binary_at_risk_table(ax, time, high_mask_plot, low_mask_plot)

    ax = axes[3]
    for mask, label, color in zip(quartile_masks, QUARTILE_LABELS, COLOR_QUARTILES):
        n = int(np.sum(mask))
        x, y = km_survival_curve(total_time[mask], total_event[mask])
        ax.step(x, y, where="post", color=color, linewidth=2.0, label=f"{label} (n={n})")

    ax.set_title(DATASET_TITLES[3], fontsize=12, fontweight="bold", pad=10)
    ax.text(
        0.05, 0.28, p_text(p_total), transform=ax.transAxes,
        bbox=dict(facecolor="white", alpha=0.9, edgecolor="gray", pad=2.5),
        fontsize=10, fontweight="bold", verticalalignment="top",
    )
    setup_km_axis(ax, 3)
    ax.legend(loc="lower left", fontsize=8.5, framealpha=0.9, edgecolor="gray", borderpad=0.6)
    add_quartile_at_risk_table(ax, total_time, quartile_masks)

def load_results(result_root):
    results = {}
    for modality in ORDER:
        path = Path(result_root) / modality_dir_name(modality) / "results.npz"
        if not path.is_file():
            raise FileNotFoundError(f"Missing results for {modality}: {path.resolve()}")
        with np.load(path, allow_pickle=True) as data:
            results[modality] = {
                key: data[key].item() if data[key].ndim == 0 else data[key]
                for key in data.files
            }
    return results


def enlarge_row(axes, font_scale):
    for col, ax in enumerate(axes):
        ax.title.set_fontsize(12 * font_scale)
        ax.xaxis.label.set_fontsize(10.5 * font_scale)
        ax.yaxis.label.set_fontsize(10.5 * font_scale)
        ax.tick_params(labelsize=10 * font_scale)
        ax.xaxis.set_label_coords(0.5, -0.16)

        risk_rows = [-0.13, -0.22] if col < 3 else [-0.13, -0.17, -0.21, -0.25]
        for text in ax.texts:
            text.set_fontsize(text.get_fontsize() * font_scale)
            x, y = text.get_position()
            if y < 0:
                row = min(range(len(risk_rows)), key=lambda i: abs(y - risk_rows[i]))
                text.set_position((x, -0.29 - 0.085 * row))
                text.set_va("center")
            else:
                text.set_position((x, 0.49 if col < 3 else 0.53))

        legend_size = (9 if col < 3 else 8.5) * font_scale
        ax.legend(
            loc="lower left",
            fontsize=legend_size,
            framealpha=0.9,
            edgecolor="gray",
            borderpad=0.6,
        )


def draw_rows(axes, modalities, results, cutoffs, font_scale):
    for row, modality in enumerate(modalities):
        draw_modality_row(axes[row], modality, results[modality], cutoffs[modality])
        enlarge_row(axes[row], font_scale)
        axes[row, 0].text(
            -0.32,
            0.5,
            modality,
            transform=axes[row, 0].transAxes,
            rotation=90,
            ha="center",
            va="center",
            fontsize=17 * font_scale,
            fontweight="bold",
        )


def make_8x4_figure(results, cutoffs, font_scale=1.25):
    configure_style()
    mpl.rcParams["font.size"] = 10 * font_scale
    fig, axes = plt.subplots(8, 4, figsize=(30, 49))
    fig.subplots_adjust(
        left=0.09,
        right=0.98,
        top=0.975,
        bottom=0.04,
        wspace=0.30,
        hspace=1.05,
    )
    draw_rows(axes, ORDER, results, cutoffs, font_scale)
    return fig


def make_4x4_figure(modalities, results, cutoffs, font_scale=1.25):
    configure_style()
    mpl.rcParams["font.size"] = 10 * font_scale
    fig, axes = plt.subplots(4, 4, figsize=(30, 25))
    fig.subplots_adjust(
        left=0.09,
        right=0.98,
        top=0.96,
        bottom=0.065,
        wspace=0.30,
        hspace=1.05,
    )
    draw_rows(axes, modalities, results, cutoffs, font_scale)
    return fig


def save_figure(fig, output_dir, stem, dpi):
    saved = []
    for suffix in ["png", "pdf", "svg"]:
        path = output_dir / f"{stem}.{suffix}"
        fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor="white")
        saved.append(path)
        print(f"Saved: {path}")
    return saved


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--result-root",
        default="./xxx/xx",
        help="Root containing modality result folders with results.npz.",
    )
    parser.add_argument(
        "--output-dir",
        default="./xxx/xx/FigS12_Nature_8modalities",
    )
    parser.add_argument("--time-cut", type=float, default=60.0)
    parser.add_argument(
        "--layout",
        choices=["8x4", "4x4", "both"],
        default="4x4",
        help="8x4 saves one figure; 4x4 saves two figures; both saves both layouts.",
    )
    parser.add_argument(
        "--font-scale",
        type=float,
        default=1.25,
        help="Multiplier for original font sizes.",
    )
    parser.add_argument("--dpi", type=int, default=600, help="PNG resolution; PDF/SVG remain vector.")
    args = parser.parse_args()
    if args.font_scale <= 0 or args.dpi <= 0:
        parser.error("--font-scale and --dpi must be positive")
    return args


def main():
    args = parse_args()
    results = load_results(args.result_root)
    cutoffs = {
        name: training_ipcw_youden_cutoff(results[name], args.time_cut)
        for name in ORDER
    }

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.layout in {"8x4", "both"}:
        fig = make_8x4_figure(results, cutoffs, args.font_scale)
        try:
            save_figure(fig, output_dir, "FigS12_KM_8modalities_8x4", args.dpi)
        finally:
            plt.close(fig)

    if args.layout in {"4x4", "both"}:
        for idx, start in enumerate(range(0, len(ORDER), 4), start=1):
            modalities = ORDER[start:start + 4]
            fig = make_4x4_figure(modalities, results, cutoffs, args.font_scale)
            try:
                save_figure(fig, output_dir, f"FigS12_KM_8modalities_4x4_part{idx}", args.dpi)
            finally:
                plt.close(fig)

    pd.DataFrame(
        [
            {"Modality": name, "Training_Set_IPCW_Youden_Cutoff": cutoffs[name]}
            for name in ORDER
        ]
    ).to_csv(output_dir / "FigS12_training_ipcw_youden_cutoffs.csv", index=False)


if __name__ == "__main__":
    main()
