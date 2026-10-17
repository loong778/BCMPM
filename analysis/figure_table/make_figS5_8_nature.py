import argparse
import os
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.gridspec import GridSpec
from scipy.stats import norm
from scipy.optimize import minimize

import warnings
warnings.filterwarnings('ignore')

RISK_MODALITY = "bcmpm"
RISK_COL = f"{RISK_MODALITY}_risk_score"
HIGH_RISK_COL = f"{RISK_MODALITY}_high_risk"
TIME_COL = "DFS"
EVENT_COL = "status"

# Required input: see README.md for the required file and directory layout.

DEFAULT_INPUT_DIR = Path('./xxx/xx')
# Example output path; generated files are written here.
DEFAULT_OUTPUT_DIR = Path('./xxx/xx')

DATASETS = [
    ("Training", "train_all_modalities_risk.csv", "FigS5"),
    ("Validation", "validation_all_modalities_risk.csv", "FigS6"),
    ("Test", "test_all_modalities_risk.csv", "FigS7"),
    ("Total", "total_all_modalities_risk.csv", "FigS8"),
]

SUBGROUP_STRUCTURE = [
    {"group_name": "Overall", "sub_items": ["Overall"], "col_name": None},
    {"group_name": "Age", "sub_items": ["<= 50", "> 50"], "col_name": "Age_group"},
    {"group_name": "Tumor size", "sub_items": ["<= 2 cm", "2-5 cm", "> 5 cm"], "col_name": "tumor size"},
    {
        "group_name": "Molecular subtype",
        "sub_items": ["Luminal A", "Luminal B", "HER2-enriched", "Triple-negative"],
        "col_name": "molecular subtypes",
    },
    {"group_name": "Nodal status", "sub_items": ["0", "1-3", ">=4"], "col_name": "lymph node metastasis"},
    {"group_name": "Clinical stage", "sub_items": ["I", "II", "III"], "col_name": "clinical stage"},
    {"group_name": "Histological grade", "sub_items": ["G1", "G2", "G3"], "col_name": "histological grade"},
]

TUMOR_SIZE_MAP = {1: "<= 2 cm", 2: "2-5 cm", 3: "> 5 cm"}
NODE_MAP = {0: "0", 1: "1-3", 2: ">=4"}
STAGE_MAP = {1: "I", 2: "II", 3: "III"}
GRADE_MAP = {1: "G1", 2: "G2", 3: "G3"}
SUBTYPE_MAP = {1: "Luminal A", 2: "Luminal B", 3: "HER2-enriched", 4: "Triple-negative"}

DISPLAY_LABELS = {
    "<= 50": "<= 50",
    "<= 2 cm": "<= 2 cm",
    "2-5 cm": "2-5 cm",
    ">=4": ">=4",
}

PURPLE = "#5A3180"


def configure_style():
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 13,
            "axes.labelsize": 14,
            "axes.titlesize": 14,
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
            "axes.linewidth": 0.9,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


def prepare_dataframe(df):
    required = [RISK_COL, HIGH_RISK_COL, TIME_COL, EVENT_COL, "age1"]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise KeyError(f"Input CSV is missing required columns: {missing}")

    out = df.copy()
    out["risk_score"] = pd.to_numeric(out[RISK_COL], errors="coerce")
    out["high_risk"] = pd.to_numeric(out[HIGH_RISK_COL], errors="coerce").astype("Int64")
    out[TIME_COL] = pd.to_numeric(out[TIME_COL], errors="coerce")
    out[EVENT_COL] = pd.to_numeric(out[EVENT_COL], errors="coerce").astype("Int64")

    out["Age_group"] = pd.cut(
        pd.to_numeric(out["age1"], errors="coerce"),
        bins=[0, 50, 200],
        labels=["<= 50", "> 50"],
        include_lowest=True,
    )
    out["tumor size"] = out["tumor size"].map(TUMOR_SIZE_MAP)
    out["lymph node metastasis"] = out["lymph node metastasis"].map(NODE_MAP)
    out["clinical stage"] = out["clinical stage"].map(STAGE_MAP)
    out["histological grade"] = out["histological grade"].map(GRADE_MAP)
    out["molecular subtypes"] = out["molecular subtypes"].map(SUBTYPE_MAP)
    return out


class FenwickTree:
    def __init__(self, n):
        self.n = n
        self.tree = np.zeros(n + 1, dtype=float)

    def add(self, idx, value):
        i = idx + 1
        while i <= self.n:
            self.tree[i] += value
            i += i & -i

    def sum(self, idx):
        if idx < 0:
            return 0.0
        i = idx + 1
        total = 0.0
        while i > 0:
            total += self.tree[i]
            i -= i & -i
        return total

    def range_sum(self, left, right):
        if right < left:
            return 0.0
        return self.sum(right) - self.sum(left - 1)


def harrell_cindex(risk, time, event):
    risk = np.asarray(risk, dtype=float)
    time = np.asarray(time, dtype=float)
    event = np.asarray(event, dtype=int)
    valid = np.isfinite(risk) & np.isfinite(time) & np.isfinite(event)
    risk, time, event = risk[valid], time[valid], event[valid]
    if len(risk) < 2 or np.sum(event == 1) == 0:
        return np.nan

    risk_values = np.sort(np.unique(risk))
    risk_rank = np.searchsorted(risk_values, risk)
    order = np.argsort(-time, kind="mergesort")
    risk_rank = risk_rank[order]
    time_sorted = time[order]
    event_sorted = event[order]

    tree = FenwickTree(len(risk_values))
    pool_count = 0.0
    concordant = 0.0
    comparable = 0.0
    start = 0
    while start < len(time_sorted):
        end = start + 1
        while end < len(time_sorted) and time_sorted[end] == time_sorted[start]:
            end += 1

        for idx in range(start, end):
            if event_sorted[idx] != 1:
                continue
            rank = risk_rank[idx]
            comparable += pool_count
            concordant += tree.sum(rank - 1) + 0.5 * tree.range_sum(rank, rank)

        for idx in range(start, end):
            tree.add(risk_rank[idx], 1.0)
            pool_count += 1.0
        start = end

    if comparable <= 0:
        return np.nan
    return float(concordant / comparable)


def calc_cindex_ci(risk, time, event, n_bootstrap=1000, random_state=2025):
    cindex = harrell_cindex(risk, time, event)
    risk = np.asarray(risk, dtype=float)
    time = np.asarray(time, dtype=float)
    event = np.asarray(event, dtype=int)
    n = len(risk)
    if n < 3 or not np.isfinite(cindex):
        return cindex, np.nan, np.nan

    rng = np.random.RandomState(random_state)
    boot_scores = []
    for _ in range(n_bootstrap):
        idx = rng.choice(np.arange(n), size=n, replace=True)
        score = harrell_cindex(risk[idx], time[idx], event[idx])
        if np.isfinite(score):
            boot_scores.append(score)

    if len(boot_scores) < 20:
        return cindex, np.nan, np.nan
    low, high = np.percentile(boot_scores, [2.5, 97.5])
    return float(cindex), float(low), float(high)


def _cox_loglik_score_info(beta, x, time, event):
    eta = np.clip(x @ beta, -50, 50)
    exp_eta = np.exp(eta)
    loglik = 0.0
    score = np.zeros_like(beta)
    info = np.zeros((len(beta), len(beta)), dtype=float)

    for t in np.unique(time[event == 1]):
        event_mask = (time == t) & (event == 1)
        risk_mask = time >= t
        d = int(np.sum(event_mask))
        if d == 0:
            continue

        x_events = x[event_mask]
        e_events = exp_eta[event_mask]
        x_risk = x[risk_mask]
        e_risk = exp_eta[risk_mask]

        s0 = float(np.sum(e_risk))
        s1 = np.sum(e_risk[:, None] * x_risk, axis=0)
        s2 = x_risk.T @ (e_risk[:, None] * x_risk)
        e0 = float(np.sum(e_events))
        e1 = np.sum(e_events[:, None] * x_events, axis=0)
        e2 = x_events.T @ (e_events[:, None] * x_events)

        loglik += float(np.sum(eta[event_mask]))
        score += np.sum(x_events, axis=0)
        for j in range(d):
            frac = j / d
            denom = s0 - frac * e0
            if denom <= 0:
                return -np.inf, score, info
            loglik -= np.log(denom)
            mean = (s1 - frac * e1) / denom
            second = (s2 - frac * e2) / denom
            score -= mean
            info += second - np.outer(mean, mean)

    return loglik, score, info


def _cox_efron_fit(x, time, event, max_iter=100, tol=1e-9):
    x = np.asarray(x, dtype=float)
    time = np.asarray(time, dtype=float)
    event = np.asarray(event, dtype=int)
    if x.ndim == 1:
        x = x.reshape(-1, 1)

    valid = np.isfinite(time) & np.isfinite(event) & np.all(np.isfinite(x), axis=1)
    x, time, event = x[valid], time[valid], event[valid]
    if len(time) < 3 or np.sum(event == 1) == 0 or x.shape[1] == 0:
        return None
    if np.any(np.nanstd(x, axis=0) <= 1e-12):
        return None

    def objective(beta):
        loglik, _, _ = _cox_loglik_score_info(beta, x, time, event)
        if not np.isfinite(loglik):
            return 1e50
        return -loglik

    def gradient(beta):
        _, score, _ = _cox_loglik_score_info(beta, x, time, event)
        return -score

    opt = minimize(
        objective,
        np.zeros(x.shape[1], dtype=float),
        jac=gradient,
        method="BFGS",
        options={"maxiter": max_iter, "gtol": tol},
    )
    if not np.all(np.isfinite(opt.x)):
        return None
    beta = opt.x
    _, _, info = _cox_loglik_score_info(beta, x, time, event)
    if not np.all(np.isfinite(info)):
        return None

    try:
        cov = np.linalg.inv(info + 1e-8 * np.eye(len(beta)))
    except np.linalg.LinAlgError:
        cov = np.linalg.pinv(info)
    se = np.sqrt(np.clip(np.diag(cov), 0, np.inf))
    z = np.divide(beta, se, out=np.full_like(beta, np.nan), where=se > 0)
    p = 2 * norm.sf(np.abs(z))
    return beta, se, p


def get_hr_ci(df, risk_col="high_risk", time_col=TIME_COL, event_col=EVENT_COL):
    dat = df[[time_col, event_col, risk_col]].dropna().copy()
    if dat.empty or dat[risk_col].nunique() < 2:
        return np.nan, np.nan, np.nan, np.nan
    fit = _cox_efron_fit(dat[risk_col].values, dat[time_col].values, dat[event_col].values)
    if fit is None:
        return np.nan, np.nan, np.nan, np.nan
    beta, se, p = fit
    if not np.isfinite(beta[0]) or not np.isfinite(se[0]) or se[0] <= 0:
        return np.nan, np.nan, np.nan, np.nan
    hr = np.exp(beta[0])
    low = np.exp(beta[0] - 1.96 * se[0])
    high = np.exp(beta[0] + 1.96 * se[0])
    return float(hr), float(low), float(high), float(p[0])


def interaction_pvalue(df, highrisk_col, subgroup_col, time_col, event_col):
    dat = df[[time_col, event_col, highrisk_col, subgroup_col]].dropna().copy()
    if dat.empty or dat[highrisk_col].nunique() < 2 or dat[subgroup_col].nunique() < 2:
        return np.nan

    dummies = pd.get_dummies(dat[subgroup_col].astype(str), prefix=subgroup_col, drop_first=True, dtype=float)
    if dummies.shape[1] == 0:
        return np.nan
    high = dat[highrisk_col].astype(float).to_numpy()
    dummy_values = dummies.to_numpy(dtype=float)
    interactions = dummy_values * high[:, None]
    x = np.column_stack([high, dummy_values, interactions])

    keep = np.nanstd(x, axis=0) > 1e-12
    x = x[:, keep]
    interaction_start = 1 + dummy_values.shape[1]
    kept_original_idx = np.arange(1 + dummy_values.shape[1] + interactions.shape[1])[keep]
    interaction_positions = np.where(kept_original_idx >= interaction_start)[0]
    if len(interaction_positions) == 0:
        return np.nan

    fit = _cox_efron_fit(x, dat[time_col].values, dat[event_col].values, max_iter=80)
    if fit is None:
        return np.nan
    _, _, p = fit
    pvals = p[interaction_positions]
    pvals = pvals[np.isfinite(pvals)]
    if len(pvals) == 0:
        return np.nan
    return float(np.max(pvals))


def format_pvalue(p):
    if not np.isfinite(p):
        return "NA"
    if p < 0.001:
        return "<0.001"
    return f"{p:.3f}"


def format_ci_text(center, low, high, digits=3):
    if not (np.isfinite(center) and np.isfinite(low) and np.isfinite(high)):
        return "NA"
    return f"{center:.{digits}f} ({low:.{digits}f}-{high:.{digits}f})"


def format_hr_text(hr, low, high):
    if not (np.isfinite(hr) and np.isfinite(low) and np.isfinite(high)):
        return "NA"
    return f"{hr:.2f} ({low:.2f}-{high:.2f})"


def display_label(label):
    return DISPLAY_LABELS.get(label, label)


def sparse_or_invalid(n_patients, n_events, min_n, min_events):
    return n_patients < min_n or n_events < min_events


def construct_forest_table(df_plot, n_bootstrap=1000, min_n=10, min_events=5):
    plot_rows = []
    df_ov = df_plot.dropna(subset=["risk_score", "high_risk", TIME_COL, EVENT_COL]).copy()
    n_total = len(df_ov)
    n_events = int(df_ov[EVENT_COL].sum())

    cindex, c_low, c_high = calc_cindex_ci(
        df_ov["risk_score"].values,
        df_ov[TIME_COL].values,
        df_ov[EVENT_COL].values,
        n_bootstrap=n_bootstrap,
    )
    if sparse_or_invalid(n_total, n_events, min_n, min_events):
        hr = lower = upper = np.nan
    else:
        hr, lower, upper, _ = get_hr_ci(df_ov, "high_risk", TIME_COL, EVENT_COL)
        if not (np.isfinite(hr) and np.isfinite(lower) and np.isfinite(upper)):
            hr = lower = upper = np.nan

    hr_text = format_hr_text(hr, lower, upper)
    c_text = "NA" if hr_text == "NA" else format_ci_text(cindex, c_low, c_high, digits=3)
    plot_rows.append(
        {
            "label": "Overall",
            "n": n_total,
            "events": n_events,
            "interaction_p": "",
            "c_idx_text": c_text,
            "hr_text": hr_text,
            "hr_val": hr,
            "lower": lower,
            "upper": upper,
            "is_header": False,
        }
    )

    for group_info in SUBGROUP_STRUCTURE[1:]:
        group_name = group_info["group_name"]
        col_name = group_info["col_name"]
        sub_items = group_info["sub_items"]
        p_interaction = interaction_pvalue(df_plot, "high_risk", col_name, TIME_COL, EVENT_COL)

        plot_rows.append(
            {
                "label": group_name,
                "n": "",
                "events": "",
                "interaction_p": format_pvalue(p_interaction),
                "c_idx_text": "",
                "hr_text": "",
                "hr_val": np.nan,
                "lower": np.nan,
                "upper": np.nan,
                "is_header": True,
            }
        )

        for item in sub_items:
            df_sub = df_plot[df_plot[col_name] == item].dropna(
                subset=["risk_score", "high_risk", TIME_COL, EVENT_COL]
            ).copy()
            if len(df_sub) == 0:
                continue
            n_patients = len(df_sub)
            n_events = int(df_sub[EVENT_COL].sum())

            if n_events < 2:
                cindex = c_low = c_high = np.nan
                c_text = "NA"
            else:
                cindex, c_low, c_high = calc_cindex_ci(
                    df_sub["risk_score"].values,
                    df_sub[TIME_COL].values,
                    df_sub[EVENT_COL].values,
                    n_bootstrap=n_bootstrap,
                )
                c_text = format_ci_text(cindex, c_low, c_high, digits=3)

            if sparse_or_invalid(n_patients, n_events, min_n, min_events):
                hr = lower = upper = np.nan
            else:
                hr, lower, upper, _ = get_hr_ci(df_sub, "high_risk", TIME_COL, EVENT_COL)
                if not (np.isfinite(hr) and np.isfinite(lower) and np.isfinite(upper)):
                    hr = lower = upper = np.nan

            hr_text = format_hr_text(hr, lower, upper)
            if hr_text == "NA":
                c_text = "NA"

            plot_rows.append(
                {
                    "label": f"    {display_label(item)}",
                    "n": n_patients,
                    "events": n_events,
                    "interaction_p": "",
                    "c_idx_text": c_text,
                    "hr_text": hr_text,
                    "hr_val": hr,
                    "lower": lower,
                    "upper": upper,
                    "is_header": False,
                }
            )

    return pd.DataFrame(plot_rows)


def draw_forest(df_forest, fig_id, dataset_name, output_dir):
    fig = plt.figure(figsize=(24, max(10, len(df_forest) * 0.45)))
    gs = GridSpec(1, 2, width_ratios=[0.60, 0.40], wspace=0.02)
    ax_table = fig.add_subplot(gs[0])
    ax_forest = fig.add_subplot(gs[1], sharey=ax_table)
    y_pos = np.arange(len(df_forest), 0, -1)

    ax_table.spines["top"].set_visible(False)
    ax_table.spines["right"].set_visible(False)
    ax_table.spines["bottom"].set_visible(False)
    ax_table.spines["left"].set_visible(False)
    ax_table.tick_params(left=False, labelleft=False, bottom=False, labelbottom=False)
    ax_table.set_xlim(0, 1)
    ax_table.set_ylim(y_pos.min() - 0.8, y_pos.max() + 0.8)

    col_label = 0.02
    col_n = 0.26
    col_event = 0.36
    col_interact = 0.50
    col_cindex = 0.70
    col_hrtext = 0.92

    header_y = len(df_forest) + 0.7
    for x, text, ha in zip(
        [col_label, col_n, col_event, col_interact, col_cindex, col_hrtext],
        ["Subgroup", "N", "Events", "Interaction P", "Harrell C-index (95% CI)", "Hazard ratio (95% CI)"],
        ["left", "center", "center", "center", "center", "center"],
    ):
        ax_table.text(x, header_y, text, fontsize=14, fontweight="bold", ha=ha, va="center")

    for i, row in df_forest.iterrows():
        y = y_pos[i]
        if row["is_header"]:
            ax_table.text(col_label, y, row["label"], fontsize=14, fontweight="bold", ha="left", va="center")
            ax_table.text(col_interact, y, row["interaction_p"], fontsize=14, ha="center", va="center")
        else:
            ax_table.text(col_label, y, row["label"], fontsize=14, ha="left", va="center")
            ax_table.text(col_n, y, str(row["n"]), fontsize=14, ha="center", va="center")
            ax_table.text(col_event, y, str(row["events"]), fontsize=14, ha="center", va="center")
            ax_table.text(col_cindex, y, row["c_idx_text"], fontsize=14, ha="center", va="center")
            ax_table.text(col_hrtext, y, row["hr_text"], fontsize=14, ha="center", va="center")

    ax_forest.spines["left"].set_visible(False)
    ax_forest.spines["top"].set_visible(False)
    ax_forest.spines["right"].set_visible(False)
    ax_forest.tick_params(left=False, labelleft=False)
    ax_forest.set_xscale("log")
    ax_forest.set_xlim(0.5, 60)
    ax_forest.set_xticks([0.5, 1, 2, 5, 10, 20, 60])
    ax_forest.set_xticklabels(["0.5", "1", "2", "5", "10", "20", "60"], fontsize=12)
    ax_forest.set_xlabel("Hazard ratio (BCMPM High-risk vs Low-risk)", fontsize=14, fontweight="bold")
    ax_forest.axvline(1, color="gray", ls="--", lw=1.3, alpha=0.6)

    for i, row in df_forest.iterrows():
        y = y_pos[i]
        if row["is_header"] or not np.isfinite(row["hr_val"]):
            continue
        hr, low, upper = row["hr_val"], row["lower"], row["upper"]
        if not (np.isfinite(hr) and np.isfinite(low) and np.isfinite(upper)):
            continue
        marker = "D" if row["label"].strip() == "Overall" else "s"
        ax_forest.errorbar(
            hr,
            y,
            xerr=[[hr - low], [upper - hr]],
            fmt="none",
            ecolor=PURPLE,
            elinewidth=1.8,
            capsize=4,
            zorder=2,
        )
        ax_forest.scatter(
            hr,
            y,
            marker=marker,
            s=95 if marker == "D" else 70,
            color=PURPLE,
            zorder=3,
        )

    ax_forest.set_yticks([])
    fig.tight_layout()

    output_dir.mkdir(parents=True, exist_ok=True)
    table_path = output_dir / f"{fig_id}_{dataset_name}_BCMPM_subgroup_forest_table.csv"
    png_path = output_dir / f"{fig_id}_{dataset_name}_BCMPM_subgroup_forest_600dpi.png"
    svg_path = output_dir / f"{fig_id}_{dataset_name}_BCMPM_subgroup_forest_600dpi.svg"

    df_forest.to_csv(table_path, index=False, encoding="utf-8-sig")
    fig.savefig(png_path, dpi=600, bbox_inches="tight")
    fig.savefig(svg_path, bbox_inches="tight")
    plt.close(fig)
    return table_path, png_path, svg_path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate FigS5-S8 BCMPM subgroup forest plots for train/validation/test/total cohorts."
    )
    parser.add_argument("--input-dir", default=str(DEFAULT_INPUT_DIR), help="Directory containing *_all_modalities_risk.csv files.")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR), help="Directory for FigS5-S8 outputs.")
    parser.add_argument("--n-bootstrap", type=int, default=1000, help="Bootstrap repeats for C-index 95% CI.")
    return parser.parse_args()


def main():
    args = parse_args()
    configure_style()
    input_dir = Path(args.input_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    print(f"Input dir : {input_dir}")
    print(f"Output dir: {output_dir}")
    print(f"C-index bootstrap repeats: {args.n_bootstrap}")

    for dataset_name, filename, fig_id in DATASETS:
        csv_path = input_dir / filename
        if not csv_path.exists():
            raise FileNotFoundError(f"Missing input CSV for {dataset_name}: {csv_path}")
        print(f"\nConstructing {fig_id} forest plot table for {dataset_name} cohort...")
        df = prepare_dataframe(pd.read_csv(csv_path))
        df_forest = construct_forest_table(df, n_bootstrap=args.n_bootstrap)
        table_path, png_path, svg_path = draw_forest(df_forest, fig_id, dataset_name, output_dir)
        print(f"  Saved table: {table_path}")
        print(f"  Saved PNG  : {png_path}")
        print(f"  Saved SVG  : {svg_path}")


if __name__ == "__main__":
    main()
