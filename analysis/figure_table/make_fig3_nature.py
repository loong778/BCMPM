import argparse
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from scipy.signal import savgol_filter
from scipy.stats import chi2
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import auc, roc_curve

try:
    from lifelines import CoxPHFitter, KaplanMeierFitter
    from lifelines.statistics import logrank_test
except ImportError:
    CoxPHFitter = None
    KaplanMeierFitter = None
    logrank_test = None


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

TARGET_MOD = "BCMPM"
TARGET_TIME = 60


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


def configure_style():
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 8,
            "axes.labelsize": 9,
            "axes.titlesize": 10,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 7,
            "axes.linewidth": 0.8,
            "xtick.major.width": 0.8,
            "ytick.major.width": 0.8,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def panel_label(ax, label, x=-0.14, y=1.08):
    ax.text(x, y, label, transform=ax.transAxes, fontsize=14, fontweight="bold", va="top", ha="left")


def km_censor_survival_at(time, event, query_times):
    time = np.asarray(time, dtype=float)
    event = np.asarray(event, dtype=int)
    censor_event = 1 - event
    query_times = np.asarray(query_times, dtype=float)
    unique_times = np.unique(time)
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


def km_curve(time, event):
    time = np.asarray(time, dtype=float)
    event = np.asarray(event, dtype=int)
    unique_times = np.unique(time)
    x = [0.0]
    s = [1.0]
    lo = [1.0]
    hi = [1.0]
    surv = 1.0
    greenwood_sum = 0.0

    for t in unique_times:
        at_risk = np.sum(time >= t)
        d = np.sum((time == t) & (event == 1))
        if at_risk > 0 and d > 0:
            surv *= 1.0 - d / at_risk
            if at_risk - d > 0:
                greenwood_sum += d / (at_risk * (at_risk - d))
        se = surv * np.sqrt(greenwood_sum) if greenwood_sum > 0 else 0.0
        x.append(float(t))
        s.append(float(surv))
        lo.append(float(np.clip(surv - 1.96 * se, 0, 1)))
        hi.append(float(np.clip(surv + 1.96 * se, 0, 1)))

    return np.asarray(x), np.asarray(s), np.asarray(lo), np.asarray(hi)


def step_value_at(x, y, query):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    query = np.asarray(query, dtype=float)
    idx = np.searchsorted(x, query, side="right") - 1
    idx = np.clip(idx, 0, len(y) - 1)
    return y[idx]


def logrank_p_value(time_a, event_a, time_b, event_b):
    time_a = np.asarray(time_a, dtype=float)
    event_a = np.asarray(event_a, dtype=int)
    time_b = np.asarray(time_b, dtype=float)
    event_b = np.asarray(event_b, dtype=int)
    event_times = np.unique(np.concatenate([time_a[event_a == 1], time_b[event_b == 1]]))
    observed = 0.0
    expected = 0.0
    variance = 0.0

    for t in event_times:
        n1 = np.sum(time_a >= t)
        n2 = np.sum(time_b >= t)
        d1 = np.sum((time_a == t) & (event_a == 1))
        d2 = np.sum((time_b == t) & (event_b == 1))
        n = n1 + n2
        d = d1 + d2
        if n <= 1 or d == 0:
            continue
        observed += d1
        expected += d * n1 / n
        variance += (n1 * n2 * d * (n - d)) / (n * n * (n - 1))

    if variance <= 0:
        return np.nan
    statistic = (observed - expected) ** 2 / variance
    return float(chi2.sf(statistic, 1))


def person_time_hr(time_high, event_high, time_low, event_low):
    events_h = np.sum(event_high)
    events_l = np.sum(event_low)
    pt_h = np.sum(time_high)
    pt_l = np.sum(time_low)
    hr = ((events_h + 0.5) / max(pt_h, 1e-8)) / ((events_l + 0.5) / max(pt_l, 1e-8))
    se = np.sqrt(1.0 / (events_h + 0.5) + 1.0 / (events_l + 0.5))
    low = np.exp(np.log(hr) - 1.96 * se)
    high = np.exp(np.log(hr) + 1.96 * se)
    return float(hr), float(low), float(high)


def cox_hr_like_figure_enhance(risk, time, event, cutoff, max_iter=100, tol=1e-9):
    """Match utils.calculate_hr(): univariable Cox model on high_risk=(risk >= cutoff)."""
    risk = np.asarray(risk, dtype=float)
    time = np.asarray(time, dtype=float)
    event = np.asarray(event, dtype=int)
    high_risk = (risk >= cutoff).astype(int)

    if CoxPHFitter is not None:
        df_cox = pd.DataFrame({"time": time, "event": event, "high_risk": high_risk})
        cph = CoxPHFitter()
        cph.fit(df_cox, duration_col="time", event_col="event")
        hr = float(np.exp(cph.params_["high_risk"]))
        ci_df = cph.confidence_intervals_
        ci_column_pairs = [
            ("95% lower-bound", "95% upper-bound"),
            ("lower 95%", "upper 95%"),
            (0.025, 0.975),
            ("lower", "upper"),
            ("lower 0.95", "upper 0.95"),
        ]
        for low_col, high_col in ci_column_pairs:
            if low_col in ci_df.columns and high_col in ci_df.columns:
                hr_low = float(np.exp(ci_df.loc["high_risk", low_col]))
                hr_high = float(np.exp(ci_df.loc["high_risk", high_col]))
                return hr, hr_low, hr_high
        raise ValueError(f"Unrecognized lifelines confidence interval columns: {list(ci_df.columns)}")

    x = high_risk.astype(float)
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
    hr_low = np.exp(beta - 1.96 * se)
    hr_high = np.exp(beta + 1.96 * se)
    return float(hr), float(hr_low), float(hr_high)


def compute_base_cumulative_hazard(time, event, risk):
    time = np.asarray(time, dtype=float)
    event = np.asarray(event, dtype=int)
    risk = np.asarray(risk, dtype=float)
    order = np.argsort(time)
    time_sorted = time[order]
    event_sorted = event[order]
    exp_risk = np.exp(risk[order])
    event_times = np.unique(time_sorted[event_sorted == 1])
    increments = []
    for t in event_times:
        risk_set_sum = np.sum(exp_risk[time_sorted >= t])
        n_events = np.sum((time_sorted == t) & (event_sorted == 1))
        increments.append(n_events / max(risk_set_sum, 1e-8))
    return event_times, np.cumsum(increments)


def predict_survival_prob(eval_time, risk, base_times, base_h0):
    if len(base_times) == 0:
        return np.ones_like(risk, dtype=float)
    h0_t = np.interp(eval_time, base_times, base_h0, left=0.0, right=base_h0[-1])
    return np.exp(-h0_t * np.exp(risk))


def binary_outcome_ipcw(time, event, eval_time):
    time = np.asarray(time, dtype=float)
    event = np.asarray(event, dtype=int)
    g_i = km_censor_survival_at(time, event, np.minimum(time, eval_time))
    g_t = float(km_censor_survival_at(time, event, np.array([eval_time]))[0])
    y = ((time <= eval_time) & (event == 1)).astype(float)
    weights = np.where(time > eval_time, 1.0 / max(g_t, 1e-8), np.where(event == 1, 1.0 / g_i, 0.0))
    return y, weights


def calibration_bin_stats(pred_survival, time, event, eval_time, n_bins=3):
    pred_survival = np.asarray(pred_survival, dtype=float)
    edges = np.quantile(pred_survival, np.linspace(0, 1, n_bins + 1))
    edges[0], edges[-1] = 0.0, 1.0
    rows = []
    for i in range(n_bins):
        if i == 0:
            mask = (pred_survival >= edges[i]) & (pred_survival <= edges[i + 1])
        else:
            mask = (pred_survival > edges[i]) & (pred_survival <= edges[i + 1])
        if np.sum(mask) < 5:
            continue
        x, s, low, high = km_curve(time[mask], event[mask])
        rows.append(
            (
                float(np.mean(pred_survival[mask])),
                float(step_value_at(x, s, eval_time)),
                float(step_value_at(x, low, eval_time)),
                float(step_value_at(x, high, eval_time)),
            )
        )
    rows = sorted(rows, key=lambda item: item[0])
    if not rows:
        return np.array([]), np.array([]), np.array([]), np.array([])
    return tuple(np.asarray(col, dtype=float) for col in zip(*rows))


def calculate_true_ici(pred_survival, time, event, eval_time):
    pred_event = 1.0 - np.asarray(pred_survival, dtype=float)
    y, weights = binary_outcome_ipcw(time, event, eval_time)
    valid = weights > 0
    if not np.any(valid):
        return np.nan
    return float(np.average(np.abs(pred_event[valid] - y[valid]), weights=weights[valid]))


def find_best_cutoff_youden(risk, time, event, eval_time=TARGET_TIME):
    y_true = ((np.asarray(time) <= eval_time) & (np.asarray(event) == 1)).astype(int)
    fpr, tpr, thresholds = roc_curve(y_true, np.asarray(risk, dtype=float))
    return float(thresholds[np.argmax(tpr - fpr)])


def time_dependent_roc_ipcw(risk, time, event, eval_time):
    y, weights = binary_outcome_ipcw(time, event, eval_time)
    if np.unique(y).size < 2:
        return np.array([0, 1]), np.array([0, 1]), np.nan, np.nan, np.nan
    fpr, tpr, _ = roc_curve(y, risk, sample_weight=weights)
    auc_val = float(auc(fpr, tpr))
    return fpr, tpr, auc_val, np.nan, np.nan


def save_figure(fig, output_dir, stem, dpi=600):
    png_path = output_dir / f"{stem}.png"
    svg_path = output_dir / f"{stem}.svg"
    fig.savefig(png_path, dpi=dpi, bbox_inches="tight")
    fig.savefig(svg_path, bbox_inches="tight")
    plt.close(fig)
    return png_path, svg_path


def calibration_context(results):
    target = results[TARGET_MOD]
    train_t = target["time"]["Train"]
    train_e = target["event"]["Train"]
    train_r = target["risk"]["Train"]
    val_t = target["time"]["Validation"]
    val_e = target["event"]["Validation"]
    val_r = target["risk"]["Validation"]
    dev_t = np.concatenate([train_t, val_t])
    dev_e = np.concatenate([train_e, val_e])
    dev_r = np.concatenate([train_r, val_r])
    base_times, base_h0 = compute_base_cumulative_hazard(train_t, train_e, train_r)

    iso_models = {}
    for eval_time in [36, 60]:
        pred_surv_dev = predict_survival_prob(eval_time, dev_r, base_times, base_h0)
        pred_event_dev = 1 - pred_surv_dev
        y_dev, weights_dev = binary_outcome_ipcw(dev_t, dev_e, eval_time)
        valid = weights_dev > 0
        model = IsotonicRegression(increasing=True, y_min=0.0, y_max=1.0, out_of_bounds="clip")
        model.fit(pred_event_dev[valid], y_dev[valid], sample_weight=weights_dev[valid])
        iso_models[eval_time] = model

    return target, base_times, base_h0, iso_models


def draw_calibration_panel(ax, target, base_times, base_h0, iso_models, cohort, label, ici_rows=None):
    palette = {36: "#4C78A8", 60: "#E45756"}
    labels = {36: "3 years", 60: "5 years"}
    time_d = target["time"][cohort]
    event_d = target["event"][cohort]
    risk_d = target["risk"][cohort]
    ax.plot([0, 1], [0, 1], color="#CFCFCF", lw=1.0, label="Ideal line")
    rug_values = []
    for eval_time in [36, 60]:
        pred_surv = predict_survival_prob(eval_time, risk_d, base_times, base_h0)
        pred_event_cal = iso_models[eval_time].predict(1 - pred_surv)
        pred_surv_cal = 1 - pred_event_cal
        rug_values.append(pred_surv_cal)
        x_m, y_o, y_l, y_h = calibration_bin_stats(pred_surv_cal, time_d, event_d, eval_time, n_bins=3)
        if len(x_m) > 0:
            y_l = np.minimum(y_l, y_o)
            y_h = np.maximum(y_h, y_o)
            ax.errorbar(
                x_m,
                y_o,
                yerr=[np.clip(y_o - y_l, 0, None), np.clip(y_h - y_o, 0, None)],
                fmt="x",
                color=palette[eval_time],
                capsize=2.5,
                lw=1.0,
                markersize=4.8,
                markeredgewidth=1.15,
                label=labels[eval_time],
            )
            ax.plot(x_m, y_o, color=palette[eval_time], lw=1.15)
        if ici_rows is not None:
            ici_rows.append(
                {
                    "Dataset": cohort,
                    "Time_point": labels[eval_time],
                    "ICI": calculate_true_ici(pred_surv_cal, time_d, event_d, eval_time),
                }
            )
    ax.vlines(np.concatenate(rug_values), 0.975, 1.0, color="black", lw=0.45, alpha=0.55, transform=ax.get_xaxis_transform())
    ax.set_xlim(-0.04, 1.04)
    ax.set_ylim(0, 1.02)
    ax.set_title(COHORT_TITLES[cohort], fontweight="bold", pad=6)
    ax.set_xlabel(f"{TARGET_MOD}-predicted survival probability")
    ax.set_ylabel("Observed survival probability")
    ax.legend(loc="lower right", frameon=False, fontsize=6.5)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.8)
    panel_label(ax, label)


def make_calibration_figure(results, output_dir):
    target, base_times, base_h0, iso_models = calibration_context(results)
    fig, axes = plt.subplots(1, 3, figsize=(13.4, 4.0))
    fig.subplots_adjust(left=0.07, right=0.98, top=0.86, bottom=0.20, wspace=0.30)
    ici_rows = []

    for ax, cohort, label in zip(axes, COHORTS, ["a", "b", "c"]):
        draw_calibration_panel(ax, target, base_times, base_h0, iso_models, cohort, label, ici_rows)

    ici_df = pd.DataFrame(ici_rows)
    ici_df.to_csv(output_dir / "figure3a_c_calibration_ici.csv", index=False, float_format="%.4f")
    return save_figure(fig, output_dir, "figure3a_c_calibration_curve")


def draw_km_panel(ax, target, cutoff, cohort, label, summary_rows=None):
    ticks = [0, 30, 60, 90, 120]
    km_colors = {"High risk": "#C6203B", "Low risk": "#2878B5"}
    time_d = np.asarray(target["time"][cohort], dtype=float)
    event_d = np.asarray(target["event"][cohort], dtype=int)
    risk_d = np.asarray(target["risk"][cohort], dtype=float)
    masks = {"High risk": risk_d > cutoff, "Low risk": risk_d <= cutoff}
    legend_labels = {"High risk": f"{TARGET_MOD} high risk", "Low risk": f"{TARGET_MOD} low risk"}

    for group_name, mask in masks.items():
        x, s, lo, hi = km_curve(time_d[mask], event_d[mask])
        ax.step(x, s, where="post", color=km_colors[group_name], lw=1.5, label=legend_labels[group_name])
        ax.fill_between(x, lo, hi, step="post", color=km_colors[group_name], alpha=0.16, linewidth=0)
        censor_times = time_d[mask & (event_d == 0)]
        censor_surv = step_value_at(x, s, censor_times)
        ax.plot(censor_times, censor_surv, "|", color=km_colors[group_name], markersize=4.6, markeredgewidth=0.75)

    high = masks["High risk"]
    low = masks["Low risk"]
    p_val = logrank_p_value(time_d[high], event_d[high], time_d[low], event_d[low])
    hr, hr_low, hr_high = cox_hr_like_figure_enhance(risk_d, time_d, event_d, cutoff)
    p_text = "p < 0.0001" if p_val < 0.0001 else f"p = {p_val:.4f}"
    hr_text = f"HR = {hr:.2f} [{hr_low:.2f}-{hr_high:.2f}]"
    ax.text(0.04, 0.18, p_text, transform=ax.transAxes, fontweight="bold", fontsize=7.0)
    ax.text(0.04, 0.125, hr_text, transform=ax.transAxes, fontweight="bold", fontsize=7.0)
    ax.set_title(f"{TARGET_MOD} in the {COHORT_TITLES[cohort].lower()}", fontweight="bold", pad=6)
    ax.set_xlabel("Time after surgery (months)")
    ax.set_ylabel("Disease-free survival")
    ax.set_xlim(0, 120)
    ax.set_ylim(0, 1.02)
    ax.set_xticks(ticks)
    ax.set_yticks(np.arange(0, 1.01, 0.2))
    ax.legend(loc="lower left", bbox_to_anchor=(0.04, 0.015), borderaxespad=0, frameon=False, fontsize=6.4, handlelength=1.9, labelspacing=0.18)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    panel_label(ax, label)

    risk_label_x = -0.23
    ax.text(risk_label_x, -0.27, "Number at risk", transform=ax.transAxes, va="top", ha="left", fontweight="bold", fontsize=6.8)
    ax.text(risk_label_x, -0.39, "High risk", transform=ax.transAxes, va="top", ha="left", color=km_colors["High risk"], fontweight="bold", fontsize=6.8)
    ax.text(risk_label_x, -0.50, "Low risk", transform=ax.transAxes, va="top", ha="left", color=km_colors["Low risk"], fontweight="bold", fontsize=6.8)
    for tick in ticks:
        x_rel = tick / 120
        ax.text(x_rel, -0.39, str(int(np.sum(time_d[high] >= tick))), transform=ax.transAxes, va="top", ha="center", fontsize=6.8)
        ax.text(x_rel, -0.50, str(int(np.sum(time_d[low] >= tick))), transform=ax.transAxes, va="top", ha="center", fontsize=6.8)
    if summary_rows is not None:
        summary_rows.append(
            {"Dataset": cohort, "Cutoff": cutoff, "HR": hr, "HR_low": hr_low, "HR_high": hr_high, "Logrank_p": p_val}
        )


def draw_km_panel_figure_enhance(ax, target, cutoff, cohort, label, summary_rows=None):
    size_label = 24
    size_title = 14
    size_axis = 14
    size_tick = 14
    size_legend = 14
    size_risk_table = 14
    size_ptext = 12
    risk_time_ticks = [0, 30, 60, 90, 120]
    km_titles = {
        "Train": f"{TARGET_MOD} in the training cohort",
        "Validation": f"{TARGET_MOD} in the internal validation cohort",
        "Test": f"{TARGET_MOD} in the external test cohort",
    }
    km_colors = {"high": "#d62728", "low": "#1f77b4"}

    time_d = np.asarray(target["time"][cohort], dtype=float)
    event_d = np.asarray(target["event"][cohort], dtype=int)
    risk_d = np.asarray(target["risk"][cohort], dtype=float)
    high_mask = risk_d > cutoff
    low_mask = risk_d <= cutoff

    if KaplanMeierFitter is not None:
        kmf_h, kmf_l = KaplanMeierFitter(), KaplanMeierFitter()
        kmf_h.fit(time_d[high_mask], event_d[high_mask])
        kmf_l.fit(time_d[low_mask], event_d[low_mask])

        surv_h = kmf_h.survival_function_
        ci_h = kmf_h.confidence_interval_
        ax.plot(surv_h.index, surv_h.iloc[:, 0], color=km_colors["high"], lw=2, drawstyle="steps-post", label="High risk")
        ax.fill_between(ci_h.index, ci_h.iloc[:, 0], ci_h.iloc[:, 1], step="post", color=km_colors["high"], alpha=0.2)

        surv_l = kmf_l.survival_function_
        ci_l = kmf_l.confidence_interval_
        ax.plot(surv_l.index, surv_l.iloc[:, 0], color=km_colors["low"], lw=2, drawstyle="steps-post", label="Low risk")
        ax.fill_between(ci_l.index, ci_l.iloc[:, 0], ci_l.iloc[:, 1], step="post", color=km_colors["low"], alpha=0.2)

        cens_h_times = time_d[high_mask & (event_d == 0)]
        cens_h_surv = np.interp(cens_h_times, surv_h.index.values, surv_h.iloc[:, 0].values)
        ax.plot(cens_h_times, cens_h_surv, "|", color=km_colors["high"], markersize=8)

        cens_l_times = time_d[low_mask & (event_d == 0)]
        cens_l_surv = np.interp(cens_l_times, surv_l.index.values, surv_l.iloc[:, 0].values)
        ax.plot(cens_l_times, cens_l_surv, "|", color=km_colors["low"], markersize=8)

        lr_res = logrank_test(time_d[high_mask], time_d[low_mask], event_d[high_mask], event_d[low_mask])
        p_val = float(lr_res.p_value)
    else:
        x_h, s_h, lo_h, hi_h = km_curve(time_d[high_mask], event_d[high_mask])
        ax.plot(x_h, s_h, color=km_colors["high"], lw=2, drawstyle="steps-post", label="High risk")
        ax.fill_between(x_h, lo_h, hi_h, step="post", color=km_colors["high"], alpha=0.2)
        cens_h_times = time_d[high_mask & (event_d == 0)]
        ax.plot(cens_h_times, step_value_at(x_h, s_h, cens_h_times), "|", color=km_colors["high"], markersize=8)

        x_l, s_l, lo_l, hi_l = km_curve(time_d[low_mask], event_d[low_mask])
        ax.plot(x_l, s_l, color=km_colors["low"], lw=2, drawstyle="steps-post", label="Low risk")
        ax.fill_between(x_l, lo_l, hi_l, step="post", color=km_colors["low"], alpha=0.2)
        cens_l_times = time_d[low_mask & (event_d == 0)]
        ax.plot(cens_l_times, step_value_at(x_l, s_l, cens_l_times), "|", color=km_colors["low"], markersize=8)

        p_val = logrank_p_value(time_d[high_mask], event_d[high_mask], time_d[low_mask], event_d[low_mask])

    p_text = "p < 0.0001" if p_val < 0.0001 else f"p = {p_val:.4f}"
    ax.text(0.03, 0.16, p_text, transform=ax.transAxes, va="bottom", ha="left", fontweight="bold", fontsize=size_ptext)

    hr, hr_low, hr_high = cox_hr_like_figure_enhance(risk_d, time_d, event_d, cutoff)
    hr_text = f"HR = {hr:.2f} [{hr_low:.2f}-{hr_high:.2f}]"
    ax.text(0.03, 0.12, hr_text, transform=ax.transAxes, va="bottom", ha="left", fontweight="bold", fontsize=size_ptext)

    ax.set_xlabel("Time after surgery (months)", fontsize=size_axis, fontweight="bold")
    ax.set_ylabel("Disease free survival", fontsize=size_axis, fontweight="bold")
    ax.set_title(km_titles[cohort], fontsize=size_title, fontweight="bold", pad=10)
    ax.set_xlim([0, 120])
    ax.set_ylim([0.0, 1.02])
    ax.set_xticks(risk_time_ticks)
    ax.set_yticks(np.arange(0, 1.01, 0.2))
    ax.tick_params(axis="both", length=4, labelsize=size_tick)
    for tick_label in ax.get_xticklabels() + ax.get_yticklabels():
        tick_label.set_fontweight("bold")

    ax.legend(loc="lower left", fontsize=size_legend)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.text(-0.20, -0.18, "Number at risk", transform=ax.transAxes, va="top", ha="left", fontweight="bold", fontsize=size_risk_table)
    ax.text(-0.20, -0.26, "High risk", color=km_colors["high"], transform=ax.transAxes, va="top", ha="left", fontweight="bold", fontsize=size_risk_table)
    ax.text(-0.20, -0.34, "Low risk", color=km_colors["low"], transform=ax.transAxes, va="top", ha="left", fontweight="bold", fontsize=size_risk_table)

    for tick in risk_time_ticks:
        x_rel = tick / 120
        n_h = int(np.sum(time_d[high_mask] >= tick))
        n_l = int(np.sum(time_d[low_mask] >= tick))
        ax.text(x_rel, -0.26, str(n_h), transform=ax.transAxes, va="top", ha="center", fontweight="bold", fontsize=size_risk_table)
        ax.text(x_rel, -0.34, str(n_l), transform=ax.transAxes, va="top", ha="center", fontweight="bold", fontsize=size_risk_table)

    ax.text(-0.18, 1.06, label, transform=ax.transAxes, fontsize=size_label, fontweight="bold", va="top")

    if summary_rows is not None:
        summary_rows.append(
            {"Dataset": cohort, "Cutoff": cutoff, "HR": hr, "HR_low": hr_low, "HR_high": hr_high, "Logrank_p": p_val}
        )


def make_km_figure(results, output_dir):
    target = results[TARGET_MOD]
    cutoff = find_best_cutoff_youden(target["risk"]["Train"], target["time"]["Train"], target["event"]["Train"])
    fig, axes = plt.subplots(1, 3, figsize=(24, 8))
    fig.subplots_adjust(left=0.06, right=0.97, top=0.88, bottom=0.15, wspace=0.28)
    summary_rows = []

    for ax, cohort, label in zip(axes, COHORTS, ["d", "e", "f"]):
        draw_km_panel_figure_enhance(ax, target, cutoff, cohort, label, summary_rows)

    pd.DataFrame(summary_rows).to_csv(output_dir / "figure3d_f_km_summary.csv", index=False, float_format="%.4f")
    return save_figure(fig, output_dir, "figure3d-f_km_survival")


def draw_roc_dca_impact_panels(ax_g, ax_h, ax_i, results, dca_rows=None, impact_rows=None):
    thresholds = np.linspace(0.01, 0.8, 200)
    smooth_window = 21
    smooth_poly = 3
    cohort = "Test"

    for modality in ORDER:
        stored = results[modality]["td_roc_raw"].get(TARGET_TIME, {}).get(cohort)
        if stored is not None:
            fpr, tpr, auc_val, ci_low, ci_high = stored
        else:
            fpr, tpr, auc_val, ci_low, ci_high = time_dependent_roc_ipcw(
                results[modality]["risk"][cohort],
                results[modality]["time"][cohort],
                results[modality]["event"][cohort],
                TARGET_TIME,
            )
        ci_text = "" if not np.isfinite(ci_low) else f" [{ci_low:.3f}-{ci_high:.3f}]"
        ax_g.step(fpr, tpr, where="post", color=COLORS[modality], lw=1.15, label=f"{modality} AUC={auc_val:.3f}{ci_text}")

    ax_g.plot([0, 1], [0, 1], color="black", lw=0.8, linestyle="--")
    ax_g.set_title("External test cohort ROC (5-year DFS)", fontweight="bold", pad=6)
    ax_g.set_xlabel("1 - Specificity")
    ax_g.set_ylabel("Sensitivity")
    ax_g.set_xlim(0, 1)
    ax_g.set_ylim(0, 1.04)
    ax_g.legend(loc="lower right", frameon=False, fontsize=5.8)
    ax_g.spines["top"].set_visible(False)
    ax_g.spines["right"].set_visible(False)
    panel_label(ax_g, "g")

    ref = results[ORDER[0]]
    y_ref, w_ref = binary_outcome_ipcw(ref["time"][cohort], ref["event"][cohort], TARGET_TIME)
    n_ref = len(y_ref)
    ax_h.plot(thresholds, np.zeros_like(thresholds), color="black", lw=1.0, linestyle="--", label="None")
    nb_all = []
    for p_t in thresholds:
        tp = np.sum(w_ref * y_ref)
        fp = np.sum(w_ref * (1 - y_ref))
        nb_all.append((tp / n_ref) - (fp / n_ref) * (p_t / (1 - p_t)))
    ax_h.plot(thresholds, savgol_filter(nb_all, smooth_window, smooth_poly, mode="nearest"), color="black", lw=1.0, label="All")

    for modality in ORDER:
        res = results[modality]
        time_d = res["time"][cohort]
        event_d = res["event"][cohort]
        risk_d = res["risk"][cohort]
        base_t, base_h = compute_base_cumulative_hazard(res["time"]["Train"], res["event"]["Train"], res["risk"]["Train"])
        pred_event = 1 - predict_survival_prob(TARGET_TIME, risk_d, base_t, base_h)
        y_true, weights = binary_outcome_ipcw(time_d, event_d, TARGET_TIME)
        n = len(time_d)
        nb_raw = []
        total_high = []
        true_positive = []
        for p_t in thresholds:
            pred_pos = pred_event >= p_t
            tp = np.sum(weights * pred_pos * y_true)
            fp = np.sum(weights * pred_pos * (1 - y_true))
            nb_raw.append((tp / n) - (fp / n) * (p_t / (1 - p_t)))
            total_high.append(np.sum(pred_pos))
            true_positive.append(np.sum(pred_pos & (event_d == 1) & (time_d <= TARGET_TIME)))
        nb_smooth = savgol_filter(nb_raw, smooth_window, smooth_poly, mode="nearest")
        high_smooth = np.clip(savgol_filter(total_high, smooth_window, smooth_poly, mode="nearest"), 0, None)
        tp_smooth = np.clip(savgol_filter(true_positive, smooth_window, smooth_poly, mode="nearest"), 0, None)
        ax_h.plot(thresholds, nb_smooth, color=COLORS[modality], lw=1.15, label=modality)
        ax_i.plot(thresholds, high_smooth, color=COLORS[modality], lw=1.15, label=modality)
        ax_i.plot(thresholds, tp_smooth, color=COLORS[modality], lw=0.95, linestyle="--", alpha=0.86)
        if dca_rows is not None:
            dca_rows.extend({"Modality": modality, "Threshold": p, "Net_benefit": nb} for p, nb in zip(thresholds, nb_raw))
        if impact_rows is not None:
            impact_rows.extend(
                {"Modality": modality, "Threshold": p, "Predicted_high_risk": h, "True_positive_cases": tp}
                for p, h, tp in zip(thresholds, total_high, true_positive)
            )

    ax_h.set_title("Decision curve analysis for 5-year DFS prediction", fontweight="bold", pad=6)
    ax_h.set_xlabel("Threshold probability")
    ax_h.set_ylabel("Net benefit (external test cohort)")
    ax_h.set_xlim(0, 0.8)
    ax_h.set_ylim(-0.12, 0.42)
    leg_h = ax_h.legend(loc="upper right", frameon=True, fontsize=5.8, handlelength=1.8, labelspacing=0.18, borderpad=0.25)
    leg_h.get_frame().set_facecolor("white")
    leg_h.get_frame().set_edgecolor("none")
    leg_h.get_frame().set_alpha(0.94)
    leg_h.set_zorder(20)
    ax_h.spines["top"].set_visible(False)
    ax_h.spines["right"].set_visible(False)
    panel_label(ax_h, "h")

    impact_handles = [
        Line2D([0], [0], color="black", lw=1.4, linestyle="-", label="Predicted high-risk patients"),
        Line2D([0], [0], color="black", lw=1.0, linestyle="--", label="True positive cases"),
    ]
    model_handles, model_labels = ax_i.get_legend_handles_labels()
    ax_i.set_title("Clinical impact curve for 5-year DFS prediction", fontweight="bold", pad=6)
    ax_i.set_xlabel("Threshold probability")
    ax_i.set_ylabel("Number of patients")
    ax_i.set_xlim(0, 0.8)
    ax_i.set_ylim(0, None)
    leg_i = ax_i.legend(
        model_handles + impact_handles,
        model_labels + [h.get_label() for h in impact_handles],
        loc="upper right",
        frameon=True,
        fontsize=5.6,
        handlelength=1.8,
        labelspacing=0.17,
        borderpad=0.25,
    )
    leg_i.get_frame().set_facecolor("white")
    leg_i.get_frame().set_edgecolor("none")
    leg_i.get_frame().set_alpha(0.94)
    leg_i.set_zorder(20)
    ax_i.spines["top"].set_visible(False)
    ax_i.spines["right"].set_visible(False)
    panel_label(ax_i, "i")


def make_roc_dca_impact_figure(results, output_dir):
    fig, axes = plt.subplots(1, 3, figsize=(13.4, 4.0))
    fig.subplots_adjust(left=0.065, right=0.985, top=0.84, bottom=0.18, wspace=0.30)
    ax_g, ax_h, ax_i = axes
    dca_rows = []
    impact_rows = []
    draw_roc_dca_impact_panels(ax_g, ax_h, ax_i, results, dca_rows, impact_rows)
    pd.DataFrame(dca_rows).to_csv(output_dir / "figure3h_dca_values.csv", index=False, float_format="%.5f")
    pd.DataFrame(impact_rows).to_csv(output_dir / "figure3i_clinical_impact_values.csv", index=False, float_format="%.5f")
    png, svg = save_figure(fig, output_dir, "figure3ghi_roc_dca_clinical_impact_labeled")
    return png, svg


def make_combined_fig3(results, output_dir):
    target, base_times, base_h0, iso_models = calibration_context(results)
    cutoff = find_best_cutoff_youden(target["risk"]["Train"], target["time"]["Train"], target["event"]["Train"])
    fig = plt.figure(figsize=(13.4, 12.4))
    gs = fig.add_gridspec(
        5,
        3,
        left=0.075,
        right=0.985,
        top=0.97,
        bottom=0.065,
        wspace=0.30,
        hspace=0.10,
        height_ratios=[1.0, 0.16, 1.0, 0.52, 1.0],
    )
    axes = np.empty((3, 3), dtype=object)
    for col in range(3):
        axes[0, col] = fig.add_subplot(gs[0, col])
        axes[1, col] = fig.add_subplot(gs[2, col])
        axes[2, col] = fig.add_subplot(gs[4, col])

    ici_rows = []
    km_rows = []
    dca_rows = []
    impact_rows = []
    for ax, cohort, label in zip(axes[0], COHORTS, ["a", "b", "c"]):
        draw_calibration_panel(ax, target, base_times, base_h0, iso_models, cohort, label, ici_rows)
    for ax, cohort, label in zip(axes[1], COHORTS, ["d", "e", "f"]):
        draw_km_panel(ax, target, cutoff, cohort, label, km_rows)
    draw_roc_dca_impact_panels(axes[2, 0], axes[2, 1], axes[2, 2], results, dca_rows, impact_rows)

    pd.DataFrame(ici_rows).to_csv(output_dir / "figure3a_c_calibration_ici.csv", index=False, float_format="%.4f")
    pd.DataFrame(km_rows).to_csv(output_dir / "figure3d_f_km_summary.csv", index=False, float_format="%.4f")
    pd.DataFrame(dca_rows).to_csv(output_dir / "figure3h_dca_values.csv", index=False, float_format="%.5f")
    pd.DataFrame(impact_rows).to_csv(output_dir / "figure3i_clinical_impact_values.csv", index=False, float_format="%.5f")
    return save_figure(fig, output_dir, "Fig3_a-i_Nature_600dpi")


def main():
    parser = argparse.ArgumentParser(description="Generate Figure 3 calibration, KM survival, ROC/DCA/clinical impact panels.")
    parser.add_argument("--result-root", default=None, help="Directory containing modality result folders.")
    parser.add_argument("--output-dir", default="./xxx/xx/Fig3_Nature", help="Output directory.")
    args = parser.parse_args()

    configure_style()
    result_root = resolve_result_root(args.result_root)
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    results = load_results(result_root)

    outputs = []
    outputs.extend(make_combined_fig3(results, output_dir))
    outputs.extend(make_calibration_figure(results, output_dir))
    outputs.extend(make_km_figure(results, output_dir))
    outputs.extend(make_roc_dca_impact_figure(results, output_dir))

    print(f"Result root: {result_root}")
    for path in outputs:
        print(f"Saved: {path}")


if __name__ == "__main__":
    main()



