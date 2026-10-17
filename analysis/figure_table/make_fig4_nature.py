import argparse
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import to_rgba
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from scipy.signal import savgol_filter
from scipy.stats import chi2


TARGET_TIME = 60
TICKS = [0, 30, 60, 90, 120]

COLORS = {
    "guideline_low": "#27A844",
    "guideline_high": "#F2C84B",
    "bcmpm_high": "#C6203B",
    "bcmpm_low": "#2E9AC2",
    "km_high": "#C6203B",
    "km_low": "#2878B5",
    "guideline_model": "#4CAF50",
    "bcmpm_model": "#E45756",
    "neutral": "#8F8F8F",
}


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


def load_data(guideline_csv, risk_csv):
    guideline = pd.read_csv(guideline_csv)
    risk = pd.read_csv(risk_csv)
    needed = ["patient_id", "bcmpm_high_risk", "bcmpm_risk60"]
    missing = [c for c in needed if c not in risk.columns]
    if missing:
        raise ValueError(f"Missing columns in risk CSV: {missing}")
    df = guideline.merge(risk[needed], on="patient_id", how="inner")
    if "Guideline_Risk_binary" not in df.columns:
        df["Guideline_Risk_binary"] = df["Guideline_Risk"].map({"LowModerate": 0, "High": 1})
    df["Guideline_Risk_binary"] = df["Guideline_Risk_binary"].astype(int)
    df["bcmpm_high_risk"] = df["bcmpm_high_risk"].astype(int)
    return df


def panel_label(ax, label, x=-0.12, y=1.08):
    ax.text(x, y, label, transform=ax.transAxes, fontsize=14, fontweight="bold", va="top", ha="left")


def interp_color(c1, c2, t):
    c1 = np.asarray(to_rgba(c1))
    c2 = np.asarray(to_rgba(c2))
    return tuple(c1 * (1.0 - t) + c2 * t)


def bezier(p0, p1, p2, p3, t):
    return (
        ((1.0 - t) ** 3) * np.asarray(p0)
        + 3.0 * ((1.0 - t) ** 2) * t * np.asarray(p1)
        + 3.0 * (1.0 - t) * (t**2) * np.asarray(p2)
        + (t**3) * np.asarray(p3)
    )


def draw_ribbon(ax, x0, y0_top, y0_bot, x1, y1_top, y1_bot, color_left, color_right, alpha=0.48, n=260):
    ts = np.linspace(0, 1, n + 1)
    dx = x1 - x0
    p0_top = (x0, y0_top)
    p3_top = (x1, y1_top)
    p0_bot = (x0, y0_bot)
    p3_bot = (x1, y1_bot)
    p1_top = (x0 + dx * 0.38, y0_top)
    p2_top = (x0 + dx * 0.62, y1_top)
    p1_bot = (x0 + dx * 0.38, y0_bot)
    p2_bot = (x0 + dx * 0.62, y1_bot)

    top = np.asarray([bezier(p0_top, p1_top, p2_top, p3_top, t) for t in ts])
    bot = np.asarray([bezier(p0_bot, p1_bot, p2_bot, p3_bot, t) for t in ts])
    for i in range(n):
        polygon = np.vstack([top[i], top[i + 1], bot[i + 1], bot[i]])
        patch = plt.Polygon(polygon, closed=True, facecolor=interp_color(color_left, color_right, i / n), edgecolor="none", alpha=alpha)
        ax.add_patch(patch)


def draw_sankey(ax, df):
    cross = pd.crosstab(df["Guideline_Risk"], df["bcmpm_high_risk"])
    val_ll = int(cross.loc["LowModerate", 0]) if "LowModerate" in cross.index and 0 in cross.columns else 0
    val_lh = int(cross.loc["LowModerate", 1]) if "LowModerate" in cross.index and 1 in cross.columns else 0
    val_hl = int(cross.loc["High", 0]) if "High" in cross.index and 0 in cross.columns else 0
    val_hh = int(cross.loc["High", 1]) if "High" in cross.index and 1 in cross.columns else 0

    left_counts = {"Inter&Low risk": val_ll + val_lh, "High risk": val_hl + val_hh}
    right_counts = {"High risk": val_lh + val_hh, "Low risk": val_ll + val_hl}
    total = sum(left_counts.values())

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    panel_label(ax, "a", x=-0.02, y=1.02)
    ax.set_title("Risk reclassification", fontweight="bold", pad=8)

    x_left, x_right = 0.14, 0.88
    bar_w = 0.028
    top = 0.69
    scale = 0.42 / total
    gap = 0.006

    left_y = {}
    y = top
    for cat in ["Inter&Low risk", "High risk"]:
        h = left_counts[cat] * scale
        left_y[cat] = (y - h, y)
        y -= h + gap

    right_y = {}
    y = top
    for cat in ["High risk", "Low risk"]:
        h = right_counts[cat] * scale
        right_y[cat] = (y - h, y)
        y -= h + gap

    segments = {}
    y = left_y["Inter&Low risk"][1]
    segments[("Inter&Low risk", "High risk")] = {"lt": y, "lb": y - val_lh * scale}
    y -= val_lh * scale
    segments[("Inter&Low risk", "Low risk")] = {"lt": y, "lb": y - val_ll * scale}
    y = left_y["High risk"][1]
    segments[("High risk", "High risk")] = {"lt": y, "lb": y - val_hh * scale}
    y -= val_hh * scale
    segments[("High risk", "Low risk")] = {"lt": y, "lb": y - val_hl * scale}

    y = right_y["High risk"][1]
    segments[("Inter&Low risk", "High risk")].update({"rt": y, "rb": y - val_lh * scale})
    y -= val_lh * scale
    segments[("High risk", "High risk")].update({"rt": y, "rb": y - val_hh * scale})
    y = right_y["Low risk"][1]
    segments[("Inter&Low risk", "Low risk")].update({"rt": y, "rb": y - val_ll * scale})
    y -= val_ll * scale
    segments[("High risk", "Low risk")].update({"rt": y, "rb": y - val_hl * scale})

    left_colors = {"Inter&Low risk": COLORS["guideline_low"], "High risk": COLORS["guideline_high"]}
    right_colors = {"High risk": COLORS["bcmpm_high"], "Low risk": COLORS["bcmpm_low"]}
    flow_order = [
        ("High risk", "High risk", val_hh),
        ("Inter&Low risk", "High risk", val_lh),
        ("Inter&Low risk", "Low risk", val_ll),
        ("High risk", "Low risk", val_hl),
    ]
    for left_cat, right_cat, value in flow_order:
        if value == 0:
            continue
        s = segments[(left_cat, right_cat)]
        draw_ribbon(ax, x_left, s["lt"], s["lb"], x_right, s["rt"], s["rb"], left_colors[left_cat], right_colors[right_cat])

    for cat in ["Inter&Low risk", "High risk"]:
        y0, y1 = left_y[cat]
        ax.add_patch(Rectangle((x_left - bar_w / 2, y0), bar_w, y1 - y0, facecolor=left_colors[cat], edgecolor="none", zorder=10))
        ax.text(x_left + 0.045, (y0 + y1) / 2, f"{cat}\n(n={left_counts[cat]})", ha="left", va="center", fontsize=9.0, fontweight="bold")
    for cat in ["High risk", "Low risk"]:
        y0, y1 = right_y[cat]
        ax.add_patch(Rectangle((x_right - bar_w / 2, y0), bar_w, y1 - y0, facecolor=right_colors[cat], edgecolor="none", zorder=10))
        ax.text(x_right - 0.045, (y0 + y1) / 2, f"{cat}\n(n={right_counts[cat]})", ha="right", va="center", fontsize=9.0, fontweight="bold")

    ax.text(0.065, 0.50, "Guideline", rotation=90, ha="center", va="center", fontsize=10, fontweight="bold", transform=ax.transAxes)
    ax.text(0.955, 0.50, "BCMPM", rotation=270, ha="center", va="center", fontsize=10, fontweight="bold", transform=ax.transAxes)
    return {
        "Guideline_InterLow_to_BCMPM_Low": val_ll,
        "Guideline_InterLow_to_BCMPM_High": val_lh,
        "Guideline_High_to_BCMPM_Low": val_hl,
        "Guideline_High_to_BCMPM_High": val_hh,
        "Guideline_InterLow": left_counts["Inter&Low risk"],
        "Guideline_High": left_counts["High risk"],
        "BCMPM_High": right_counts["High risk"],
        "BCMPM_Low": right_counts["Low risk"],
    }


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
        n1, n2, n, d = map(float, (n1, n2, n, d))
        observed += d1
        expected += d * n1 / n
        variance += (n1 * n2 * d * (n - d)) / (n * n * (n - 1))
    if variance <= 0:
        return np.nan
    return float(chi2.sf((observed - expected) ** 2 / variance, 1))


def cox_hr_ci(time, event, group, max_iter=50, tol=1e-8):
    time = np.asarray(time, dtype=float)
    event = np.asarray(event, dtype=int)
    group = np.asarray(group, dtype=float)
    if np.unique(group).size < 2:
        return np.nan, np.nan, np.nan
    beta = 0.0
    event_times = np.unique(time[event == 1])
    for _ in range(max_iter):
        score = 0.0
        info = 0.0
        exp_bg = np.exp(np.clip(beta * group, -40, 40))
        for t in event_times:
            event_mask = (time == t) & (event == 1)
            risk_mask = time >= t
            d = np.sum(event_mask)
            s0 = np.sum(exp_bg[risk_mask])
            s1 = np.sum(exp_bg[risk_mask] * group[risk_mask])
            s2 = np.sum(exp_bg[risk_mask] * group[risk_mask] * group[risk_mask])
            mean = s1 / max(s0, 1e-12)
            score += np.sum(group[event_mask]) - d * mean
            info += d * (s2 / max(s0, 1e-12) - mean * mean)
        if info <= 1e-12:
            break
        step = score / info
        beta += step
        if abs(step) < tol:
            break
    exp_bg = np.exp(np.clip(beta * group, -40, 40))
    info = 0.0
    for t in event_times:
        event_mask = (time == t) & (event == 1)
        risk_mask = time >= t
        d = np.sum(event_mask)
        s0 = np.sum(exp_bg[risk_mask])
        s1 = np.sum(exp_bg[risk_mask] * group[risk_mask])
        s2 = np.sum(exp_bg[risk_mask] * group[risk_mask] * group[risk_mask])
        mean = s1 / max(s0, 1e-12)
        info += d * (s2 / max(s0, 1e-12) - mean * mean)
    se = np.sqrt(1.0 / info) if info > 1e-12 else np.nan
    hr = np.exp(beta)
    low = np.exp(beta - 1.96 * se) if np.isfinite(se) else np.nan
    high = np.exp(beta + 1.96 * se) if np.isfinite(se) else np.nan
    return float(hr), float(low), float(high)


def draw_risk_table(ax, groups, times=TICKS):
    label_x = -0.29
    ax.text(label_x, -0.18, "At risk", transform=ax.transAxes, ha="left", va="top", fontsize=5.3, fontweight="bold", clip_on=False)
    for row, item in enumerate(groups):
        y = -0.31 - row * 0.13
        ax.text(label_x, y, item["label"], color=item["color"], transform=ax.transAxes, ha="left", va="top", fontsize=5.2, fontweight="bold", clip_on=False)
        for t in times:
            ax.text(t / max(times), y, str(int(np.sum(item["time"] >= t))), transform=ax.transAxes, ha="center", va="top", fontsize=5.2, color=item["color"], clip_on=False)


def draw_km_panel(ax, df, group_col, title, label, legend_labels=None):
    time = df["DFS"].to_numpy(dtype=float)
    event = df["status"].to_numpy(dtype=int)
    group = df[group_col].to_numpy(dtype=int)
    if legend_labels is None:
        legend_labels = {0: "Low risk", 1: "High risk"}

    plot_items = [
        (0, legend_labels[0], COLORS["km_low"]),
        (1, legend_labels[1], COLORS["km_high"]),
    ]
    risk_items = []
    for value, legend_label, color in plot_items:
        mask = group == value
        if np.sum(mask) == 0:
            continue
        x, s, lo, hi = km_curve(time[mask], event[mask])
        ax.step(x, s, where="post", color=color, lw=1.2, label=legend_label)
        ax.fill_between(x, lo, hi, step="post", color=color, alpha=0.16, linewidth=0)
        censor_times = time[mask & (event == 0)]
        ax.plot(censor_times, step_value_at(x, s, censor_times), "|", color=color, markersize=3.0, markeredgewidth=0.6)
        risk_items.append({"label": legend_label, "color": color, "time": time[mask]})

    p_val = logrank_p_value(time[group == 1], event[group == 1], time[group == 0], event[group == 0])
    hr, ci_low, ci_high = cox_hr_ci(time, event, group)
    p_text = "P < 0.0001" if p_val < 0.0001 else f"P = {p_val:.4f}"
    ax.text(0.04, 0.18, p_text, transform=ax.transAxes, fontsize=5.5, fontweight="bold")
    ax.text(0.04, 0.125, f"HR = {hr:.2f} [{ci_low:.2f}-{ci_high:.2f}]", transform=ax.transAxes, fontsize=5.5, fontweight="bold")
    ax.set_title(title, fontweight="bold", pad=5)
    ax.set_xlabel("Time(months)")
    ax.set_ylabel("Survival Probability")
    ax.set_xlim(0, 120)
    ax.set_ylim(0, 1.03)
    ax.set_xticks(TICKS)
    ax.legend(loc="lower left", bbox_to_anchor=(0.04, 0.015), frameon=False, fontsize=5.2, borderaxespad=0, handlelength=1.7, labelspacing=0.15)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    panel_label(ax, label)
    draw_risk_table(ax, risk_items)
    return {"Panel": label, "HR": hr, "HR_low": ci_low, "HR_high": ci_high, "Logrank_p": p_val}


def calculate_dca(y_true, y_prob, thresholds):
    y_true = np.asarray(y_true, dtype=int)
    y_prob = np.asarray(y_prob, dtype=float)
    n = len(y_true)
    net_benefit = []
    for threshold in thresholds:
        pred = y_prob >= threshold
        tp = np.sum((y_true == 1) & pred)
        fp = np.sum((y_true == 0) & pred)
        net_benefit.append((tp / n) - (fp / n) * (threshold / (1.0 - threshold)))
    return np.asarray(net_benefit)


def draw_dca_panel(ax, df):
    y_true = df["status"].to_numpy(dtype=int)
    thresholds = np.linspace(0.01, 0.80, 120)
    nb_bcmpm = calculate_dca(y_true, df["bcmpm_risk60"].to_numpy(dtype=float), thresholds)
    nb_guideline = calculate_dca(y_true, df["Guideline_Risk_binary"].to_numpy(dtype=float), thresholds)
    prevalence = np.mean(y_true)
    nb_none = np.zeros_like(thresholds)
    nb_all = prevalence - (1.0 - prevalence) * thresholds / (1.0 - thresholds)

    window = 13
    nb_bcmpm_smooth = savgol_filter(nb_bcmpm, window, 3, mode="nearest")
    nb_guideline_smooth = savgol_filter(nb_guideline, window, 3, mode="nearest")
    ax.plot(thresholds, nb_bcmpm_smooth, color=COLORS["bcmpm_model"], lw=1.5, label="BCMPM Model")
    ax.plot(thresholds, nb_guideline_smooth, color=COLORS["guideline_model"], lw=1.4, label="Guideline Model")
    ax.plot(thresholds, nb_none, color="black", lw=0.9, ls="--", label="Treat None")
    ax.plot(thresholds, nb_all, color=COLORS["neutral"], lw=0.9, ls="-.", label="Treat All")
    ax.set_title("Clinical utility (DCA)", fontweight="bold", pad=5)
    ax.set_xlabel("Threshold Probability")
    ax.set_ylabel("Net Benefit")
    ax.set_xlim(0, 0.8)
    y_max = max(0.42, float(np.nanmax([nb_bcmpm_smooth.max(), nb_guideline_smooth.max()])) * 1.05)
    ax.set_ylim(-0.02, y_max)
    leg = ax.legend(loc="upper right", frameon=True, fontsize=5.8, handlelength=1.7, borderpad=0.25, labelspacing=0.2)
    leg.get_frame().set_facecolor("white")
    leg.get_frame().set_edgecolor("none")
    leg.get_frame().set_alpha(0.94)
    leg.set_zorder(20)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    panel_label(ax, "e")
    rows = []
    for threshold, bcmpm, guideline, none, all_nb in zip(thresholds, nb_bcmpm, nb_guideline, nb_none, nb_all):
        rows.append({"Threshold": threshold, "BCMPM": bcmpm, "Guideline": guideline, "Treat_None": none, "Treat_All": all_nb})
    return rows


def make_figure4(df, output_dir):
    fig = plt.figure(figsize=(13.4, 7.2))
    gs = fig.add_gridspec(
        2,
        3,
        width_ratios=[1.48, 1.0, 1.0],
        height_ratios=[1.0, 1.0],
        left=0.055,
        right=0.985,
        top=0.94,
        bottom=0.13,
        wspace=0.34,
        hspace=0.70,
    )
    ax_a = fig.add_subplot(gs[:, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[0, 2])
    ax_d = fig.add_subplot(gs[1, 1])
    ax_e = fig.add_subplot(gs[1, 2])

    sankey_counts = draw_sankey(ax_a, df)
    km_rows = []
    km_rows.append(
        draw_km_panel(
            ax_b,
            df,
            "Guideline_Risk_binary",
            "Guideline (St. Gallen)",
            "b",
            legend_labels={0: "Inter&low risk", 1: "High risk"},
        )
    )
    low_intermediate = df[df["Guideline_Risk_binary"] == 0].copy()
    km_rows.append(
        draw_km_panel(
            ax_c,
            low_intermediate,
            "bcmpm_high_risk",
            "Low/intermediate-risk refinement",
            "c",
            legend_labels={0: "BCMPM low risk", 1: "BCMPM high risk"},
        )
    )
    guideline_high = df[df["Guideline_Risk_binary"] == 1].copy()
    km_rows.append(
        draw_km_panel(
            ax_d,
            guideline_high,
            "bcmpm_high_risk",
            "High-risk refinement",
            "d",
            legend_labels={0: "BCMPM low risk", 1: "BCMPM high risk"},
        )
    )
    dca_rows = draw_dca_panel(ax_e, df)

    output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([sankey_counts]).to_csv(output_dir / "figure4_reclassification_counts.csv", index=False)
    pd.DataFrame(km_rows).to_csv(output_dir / "figure4_km_statistics.csv", index=False, float_format="%.5f")
    pd.DataFrame(dca_rows).to_csv(output_dir / "figure4_dca_values.csv", index=False, float_format="%.6f")

    png_path = output_dir / "Fig4_a-e_Nature_600dpi.png"
    svg_path = output_dir / "Fig4_a-e_Nature_600dpi.svg"
    fig.savefig(png_path, dpi=600, bbox_inches="tight")
    fig.savefig(svg_path, bbox_inches="tight")
    plt.close(fig)
    return png_path, svg_path


def main():
    parser = argparse.ArgumentParser(description="Generate publication-ready Figure 4 from St. Gallen guideline and BCMPM risk data.")
    parser.add_argument("--guideline-csv", default="./xxx/xx/total991_Guideline_informed_risk_v3_treatment.csv")
    parser.add_argument("--risk-csv", default="./xxx/xx/total_all_modalities_risk60.csv")
    parser.add_argument("--output-dir", default="./xxx/xx/Fig4_Nature")
    args = parser.parse_args()

    configure_style()
    df = load_data(args.guideline_csv, args.risk_csv)
    output_dir = Path(args.output_dir).resolve()
    png, svg = make_figure4(df, output_dir)
    print(f"Loaded patients: {len(df)}")
    print(f"Saved: {png}")
    print(f"Saved: {svg}")


if __name__ == "__main__":
    main()
