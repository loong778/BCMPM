"""
Generate FigS13-FigS18 subgroup reclassification figures.

The panel-level methods are copied from:
  1. 01_predict_60m_risk.py
     Risk(60) = 1 - S0(60) ** exp(linear_predictor)
  2. Fig5_supplement.py
     Kaplan-Meier curves, risk tables, Sankey reclassification, and DCA.

Outputs
-------
FigS13: Age
FigS14: Tumor Size
FigS15: Nodal_Status
FigS16: Clinical_Stage
FigS17: Histological_Grade
FigS18: Molecular Subtype

Each figure is saved as a 600-dpi PNG plus SVG and PDF vector files.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lifelines import CoxPHFitter, KaplanMeierFitter
from lifelines.statistics import logrank_test, multivariate_logrank_test
from matplotlib.colors import to_rgba
from matplotlib.patches import Rectangle


# -----------------------------------------------------------------------------
# Paths
# -----------------------------------------------------------------------------
# Required input: see README.md for the required file and directory layout.
SCRIPT_DIR = Path(__file__).resolve().parent
# Required input: see README.md for the required file and directory layout.
RISK_DIR = Path("./xxx/xx")  # Required input: train/validation/test all-modality risk CSV files.

# Required input: see README.md for the required file and directory layout.

TRAIN_FILE = RISK_DIR / "train_all_modalities_risk.csv"
# Required input: see README.md for the required file and directory layout.
VALIDATION_FILE = RISK_DIR / "validation_all_modalities_risk.csv"
# Required input: see README.md for the required file and directory layout.
TEST_FILE = RISK_DIR / "test_all_modalities_risk.csv"

# Preferred source of the already calculated 60-month absolute BCMPM risk.
# Required input: see README.md for the required file and directory layout.
RISK60_TOTAL_FILE = Path("./xxx/xx/total_all_modalities_risk60.csv")

# Used only when RISK60_TOTAL_FILE does not provide bcmpm_risk60.
# Required input: see README.md for the required file and directory layout.
BASELINE_60M_FILE = (
    Path("./xxx/xx/baseline_60m.csv")
)

# Example output path; generated files are written here.

OUTPUT_DIR = Path("./xxx/xx")  # Example output directory for generated figures and CSV summaries.

TIME_COL = "DFS"
EVENT_COL = "status"
PATIENT_ID_COL = "patient_id"
BCMPM_LP_COL = "bcmpm_risk_score"
BCMPM_GROUP_COL = "bcmpm_high_risk"
BCMPM_RISK60_COL = "bcmpm_risk60"


# -----------------------------------------------------------------------------
# Figure style copied from Fig5_supplement.py, with 600-dpi raster export.
# -----------------------------------------------------------------------------
plt.rcParams.update(
    {
        "font.family": "Arial",
        "font.sans-serif": ["Arial", "Helvetica"],
        "axes.linewidth": 0.8,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "axes.labelsize": 12,
        "legend.fontsize": 9,
        "savefig.dpi": 600,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
        "figure.dpi": 100,
    }
)


# -----------------------------------------------------------------------------
# Six requested subgroup definitions.
# -----------------------------------------------------------------------------
SUBGROUP_CONFIGS = [
    {
        "figure": "FigS13",
        "key": "Age",
        "name": "Age",
        "column": "age",
        "categories": [
            {"code": 1, "label": "Age \u2264 50", "color": "#1f77b4"},
            {"code": 2, "label": "Age > 50", "color": "#d62728"},
        ],
        "code_map": lambda x: x,
    },
    {
        "figure": "FigS14",
        "key": "Tumor_Size",
        "name": "Tumor Size",
        "column": "tumor size",
        "categories": [
            {"code": 1, "label": "Size \u2264 2 cm", "color": "#1f77b4"},
            {"code": 2, "label": "Size 2-5 cm", "color": "#ff7f0e"},
            {"code": 3, "label": "Size > 5 cm", "color": "#d62728"},
        ],
        "code_map": lambda x: x,
    },
    {
        "figure": "FigS15",
        "key": "Nodal_Status",
        "name": "Nodal Status",
        "column": "lymph node metastasis",
        "categories": [
            {"code": 0, "label": "N0", "color": "#1f77b4"},
            {"code": 1, "label": "N1-3", "color": "#ff7f0e"},
            {"code": 2, "label": "N\u22654", "color": "#d62728"},
        ],
        "code_map": lambda x: x,
    },
    {
        "figure": "FigS16",
        "key": "Clinical_Stage",
        "name": "Clinical Stage",
        "column": "clinical stage",
        "categories": [
            {"code": 1, "label": "Stage I", "color": "#1f77b4"},
            {"code": 2, "label": "Stage II", "color": "#ff7f0e"},
            {"code": 3, "label": "Stage III", "color": "#d62728"},
        ],
        "code_map": lambda x: x,
    },
    {
        "figure": "FigS17",
        "key": "Histological_Grade",
        "name": "Histological Grade",
        "column": "histological grade",
        "categories": [
            {"code": 1, "label": "Grade 1", "color": "#b19cd9"},
            {"code": 2, "label": "Grade 2", "color": "#8dd3c7"},
            {"code": 3, "label": "Grade 3", "color": "#ffd97a"},
        ],
        "code_map": lambda x: x,
    },
    {
        "figure": "FigS18",
        "key": "Molecular_Subtype",
        "name": "Molecular Subtype",
        "column": "molecular subtypes",
        "categories": [
            {"code": 0, "label": "Luminal", "color": "#1f77b4"},
            {"code": 1, "label": "HER2-enriched", "color": "#ff7f0e"},
            {"code": 2, "label": "Triple-negative", "color": "#d62728"},
        ],
        # Luminal A (1) + Luminal B (2) -> Luminal; 3 -> HER2; 4 -> TNBC.
        "code_map": lambda x: 0 if x in [1, 2] else (1 if x == 3 else 2),
    },
]


# -----------------------------------------------------------------------------
# 60-month absolute risk: copied from 01_predict_60m_risk.py.
# -----------------------------------------------------------------------------
def predict_60m_absolute_risk(linear_predictor, baseline_survival_60m):
    linear_predictor = np.asarray(linear_predictor, dtype=float)
    risk60 = 1.0 - np.power(
        float(baseline_survival_60m), np.exp(linear_predictor)
    )
    return np.clip(risk60, 0.0, 1.0)


def load_bcmpm_baseline_survival(path):
    if not path.exists():
        raise FileNotFoundError(
            "Cannot calculate bcmpm_risk60 because the baseline file is missing: "
            f"{path}"
        )
    baseline = pd.read_csv(path)
    if "S0" not in baseline.columns or baseline.empty:
        raise ValueError(f"Expected a non-empty 'S0' column in {path}")
    return float(baseline["S0"].iloc[0])


def load_analysis_data():
    for path in (TRAIN_FILE, VALIDATION_FILE, TEST_FILE):
        if not path.exists():
            raise FileNotFoundError(path)

    train_df = pd.read_csv(TRAIN_FILE)
    validation_df = pd.read_csv(VALIDATION_FILE)
    test_df = pd.read_csv(TEST_FILE)
    total_df = pd.concat(
        [train_df, validation_df, test_df], axis=0, ignore_index=True
    )
    total_df[PATIENT_ID_COL] = total_df[PATIENT_ID_COL].astype(str)

    # Reuse an existing risk60 file when available, as in Fig5_supplement.py.
    if RISK60_TOTAL_FILE.exists():
        risk60_df = pd.read_csv(RISK60_TOTAL_FILE)
        if BCMPM_RISK60_COL in risk60_df.columns:
            risk60_df[PATIENT_ID_COL] = risk60_df[PATIENT_ID_COL].astype(str)
            risk60_df = risk60_df[[PATIENT_ID_COL, BCMPM_RISK60_COL]]
            total_df = total_df.merge(risk60_df, on=PATIENT_ID_COL, how="inner")

    # If the merged file did not supply the column, calculate it exactly as in
    # 01_predict_60m_risk.py.
    if BCMPM_RISK60_COL not in total_df.columns:
        baseline_s0 = load_bcmpm_baseline_survival(BASELINE_60M_FILE)
        total_df[BCMPM_RISK60_COL] = predict_60m_absolute_risk(
            total_df[BCMPM_LP_COL].to_numpy(), baseline_s0
        )

    required = {
        PATIENT_ID_COL,
        TIME_COL,
        EVENT_COL,
        BCMPM_GROUP_COL,
        BCMPM_RISK60_COL,
    }
    required.update(cfg["column"] for cfg in SUBGROUP_CONFIGS)
    missing = sorted(required.difference(total_df.columns))
    if missing:
        raise KeyError(f"Missing required columns: {missing}")

    return total_df


# -----------------------------------------------------------------------------
# Statistical helpers copied from Fig5_supplement.py.
# -----------------------------------------------------------------------------
def calculate_dca(y_true, y_pred_prob, thresholds):
    net_benefits = []
    n = len(y_true)
    for threshold in thresholds:
        y_pred = (y_pred_prob >= threshold).astype(int)
        tp = np.sum((y_true == 1) & (y_pred == 1))
        fp = np.sum((y_true == 0) & (y_pred == 1))
        if threshold == 1:
            net_benefit = 0
        else:
            net_benefit = (
                (tp / n)
                - (fp / n) * (threshold / (1 - threshold))
            )
        net_benefits.append(net_benefit)
    return np.asarray(net_benefits)


def safe_cox_hr(
    df,
    time_col,
    event_col,
    group_col,
    min_group_size=2,
    min_events=2,
):
    low = df[df[group_col] == 0]
    high = df[df[group_col] == 1]

    if min(len(low), len(high)) < min_group_size:
        return None
    if min(low[event_col].sum(), high[event_col].sum()) < min_events:
        return None

    try:
        cph = CoxPHFitter()
        cph.fit(
            df[[time_col, event_col, group_col]],
            duration_col=time_col,
            event_col=event_col,
        )
        hr = cph.hazard_ratios_[group_col]
        ci = np.exp(cph.confidence_intervals_.loc[group_col])
        lower = ci.iloc[0]
        upper = ci.iloc[1]
        invalid = (
            np.isnan(hr)
            or np.isinf(hr)
            or hr > 100
            or np.isnan(lower)
            or np.isnan(upper)
            or np.isinf(lower)
            or np.isinf(upper)
        )
        if invalid:
            return None
        return hr, lower, upper
    except Exception:
        return None


def get_hr_pval(time, event, group):
    group = np.asarray(group).astype(int)
    if len(np.unique(group)) < 2:
        return None

    high = group == 1
    low = group == 0
    logrank = logrank_test(
        time[high], time[low], event[high], event[low]
    )

    cox_df = pd.DataFrame(
        {
            TIME_COL: np.asarray(time),
            EVENT_COL: np.asarray(event),
            "G": group,
        }
    )
    result = safe_cox_hr(cox_df, TIME_COL, EVENT_COL, "G")
    if result is None:
        return np.nan, np.nan, np.nan, logrank.p_value
    hr, lower, upper = result
    return hr, lower, upper, logrank.p_value


def get_multigroup_p(time, event, group):
    group = np.asarray(group).astype(int)
    if len(np.unique(group)) < 2:
        return None
    return multivariate_logrank_test(time, group, event).p_value


def format_p_value(p_value):
    return "P < 0.0001" if p_value < 0.0001 else f"P = {p_value:.4f}"


# -----------------------------------------------------------------------------
# KM risk table copied from Fig5_supplement.py.
# -----------------------------------------------------------------------------
def add_custom_risk_table(ax, kmfs, labels, colors, times):
    y_positions = [-0.25 - i * 0.09 for i in range(len(labels))]
    ax.text(
        -0.26,
        -0.14,
        "At risk",
        transform=ax.transAxes,
        fontsize=9,
        ha="left",
        va="center",
        color="black",
        clip_on=False,
    )
    for index, (kmf, label, color) in enumerate(zip(kmfs, labels, colors)):
        y_position = y_positions[index]
        ax.text(
            -0.26,
            y_position,
            label,
            transform=ax.transAxes,
            fontsize=8.5,
            color=color,
            ha="left",
            va="center",
            clip_on=False,
        )
        for time_point in times:
            number_at_risk = np.sum(kmf.durations >= time_point)
            ax.text(
                time_point,
                y_position,
                str(int(number_at_risk)),
                fontsize=8.5,
                ha="center",
                va="center",
                color=color,
                transform=ax.get_xaxis_transform(),
                clip_on=False,
            )


def style_km_axis(ax):
    ax.set_xlabel("Time after surgery (months)", fontsize=10, fontweight="bold")
    ax.set_ylabel("Disease free survival", fontsize=10, fontweight="bold")
    ax.set_xlim(0, 120)
    ax.set_xticks([0, 30, 60, 90, 120])
    ax.set_ylim(-0.05, 1.05)
    ax.set_yticks([0.00, 0.25, 0.50, 0.75, 1.00])
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def plot_overall_km(ax, df, subgroup_cfg):
    kmfs = []
    labels = []
    colors = []
    time = df[TIME_COL].to_numpy()
    event = df[EVENT_COL].to_numpy()

    for category in subgroup_cfg["categories"]:
        mask = df["subgroup_cat"] == category["code"]
        if mask.sum() < 2:
            continue
        kmf = KaplanMeierFitter()
        kmf.fit(time[mask], event[mask], label=category["label"])
        kmf.plot_survival_function(
            ax=ax,
            ci_show=True,
            ci_alpha=0.25,
            show_censors=True,
            censor_styles={"ms": 5, "marker": "|"},
            color=category["color"],
            linewidth=1.5,
        )
        kmfs.append(kmf)
        labels.append(category["label"])
        colors.append(category["color"])

    p_value = get_multigroup_p(
        time, event, df["subgroup_cat"].to_numpy()
    )
    if p_value is not None:
        ax.text(
            0.02,
            0.18,
            format_p_value(p_value),
            transform=ax.transAxes,
            fontsize=9,
            va="bottom",
        )

    style_km_axis(ax)
    ax.set_title(
        f"{subgroup_cfg['name']}(n={len(df)})",
        fontsize=11,
        fontweight="bold",
        pad=8,
    )
    ax.legend(loc="lower left", bbox_to_anchor=(0.00, -0.03),frameon=False, fontsize=8)
    add_custom_risk_table(
        ax, kmfs, labels, colors, [0, 30, 60, 90, 120]
    )


def plot_bcmpm_km(ax, subgroup_df, category):
    low = subgroup_df[BCMPM_GROUP_COL] == 0
    high = subgroup_df[BCMPM_GROUP_COL] == 1

    kmf_low = KaplanMeierFitter()
    kmf_low.fit(
        subgroup_df.loc[low, TIME_COL],
        subgroup_df.loc[low, EVENT_COL],
        label="BCMPM Low risk",
    )
    kmf_high = KaplanMeierFitter()
    kmf_high.fit(
        subgroup_df.loc[high, TIME_COL],
        subgroup_df.loc[high, EVENT_COL],
        label="BCMPM High risk",
    )

    kmf_low.plot_survival_function(
        ax=ax,
        ci_show=True,
        ci_alpha=0.25,
        show_censors=True,
        censor_styles={"ms": 5, "marker": "|"},
        color="#1f77b4",
        linewidth=1.5,
    )
    kmf_high.plot_survival_function(
        ax=ax,
        ci_show=True,
        ci_alpha=0.25,
        show_censors=True,
        censor_styles={"ms": 5, "marker": "|"},
        color="#d62728",
        linewidth=1.5,
    )

    result = get_hr_pval(
        subgroup_df[TIME_COL].to_numpy(),
        subgroup_df[EVENT_COL].to_numpy(),
        subgroup_df[BCMPM_GROUP_COL].to_numpy(),
    )
    if result is not None:
        hr, lower, upper, p_value = result
        if np.isnan(hr):
            hr_text = "HR = NA (unstable estimate)"
        else:
            hr_text = f"HR = {hr:.2f} ({lower:.2f}-{upper:.2f})"
        ax.text(
            0.02,
            0.14,
            f"{hr_text}\n{format_p_value(p_value)}",
            transform=ax.transAxes,
            fontsize=8.5,
            va="bottom",
        )

    style_km_axis(ax)
    ax.set_title(
        f"BCMPM at {category['label']}(n={len(subgroup_df)})",
        fontsize=11,
        fontweight="bold",
        pad=8,
    )
    ax.legend(loc="lower left", bbox_to_anchor=(0.00, -0.03),frameon=False, fontsize=8)
    add_custom_risk_table(
        ax,
        [kmf_low, kmf_high],
        ["Low risk", "High risk"],
        ["#1f77b4", "#d62728"],
        [0, 30, 60, 90, 120],
    )


# -----------------------------------------------------------------------------
# Sankey ribbon method copied from Fig5_supplement.py.
# -----------------------------------------------------------------------------
def interp_color(color_1, color_2, fraction):
    color_1 = np.asarray(to_rgba(color_1))
    color_2 = np.asarray(to_rgba(color_2))
    return tuple(color_1 * (1 - fraction) + color_2 * fraction)


def bezier(point_0, point_1, point_2, point_3, fraction):
    return (
        (1 - fraction) ** 3 * np.asarray(point_0)
        + 3 * (1 - fraction) ** 2 * fraction * np.asarray(point_1)
        + 3 * (1 - fraction) * fraction**2 * np.asarray(point_2)
        + fraction**3 * np.asarray(point_3)
    )


def draw_ribbon(
    ax,
    x0,
    y0_top,
    y0_bottom,
    x1,
    y1_top,
    y1_bottom,
    color_left,
    color_right,
    alpha=0.6,
    n=200,
    flow_type="curve",
):
    fractions = np.linspace(0, 1, n + 1)
    top = []
    bottom = []
    p0_top = (x0, y0_top)
    p3_top = (x1, y1_top)
    p0_bottom = (x0, y0_bottom)
    p3_bottom = (x1, y1_bottom)
    dx = x1 - x0

    if flow_type == "curve":
        p1_top = (x0 + dx * 0.4, y0_top)
        p2_top = (x0 + dx * 0.6, y1_top)
        p1_bottom = (x0 + dx * 0.4, y0_bottom)
        p2_bottom = (x0 + dx * 0.6, y1_bottom)
    else:
        p1_top = (x0 + dx / 3, y0_top)
        p2_top = (x0 + dx * 2 / 3, y1_top)
        p1_bottom = (x0 + dx / 3, y0_bottom)
        p2_bottom = (x0 + dx * 2 / 3, y1_bottom)

    for fraction in fractions:
        top.append(bezier(p0_top, p1_top, p2_top, p3_top, fraction))
        bottom.append(
            bezier(
                p0_bottom,
                p1_bottom,
                p2_bottom,
                p3_bottom,
                fraction,
            )
        )

    top = np.asarray(top)
    bottom = np.asarray(bottom)
    for index in range(n):
        polygon = np.vstack(
            [top[index], top[index + 1], bottom[index + 1], bottom[index]]
        )
        color = interp_color(color_left, color_right, index / n)
        ax.add_patch(
            plt.Polygon(
                polygon,
                closed=True,
                facecolor=color,
                edgecolor="none",
                alpha=alpha,
            )
        )


def plot_reclassification_sankey(
    ax,
    df,
    subgroup_cfg,
    compact=False,
):
    """
    Plot subgroup -> BCMPM risk reclassification Sankey.

    Parameters
    ----------
    compact : bool
        False:
            Original Sankey layout, suitable for FigS13 where panel a
            occupies the full left column across two rows.

        True:
            Compact/full-panel Sankey layout for FigS14-FigS18.
            The Sankey fills a standard subplot cell so that panel a
            has the same visual size as panels b-f.
    """

    categories = subgroup_cfg["categories"]
    category_codes = [
        category["code"]
        for category in categories
    ]

    cross_tab = pd.crosstab(
        df["subgroup_cat"],
        df[BCMPM_GROUP_COL]
    )

    cross_tab = cross_tab.reindex(
        index=category_codes,
        columns=[0, 1],
        fill_value=0
    )

    right_labels = [
        "Low risk",
        "High risk"
    ]

    right_colors = [
        "#1f77b4",
        "#d62728"
    ]

    right_codes = [0, 1]

    left_counts = {
        category["code"]:
        int(
            cross_tab.loc[
                category["code"]
            ].sum()
        )
        for category in categories
    }

    right_counts = {
        code: int(cross_tab[code].sum())
        for code in right_codes
    }

    total_n = len(df)

    # ============================================================
    # Axis
    # ============================================================
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    ax.set_title(
        "Risk reclassification",
        fontsize=11,
        fontweight="bold",
        pad=8
    )

    # ============================================================
    # Layout parameters
    # ============================================================
    if compact:
        # --------------------------------------------------------
        # FigS14-FigS18:
        # Make Sankey fill almost the entire subplot.
        # This produces an a-panel visually comparable to KM/DCA.
        # --------------------------------------------------------
        x_left = 0.08
        x_right = 0.92

        bar_width = 0.035

        top_y = 0.95
        bottom_y = 0.08

        usable_height = top_y - bottom_y

        # Small category gap because the panel is shorter.
        gap = 0.007

        total_gap = (
            len(categories) - 1
        ) * gap

        scale = (
            usable_height - total_gap
        ) / total_n

        left_label_x = x_left + 0.055
        right_label_x = x_right - 0.055

        left_side_label_x = 0.035
        right_side_label_x = 0.965

        node_fontsize = 8.5
        side_fontsize = 10

    else:
        # --------------------------------------------------------
        # FigS13:
        # Original wider/taller Sankey layout.
        # --------------------------------------------------------
        x_left = 0.15
        x_right = 0.85

        bar_width = 0.025

        scale = 0.80 / total_n
        top_y = 0.90

        gap = 0.005

        left_label_x = x_left + 0.04
        right_label_x = x_right - 0.04

        left_side_label_x = 0.08
        right_side_label_x = 0.92

        node_fontsize = 9.5
        side_fontsize = 11

    # ============================================================
    # Left node positions
    # ============================================================
    left_y_positions = {}

    current_y = top_y

    for category in categories:

        code = category["code"]

        height = (
            left_counts[code] * scale
        )

        y1 = current_y
        y0 = y1 - height

        left_y_positions[code] = (
            y0,
            y1
        )

        current_y = y0 - gap

    # ============================================================
    # Right node positions
    # ============================================================
    right_y_positions = {}

    current_y = top_y

    for code in right_codes:

        height = (
            right_counts[code] * scale
        )

        y1 = current_y
        y0 = y1 - height

        right_y_positions[code] = (
            y0,
            y1
        )

        current_y = y0 - gap

    # ============================================================
    # Build flows
    # ============================================================
    flows = []

    left_current_top = top_y

    for category in categories:

        code = category["code"]

        number_to_low = cross_tab.loc[
            code,
            0
        ]

        number_to_high = cross_tab.loc[
            code,
            1
        ]

        left_low_top = left_current_top

        left_low_bottom = (
            left_low_top
            - number_to_low * scale
        )

        left_high_top = left_low_bottom

        left_high_bottom = (
            left_high_top
            - number_to_high * scale
        )

        left_current_top = (
            left_high_bottom
            - gap
        )

        flows.append(
            {
                "right_code": 0,
                "left_top": left_low_top,
                "left_bottom": left_low_bottom,
                "color_left": category["color"],
                "color_right": right_colors[0],
            }
        )

        flows.append(
            {
                "right_code": 1,
                "left_top": left_high_top,
                "left_bottom": left_high_bottom,
                "color_left": category["color"],
                "color_right": right_colors[1],
            }
        )

    # ============================================================
    # Right offsets
    # ============================================================
    right_offsets = {
        code:
        right_y_positions[code][1]
        for code in right_codes
    }

    for flow in flows:

        right_code = flow["right_code"]

        height = (
            flow["left_top"]
            - flow["left_bottom"]
        )

        flow["right_top"] = (
            right_offsets[right_code]
        )

        flow["right_bottom"] = (
            right_offsets[right_code]
            - height
        )

        right_offsets[right_code] -= height

    # ============================================================
    # Draw ribbons
    # ============================================================
    for flow in flows:

        draw_ribbon(
            ax,
            x_left,
            flow["left_top"],
            flow["left_bottom"],
            x_right,
            flow["right_top"],
            flow["right_bottom"],
            flow["color_left"],
            flow["color_right"],
            alpha=0.55,
            n=300,
            flow_type="curve",
        )

    # ============================================================
    # Left bars and labels
    # ============================================================
    for category in categories:

        code = category["code"]

        y0, y1 = (
            left_y_positions[code]
        )

        ax.add_patch(
            Rectangle(
                (
                    x_left
                    - bar_width / 2,
                    y0
                ),
                bar_width,
                y1 - y0,
                facecolor=category["color"],
                alpha=0.95,
                zorder=10,
            )
        )

        ax.text(
            left_label_x,
            (y0 + y1) / 2,
            (
                f"{category['label']}\n"
                f"(n={left_counts[code]})"
            ),
            ha="left",
            va="center",
            fontsize=node_fontsize,
            fontweight="bold",
        )

    # ============================================================
    # Right bars and labels
    # ============================================================
    for index, code in enumerate(
        right_codes
    ):

        y0, y1 = (
            right_y_positions[code]
        )

        ax.add_patch(
            Rectangle(
                (
                    x_right
                    - bar_width / 2,
                    y0
                ),
                bar_width,
                y1 - y0,
                facecolor=right_colors[index],
                alpha=0.95,
                zorder=10,
            )
        )

        ax.text(
            right_label_x,
            (y0 + y1) / 2,
            (
                f"{right_labels[index]}\n"
                f"(n={right_counts[code]})"
            ),
            ha="right",
            va="center",
            fontsize=node_fontsize,
            fontweight="bold",
        )

    # ============================================================
    # Vertical side labels
    # ============================================================
    ax.text(
        left_side_label_x,
        0.5,
        subgroup_cfg["name"],
        rotation=90,
        ha="center",
        va="center",
        fontsize=side_fontsize,
        fontweight="bold",
        transform=ax.transAxes,
    )

    ax.text(
        right_side_label_x,
        0.5,
        "BCMPM",
        rotation=270,
        ha="center",
        va="center",
        fontsize=side_fontsize,
        fontweight="bold",
        transform=ax.transAxes,
    )


# -----------------------------------------------------------------------------
# DCA copied from Fig5_supplement.py.
# -----------------------------------------------------------------------------
def plot_dca(ax, df, subgroup_cfg):
    number_of_categories = len(subgroup_cfg["categories"])
    y_true = df[EVENT_COL].to_numpy()
    bcmpm_probability = df[BCMPM_RISK60_COL].to_numpy()

    # Kept exactly compatible with Fig5_supplement.py.
    subgroup_probability = (
        df["subgroup_cat"].to_numpy() / (number_of_categories - 1)
    )

    thresholds = np.linspace(0.01, 0.99, 50)
    nb_bcmpm = calculate_dca(y_true, bcmpm_probability, thresholds)
    nb_subgroup = calculate_dca(y_true, subgroup_probability, thresholds)
    nb_none = np.zeros_like(thresholds)
    event_rate = y_true.sum() / len(y_true)
    nb_all = event_rate - (1 - event_rate) * (
        thresholds / (1 - thresholds)
    )

    ax.plot(
        thresholds,
        nb_bcmpm,
        label="BCMPM Model",
        color="#d62728",
        linewidth=2,
    )
    ax.plot(
        thresholds,
        nb_subgroup,
        label=f"{subgroup_cfg['name']} Model",
        color=subgroup_cfg["categories"][0]["color"],
        linewidth=2,
    )
    ax.plot(
        thresholds,
        nb_none,
        label="Treat None",
        color="black",
        linestyle="--",
    )
    ax.plot(
        thresholds,
        nb_all,
        label="Treat All",
        color="grey",
        linestyle="--",
    )

    ax.set_xlim(0, 0.8)
    y_max = max(nb_bcmpm.max(), nb_subgroup.max())
    ax.set_ylim(-0.05, y_max * 1.1 if y_max > 0 else 0.2)
    ax.set_xlabel("Threshold Probability", fontsize=10, fontweight="bold")
    ax.set_ylabel("Net Benefit", fontsize=10, fontweight="bold")
    ax.set_title("Clinical utility (DCA)", fontsize=11, fontweight="bold", pad=8)
    ax.legend(loc="upper right", frameon=False, fontsize=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


# -----------------------------------------------------------------------------
# Composite layout.
# Binary subgroup: reference layout (Sankey left, 2 x 2 panels right).
# Three-category subgroup: Sankey left, 2 x 3 panels right, preserving all KMs.
# -----------------------------------------------------------------------------
def add_panel_label(ax, label, x=-0.18, y=1.12):
    ax.text(
        x,
        y,
        label,
        transform=ax.transAxes,
        fontsize=14,
        fontweight="bold",
        ha="left",
        va="top",
        clip_on=False,
    )


def prepare_subgroup_data(total_df, subgroup_cfg):
    df = total_df.copy()
    df["subgroup_cat"] = df[subgroup_cfg["column"]].map(
        subgroup_cfg["code_map"]
    )
    df = df.dropna(
        subset=[
            "subgroup_cat",
            TIME_COL,
            EVENT_COL,
            BCMPM_GROUP_COL,
            BCMPM_RISK60_COL,
        ]
    )
    df["subgroup_cat"] = df["subgroup_cat"].astype(int)
    category_codes = [
        category["code"] for category in subgroup_cfg["categories"]
    ]
    df = df[df["subgroup_cat"].isin(category_codes)].copy()

    if df["subgroup_cat"].nunique() < 2:
        raise ValueError(
            f"{subgroup_cfg['name']} has fewer than two valid categories."
        )
    return df


def create_subgroup_figure(
    total_df,
    subgroup_cfg
):
    """
    Create subgroup supplementary figures.

    FigS13 Age:
        a | b | c
        a | d | e

    FigS14-FigS18:
        a | b | c
        d | e | f

    For FigS14-FigS18 all six subplot cells have identical
    physical dimensions.
    """

    df = prepare_subgroup_data(
        total_df,
        subgroup_cfg
    )

    categories = (
        subgroup_cfg["categories"]
    )

    number_of_categories = len(
        categories
    )

    # ================================================================
    # FigS13: binary subgroup
    # ================================================================
    if number_of_categories == 2:

        fig = plt.figure(
            figsize=(16.5, 8.8)
        )

        grid = fig.add_gridspec(
            nrows=2,
            ncols=3,

            # Sankey column slightly wider
            width_ratios=[
                1.35,
                1.0,
                1.0
            ],

            height_ratios=[
                1.0,
                1.0
            ],

            hspace=0.78,
            wspace=0.34,
        )

        # ------------------------------------------------------------
        # a
        # ------------------------------------------------------------
        sankey_ax = fig.add_subplot(
            grid[:, 0]
        )

        plot_reclassification_sankey(
            sankey_ax,
            df,
            subgroup_cfg,
            compact=False,
        )

        add_panel_label(
            sankey_ax,
            "a",
            x=-0.04,
            y=1.03
        )

        # ------------------------------------------------------------
        # b
        # ------------------------------------------------------------
        overall_ax = fig.add_subplot(
            grid[0, 1]
        )

        plot_overall_km(
            overall_ax,
            df,
            subgroup_cfg
        )

        add_panel_label(
            overall_ax,
            "b"
        )

        # ------------------------------------------------------------
        # c
        # ------------------------------------------------------------
        category = categories[0]

        subgroup_df = df[
            df["subgroup_cat"]
            == category["code"]
        ].copy()

        if (
            len(subgroup_df) < 10
            or subgroup_df[
                BCMPM_GROUP_COL
            ].nunique() < 2
        ):
            raise ValueError(
                f"{subgroup_cfg['name']} / "
                f"{category['label']} cannot "
                "be plotted because the subgroup "
                "is too small or contains only "
                "one BCMPM risk group."
            )

        km_ax_c = fig.add_subplot(
            grid[0, 2]
        )

        plot_bcmpm_km(
            km_ax_c,
            subgroup_df,
            category
        )

        add_panel_label(
            km_ax_c,
            "c"
        )

        # ------------------------------------------------------------
        # d
        # ------------------------------------------------------------
        category = categories[1]

        subgroup_df = df[
            df["subgroup_cat"]
            == category["code"]
        ].copy()

        if (
            len(subgroup_df) < 10
            or subgroup_df[
                BCMPM_GROUP_COL
            ].nunique() < 2
        ):
            raise ValueError(
                f"{subgroup_cfg['name']} / "
                f"{category['label']} cannot "
                "be plotted because the subgroup "
                "is too small or contains only "
                "one BCMPM risk group."
            )

        km_ax_d = fig.add_subplot(
            grid[1, 1]
        )

        plot_bcmpm_km(
            km_ax_d,
            subgroup_df,
            category
        )

        add_panel_label(
            km_ax_d,
            "d"
        )

        # ------------------------------------------------------------
        # e
        # ------------------------------------------------------------
        dca_ax = fig.add_subplot(
            grid[1, 2]
        )

        plot_dca(
            dca_ax,
            df,
            subgroup_cfg
        )

        add_panel_label(
            dca_ax,
            "e"
        )

        fig.subplots_adjust(
            left=0.045,
            right=0.985,
            top=0.94,
            bottom=0.13,
        )

        # Sankey vertically compressed and centered
        pos = sankey_ax.get_position()

        new_height = (
            pos.height * 0.62
        )

        new_bottom = (
            pos.y0
            + (
                pos.height
                - new_height
            ) / 2
        )

        sankey_ax.set_position(
            [
                pos.x0,
                new_bottom,
                pos.width,
                new_height
            ]
        )

        return fig, len(df)


    #
    # Therefore a has exactly the same subplot size as b/c/d/e/f.
    # ================================================================
    elif number_of_categories == 3:

        fig = plt.figure(
            figsize=(16.5, 9.2)
        )

        grid = fig.add_gridspec(
            nrows=2,
            ncols=3,

            # ----------------------------------------
            # Equal column widths
            # ----------------------------------------
            width_ratios=[
                1.0,
                1.0,
                1.0
            ],

            # ----------------------------------------
            # Equal row heights
            # ----------------------------------------
            height_ratios=[
                1.0,
                1.0
            ],

            hspace=0.78,
            wspace=0.34,
        )

        # ============================================================
        # a — Sankey
        # ============================================================
        sankey_ax = fig.add_subplot(
            grid[0, 0]
        )

        plot_reclassification_sankey(
            sankey_ax,
            df,
            subgroup_cfg,

            # CRITICAL:
            # fill one standard subplot cell
            compact=True,
        )

        add_panel_label(
            sankey_ax,
            "a",
            x=-0.18,
            y=1.12
        )

        # ============================================================
        # b — Overall KM
        # ============================================================
        overall_ax = fig.add_subplot(
            grid[0, 1]
        )

        plot_overall_km(
            overall_ax,
            df,
            subgroup_cfg
        )

        add_panel_label(
            overall_ax,
            "b"
        )

        # ============================================================
        # c — first subgroup
        # ============================================================
        category = categories[0]

        subgroup_df = df[
            df["subgroup_cat"]
            == category["code"]
        ].copy()

        if (
            len(subgroup_df) < 10
            or subgroup_df[
                BCMPM_GROUP_COL
            ].nunique() < 2
        ):
            raise ValueError(
                f"{subgroup_cfg['name']} / "
                f"{category['label']} cannot "
                "be plotted because the subgroup "
                "is too small or contains only "
                "one BCMPM risk group."
            )

        km_ax_c = fig.add_subplot(
            grid[0, 2]
        )

        plot_bcmpm_km(
            km_ax_c,
            subgroup_df,
            category
        )

        add_panel_label(
            km_ax_c,
            "c"
        )

        # ============================================================
        # d — second subgroup
        # ============================================================
        category = categories[1]

        subgroup_df = df[
            df["subgroup_cat"]
            == category["code"]
        ].copy()

        if (
            len(subgroup_df) < 10
            or subgroup_df[
                BCMPM_GROUP_COL
            ].nunique() < 2
        ):
            raise ValueError(
                f"{subgroup_cfg['name']} / "
                f"{category['label']} cannot "
                "be plotted because the subgroup "
                "is too small or contains only "
                "one BCMPM risk group."
            )

        km_ax_d = fig.add_subplot(
            grid[1, 0]
        )

        plot_bcmpm_km(
            km_ax_d,
            subgroup_df,
            category
        )

        add_panel_label(
            km_ax_d,
            "d"
        )

        # ============================================================
        # e — third subgroup
        # ============================================================
        category = categories[2]

        subgroup_df = df[
            df["subgroup_cat"]
            == category["code"]
        ].copy()

        if (
            len(subgroup_df) < 10
            or subgroup_df[
                BCMPM_GROUP_COL
            ].nunique() < 2
        ):
            raise ValueError(
                f"{subgroup_cfg['name']} / "
                f"{category['label']} cannot "
                "be plotted because the subgroup "
                "is too small or contains only "
                "one BCMPM risk group."
            )

        km_ax_e = fig.add_subplot(
            grid[1, 1]
        )

        plot_bcmpm_km(
            km_ax_e,
            subgroup_df,
            category
        )

        add_panel_label(
            km_ax_e,
            "e"
        )

        # ============================================================
        # f — DCA
        # ============================================================
        dca_ax = fig.add_subplot(
            grid[1, 2]
        )

        plot_dca(
            dca_ax,
            df,
            subgroup_cfg
        )

        add_panel_label(
            dca_ax,
            "f"
        )

        # ============================================================
        # Global margins
        # ============================================================
        fig.subplots_adjust(
            left=0.055,
            right=0.985,
            top=0.94,
            bottom=0.13,
        )

        return fig, len(df)

    else:

        raise ValueError(
            "This script supports only "
            "two- or three-category subgroups."
        )


def save_figure(fig, subgroup_cfg):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    stem = f"{subgroup_cfg['figure']}_{subgroup_cfg['key']}"
    fig.savefig(
        OUTPUT_DIR / f"{stem}_600dpi.png",
        dpi=600,
        bbox_inches="tight",
        facecolor="white",
    )
    fig.savefig(
        OUTPUT_DIR / f"{stem}.svg",
        bbox_inches="tight",
        facecolor="white",
    )
    fig.savefig(
        OUTPUT_DIR / f"{stem}.pdf",
        bbox_inches="tight",
        facecolor="white",
    )
    plt.close(fig)


def main():
    total_df = load_analysis_data()
    print(f"Loaded total cohort: n={len(total_df)}")
    print(f"Output directory: {OUTPUT_DIR}")

    for subgroup_cfg in SUBGROUP_CONFIGS:
        fig, valid_n = create_subgroup_figure(total_df, subgroup_cfg)
        save_figure(fig, subgroup_cfg)
        print(
            f"Saved {subgroup_cfg['figure']} ({subgroup_cfg['name']}), "
            f"valid n={valid_n}"
        )


if __name__ == "__main__":
    main()




