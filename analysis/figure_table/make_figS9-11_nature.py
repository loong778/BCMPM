import os
import string
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from lifelines import KaplanMeierFitter, CoxPHFitter
from lifelines.statistics import logrank_test
from matplotlib.lines import Line2D


# ============================================================
# 1. Global settings
# ============================================================

plt.rcParams.update({
    'font.family': 'Arial',
    'font.sans-serif': ['Arial', 'Helvetica'],

    'axes.linewidth': 0.7,

    'xtick.direction': 'out',
    'ytick.direction': 'out',

    'xtick.major.width': 0.7,
    'ytick.major.width': 0.7,

    'pdf.fonttype': 42,
    'ps.fonttype': 42,

    'savefig.dpi': 600,
    'figure.dpi': 120
})


# ============================================================
# 2. Paths
# ============================================================

# Required input: see README.md for the required file and directory layout.

SCRIPT_DIR = './xxx/xx'


# Example output path; generated files are written here.


OUTPUT_DIR = './xxx/xx'
os.makedirs(OUTPUT_DIR, exist_ok=True)

TIME_COL = "DFS"
EVENT_COL = "status"
GROUP_COL = "bcmpm_high_risk"


# ============================================================
# 3. Subgroups
# ============================================================

SUBGROUPS = {

    'Age ≤ 50':
        lambda df: df['age'] == 1,

    'Age > 50':
        lambda df: df['age'] == 2,

    'Tumor Size ≤ 2cm':
        lambda df: df['tumor size'] == 1,

    'Tumor Size 2-5cm':
        lambda df: df['tumor size'] == 2,

    'Tumor Size > 5cm':
        lambda df: df['tumor size'] == 3,

    'Luminal A':
        lambda df: df['molecular subtypes'] == 1,

    'Luminal B':
        lambda df: df['molecular subtypes'] == 2,

    'HER2-enriched':
        lambda df: df['molecular subtypes'] == 3,

    'Triple-negative':
        lambda df: df['molecular subtypes'] == 4,

    'N0':
        lambda df: df['lymph node metastasis'] == 0,

    'N1-3':
        lambda df: df['lymph node metastasis'] == 1,

    'N≥4':
        lambda df: df['lymph node metastasis'] == 2,

    'Stage I':
        lambda df: df['clinical stage'] == 1,

    'Stage II':
        lambda df: df['clinical stage'] == 2,

    'Stage III':
        lambda df: df['clinical stage'] == 3,

    'G1':
        lambda df: df['histological grade'] == 1,

    'G2':
        lambda df: df['histological grade'] == 2,

    'G3':
        lambda df: df['histological grade'] == 3,
}


# ============================================================
# 4. Panel orders
# ============================================================

# Fig. S9 / Fig. S10
# 5 rows × 4 columns
# 18 panels + 2 blank cells

PANEL_ORDER_18 = [
    'Age ≤ 50',
    'Age > 50',
    'Tumor Size ≤ 2cm',
    'Tumor Size 2-5cm',

    'Tumor Size > 5cm',
    'Luminal A',
    'Luminal B',
    'HER2-enriched',

    'Triple-negative',
    'N0',
    'N1-3',
    'N≥4',

    'Stage I',
    'Stage II',
    'Stage III',
    'G1',

    'G2',
    'G3'
]


# Fig. S11
# 4 rows × 4 columns
# 16 panels
#
# Remove:
#   Tumor Size > 5cm
#   G1

PANEL_ORDER_16_EXTERNAL = [
    'Age ≤ 50',
    'Age > 50',
    'Tumor Size ≤ 2cm',
    'Tumor Size 2-5cm',

    'Luminal A',
    'Luminal B',
    'HER2-enriched',
    'Triple-negative',

    'N0',
    'N1-3',
    'N≥4',
    'Stage I',

    'Stage II',
    'Stage III',
    'G2',
    'G3'
]


DISPLAY_NAMES = {

    'Age ≤ 50': 'Age ≤ 50',
    'Age > 50': 'Age > 50',

    'Tumor Size ≤ 2cm': 'Size ≤ 2 cm',
    'Tumor Size 2-5cm': 'Size 2–5 cm',
    'Tumor Size > 5cm': 'Size > 5 cm',

    'Luminal A': 'Luminal A',
    'Luminal B': 'Luminal B',
    'HER2-enriched': 'HER2-enriched',
    'Triple-negative': 'Triple-negative',

    'N0': 'N0',
    'N1-3': 'N1–3',
    'N≥4': 'N≥4',

    'Stage I': 'Stage I',
    'Stage II': 'Stage II',
    'Stage III': 'Stage III',

    'G1': 'G1',
    'G2': 'G2',
    'G3': 'G3'
}


# ============================================================
# 5. Colors
# ============================================================

COLOR_LOW = '#1f77b4'
COLOR_HIGH = '#d62728'


# ============================================================
# 6. Safe Cox HR
# ============================================================

def safe_cox_hr(
        df,
        time_col,
        event_col,
        group_col,
        min_group_size=2,
        min_events=2):

    low = df[df[group_col] == 0]
    high = df[df[group_col] == 1]

    if len(low) < min_group_size or len(high) < min_group_size:
        return None

    event_low = low[event_col].sum()
    event_high = high[event_col].sum()

    if event_low < min_events or event_high < min_events:
        return None

    try:

        cph = CoxPHFitter()

        cph.fit(
            df[[time_col, event_col, group_col]],
            duration_col=time_col,
            event_col=event_col
        )

        hr = cph.hazard_ratios_[group_col]

        ci = np.exp(
            cph.confidence_intervals_.loc[group_col]
        )

        lower = ci.iloc[0]
        upper = ci.iloc[1]

        if (
            np.isnan(hr)
            or np.isinf(hr)
            or np.isnan(lower)
            or np.isnan(upper)
            or np.isinf(lower)
            or np.isinf(upper)
            or hr > 100
        ):
            return None

        return hr, lower, upper

    except Exception:
        return None


# ============================================================
# 7. Compact at-risk table
# ============================================================

def add_compact_risk_table(
        ax,
        kmf_low,
        kmf_high,
        times=(0, 30, 60, 90, 120)):

    """
    Compact risk table designed for multi-panel Supplementary Figures.
    """

    y_header = -0.20
    y_low = -0.31
    y_high = -0.42

    # --------------------------------------------------------
    # At risk header
    # --------------------------------------------------------

    ax.text(
        -0.12,
        y_header,
        "At risk",
        transform=ax.transAxes,
        fontsize=5.3,
        ha='left',
        va='center',
        color='black',
        clip_on=False
    )

    # --------------------------------------------------------
    # Low / High labels
    # --------------------------------------------------------

    ax.text(
        -0.12,
        y_low,
        "Low",
        transform=ax.transAxes,
        fontsize=5.3,
        ha='left',
        va='center',
        color=COLOR_LOW,
        clip_on=False
    )

    ax.text(
        -0.12,
        y_high,
        "High",
        transform=ax.transAxes,
        fontsize=5.3,
        ha='left',
        va='center',
        color=COLOR_HIGH,
        clip_on=False
    )

    # --------------------------------------------------------
    # Numbers
    # --------------------------------------------------------

    for t in times:

        low_n = np.sum(kmf_low.durations >= t)
        high_n = np.sum(kmf_high.durations >= t)

        ax.text(
            t,
            y_low,
            str(int(low_n)),
            transform=ax.get_xaxis_transform(),
            fontsize=5.3,
            color=COLOR_LOW,
            ha='center',
            va='center',
            clip_on=False
        )

        ax.text(
            t,
            y_high,
            str(int(high_n)),
            transform=ax.get_xaxis_transform(),
            fontsize=5.3,
            color=COLOR_HIGH,
            ha='center',
            va='center',
            clip_on=False
        )


# ============================================================
# 8. Draw one KM panel
# ============================================================

def plot_km_panel(
        ax,
        df,
        subgroup_name,
        panel_letter):

    # --------------------------------------------------------
    # Subgroup selection
    # --------------------------------------------------------

    try:

        mask = SUBGROUPS[subgroup_name](df)

        sg_df = df.loc[mask].copy()

    except Exception:

        sg_df = pd.DataFrame()

    sg_df = sg_df.dropna(
        subset=[
            TIME_COL,
            EVENT_COL,
            GROUP_COL
        ]
    )

    title_name = DISPLAY_NAMES[subgroup_name]

    # --------------------------------------------------------
    # Panel label
    # --------------------------------------------------------

    ax.text(
        -0.18,
        1.10,
        panel_letter,
        transform=ax.transAxes,
        fontsize=8,
        fontweight='bold',
        ha='left',
        va='top',
        clip_on=False
    )

    # --------------------------------------------------------
    # No data
    # --------------------------------------------------------

    if len(sg_df) == 0:

        ax.set_title(
            f"{title_name} (n=0)",
            fontsize=7.2,
            fontweight='bold',
            pad=3
        )

        ax.text(
            0.5,
            0.5,
            "No data",
            transform=ax.transAxes,
            ha='center',
            va='center',
            fontsize=6
        )

        ax.set_xlim(0, 120)
        ax.set_ylim(0, 1.05)

        ax.set_xticks(
            [0, 30, 60, 90, 120]
        )

        ax.set_yticks(
            [0.00, 0.25, 0.50, 0.75, 1.00]
        )

        return

    low = sg_df[
        sg_df[GROUP_COL] == 0
    ]

    high = sg_df[
        sg_df[GROUP_COL] == 1
    ]

    # --------------------------------------------------------
    # Title
    # --------------------------------------------------------

    ax.set_title(
        f"{title_name} (n={len(sg_df)})",
        fontsize=7.2,
        fontweight='bold',
        pad=3
    )

    # --------------------------------------------------------
    # Insufficient group size
    # --------------------------------------------------------

    if len(low) < 2 or len(high) < 2:

        ax.text(
            0.5,
            0.5,
            "Insufficient group size",
            transform=ax.transAxes,
            ha='center',
            va='center',
            fontsize=6
        )

        ax.set_xlim(0, 120)
        ax.set_ylim(0, 1.05)

        ax.set_xticks(
            [0, 30, 60, 90, 120]
        )

        ax.set_yticks(
            [0.00, 0.25, 0.50, 0.75, 1.00]
        )

        return

    # --------------------------------------------------------
    # KM fitting
    # --------------------------------------------------------

    kmf_low = KaplanMeierFitter()
    kmf_high = KaplanMeierFitter()

    kmf_low.fit(
        low[TIME_COL],
        event_observed=low[EVENT_COL],
        label='Low risk'
    )

    kmf_high.fit(
        high[TIME_COL],
        event_observed=high[EVENT_COL],
        label='High risk'
    )

    # --------------------------------------------------------
    # KM plots
    # --------------------------------------------------------

    kmf_low.plot_survival_function(
        ax=ax,
        ci_show=True,
        ci_alpha=0.20,
        show_censors=True,
        censor_styles={
            'ms': 3.5,
            'marker': '|',
            'mew': 0.8
        },
        color=COLOR_LOW,
        linewidth=1.15,
        legend=False
    )

    kmf_high.plot_survival_function(
        ax=ax,
        ci_show=True,
        ci_alpha=0.20,
        show_censors=True,
        censor_styles={
            'ms': 3.5,
            'marker': '|',
            'mew': 0.8
        },
        color=COLOR_HIGH,
        linewidth=1.15,
        legend=False
    )

    # --------------------------------------------------------
    # Log-rank
    # --------------------------------------------------------

    lr = logrank_test(
        low[TIME_COL],
        high[TIME_COL],
        event_observed_A=low[EVENT_COL],
        event_observed_B=high[EVENT_COL]
    )

    p = lr.p_value

    if p < 0.0001:

        p_text = r"$P$ < 0.0001"

    elif p < 0.001:

        p_text = rf"$P$ = {p:.4f}"

    else:

        p_text = rf"$P$ = {p:.3f}"

    # --------------------------------------------------------
    # HR
    # --------------------------------------------------------

    hr_result = safe_cox_hr(
        sg_df,
        TIME_COL,
        EVENT_COL,
        GROUP_COL
    )

    if hr_result is None:

        stat_text = (
            "HR = NA\n"
            f"{p_text}"
        )

    else:

        hr, lower, upper = hr_result

        stat_text = (
            f"HR = {hr:.2f}\n"
            f"({lower:.2f}–{upper:.2f})\n"
            f"{p_text}"
        )

    # --------------------------------------------------------
    # Statistics annotation
    # --------------------------------------------------------

    ax.text(
        0.04,
        0.06,
        stat_text,
        transform=ax.transAxes,
        fontsize=5.5,
        ha='left',
        va='bottom',
        linespacing=1.05
    )

    # --------------------------------------------------------
    # Axes
    # --------------------------------------------------------

    ax.set_xlim(
        0,
        120
    )

    ax.set_xticks(
        [0, 30, 60, 90, 120]
    )

    ax.set_ylim(
        0,
        1.05
    )

    ax.set_yticks(
        [0.00, 0.25, 0.50, 0.75, 1.00]
    )

    ax.tick_params(
        axis='both',
        labelsize=5.5,
        length=2.5,
        pad=1.5
    )

    ax.set_xlabel("")
    ax.set_ylabel("")

    # --------------------------------------------------------
    # Nature-like frame
    # --------------------------------------------------------

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    ax.spines['bottom'].set_linewidth(0.7)
    ax.spines['left'].set_linewidth(0.7)

    # --------------------------------------------------------
    # At-risk table
    # --------------------------------------------------------

    add_compact_risk_table(
        ax,
        kmf_low,
        kmf_high,
        times=(0, 30, 60, 90, 120)
    )


# ============================================================
# 9. Cohort-level figure
# ============================================================

def create_cohort_figure(
        df,
        cohort_name,
        figure_number,
        output_dir):

    print(
        f"\nGenerating {figure_number}: "
        f"{cohort_name}, n={len(df)}"
    )

    # ========================================================
    # Fig. S9 / Fig. S10
    # 5 rows × 4 columns
    # ========================================================

    if figure_number in [
        "FigS9",
        "FigS10"
    ]:

        n_rows = 5
        n_cols = 4

        panel_order = PANEL_ORDER_18

        # Wider landscape-like format
        figsize = (
            9.2,
            10.5
        )

        layout_adjust = {
            "left": 0.070,
            "right": 0.985,
            "top": 0.950,
            "bottom": 0.075,
            "wspace": 0.34,
            "hspace": 0.92
        }

    # ========================================================
    # Fig. S11
    # 4 rows × 4 columns
    # ========================================================

    elif figure_number == "FigS11":

        n_rows = 4
        n_cols = 4

        panel_order = PANEL_ORDER_16_EXTERNAL

        figsize = (
            9.2,
            8.8
        )

        layout_adjust = {
            "left": 0.070,
            "right": 0.985,
            "top": 0.945,
            "bottom": 0.085,
            "wspace": 0.34,
            "hspace": 0.92
        }

    # ========================================================
    # Fallback
    # ========================================================

    else:

        n_rows = 5
        n_cols = 4

        panel_order = PANEL_ORDER_18

        figsize = (
            9.2,
            10.5
        )

        layout_adjust = {
            "left": 0.070,
            "right": 0.985,
            "top": 0.950,
            "bottom": 0.075,
            "wspace": 0.34,
            "hspace": 0.92
        }

    # --------------------------------------------------------
    # Create subplot grid
    # --------------------------------------------------------

    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=figsize
    )

    axes = np.asarray(
        axes
    ).flatten()

    # --------------------------------------------------------
    # Panel letters
    # --------------------------------------------------------

    letters = list(
        string.ascii_lowercase[
            :len(panel_order)
        ]
    )

    # --------------------------------------------------------
    # Draw KM panels
    # --------------------------------------------------------

    for i, subgroup_name in enumerate(
        panel_order
    ):

        plot_km_panel(
            axes[i],
            df,
            subgroup_name,
            letters[i]
        )

    # --------------------------------------------------------
    # Turn off unused cells
    #
    # Fig. S9 / S10:
    # 20 cells - 18 panels = 2 blank cells
    # --------------------------------------------------------

    for i in range(
        len(panel_order),
        len(axes)
    ):

        axes[i].axis('off')

    # ========================================================
    # Shared legend
    # ========================================================

    legend_handles = [

        Line2D(
            [0],
            [0],
            color=COLOR_LOW,
            lw=1.5,
            label='Low risk'
        ),

        Line2D(
            [0],
            [0],
            color=COLOR_HIGH,
            lw=1.5,
            label='High risk'
        )
    ]

    fig.legend(
        handles=legend_handles,
        loc='upper center',
        bbox_to_anchor=(
            0.52,
            0.992
        ),
        ncol=2,
        frameon=False,
        fontsize=7.5,
        handlelength=2.3,
        columnspacing=2.0
    )

    # ========================================================
    # Shared x-axis label
    # ========================================================

    fig.text(
        0.52,
        0.018,
        "Time after surgery (months)",
        ha='center',
        va='center',
        fontsize=8,
        fontweight='bold'
    )

    # ========================================================
    # Shared y-axis label
    # ========================================================

    fig.text(
        0.014,
        0.52,
        "Disease-free survival",
        ha='center',
        va='center',
        rotation=90,
        fontsize=8,
        fontweight='bold'
    )

    # ========================================================
    # Layout
    # ========================================================

    plt.subplots_adjust(
        left=layout_adjust["left"],
        right=layout_adjust["right"],
        top=layout_adjust["top"],
        bottom=layout_adjust["bottom"],
        wspace=layout_adjust["wspace"],
        hspace=layout_adjust["hspace"]
    )

    # ========================================================
    # Save
    # ========================================================

    filename_base = os.path.join(
        output_dir,
        f"{figure_number}_"
        f"{cohort_name.replace(' ', '_')}_"
        f"Subgroup_KM"
    )

    fig.savefig(
        filename_base + ".pdf",
        bbox_inches='tight',
        facecolor='white'
    )

    fig.savefig(
        filename_base + ".png",
        dpi=600,
        bbox_inches='tight',
        facecolor='white'
    )

    plt.close(fig)

    print(
        f"✓ Saved:\n"
        f"  {filename_base}.pdf\n"
        f"  {filename_base}.png"
    )


# ============================================================
# 10. Main
# ============================================================

def main():

    print("=" * 72)

    print(
        "Supplementary Fig. S9-S11 "
        "BC-MPM subgroup Kaplan-Meier analysis"
    )

    print("=" * 72)

    files = {

        "Training cohort": os.path.join(
            SCRIPT_DIR,
            "train_all_modalities_risk.csv"
        ),

        "Internal validation cohort": os.path.join(
            SCRIPT_DIR,
            "validation_all_modalities_risk.csv"
        ),

        "External test cohort": os.path.join(
            SCRIPT_DIR,
            "test_all_modalities_risk.csv"
        )
    }

    figure_numbers = {

        "Training cohort":
            "FigS9",

        "Internal validation cohort":
            "FigS10",

        "External test cohort":
            "FigS11"
    }

    # ========================================================
    # Process each cohort
    # ========================================================

    for cohort_name, path in files.items():

        if not os.path.exists(path):

            print(
                f"⚠️ File not found: "
                f"{path}"
            )

            continue

        # ----------------------------------------------------
        # Load CSV
        # ----------------------------------------------------

        df = pd.read_csv(
            path
        )

        print(
            f"✓ Loaded {cohort_name}: "
            f"n={len(df)}"
        )

        # ----------------------------------------------------
        # Age mapping
        # ----------------------------------------------------

        if "age" not in df.columns:

            if "age1" in df.columns:

                df["age"] = np.where(
                    df["age1"] <= 50,
                    1,
                    2
                )

                print(
                    "  ✓ age variable generated "
                    "from age1"
                )

        # ----------------------------------------------------
        # Generate figure
        # ----------------------------------------------------

        create_cohort_figure(
            df=df,
            cohort_name=cohort_name,
            figure_number=figure_numbers[
                cohort_name
            ],
            output_dir=OUTPUT_DIR
        )

    print("=" * 72)

    print(
        "✓ Fig. S9-S11 generation complete"
    )

    print(
        f"Output directory: "
        f"{OUTPUT_DIR}"
    )

    print("=" * 72)


# ============================================================
# Run
# ============================================================

if __name__ == "__main__":
    main()