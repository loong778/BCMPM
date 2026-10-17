import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# ===============================
# See README.md for project setup and required files.
# ===============================

# Required input: see README.md for the required file and directory layout.

SCRIPT_DIR = './xxx/xx'

FILE = "NRI_IDI_Continuous_IPCW_Test_all.csv"



OUTPUT = './xxx/xx/FigS4'
# ===============================
# See README.md for project setup and required files.
# ===============================

df = pd.read_csv(
    os.path.join(SCRIPT_DIR, FILE)
)


# ===============================
# See README.md for project setup and required files.
# ===============================

order = [
    "Overall",

    "Age ≤ 50",
    "Age > 50",

    "G2",
    "G3",

    "Luminal A",
    "Luminal B",
    "HER2-enriched",
    "Triple-negative",

    "Stage I",
    "Stage II",
    "Stage III",

    "N0",
    "N1-3",
    "N≥4",

    "Size ≤ 2cm",
    "Size 2-5cm",

]


df_plot = (
    df.set_index("Subgroup")
      .loc[order]
      .reset_index()
)


# ===============================
# See README.md for project setup and required files.
# ===============================

labels = df_plot["Subgroup"].values


nri = df_plot["Cont_NRI"].values


nri_err = np.vstack([
    nri - df_plot["Cont_NRI_CI_Low"].values,
    df_plot["Cont_NRI_CI_High"].values - nri
])


idi = df_plot["IDI"].values


idi_err = np.vstack([
    idi - df_plot["IDI_CI_Low"].values,
    df_plot["IDI_CI_High"].values - idi
])



# ===============================
# See README.md for project setup and required files.
# ===============================

plt.rcParams.update({

    "font.family":"Arial",

    "font.size":12,

    "axes.linewidth":1.2
})


fig, ax = plt.subplots(
    figsize=(11,5.5)
)


x = np.arange(len(labels))


bar_width = 0.36



# ===============================
# See README.md for project setup and required files.
# ===============================

bar1=ax.bar(

    x-bar_width/2,

    nri,

    width=bar_width,

    color="#8E63B7",

    edgecolor="black",

    linewidth=0.5,

    yerr=nri_err,

    capsize=3,

    error_kw={
        "elinewidth":1
    },

    label="NRI"

)



# ===============================
# See README.md for project setup and required files.
# ===============================


bar2=ax.bar(

    x+bar_width/2,

    idi,

    width=bar_width,

    color="#62A86B",

    edgecolor="black",

    linewidth=0.5,

    yerr=idi_err,

    capsize=3,

    error_kw={
        "elinewidth":1
    },

    label="IDI"

)



# ===============================
# See README.md for project setup and required files.
# ===============================


# See README.md for project setup and required files.
for i, v in enumerate(nri):
    upper = df_plot["Cont_NRI_CI_High"].iloc[i]
    ax.text(
        x[i]-bar_width/2,
        upper + 0.04,
        f"{v:.2f}",
        ha='center',
        va='bottom',
        fontsize=11,
        fontweight='bold'
    )




# See README.md for project setup and required files.
for i, v in enumerate(idi):
    upper = df_plot["IDI_CI_High"].iloc[i]
    ax.text(
        x[i]+bar_width/2,
        upper + 0.01,
        f"{v:.2f}",
        ha='center',
        va='bottom',
        fontsize=10
    )



# ===============================
# See README.md for project setup and required files.
# ===============================


ax.set_xticks(x)

ax.set_xticklabels(

    labels,

    rotation=45,

    ha="right",

    fontsize=11

)


ax.set_ylabel(

    "Improvement",

    fontsize=13,

    fontweight="bold"

)


# See README.md for project setup and required files.
ax.set_ylim(
    -0.05,
    max(df_plot["Cont_NRI_CI_High"]) + 0.25
)



# See README.md for project setup and required files.

ax.axhline(

    0,

    color="black",

    linewidth=1

)



# See README.md for project setup and required files.

for spine in [

    "top",

    "right"

]:

    ax.spines[spine].set_visible(False)



# legend

# legend
ax.legend(
    loc='upper left',
    bbox_to_anchor=(1.01, 1.00),
    frameon=False,
    fontsize=13
)



# plt.tight_layout()


plt.subplots_adjust(
    bottom=0.24,
    left=0.08,
    right=0.88,
    top=0.95
)
# ===============================
# See README.md for project setup and required files.
# ===============================


plt.savefig(

    OUTPUT,

    dpi=600,

    bbox_inches="tight",

    facecolor="white"

)


plt.close()


print(
    "Saved:",
    OUTPUT
)