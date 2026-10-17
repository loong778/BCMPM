import os
import numpy as np
import pandas as pd
import warnings
from sklearn.metrics import (
    roc_auc_score,
    confusion_matrix
)

from lifelines import CoxPHFitter
from lifelines import KaplanMeierFitter

warnings.filterwarnings('ignore', 'RuntimeWarning')

# See README.md for project setup and required files.
# See README.md for project setup and required files.


# Required input: see README.md for the required file and directory layout.


BASE_RESULT_DIR = './xxx/xx/all_modalities_risk'
# See README.md for project setup and required files.

RISK_MODALITY = 'bcmpm'
# ==========================================
# See README.md for project setup and required files.
train_path = os.path.join(BASE_RESULT_DIR, 'train_all_modalities_risk.csv')
val_path   = os.path.join(BASE_RESULT_DIR, 'validation_all_modalities_risk.csv')
test_path  = os.path.join(BASE_RESULT_DIR, 'test_all_modalities_risk.csv')

# See README.md for project setup and required files.
train_df = pd.read_csv(train_path)
val_df   = pd.read_csv(val_path)
test_df  = pd.read_csv(test_path)

# See README.md for project setup and required files.
risk_col = f'{RISK_MODALITY}_risk_score'
high_col = f'{RISK_MODALITY}_high_risk'

if risk_col not in train_df.columns:
    raise KeyError(f"Status '{risk_col}' Status RISK_MODALITY Status")
if high_col not in train_df.columns:
    raise KeyError(f"Status '{high_col}' Status RISK_MODALITY Status")

train_df['risk_score'] = train_df[risk_col]
train_df['high_risk']  = train_df[high_col]
val_df['risk_score']   = val_df[risk_col]
val_df['high_risk']    = val_df[high_col]
test_df['risk_score']  = test_df[risk_col]
test_df['high_risk']   = test_df[high_col]

# See README.md for project setup and required files.
total_df = pd.concat([train_df, val_df, test_df], axis=0)

print(f"✅ Status。Status: {len(total_df)}")
print(f"   Status: {RISK_MODALITY}")


from sksurv.metrics import concordance_index_censored
import numpy as np

def harrell_cindex(risk, time, event):
    """
    Standard Harrell's concordance index
    """

    risk = np.asarray(risk, dtype=float).reshape(-1)
    time = np.asarray(time, dtype=float).reshape(-1)
    event = np.asarray(event, dtype=bool).reshape(-1)

    return concordance_index_censored(
        event_indicator=event,
        event_time=time,
        estimate=risk
    )[0]


# ==========================================================
# See README.md for project setup and required files.
# ==========================================================
def calc_5yr_dfs(df):

    km = KaplanMeierFitter()

    km.fit(
        durations=df["DFS"],
        event_observed=df["status"]
    )

    try:
        dfs = km.predict(60)
    except:
        dfs = np.nan

    return dfs * 100


# ==========================================================
# See README.md for project setup and required files.
# ==========================================================
def calc_binary_metrics(df):

    y_true = (
        (df["DFS"] <= 60) &
        (df["status"] == 1)
    ).astype(int)

    y_pred = df["high_risk"].astype(int)

    if len(np.unique(y_true)) < 2:
        return np.nan, np.nan

    tn, fp, fn, tp = confusion_matrix(
        y_true,
        y_pred
    ).ravel()

    sensitivity = tp / (tp + fn)

    specificity = tn / (tn + fp)

    return sensitivity, specificity



df_all = total_df.copy()

subtype_map = {1: 'Luminal A', 2: 'Luminal B', 3: 'HER2-enriched', 4: 'Triple-negative'}
df_all['molecular subtypes'] = df_all['molecular subtypes'].map(subtype_map)

node_map = {0: '0', 1: '1-3', 2: '≥4'}
df_all['lymph node metastasis'] = df_all['lymph node metastasis'].map(node_map)

stage_map = {1: 'I', 2: 'II', 3: 'III'}
df_all['clinical stage'] = df_all['clinical stage'].map(stage_map)

grade_map = {1: 'G1', 2: 'G2', 3: 'G3'}
df_all['histological grade'] = df_all['histological grade'].map(grade_map)

tsize_map = {1: '≤2cm', 2: '2-5cm', 3: '>5cm'}
df_all['tumor size'] = df_all['tumor size'].map(tsize_map)

df_all['Age_group'] = pd.cut(df_all['age1'], bins=[0, 50, 200], labels=['≤50', '>50'])

# See README.md for project setup and required files.
subgroups = {
    'Age ≤ 50': df_all['Age_group'] == '≤50',
    'Age > 50': df_all['Age_group'] == '>50',
    'Tumor Size ≤ 2cm': df_all['tumor size'] == '≤2cm',
    'Tumor Size 2-5cm': df_all['tumor size'] == '2-5cm',
    'Tumor Size > 5cm': df_all['tumor size'] == '>5cm',
    'Molecular Subtype (Luminal A)': df_all['molecular subtypes'] == 'Luminal A',
    'Molecular Subtype (Luminal B)': df_all['molecular subtypes'] == 'Luminal B',
    'Molecular Subtype (HER2-enriched)': df_all['molecular subtypes'] == 'HER2-enriched',
    'Molecular Subtype (Triple-negative)': df_all['molecular subtypes'] == 'Triple-negative',
    'Nodal Status (0)': df_all['lymph node metastasis'] == '0',
    'Nodal Status (1-3)': df_all['lymph node metastasis'] == '1-3',
    'Nodal Status (≥4)': df_all['lymph node metastasis'] == '≥4',
    'Clinical Stage (I)': df_all['clinical stage'] == 'I',
    'Clinical Stage (II)': df_all['clinical stage'] == 'II',
    'Clinical Stage (III)': df_all['clinical stage'] == 'III',
    'Histological Grade (G1)': df_all['histological grade'] == 'G1',
    'Histological Grade (G2)': df_all['histological grade'] == 'G2',
    'Histological Grade (G3)': df_all['histological grade'] == 'G3'
}

results = []

############################################################
# Overall
############################################################

df_sub = df_all.copy()

cph = CoxPHFitter()
cph.fit(
    df_sub[["DFS","status","high_risk"]],
    duration_col="DFS",
    event_col="status"
)

hr = np.exp(cph.params_["high_risk"])

ci = np.exp(
    cph.confidence_intervals_.loc["high_risk"]
)

p = cph.summary.loc["high_risk","p"]

dfs5 = calc_5yr_dfs(df_sub)

low_n = (df_sub.high_risk==0).sum()
low_pct = low_n/len(df_sub)*100

y_true = (
    (df_sub.DFS<=60)&
    (df_sub.status==1)
)

auc = roc_auc_score(
    y_true,
    df_sub.risk_score
)

cindex = harrell_cindex(
    df_sub.risk_score.values,
    df_sub.DFS.values,
    df_sub.status.values
)

sen,spe = calc_binary_metrics(df_sub)

results.append({

    "Variable":"Overall",

    "N":len(df_sub),

    "5-yr DFS":f"{dfs5:.1f}%",

    "Low risk":f"{low_n} ({low_pct:.1f}%)",

    "HR":f"{hr:.2f}",

    "95% CI":f"{ci.iloc[0]:.2f}-{ci.iloc[1]:.2f}",

    "P value":"<0.0001" if p<1e-4 else f"{p:.4f}",

    "AUC":f"{auc:.3f}",

    "C-index":f"{cindex:.3f}",

    "Sensitivity":f"{sen:.3f}",

    "Specificity":f"{spe:.3f}"

})

for name, mask in subgroups.items():

    df_sub = df_all.loc[mask].copy()

    if len(df_sub) < 10:
        continue

    ############################################################
    # Cox
    ############################################################

    try:

        cph = CoxPHFitter()

        cph.fit(
            df_sub[["DFS", "status", "high_risk"]],
            duration_col="DFS",
            event_col="status"
        )

        hr = np.exp(cph.params_["high_risk"])

        ci = np.exp(
            cph.confidence_intervals_.loc["high_risk"]
        )

        p = cph.summary.loc["high_risk", "p"]

    except:

        hr = np.inf
        ci = [np.nan, np.nan]
        p = np.nan

    ############################################################
    # 5-year DFS
    ############################################################

    dfs5 = calc_5yr_dfs(df_sub)

    ############################################################
    # Low-risk
    ############################################################

    low_n = (df_sub.high_risk == 0).sum()

    low_pct = low_n / len(df_sub) * 100

    ############################################################
    # AUC
    ############################################################

    y_true = (
        (df_sub.DFS <= 60) &
        (df_sub.status == 1)
    )

    if y_true.sum() > 0:

        auc = roc_auc_score(
            y_true,
            df_sub.risk_score
        )

    else:

        auc = np.nan

    ############################################################
    # C-index
    ############################################################

    cindex = harrell_cindex(
        df_sub.risk_score.values,
        df_sub.DFS.values,
        df_sub.status.values
    )

    ############################################################
    # Sensitivity Specificity
    ############################################################

    sen, spe = calc_binary_metrics(df_sub)

    ############################################################
    results.append({

        "Variable": name,

        "N": len(df_sub),

        "5-yr DFS":
            f"{dfs5:.1f}%",

        "Low risk":
            f"{low_n} ({low_pct:.1f}%)",

        "HR":
            f"{hr:.2f}",

        "95% CI":
            f"{ci.iloc[0]:.2f}-{ci.iloc[1]:.2f}"
            if np.isfinite(hr)
            else "NA",

        "P value":
            "<0.0001"
            if pd.notna(p) and p < 1e-4
            else f"{p:.4f}",

        "AUC":
            f"{auc:.3f}",

        "C-index":
            f"{cindex:.3f}",

        "Sensitivity":
            f"{sen:.3f}",

        "Specificity":
            f"{spe:.3f}"

    })


table1 = pd.DataFrame(results)


output_table_path = './xxx/xx/TableS1.csv'
table1.to_csv(output_table_path, index=False)

print(table1)