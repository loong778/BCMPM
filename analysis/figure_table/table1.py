# See README.md for project setup and required files.
import os
import warnings
import numpy as np
import pandas as pd
from scipy.stats import kruskal, chi2_contingency
from lifelines import CoxPHFitter
from lifelines.statistics import multivariate_logrank_test
from lifelines.exceptions import ConvergenceWarning
from sklearn.metrics import roc_auc_score

# See README.md for project setup and required files.
warnings.filterwarnings('ignore', category=ConvergenceWarning)
warnings.filterwarnings('ignore', 'overflow encountered in exp')

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

# See README.md for project setup and required files.
def format_p_value(p, threshold=0.001):
    """See README.md for the script purpose and required inputs."""
    if pd.isna(p):
        return ""
    if p < threshold:
        return f"<{threshold}"
    elif p > 0.999:
        return ">0.999"
    else:
        return f"{p:.3f}"

def median_iqr_str(series):
    """See README.md for the script purpose and required inputs."""
    series = series.dropna()
    if len(series) == 0:
        return "NA"
    med = series.median()
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    return f"{med:.1f} ({q1:.1f}–{q3:.1f})"

def cat_count_pct_str(count, total):
    """See README.md for the script purpose and required inputs."""
    return f"{count} ({count / total * 100:.1f}%)"

# See README.md for project setup and required files.
def create_table_1(total_df, train_df, val_df, test_df, col_features):
    """See README.md for the script purpose and required inputs."""
    # See README.md for project setup and required files.
    datasets = {
        'Overall': total_df,
        'Training': train_df,
        'Internal Validation': val_df,
        'Test': test_df
    }
    group_names = ['Overall', 'Training', 'Internal Validation', 'Test']
    compare_groups = ['Training', 'Internal Validation', 'Test']  # See README.md for project setup and required files.

    rows = []
    continuous_vars = {'DFS'}   # See README.md for project setup and required files.

    for feat in col_features:
        if feat not in total_df.columns:
            print(f"⚠️ Status: Status '{feat}' Status")
            continue

        # See README.md for project setup and required files.
        sizes = {grp: len(datasets[grp]) for grp in group_names}

        if feat in continuous_vars:
            # See README.md for project setup and required files.
            row = {'Characteristic': feat}
            for grp in group_names:
                series = datasets[grp][feat]
                row[grp] = median_iqr_str(series)

            # See README.md for project setup and required files.
            groups = [datasets[grp][feat].dropna().values for grp in compare_groups]
            if all(len(g) > 0 for g in groups):
                try:
                    _, p_val = kruskal(*groups)
                except Exception as e:
                    print(f"  ⚠️ {feat} Kruskal-Wallis Status: {e}")
                    p_val = np.nan
            else:
                p_val = np.nan
            row['P value'] = format_p_value(p_val)
            rows.append(row)

        else:
            # See README.md for project setup and required files.
            # See README.md for project setup and required files.
            all_cats = set()
            for grp in group_names:
                all_cats.update(datasets[grp][feat].dropna().unique())
            all_cats = sorted([cat for cat in all_cats if not pd.isna(cat)], key=str)

            if not all_cats:
                continue

            # See README.md for project setup and required files.
            contingency = []
            for grp in compare_groups:
                counts = datasets[grp][feat].value_counts()
                contingency.append([counts.get(cat, 0) for cat in all_cats])
            contingency = np.array(contingency)

            # See README.md for project setup and required files.
            if contingency.size > 0 and contingency.shape[1] > 1:
                chi2, p_val_chi, dof, expected = chi2_contingency(contingency, correction=False)
                # See README.md for project setup and required files.
                if np.any(expected < 5):
                    try:
                        # See README.md for project setup and required files.
                        _, p_val = chi2_contingency(contingency, method='monte-carlo', n_resamples=10000, correction=False)
                    except TypeError:
                        # See README.md for project setup and required files.
                        p_val = p_val_chi
                        print(f"  ⚠️ {feat}: Status<5Status SciPy Status p Status")
                else:
                    p_val = p_val_chi
            else:
                p_val = np.nan

            # See README.md for project setup and required files.
            var_row = {'Characteristic': feat}
            for grp in group_names:
                var_row[grp] = ""
            var_row['P value'] = format_p_value(p_val)
            rows.append(var_row)

            # See README.md for project setup and required files.
            for cat in all_cats:
                cat_row = {'Characteristic': f"  {cat}"}   # See README.md for project setup and required files.
                for grp in group_names:
                    df = datasets[grp]
                    count = (df[feat] == cat).sum()
                    total = sizes[grp]
                    cat_row[grp] = cat_count_pct_str(count, total)
                cat_row['P value'] = ""
                rows.append(cat_row)

    # See README.md for project setup and required files.
    col_order = ['Characteristic'] + group_names + ['P value']
    table_df = pd.DataFrame(rows, columns=col_order)
    return table_df

# See README.md for project setup and required files.
col_list = ['age', 'tumor size', 'molecular subtypes', 'lymph node metastasis', 'clinical stage', 'histological grade','CT', 'RT', 'DFS', 'status']

# See README.md for project setup and required files.
table_1 = create_table_1(total_df, train_df, val_df, test_df, col_list)
# output_path = os.path.join(BASE_RESULT_DIR, 'Table1_Baseline.csv')
output_path = './xxx/xx/Table1.csv'
table_1.to_csv(output_path, index=False)
print(f"✅ Table 1 (Baseline) Status: {output_path}")

