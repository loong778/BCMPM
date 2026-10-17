# See README.md for project setup and required files.
import os
import warnings
import numpy as np
import pandas as pd
from lifelines import CoxPHFitter
from lifelines.statistics import multivariate_logrank_test
from lifelines.exceptions import ConvergenceWarning

# See README.md for project setup and required files.
warnings.filterwarnings('ignore', category=ConvergenceWarning)
warnings.filterwarnings('ignore', 'overflow encountered in exp')

# See README.md for project setup and required files.
# Required input: see README.md for the required file and directory layout.
BASE_RESULT_DIR = './xxx/xx/all_modalities_risk'

# See README.md for project setup and required files.
# See README.md for project setup and required files.


training_order = [
    'clinical_only',
    'he_only',
    'he_clinical',
    'mpm_only',
    'mpm_clinical',
    'he_mpm',
    'bica',
    'bcmpm',  # See README.md for project setup and required files.
]
# ==========================================

# See README.md for project setup and required files.
train_path = os.path.join(BASE_RESULT_DIR, 'train_all_modalities_risk.csv')
val_path   = os.path.join(BASE_RESULT_DIR, 'validation_all_modalities_risk.csv')
test_path  = os.path.join(BASE_RESULT_DIR, 'test_all_modalities_risk.csv')

train_df = pd.read_csv(train_path)
val_df   = pd.read_csv(val_path)
test_df  = pd.read_csv(test_path)
total_df = pd.concat([train_df, val_df, test_df], axis=0)   # See README.md for project setup and required files.

print(f"✅ Status。Status: {len(total_df)}")

# See README.md for project setup and required files.
trend_list = []

for mod in training_order:
    # See README.md for project setup and required files.
    risk_col = f'{mod}_risk_score'
    if risk_col not in test_df.columns:
        print(f"⚠️ Status '{risk_col}' Status {mod}")
        continue

    # See README.md for project setup and required files.
    risk_test = test_df[risk_col].values
    time_test = test_df['DFS'].values        # See README.md for project setup and required files.
    event_test = test_df['status'].values    # See README.md for project setup and required files.

    df = pd.DataFrame({'risk_score': risk_test, 'DFS': time_test, 'status': event_test})
    # See README.md for project setup and required files.
    df = df.astype({'risk_score': float, 'DFS': float, 'status': int})

    # See README.md for project setup and required files.
    df['quartile'] = pd.qcut(df['risk_score'], q=4, labels=['Q1', 'Q2', 'Q3', 'Q4'])
    df['quartile_num'] = df['quartile'].map({'Q1': 1, 'Q2': 2, 'Q3': 3, 'Q4': 4})

    # See README.md for project setup and required files.
    dummies = pd.get_dummies(df['quartile'], prefix='quartile')
    dummies = dummies.drop(columns='quartile_Q1', errors='ignore')
    model_cols = ['DFS', 'status'] + list(dummies.columns)
    model_df = pd.concat([df[['DFS', 'status']], dummies], axis=1).dropna()

    # See README.md for project setup and required files.
    cph_qt = CoxPHFitter()
    try:
        cph_qt.fit(model_df, duration_col='DFS', event_col='status')
    except Exception as e:
        print(f"❌ Status {mod} Status: {e}")
        continue

    # See README.md for project setup and required files.
    trend_test = multivariate_logrank_test(df['DFS'], df['quartile_num'], df['status'])
    trend_p = trend_test.p_value
    trend_p_str = f"{trend_p:.2e}" if trend_p >= 1e-16 else "<1e-16"

    # See README.md for project setup and required files.
    def safe_hr_ci(cph, var_name):
        ci_df = cph.confidence_intervals_
        # See README.md for project setup and required files.
        if 0.025 in ci_df.columns:
            ci = np.exp(ci_df.loc[var_name, [0.025, 0.975]])
        elif '95% lower-bound' in ci_df.columns:
            ci = np.exp(ci_df.loc[var_name, ['95% lower-bound', '95% upper-bound']])
        elif 'lower 95%' in ci_df.columns:
            ci = np.exp(ci_df.loc[var_name, ['lower 95%', 'upper 95%']])
        else:
            raise KeyError("Processing...")
        return ci

    # See README.md for project setup and required files.
    rows_mod = [{
        'Modality': mod,
        'Quartile': 'Q1',
        'HR (95% CI)': '1.0 (Reference)',
        'P Value': '-',
        'Trend P-value': trend_p_str
    }]

    for q in ['Q2', 'Q3', 'Q4']:
        col_name = f'quartile_{q}'
        if col_name in cph_qt.params_:
            coef = cph_qt.params_[col_name]
            if np.abs(coef) > 700:   # See README.md for project setup and required files.
                hr_str = 'Inf (Perfect separation)'
                p_str = '-'
            else:
                hr = np.exp(coef)
                ci = safe_hr_ci(cph_qt, col_name)
                hr_str = f'{hr:.2f} ({ci.iloc[0]:.2f}-{ci.iloc[1]:.2f})'
                p_val = cph_qt.summary.loc[col_name, 'p']
                if p_val == 0:
                    p_str = '<1e-16'
                else:
                    p_str = f"{p_val:.2e}"
        else:
            hr_str = 'Calculation Error'
            p_str = '-'
        rows_mod.append({
            'Modality': mod,
            'Quartile': q,
            'HR (95% CI)': hr_str,
            'P Value': p_str,
            'Trend P-value': trend_p_str
        })

    trend_list.extend(rows_mod)

# See README.md for project setup and required files.
table_s6 = pd.DataFrame(trend_list)

output_table_path = './xxx/xx/TableS6.csv'
table_s6.to_csv(output_table_path, index=False)
print("✅ Table S6 (Quartile Trend for all modalities) saved.")