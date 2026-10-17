import os
import warnings
import numpy as np
import pandas as pd
from lifelines import CoxPHFitter
from lifelines.exceptions import ConvergenceWarning

# See README.md for project setup and required files.
from sklearn.preprocessing import StandardScaler

# See README.md for project setup and required files.
warnings.filterwarnings('ignore', category=ConvergenceWarning)
warnings.filterwarnings('ignore', 'overflow encountered in exp')

# See README.md for project setup and required files.
# See README.md for project setup and required files.
# Required input: see README.md for the required file and directory layout.
BASE_RESULT_DIR = './xxx/xx/all_modalities_risk'
RISK_MODALITY = 'bcmpm'  # See README.md for project setup and required files.

# ==========================================

# See README.md for project setup and required files.
train_path = os.path.join(BASE_RESULT_DIR, 'train_all_modalities_risk.csv')
val_path = os.path.join(BASE_RESULT_DIR, 'validation_all_modalities_risk.csv')
test_path = os.path.join(BASE_RESULT_DIR, 'test_all_modalities_risk.csv')

# See README.md for project setup and required files.
train_df = pd.read_csv(train_path)
val_df = pd.read_csv(val_path)
test_df = pd.read_csv(test_path)

# See README.md for project setup and required files.
risk_col = f'{RISK_MODALITY}_risk_score'
if risk_col not in train_df.columns:
    raise KeyError(f"Status '{risk_col}' Status RISK_MODALITY Status")

train_df['risk_score'] = train_df[risk_col]
val_df['risk_score'] = val_df[risk_col]
test_df['risk_score'] = test_df[risk_col]

# See README.md for project setup and required files.
# See README.md for project setup and required files.
scaler = StandardScaler()
scaler.fit(train_df[['risk_score']])

# See README.md for project setup and required files.
train_df['risk_score'] = scaler.transform(train_df[['risk_score']])
val_df['risk_score']   = scaler.transform(val_df[['risk_score']])
test_df['risk_score']  = scaler.transform(test_df[['risk_score']])
# ===========================================================

print(f"✅ Status。Status: {len(train_df)}, Status: {len(val_df)}, Status: {len(test_df)}")
print(
    f"   Status: Age, CT, RT, molecular subtypes, tumor size, Nodal status, clinical stage, histological grade, risk_score (Standardized)"
)


# See README.md for project setup and required files.
def safe_hr_ci(cph, var_name, ci_df):
    """See README.md for the script purpose and required inputs."""
    coef = cph.params_[var_name]
    if np.abs(coef) > 700:
        hr = np.inf
        ci = [np.inf, np.inf]
    else:
        hr = np.exp(coef)
        if 0.025 in ci_df.columns:
            ci = np.exp(ci_df.loc[var_name, [0.025, 0.975]])
        elif '95% lower-bound' in ci_df.columns and '95% upper-bound' in ci_df.columns:
            ci = np.exp(ci_df.loc[var_name, ['95% lower-bound', '95% upper-bound']])
        else:
            ci = np.exp(ci_df.loc[var_name, ['lower 95%', 'upper 95%']])
    return hr, ci


def format_p_value(p):
    """See README.md for the script purpose and required inputs."""
    if p < 0.001:
        return '<0.001'
    else:
        return f'{p:.3f}'


# See README.md for project setup and required files.
def generate_full_clinical_risk_analysis(train_df, val_df, test_df, time_col='DFS', event_col='status'):
    """See README.md for the script purpose and required inputs."""
    cohorts = {
        'Training cohort': train_df,
        'Internal validation cohort': val_df,
        'External validation cohort': test_df
    }

    all_results = []

    # See README.md for project setup and required files.
    vars_config = {
        'Age': {
            'type': 'categorical', 'col': 'age1', 'ref': '≤50',
            'func': lambda x: '≤50' if x <= 50 else '>50'
        },
        'CT': {
            'type': 'categorical', 'col': 'CT', 'ref': 'Yes',
            'mapping': {0: 'No', 1: 'Yes'}
        },
        'RT': {
            'type': 'categorical', 'col': 'RT', 'ref': 'Yes',
            'mapping': {0: 'No', 1: 'Yes'}
        },
        'molecular subtypes': {
            'type': 'categorical', 'col': 'molecular subtypes', 'ref': 'Luminal A',
            'mapping': {1: 'Luminal A', 2: 'Luminal B', 3: 'HER2-enriched', 4: 'Triple-negative'}
        },
        'tumor size': {
            'type': 'categorical', 'col': 'tumor size', 'ref': '≤2cm',
            'mapping': {1: '≤2cm', 2: '2-5cm', 3: '>5cm'}
        },
        'Nodal status': {
            'type': 'categorical', 'col': 'lymph node metastasis', 'ref': '0',
            'mapping': {0: '0', 1: '1-3', 2: '≥4'}
        },
        'clinical stage': {
            'type': 'categorical', 'col': 'clinical stage', 'ref': 'I',
            'mapping': {1: 'I', 2: 'II', 3: 'III'}
        },
        'histological grade': {
            'type': 'categorical', 'col': 'histological grade', 'ref': 'G1',
            'mapping': {1: 'G1', 2: 'G2', 3: 'G3'}
        },
        'risk_score': {
            'type': 'continuous', 'col': 'risk_score',
            'label': 'Risk score (per 1-SD increase)'  # See README.md for project setup and required files.
        }
    }

    # See README.md for project setup and required files.
    vars_order = [
        'Age', 'CT', 'RT', 'molecular subtypes', 'tumor size',
        'Nodal status', 'clinical stage', 'histological grade', 'risk_score'
    ]

    for cohort_name, df in cohorts.items():
        df = df.copy()

        # See README.md for project setup and required files.
        for var_key, config in vars_config.items():
            if config['type'] == 'categorical':
                col_name = config['col']
                # See README.md for project setup and required files.
                if 'func' in config:
                    df[var_key] = df[col_name].apply(config['func'])
                else:
                    if col_name in df.columns:
                        df[var_key] = df[col_name].map(config['mapping']).fillna(df[col_name].astype(str))
                    else:
                        df[var_key] = 'Unknown'
                df[var_key] = df[var_key].astype(str)

        # See README.md for project setup and required files.
        univariate_results = []

        for var_key in vars_order:
            config = vars_config[var_key]

            # See README.md for project setup and required files.
            if config['type'] == 'continuous':
                col = config['col']
                df_temp = df[[time_col, event_col, col]].dropna()
                try:
                    cph = CoxPHFitter()
                    cph.fit(df_temp, duration_col=time_col, event_col=event_col)
                    hr, ci = safe_hr_ci(cph, col, cph.confidence_intervals_)
                    p = cph.summary.loc[col, 'p']

                    ci_str = f'{ci.iloc[0]:.2f}, {ci.iloc[1]:.2f}' if np.isfinite(hr) else 'Inf'
                    hr_str = f'{hr:.2f}' if np.isfinite(hr) else 'Inf'

                    univariate_results.append({
                        'Variable': config['label'],
                        'Univariate HR (95% CI)': f'{hr_str} ({ci_str})',
                        'Univariate P value': format_p_value(p),
                        'Cohort': cohort_name
                    })
                except Exception as e:
                    univariate_results.append({
                        'Variable': config['label'],
                        'Univariate HR (95% CI)': 'N/A',
                        'Univariate P value': 'N/A',
                        'Cohort': cohort_name
                    })

            # See README.md for project setup and required files.
            else:
                ref_cat = config['ref']
                categories = df[var_key].dropna().unique()
                if ref_cat not in categories:
                    ref_cat = categories[0] if len(categories) > 0 else 'N/A'

                for cat in categories:
                    if cat == 'Unknown':
                        continue
                    # See README.md for project setup and required files.
                    if cat == ref_cat:
                        univariate_results.append({
                            'Variable': f'{var_key} ({cat})',
                            'Univariate HR (95% CI)': 'Ref.',
                            'Univariate P value': '-',
                            'Cohort': cohort_name
                        })
                        continue

                    df_temp = df[[time_col, event_col, var_key]].dropna()
                    df_temp['temp_var'] = (df_temp[var_key] == cat).astype(int)
                    try:
                        cph = CoxPHFitter()
                        cph.fit(df_temp[[time_col, event_col, 'temp_var']],
                                duration_col=time_col, event_col=event_col)
                        hr, ci = safe_hr_ci(cph, 'temp_var', cph.confidence_intervals_)
                        p = cph.summary.loc['temp_var', 'p']

                        ci_str = f'{ci.iloc[0]:.2f}, {ci.iloc[1]:.2f}' if np.isfinite(hr) else 'Inf'
                        hr_str = f'{hr:.2f}' if np.isfinite(hr) else 'Inf'

                        univariate_results.append({
                            'Variable': f'{var_key} ({cat})',
                            'Univariate HR (95% CI)': f'{hr_str} ({ci_str})',
                            'Univariate P value': format_p_value(p),
                            'Cohort': cohort_name
                        })
                    except Exception:
                        univariate_results.append({
                            'Variable': f'{var_key} ({cat})',
                            'Univariate HR (95% CI)': 'N/A',
                            'Univariate P value': 'N/A',
                            'Cohort': cohort_name
                        })

        # See README.md for project setup and required files.
        df_multi = df.copy()
        dummies_to_keep = []

        for var_key in vars_order:
            config = vars_config[var_key]
            if config['type'] == 'continuous':
                # See README.md for project setup and required files.
                dummies_to_keep.append(config['col'])
            else:
                # See README.md for project setup and required files.
                ref_cat = config['ref']
                dummies = pd.get_dummies(df_multi[var_key], prefix=var_key, drop_first=False)
                drop_col = f'{var_key}_{ref_cat}'
                if drop_col in dummies.columns:
                    dummies = dummies.drop(columns=[drop_col])
                df_multi = pd.concat([df_multi, dummies], axis=1)
                dummies_to_keep.extend([c for c in dummies.columns if c.startswith(f'{var_key}_')])

        multi_cols = [time_col, event_col] + dummies_to_keep

        multivariate_results = []
        try:
            cph_multi = CoxPHFitter()
            cph_multi.fit(df_multi[multi_cols].dropna(),
                          duration_col=time_col, event_col=event_col)

            # See README.md for project setup and required files.
            for var_key in vars_order:
                config = vars_config[var_key]
                if config['type'] == 'categorical':
                    multivariate_results.append({
                        'Variable': f'{var_key} ({config["ref"]})',
                        'Multivariate HR (95% CI)': 'Ref.',
                        'Multivariate P value': '-',
                        'Cohort': cohort_name
                    })

            # See README.md for project setup and required files.
            for var in cph_multi.params_.index:
                hr, ci = safe_hr_ci(cph_multi, var, cph_multi.confidence_intervals_)
                p = cph_multi.summary.loc[var, 'p']

                # See README.md for project setup and required files.
                if var == 'risk_score':
                    pretty_var = 'Risk score (per 1-SD increase)'  # See README.md for project setup and required files.
                elif var.startswith('Age_'):
                    pretty_var = f'Age ({var.split("_")[-1]})'
                elif var.startswith('CT_'):
                    pretty_var = f'CT ({var.split("_")[-1]})'
                elif var.startswith('RT_'):
                    pretty_var = f'RT ({var.split("_")[-1]})'
                elif var.startswith('molecular subtypes_'):
                    pretty_var = f'molecular subtypes ({var.split("_")[-1]})'
                elif var.startswith('tumor size_'):
                    pretty_var = f'tumor size ({var.split("_")[-1]})'
                elif var.startswith('Nodal status_'):
                    pretty_var = f'Nodal status ({var.split("_")[-1]})'
                elif var.startswith('clinical stage_'):
                    pretty_var = f'clinical stage ({var.split("_")[-1]})'
                elif var.startswith('histological grade_'):
                    pretty_var = f'histological grade ({var.split("_")[-1]})'
                else:
                    pretty_var = var

                ci_str = f'{ci.iloc[0]:.2}, {ci.iloc[1]:.2f}' if np.isfinite(hr) else 'Inf'
                hr_str = f'{hr:.2f}' if np.isfinite(hr) else 'Inf'

                multivariate_results.append({
                    'Variable': pretty_var,
                    'Multivariate HR (95% CI)': f'{hr_str} ({ci_str})',
                    'Multivariate P value': format_p_value(p),
                    'Cohort': cohort_name
                })

        except Exception as e:
            print(f"Status Cox Status - {cohort_name}: {e}")
            multivariate_results = []

        # See README.md for project setup and required files.
        df_uni = pd.DataFrame(univariate_results)
        df_multi_res = pd.DataFrame(multivariate_results)

        cohort_res = df_uni.merge(df_multi_res, on=['Variable', 'Cohort'], how='left')
        cohort_res['Multivariate HR (95% CI)'] = cohort_res['Multivariate HR (95% CI)'].fillna('N/A')
        cohort_res['Multivariate P value'] = cohort_res['Multivariate P value'].fillna('N/A')
        all_results.append(cohort_res)

    # See README.md for project setup and required files.
    final_df = pd.concat(all_results, axis=0)

    # See README.md for project setup and required files.
    def sort_key(var_str):
        if 'Age' in var_str: return 1
        if 'molecular subtypes' in var_str: return 2
        if 'tumor size' in var_str: return 3
        if 'Nodal status' in var_str: return 4
        if 'clinical stage' in var_str: return 5
        if 'histological grade' in var_str: return 6
        if 'CT' in var_str: return 7
        if 'RT' in var_str: return 8
        if 'Risk score' in var_str: return 9
        return 99

    final_df['sort_priority'] = final_df['Variable'].apply(sort_key)

    # See README.md for project setup and required files.
    cohort_order = ['Training cohort', 'Internal validation cohort', 'External validation cohort']
    final_df['Cohort'] = pd.Categorical(final_df['Cohort'], categories=cohort_order, ordered=True)

    # See README.md for project setup and required files.
    final_df = final_df.sort_values(by=['Cohort', 'sort_priority'])
    final_df = final_df.drop(columns=['sort_priority'])

    # See README.md for project setup and required files.
    final_df.columns = ['Characteristic', 'Univariable HR (95% CI)', 'Univariable P value',
                        'Cohort', 'Multivariable HR (95% CI)', 'Multivariable P value']

    return final_df


# See README.md for project setup and required files.
# See README.md for project setup and required files.
table_full_clinical_risk_df = generate_full_clinical_risk_analysis(train_df, val_df, test_df)

# See README.md for project setup and required files.
# output_table_path = os.path.join(BASE_RESULT_DIR, 'Table_2.csv')
output_table_path = './xxx/xx/Table2_S3-S5.csv'
table_full_clinical_risk_df.to_csv(output_table_path, index=False)

print(f"✅ Status8Status + Status risk_scoreStatus")
print(f"   Status: {output_table_path}")