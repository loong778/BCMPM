import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.decomposition import PCA
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import KFold
from sklearn.cross_decomposition import PLSRegression
from lifelines import KaplanMeierFitter
from scipy.interpolate import interp1d
import os
import pickle
import random
import warnings

warnings.filterwarnings('ignore')

SEED = 42
random.seed(SEED)
os.environ['PYTHONHASHSEED'] = str(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.enabled = False

# Required input: see README.md for the required file and directory layout.

BASE_RESULT_DIR = './xxx/xx'

class SupervisedPCARegression:
    """See README.md for the script purpose and required inputs."""

    def __init__(self, n_components=2, random_state=42):
        self.n_components = n_components
        self.random_state = random_state
        self.components_ = None
        self.mean_ = None
        self.explained_variance_ = None

    def fit(self, X, y):
        """See README.md for the script purpose and required inputs."""
        np.random.seed(self.random_state)
        n_samples, n_features = X.shape

        self.mean_ = np.mean(X, axis=0)
        X_centered = X - self.mean_

        time = y['time']
        event = y['event'].astype(bool)

        event_mask = event
        X_event = X_centered[event_mask]
        time_event = time[event_mask]

        corrs = np.zeros(n_features)
        for i in range(n_features):
            corrs[i] = np.corrcoef(X_event[:, i], time_event)[0, 1]

        abs_corrs = np.abs(corrs)
        threshold = 0.1
        selected_features = abs_corrs > threshold

        if np.sum(selected_features) < self.n_components:
            
            top_indices = np.argsort(abs_corrs)[::-1][:self.n_components * 2]
            selected_features = np.zeros(n_features, dtype=bool)
            selected_features[top_indices] = True

        X_selected = X_centered[:, selected_features]
        pca = PCA(n_components=self.n_components, random_state=self.random_state)
        X_pca = pca.fit_transform(X_selected)

        self.components_ = np.zeros((self.n_components, n_features))
        self.components_[:, selected_features] = pca.components_

        self.explained_variance_ = pca.explained_variance_

        return self

    def transform(self, X):
        """See README.md for the script purpose and required inputs."""
        X_centered = X - self.mean_
        return np.dot(X_centered, self.components_.T)

    def fit_transform(self, X, y):
        """See README.md for the script purpose and required inputs."""
        self.fit(X, y)
        return self.transform(X)

class CoxLinear(nn.Module):
    def __init__(self, n_features, dropout=0.5):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(n_features, 1, bias=False)

    def forward(self, x):
        x = self.dropout(x)
        return self.linear(x).squeeze(1)

from sksurv.metrics import concordance_index_censored

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

def cox_ph_loss(risk, time, event):
    sorted_idx = torch.argsort(time, descending=True)
    risk = risk[sorted_idx]
    event = event[sorted_idx]
    log_cumsum_hazard = torch.logcumsumexp(risk, dim=0)
    return -torch.mean((risk - log_cumsum_hazard) * event)

def bootstrap_cindex(
        risk,
        time,
        event,
        n_boot=1000):

    rng=np.random.default_rng(42)

    values=[]

    n=len(risk)

    for _ in range(n_boot):

        idx=rng.choice(n,n,replace=True)

        c=concordance_index_censored(

            event[idx].astype(bool),

            time[idx],

            risk[idx]

        )[0]

        values.append(c)

    values=np.asarray(values)

    return (

        values.mean(),

        np.percentile(values,2.5),

        np.percentile(values,97.5)

    )

def time_dependent_roc_ipcw(risk, time, event, t, cens_dist=None):
    if cens_dist is None:
        kmc = KaplanMeierFitter()
        kmc.fit(time, event == 0, label='Censoring')
        try:
            surv_idx = kmc.survival_function_.index.values
            surv_vals = kmc.survival_function_.values.ravel()
            cens_surv = interp1d(surv_idx, surv_vals, kind='previous',
                                 bounds_error=False, fill_value=(1.0, 0.0))
            G = cens_surv(np.minimum(time, t))
            G_t = cens_surv(t)
        except Exception:
            G = np.ones_like(time)
            G_t = 1.0
    else:
        G = cens_dist(time) if callable(cens_dist) else cens_dist
        G_t = cens_dist(t) if callable(cens_dist) else cens_dist

    y_true = (time <= t) & (event == 1)
    w = np.where(y_true, 1.0 / np.maximum(G, 1e-8), 1.0 / np.maximum(G_t, 1e-8))
    fpr, tpr, thresholds = roc_curve(y_true, risk, sample_weight=w)
    auc_val = auc(fpr, tpr)
    return auc_val, fpr, tpr, thresholds

def bootstrap_auc_ci(risk, time, event, t, n_boot=1000, seed=42):
    rng = np.random.RandomState(seed)
    n = len(time)
    aucs = []
    for _ in range(n_boot):
        idx = rng.choice(n, n, replace=True)
        auc_val, _, _, _ = time_dependent_roc_ipcw(risk[idx], time[idx], event[idx], t)
        if auc_val is not None and not np.isnan(auc_val):
            aucs.append(auc_val)
    if len(aucs) == 0:
        return np.nan, np.nan
    ci_low, ci_high = np.percentile(aucs, [2.5, 97.5])
    return ci_low, ci_high

def estimate_baseline_hazard(risk, time, event):
    
    event_times = np.sort(np.unique(time[event == 1]))
    exp_risk = np.exp(risk)
    cum_haz = 0.0
    cum_haz_list, times_list = [], []
    for t_ in event_times:
        at_risk = (time >= t_)
        sum_exp = np.sum(exp_risk[at_risk])
        n_event = np.sum((time == t_) & (event == 1))
        if sum_exp > 0:
            cum_haz += n_event / sum_exp
        cum_haz_list.append(cum_haz)
        times_list.append(t_)
    return interp1d(times_list, cum_haz_list, kind='previous',
                    bounds_error=False, fill_value=(0.0, cum_haz_list[-1]))

def calibration_curve_calibrated(cal_surv, time, event, t=60, n_groups=10):
    cal_surv = np.clip(cal_surv, 1e-8, 1 - 1e-8)
    n_samples = len(cal_surv)
    n_groups = min(n_groups, n_samples // 5)
    n_groups = max(n_groups, 3)

    sort_idx = np.argsort(cal_surv)
    cal_sorted = cal_surv[sort_idx]
    samples_per_bin = n_samples // n_groups
    bin_indices = np.zeros(n_samples, dtype=int)
    for i in range(n_groups):
        start = i * samples_per_bin
        end = (i + 1) * samples_per_bin if i < n_groups - 1 else n_samples
        bin_indices[start:end] = i
    bin_idx_original = np.zeros(n_samples, dtype=int)
    bin_idx_original[sort_idx] = bin_indices

    mean_pred = []
    obs_surv = []
    for i in range(n_groups):
        mask = (bin_idx_original == i)
        if np.sum(mask) < 3:
            continue
        mean_pred.append(np.mean(cal_surv[mask]))
        kmf = KaplanMeierFitter()
        kmf.fit(time[mask], event[mask])
        surv_at_t = kmf.survival_function_at_times(t)
        if not surv_at_t.empty:
            obs = surv_at_t.values[0]
        else:
            obs = 1.0 - np.mean((time[mask] <= t) & (event[mask] == 1))
        obs_surv.append(obs)

    mean_pred = np.array(mean_pred)
    obs_surv = np.array(obs_surv)
    ici = np.mean(np.abs(mean_pred - obs_surv))
    return mean_pred, obs_surv, ici

def evaluate_metrics_at_t(risk, time_, event, t, threshold):
    """See README.md for the script purpose and required inputs."""
    n = len(risk)
    pred = (risk >= threshold).astype(int)

    km_censor = KaplanMeierFitter()
    
    km_censor.fit(time_, event_observed=1 - event)
    G_t = km_censor.survival_function_at_times(t).values[0]
    if G_t == 0:
        G_t = 1e-10  

    weights = np.zeros(n)
    for i in range(n):
        if event[i] == 1 and time_[i] <= t:
            
            G_time = km_censor.survival_function_at_times(time_[i]).values[0]
            weights[i] = 1.0 / max(G_time, 1e-10)
        elif time_[i] > t:
            
            weights[i] = 1.0 / G_t
        else:
            
            weights[i] = 0.0

    TP = np.sum(weights[(pred == 1) & (event == 1) & (time_ <= t)])
    FP = np.sum(weights[(pred == 1) & ~((event == 1) & (time_ <= t))])
    FN = np.sum(weights[(pred == 0) & (event == 1) & (time_ <= t)])
    TN = np.sum(weights[(pred == 0) & ~((event == 1) & (time_ <= t))])

    sens = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    spec = TN / (TN + FP) if (TN + FP) > 0 else 0.0
    ppv = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    npv = TN / (TN + FN) if (TN + FN) > 0 else 0.0
    acc = (TP + TN) / (TP + TN + FP + FN) if (TP + TN + FP + FN) > 0 else 0.0
    f1 = 2 * (ppv * sens) / (ppv + sens) if (ppv + sens) > 0 else 0.0

    conf_mat = np.array([[TN, FP], [FN, TP]])

    return {
        'sensitivity': sens,
        'specificity': spec,
        'precision': ppv,
        'f1': f1,
        'accuracy': acc,
        'ppv': ppv,
        'npv': npv,
        'confusion_matrix': conf_mat
    }

def load_data(csv_path, npz_dir, id_col='patient_id'):
    """See README.md for the script purpose and required inputs."""
    df = pd.read_csv(csv_path)
    ids = df[id_col].values.astype(str)
    all_he = []
    all_mpm = []
    all_fused = []

    for pid in ids:
        npz_path = os.path.join(npz_dir, f"{pid}_features.npz")
        if not os.path.exists(npz_path):
            raise FileNotFoundError(f"NPZ file not found: {npz_path}")
        data = np.load(npz_path)

        he_feat = data['he_features']  
        mpm_feat = data['mpm_features']  
        fused_feat = data['fused_features']  

        patient_he = np.mean(he_feat, axis=0)
        patient_mpm = np.mean(mpm_feat, axis=0)
        patient_fused = np.mean(fused_feat, axis=0)

        all_he.append(patient_he)
        all_mpm.append(patient_mpm)
        all_fused.append(patient_fused)

    orig_he = np.stack(all_he, axis=0).astype(np.float32)
    orig_mpm = np.stack(all_mpm, axis=0).astype(np.float32)
    orig_fused = np.stack(all_fused, axis=0).astype(np.float32)

    CT = df['CT'].values.astype(np.float32).reshape(-1, 1)
    RT = df['RT'].values.astype(np.float32).reshape(-1, 1)
    tumor_size = df['tumor size'].values.astype(np.float32).reshape(-1, 1)
    age = df['age1'].values.astype(np.float32).reshape(-1, 1)
    molecular = df['molecular subtypes'].values.reshape(-1, 1)
    lymph_node = df['lymph node metastasis'].values.reshape(-1, 1)
    clinical_stage = df['clinical stage'].values.reshape(-1, 1)
    hist_grade = df['histological grade'].values.reshape(-1, 1)
    time = df['DFS'].values.astype(np.float32)
    event = df['status'].values.astype(np.float32)

    return orig_he, orig_mpm, orig_fused, CT, RT, tumor_size, age, molecular, lymph_node, clinical_stage, hist_grade, time, event

def prepare_all_modalities(train_csv, val_csv, test_csv, npz_root):
    """Load shared modality features and labels for the training entry point."""
    npz_train = os.path.join(npz_root, 'train')
    npz_val = os.path.join(npz_root, 'val')
    npz_test = os.path.join(npz_root, 'test')

    (orig_he_tr, orig_mpm_tr, orig_fused_tr, CT_tr, RT_tr, ts_tr, age_tr, mol_tr, ln_tr, cs_tr, hg_tr, T_tr,
     E_tr) = load_data(train_csv, npz_train)
    (orig_he_val, orig_mpm_val, orig_fused_val, CT_val, RT_val, ts_val, age_val, mol_val, ln_val, cs_val, hg_val, T_val,
     E_val) = load_data(val_csv, npz_val)
    (orig_he_test, orig_mpm_test, orig_fused_test, CT_test, RT_test, ts_test, age_test, mol_test, ln_test, cs_test,
     hg_test, T_test, E_test) = load_data(test_csv, npz_test)

    he_scaler = StandardScaler()
    he_train = he_scaler.fit_transform(orig_he_tr)
    he_val = he_scaler.transform(orig_he_val)
    he_test = he_scaler.transform(orig_he_test)

    mpm_scaler = StandardScaler()
    mpm_train = mpm_scaler.fit_transform(orig_mpm_tr)
    mpm_val = mpm_scaler.transform(orig_mpm_val)
    mpm_test = mpm_scaler.transform(orig_mpm_test)

    fused_scaler = StandardScaler()
    fused_train = fused_scaler.fit_transform(orig_fused_tr)
    fused_val = fused_scaler.transform(orig_fused_val)
    fused_test = fused_scaler.transform(orig_fused_test)

    age_mean = age_tr.mean()
    age_std = age_tr.std() + 1e-8
    age_tr_std = (age_tr - age_mean) / age_std
    age_val_std = (age_val - age_mean) / age_std
    age_test_std = (age_test - age_mean) / age_std

    ohe = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    cat_tr = np.hstack([mol_tr, ln_tr, cs_tr, hg_tr, ts_tr])
    cat_val = np.hstack([mol_val, ln_val, cs_val, hg_val, ts_val])
    cat_test = np.hstack([mol_test, ln_test, cs_test, hg_test, ts_test])
    ohe.fit(cat_tr)
    cat_tr_enc = ohe.transform(cat_tr)
    cat_val_enc = ohe.transform(cat_val)
    cat_test_enc = ohe.transform(cat_test)

    X_clin_tr = np.hstack([CT_tr, RT_tr, age_tr_std, cat_tr_enc])
    X_clin_val = np.hstack([CT_val, RT_val, age_val_std, cat_val_enc])
    X_clin_test = np.hstack([CT_test, RT_test, age_test_std, cat_test_enc])

    modalities = {
        'Clinical-Only': ((X_clin_tr.astype(np.float32), T_tr, E_tr),
                          (X_clin_val.astype(np.float32), T_val, E_val),
                          (X_clin_test.astype(np.float32), T_test, E_test)),
        'H&E-Only': ((he_train.astype(np.float32), T_tr, E_tr),
                    (he_val.astype(np.float32), T_val, E_val),
                    (he_test.astype(np.float32), T_test, E_test)),
        'H&E-Clinical': ((np.hstack([he_train, X_clin_tr]).astype(np.float32), T_tr, E_tr),
                        (np.hstack([he_val, X_clin_val]).astype(np.float32), T_val, E_val),
                        (np.hstack([he_test, X_clin_test]).astype(np.float32), T_test, E_test)),
        'MPM-Only': ((mpm_train.astype(np.float32), T_tr, E_tr),
                     (mpm_val.astype(np.float32), T_val, E_val),
                     (mpm_test.astype(np.float32), T_test, E_test)),
        'MPM-Clinical': ((np.hstack([mpm_train, X_clin_tr]).astype(np.float32), T_tr, E_tr),
                         (np.hstack([mpm_val, X_clin_val]).astype(np.float32), T_val, E_val),
                         (np.hstack([mpm_test, X_clin_test]).astype(np.float32), T_test, E_test)),
        'H&E-MPM': ((fused_train.astype(np.float32), T_tr, E_tr),
                   (fused_val.astype(np.float32), T_val, E_val),
                   (fused_test.astype(np.float32), T_test, E_test)),

        'BCMPM': ((np.hstack([fused_train, X_clin_tr]).astype(np.float32), T_tr, E_tr),
                            (np.hstack([fused_val, X_clin_val]).astype(np.float32), T_val, E_val),
                            (np.hstack([fused_test, X_clin_test]).astype(np.float32), T_test, E_test))
    }
    return modalities

def search_best_dim_cv(X_train, T_train, E_train,
                       candidate_dims=list(range(4, 257, 4)),
                       n_folds=5,
                       dropout=0.4, lr=5e-4, weight_decay=0.05,
                       epochs=100, patience=10, seed=42,
                       method='standard'):
    """See README.md for the script purpose and required inputs."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    max_dim = min(X_train.shape[1], X_train.shape[0] - 1)
    candidate_dims = [d for d in candidate_dims if 2 <= d <= max_dim]
    if not candidate_dims:
        candidate_dims = [max_dim]

    n_samples = X_train.shape[0]
    kfold = KFold(n_splits=n_folds, shuffle=True, random_state=seed)

    dim_mean_cindex = {}

    y_train = np.rec.fromarrays([E_train.astype(bool), T_train], names=['event', 'time'])

    for dim in candidate_dims:
        print(f"  Cross-validating {method.upper()} dimension: {dim}")
        fold_cindices = []

        for fold_idx, (train_idx, val_idx) in enumerate(kfold.split(X_train)):
            X_tr_fold, X_val_fold = X_train[train_idx], X_train[val_idx]
            T_tr_fold, T_val_fold = T_train[train_idx], T_train[val_idx]
            E_tr_fold, E_val_fold = E_train[train_idx], E_train[val_idx]
            y_tr_fold = np.rec.fromarrays([E_tr_fold.astype(bool), T_tr_fold], names=['event', 'time'])

            if method == 'standard':
                reducer = PCA(n_components=dim, random_state=seed)
                X_tr_pca = reducer.fit_transform(X_tr_fold)
                X_val_pca = reducer.transform(X_val_fold)
            elif method == 'spca':
                
                reducer = SupervisedPCARegression(n_components=dim, random_state=seed)
                reducer.fit(X_tr_fold, y_tr_fold)
                X_tr_pca = reducer.transform(X_tr_fold)
                X_val_pca = reducer.transform(X_val_fold)
            elif method == 'pls':
                
                reducer = PLSRegression(n_components=dim, scale=False)
                reducer.fit(X_tr_fold, T_tr_fold)
                X_tr_pca = reducer.transform(X_tr_fold)
                X_val_pca = reducer.transform(X_val_fold)
            else:
                raise ValueError(f"Unsupported dimensionality reduction method: {method}")

            train_dataset = TensorDataset(torch.FloatTensor(X_tr_pca),
                                          torch.FloatTensor(T_tr_fold),
                                          torch.FloatTensor(E_tr_fold))
            val_dataset = TensorDataset(torch.FloatTensor(X_val_pca),
                                        torch.FloatTensor(T_val_fold),
                                        torch.FloatTensor(E_val_fold))
            train_loader = DataLoader(train_dataset, batch_size=64, shuffle=False)
            
            val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

            model = CoxLinear(n_features=dim, dropout=dropout).to(device)
            optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

            best_fold_cindex = -1.0
            patience_counter = 0

            for epoch in range(epochs):
                model.train()
                for x, t, e in train_loader:
                    x, t, e = x.to(device), t.to(device), e.to(device)
                    optimizer.zero_grad()
                    risk = model(x)
                    loss = cox_ph_loss(risk, t, e)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()
                scheduler.step()

                model.eval()
                val_risks, val_times, val_events = [], [], []
                with torch.no_grad():
                    for x, t, e in val_loader:
                        x = x.to(device)
                        risk = model(x).cpu().numpy()
                        val_risks.extend(risk)
                        val_times.extend(t.numpy())
                        val_events.extend(e.numpy())
                val_risks = np.array(val_risks)
                val_cindex = harrell_cindex(val_risks, val_times, val_events)

                if val_cindex > best_fold_cindex:
                    best_fold_cindex = val_cindex
                    patience_counter = 0
                else:
                    patience_counter += 1
                if patience_counter >= patience:
                    break

            fold_cindices.append(best_fold_cindex)
            
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        mean_cindex = np.mean(fold_cindices)
        std_cindex = np.std(fold_cindices)
        dim_mean_cindex[dim] = (mean_cindex, std_cindex)
    print(f"    Dimension {dim}: mean C-index = {mean_cindex:.4f} (±{std_cindex:.4f})")

    best_dim = max(dim_mean_cindex, key=lambda d: dim_mean_cindex[d][0])
    best_mean_cindex = dim_mean_cindex[best_dim][0]
    print(f"  Best {method.upper()} dimension: {best_dim}, mean validation C-index: {best_mean_cindex:.4f}")

    if method == 'standard':
        best_reducer = PCA(n_components=best_dim, random_state=seed)
        best_reducer.fit(X_train)
    elif method == 'spca':
        best_reducer = SupervisedPCARegression(n_components=best_dim, random_state=seed)
        best_reducer.fit(X_train, y_train)
    elif method == 'pls':
        best_reducer = PLSRegression(n_components=best_dim, scale=False)
        best_reducer.fit(X_train, T_train)

    return best_reducer, best_dim, best_mean_cindex

def train_single_modality(mod_name, X_tr, T_tr, E_tr, X_val, T_val, E_val, X_test, T_test, E_test,
                          config, image_slice, cv_params, train_params, noise_std, SEED):
    """See README.md for the script purpose and required inputs."""
    print(f"\n{'=' * 60}")
    print(f"Training modality: {mod_name}")
    print(f"Dimensionality reduction method: {config['method'].upper()}")
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    variance_explained = 1.0

    if config['method'] == 'pca':
        from sklearn.decomposition import PCA
        from sklearn.preprocessing import StandardScaler

        X_img_tr = X_tr[:, image_slice]
        
        var_thresh = config['variance_threshold']
        print(f"Using unsupervised PCA to retain {var_thresh * 100:.0f}% of the variance...")

        scaler = StandardScaler()
        X_img_tr_scaled = scaler.fit_transform(X_img_tr)

        pca = PCA(n_components=var_thresh, random_state=SEED)
        X_img_tr_pca = pca.fit_transform(X_img_tr_scaled)
        best_dim = X_img_tr_pca.shape[1]
        variance_explained = np.sum(pca.explained_variance_ratio_)

        print(f"  Selected PCA dimension: {best_dim}")
        print(f"  Explained variance: {variance_explained:.4f}")

        X_img_val_scaled = scaler.transform(X_val[:, image_slice])
        X_img_val_pca = pca.transform(X_img_val_scaled)
        X_img_test_scaled = scaler.transform(X_test[:, image_slice])
        X_img_test_pca = pca.transform(X_img_test_scaled)

        reducer = {'scaler': scaler, 'pca': pca}

        if image_slice == slice(None):
            X_tr_final = X_img_tr_pca
            X_val_final = X_img_val_pca
            X_test_final = X_img_test_pca
        else:
            clinical_tr = X_tr[:, image_slice.stop:]
            clinical_val = X_val[:, image_slice.stop:]
            clinical_test = X_test[:, image_slice.stop:]
            X_tr_final = np.hstack([X_img_tr_pca, clinical_tr])
            X_val_final = np.hstack([X_img_val_pca, clinical_val])
            X_test_final = np.hstack([X_img_test_pca, clinical_test])

        print(f"  Final feature dimensions: {X_tr_final.shape[1]} (image PCA dimension: {best_dim})")

    elif config['method'] != 'none':
        X_img_tr = X_tr[:, image_slice]
        print(f"Searching best {config['method'].upper()} dimension using {cv_params['n_folds']}-fold CV...")
        reducer, best_dim, best_cv_cindex = search_best_dim_cv(
            X_img_tr, T_tr, E_tr,
            candidate_dims=config['candidate_dims'],
            **cv_params,
            method=config['method']
        )
        variance_explained = None  
        X_img_tr_pca = reducer.transform(X_img_tr)
        X_img_val_pca = reducer.transform(X_val[:, image_slice])
        X_img_test_pca = reducer.transform(X_test[:, image_slice])

        if image_slice == slice(None):
            X_tr_final = X_img_tr_pca
            X_val_final = X_img_val_pca
            X_test_final = X_img_test_pca
        else:
            
            clinical_tr = X_tr[:, image_slice.stop:]
            clinical_val = X_val[:, image_slice.stop:]
            clinical_test = X_test[:, image_slice.stop:]
            X_tr_final = np.hstack([X_img_tr_pca, clinical_tr])
            X_val_final = np.hstack([X_img_val_pca, clinical_val])
            X_test_final = np.hstack([X_img_test_pca, clinical_test])

        print(f"  Final feature dimensions: {X_tr_final.shape[1]} (image reduction dimension: {best_dim})")

    else:
        X_tr_final = X_tr
        X_val_final = X_val
        X_test_final = X_test
        reducer = None
        best_dim = None
        variance_explained = 1.0  
        print(f"  Final feature dimensions: {X_tr_final.shape[1]} (clinical features only)")

    train_dataset = TensorDataset(torch.FloatTensor(X_tr_final),
                                  torch.FloatTensor(T_tr),
                                  torch.FloatTensor(E_tr))
    val_dataset = TensorDataset(torch.FloatTensor(X_val_final),
                                torch.FloatTensor(T_val),
                                torch.FloatTensor(E_val))
    test_dataset = TensorDataset(torch.FloatTensor(X_test_final),
                                 torch.FloatTensor(T_test),
                                 torch.FloatTensor(E_test))

    batch_size = 64
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    model = CoxLinear(n_features=X_tr_final.shape[1], dropout=train_params['dropout'])
    optimizer = torch.optim.AdamW(model.parameters(), lr=train_params['lr'], weight_decay=train_params['weight_decay'])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=train_params['num_epochs'], eta_min=1e-6)

    def train_epoch(model, loader, optimizer, add_noise=False, noise_std=0.0):
        model.train()
        total_loss = 0.0
        for x, t, e in loader:
            if add_noise and noise_std > 0:
                noise = torch.randn_like(x) * noise_std
                x = x + noise
            optimizer.zero_grad()
            risk = model(x)
            loss = cox_ph_loss(risk, t, e)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += loss.item() * len(x)
        return total_loss / len(loader.dataset)

    def predict(model, loader):
        model.eval()
        all_risk, all_time, all_event = [], [], []
        with torch.no_grad():
            for x, t, e in loader:
                risk = model(x)
                all_risk.append(risk.numpy())
                all_time.append(t.numpy())
                all_event.append(e.numpy())
        return np.concatenate(all_risk), np.concatenate(all_time), np.concatenate(all_event)

    best_val_cindex = -1
    best_model_state = None
    patience_counter = 0

    for epoch in range(train_params['num_epochs']):
        train_loss = train_epoch(model, train_loader, optimizer,
                                 add_noise=(noise_std > 0), noise_std=noise_std)
        val_risk, val_time, val_event = predict(model, val_loader)
        val_cindex = harrell_cindex(val_risk, val_time, val_event)
        scheduler.step()

        if val_cindex > best_val_cindex:
            best_val_cindex = val_cindex
            best_model_state = model.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1

        if (epoch + 1) % 20 == 0:
            current_lr = optimizer.param_groups[0]['lr']
            print(
                f"Epoch {epoch + 1:3d} | Train Loss: {train_loss:.4f} | Val C-index: {val_cindex:.4f} | LR: {current_lr:.2e}")

        if patience_counter >= train_params['patience']:
            print(f"Early stopping at epoch {epoch + 1}")
            break

    model.load_state_dict(best_model_state)

    train_risk, train_time, train_event = predict(model, DataLoader(train_dataset, batch_size=64))
    val_risk, val_time, val_event = predict(model, DataLoader(val_dataset, batch_size=64))
    test_risk, test_time, test_event = predict(model, test_loader)

    risk_dict = {'Train': train_risk, 'Validation': val_risk, 'Test': test_risk}
    time_dict = {'Train': train_time, 'Validation': val_time, 'Test': test_time}
    event_dict = {'Train': train_event, 'Validation': val_event, 'Test': test_event}

    cidx = {}
    boot_cidx = {}
    for dset, risk_arr in risk_dict.items():
        time_arr = time_dict[dset]
        event_arr = event_dict[dset]
        cidx[dset] = harrell_cindex(risk_arr, time_arr, event_arr)
        boot_cidx[dset] = bootstrap_cindex(risk_arr, time_arr, event_arr)

    time_points = [12, 36, 60]
    td_roc_raw = {t: {} for t in time_points}

    for t in time_points:
        for dset in ['Train', 'Validation', 'Test']:
            auc_val, fpr, tpr, _ = time_dependent_roc_ipcw(
                risk_dict[dset], time_dict[dset], event_dict[dset], t
            )
            if fpr[0] != 0.0 or tpr[0] != 0.0:
                fpr = np.concatenate([[0.0], fpr])
                tpr = np.concatenate([[0.0], tpr])
            if fpr[-1] != 1.0 or tpr[-1] != 1.0:
                fpr = np.concatenate([fpr, [1.0]])
                tpr = np.concatenate([tpr, [1.0]])
            ci_low, ci_high = bootstrap_auc_ci(
                risk_dict[dset], time_dict[dset], event_dict[dset], t, n_boot=1000, seed=SEED
            )
            td_roc_raw[t][dset] = (fpr, tpr, auc_val, ci_low, ci_high)

    baseline_hazard_fn = estimate_baseline_hazard(train_risk, train_time, train_event)
    H0_60 = baseline_hazard_fn(60)
    train_pred_surv = np.exp(-np.exp(train_risk) * H0_60)
    val_pred_surv = np.exp(-np.exp(val_risk) * H0_60)
    test_pred_surv = np.exp(-np.exp(test_risk) * H0_60)

    keep_val = ~((val_time <= 60) & (val_event == 0))
    val_pred_event_cal = 1 - val_pred_surv[keep_val]
    y_event_val = np.where((val_time[keep_val] <= 60) & (val_event[keep_val] == 1), 1, 0)
    iso = IsotonicRegression(out_of_bounds='clip')
    iso.fit(val_pred_event_cal, y_event_val)
    train_cal_surv = 1 - iso.predict(1 - train_pred_surv)
    val_cal_surv = 1 - iso.predict(1 - val_pred_surv)
    test_cal_surv = 1 - iso.predict(1 - test_pred_surv)

    cal_curves = {
        'Train': calibration_curve_calibrated(train_cal_surv, train_time, train_event, t=60),
        'Validation': calibration_curve_calibrated(val_cal_surv, val_time, val_event, t=60),
        'Test': calibration_curve_calibrated(test_cal_surv, test_time, test_event, t=60)
    }

    return {
        'risk': risk_dict,
        'time': time_dict,
        'event': event_dict,
        'cindex': cidx,
        'bootstrap_cindex': boot_cidx,
        'td_roc_raw': td_roc_raw,
        'train_pred_surv': train_pred_surv,
        'val_pred_surv': val_pred_surv,
        'test_pred_surv': test_pred_surv,
        'cal_curves': cal_curves,
        'baseline_hazard_fn': baseline_hazard_fn,
        'reduction_method': config['method'],
        'best_dim': best_dim,
        'variance_explained': variance_explained,  
        'reducer': reducer,
        'model_state': best_model_state
    }

def save_modality_result(result, mod_name, result_dir):
    """See README.md for the script purpose and required inputs."""
    mod_dir = os.path.join(result_dir, mod_name.lower().replace('-', '_'))
    os.makedirs(mod_dir, exist_ok=True)

    save_dict = {k: v for k, v in result.items() if k not in ['baseline_hazard_fn', 'reducer', 'model_state']}
    np.savez_compressed(os.path.join(mod_dir, 'results.npz'), **save_dict)

    if result['reducer'] is not None:
        with open(os.path.join(mod_dir, 'reducer.pkl'), 'wb') as f:
            pickle.dump(result['reducer'], f)

    torch.save(result['model_state'], os.path.join(mod_dir, 'cox_model.pth'))

    print(f"✅ Saved {mod_name} results to: {mod_dir}")

def load_modality_result(mod_name, result_dir):
    """See README.md for the script purpose and required inputs."""
    mod_dir = os.path.join(result_dir, mod_name.lower().replace('-', '_'))

    # See README.md for project setup and required files.
    data = np.load(os.path.join(mod_dir, 'results.npz'), allow_pickle=True)
    result = {k: data[k].item() if data[k].ndim == 0 else data[k] for k in data.files}

    # See README.md for project setup and required files.
    reducer_path = os.path.join(mod_dir, 'reducer.pkl')
    if os.path.exists(reducer_path):
        with open(reducer_path, 'rb') as f:
            result['reducer'] = pickle.load(f)
    else:
        result['reducer'] = None

    # See README.md for project setup and required files.
    result['model_state'] = torch.load(os.path.join(mod_dir, 'cox_model.pth'), map_location='cpu')

    # See README.md for project setup and required files.
    baseline_hazard_fn = estimate_baseline_hazard(
        result['risk']['Train'], result['time']['Train'], result['event']['Train']
    )
    result['baseline_hazard_fn'] = baseline_hazard_fn

    return result

def find_best_cutoff_youden(risk, time, event, time_cut=60):
    """See README.md for the script purpose and required inputs."""
    
    y_true = np.where((time <= time_cut) & (event == 1), 1, 0)

    fpr, tpr, thresholds = roc_curve(y_true, risk)

    youden = tpr - fpr

    best_idx = np.argmax(youden)
    best_cut = thresholds[best_idx]
    best_youden = youden[best_idx]

    return best_cut, best_youden

def calculate_all_binary_metrics(risk_dict, time_dict, event_dict, best_cut, time_cut=60):
    """See README.md for the script purpose and required inputs."""
    metrics_dict = {}
    for dset in ['Train', 'Validation', 'Test']:
        metrics_dict[dset] = evaluate_metrics_at_t(
            risk_dict[dset],
            time_dict[dset],
            event_dict[dset],
            time_cut,
            best_cut
        )
    return metrics_dict
