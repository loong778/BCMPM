"""See README.md for the script purpose and required inputs."""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import pickle
from sklearn.preprocessing import StandardScaler

# See README.md for project setup and required files.
from utils import (
    load_modality_result,
    SEED,
    time_dependent_roc_ipcw
)

np.random.seed(SEED)

# See README.md for project setup and required files.
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']


# See README.md for project setup and required files.
# See README.md for project setup and required files.
SPLITS = ['train', 'val', 'test']

# See README.md for project setup and required files.
CSV_PATHS = {
    'train': './xxx/xx/data/train398.csv',
    'val': './xxx/xx/data/val330.csv',
    'test': './xxx/xx/data/test263.csv'
}

# See README.md for project setup and required files.
TOP_N_ROI = 20

# See README.md for project setup and required files.
# Example output path; generated files are written here.
OUTPUT_HEATMAP_PATH = None

# See README.md for project setup and required files.
# Required input: see README.md for the required file and directory layout.
NPZ_ROOT_DIR = './xxx/xx'

# See README.md for project setup and required files.
# Required input: see README.md for the required file and directory layout.
RESULT_DIR = './xxx/xx'
MODALITY_NAME = 'BCMPM'

# See README.md for project setup and required files.
ID_COL = 'patient_id'


# See README.md for project setup and required files.
def get_or_create_fused_scaler(train_csv, npz_train_dir, modality_dir, id_col=ID_COL):
    scaler_path = os.path.join(modality_dir, 'fused_scaler.pkl')
    if os.path.exists(scaler_path):
        print(f"Status fused_scaler: {scaler_path}")
        with open(scaler_path, 'rb') as f:
            scaler = pickle.load(f)
        return scaler

    print("Processing...")
    df_train = pd.read_csv(train_csv)
    ids = df_train[id_col].astype(str).values

    all_fused = []
    for pid in ids:
        npz_path = os.path.join(npz_train_dir, f"{pid}_features.npz")
        if not os.path.exists(npz_path):
            raise FileNotFoundError(f"Status NPZ Status: {npz_path}")
        data = np.load(npz_path)
        fused_feat = data['fused_features']
        patient_fused = np.mean(fused_feat, axis=0)
        all_fused.append(patient_fused)

    orig_fused = np.stack(all_fused, axis=0).astype(np.float32)
    scaler = StandardScaler()
    scaler.fit(orig_fused)

    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)
    print(f"fused_scaler Status: {scaler_path}")
    return scaler


# See README.md for project setup and required files.
def compute_roi_contributions(patient_id, npz_data_dir, fused_scaler, reducer, w_img, reduction_method):
    npz_path = os.path.join(npz_data_dir, f"{patient_id}_features.npz")
    if not os.path.exists(npz_path):
        return None

    data = np.load(npz_path)
    fused_features = data['fused_features']

    fused_std1 = fused_scaler.transform(fused_features)
    if reduction_method == 'pca':
        scaler2 = reducer['scaler']
        pca_model = reducer['pca']
        fused_std2 = scaler2.transform(fused_std1)
        roi_lowdim = pca_model.transform(fused_std2)
    else:
        roi_lowdim = reducer.transform(fused_std1)

    n_roi = fused_features.shape[0]
    raw_scores = np.dot(roi_lowdim, w_img)
    roi_contributions = raw_scores / n_roi
    return roi_contributions


# See README.md for project setup and required files.
def plot_roi_contribution_heatmap(patient_df, cutoff, save_path, cmap_name='coolwarm'):
    """See README.md for the script purpose and required inputs."""
    n_patients = len(patient_df)

    # See README.md for project setup and required files.
    max_roi = patient_df['roi_count'].max()
    heatmap_matrix = np.full((max_roi, n_patients), np.nan)

    for i, (_, row) in enumerate(patient_df.iterrows()):
        contrib = row['top_contrib']
        heatmap_matrix[:len(contrib), i] = contrib

    # See README.md for project setup and required files.
    heatmap_matrix = heatmap_matrix[~np.all(np.isnan(heatmap_matrix), axis=1)]
    n_roi_effective = heatmap_matrix.shape[0]

    # See README.md for project setup and required files.
    valid_vals = heatmap_matrix[~np.isnan(heatmap_matrix)]
    if valid_vals.size == 0:
        print("Processing...")
        return

    vmin, vmax = np.percentile(valid_vals, [5, 95])
    norm = mcolors.TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)

    cmap = plt.cm.get_cmap(cmap_name).copy()
    cmap.set_bad(color='white')

    # See README.md for project setup and required files.
    n_low = np.sum(patient_df['risk'] < cutoff)

    # See README.md for project setup and required files.
    # See README.md for project setup and required files.
    # See README.md for project setup and required files.
    # See README.md for project setup and required files.
    fig = plt.figure(figsize=(22, 6))
    gs = fig.add_gridspec(
        nrows=2, ncols=2,
        height_ratios=[0.06, 1],
        width_ratios=[1, 0.025],
        hspace=0.03,
        wspace=0.02
    )

    # See README.md for project setup and required files.
    ax_band = fig.add_subplot(gs[0, 0])
    band_colors = np.zeros((1, n_patients, 3))
    band_colors[0, :n_low, :] = mcolors.to_rgb('#1f77b4')  # See README.md for project setup and required files.
    band_colors[0, n_low:, :] = mcolors.to_rgb('#d62728')  # See README.md for project setup and required files.

    ax_band.imshow(band_colors, aspect='auto', extent=[0, n_patients, 0, 1])
    ax_band.set_xticks([])
    ax_band.set_yticks([])
    ax_band.spines[:].set_visible(False)

    ax_band.text(n_low / 2, 0.5, 'BCMPM low-risk', ha='center', va='center',
                 fontsize=18, fontweight='bold', color='white')
    ax_band.text(n_low + (n_patients - n_low) / 2, 0.5, 'BCMPM high-risk', ha='center', va='center',
                 fontsize=18, fontweight='bold', color='white')

    # See README.md for project setup and required files.
    ax_heat = fig.add_subplot(gs[1, 0])

    im = ax_heat.imshow(heatmap_matrix, cmap=cmap, norm=norm, aspect='auto',
                        origin='upper', interpolation='nearest',
                        extent=[0, n_patients, n_roi_effective, 0])

    # See README.md for project setup and required files.
    ax_band.set_xlim(ax_heat.get_xlim())

    # See README.md for project setup and required files.
    ax_heat.axvline(x=n_low, color='black', linestyle='--', linewidth=1.5)

    # See README.md for project setup and required files.
    ax_heat.set_xlabel('Patients ranked by BCMPM-derived risk score', fontweight='bold', fontsize=18, labelpad=10)
    ax_heat.set_ylabel('ROI contribution rank within each patient', fontweight='bold',fontsize=18, labelpad=10)
    ax_heat.set_xticks([])
    ax_heat.set_yticks([])
    ax_heat.spines['top'].set_visible(False)
    ax_heat.spines['right'].set_visible(False)

    # See README.md for project setup and required files.
    cax = fig.add_subplot(gs[1, 1])
    cbar = fig.colorbar(im, cax=cax)

    # See README.md for project setup and required files.
    cbar.ax.set_yticks([])
    cbar.ax.text(1.05, 1.0, 'Positive', va='top', ha='left', fontweight='bold',rotation=90, fontsize=16, transform=cbar.ax.transAxes)
    cbar.ax.text(1.05, 0.0, 'Negative', va='bottom', ha='left', fontweight='bold',rotation=90, fontsize=16, transform=cbar.ax.transAxes)
    cbar.ax.text(1.65, 0.5, 'Contribution', va='center', ha='left', fontweight='bold',rotation=90, fontsize=16, transform=cbar.ax.transAxes)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"ROIStatus: {save_path}")


# See README.md for project setup and required files.
def main():
    if isinstance(SPLITS, str):
        splits = [SPLITS]
    else:
        splits = SPLITS

    for split in splits:
        if split not in CSV_PATHS:
            raise ValueError(f"SPLITS Status '{split}'Status 'train', 'val' Status 'test' Status")

    mod_dir = os.path.join(RESULT_DIR, MODALITY_NAME.lower().replace('-', '_'))
    if not os.path.exists(mod_dir):
        raise FileNotFoundError(f"BCMPM Status: {mod_dir}")
    result = load_modality_result(MODALITY_NAME, RESULT_DIR)

    reduction_method = result['reduction_method']
    if reduction_method == 'none':
        raise ValueError("Processing...")

    reducer = result['reducer']
    d_img = result['best_dim']
    print(f"BCMPM Status: {reduction_method}, Status: {d_img}")

    model_state = result['model_state']
    weight_full = model_state['linear.weight'].cpu().numpy().flatten()
    total_dim = len(weight_full)
    d_clin = total_dim - d_img
    if total_dim <= d_img:
        raise ValueError(f"Status({total_dim}) <= Status({d_img})Status")
    w_img = weight_full[:d_img]
    print(f"Status: {total_dim}, Status: {d_img}, Status: {d_clin}")

    # See README.md for project setup and required files.
    train_risk = result['risk']['Train']
    train_time = result['time']['Train']
    train_event = result['event']['Train']
    _, fpr, tpr, thresh = time_dependent_roc_ipcw(train_risk, train_time, train_event, 60)
    best_cutoff = thresh[np.argmax(tpr - fpr)]
    print(f"Status: {best_cutoff:.4f}")

    # See README.md for project setup and required files.
    npz_train_dir = os.path.join(NPZ_ROOT_DIR, 'train')
    fused_scaler = get_or_create_fused_scaler(CSV_PATHS['train'], npz_train_dir, mod_dir, ID_COL)

    # See README.md for project setup and required files.
    all_patients = []
    split_risk_key = {'train': 'Train', 'val': 'Validation', 'test': 'Test'}

    for split in splits:
        df_csv = pd.read_csv(CSV_PATHS[split])
        pids = df_csv[ID_COL].astype(str).values
        risks = result['risk'][split_risk_key[split]]

        if len(pids) != len(risks):
            raise ValueError(f"{split}StatusCSVStatus{len(pids)}Status{len(risks)}Status")

        for pid, risk in zip(pids, risks):
            all_patients.append({
                'patient_id': pid,
                'risk': risk,
                'split': split
            })

    patient_df = pd.DataFrame(all_patients)
    print(f"Status: {len(patient_df)}")

    # See README.md for project setup and required files.
    roi_count_list = []
    top_contrib_list = []

    for _, row in patient_df.iterrows():
        pid = row['patient_id']
        split = row['split']
        npz_dir = os.path.join(NPZ_ROOT_DIR, split)

        contrib = compute_roi_contributions(
            pid, npz_dir, fused_scaler, reducer, w_img, reduction_method
        )

        if contrib is None:
            top_contrib_list.append(np.array([]))
            roi_count_list.append(0)
            continue

        roi_count_list.append(len(contrib))
        # See README.md for project setup and required files.
        sorted_contrib = np.sort(contrib)[::-1]
        # See README.md for project setup and required files.
        top_contrib = sorted_contrib[:TOP_N_ROI]
        top_contrib_list.append(top_contrib)

    patient_df['top_contrib'] = top_contrib_list
    patient_df['roi_count'] = roi_count_list

    # See README.md for project setup and required files.
    patient_df = patient_df[patient_df['roi_count'] > 0].reset_index(drop=True)

    # See README.md for project setup and required files.
    roi_counts = patient_df['roi_count'].values
    print(f"\n✅ StatusROIStatus: {len(patient_df)}")
    print(f"ROIStatus={roi_counts.min()}, Status={roi_counts.max()}, Status={np.median(roi_counts):.0f}, Status={roi_counts.mean():.1f}")

    # See README.md for project setup and required files.
    patient_df_sorted = patient_df.sort_values(by='risk', ascending=True).reset_index(drop=True)

    # See README.md for project setup and required files.
    output_path = OUTPUT_HEATMAP_PATH
    if output_path is None:
        splits_str = "_".join(splits)
        output_path = f"./{splits_str}_roi_contribution_heatmap_isolated_cbar724.png"

    plot_roi_contribution_heatmap(patient_df_sorted, best_cutoff, output_path)

    print("Processing...")


if __name__ == '__main__':
    main()