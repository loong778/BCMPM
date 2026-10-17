# See README.md for project setup and required files.
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap
import torch
import os
from utils import (
     prepare_all_modalities,
     load_modality_result,
     SEED,
     BASE_RESULT_DIR,
     CoxLinear
 )

# See README.md for project setup and required files.
MOD_NAME = 'BCMPM'

# See README.md for project setup and required files.
MAX_FEATURES = 15  # See README.md for project setup and required files.
# ===============================================

# Required input: see README.md for the required file and directory layout.

TRAIN_CSV = './xxx/xx/train.csv'
# Required input: see README.md for the required file and directory layout.
VAL_CSV = './xxx/xx/val.csv'
# Required input: see README.md for the required file and directory layout.
TEST_CSV = './xxx/xx/test.csv'
# Required input: see README.md for the required file and directory layout.
NPZ_ROOT = './xxx/xx'

FIG_SIZE = (14, 8)
DPI = 600
FONT_SIZE = 18
# Example output path; generated files are written here.
SAVE_PATH = './xxx/xx'

# See README.md for project setup and required files.
N_FUSED = 256  # See README.md for project setup and required files.
N_CLINICAL = 19  # See README.md for project setup and required files.
# ======================================================================

# See README.md for project setup and required files.
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.size'] = FONT_SIZE
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['axes.spines.top'] = False
plt.rcParams['axes.spines.right'] = False
plt.rcParams['axes.spines.left'] = True
plt.rcParams['axes.spines.bottom'] = True
#
device = torch.device('cpu')


def load_model_and_extract_coeffs(result):
     """See README.md for the script purpose and required inputs."""
     model_state = result['model_state']

     weight_tensor = model_state.get('linear.weight')
     if weight_tensor is None:
         raise KeyError("Processing...")

     weight_full = weight_tensor.cpu().numpy().flatten()

     bias_tensor = model_state.get('linear.bias')
     intercept = bias_tensor.cpu().numpy()[0] if bias_tensor is not None else 0.0

     best_dim = result['best_dim']
     reducer = result['reducer']
     if isinstance(reducer, dict) and 'pca' in reducer:
         pca = reducer['pca']
     else:
         pca = reducer

     w_img = weight_full[:best_dim]
     w_clin = weight_full[best_dim:]

     w_img_orig = pca.components_.T @ w_img
     weights_equiv = np.concatenate([w_img_orig, w_clin])
     total_orig_dim = len(weights_equiv)

     print(f"✅ BCMPM Status")
     print(f"  StatusPCAStatus: {best_dim}")
     print(f"  Status: {N_FUSED}")
     print(f"  Status: {total_orig_dim} (Status: 275Status)")

     return weights_equiv, intercept, total_orig_dim


def prepare_shap_data():
    """See README.md for the script purpose and required inputs."""
    print("\nLoading data for SHAP analysis (full cohort, n=991)...")
    modal_data = prepare_all_modalities(TRAIN_CSV, VAL_CSV, TEST_CSV, NPZ_ROOT)
    (X_tr, _, _), (X_val, _, _), (X_test, _, _) = modal_data[MOD_NAME]

    X_bg = X_tr
    X_all = np.concatenate([X_tr, X_val, X_test], axis=0)

    print(f"Status: {X_bg.shape[0]} Status")
    print(f"Status: {X_all.shape[0]} Status")
    print(f"BCMPM Status: {X_all.shape[1]} (Status: 275Status)")

    return X_bg, X_all


def filter_top_image_features(shap_values, X, max_features):
     """See README.md for the script purpose and required inputs."""
     # See README.md for project setup and required files.
     img_shap = shap_values[:, :N_FUSED]
     img_X = X[:, :N_FUSED]

     # See README.md for project setup and required files.
     importance = np.mean(np.abs(img_shap), axis=0)

     # See README.md for project setup and required files.
     sorted_idx = np.argsort(importance)[::-1]
     final_idx = sorted_idx[:max_features]

     # See README.md for project setup and required files.
     sorted_order = np.argsort(importance[final_idx])[::-1]
     final_idx = final_idx[sorted_order]

     filtered_shap = img_shap[:, final_idx]
     filtered_feat = img_X[:, final_idx]
     feature_names = [f'Fused_{i + 1}' for i in final_idx]

     return filtered_shap, filtered_feat, feature_names

def plot_shap_beeswarm(merged_shap, merged_features, feature_names):
     """See README.md for the script purpose and required inputs."""
     X_LABEL_FONT_SIZE = 18
     Y_LABEL_FONT_SIZE = 18
     TICK_FONT_SIZE = 18
     LEGEND_TITLE_SIZE = 18
     LEGEND_TICK_SIZE = 18
     ZERO_LINE_WIDTH = 2.0
     POINT_ALPHA = 0.6
     POINT_SIZE = 14
     AXIS_SPINE_WIDTH = 1.0
     GRID_LINE_WIDTH = 0.8

     # See README.md for project setup and required files.
     normalized_features = np.zeros_like(merged_features)
     n_features = merged_features.shape[1]

     for i in range(n_features):
         feat_vals = merged_features[:, i]
         feat_min = np.min(feat_vals)
         feat_max = np.max(feat_vals)

         if feat_max > feat_min:
             normalized_features[:, i] = (feat_vals - feat_min) / (feat_max - feat_min)
         else:
             normalized_features[:, i] = 0.5

     # See README.md for project setup and required files.
     exp = shap.Explanation(
         values=merged_shap,
         base_values=np.zeros(merged_shap.shape[0]),
         data=normalized_features,
         feature_names=feature_names
     )

     plt.figure(figsize=FIG_SIZE, dpi=DPI)

     result = shap.plots.beeswarm(
         exp,
         show=False,
         color_bar=True,
         max_display=len(feature_names),
         plot_size=None,
         alpha=POINT_ALPHA,
         s=POINT_SIZE
     )

     if isinstance(result, tuple) and len(result) == 2:
         ax, cbar = result
     else:
         ax = result
         cbar = None
         for fig_ax in plt.gcf().axes:
             if fig_ax is not ax:
                 cbar = fig_ax
                 break

     ax.set_yticklabels(feature_names[::-1], fontsize=Y_LABEL_FONT_SIZE)
     ax.invert_yaxis()

     # See README.md for project setup and required files.
     ax.axvline(x=0, color='gray', linestyle='-', linewidth=ZERO_LINE_WIDTH)

     # See README.md for project setup and required files.
     ax.set_xlim(-0.95, 0.75)
     ax.set_xticks([-0.8, -0.6, -0.4, -0.2, 0.0, 0.2, 0.4, 0.6])

     for spine in ax.spines.values():
         spine.set_linewidth(AXIS_SPINE_WIDTH)

     ax.set_xlabel('SHAP value (impact on model output)', fontsize=X_LABEL_FONT_SIZE, fontweight='bold')
     ax.set_ylabel('')
     ax.tick_params(axis='x', labelsize=TICK_FONT_SIZE)

     ax.yaxis.grid(True, linestyle='--', alpha=0.3, linewidth=GRID_LINE_WIDTH)
     ax.xaxis.grid(False)

     # See README.md for project setup and required files.
     if cbar is not None:
         if hasattr(cbar, 'ax'):
             cbar_ax = cbar.ax
             cbar_ax.yaxis.set_label_position('right')
             cbar_ax.set_yticks([])
             cbar.set_label('Feature value', fontsize=LEGEND_TITLE_SIZE, labelpad=30, fontweight='bold')

             cbar_ax.text(1.8, -0.05, 'Low', transform=cbar_ax.transAxes,
                          fontsize=LEGEND_TICK_SIZE, va='bottom', ha='left', rotation=90)
             cbar_ax.text(1.8, 1.05, 'High', transform=cbar_ax.transAxes,
                          fontsize=LEGEND_TICK_SIZE, va='top', ha='left', rotation=90)
         else:
             cbar.yaxis.set_label_position('right')
             cbar.set_ylabel('Feature value', fontsize=LEGEND_TITLE_SIZE, labelpad=30, fontweight='bold')
             cbar.tick_params(labelsize=LEGEND_TICK_SIZE)
             cbar.set_yticks([])
             cbar.text(1.8, -0.05, 'Low', transform=cbar.transAxes,
                       fontsize=LEGEND_TICK_SIZE, va='bottom', ha='left', rotation=90)
             cbar.text(1.8, 1.05, 'High', transform=cbar.transAxes,
                       fontsize=LEGEND_TICK_SIZE, va='top', ha='left', rotation=90)

     plt.tight_layout()
     os.makedirs(os.path.dirname(SAVE_PATH), exist_ok=True)
     plt.savefig(SAVE_PATH, dpi=DPI, bbox_inches='tight', pad_inches=0.3)
     plt.show()



if __name__ == '__main__':
     print("=" * 70)
     print(f"  {MOD_NAME} Status SHAP Status 256 Status HE-MPM Status")
     print("=" * 70)

     print(f"Status {MOD_NAME} Status...")
     result = load_modality_result(MOD_NAME, BASE_RESULT_DIR)
     weights_equiv, intercept, total_dim = load_model_and_extract_coeffs(result)

     X_bg, X_all = prepare_shap_data()

     # See README.md for project setup and required files.
     print("\nCalculating SHAP values on original 275-d space using LinearExplainer...")
     explainer = shap.LinearExplainer(
         (weights_equiv, intercept),
         X_bg,
         feature_perturbation="interventional"
     )
     shap_values = explainer.shap_values(X_all)
     print("Processing...")

     # See README.md for project setup and required files.
     print(f"\nStatus {MAX_FEATURES} Status...")
     filtered_shap, filtered_features, feature_names = filter_top_image_features(
         shap_values, X_all, MAX_FEATURES
     )

     print("Processing...")
     plot_shap_beeswarm(filtered_shap, filtered_features, feature_names)

     print(f"\n📊 Status {MAX_FEATURES} Status")
     print("-" * 60)
     for i, name in enumerate(feature_names):
         mean_abs_shap = np.mean(np.abs(filtered_shap[:, i]))
         print(f"{i + 1:2d}. {name:<25} {mean_abs_shap:.4f}")

     print("=" * 70)