import argparse
import pickle
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.cm as cm
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import OneHotEncoder, StandardScaler


ID_COL = "patient_id"
N_FUSED = 256
CLINICAL_FEATURE_MAP = {
    "Chemotherapy": [0],
    "Radiotherapy": [1],
    "Age": [2],
    "Molecular Subtype": [3, 4, 5, 6],
    "Nodal status": [7, 8, 9],
    "Clinical Stage": [10, 11, 12],
    "Histological Grade": [13, 14, 15],
    "Tumor Size": [16, 17, 18],
}
CATEGORY_VALUE_MAP = {
    "Chemotherapy": ([0], None),
    "Radiotherapy": ([1], None),
    "Age": ([2], None),
    "Molecular Subtype": ([3, 4, 5, 6], [1, 2, 3, 4]),
    "Nodal status": ([7, 8, 9], [0, 1, 2]),
    "Clinical Stage": ([10, 11, 12], [1, 2, 3]),
    "Histological Grade": ([13, 14, 15], [1, 2, 3]),
    "Tumor Size": ([16, 17, 18], [1, 2, 3]),
}
SPLIT_TO_COHORT = {"train": "Training", "val": "Validation", "test": "Test"}
COHORT_TO_SPLIT = {v: k for k, v in SPLIT_TO_COHORT.items()}
IMAGE_COLOR = "#D32F2F"
CLINICAL_COLOR = "#4A90E2"
LOW_COLOR = "#2B83BA"
HIGH_COLOR = "#D73027"


def configure_style():
    plt.rcParams.update(
        {
            "font.family": "Arial",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 16,
            "font.weight": "bold",
            "axes.unicode_minus": False,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.spines.left": True,
            "axes.spines.bottom": True,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def panel_label(ax, label, x=-0.20, y=1.04):
    ax.text(x, y, label, transform=ax.transAxes, ha="left", va="bottom", fontsize=24, fontweight="bold")


def save_figure(fig, output_dir, stem):
    output_dir.mkdir(parents=True, exist_ok=True)
    png = output_dir / f"{stem}_600dpi.png"
    svg = output_dir / f"{stem}_600dpi.svg"
    fig.savefig(png, dpi=600, bbox_inches="tight", pad_inches=0.3)
    fig.savefig(svg, bbox_inches="tight", pad_inches=0.3)
    plt.close(fig)
    return png, svg


def load_bcmpm_result(result_dir):
    result_dir = Path(result_dir)
    data = np.load(result_dir / "results.npz", allow_pickle=True)
    result = {k: data[k].item() if data[k].ndim == 0 else data[k] for k in data.files}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with open(result_dir / "reducer.pkl", "rb") as f:
            result["reducer"] = pickle.load(f)
    try:
        result["model_state"] = torch.load(result_dir / "cox_model.pth", map_location="cpu", weights_only=True)
    except TypeError:
        result["model_state"] = torch.load(result_dir / "cox_model.pth", map_location="cpu")
    return result


def load_patient_features(csv_path, npz_dir):
    df = pd.read_csv(csv_path)
    ids = df[ID_COL].astype(str).to_numpy()
    fused = []
    for pid in ids:
        npz_path = Path(npz_dir) / f"{pid}_features.npz"
        if not npz_path.exists():
            raise FileNotFoundError(f"NPZ file not found: {npz_path}")
        data = np.load(npz_path)
        fused.append(np.mean(data["fused_features"], axis=0))
    return df, np.stack(fused, axis=0).astype(np.float32)


def build_bcmpm_feature_matrix(train_csv, val_csv, test_csv, npz_root):
    npz_root = Path(npz_root)
    train_df, fused_train_raw = load_patient_features(train_csv, npz_root / "train")
    val_df, fused_val_raw = load_patient_features(val_csv, npz_root / "val")
    test_df, fused_test_raw = load_patient_features(test_csv, npz_root / "test")

    fused_scaler = StandardScaler()
    fused_train = fused_scaler.fit_transform(fused_train_raw)
    fused_val = fused_scaler.transform(fused_val_raw)
    fused_test = fused_scaler.transform(fused_test_raw)

    age_mean = train_df["age1"].to_numpy(dtype=np.float32).mean()
    age_std = train_df["age1"].to_numpy(dtype=np.float32).std() + 1e-8

    cat_cols = ["molecular subtypes", "lymph node metastasis", "clinical stage", "histological grade", "tumor size"]
    encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
    encoder.fit(train_df[cat_cols].to_numpy())

    def clinical_matrix(df):
        ct = df["CT"].to_numpy(dtype=np.float32).reshape(-1, 1)
        rt = df["RT"].to_numpy(dtype=np.float32).reshape(-1, 1)
        age = ((df["age1"].to_numpy(dtype=np.float32).reshape(-1, 1) - age_mean) / age_std).astype(np.float32)
        cat = encoder.transform(df[cat_cols].to_numpy()).astype(np.float32)
        return np.hstack([ct, rt, age, cat]).astype(np.float32)

    x_train = np.hstack([fused_train, clinical_matrix(train_df)]).astype(np.float32)
    x_val = np.hstack([fused_val, clinical_matrix(val_df)]).astype(np.float32)
    x_test = np.hstack([fused_test, clinical_matrix(test_df)]).astype(np.float32)
    patient_table = pd.concat(
        [
            train_df.assign(split="train", Cohort="Training"),
            val_df.assign(split="val", Cohort="Validation"),
            test_df.assign(split="test", Cohort="Test"),
        ],
        ignore_index=True,
    )
    return x_train, np.vstack([x_train, x_val, x_test]), patient_table, fused_scaler


def equivalent_original_weights(result):
    weight_full = result["model_state"]["linear.weight"].detach().cpu().numpy().ravel()
    best_dim = int(result["best_dim"])
    reducer = result["reducer"]
    w_img = weight_full[:best_dim]
    w_clin = weight_full[best_dim:]

    if isinstance(reducer, dict):
        pca = reducer.get("pca")
        scaler2 = reducer.get("scaler")
        w_img_orig = pca.components_.T @ w_img
        if scaler2 is not None and hasattr(scaler2, "scale_"):
            w_img_orig = w_img_orig / scaler2.scale_
    else:
        w_img_orig = reducer.components_.T @ w_img if hasattr(reducer, "components_") else w_img

    return np.concatenate([w_img_orig, w_clin]), weight_full


def calculate_linear_shap_values(x_background, x_all, weights_equiv):
    baseline = np.mean(x_background, axis=0, keepdims=True)
    return (x_all - baseline) * weights_equiv.reshape(1, -1)


def group_clinical_features(shap_values, feature_values):
    img_shap = shap_values[:, :N_FUSED]
    clin_shap = shap_values[:, N_FUSED:]
    img_x = feature_values[:, :N_FUSED]
    clin_x = feature_values[:, N_FUSED:]

    grouped_shap = []
    grouped_x = []
    grouped_names = []
    for name, cols in CLINICAL_FEATURE_MAP.items():
        cols_idx, labels = CATEGORY_VALUE_MAP[name]
        feat_shap = clin_shap[:, cols]
        feat_x = clin_x[:, cols]
        if len(cols_idx) > 1:
            grouped_shap.append(np.sum(feat_shap, axis=1))
            active_idx = np.argmax(feat_x, axis=1)
            grouped_x.append(np.asarray([labels[i] for i in active_idx], dtype=float))
        else:
            grouped_shap.append(feat_shap.ravel())
            grouped_x.append(feat_x.ravel())
        grouped_names.append(name)

    combined_shap = np.hstack([img_shap, np.column_stack(grouped_shap)])
    combined_x = np.hstack([img_x, np.column_stack(grouped_x)])
    combined_names = [f"Fused_{i + 1}" for i in range(N_FUSED)] + grouped_names
    return combined_shap, combined_x, combined_names


def select_top_features(shap_values, feature_values, feature_names, max_features=15, top_clinical=1):
    importance = np.mean(np.abs(shap_values), axis=0)
    if top_clinical > 0:
        image_order = list(np.argsort(importance[:N_FUSED])[::-1])
        clinical_order = list(np.argsort(importance[N_FUSED:])[::-1] + N_FUSED)
        chosen = image_order[: max_features - top_clinical] + clinical_order[:top_clinical]
    else:
        chosen = list(np.argsort(importance)[::-1][:max_features])

    chosen = np.asarray(chosen, dtype=int)
    chosen = chosen[np.argsort(importance[chosen])[::-1]]
    return shap_values[:, chosen], feature_values[:, chosen], [feature_names[i] for i in chosen], importance[chosen]


def normalize_columns(values):
    normed = np.zeros_like(values, dtype=float)
    for i in range(values.shape[1]):
        col = values[:, i]
        low = np.nanmin(col)
        high = np.nanmax(col)
        normed[:, i] = (col - low) / (high - low) if high > low else 0.5
    return normed


def draw_fig6a(ax, shap_top, feature_names, mean_abs):
    colors = [IMAGE_COLOR if name.startswith("Fused_") else CLINICAL_COLOR for name in feature_names]
    y_pos = np.arange(len(feature_names))[::-1]
    bars = ax.barh(y_pos, mean_abs, color=colors, height=0.65, zorder=3)

    for bar in bars:
        width = bar.get_width()
        ax.text(
            width + 0.005,
            bar.get_y() + bar.get_height() / 2,
            f"{width:.3f}",
            ha="left",
            va="center",
            fontsize=18,
            fontweight="bold",
        )

    ax.set_yticks(y_pos)
    ax.set_yticklabels(feature_names, fontsize=16, fontweight="bold")
    ax.set_xlim(0, 0.27)
    ax.set_xticks([0, 0.05, 0.10, 0.15, 0.20, 0.25])
    ax.tick_params(axis="x", labelsize=18)
    for label in ax.get_xticklabels():
        label.set_fontweight("bold")

    ax.set_xlabel("Mean |SHAP|", fontsize=20, labelpad=10, fontweight="bold")
    ax.text(0.0, 1.01, f"Top {len(feature_names)} features", transform=ax.transAxes, fontsize=18, va="bottom", ha="left", fontweight="bold")
    ax.text(1.0, 1.01, "Mean |SHAP|", transform=ax.transAxes, fontsize=18, va="bottom", ha="right", fontweight="bold")

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    legend_handles = [
        plt.Rectangle((0, 0), 1, 1, color=IMAGE_COLOR, label="Fused-image features"),
        plt.Rectangle((0, 0), 1, 1, color=CLINICAL_COLOR, label="Clinical features"),
    ]
    ax.legend(handles=legend_handles, loc="lower right", frameon=True, edgecolor="#DDDDDD", fontsize=8)
    #panel_label(ax, "a", x=-0.30, y=1.05)


def require_shap():
    try:
        import shap
    except ImportError as exc:
        raise ImportError(
            "Fig6b now directly reuses the shap.Explanation + shap.plots.beeswarm method from Fig6b.py. "
            "Please install shap in this Python environment before running make_fig6_nature.py."
        ) from exc
    return shap


def draw_fig6b(shap_top, features_top, feature_names):
    shap = require_shap()
    x_label_font_size = 18
    y_label_font_size = 18
    tick_font_size = 18
    legend_title_size = 18
    legend_tick_size = 18
    zero_line_width = 2.0
    point_alpha = 0.6
    point_size = 14
    axis_spine_width = 1.0
    grid_line_width = 0.8

    normalized_features = normalize_columns(features_top)

    exp = shap.Explanation(
        values=shap_top,
        base_values=np.zeros(shap_top.shape[0]),
        data=normalized_features,
        feature_names=feature_names,
    )

    fig = plt.figure(figsize=(14, 8), dpi=600)
    result = shap.plots.beeswarm(
        exp,
        show=False,
        color_bar=True,
        max_display=len(feature_names),
        plot_size=None,
        alpha=point_alpha,
        s=point_size,
    )

    if isinstance(result, tuple) and len(result) == 2:
        ax, cbar = result
    else:
        fig_axes = plt.gcf().axes
        ax = result if result is not None else max(fig_axes, key=lambda a: a.get_position().width * a.get_position().height)
        cbar = None
        for fig_ax in fig_axes:
            if fig_ax is not ax:
                cbar = fig_ax
                break

    ax.set_yticklabels(feature_names[::-1], fontsize=y_label_font_size, fontweight="bold")
    ax.invert_yaxis()
    ax.set_xlim(-0.95, 0.75)
    ax.set_xticks([-0.8, -0.6, -0.4, -0.2, 0.0, 0.2, 0.4, 0.6])
    ax.set_xticklabels([-0.8, -0.6, -0.4, -0.2, 0.0, 0.2, 0.4, 0.6], fontsize=tick_font_size, fontweight="bold")
    ax.axvline(x=0, color="gray", linestyle="-", linewidth=zero_line_width)
    ax.set_xlabel("SHAP value (impact on model output)", fontsize=x_label_font_size, fontweight="bold")
    ax.set_ylabel("")
    for spine in ax.spines.values():
        spine.set_linewidth(axis_spine_width)
    ax.yaxis.grid(True, linestyle="--", alpha=0.3, linewidth=grid_line_width)
    ax.xaxis.grid(False)

    if cbar is not None:
        if hasattr(cbar, "ax"):
            cbar_ax = cbar.ax
            cbar_ax.yaxis.set_label_position("right")
            cbar_ax.set_yticks([])
            cbar.set_label("Feature value", fontsize=legend_title_size, labelpad=30, fontweight="bold")
            cbar_ax.text(1.8, -0.05, "Low", transform=cbar_ax.transAxes, fontsize=legend_tick_size, va="bottom", ha="left", rotation=90, fontweight="bold")
            cbar_ax.text(1.8, 1.05, "High", transform=cbar_ax.transAxes, fontsize=legend_tick_size, va="top", ha="left", rotation=90, fontweight="bold")
        else:
            cbar.yaxis.set_label_position("right")
            cbar.set_ylabel("Feature value", fontsize=legend_title_size, labelpad=30, fontweight="bold")
            cbar.tick_params(labelsize=legend_tick_size)
            cbar.set_yticks([])
            cbar.text(1.8, -0.05, "Low", transform=cbar.transAxes, fontsize=legend_tick_size, va="bottom", ha="left", rotation=90, fontweight="bold")
            cbar.text(1.8, 1.05, "High", transform=cbar.transAxes, fontsize=legend_tick_size, va="top", ha="left", rotation=90, fontweight="bold")

    #panel_label(ax, "b", x=-0.30, y=1.05)
    plt.tight_layout()
    return fig


def find_patient_row(risk_table, patient_id=None):
    table = risk_table.copy()
    table[ID_COL] = table[ID_COL].astype(str)
    if patient_id:
        matched = table[table[ID_COL] == str(patient_id)]
        if matched.empty:
            raise ValueError(f"Patient ID not found in risk table: {patient_id}")
        return matched.iloc[0]
    candidates = table[(table["bcmpm_high_risk"] == 0) & (table["status"] == 0)].copy()
    if candidates.empty:
        candidates = table.copy()
    candidates["target_distance"] = np.abs(candidates["bcmpm_risk_score"] + 1.51)
    return candidates.sort_values(["target_distance", "bcmpm_risk_score"]).iloc[0]


def select_patient_and_contributions(risk_table, patient_id, npz_root, fused_scaler, result, weight_full, top_n):
    if patient_id:
        row = find_patient_row(risk_table, patient_id)
        split = COHORT_TO_SPLIT.get(str(row["Cohort"]), str(row.get("split", "train")))
        contributions = compute_patient_roi_contributions(str(row[ID_COL]), split, npz_root, fused_scaler, result, weight_full)
        return row, split, contributions

    candidates = risk_table[(risk_table["bcmpm_high_risk"] == 0) & (risk_table["status"] == 0)].copy()
    if candidates.empty:
        candidates = risk_table.copy()
    candidates["target_distance"] = np.abs(candidates["bcmpm_risk_score"] + 1.51)

    fallback = None
    for _, row in candidates.sort_values(["target_distance", "bcmpm_risk_score"]).iterrows():
        split = COHORT_TO_SPLIT.get(str(row["Cohort"]), str(row.get("split", "train")))
        contributions = compute_patient_roi_contributions(str(row[ID_COL]), split, npz_root, fused_scaler, result, weight_full)
        top = contributions[np.argsort(np.abs(contributions))[::-1][:top_n]]
        if fallback is None:
            fallback = (row, split, contributions)
        if len(top) >= top_n and np.any(top > 0) and np.any(top < 0):
            return row, split, contributions
    return fallback


def transform_roi_features(fused_features, fused_scaler, reducer, reduction_method):
    fused_std = fused_scaler.transform(fused_features)
    if isinstance(reducer, dict):
        scaler2 = reducer.get("scaler")
        pca = reducer.get("pca")
        if scaler2 is not None:
            fused_std = scaler2.transform(fused_std)
        return pca.transform(fused_std)
    if reduction_method == "none" or reducer is None:
        return fused_std
    return reducer.transform(fused_std)


def compute_patient_roi_contributions(patient_id, split, npz_root, fused_scaler, result, weight_full):
    npz_path = Path(npz_root) / split / f"{patient_id}_features.npz"
    if not npz_path.exists():
        raise FileNotFoundError(f"Patient NPZ file not found: {npz_path}")
    data = np.load(npz_path)
    fused_features = data["fused_features"]
    roi_lowdim = transform_roi_features(fused_features, fused_scaler, result["reducer"], str(result["reduction_method"]))
    w_img = weight_full[: int(result["best_dim"])]
    return (roi_lowdim @ w_img) / fused_features.shape[0]


def draw_fig6c(fig, ax, contributions, patient_row, top_n=8):
    order = np.argsort(np.abs(contributions))[::-1][:top_n]
    values = contributions[order]
    x = np.arange(len(values))
    cmap = plt.get_cmap("coolwarm")
    norm = plt.Normalize(vmin=float(np.min(values)), vmax=float(np.max(values)))
    ax.bar(x, values, color=cmap(norm(values)), width=0.8, edgecolor="none")
    ax.axhline(0, color="black", linewidth=1.5, alpha=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels([f"ROI {i + 1}" for i in range(len(values))], rotation=0, fontsize=18)
    ax.tick_params(axis="x", pad=10)
    ax.set_ylabel("ROI Contribution", fontsize=22)
    ax.tick_params(axis="y", labelsize=16)
    pred = "High risk" if int(patient_row["bcmpm_high_risk"]) == 1 else "Low risk"
    outcome = "Recurrence" if int(patient_row["status"]) == 1 else "Non-recurrence"
    ax.set_title(f"Patient A: H&E-MPM (BiCA)\npredicted: {pred}, {outcome}", loc="left", fontweight="bold", pad=6)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(1.5)
    ax.spines["bottom"].set_linewidth(1.5)
    sm = cm.ScalarMappable(norm=norm, cmap=cmap)
    sm._A = []
    cbar = fig.colorbar(sm, ax=ax, orientation="vertical", pad=0.04, aspect=30)
    cbar.ax.set_yticks([])
    cbar.ax.text(1.05, 0.0, "Low risk", va="bottom", ha="left", rotation=90, fontsize=18, transform=cbar.ax.transAxes)
    cbar.ax.text(1.05, 1.0, "High risk", va="top", ha="left", rotation=90, fontsize=18, transform=cbar.ax.transAxes)
    #panel_label(ax, "c", x=-0.30, y=1.05)
    return pd.DataFrame({"display_roi": [f"ROI {i + 1}" for i in range(len(values))], "original_roi_index": order + 1, "contribution": values})


def make_importance_score_legend(output_dir):
    fig = plt.figure(figsize=(3.2, 0.78))
    ax = fig.add_axes([0.12, 0.48, 0.76, 0.20])
    norm = mcolors.Normalize(vmin=0, vmax=1)
    sm = cm.ScalarMappable(norm=norm, cmap="magma")
    cbar = fig.colorbar(sm, cax=ax, orientation="horizontal")
    cbar.set_ticks(np.linspace(0, 1, 6))
    cbar.ax.tick_params(labelsize=5.8, length=2, pad=1)
    cbar.set_label("importance score", fontsize=8, labelpad=4)
    fig.text(0.01, 0.88, "d", fontsize=12, fontweight="bold", ha="left", va="top")
    return save_figure(fig, output_dir, "Fig6d_importance_score_legend")


def main():
    parser = argparse.ArgumentParser(description="Generate Fig6a, Fig6b, and Fig6c with the same color and layout style as the provided scripts.")
    parser.add_argument("--train-csv", default="./xxx/xx/train398.csv")
    parser.add_argument("--val-csv", default="./xxx/xx/val330.csv")
    parser.add_argument("--test-csv", default="./xxx/xx/test263.csv")
    parser.add_argument("--risk-csv", default="./xxx/xx/total_all_modalities_risk60.csv")
    parser.add_argument("--npz-root", default="./xxx/xx/patch_tacs_attention_991uni_vahadane_features_256d")
    parser.add_argument("--bcmpm-result-dir", default="./xxx/xx/bcmpm")
    parser.add_argument("--output-dir", default="./xxx/xx/Fig6_Nature")
    parser.add_argument("--max-features", type=int, default=15)
    parser.add_argument("--top-clinical", type=int, default=0)
    parser.add_argument("--patient-id", default=None)
    parser.add_argument("--top-roi", type=int, default=8)
    args = parser.parse_args()

    configure_style()
    output_dir = Path(args.output_dir).resolve()
    result = load_bcmpm_result(args.bcmpm_result_dir)
    x_bg, x_all, patient_table, fused_scaler = build_bcmpm_feature_matrix(args.train_csv, args.val_csv, args.test_csv, args.npz_root)
    weights_equiv, weight_full = equivalent_original_weights(result)
    shap_values = calculate_linear_shap_values(x_bg, x_all, weights_equiv)
    grouped_shap, grouped_x, grouped_names = group_clinical_features(shap_values, x_all)
    shap_top, features_top, feature_names, mean_abs = select_top_features(grouped_shap, grouped_x, grouped_names, args.max_features, args.top_clinical)

    top_df = pd.DataFrame({"feature": feature_names, "mean_abs_shap": mean_abs, "group": ["Fused-image features" if n.startswith("Fused_") else "Clinical features" for n in feature_names]})
    output_dir.mkdir(parents=True, exist_ok=True)
    top_df.to_csv(output_dir / "figure6ab_top_shap_features.csv", index=False)

    fig_a, ax_a = plt.subplots(figsize=(14, 8), dpi=600)
    draw_fig6a(ax_a, shap_top, feature_names, mean_abs)
    fig_a.subplots_adjust(top=0.92, bottom=0.08, left=0.25, right=0.85)
    fig6a = save_figure(fig_a, output_dir, "Fig6a_global_feature_importance")

    fig_b = draw_fig6b(shap_top, features_top, feature_names)
    fig6b = save_figure(fig_b, output_dir, "Fig6b_shap_beeswarm")

    risk_table = pd.read_csv(args.risk_csv)
    patient_row, split, contributions = select_patient_and_contributions(
        risk_table, args.patient_id, args.npz_root, fused_scaler, result, weight_full, args.top_roi
    )

    fig_c, ax_c = plt.subplots(figsize=(max(8, args.top_roi * 1.5), 6.5), dpi=600)
    roi_df = draw_fig6c(fig_c, ax_c, contributions, patient_row, args.top_roi)
    fig_c.tight_layout()
    fig_c.subplots_adjust(bottom=0.15)
    fig6c = save_figure(fig_c, output_dir, "Fig6c_patient_roi_contribution")
    roi_df.insert(0, ID_COL, str(patient_row[ID_COL]))
    roi_df.insert(1, "split", split)
    roi_df.to_csv(output_dir / "figure6c_patient_roi_contributions.csv", index=False)

    print(f"Loaded patients for SHAP: {x_all.shape[0]}")
    print(f"Selected Patient A: {patient_row[ID_COL]} ({split}), risk={patient_row['bcmpm_risk_score']:.3f}")
    for path in [*fig6a, *fig6b, *fig6c]:
        print(f"Saved: {path}")


if __name__ == "__main__":
    main()
