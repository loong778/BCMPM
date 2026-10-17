import argparse
import pickle
import warnings
from pathlib import Path

import matplotlib as mpl
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from matplotlib.patches import Rectangle
from sklearn.metrics import roc_curve
from sklearn.preprocessing import StandardScaler


TARGET_MOD = "BCMPM"
TARGET_TIME = 60
ID_COL = "patient_id"
SPLIT_MAP = {"Training": "train", "Validation": "val", "Test": "test"}

COLORS = {
    "low": "#1f77b4",
    "high": "#d62728",
    "event": "#E45756",
    "censor": "#74ADD1",
    "divider": "#253494",
}

FEATURE_COLORS = {
    "BC-MPM": {"Low risk": "#1f77b4", "High risk": "#d62728"},
    "Recurrence": {"No": "#D9D9D9", "Yes": "#000000"},
    "Group": {"Training cohort": "#D9D9D9", "Validation cohort": "#969696", "Test cohort": "#525252"},
    "Age": {1: "#DADAEB", 2: "#807DBA"},
    "Molecular subtypes": {1: "#BDBDBD", 2: "#FFF7A8", 3: "#FDD049", 4: "#B15928"},
    "Tumor size": {1: "#BDBDBD", 2: "#9ECAE1", 3: "#3182BD"},
    "Nodal status": {0: "#BDBDBD", 1: "#A6CEE3", 2: "#1F78B4"},
    "Clinical stage": {1: "#FDBF6F", 2: "#FF7F00", 3: "#D94801"},
    "Histological grade": {1: "#B2DF8A", 2: "#33A02C", 3: "#1B7837"},
    "Chemotherapy": {0: "#D9D9D9", 1: "#737373"},
    "Radiation Therapy": {0: "#B3CDE3", 1: "#1F78B4"},
}

FEATURE_COLUMNS = {
    "Age": "age",
    "Molecular subtypes": "molecular subtypes",
    "Tumor size": "tumor size",
    "Nodal status": "lymph node metastasis",
    "Clinical stage": "clinical stage",
    "Histological grade": "histological grade",
    "Chemotherapy": "CT",
    "Radiation Therapy": "RT",
}

FEATURE_LABELS = {
    "BC-MPM": {"Low risk": "Low risk", "High risk": "High risk"},
    "Recurrence": {"No": "No", "Yes": "Yes"},
    "Group": {"Training cohort": "Training cohort", "Validation cohort": "Validation cohort", "Test cohort": "Test cohort"},
    "Age": {1: "≤50", 2: ">50"},
    "Molecular subtypes": {1: "Luminal A", 2: "Luminal B", 3: "HER2-enriched", 4: "Triple-negative"},
    "Tumor size": {1: "≤2cm", 2: "2-5cm", 3: ">5cm"},
    "Nodal status": {0: "0", 1: "1-3", 2: "≥4"},
    "Clinical stage": {1: "I", 2: "II", 3: "III"},
    "Histological grade": {1: "G1", 2: "G2", 3: "G3"},
    "Chemotherapy": {0: "No", 1: "Yes"},
    "Radiation Therapy": {0: "No", 1: "Yes"},
}


def configure_style():
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 7,
            "axes.labelsize": 7,
            "axes.titlesize": 8,
            "xtick.labelsize": 6.5,
            "ytick.labelsize": 6.5,
            "legend.fontsize": 5.6,
            "axes.linewidth": 0.8,
            "xtick.major.width": 0.8,
            "ytick.major.width": 0.8,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def panel_label(ax, label, x=-0.08, y=1.06, fontsize=13):
    ax.text(x, y, label, transform=ax.transAxes, fontsize=fontsize, fontweight="bold", ha="left", va="top")


def load_bcmpm_result(result_dir):
    result_dir = Path(result_dir)
    result_npz = result_dir / "results.npz"
    reducer_path = result_dir / "reducer.pkl"
    model_path = result_dir / "cox_model.pth"
    if not result_npz.exists() or not reducer_path.exists() or not model_path.exists():
        raise FileNotFoundError(f"BCMPM result files are incomplete in {result_dir}")
    data = np.load(result_npz, allow_pickle=True)
    result = {key: data[key].item() if data[key].shape == () else data[key] for key in data.files}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with open(reducer_path, "rb") as handle:
            result["reducer"] = pickle.load(handle)
    try:
        result["model_state"] = torch.load(model_path, map_location="cpu", weights_only=True)
    except TypeError:
        result["model_state"] = torch.load(model_path, map_location="cpu")
    return result


def derive_cutoff(risk, high_risk):
    risk = np.asarray(risk, dtype=float)
    high_risk = np.asarray(high_risk, dtype=int)
    low_values = risk[high_risk == 0]
    high_values = risk[high_risk == 1]
    if low_values.size and high_values.size:
        return float((low_values.max() + high_values.min()) / 2.0)
    return float(np.median(risk))


def find_training_youden_cutoff(risk, time, event, eval_time=TARGET_TIME):
    y_true = ((np.asarray(time) <= eval_time) & (np.asarray(event) == 1)).astype(int)
    fpr, tpr, thresholds = roc_curve(y_true, np.asarray(risk, dtype=float))
    return float(thresholds[np.argmax(tpr - fpr)])


def build_patient_table_from_model(result, train_csv, val_csv, test_csv):
    train_df = pd.read_csv(train_csv)
    val_df = pd.read_csv(val_csv)
    test_df = pd.read_csv(test_csv)
    clinical_df = pd.concat([train_df, val_df, test_df], ignore_index=True).copy()

    risk_train = np.asarray(result["risk"]["Train"], dtype=float)
    risk_val = np.asarray(result["risk"]["Validation"], dtype=float)
    risk_test = np.asarray(result["risk"]["Test"], dtype=float)
    time_train = np.asarray(result["time"]["Train"], dtype=float)
    time_val = np.asarray(result["time"]["Validation"], dtype=float)
    time_test = np.asarray(result["time"]["Test"], dtype=float)
    event_train = np.asarray(result["event"]["Train"], dtype=int)
    event_val = np.asarray(result["event"]["Validation"], dtype=int)
    event_test = np.asarray(result["event"]["Test"], dtype=int)

    expected_len = len(risk_train) + len(risk_val) + len(risk_test)
    if len(clinical_df) != expected_len:
        raise ValueError(
            f"Clinical table length ({len(clinical_df)}) does not match model outputs ({expected_len})."
        )

    all_risk = np.concatenate([risk_train, risk_val, risk_test])
    all_time = np.concatenate([time_train, time_val, time_test])
    all_event = np.concatenate([event_train, event_val, event_test])
    cohorts = np.array(["Training"] * len(risk_train) + ["Validation"] * len(risk_val) + ["Test"] * len(risk_test))
    cutoff = find_training_youden_cutoff(risk_train, time_train, event_train, TARGET_TIME)

    clinical_df[ID_COL] = clinical_df[ID_COL].astype(str)
    clinical_df["bcmpm_risk_score"] = all_risk
    clinical_df["bcmpm_high_risk"] = (all_risk >= cutoff).astype(int)
    clinical_df["DFS"] = all_time
    clinical_df["status"] = all_event
    clinical_df["Cohort"] = cohorts
    return clinical_df, cutoff


def load_patient_table(risk_csv):
    df = pd.read_csv(risk_csv)
    required = [
        ID_COL,
        "bcmpm_risk_score",
        "bcmpm_high_risk",
        "DFS",
        "status",
        "Cohort",
        "age",
        "molecular subtypes",
        "tumor size",
        "lymph node metastasis",
        "clinical stage",
        "histological grade",
        "CT",
        "RT",
    ]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in {risk_csv}: {missing}")
    df[ID_COL] = df[ID_COL].astype(str)
    return df


def draw_feature_legend(ax):
    ax.axis("off")
    size_legend_title = 16
    size_legend_text = 14
    legend_x_start = 0.015
    legend_x_end = 0.985
    legend_v_gap = 0.24
    legend_text_gap = 0.01
    legend_square_size = 200
    legend_vars = list(FEATURE_COLORS.keys())
    x_centers = np.linspace(legend_x_start, legend_x_end, len(legend_vars))
    for x, var in zip(x_centers, legend_vars):
        ax.text(x, 0.95, var, transform=ax.transAxes, ha="center", fontsize=size_legend_title, fontweight="bold")
        y = 0.75
        for value, color in FEATURE_COLORS[var].items():
            ax.scatter(x, y, marker="s", s=legend_square_size, color=color, transform=ax.transAxes, linewidths=0)
            ax.text(
                x + legend_text_gap,
                y,
                FEATURE_LABELS[var].get(value, str(value)),
                transform=ax.transAxes,
                ha="left",
                va="center",
                fontsize=size_legend_text,
            )
            y -= legend_v_gap


def draw_feature_band(ax, values, color_map, label):
    color_list = []
    for value in values:
        if value in color_map:
            color_list.append(color_map[value])
        elif str(value) in color_map:
            color_list.append(color_map[str(value)])
        else:
            color_list.append("#CCCCCC")
    matrix = np.asarray([mcolors.to_rgb(color) for color in color_list], dtype=float).reshape(1, -1, 3)
    ax.imshow(matrix, aspect="auto", interpolation="nearest", extent=[0, len(values), 0, 1])
    ax.set_xticks([])
    ax.set_yticks([0.5])
    ax.set_yticklabels([label], fontsize=16, rotation=0, va="center")
    ax.tick_params(axis="y", length=0, pad=2)
    for spine in ax.spines.values():
        spine.set_visible(False)


def draw_risk_landscape(fig, outer_spec, df, cutoff):
    size_axis_label = 16
    size_tick = 14
    size_cutoff = 18
    panel = fig.add_subplot(outer_spec)
    panel.axis("off")
    panel_label(panel, "a", x=-0.055, y=1.095, fontsize=24)
    panel.text(0.5, 1.095, f"Whole cohort risk landscape(n={len(df)})", transform=panel.transAxes, ha="center", va="top", fontsize=18, fontweight="bold")

    inner = outer_spec.subgridspec(
        16,
        1,
        height_ratios=[
            1.00,
            2.20,
            0.40,
            2.20,
            0.20,
            0.20,
            0.20,
            0.30,
            0.20,
            0.20,
            0.20,
            0.20,
            0.20,
            0.20,
            0.20,
            0.20,
        ],
        hspace=0.08,
    )
    ax_legend = fig.add_subplot(inner[0])
    draw_feature_legend(ax_legend)

    order = np.argsort(df["bcmpm_risk_score"].to_numpy(dtype=float))
    sorted_df = df.iloc[order].reset_index(drop=True)
    risk_sorted = sorted_df["bcmpm_risk_score"].to_numpy(dtype=float)
    event_sorted = sorted_df["status"].to_numpy(dtype=int)
    time_sorted = sorted_df["DFS"].to_numpy(dtype=float)
    high_sorted = sorted_df["bcmpm_high_risk"].to_numpy(dtype=int)
    x = np.arange(len(sorted_df))
    n_low = int(np.sum(high_sorted == 0))

    ax_risk = fig.add_subplot(inner[1])
    ax_risk.fill_between(x, risk_sorted, cutoff, where=risk_sorted < cutoff, color="#1f77b4", alpha=0.90)
    ax_risk.fill_between(x, cutoff, risk_sorted, where=risk_sorted >= cutoff, color="#d62728", alpha=0.90)
    ax_risk.plot(x, risk_sorted, color="black", lw=1.20)
    ax_risk.axvline(n_low, color="navy", lw=1.20, ls="--")
    ax_risk.text(
        0.62,
        0.88,
        f"Cutoff = {cutoff:.4f}",
        transform=ax_risk.transAxes,
        ha="left",
        va="top",
        fontsize=size_cutoff,
        fontweight="bold",
        color="black",
        bbox=dict(facecolor="white", alpha=0.8, edgecolor="none", pad=1.5),
    )
    ax_risk.set_ylabel("Risk score", fontsize=size_axis_label, fontweight="bold")
    ax_risk.set_xticks([])
    ax_risk.set_xlim(0, len(sorted_df))
    ax_risk.tick_params(axis="y", labelsize=size_tick)
    for side in ["top", "right", "bottom"]:
        ax_risk.spines[side].set_visible(False)

    ax_gap = fig.add_subplot(inner[2])
    ax_gap.axis("off")

    ax_time = fig.add_subplot(inner[3])
    recurrence = event_sorted == 1
    ax_time.scatter(x[~recurrence], time_sorted[~recurrence], s=8, facecolors="none", edgecolors="#1f77b4", linewidths=0.7, alpha=0.70)
    ax_time.scatter(x[recurrence], time_sorted[recurrence], s=8, c="#d62728", linewidths=0, alpha=0.70)
    ax_time.axvline(n_low, color="navy", lw=1.20, ls="--")
    ax_time.set_ylabel("DFS (month)", fontsize=size_axis_label, fontweight="bold")
    ax_time.set_xticks([])
    ax_time.set_xlim(0, len(sorted_df))
    ax_time.set_ylim(0, float(np.nanmax(time_sorted)) + 5)
    ax_time.tick_params(axis="y", labelsize=size_tick)
    for side in ["top", "right"]:
        ax_time.spines[side].set_visible(False)

    band_values = {
        "BC-MPM": np.where(risk_sorted < cutoff, "Low risk", "High risk"),
        "Recurrence": np.where(event_sorted == 1, "Yes", "No"),
        "Group": sorted_df["Cohort"].map({"Training": "Training cohort", "Validation": "Validation cohort", "Test": "Test cohort"}).to_numpy(),
    }
    for feature, col in FEATURE_COLUMNS.items():
        band_values[feature] = sorted_df[col].to_numpy()

    bands = ["BC-MPM", "Recurrence", "Group", "Age", "Molecular subtypes", "Tumor size", "Nodal status", "Clinical stage", "Histological grade", "Chemotherapy", "Radiation Therapy"]
    band_rows = [4, 5, 6, 8, 9, 10, 11, 12, 13, 14, 15]
    for feature, row in zip(bands, band_rows):
        ax = fig.add_subplot(inner[row])
        draw_feature_band(ax, band_values[feature], FEATURE_COLORS[feature], feature)
        ax.set_xlim(0, len(sorted_df) - 1)

    ax_group_age_gap = fig.add_subplot(inner[7])
    ax_group_age_gap.axis("off")

    return sorted_df


def fit_fused_scaler(train_csv, npz_train_dir):
    train_df = pd.read_csv(train_csv)
    features = []
    for pid in train_df[ID_COL].astype(str):
        path = Path(npz_train_dir) / f"{pid}_features.npz"
        if not path.exists():
            continue
        data = np.load(path)
        features.append(np.mean(data["fused_features"], axis=0))
    if not features:
        raise FileNotFoundError(f"No training ROI feature files found in {npz_train_dir}")
    scaler = StandardScaler()
    scaler.fit(np.vstack(features).astype(np.float32))
    return scaler


def compute_roi_contributions(patient_id, split, feature_root, fused_scaler, reducer, w_img, reduction_method):
    npz_path = Path(feature_root) / split / f"{patient_id}_features.npz"
    if not npz_path.exists():
        return None
    data = np.load(npz_path)
    fused_features = data["fused_features"]
    fused_std = fused_scaler.transform(fused_features)
    if reduction_method == "pca" and isinstance(reducer, dict):
        fused_std = reducer["scaler"].transform(fused_std)
        roi_lowdim = reducer["pca"].transform(fused_std)
    else:
        roi_lowdim = reducer.transform(fused_std)
    return np.dot(roi_lowdim, w_img) / max(fused_features.shape[0], 1)


def build_roi_matrix(df, result, feature_root, train_csv, top_n):
    reducer = result["reducer"]
    reduction_method = str(result["reduction_method"])
    d_img = int(result["best_dim"])
    weight_full = result["model_state"]["linear.weight"].detach().cpu().numpy().ravel()
    w_img = weight_full[:d_img]
    fused_scaler = fit_fused_scaler(train_csv, Path(feature_root) / "train")

    rows = []
    for _, row in df.iterrows():
        split = SPLIT_MAP.get(row["Cohort"])
        if split is None:
            continue
        contrib = compute_roi_contributions(row[ID_COL], split, feature_root, fused_scaler, reducer, w_img, reduction_method)
        if contrib is None or len(contrib) == 0:
            continue
        top = np.sort(np.asarray(contrib, dtype=float))[::-1][:top_n]
        rows.append({"patient_id": row[ID_COL], "risk": row["bcmpm_risk_score"], "high_risk": row["bcmpm_high_risk"], "top_contrib": top, "roi_count": len(contrib)})
    roi_df = pd.DataFrame(rows).sort_values("risk", ascending=True).reset_index(drop=True)
    if roi_df.empty:
        raise ValueError("No ROI contributions could be computed.")
    max_len = max(len(x) for x in roi_df["top_contrib"])
    matrix = np.full((max_len, len(roi_df)), np.nan)
    for i, contrib in enumerate(roi_df["top_contrib"]):
        matrix[: len(contrib), i] = contrib
    matrix = matrix[~np.all(np.isnan(matrix), axis=1)]
    return roi_df, matrix


def draw_roi_landscape(fig, outer_spec, df, result, feature_root, train_csv, top_n):
    panel = fig.add_subplot(outer_spec)
    panel.axis("off")
    panel_label(panel, "b", x=-0.07, y=1.02)
    panel.set_title("ROI-level contribution landscape of BCMPM risk prediction", fontweight="bold", pad=4)

    inner = outer_spec.subgridspec(2, 2, height_ratios=[0.10, 1.0], width_ratios=[1.0, 0.030], hspace=0.025, wspace=0.018)
    ax_band = fig.add_subplot(inner[0, 0])
    ax_heat = fig.add_subplot(inner[1, 0])
    cax = fig.add_subplot(inner[1, 1])

    roi_df, matrix = build_roi_matrix(df, result, feature_root, train_csv, top_n)
    n_patients = len(roi_df)
    n_low = int(np.sum(roi_df["high_risk"].to_numpy(dtype=int) == 0))

    band = np.zeros((1, n_patients, 3), dtype=float)
    band[0, :n_low, :] = mcolors.to_rgb(COLORS["low"])
    band[0, n_low:, :] = mcolors.to_rgb(COLORS["high"])
    ax_band.imshow(band, aspect="auto", interpolation="nearest", extent=[0, n_patients, 0, 1])
    ax_band.text(n_low / 2, 0.5, "BCMPM low-risk", ha="center", va="center", color="white", fontsize=6.0, fontweight="bold")
    ax_band.text(n_low + (n_patients - n_low) / 2, 0.5, "BCMPM high-risk", ha="center", va="center", color="white", fontsize=6.0, fontweight="bold")
    ax_band.set_xticks([])
    ax_band.set_yticks([])
    for spine in ax_band.spines.values():
        spine.set_visible(False)

    valid = matrix[~np.isnan(matrix)]
    vmin, vmax = np.percentile(valid, [3, 97])
    abs_max = max(abs(vmin), abs(vmax), 1e-8)
    norm = mcolors.TwoSlopeNorm(vmin=-abs_max, vcenter=0, vmax=abs_max)
    cmap = plt.get_cmap("coolwarm").copy()
    cmap.set_bad("white")
    image = ax_heat.imshow(matrix, aspect="auto", interpolation="nearest", cmap=cmap, norm=norm, extent=[0, n_patients, matrix.shape[0], 0])
    ax_heat.axvline(n_low, color="#4D4D4D", lw=0.75, ls="--")
    ax_heat.set_xlabel("Patients ranked by BCMPM-derived risk score", fontsize=6.0, fontweight="bold", labelpad=4)
    ax_heat.set_ylabel("ROI contribution rank within each patient", fontsize=6.0)
    ax_heat.set_xticks([])
    ax_heat.set_yticks([])
    for side in ["top", "right"]:
        ax_heat.spines[side].set_visible(False)

    cbar = fig.colorbar(image, cax=cax)
    cbar.ax.set_yticks([])
    cbar.ax.text(1.2, 1.0, "Positive", transform=cbar.ax.transAxes, rotation=90, ha="left", va="top", fontsize=5.6, fontweight="bold")
    cbar.ax.text(1.2, 0.0, "Negative", transform=cbar.ax.transAxes, rotation=90, ha="left", va="bottom", fontsize=5.6, fontweight="bold")
    cbar.ax.text(2.4, 0.5, "Contribution", transform=cbar.ax.transAxes, rotation=90, ha="left", va="center", fontsize=5.6, fontweight="bold")
    return roi_df


def make_fig5a(df, output_dir, cutoff):
    with mpl.rc_context({"font.family": "Arial", "font.size": 14, "axes.linewidth": 1.0}):
        fig = plt.figure(figsize=(28.0, 13.0))
        gs = fig.add_gridspec(1, 1, left=0.055, right=0.985, top=0.920, bottom=0.030)
        sorted_df = draw_risk_landscape(fig, gs[0], df, cutoff)

        output_dir.mkdir(parents=True, exist_ok=True)
        sorted_df[[ID_COL, "Cohort", "bcmpm_risk_score", "bcmpm_high_risk", "DFS", "status"]].to_csv(output_dir / "figure5a_sorted_risk_landscape.csv", index=False)

        png = output_dir / "Fig5a_risk_landscape_600dpi.png"
        svg = output_dir / "Fig5a_risk_landscape_600dpi.svg"
        fig.savefig(png, dpi=600, bbox_inches="tight")
        fig.savefig(svg, bbox_inches="tight")
        plt.close(fig)
    return png, svg


def make_fig5b(df, result, output_dir, feature_root, train_csv, top_n):
    fig = plt.figure(figsize=(9.0, 3.7))
    gs = fig.add_gridspec(1, 1, left=0.090, right=0.935, top=0.860, bottom=0.180)
    roi_df = draw_roi_landscape(fig, gs[0], df, result, feature_root, train_csv, top_n)

    output_dir.mkdir(parents=True, exist_ok=True)
    roi_df[[ID_COL, "risk", "high_risk", "roi_count"]].to_csv(output_dir / "figure5b_roi_landscape_summary.csv", index=False)

    png = output_dir / "Fig5b_roi_contribution_landscape_600dpi.png"
    svg = output_dir / "Fig5b_roi_contribution_landscape_600dpi.svg"
    fig.savefig(png, dpi=600, bbox_inches="tight")
    fig.savefig(svg, bbox_inches="tight")
    plt.close(fig)
    return png, svg


def main():
    parser = argparse.ArgumentParser(description="Generate Nature-style Figure 5 panels.")
    parser.add_argument("--bcmpm-result-dir", default="./xxx/xx/bcmpm")
    parser.add_argument("--feature-root", default="./xxx/xx/patch_tacs_attention_991uni_vahadane_features_256d")
    parser.add_argument("--train-csv", default="./xxx/xx/train398.csv")
    parser.add_argument("--val-csv", default="./xxx/xx/val330.csv")
    parser.add_argument("--test-csv", default="./xxx/xx/test263.csv")
    parser.add_argument("--output-dir", default="./xxx/xx/Fig5_Nature")
    parser.add_argument("--top-n-roi", type=int, default=20)
    args = parser.parse_args()

    configure_style()
    result = load_bcmpm_result(args.bcmpm_result_dir)
    df, cutoff = build_patient_table_from_model(result, args.train_csv, args.val_csv, args.test_csv)
    output_dir = Path(args.output_dir).resolve()
    fig5a_png, fig5a_svg = make_fig5a(df, output_dir, cutoff)

    fig5b_png, fig5b_svg = make_fig5b(df, result, output_dir, args.feature_root, args.train_csv, args.top_n_roi)

    print(f"Loaded patients: {len(df)}")
    print(f"Training-set cutoff: {cutoff:.4f}")
    print(f"Saved: {fig5a_png}")
    print(f"Saved: {fig5a_svg}")
    print(f"Saved: {fig5b_png}")
    print(f"Saved: {fig5b_svg}")


if __name__ == "__main__":
    main()
