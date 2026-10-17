import argparse
import os

import numpy as np
import pandas as pd

try:
    from utils import BASE_RESULT_DIR, find_best_cutoff_youden
except ModuleNotFoundError as exc:
    if exc.name != "lifelines":
        raise
    from sklearn.metrics import roc_curve

    # Required input: see README.md for the required file and directory layout.

    BASE_RESULT_DIR = "./xxx/xx/results"

    def find_best_cutoff_youden(risk, time, event, time_cut=60):
        y_true = np.where((time <= time_cut) & (event == 1), 1, 0)
        fpr, tpr, thresholds = roc_curve(y_true, risk)
        youden = tpr - fpr
        best_idx = np.argmax(youden)
        return thresholds[best_idx], youden[best_idx]


ORDER = [
    "MPM-Only",
    "MPM-Clinical",
    "Clinical-Only",
    "H&E-Only",
    "H&E-Clinical",
    "H&E-MPM",
    "BiCA",
    "BCMPM",
]

MODALITY_SUFFIX = {
    "MPM-Only": "mpm_only",
    "MPM-Clinical": "mpm_clinical",
    "Clinical-Only": "clinical_only",
    "H&E-Only": "he_only",
    "H&E-Clinical": "he_clinical",
    "H&E-MPM": "he_mpm",
    "BiCA": "bica",
    "BCMPM": "bcmpm",
}

DATASETS = ["Train", "Validation", "Test"]
DEFAULT_CSV_PATHS = {
    "Train": os.path.join('./xxx/xx', "train398.csv"),
    "Validation": os.path.join('./xxx/xx', "val330.csv"),
    "Test": os.path.join('./xxx/xx', "test263.csv"),
}


def read_clinical_csvs(train_csv, val_csv, test_csv):
    csv_paths = {
        "Train": train_csv,
        "Validation": val_csv,
        "Test": test_csv,
    }
    clinical_dfs = {}
    for dset, csv_path in csv_paths.items():
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"{dset} clinical CSV not found: {csv_path}")
        clinical_dfs[dset] = pd.read_csv(csv_path)
    return clinical_dfs


def load_all_results(result_root):
    all_results = {}
    for modality in ORDER:
        npz_path = os.path.join(result_root, modality.lower().replace("-", "_"), "results.npz")
        if not os.path.exists(npz_path):
            raise FileNotFoundError(f"{modality} results.npz not found: {npz_path}")
        data = np.load(npz_path, allow_pickle=True)
        all_results[modality] = {
            key: data[key].item() if data[key].ndim == 0 else data[key]
            for key in data.files
        }
        print(f"Loaded: {modality}")
    return all_results


def calculate_training_cutoffs(all_results, time_cut):
    cutoff_rows = []
    for modality in ORDER:
        result = all_results[modality]
        best_cutoff, best_youden = find_best_cutoff_youden(
            np.asarray(result["risk"]["Train"], dtype=float),
            np.asarray(result["time"]["Train"], dtype=float),
            np.asarray(result["event"]["Train"], dtype=int),
            time_cut=time_cut,
        )
        result["best_threshold"] = float(best_cutoff)
        result["best_youden"] = float(best_youden)
        cutoff_rows.append(
            {
                "Modality": modality,
                "Training_Set_Cutoff": float(best_cutoff),
                "Training_Set_Youden_Index": float(best_youden),
                "Time_Cut_Months": time_cut,
            }
        )
        print(
            f"{modality}: training-set Youden cutoff = {best_cutoff:.6f}, "
            f"Youden index = {best_youden:.6f}"
        )
    return pd.DataFrame(cutoff_rows)


def append_risk_columns(orig_df, all_results, dset_name):
    out_df = orig_df.copy()
    for modality in ORDER:
        result = all_results[modality]
        risk_values = np.asarray(result["risk"][dset_name], dtype=float).reshape(-1)
        cutoff = result["best_threshold"]
        suffix = MODALITY_SUFFIX[modality]

        if len(risk_values) != len(out_df):
            raise ValueError(
                f"{modality} {dset_name}: risk length ({len(risk_values)}) "
                f"does not match clinical CSV rows ({len(out_df)})."
            )

        out_df[f"{suffix}_risk_score"] = risk_values
        out_df[f"{suffix}_high_risk"] = (risk_values >= cutoff).astype(int)
    return out_df


def export_all_modality_risk_csvs(clinical_dfs, all_results, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    exported_paths = {}
    cohort_frames = []

    for dset_name in DATASETS:
        print(f"Processing {dset_name} cohort...")
        risk_df = append_risk_columns(clinical_dfs[dset_name], all_results, dset_name)
        output_name = f"{dset_name.lower()}_all_modalities_risk.csv"
        output_path = os.path.join(output_dir, output_name)
        risk_df.to_csv(output_path, index=False, encoding="utf-8-sig")
        exported_paths[dset_name] = output_path

        total_part = risk_df.copy()
        total_part.insert(0, "Cohort", dset_name)
        cohort_frames.append(total_part)
        print(f"  Saved: {output_path}")

    total_df = pd.concat(cohort_frames, axis=0, ignore_index=True)
    total_path = os.path.join(output_dir, "total_all_modalities_risk.csv")
    total_df.to_csv(total_path, index=False, encoding="utf-8-sig")
    exported_paths["Total"] = total_path
    print("Processing Total cohort...")
    print(f"  Saved: {total_path}")

    survival_keys = {
        "Train": "train_pred_surv",
        "Validation": "val_pred_surv",
        "Test": "test_pred_surv",
    }
    bcmpm = all_results["BCMPM"]
    risk60_parts = []
    for dset_name in DATASETS:
        survival_key = survival_keys[dset_name]
        if survival_key not in bcmpm:
            raise KeyError(
                f"BCMPM results are missing '{survival_key}'. "
                "Run training with a version that saves 60-month survival predictions."
            )
        survival = np.asarray(bcmpm[survival_key], dtype=float).reshape(-1)
        expected_rows = len(clinical_dfs[dset_name])
        if len(survival) != expected_rows:
            raise ValueError(
                f"BCMPM {dset_name}: survival prediction length ({len(survival)}) "
                f"does not match clinical CSV rows ({expected_rows})."
            )
        risk60_parts.append(1.0 - survival)

    total_risk60_df = total_df.copy()
    total_risk60_df["bcmpm_risk60"] = np.concatenate(risk60_parts)
    risk60_path = os.path.join(output_dir, "total_all_modalities_risk60.csv")
    total_risk60_df.to_csv(risk60_path, index=False, encoding="utf-8-sig")
    exported_paths["Total_Risk60"] = risk60_path
    print(f"  Saved: {risk60_path}")

    return exported_paths


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Export train/validation/test/total CSV files containing risk scores "
            "and high-risk groups for all 9 modalities. Cutoffs are determined "
            "only from the training cohort using find_best_cutoff_youden."
        )
    )
    parser.add_argument(
        "--result-root",
        default=BASE_RESULT_DIR,
        help="Root directory containing modality result folders.",
    )
    parser.add_argument(
        "--train-csv",
        default=DEFAULT_CSV_PATHS["Train"],
        help="Training cohort clinical CSV.",
    )
    parser.add_argument(
        "--val-csv",
        default=DEFAULT_CSV_PATHS["Validation"],
        help="Internal validation cohort clinical CSV.",
    )
    parser.add_argument(
        "--test-csv",
        default=DEFAULT_CSV_PATHS["Test"],
        help="External test cohort clinical CSV.",
    )
    parser.add_argument(
        "--time-cut",
        type=float,
        default=60.0,
        help="Time cutoff in months for Youden index, default: 60.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory. Default: <result-root>/all_modalities_risk.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    result_root = os.path.abspath(args.result_root)
    output_dir = args.output_dir
    if output_dir is None:
        output_dir = os.path.join(result_root, "all_modalities_risk")
    output_dir = os.path.abspath(output_dir)

    print("=" * 80)
    print("  Exporting risk-score CSV files for all 9 modalities")
    print("=" * 80)
    print(f"Result root: {result_root}")
    print(f"Output dir : {output_dir}")
    print(f"Youden cutoff time: {args.time_cut:g} months")

    clinical_dfs = read_clinical_csvs(args.train_csv, args.val_csv, args.test_csv)
    all_results = load_all_results(result_root)
    calculate_training_cutoffs(all_results, args.time_cut)

    exported_paths = export_all_modality_risk_csvs(clinical_dfs, all_results, output_dir)

    print("\nDone. Generated risk CSV files:")
    for dset in ["Train", "Validation", "Test", "Total", "Total_Risk60"]:
        print(f"  {dset}: {exported_paths[dset]}")


if __name__ == "__main__":
    main()