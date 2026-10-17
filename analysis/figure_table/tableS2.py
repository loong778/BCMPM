from pathlib import Path

import numpy as np
import pandas as pd


# Required input: see README.md for the required file and directory layout.


BASE_DIR = Path(__file__).resolve().parent
# Required input: see README.md for the required file and directory layout.
RISK_DIR = Path("./xxx/xx")  # Required input: modality risk CSV files listed in README.md.
# Example output path; generated files are written here.
OUTPUT_DIR = Path("./xxx/xx")  # Example output directory for generated table CSV files.
# Example output path: ./xxx/xx

TIME_CUT_MONTHS = 60
N_BOOTSTRAP = 1000
RANDOM_SEED = 42

MODALITIES = [
    ("Clinical-Only", "clinical_only"),
    ("MPM-Only", "mpm_only"),
    ("MPM-Clinical", "mpm_clinical"),
    ("H&E-Only", "he_only"),
    ("H&E-Clinical", "he_clinical"),
    ("H&E-MPM", "he_mpm"),
    ("BiCA", "bica"),
    ("BCMPM", "bcmpm"),
]

DATASETS = [
    ("Training", "train_all_modalities_risk.csv"),
    ("Internal-Validation", "validation_all_modalities_risk.csv"),
    ("External-testing", "test_all_modalities_risk.csv"),
]

METRICS = [
    ("sens", "Sens(95%CI)"),
    ("spec", "Spec(95%CI)"),
    ("acc", "Acc(95%CI)"),
    ("ppv", "PPV(95%CI)"),
    ("npv", "NPV(95%CI)"),
]


def get_5year_event_label(df: pd.DataFrame) -> np.ndarray:
    if "status5" in df.columns:
        return df["status5"].astype(int).to_numpy()

    required = {"DFS", "status"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            "Cannot build 5-year event label. Missing columns: "
            + ", ".join(sorted(missing))
        )
    return ((df["DFS"].to_numpy() <= TIME_CUT_MONTHS) & (df["status"].to_numpy() == 1)).astype(int)


def binary_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)

    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))

    sens = safe_div(tp, tp + fn)
    spec = safe_div(tn, tn + fp)
    ppv = safe_div(tp, tp + fp)
    npv = safe_div(tn, tn + fn)
    acc = safe_div(tp + tn, tp + tn + fp + fn)
    # f1 = safe_div(2 * ppv * sens, ppv + sens)

    return {
        "sens": sens,
        "spec": spec,
        "acc": acc,
        # "f1": f1,
        "ppv": ppv,
        "npv": npv,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "n": len(y_true),
        "events": int(np.sum(y_true == 1)),
    }


def safe_div(numer: float, denom: float) -> float:
    if denom == 0:
        return np.nan
    return float(numer) / float(denom)


def bootstrap_metric_ci(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metric_name: str,
    rng: np.random.Generator,
) -> tuple[float, float]:
    n = len(y_true)
    boot_values = []

    for _ in range(N_BOOTSTRAP):
        idx = rng.integers(0, n, size=n)
        value = binary_metrics(y_true[idx], y_pred[idx])[metric_name]
        if not np.isnan(value):
            boot_values.append(value)

    if not boot_values:
        return np.nan, np.nan

    low, high = np.percentile(boot_values, [2.5, 97.5])
    return float(low), float(high)


def format_metric(value: float, low: float, high: float) -> str:
    if np.isnan(value) or np.isnan(low) or np.isnan(high):
        return "NA"
    return f"{value:.3f}({low:.3f}-{high:.3f})"


def validate_columns(df: pd.DataFrame, suffix: str, dataset_name: str) -> None:
    pred_col = f"{suffix}_high_risk"
    if pred_col not in df.columns:
        raise ValueError(f"{dataset_name}: missing required column {pred_col}")


def compute_rows() -> tuple[pd.DataFrame, pd.DataFrame]:
    numeric_rows = []
    formatted_rows = []
    rng = np.random.default_rng(RANDOM_SEED)

    for method, suffix in MODALITIES:
        for dataset_name, filename in DATASETS:
            csv_path = RISK_DIR / filename
            if not csv_path.exists():
                raise FileNotFoundError(f"Missing input file: {csv_path}")

            df = pd.read_csv(csv_path)
            validate_columns(df, suffix, dataset_name)

            y_true = get_5year_event_label(df)
            y_pred = df[f"{suffix}_high_risk"].astype(int).to_numpy()

            valid = ~pd.isna(y_true) & ~pd.isna(y_pred)
            y_true = y_true[valid].astype(int)
            y_pred = y_pred[valid].astype(int)

            point = binary_metrics(y_true, y_pred)

            row_numeric = {
                "Method": method,
                "Dataset": dataset_name,
                "N": point["n"],
                "Events": point["events"],
                "TP": point["tp"],
                "TN": point["tn"],
                "FP": point["fp"],
                "FN": point["fn"],
            }
            row_formatted = {
                "Method": method,
                "Dataset": dataset_name,
            }

            for metric_key, display_name in METRICS:
                low, high = bootstrap_metric_ci(y_true, y_pred, metric_key, rng)
                row_numeric[metric_key] = point[metric_key]
                row_numeric[f"{metric_key}_95CI_low"] = low
                row_numeric[f"{metric_key}_95CI_high"] = high
                row_formatted[display_name] = format_metric(point[metric_key], low, high)

            numeric_rows.append(row_numeric)
            formatted_rows.append(row_formatted)

    numeric_df = pd.DataFrame(numeric_rows)
    formatted_df = pd.DataFrame(formatted_rows)
    return numeric_df, formatted_df


def add_blank_method_cells_for_word_table(formatted_df: pd.DataFrame) -> pd.DataFrame:
    table_df = formatted_df.copy()
    duplicated = table_df["Method"].eq(table_df["Method"].shift())
    table_df.loc[duplicated, "Method"] = ""
    return table_df


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    numeric_df, formatted_df = compute_rows()
    word_ready_df = add_blank_method_cells_for_word_table(formatted_df)


    word_ready_path = OUTPUT_DIR / "TableS2_modalities_metrics_word_ready.csv"



    word_ready_df.to_csv(word_ready_path, index=False, encoding="utf-8-sig")


    print(f"Saved Word-ready table: {word_ready_path}")


if __name__ == "__main__":
    main()
