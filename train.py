import argparse
import os

from utils import *


MODALITY = 'H&E-MPM'
TIME_POINT = 60

# Required input: see README.md for the required file and directory layout.

# Required input: patient-level CSV with patient_id, DFS, and status columns.
TRAIN_CSV = './xxx/xx/train398.csv'
# Required input: see README.md for the required file and directory layout.
VAL_CSV = './xxx/xx/val330.csv'
# Required input: see README.md for the required file and directory layout.
TEST_CSV = './xxx/xx/test263.csv'

# Required input: see README.md for the required file and directory layout.

# Required input: train/val/test folders of {patient_id}_features.npz archives.
TACS_NPZ_ROOT = './xxx/xx'


ORDER = [
    'Clinical-Only',
    'H&E-Only',
    'H&E-Clinical',
    'MPM-Only',
    'MPM-Clinical',
    'H&E-MPM',
    'BiCA',
    'BCMPM',
]


def bcmpm_train_params():
    return {
        'dropout': 0.5,
        'lr': 5e-4,
        'weight_decay': 0.05,
        'num_epochs': 300,
        'patience': 25,
    }


def bcmpm_cv_params():
    return {
        'n_folds': 5,
        'dropout': 0.4,
        'lr': 5e-4,
        'weight_decay': 0.05,
        'epochs': 100,
        'patience': 10,
        'seed': SEED,
    }


def add_binary_metrics(result):
    risk_dict = result['risk']
    time_dict = result['time']
    event_dict = result['event']

    best_cut, best_youden = find_best_cutoff_youden(
        risk_dict['Train'], time_dict['Train'], event_dict['Train'], time_cut=60
    )
    binary_metrics = calculate_all_binary_metrics(
        risk_dict, time_dict, event_dict, best_cut, time_cut=60
    )

    result['best_cutoff'] = best_cut
    result['binary_metrics'] = binary_metrics
    return best_cut, best_youden, binary_metrics


def print_basic_metrics(mod_name, result, T_tr, E_tr, T_val, E_val, T_test, E_test):
    train_risk = result['risk']['Train']
    val_risk = result['risk']['Validation']
    test_risk = result['risk']['Test']

    train_auc, _, _, _ = time_dependent_roc_ipcw(train_risk, T_tr, E_tr, TIME_POINT)
    val_auc, _, _, _ = time_dependent_roc_ipcw(val_risk, T_val, E_val, TIME_POINT)
    test_auc, _, _, _ = time_dependent_roc_ipcw(test_risk, T_test, E_test, TIME_POINT)

    print(f"\n✅ {mod_name} training complete")
    if result.get('best_dim') is not None:
        print(f"  Selected PCA dimension: {result['best_dim']}")

    print(f"\n📊 C-index results")
    print(f"  Train C-index: {result['cindex']['Train']:.4f}")
    print(f"  Validation C-index: {result['cindex']['Validation']:.4f}")
    print(f"  Test C-index: {result['cindex']['Test']:.4f}")

    print(f"\n📊 {TIME_POINT}-month time-dependent AUC")
    print(f"  Train AUC: {train_auc:.4f}")
    print(f"  Validation AUC: {val_auc:.4f}")
    print(f"  Test AUC: {test_auc:.4f}")


def print_binary_metrics(best_cut, best_youden, binary_metrics):
    print("\n🔍 Training-set Youden threshold")
    print(f"  Best cutoff: {best_cut:.4f} | Youden index: {best_youden:.4f}")
    print("\n📊 Binary metrics at the fixed training-set cutoff")
    for dset in ['Train', 'Validation', 'Test']:
        m = binary_metrics[dset]
        print(f"\n  {dset} cohort")
        print(f"    Sensitivity: {m['sensitivity']:.4f} | Specificity: {m['specificity']:.4f}")
        print(f"    Precision: {m['precision']:.4f} | F1: {m['f1']:.4f}")
        print(f"    Accuracy: {m['accuracy']:.4f} | PPV: {m['ppv']:.4f} | NPV: {m['npv']:.4f}")
        print(f"    Confusion matrix:\n{m['confusion_matrix']}")


def run_prepare_modality(mod_name, source_mod_name, npz_root, config, image_slice,
                         cv_params, train_params, add_binary_before_save=False,
                         show_binary=False):
    noise_std = 0.05

    print("=" * 70)
    print(f"  Training modality: {mod_name}")
    print("=" * 70)
    print("Loading data...")

    modal_data = prepare_all_modalities(TRAIN_CSV, VAL_CSV, TEST_CSV, npz_root)
    (X_tr, T_tr, E_tr), (X_val, T_val, E_val), (X_test, T_test, E_test) = modal_data[source_mod_name]

    result = train_single_modality(
        mod_name, X_tr, T_tr, E_tr, X_val, T_val, E_val, X_test, T_test, E_test,
        config, image_slice, cv_params, train_params, noise_std, SEED
    )

    best_cut = best_youden = binary_metrics = None
    if add_binary_before_save:
        best_cut, best_youden, binary_metrics = add_binary_metrics(result)

    save_modality_result(result, mod_name, BASE_RESULT_DIR)
    print_basic_metrics(mod_name, result, T_tr, E_tr, T_val, E_val, T_test, E_test)

    if show_binary:
        if binary_metrics is None:
            best_cut, best_youden, binary_metrics = add_binary_metrics(result)
        print_binary_metrics(best_cut, best_youden, binary_metrics)

    print("=" * 70)


def load_he_mpm_tacs_split(csv_path, npz_dir):
    df = pd.read_csv(csv_path, dtype={"patient_id": str})
    features = []

    for pid in df["patient_id"]:
        path = os.path.join(npz_dir, f"{pid}_features.npz")
        with np.load(path, allow_pickle=False) as data:
            he = np.asarray(data["he_features"])
            mpm = np.asarray(data["mpm_features"])
            labels = np.asarray(data["tacs_types"])

        if (
            he.ndim != 2
            or mpm.ndim != 2
            or labels.ndim != 1
            or len(labels) == 0
            or he.shape[0] != len(labels)
            or mpm.shape[0] != len(labels)
        ):
            raise ValueError(f"{path}: expected matching, nonempty ROI arrays")

        if (
            not np.isfinite(he).all()
            or not np.isfinite(mpm).all()
            or not np.isfinite(labels).all()
            or not np.equal(labels, np.floor(labels)).all()
            or (labels < 0).any()
            or (labels > 8).any()
        ):
            raise ValueError(f"{path}: invalid features or TACS labels, expected 0..8")

        he_patient = he.mean(axis=0)
        mpm_patient = mpm.mean(axis=0)
        tacs_patient = np.bincount(labels.astype(np.int64), minlength=9) / len(labels)
        features.append(np.concatenate([he_patient, mpm_patient, tacs_patient]))

    return (
        np.stack(features).astype(np.float32),
        df["DFS"].to_numpy(dtype=np.float32),
        df["status"].to_numpy(dtype=np.float32),
    )


def run_he_mpm_tacs():
    train = load_he_mpm_tacs_split(
        TRAIN_CSV,
        os.path.join(TACS_NPZ_ROOT, "train"),
    )
    val = load_he_mpm_tacs_split(
        VAL_CSV,
        os.path.join(TACS_NPZ_ROOT, "val"),
    )
    test = load_he_mpm_tacs_split(
        TEST_CSV,
        os.path.join(TACS_NPZ_ROOT, "test"),
    )

    (X_tr, T_tr, E_tr), (X_val, T_val, E_val), (X_test, T_test, E_test) = train, val, test

    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_tr).astype(np.float32)
    X_val = scaler.transform(X_val).astype(np.float32)
    X_test = scaler.transform(X_test).astype(np.float32)

    print("=" * 70)
    print(f"  Training modality: {MODALITY}")
    print("=" * 70)
    print(f"{MODALITY}: input dimension = {X_tr.shape[1]}")
    print("Input = mean(he_features) + mean(mpm_features) + TACS proportions 0..8")
    print("Note: fused_features is not used for this modality.")

    result = train_single_modality(
        MODALITY,
        X_tr,
        T_tr,
        E_tr,
        X_val,
        T_val,
        E_val,
        X_test,
        T_test,
        E_test,
        {"method": "standard", "candidate_dims": list(range(16, 257, 16))},
        slice(None),
        bcmpm_cv_params(),
        bcmpm_train_params(),
        0.05,
        SEED,
    )

    result["input_feature_definition"] = (
        "mean(he_features) + mean(mpm_features) + proportion(tacs_types=0..8)"
    )
    save_modality_result(result, MODALITY, BASE_RESULT_DIR)

    result_dir = os.path.join(BASE_RESULT_DIR, MODALITY.lower().replace("-", "_"))
    with open(os.path.join(result_dir, "input_scaler.pkl"), "wb") as handle:
        pickle.dump(scaler, handle)

    print_basic_metrics(MODALITY, result, T_tr, E_tr, T_val, E_val, T_test, E_test)
    print("=" * 70)



def run_modality(mod_name):
    if mod_name == MODALITY:
        run_he_mpm_tacs()
        return

    configs = {
        'Clinical-Only': ('Clinical-Only', {'method': 'none', 'candidate_dims': []}, slice(None), True),
        'H&E-Only': ('H&E-Only', {'method': 'pca', 'variance_threshold': 0.90, 'random_state': SEED}, slice(None), False),
        'H&E-Clinical': ('H&E-Clinical', {'method': 'pca', 'variance_threshold': 0.90, 'random_state': SEED}, slice(0, 1024), True),
        'MPM-Only': ('MPM-Only', {'method': 'standard', 'candidate_dims': list(range(16, 257, 16))}, slice(None), False),
        'MPM-Clinical': ('MPM-Clinical', {'method': 'standard', 'candidate_dims': list(range(16, 257, 16))}, slice(0, 1024), False),
        'BiCA': ('BiCA', {'method': 'standard', 'candidate_dims': list(range(16, 257, 16))}, slice(None), False),
        'BCMPM': ('BCMPM', {'method': 'standard', 'candidate_dims': list(range(16, 257, 16))}, slice(0, 256), False),
    }
    source, config, image_slice, show_binary = configs[mod_name]
    run_prepare_modality(
        mod_name, source, TACS_NPZ_ROOT, config, image_slice,
        bcmpm_cv_params(), bcmpm_train_params(),
        add_binary_before_save=(mod_name in {'Clinical-Only', 'H&E-Clinical'}),
        show_binary=show_binary,
    )



ALIASES = {'H&E-MPM-TACS': 'H&E-MPM'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--modality', default='all', choices=['all'] + ORDER + sorted(ALIASES),
        help='Train all eight modalities or select one',
    )
    # Example output directory for fitted models, metrics, and result archives.
    parser.add_argument('--result-dir', default='./xxx/xx')
    args = parser.parse_args()

    global BASE_RESULT_DIR
    # Required input: see README.md for the required file and directory layout.
    BASE_RESULT_DIR = args.result_dir
    selected = ORDER if args.modality == 'all' else [ALIASES.get(args.modality, args.modality)]
    for name in selected:
        run_modality(name)


if __name__ == '__main__':
    main()
