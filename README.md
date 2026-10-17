# BCMPM

**BCMPM: Breast Cancer Multiphoton Multimodal Prognostic Model**

This repository contains the code and configuration for the BCMPM workflow and manuscript analyses. It is intended to support review and reproduction of the reported methods and results. Patient-level data, derived feature archives, model-result files, and microscopy images are not included in this public code repository.

All paths shown in the scripts and examples are placeholders (for example, `./xxx/xx`). Replace them with paths to the authorized data and model files available in your environment. Do not commit restricted data or model artifacts to the public repository.

## Project Workflow

1. **Stage 1 — Data preprocessing:** `data_preprocess.py` reads paired H&E and multiphoton microscopy (MPM) ROI images, checks the dataset structure, and writes metadata and preprocessing summaries.
2. **Stage 2 — BiCA feature fusion:** `BiCA_features.py` extracts H&E and MPM features and applies the configured BiCA fusion method. It writes fused feature archives for downstream modeling.
3. **Model training:** `train.py` trains the selected modality-specific models and BCMPM. `utils.py` provides data loading, model fitting, metric calculation, and result serialization functions called by the training script.
4. **Manuscript analysis:** scripts in `analysis/` calculate risk and clinical analyses and generate figures and tables. Figure and table scripts are grouped under `analysis/figure_table/`.

## Repository Structure

```text
BCMPM/
├── data_preprocess.py
├── BiCA_features.py
├── train.py
├── utils.py
├── config-UNI-feature_crossattn.yaml
├── requirements.txt
├── analysis/
│   ├── model_risk_data.py
│   ├── roi_risk_heatmap.py
│   ├── St. Gallen-data.py
│   └── figure_table/
│       ├── make_fig*.py
│       ├── make_figS*.py
│       ├── table*.py
│       └── ROI_Heatmap/
└── README.md
```

## Environment

Use Python 3 and install the listed dependencies:

```bash
python -m pip install -r requirements.txt
```

The project includes PyTorch and torchvision dependencies. Install versions compatible with the operating system, Python version, and CUDA runtime used for feature extraction or training. Some analysis scripts may require additional packages depending on the selected analysis. The repository does not pin exact package versions; record the environment used for each reproduction run.

## Input Data

### Stage 1: paired H&E and MPM ROI images

Provide paired TIFF ROI files arranged by split and modality. Keep corresponding patient IDs and ROI filenames aligned between modalities. TACS labels are parsed from the image filenames by the preprocessing code.

```text
./xxx/xx/paired_rois/
├── train/
│   ├── he/<patient_id>/*.tif
│   └── mpm/<patient_id>/*.tif
├── val/
│   ├── he/<patient_id>/*.tif
│   └── mpm/<patient_id>/*.tif
└── test/
    ├── he/<patient_id>/*.tif
    └── mpm/<patient_id>/*.tif
```

Preprocessing writes dataset metadata, ROI metadata, and summaries to the selected output directory. Patient IDs must be consistent with the IDs in the cohort CSV files.

### Stage 2: feature fusion

`config-UNI-feature_crossattn.yaml` is the example configuration. Set `data.processed_data_dir` to the Stage 1 output directory containing `dataset_metadata.json` (or the ROI metadata file supported by the script). Set the feature output path to a writable directory. Feature extraction also requires the configured H&E model weights and the MPM feature extractor/model dependencies.

The fused feature output is expected to provide split-specific feature archives for training and analysis:

```text
./xxx/xx/fused_features/
├── train/<patient_id>_features.npz
├── val/<patient_id>_features.npz
└── test/<patient_id>_features.npz
```

The NPZ files must contain the feature arrays and keys expected by `train.py` and `utils.py`.

### Model training

Training requires the training, validation, and test cohort CSV files and the corresponding feature archives. At minimum, cohort files require patient IDs, survival time (`DFS`), event status (`status`), and the clinical variables used by the selected model. Update `TRAIN_CSV`, `VAL_CSV`, `TEST_CSV`, and `TACS_NPZ_ROOT` near the top of `train.py` to point to the authorized inputs. The `--result-dir` argument selects where fitted models, metrics, and result archives are written.

### Figure and table analyses

The analysis scripts consume different combinations of the following inputs:

- modality-specific `results.npz` files and associated fitted model artifacts;
- training, validation, and test cohort CSV files;
- split-specific and total-cohort risk CSV files, including 60-month risk estimates;
- guideline-informed risk CSV files and NRI/IDI summary tables;
- patient-level fused feature archives for feature-importance and ROI-contribution analyses;
- TIFF ROI images and trained or pretrained checkpoints for heatmap generation.

Each analysis script has its own path arguments or path constants. Consult the script before running it and update its example paths to the authorized input files. The public repository does not contain these data or result files. Any permitted reviewer materials should be supplied separately through the journal's confidential peer-review process, not committed to the public code repository.

## Example Commands

Run commands from the repository root and replace each placeholder path with an authorized local path.

### Stage 1: preprocess paired ROIs

```bash
python data_preprocess.py \
  --dataset_root ./xxx/xx/paired_rois \
  --output_dir ./xxx/xx/processed_data
```

### Stage 2: generate fused features

```bash
python BiCA_features.py \
  --config config-UNI-feature_crossattn.yaml \
  --metadata ./xxx/xx/processed_data/dataset_metadata.json \
  --output_dir ./xxx/xx/fused_features \
  --device cuda
```

### Train all supported modalities

Before training, set the cohort CSV and feature-root constants in `train.py` to the correct inputs.

```bash
python train.py --modality all --result-dir ./xxx/xx/model_results
```

### Generate Supplementary Figure S12

Provide a result root containing one subdirectory per modality, each with a `results.npz` file.

```bash
python analysis/figure_table/make_figS12_nature_8modalities.py \
  --result-root ./xxx/xx/model_results \
  --output-dir ./xxx/xx/figures/FigS12 \
  --layout both
```

The script writes PNG, PDF, and SVG figures, along with a CSV summary of the training-set IPCW Youden cutoffs.

## Data Availability, Ethics, and Reproducibility Limits

This public repository contains code and configuration only. Cohort data, derived risk tables, fitted model results, and other restricted inputs are not included. If a subset of those materials is approved for peer review, provide it separately through the journal's confidential submission or review channel and in accordance with applicable ethics approvals, consent terms, and institutional data-use agreements.

Patient-level fused feature NPZ archives and original TIFF ROI images are withheld because of medical ethics and patient privacy requirements. Consequently, analyses that require these files—including some Figure 5/6 feature and ROI analyses and ROI heatmap regeneration—cannot be fully reproduced from this public repository alone. The scripts document the required input formats and expected paths; access to the code does not imply access to restricted patient data.

