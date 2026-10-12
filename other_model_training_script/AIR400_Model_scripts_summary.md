# AIR-400 EfficientPhys Script Summary

## `build_air400_layout.py`
- Reorganizes the original AIR-400 data from 45 session folders into 15 subject-level folders (`S01`–`S15`) expected by the training repository.
- Uses hard links so the original files are not modified, and writes `clip_remap.csv` to preserve the mapping between new clip IDs and the original data.
- This script is unnecessary if the dataset has already been reorganized into the required subject-based structure.

## `make_fold_configs.py`
- Splits the 15 subjects into predefined Train / Validation / Test groups and generates one YAML configuration file for each fold.
- Disables the repository’s automatic cross-validation (`ENABLE_CV: False`) and fixes the split using `SPLIT_TYPE: custom` and `SPLIT_SUBJECTS`.
- This is the main configuration-generation script when retraining EfficientPhys with a new dataset split.

## `run_step1.py`
- Serves as the main execution wrapper for the whole pipeline: dataset layout generation, fold configuration generation, preprocessing, and training/testing.
- Stage 1 largely duplicates `build_air400_layout.py`, while Stage 2 largely duplicates `make_fold_configs.py`; Stages 3 and 4 call the repository’s `main.py`.
- In practice, this is the central script for running the experiment.

## `run_folds.py`
- Repeatedly runs the training stage of `run_step1.py` for multiple folds so that the full cross-validation experiment can run automatically.
- Extracts per-fold metrics such as MAE, RMSE, Pearson correlation, best epoch, runtime, and GPU usage, then writes them to `folds_summary.txt`.
- This script is mainly used to automate the complete 6-fold experiment.

## `dump_predictions.py`
- Uses already-trained checkpoints to extract per-clip respiration predictions (`pred_rr`) and ground-truth values (`gt_rr`) without retraining the model.
- Generates prediction CSV files for each fold and merges them with `clip_remap.csv` to attach subject, posture, FPS, and other clip-level metadata.
- It is optional for simple fold-level evaluation, but necessary for OOF, posture-wise, and clip-level analyses.
