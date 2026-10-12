"""
make_fold_configs.py

Emit 6 fold-specific config YAMLs that bypass the repo's automatic CV.

Why bypass it: main.py:handle_cross_validation() shuffles the subject list with
np.random.shuffle and slices it into folds with fold_size = n // CV_FOLD_NUM.
With our 15 subjects that yields folds of 2,2,2,2,2,5 in an order we do not
control, so it would not reproduce the DSP folds. Setting ENABLE_CV: False and
supplying SPLIT_TYPE: custom + SPLIT_SUBJECTS per split pins the folds exactly.
base_dataset.py:177-186 handles the custom path.

Test folds come from run_cv_experiment.py:42-49 (the DSP 6-fold definition).
Validation subjects are fixed here so the runs are reproducible and the table
can be reprinted in an appendix.

Usage (PowerShell, one line):
  python make_fold_configs.py `
      --base "D:\\air400_dl\\configs\\body_roi\\efficientphys_6_fold_cv_air400_with_roi_farneback_of_body.yaml" `
      --air400-dir "D:\\hdd\\RE_baby_air400" `
      --cache-dir "D:\\air400_cache" `
      --output-dir "D:\\air400_runs" `
      --body-detector "D:\\air400_dl\\weights\\yolov8n.pt" `
      --model EfficientPhys --of farneback --tag pilot `
      --out-dir "D:\\air400_dl\\configs\\ours"

Then run one fold:
  python main.py --config D:\\air400_dl\\configs\\ours\\efficientphys_farneback_pilot_fold1.yaml
"""

from __future__ import annotations

import argparse
import copy
import os
import sys

import yaml

ALL_SUBJECTS = [f"S{i:02d}" for i in range(1, 16)]

# run_cv_experiment.py:42-49
FOLD_TEST = {
    1: ["S01", "S02"],
    2: ["S05", "S11"],
    3: ["S06", "S08"],
    4: ["S04", "S09", "S12"],
    5: ["S03", "S07", "S13"],
    6: ["S10", "S14", "S15"],
}

# Fixed validation subjects. Constraints applied when choosing these:
#   (1) disjoint from that fold's test set
#   (2) no subject used for validation more than once across the 6 folds
#   (3) low-clip-count subjects preferred, so the training set stays large and
#       roughly even across folds (train clips end up in 245-289, not 225-289)
# The three highest-count subjects (S05=45, S06=50, S14=25) are never used for
# validation and therefore always contribute to training.
FOLD_VALID = {
    1: ["S03", "S12"],
    2: ["S13", "S15"],
    3: ["S09", "S10"],
    4: ["S01", "S08"],
    5: ["S04", "S11"],
    6: ["S02", "S07"],
}


def build_split(fold: int):
    test = sorted(FOLD_TEST[fold])
    valid = sorted(FOLD_VALID[fold])
    overlap = set(test) & set(valid)
    if overlap:
        raise ValueError(f"fold {fold}: test/valid overlap {sorted(overlap)}")
    train = sorted(set(ALL_SUBJECTS) - set(test) - set(valid))
    if len(train) + len(valid) + len(test) != 15:
        raise ValueError(f"fold {fold}: subjects do not sum to 15")
    return train, valid, test


def data_block(subjects, data_format, preprocess):
    return {
        "DATASET": "AIR_400",
        "SPLIT_TYPE": "custom",
        "SPLIT_SUBJECTS": list(subjects),
        "SPLIT_RATIO": 0.0,
        "DATA_FORMAT": data_format,
        "PREPROCESS": copy.deepcopy(preprocess),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True, help="an existing repo config to use as template")
    ap.add_argument("--air400-dir", required=True, help="restructured 15-subject root")
    ap.add_argument("--cache-dir", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--body-detector", required=True, help="yolov8 weights path")
    ap.add_argument("--face-detector", default="")
    ap.add_argument("--out-dir", required=True, help="where to write the 6 YAMLs")
    ap.add_argument("--model", default="EfficientPhys",
                    choices=["EfficientPhys", "VIRENet", "DeepPhys", "TSCAN"])
    ap.add_argument("--of", default="farneback",
                    choices=["farneback", "coarse2fine", "deep", "pca", "tvl1", "raft"])
    ap.add_argument("--tag", default="run")
    ap.add_argument("--folds", default="1,2,3,4,5,6")
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--epochs", type=int, default=None)
    args = ap.parse_args()

    with open(args.base, "r", encoding="utf-8") as f:
        base = yaml.safe_load(f)

    train_data = base["TRAIN"]["DATA"]
    if isinstance(train_data, list):
        template = train_data[0]
    else:
        template = train_data
    data_format = template["DATA_FORMAT"]
    preprocess = copy.deepcopy(template["PREPROCESS"])

    preprocess["DO_OPTICAL_FLOW"] = True
    preprocess["OF_METHOD"] = args.of

    os.makedirs(args.out_dir, exist_ok=True)
    folds = [int(x) for x in args.folds.split(",") if x.strip()]

    print(f"model={args.model}  of={args.of}  tag={args.tag}")
    print(f"preprocess: INFANT_REGION={preprocess.get('INFANT_REGION')} "
          f"CHUNK_LENGTH={preprocess.get('CHUNK_LENGTH')} "
          f"train_size={preprocess.get('DOWNSAMPLE_SIZE_BEFORE_TRAINING')}")
    print()

    written = []
    for fold in folds:
        train, valid, test = build_split(fold)
        cfg = copy.deepcopy(base)

        name = f"{args.model.lower()}_{args.of}_{args.tag}_fold{fold}"
        cfg["NAME"] = name

        cfg["CV"] = {"ENABLE_CV": False}

        cfg["DATA_PATH"] = {
            "AIR_125": args.air400_dir,   # unused; ENABLE_CV is False and no AIR_125 split exists
            "AIR_400": args.air400_dir,
            "COHFACE": "",
            "CACHE_DIR": args.cache_dir,
            "OUTPUT_DIR": args.output_dir,
            "BODY_DETECTOR_PATH": args.body_detector,
            "FACE_DETECTOR_PATH": args.face_detector,
        }

        cfg["TRAIN"]["DATA"] = [data_block(train, data_format, preprocess)]
        cfg["VALID"]["DATA"] = data_block(valid, data_format, preprocess)
        cfg["TEST"]["DATA"] = data_block(test, data_format, preprocess)

        cfg["MODEL"]["NAME"] = args.model
        if args.batch_size is not None:
            cfg["TRAIN"]["BATCH_SIZE"] = args.batch_size
        if args.epochs is not None:
            cfg["TRAIN"]["EPOCHES"] = args.epochs

        cfg.setdefault("INFERENCE", {})
        cfg["INFERENCE"]["EVALUATION_METHOD"] = "FFT"
        cfg["INFERENCE"]["USE_POST_PROCESS"] = True

        path = os.path.join(args.out_dir, name + ".yaml")
        with open(path, "w", encoding="utf-8") as f:
            yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True, default_flow_style=False)
        written.append(path)

        print(f"fold {fold}: train {len(train):2d} | valid {len(valid)} {valid} | test {len(test)} {test}")
        print(f"         -> {path}")

    print(f"\nwrote {len(written)} config(s)")
    print("\nSanity check before training:")
    print("  - every subject appears in exactly one test fold across folds 1-6")
    seen = sorted(s for f in FOLD_TEST.values() for s in f)
    print(f"  - union of test sets = {len(seen)} entries, {len(set(seen))} unique")
    if seen != ALL_SUBJECTS:
        print("  [WARN] test sets do not partition the 15 subjects")
    return 0


if __name__ == "__main__":
    sys.exit(main())
