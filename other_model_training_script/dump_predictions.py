"""
dump_predictions.py  --  extract per-clip predicted / ground-truth respiration
rates from already-trained checkpoints. No training happens.

Why this exists: the repo computes test metrics but never writes the per-clip
values to disk, so there is no way to do clip-level comparisons, per-posture
breakdowns, or a trustworthy Pearson (the repo's own Pearson is computed over
the 2-3 test *subjects* in a fold, which is meaningless at that sample size).

How it works: main.py is run unchanged, with two runtime patches.
  1. Trainer.train  -> no-op, so the existing best checkpoint is used as-is.
  2. Trainer._calculate_metrics -> intercepted; it receives exactly the
     per-clip post-processed rates that base_trainer.test() built
     (base_trainer.py:468-501), writes them out, then calls the original.
Nothing in the repo is edited on disk.

Usage (PowerShell, one line):
  python dump_predictions.py --tag b2e15 --folds 1,2,3,4,5,6

Output:
  D:\\air400_dl\\preds\\pred_<tag>_fold<N>.csv   per fold
  D:\\air400_dl\\preds\\pred_<tag>_all.csv       merged, with posture + fold
"""

from __future__ import annotations

import argparse
import glob
import importlib
import os
import sys
import traceback

import pandas as pd

FOLD_TEST = {
    1: ["S01", "S02"],
    2: ["S05", "S11"],
    3: ["S06", "S08"],
    4: ["S04", "S09", "S12"],
    5: ["S03", "S07", "S13"],
    6: ["S10", "S14", "S15"],
}


def _to_float(v):
    """final_predictions[clip] is {0: tensor([[rr]])}; unwrap to a python float."""
    if v is None:
        return float("nan")
    if isinstance(v, dict):
        if not v:
            return float("nan")
        v = next(iter(v.values()))
    detach = getattr(v, "detach", None)
    if callable(detach):
        try:
            v = detach().cpu().numpy()
        except Exception:
            pass
    reshape = getattr(v, "reshape", None)
    if callable(reshape):
        try:
            flat = reshape(-1)
            if len(flat) == 0:
                return float("nan")
            return float(flat[0])
        except Exception:
            pass
    try:
        return float(v)
    except Exception:
        return float("nan")


_CAPTURED: dict = {}
_PATCHED: set = set()


def _install_patches() -> bool:
    """Patch each defining class exactly once; forward args untouched."""
    bt = importlib.import_module("trainers.base_trainer")

    metric_cls = [obj for obj in vars(bt).values()
                  if isinstance(obj, type) and "_calculate_metrics" in obj.__dict__]
    train_cls = [obj for obj in vars(bt).values()
                 if isinstance(obj, type) and "train" in obj.__dict__]
    if not metric_cls:
        print("[FAIL] no class defines _calculate_metrics in trainers.base_trainer")
        return False

    for cls in metric_cls:
        key = (cls.__module__, cls.__name__, "metrics")
        if key in _PATCHED:
            continue
        orig = cls.__dict__["_calculate_metrics"]

        def patched_metrics(self, *args, _orig=orig, **kwargs):
            # base_trainer.test() passes the per-clip post-processed rates here
            preds = args[0] if len(args) > 0 else kwargs.get("predictions")
            labels = args[1] if len(args) > 1 else kwargs.get("labels")
            if isinstance(preds, dict):
                for k in preds:
                    g = labels.get(k) if isinstance(labels, dict) else None
                    _CAPTURED[str(k)] = (_to_float(preds[k]), _to_float(g))
                print(f"[capture] {len(_CAPTURED)} clips")
            return _orig(self, *args, **kwargs)

        cls._calculate_metrics = patched_metrics
        _PATCHED.add(key)
        print(f"[patch] {cls.__name__}._calculate_metrics")

    for cls in train_cls:
        key = (cls.__module__, cls.__name__, "train")
        if key in _PATCHED:
            continue

        def patched_train(self, *a, **kw):
            print("[patch] train() skipped; using existing best checkpoint")
            return None

        cls.train = patched_train
        _PATCHED.add(key)
        print(f"[patch] {cls.__name__}.train -> no-op")

    return True


def run_fold(repo: str, cfg_path: str, out_csv: str) -> bool:
    print(f"\n{'='*70}\nconfig: {cfg_path}\n{'='*70}", flush=True)
    if not os.path.exists(cfg_path):
        print(f"[skip] config not found: {cfg_path}")
        return False

    if repo not in sys.path:
        sys.path.insert(0, repo)
    os.chdir(repo)

    if not _install_patches():
        return False

    _CAPTURED.clear()

    main_mod = importlib.import_module("main")
    old_argv = sys.argv[:]
    sys.argv = ["main.py", "--config", cfg_path]
    try:
        main_mod.main()
    except SystemExit:
        pass
    except Exception:
        traceback.print_exc()
        print("[warn] main() raised after capture; continuing if data was collected")
    finally:
        sys.argv = old_argv

    if not _CAPTURED:
        print("[FAIL] nothing captured -- test() may not have run")
        return False

    rows = [{"new_index": k, "pred_rr": p, "gt_rr": g}
            for k, (p, g) in sorted(_CAPTURED.items())]
    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    df.to_csv(out_csv, index=False)
    print(f"[write] {out_csv}  ({len(df)} clips)")
    print(df.head(5).to_string(index=False))
    return True


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default=r"D:\air400_dl")
    ap.add_argument("--config-dir", default=r"D:\air400_dl\configs\ours")
    ap.add_argument("--out-dir", default=r"D:\air400_dl\preds")
    ap.add_argument("--remap", default=r"D:\air400_dl\clip_remap.csv")
    ap.add_argument("--model", default="efficientphys")
    ap.add_argument("--of", default="farneback")
    ap.add_argument("--tag", default="b2e15")
    ap.add_argument("--folds", default="1,2,3,4,5,6")
    args = ap.parse_args()

    folds = [int(x) for x in args.folds.split(",") if x.strip()]
    done = []

    for f in folds:
        name = f"{args.model}_{args.of}_{args.tag}_fold{f}"
        cfg = os.path.join(args.config_dir, name + ".yaml")
        out = os.path.join(args.out_dir, f"pred_{args.tag}_fold{f}.csv")
        # each fold runs in a fresh interpreter state as far as the patches go;
        # re-importing is fine because the patch is idempotent per class
        ok = run_fold(args.repo, cfg, out)
        if ok:
            done.append((f, out))

    if not done:
        print("\nno folds produced output")
        return 1

    # merge + annotate
    parts = []
    for f, path in done:
        d = pd.read_csv(path)
        d["fold"] = f
        parts.append(d)
    allp = pd.concat(parts, ignore_index=True)

    if os.path.exists(args.remap):
        rm = pd.read_csv(args.remap)
        keep = [c for c in ("new_index", "new_subject", "posture_measured",
                            "official_gt_rr", "fps_measured", "local_clip_id")
                if c in rm.columns]
        allp = allp.merge(rm[keep], on="new_index", how="left")

    merged = os.path.join(args.out_dir, f"pred_{args.tag}_all.csv")
    allp.to_csv(merged, index=False)
    print(f"\n[write] {merged}  ({len(allp)} rows)")

    # quick clip-level summary so the numbers are visible immediately
    import numpy as np
    print("\nclip-level metrics (pred_rr vs gt_rr from the repo's own pipeline)")
    print(f"{'fold':<6}{'n':>5}{'MAE':>9}{'RMSE':>9}{'Pearson':>10}")
    for f in sorted(allp["fold"].unique()):
        d = allp[allp["fold"] == f].dropna(subset=["pred_rr", "gt_rr"])
        e = d["pred_rr"] - d["gt_rr"]
        r = float(np.corrcoef(d["pred_rr"], d["gt_rr"])[0, 1]) if len(d) > 2 else float("nan")
        print(f"{f:<6}{len(d):>5}{e.abs().mean():>9.3f}"
              f"{float(np.sqrt((e**2).mean())):>9.3f}{r:>10.3f}")
    d = allp.dropna(subset=["pred_rr", "gt_rr"])
    e = d["pred_rr"] - d["gt_rr"]
    r = float(np.corrcoef(d["pred_rr"], d["gt_rr"])[0, 1]) if len(d) > 2 else float("nan")
    print(f"{'ALL':<6}{len(d):>5}{e.abs().mean():>9.3f}"
          f"{float(np.sqrt((e**2).mean())):>9.3f}{r:>10.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
