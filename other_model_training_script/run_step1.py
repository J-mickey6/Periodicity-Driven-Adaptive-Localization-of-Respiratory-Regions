"""
run_step1.py  --  one-shot Step 1 runner for the AIR-400 DL comparison.

Stages (run in order, each aborts the run if it fails):
  1  LAYOUT     45 session folders -> 15 subject folders, via hard links
  2  CONFIGS    6 fold YAMLs with ENABLE_CV: False + SPLIT_TYPE: custom
  3  PREPROCESS main.py --preprocess for the pilot fold
  4  TRAIN      main.py for the pilot fold

Everything is logged in full to <report-dir>\\step1_full.log.
At the end a short human-readable digest is written to
<report-dir>\\step1_summary.txt -- that is the only file you need to paste back.

Defaults match the paths confirmed during Step 0, so the usual invocation is:

  python run_step1.py --body-detector "D:\\air400_dl\\weights\\yolov8n.pt"

Useful flags:
  --dry-run          plan only, touch nothing
  --stages 1,2       stop after configs (skip the heavy stages)
  --pilot-fold 1     which fold to pilot
  --batch-size 2     shrink if CUDA runs out of memory
  --epochs 2         short pilot; leave unset to keep the repo default (15)
  --model VIRENet    EfficientPhys | VIRENet | DeepPhys | TSCAN
  --of coarse2fine   only if pyflow has actually been built
"""

from __future__ import annotations

import argparse
import copy
import datetime
import os
import re
import shutil
import subprocess
import sys
import time

import pandas as pd
import yaml

# ----------------------------------------------------------------------------
# Step 0 audit constants -- the script refuses to proceed if reality disagrees
# ----------------------------------------------------------------------------
EXPECTED_TOTAL = 374
EXPECTED_PER_SUBJECT = {
    "S01": 20, "S02": 40, "S03": 10, "S04": 25, "S05": 45,
    "S06": 50, "S07": 25, "S08": 20, "S09": 20, "S10": 20,
    "S11": 21, "S12": 15, "S13": 19, "S14": 25, "S15": 19,
}
ALL_SUBJECTS = [f"S{i:02d}" for i in range(1, 16)]

# run_cv_experiment.py:42-49  (the DSP 6-fold definition)
FOLD_TEST = {
    1: ["S01", "S02"],
    2: ["S05", "S11"],
    3: ["S06", "S08"],
    4: ["S04", "S09", "S12"],
    5: ["S03", "S07", "S13"],
    6: ["S10", "S14", "S15"],
}
# Fixed validation subjects: disjoint from that fold's test set, never reused
# across folds, and biased toward low-clip-count subjects so the training set
# stays large. S05/S06/S14 (the three largest) always stay in training.
FOLD_VALID = {
    1: ["S03", "S12"],
    2: ["S13", "S15"],
    3: ["S09", "S10"],
    4: ["S01", "S08"],
    5: ["S04", "S11"],
    6: ["S02", "S07"],
}

SESSION_RE = re.compile(r"^(?P<subj>S\d+)_(?P<sess>\d+)$")


def _split_path(p: str) -> list[str]:
    """Split on both separators so the logic is identical on any host OS."""
    return [seg for seg in re.split(r"[\\/]+", str(p).strip()) if seg]


def parse_from_hdf5(hdf5_path: str):
    """
    Derive (subject, session_num, clip_num) from the real file location, not
    from the local_clip_id string.

    Rationale: 9 rows in S02_3 carry ids like 'S02_3/tmp_12.mp4' whose numeric
    part does not match the file on disk (verified: tmp_12..tmp_20 map to
    2.hdf5..10.hdf5). The path is the ground truth; the id string is not.
    Layout: ...\\{posture}\\{Sxx}_{sess}\\out\\{clip}.hdf5
    """
    parts = _split_path(hdf5_path)
    if len(parts) < 3:
        raise ValueError(f"path too short: {hdf5_path}")
    if parts[-2].lower() != "out":
        raise ValueError(f"hdf5 not inside an 'out' folder: {hdf5_path}")
    m = SESSION_RE.match(parts[-3])
    if not m:
        raise ValueError(f"unexpected session folder name: {parts[-3]}")
    stem = os.path.splitext(parts[-1])[0]
    if not stem.isdigit():
        raise ValueError(f"non-numeric hdf5 stem: {hdf5_path}")
    return m.group("subj"), int(m.group("sess")), int(stem)

LOG_LINES: list[str] = []
SUMMARY: list[str] = []


def log(msg: str = "") -> None:
    print(msg, flush=True)
    LOG_LINES.append(msg)


def summ(msg: str = "") -> None:
    SUMMARY.append(msg)


def fail(msg: str) -> None:
    log(f"[FAIL] {msg}")
    summ(f"FAILED: {msg}")


# ----------------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------------
def mp4_from_hdf5(hdf5_path: str) -> str:
    """...<sep>S02_3<sep>out<sep>2.hdf5  ->  ...<sep>S02_3<sep>2.mp4"""
    raw = str(hdf5_path)
    parts = _split_path(raw)
    if len(parts) < 3 or parts[-2].lower() != "out":
        raise ValueError(f"hdf5 not inside an 'out' folder: {hdf5_path}")
    sep = "\\" if "\\" in raw else "/"
    tail = sep.join(parts[-2:])              # e.g. 'out\\2.hdf5'
    idx = raw.rfind(tail)
    stem = os.path.splitext(parts[-1])[0]
    return raw[:idx] + stem + ".mp4"


def same_volume(a: str, b: str) -> bool:
    return (os.path.splitdrive(os.path.abspath(a))[0].upper()
            == os.path.splitdrive(os.path.abspath(b))[0].upper())


def dir_size_gb(path: str) -> float:
    total = 0
    for root, _dirs, files in os.walk(path):
        for f in files:
            try:
                total += os.path.getsize(os.path.join(root, f))
            except OSError:
                pass
    return total / (1024 ** 3)


def free_space_gb(path: str) -> float:
    drive = os.path.splitdrive(os.path.abspath(path))[0] + os.sep
    return shutil.disk_usage(drive).free / (1024 ** 3)


def build_split(fold: int):
    test = sorted(FOLD_TEST[fold])
    valid = sorted(FOLD_VALID[fold])
    if set(test) & set(valid):
        raise ValueError(f"fold {fold}: test/valid overlap")
    train = sorted(set(ALL_SUBJECTS) - set(test) - set(valid))
    if len(train) + len(valid) + len(test) != 15:
        raise ValueError(f"fold {fold}: subjects do not sum to 15")
    return train, valid, test


def run_cmd(cmd: list[str], cwd: str, tag: str, tail: int = 40):
    """Run in the foreground, stream output, return (returncode, seconds, tail_lines)."""
    log(f"\n[{tag}] $ {' '.join(cmd)}")
    log(f"[{tag}] cwd = {cwd}")
    t0 = time.time()
    lines: list[str] = []
    proc = subprocess.Popen(cmd, cwd=cwd, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, text=True,
                            encoding="utf-8", errors="replace", bufsize=1)
    assert proc.stdout is not None
    for line in proc.stdout:
        line = line.rstrip("\n")
        print(line, flush=True)
        lines.append(line)
    proc.wait()
    dt = time.time() - t0
    LOG_LINES.extend(lines)
    log(f"[{tag}] exit={proc.returncode} elapsed={dt/60:.1f} min")
    return proc.returncode, dt, lines[-tail:]


# ----------------------------------------------------------------------------
# stage 1: layout
# ----------------------------------------------------------------------------
def stage_layout(args) -> pd.DataFrame | None:
    log("\n" + "=" * 70)
    log("STAGE 1  LAYOUT")
    log("=" * 70)

    cmap = pd.read_csv(args.map)
    meta = pd.read_csv(args.meta)
    log(f"clip_path_map rows = {len(cmap)}   metadata rows = {len(meta)}")

    for col in ("local_clip_id", "hdf5_abs_path", "official_subject_id"):
        if col not in cmap.columns:
            fail(f"clip_path_map.csv missing column {col}")
            return None
    if "posture_measured" not in meta.columns:
        fail("metadata csv missing posture_measured")
        return None

    keep = [c for c in ("local_clip_id", "posture_measured", "fps_measured",
                        "official_gt_rr") if c in meta.columns]
    df = cmap.merge(meta[keep], on="local_clip_id", how="inner")
    log(f"merged rows = {len(df)}")
    if len(df) != EXPECTED_TOTAL:
        fail(f"expected {EXPECTED_TOTAL} rows after merge, got {len(df)}")
        return None

    parsed = df["hdf5_abs_path"].map(parse_from_hdf5)
    df["local_subject"] = [p[0] for p in parsed]
    df["session_num"] = [p[1] for p in parsed]
    df["clip_num"] = [p[2] for p in parsed]

    # local_clip_id is informational only; flag rows where its numeric part
    # disagrees with the file actually on disk, but do not fail on them.
    id_re = re.compile(r"^(S\d+)_(\d+)/(?:tmp_)?(\d+)\.mp4$")
    mismatched = []
    for _, r in df.iterrows():
        m = id_re.match(str(r["local_clip_id"]).replace("\\", "/").strip())
        if m is None:
            mismatched.append((r["local_clip_id"], "unparsable id"))
        elif (m.group(1), int(m.group(2)), int(m.group(3))) != (
                r["local_subject"], r["session_num"], r["clip_num"]):
            mismatched.append((r["local_clip_id"],
                               f"disk={r['local_subject']}_{r['session_num']}/{r['clip_num']}"))
    if mismatched:
        log(f"note: {len(mismatched)} local_clip_id value(s) disagree with the file on disk; "
            f"the disk path wins. Examples:")
        for cid, why in mismatched[:10]:
            log(f"    {cid}  ({why})")
        summ(f"                {len(mismatched)} clip ids disagree with disk paths "
             f"(disk path used; see clip_remap.csv)")

    dup = df["hdf5_abs_path"].duplicated(keep=False)
    if dup.any():
        fail(f"{int(dup.sum())} rows share an hdf5 path; mapping is not 1:1")
        for p in sorted(set(df.loc[dup, "hdf5_abs_path"]))[:10]:
            log(f"    {p}")
        return None

    counts = df["local_subject"].value_counts().to_dict()
    bad = {k: (counts.get(k, 0), v) for k, v in EXPECTED_PER_SUBJECT.items()
           if counts.get(k, 0) != v}
    if bad:
        fail("per-subject clip counts disagree with the Step 0 audit")
        for k, (got, exp) in sorted(bad.items()):
            log(f"    {k}: got {got}, expected {exp}")
        return None
    log("per-subject clip counts match the audit for all 15 subjects")

    df = df.sort_values(["local_subject", "session_num", "clip_num"]).reset_index(drop=True)
    df["new_clip_num"] = df.groupby("local_subject").cumcount() + 1
    df["new_subject"] = df["local_subject"]
    df["new_index"] = df["new_subject"] + "_" + df["new_clip_num"].map("{:03d}".format)

    src_mp4, src_h5, missing = [], [], []
    for h5 in df["hdf5_abs_path"].astype(str):
        mp4 = mp4_from_hdf5(h5)
        src_h5.append(h5)
        src_mp4.append(mp4)
        for p in (h5, mp4):
            if not os.path.exists(p):
                missing.append(p)
    df["src_mp4"] = src_mp4
    df["src_hdf5"] = src_h5
    if missing:
        fail(f"{len(missing)} source files not found (first: {missing[0]})")
        return None
    log(f"all {2 * len(df)} source files exist")

    dest = os.path.abspath(args.dest)
    df["dst_mp4"] = [os.path.join(dest, s, f"{n}.mp4")
                     for s, n in zip(df["new_subject"], df["new_clip_num"])]
    df["dst_hdf5"] = [os.path.join(dest, s, "out", f"{n}.hdf5")
                      for s, n in zip(df["new_subject"], df["new_clip_num"])]

    if not same_volume(df["src_hdf5"].iloc[0], dest):
        fail("source and dest are on different volumes; hard links impossible")
        return None
    log("source and dest share a volume, hard links are usable")

    if args.dry_run:
        log("\n[dry-run] first 3 planned links:")
        for _, r in df.head(3).iterrows():
            log(f"  {r['local_clip_id']} -> {r['new_index']}")
            log(f"      {r['src_mp4']}")
            log(f"   -> {r['dst_mp4']}")
        log(f"[dry-run] would create {2 * len(df)} hard links under {dest}")
        summ(f"stage1 LAYOUT   dry-run OK, {2 * len(df)} links planned")
        return df

    if os.path.isdir(dest) and os.listdir(dest) and not args.force:
        fail(f"dest exists and is not empty: {dest}  (use --force)")
        return None

    for s in sorted(df["new_subject"].unique()):
        os.makedirs(os.path.join(dest, s, "out"), exist_ok=True)

    made = 0
    for _, r in df.iterrows():
        for src, dst in ((r["src_mp4"], r["dst_mp4"]), (r["src_hdf5"], r["dst_hdf5"])):
            if os.path.exists(dst):
                os.remove(dst)
            os.link(src, dst)
            made += 1
    log(f"created {made} hard links")

    problems = 0
    for _, r in df.iterrows():
        for src, dst in ((r["src_mp4"], r["dst_mp4"]), (r["src_hdf5"], r["dst_hdf5"])):
            if not os.path.exists(dst) or os.path.getsize(src) != os.path.getsize(dst):
                log(f"[VERIFY-FAIL] {dst}")
                problems += 1
    for s, exp in EXPECTED_PER_SUBJECT.items():
        n_mp4 = len([f for f in os.listdir(os.path.join(dest, s)) if f.endswith(".mp4")])
        n_h5 = len(os.listdir(os.path.join(dest, s, "out")))
        if n_mp4 != exp or n_h5 != exp:
            log(f"[VERIFY-FAIL] {s}: mp4={n_mp4} hdf5={n_h5} expected={exp}")
            problems += 1
    subject_dirs = [d for d in os.listdir(dest)
                    if d.startswith("S") and os.path.isdir(os.path.join(dest, d))]
    log(f"folders visible to glob('S*') = {len(subject_dirs)}")
    if len(subject_dirs) != 15:
        log("[VERIFY-FAIL] expected exactly 15 subject folders")
        problems += 1

    remap = os.path.join(args.report_dir, "clip_remap.csv")
    cols = ["new_subject", "new_clip_num", "new_index", "local_clip_id",
            "local_subject", "session_num", "clip_num", "official_subject_id",
            "posture_measured"]
    cols += [c for c in ("fps_measured", "official_gt_rr") if c in df.columns]
    cols += ["src_mp4", "src_hdf5", "dst_mp4", "dst_hdf5"]
    df[cols].to_csv(remap, index=False, encoding="utf-8-sig")
    log(f"wrote {remap}")

    log("\nposture x subject:")
    log(str(pd.crosstab(df["new_subject"], df["posture_measured"], margins=True)))

    if problems:
        fail(f"layout verification found {problems} problem(s)")
        return None
    summ(f"stage1 LAYOUT   OK   {made} hard links, 15 subject folders, {len(df)} clips")
    summ(f"                remap csv: {remap}")
    return df


# ----------------------------------------------------------------------------
# stage 2: configs
# ----------------------------------------------------------------------------
def stage_configs(args) -> list[str] | None:
    log("\n" + "=" * 70)
    log("STAGE 2  CONFIGS")
    log("=" * 70)

    if not os.path.exists(args.base):
        fail(f"base config not found: {args.base}")
        return None
    with open(args.base, "r", encoding="utf-8") as f:
        base = yaml.safe_load(f)

    td = base["TRAIN"]["DATA"]
    template = td[0] if isinstance(td, list) else td
    data_format = template["DATA_FORMAT"]
    preprocess = copy.deepcopy(template["PREPROCESS"])
    preprocess["DO_OPTICAL_FLOW"] = True
    preprocess["OF_METHOD"] = args.of

    log(f"template: INFANT_REGION={preprocess.get('INFANT_REGION')} "
        f"CHUNK_LENGTH={preprocess.get('CHUNK_LENGTH')} "
        f"train_size={preprocess.get('DOWNSAMPLE_SIZE_BEFORE_TRAINING')} "
        f"OF_METHOD={preprocess.get('OF_METHOD')}")

    def block(subjects):
        return {"DATASET": "AIR_400", "SPLIT_TYPE": "custom",
                "SPLIT_SUBJECTS": list(subjects), "SPLIT_RATIO": 0.0,
                "DATA_FORMAT": data_format, "PREPROCESS": copy.deepcopy(preprocess)}

    os.makedirs(args.config_out, exist_ok=True)
    written = []
    for fold in range(1, 7):
        train, valid, test = build_split(fold)
        cfg = copy.deepcopy(base)
        name = f"{args.model.lower()}_{args.of}_{args.tag}_fold{fold}"
        cfg["NAME"] = name
        cfg["CV"] = {"ENABLE_CV": False}
        cfg["DATA_PATH"] = {
            "AIR_125": args.dest,
            "AIR_400": args.dest,
            "COHFACE": "",
            "CACHE_DIR": args.cache_dir,
            "OUTPUT_DIR": args.output_dir,
            "BODY_DETECTOR_PATH": args.body_detector,
            "FACE_DETECTOR_PATH": args.face_detector,
        }
        cfg["TRAIN"]["DATA"] = [block(train)]
        cfg["VALID"]["DATA"] = block(valid)
        cfg["TEST"]["DATA"] = block(test)
        cfg["MODEL"]["NAME"] = args.model
        if args.batch_size is not None:
            cfg["TRAIN"]["BATCH_SIZE"] = args.batch_size
        if args.epochs is not None:
            cfg["TRAIN"]["EPOCHES"] = args.epochs
        cfg.setdefault("INFERENCE", {})
        cfg["INFERENCE"]["EVALUATION_METHOD"] = "FFT"
        cfg["INFERENCE"]["USE_POST_PROCESS"] = True

        path = os.path.join(args.config_out, name + ".yaml")
        if not args.dry_run:
            with open(path, "w", encoding="utf-8") as f:
                yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True,
                               default_flow_style=False)
        written.append(path)
        n = EXPECTED_PER_SUBJECT
        log(f"fold {fold}: test {test} ({sum(n[s] for s in test)} clips) | "
            f"val {valid} ({sum(n[s] for s in valid)}) | "
            f"train {len(train)} subj ({sum(n[s] for s in train)})")

    union = sorted(s for f in FOLD_TEST.values() for s in f)
    ok = union == ALL_SUBJECTS
    log(f"test sets partition the 15 subjects: {ok}")
    if not ok:
        fail("test folds do not partition the subjects")
        return None

    summ(f"stage2 CONFIGS  OK   6 yaml files in {args.config_out}")
    summ(f"                model={args.model} of={args.of} "
         f"batch={args.batch_size or 'base'} epochs={args.epochs or 'base'}")
    return written


# ----------------------------------------------------------------------------
# stages 3 and 4: pilot
# ----------------------------------------------------------------------------
def stage_preprocess(args, cfg_path: str) -> bool:
    log("\n" + "=" * 70)
    log(f"STAGE 3  PREPROCESS  (fold {args.pilot_fold})")
    log("=" * 70)
    before = free_space_gb(args.cache_dir) if os.path.isdir(args.cache_dir) else free_space_gb(
        os.path.splitdrive(os.path.abspath(args.cache_dir))[0] + os.sep)
    rc, dt, tail = run_cmd([sys.executable, "main.py", "--config", cfg_path, "--preprocess"],
                           cwd=args.repo, tag="preprocess")
    cache_gb = dir_size_gb(args.cache_dir) if os.path.isdir(args.cache_dir) else 0.0
    n_npy = 0
    if os.path.isdir(args.cache_dir):
        for _r, _d, files in os.walk(args.cache_dir):
            n_npy += sum(1 for f in files if f.endswith(".npy"))
    log(f"cache size = {cache_gb:.2f} GB   npy files = {n_npy}   "
        f"free before = {before:.1f} GB   free now = {free_space_gb(args.cache_dir):.1f} GB")
    summ(f"stage3 PREPROC  {'OK' if rc == 0 else 'FAILED rc=' + str(rc)}   "
         f"{dt/60:.1f} min, cache {cache_gb:.2f} GB, {n_npy} npy files")
    if rc != 0:
        summ("  last lines:")
        for line in tail[-12:]:
            summ("    " + line)
    return rc == 0


def stage_train(args, cfg_path: str) -> bool:
    log("\n" + "=" * 70)
    log(f"STAGE 4  TRAIN  (fold {args.pilot_fold})")
    log("=" * 70)
    rc, dt, tail = run_cmd([sys.executable, "main.py", "--config", cfg_path],
                           cwd=args.repo, tag="train", tail=60)

    metrics = [ln for ln in tail if re.search(r"(MAE|RMSE|Pearson|Corr)", ln, re.I)]
    oom = any("out of memory" in ln.lower() for ln in tail)
    summ(f"stage4 TRAIN    {'OK' if rc == 0 else 'FAILED rc=' + str(rc)}   {dt/60:.1f} min")
    if oom:
        summ("  CUDA OUT OF MEMORY -- rerun with --batch-size 2")
    if metrics:
        summ("  test metrics:")
        for ln in metrics[-8:]:
            summ("    " + ln.strip())
    elif rc != 0:
        summ("  last lines:")
        for ln in tail[-15:]:
            summ("    " + ln)
    return rc == 0


# ----------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default=r"D:\air400_dl")
    ap.add_argument("--map",
                    default=r"C:\Users\USER\Desktop\Baby_Respiration_Archive\clip_path_map.csv")
    ap.add_argument("--meta",
                    default=r"C:\Users\USER\Desktop\re_baby_374_metadata_master.csv")
    ap.add_argument("--dest", default=r"D:\hdd\RE_baby_air400")
    ap.add_argument("--cache-dir", default=r"D:\air400_cache")
    ap.add_argument("--output-dir", default=r"D:\air400_runs")
    ap.add_argument("--config-out", default=r"D:\air400_dl\configs\ours")
    ap.add_argument("--report-dir", default=r"D:\air400_dl")
    ap.add_argument("--base", default="")
    ap.add_argument("--body-detector", required=True)
    ap.add_argument("--face-detector", default="")
    ap.add_argument("--model", default="EfficientPhys",
                    choices=["EfficientPhys", "VIRENet", "DeepPhys", "TSCAN"])
    ap.add_argument("--of", default="farneback",
                    choices=["farneback", "coarse2fine", "deep", "pca", "tvl1", "raft"])
    ap.add_argument("--tag", default="pilot")
    ap.add_argument("--pilot-fold", type=int, default=1)
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--stages", default="1,2,3,4")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    if not args.base:
        args.base = os.path.join(
            args.repo, "configs", "body_roi",
            f"{args.model.lower()}_6_fold_cv_air400_with_roi_{args.of}_of_body.yaml")

    stages = {int(x) for x in args.stages.split(",") if x.strip()}
    started = datetime.datetime.now()
    os.makedirs(args.report_dir, exist_ok=True)

    log("=" * 70)
    log(f"run_step1.py   started {started:%Y-%m-%d %H:%M:%S}")
    log("=" * 70)
    log(f"repo         {args.repo}")
    log(f"dest         {args.dest}")
    log(f"cache        {args.cache_dir}  (free {free_space_gb(args.cache_dir):.0f} GB)")
    log(f"base config  {args.base}")
    log(f"stages       {sorted(stages)}   dry_run={args.dry_run}")
    summ(f"run_step1.py  {started:%Y-%m-%d %H:%M}  "
         f"model={args.model} of={args.of} fold={args.pilot_fold} dry_run={args.dry_run}")
    summ("")

    ok = True
    try:
        if 1 in stages:
            if stage_layout(args) is None:
                ok = False
        if ok and 2 in stages:
            if stage_configs(args) is None:
                ok = False
        cfg_path = os.path.join(
            args.config_out,
            f"{args.model.lower()}_{args.of}_{args.tag}_fold{args.pilot_fold}.yaml")
        if ok and not args.dry_run and 3 in stages:
            ok = stage_preprocess(args, cfg_path)
        if ok and not args.dry_run and 4 in stages:
            ok = stage_train(args, cfg_path)
    except Exception as exc:  # noqa: BLE001
        import traceback
        tb = traceback.format_exc()
        log(tb)
        summ(f"EXCEPTION: {exc}")
        summ(tb.strip().splitlines()[-1])
        ok = False

    elapsed = (datetime.datetime.now() - started).total_seconds() / 60
    summ("")
    summ(f"total elapsed {elapsed:.1f} min   result: {'OK' if ok else 'FAILED'}")

    full = os.path.join(args.report_dir, "step1_full.log")
    with open(full, "w", encoding="utf-8") as f:
        f.write("\n".join(LOG_LINES))
    digest = os.path.join(args.report_dir, "step1_summary.txt")
    with open(digest, "w", encoding="utf-8") as f:
        f.write("\n".join(SUMMARY))

    print("\n" + "=" * 70)
    print("SUMMARY  (paste this back)")
    print("=" * 70)
    print("\n".join(SUMMARY))
    print("=" * 70)
    print(f"full log: {full}")
    print(f"summary : {digest}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
