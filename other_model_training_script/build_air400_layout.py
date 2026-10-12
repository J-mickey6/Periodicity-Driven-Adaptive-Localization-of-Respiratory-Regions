"""
build_air400_layout.py

45 session folders (D:\\hdd\\RE_baby\\{posture}\\{Sxx}_{sess}\\) ->
15 subject folders (dest\\{Sxx}\\) expected by AIR400Dataset.get_raw_data().

- Uses hard links. Source files are never modified, moved, or deleted.
- Clips are renumbered 1..N within each subject, in a deterministic order.
- Writes clip_remap.csv so every new clip can be traced back to its origin
  (needed later for the per-posture error breakdown).

Verified assumptions (Step 0):
  air400_dataset.py:42  glob("S*") over data_path            -> folder name must start with 'S'
  air400_dataset.py:52  subject = os.path.basename(dir)      -> folder name IS the subject id
  air400_dataset.py:60  mp4 files directly inside subject dir
  air400_dataset.py:67  hdf5 at subject_dir/out/{stem}.hdf5
  air400_dataset.py:82  index = f"{subject}_{int(recording_id):03d}"  -> stem must be int-parsable

Usage (PowerShell, one line):
  python build_air400_layout.py `
      --map "C:\\Users\\USER\\Desktop\\Baby_Respiration_Archive\\clip_path_map.csv" `
      --meta "C:\\Users\\USER\\Desktop\\re_baby_374_metadata_master.csv" `
      --dest "D:\\hdd\\RE_baby_air400"

Add --dry-run first to see what would happen without touching the disk.
"""

from __future__ import annotations

import argparse
import os
import re
import sys

import pandas as pd

EXPECTED_TOTAL = 374
EXPECTED_PER_SUBJECT = {
    "S01": 20, "S02": 40, "S03": 10, "S04": 25, "S05": 45,
    "S06": 50, "S07": 25, "S08": 20, "S09": 20, "S10": 20,
    "S11": 21, "S12": 15, "S13": 19, "S14": 25, "S15": 19,
}

CLIP_RE = re.compile(r"^(?P<subj>S\d+)_(?P<sess>\d+)/(?P<clip>\d+)\.mp4$")


def parse_clip_id(local_clip_id: str):
    """'S01_2/3.mp4' -> ('S01', 2, 3). Raises on any unexpected format."""
    m = CLIP_RE.match(str(local_clip_id).replace("\\", "/").strip())
    if not m:
        raise ValueError(f"unparsable local_clip_id: {local_clip_id!r}")
    return m.group("subj"), int(m.group("sess")), int(m.group("clip"))


def mp4_from_hdf5(hdf5_path: str) -> str:
    """...\\S01_1\\out\\3.hdf5 -> ...\\S01_1\\3.mp4"""
    out_dir = os.path.dirname(hdf5_path)
    if os.path.basename(out_dir).lower() != "out":
        raise ValueError(f"hdf5 not inside an 'out' folder: {hdf5_path}")
    session_dir = os.path.dirname(out_dir)
    stem = os.path.splitext(os.path.basename(hdf5_path))[0]
    return os.path.join(session_dir, stem + ".mp4")


def same_volume(a: str, b: str) -> bool:
    da = os.path.splitdrive(os.path.abspath(a))[0].upper()
    db = os.path.splitdrive(os.path.abspath(b))[0].upper()
    return da == db


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--map", required=True, help="clip_path_map.csv")
    ap.add_argument("--meta", required=True, help="re_baby_374_metadata_master.csv")
    ap.add_argument("--dest", required=True, help="output root, e.g. D:\\hdd\\RE_baby_air400")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--force", action="store_true", help="allow a non-empty dest")
    args = ap.parse_args()

    # ---------- load ----------
    cmap = pd.read_csv(args.map)
    meta = pd.read_csv(args.meta)
    print(f"[load] clip_path_map rows = {len(cmap)}")
    print(f"[load] metadata     rows = {len(meta)}")

    for col in ("local_clip_id", "hdf5_abs_path", "official_subject_id"):
        if col not in cmap.columns:
            print(f"[FAIL] clip_path_map.csv missing column: {col}")
            return 1
    for col in ("local_clip_id", "posture_measured"):
        if col not in meta.columns:
            print(f"[FAIL] metadata csv missing column: {col}")
            return 1

    meta_cols = [c for c in
                 ("local_clip_id", "posture_measured", "fps_measured", "official_gt_rr")
                 if c in meta.columns]
    df = cmap.merge(meta[meta_cols], on="local_clip_id", how="inner")
    print(f"[merge] rows = {len(df)}")
    if len(df) != EXPECTED_TOTAL:
        print(f"[FAIL] expected {EXPECTED_TOTAL} rows after merge, got {len(df)}")
        return 1

    # ---------- parse + deterministic ordering ----------
    parsed = df["local_clip_id"].map(parse_clip_id)
    df["local_subject"] = [p[0] for p in parsed]
    df["session_num"] = [p[1] for p in parsed]
    df["clip_num"] = [p[2] for p in parsed]

    counts = df["local_subject"].value_counts().to_dict()
    bad = {k: (counts.get(k, 0), v) for k, v in EXPECTED_PER_SUBJECT.items()
           if counts.get(k, 0) != v}
    if bad:
        print("[FAIL] per-subject clip counts do not match the Step 0 audit:")
        for k, (got, exp) in sorted(bad.items()):
            print(f"       {k}: got {got}, expected {exp}")
        return 1
    print(f"[check] per-subject counts match audit for all {len(EXPECTED_PER_SUBJECT)} subjects")

    df = df.sort_values(["local_subject", "session_num", "clip_num"]).reset_index(drop=True)
    df["new_clip_num"] = df.groupby("local_subject").cumcount() + 1
    df["new_subject"] = df["local_subject"]
    df["new_index"] = df["new_subject"] + "_" + df["new_clip_num"].map("{:03d}".format)

    # resolve source mp4 paths
    src_mp4, src_h5, missing = [], [], []
    for _, r in df.iterrows():
        h5 = str(r["hdf5_abs_path"])
        mp4 = mp4_from_hdf5(h5)
        src_h5.append(h5)
        src_mp4.append(mp4)
        if not os.path.exists(h5):
            missing.append(h5)
        if not os.path.exists(mp4):
            missing.append(mp4)
    df["src_hdf5"] = src_h5
    df["src_mp4"] = src_mp4
    if missing:
        print(f"[FAIL] {len(missing)} source files not found. First 10:")
        for p in missing[:10]:
            print("       " + p)
        return 1
    print("[check] all 748 source files (374 mp4 + 374 hdf5) exist")

    dest_root = os.path.abspath(args.dest)
    df["dst_mp4"] = [os.path.join(dest_root, s, f"{n}.mp4")
                     for s, n in zip(df["new_subject"], df["new_clip_num"])]
    df["dst_hdf5"] = [os.path.join(dest_root, s, "out", f"{n}.hdf5")
                      for s, n in zip(df["new_subject"], df["new_clip_num"])]

    if not same_volume(df["src_hdf5"].iloc[0], dest_root):
        print("[FAIL] source and dest are on different volumes; hard links are impossible.")
        print("       Pick a --dest on the same drive as D:\\hdd\\RE_baby.")
        return 1
    print("[check] source and dest are on the same volume (hard links OK)")

    if os.path.isdir(dest_root) and os.listdir(dest_root) and not args.force:
        print(f"[FAIL] dest already exists and is not empty: {dest_root}")
        print("       Remove it manually or pass --force.")
        return 1

    remap_path = os.path.join(os.path.dirname(os.path.abspath(args.map)), "clip_remap.csv")

    if args.dry_run:
        print("\n[dry-run] first 5 planned links:")
        for _, r in df.head(5).iterrows():
            print(f"  {r['local_clip_id']:>16}  ->  {r['new_index']}")
            print(f"      {r['src_mp4']}\n        -> {r['dst_mp4']}")
        print(f"\n[dry-run] would create {len(df)*2} hard links under {dest_root}")
        print(f"[dry-run] would write {remap_path}")
        return 0

    # ---------- link ----------
    for s in sorted(df["new_subject"].unique()):
        os.makedirs(os.path.join(dest_root, s, "out"), exist_ok=True)

    made = 0
    for _, r in df.iterrows():
        for src, dst in ((r["src_mp4"], r["dst_mp4"]), (r["src_hdf5"], r["dst_hdf5"])):
            if os.path.exists(dst):
                os.remove(dst)
            os.link(src, dst)
            made += 1
    print(f"[link] created {made} hard links")

    # ---------- verify ----------
    problems = 0
    for _, r in df.iterrows():
        for src, dst in ((r["src_mp4"], r["dst_mp4"]), (r["src_hdf5"], r["dst_hdf5"])):
            if not os.path.exists(dst):
                print(f"[VERIFY-FAIL] missing {dst}")
                problems += 1
                continue
            if os.path.getsize(src) != os.path.getsize(dst):
                print(f"[VERIFY-FAIL] size mismatch {dst}")
                problems += 1

    for s, exp in EXPECTED_PER_SUBJECT.items():
        n_mp4 = len([f for f in os.listdir(os.path.join(dest_root, s)) if f.endswith(".mp4")])
        n_h5 = len(os.listdir(os.path.join(dest_root, s, "out")))
        if n_mp4 != exp or n_h5 != exp:
            print(f"[VERIFY-FAIL] {s}: mp4={n_mp4} hdf5={n_h5} expected={exp}")
            problems += 1

    subject_dirs = [d for d in os.listdir(dest_root)
                    if d.startswith("S") and os.path.isdir(os.path.join(dest_root, d))]
    print(f"[verify] subject folders visible to glob('S*') = {len(subject_dirs)}")
    if len(subject_dirs) != 15:
        print("[VERIFY-FAIL] expected exactly 15 subject folders")
        problems += 1

    out_cols = ["new_subject", "new_clip_num", "new_index",
                "local_clip_id", "local_subject", "session_num", "clip_num",
                "official_subject_id", "posture_measured"]
    out_cols += [c for c in ("fps_measured", "official_gt_rr") if c in df.columns]
    out_cols += ["src_mp4", "src_hdf5", "dst_mp4", "dst_hdf5"]
    df[out_cols].to_csv(remap_path, index=False, encoding="utf-8-sig")
    print(f"[write] {remap_path}  ({len(df)} rows)")

    print("\n[posture x new_subject]")
    print(pd.crosstab(df["new_subject"], df["posture_measured"], margins=True))

    if problems:
        print(f"\n[RESULT] FAILED with {problems} problem(s)")
        return 1
    print("\n[RESULT] OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
