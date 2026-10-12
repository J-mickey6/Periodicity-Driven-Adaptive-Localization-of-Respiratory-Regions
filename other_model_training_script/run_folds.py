"""
run_folds.py  --  run several folds back to back, unattended.

Calls run_step1.py once per fold (stage 4 only; the cache is already built),
parses the test metrics out of each run, and writes one combined report.
A failing fold does not stop the queue.

Typical overnight use:

  python run_folds.py --folds 2,3,4,5,6 ^
      --body-detector "D:\\hdd\\RE_baby\\ROI Detectors\\yolov8m.pt" ^
      --face-detector "D:\\hdd\\RE_baby\\ROI Detectors\\yolov8n-face.pt"

Read D:\\air400_dl\\folds_summary.txt in the morning. Per-fold console output is
kept in D:\\air400_dl\\fold_logs\\fold{N}.log.
"""

from __future__ import annotations

import argparse
import datetime
import os
import re
import subprocess
import sys
import time

METRIC_RE = {
    "mae": re.compile(r"test_mae:\s*([0-9.]+)"),
    "rmse": re.compile(r"test_rmse:\s*([0-9.]+)"),
    "mse": re.compile(r"test_mse:\s*([0-9.]+)"),
    "pearson": re.compile(r"test_pearson_r:\s*(-?[0-9.]+)"),
}
SIT_RE = re.compile(r"(\d+\.\d+)s/it")
BEST_RE = re.compile(r"Best trained epoch:\s*(\d+),\s*min_val_loss:\s*([0-9.eE+-]+)")


def gpu_mem() -> str:
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used,memory.total", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=30)
        return out.stdout.strip().replace("\n", " | ")
    except Exception as exc:  # noqa: BLE001
        return f"(nvidia-smi failed: {exc})"


def run_one(fold: int, args) -> dict:
    log_dir = os.path.join(args.report_dir, "fold_logs")
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, f"fold{fold}.log")

    cmd = [
        sys.executable, args.runner,
        "--body-detector", args.body_detector,
        "--face-detector", args.face_detector,
        "--stages", "4",
        "--epochs", str(args.epochs),
        "--batch-size", str(args.batch_size),
        "--tag", args.tag,
        "--pilot-fold", str(fold),
    ]

    print("\n" + "=" * 70, flush=True)
    print(f"FOLD {fold}  start {datetime.datetime.now():%Y-%m-%d %H:%M:%S}", flush=True)
    print(f"  gpu before: {gpu_mem()}", flush=True)
    print("  " + " ".join(cmd), flush=True)
    print("=" * 70, flush=True)

    t0 = time.time()
    lines: list[str] = []
    gpu_during = ""
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            text=True, encoding="utf-8", errors="replace", bufsize=1)
    assert proc.stdout is not None
    for line in proc.stdout:
        line = line.rstrip("\n")
        print(line, flush=True)
        lines.append(line)
        # sample GPU memory once, shortly after training actually starts
        if not gpu_during and "Train epoch 0" in line and "s/it" in line:
            gpu_during = gpu_mem()
    proc.wait()
    dt = (time.time() - t0) / 60.0

    with open(log_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    text = "\n".join(lines)
    res = {
        "fold": fold,
        "rc": proc.returncode,
        "minutes": dt,
        "gpu_during": gpu_during or gpu_mem(),
        "log": log_path,
    }
    for k, rx in METRIC_RE.items():
        m = rx.search(text)
        res[k] = m.group(1) if m else None

    sits = SIT_RE.findall(text)
    res["first_sit"] = sits[0] if sits else None
    res["last_sit"] = sits[-1] if sits else None

    b = BEST_RE.search(text)
    res["best_epoch"] = b.group(1) if b else None
    res["min_val_loss"] = b.group(2) if b else None

    res["oom"] = "out of memory" in text.lower()

    print(f"\nFOLD {fold}  done  rc={proc.returncode}  {dt:.1f} min  "
          f"mae={res['mae']} rmse={res['rmse']} r={res['pearson']}", flush=True)
    return res


def write_report(results: list[dict], args, started) -> str:
    path = os.path.join(args.report_dir, "folds_summary.txt")
    ok = [r for r in results if r["rc"] == 0 and r["mae"]]

    out = []
    out.append(f"run_folds.py   started {started:%Y-%m-%d %H:%M}  "
               f"finished {datetime.datetime.now():%Y-%m-%d %H:%M}")
    out.append(f"model=EfficientPhys of=farneback batch={args.batch_size} "
               f"epochs={args.epochs} tag={args.tag}")
    out.append("")
    out.append(f"{'fold':<5} {'rc':<4} {'min':>7} {'MAE':>8} {'RMSE':>8} {'Pearson':>9} "
               f"{'best_ep':>8} {'s/it':>7}")
    out.append("-" * 62)
    for r in results:
        out.append(f"{r['fold']:<5} {r['rc']:<4} {r['minutes']:>7.1f} "
                   f"{(r['mae'] or '-'):>8} {(r['rmse'] or '-'):>8} "
                   f"{(r['pearson'] or '-'):>9} {(r['best_epoch'] or '-'):>8} "
                   f"{(r['last_sit'] or '-'):>7}")
        if r["oom"]:
            out.append(f"      fold {r['fold']}: CUDA OUT OF MEMORY")

    if len(ok) >= 2:
        import statistics as st
        for key, label in (("mae", "MAE"), ("rmse", "RMSE"), ("pearson", "Pearson")):
            vals = [float(r[key]) for r in ok if r[key] is not None]
            if len(vals) >= 2:
                out.append(f"{label:>8} across {len(vals)} folds: "
                           f"mean={st.mean(vals):.4f}  sd={st.stdev(vals):.4f}")
    out.append("")
    out.append("NOTE: these are per-fold test metrics only. The cross-validated number for")
    out.append("the paper is the mean over all six folds, so fold 1 must be included when")
    out.append("assembling the final table.")
    out.append("")
    for r in results:
        out.append(f"  fold {r['fold']} gpu during training: {r['gpu_during']}")
        out.append(f"  fold {r['fold']} log: {r['log']}")

    text = "\n".join(out)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)
    return text


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runner", default=r"D:\air400_dl\run_step1.py")
    ap.add_argument("--report-dir", default=r"D:\air400_dl")
    ap.add_argument("--body-detector", required=True)
    ap.add_argument("--face-detector", default="")
    ap.add_argument("--folds", default="2,3,4,5,6")
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--tag", default="b4e15")
    args = ap.parse_args()

    folds = [int(x) for x in args.folds.split(",") if x.strip()]
    started = datetime.datetime.now()

    print(f"run_folds.py  {started:%Y-%m-%d %H:%M}  folds={folds}", flush=True)
    print(f"gpu at start: {gpu_mem()}", flush=True)

    results = []
    for fold in folds:
        try:
            results.append(run_one(fold, args))
        except KeyboardInterrupt:
            print("\ninterrupted by user; writing partial report", flush=True)
            break
        except Exception as exc:  # noqa: BLE001
            import traceback
            traceback.print_exc()
            results.append({"fold": fold, "rc": -1, "minutes": 0.0,
                            "gpu_during": "", "log": "", "mae": None, "rmse": None,
                            "mse": None, "pearson": None, "first_sit": None,
                            "last_sit": None, "best_epoch": None,
                            "min_val_loss": None, "oom": False})

    text = write_report(results, args, started)
    print("\n" + "=" * 70)
    print("SUMMARY  (paste this back)")
    print("=" * 70)
    print(text)
    print("=" * 70)
    print(f"saved: {os.path.join(args.report_dir, 'folds_summary.txt')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
