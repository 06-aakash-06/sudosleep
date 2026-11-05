"""Audit DREAMT event, stage, timestamp and accelerometer coverage locally."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import json
import re
import time

import numpy as np
import pandas as pd

EVENTS = ["Obstructive_Apnea", "Central_Apnea", "Hypopnea", "Multiple_Events"]
STAGES = ["W", "N1", "N2", "N3", "R", "P", "Missing"]


def audit_file(path: Path) -> dict:
    subject = path.name.split("_")[0]
    needed = ["TIMESTAMP", "Sleep_Stage", "ACC_X", "ACC_Y", "ACC_Z", *EVENTS]
    headers = list(pd.read_csv(path, nrows=0).columns)
    missing = [x for x in needed if x not in headers]
    if missing:
        return {"subject": subject, "file": path.name, "error": f"Missing columns: {missing}"}
    n = 0
    first = last = None
    gaps = reverse = nonfinite_acc = positive_rows = event_runs = sleep_rows = event_sleep_rows = 0
    type_positive = {x: 0 for x in EVENTS}
    type_other = {x: set() for x in EVENTS}
    stage_counts = {x: 0 for x in STAGES}
    other_stages: set[str] = set()
    previous_event = False
    sampling_hz = 100.0 if path.name.endswith("_PSG_df.csv") else 64.0
    expected_step = 1 / sampling_hz
    with pd.read_csv(path, usecols=needed, chunksize=200_000, low_memory=False) as reader:
        for chunk in reader:
            times = pd.to_numeric(chunk["TIMESTAMP"], errors="coerce").to_numpy(dtype=float)
            if not len(times):
                continue
            if first is None:
                first = float(times[0])
            diffs = np.diff(times, prepend=last if last is not None else times[0] - expected_step)
            gaps += int(np.sum(diffs > expected_step * 1.5))
            reverse += int(np.sum(~np.isfinite(diffs) | (diffs <= 0)))
            last = float(times[-1])
            n += len(chunk)
            axes = chunk[["ACC_X", "ACC_Y", "ACC_Z"]].apply(pd.to_numeric, errors="coerce").to_numpy()
            nonfinite_acc += int(np.sum(~np.isfinite(axes).all(axis=1)))
            stage = chunk["Sleep_Stage"].fillna("<blank>").astype(str).to_numpy()
            for name in STAGES:
                stage_counts[name] += int(np.sum(stage == name))
            other_stages.update(np.unique(stage[~np.isin(stage, STAGES)]).tolist())
            asleep = np.isin(stage, ["N1", "N2", "N3", "R"])
            sleep_rows += int(asleep.sum())
            event = np.zeros(len(chunk), dtype=bool)
            for name in EVENTS:
                raw = chunk[name]
                values = pd.to_numeric(raw, errors="coerce").to_numpy(dtype=float)
                one = values == 1
                event |= one
                type_positive[name] += int(one.sum())
                unusual = raw[raw.notna() & ~raw.astype(str).isin(["1", "1.0"])].astype(str).unique()
                type_other[name].update(unusual[:20].tolist())
            positive_rows += int(event.sum())
            event_sleep_rows += int((event & asleep).sum())
            previous = np.r_[previous_event, event[:-1]]
            event_runs += int(np.sum(event & (~previous | (diffs > expected_step * 1.5) | (diffs <= 0))))
            previous_event = bool(event[-1])
    hours = sleep_rows / sampling_hz / 3600.0
    return {"subject": subject, "file": path.name, "rows": n, "first_s": first, "last_s": last,
            "elapsed_h": ((last - first) / 3600 if n else None), "sleep_h": hours,
            "positive_rows": positive_rows, "positive_sleep_rows": event_sleep_rows,
            "event_runs_all_stages": event_runs, "runs_per_sleep_h": event_runs / hours if hours else None,
            "gaps": gaps, "nonmonotonic_or_invalid": reverse, "invalid_acc_rows": nonfinite_acc,
            "stage_counts": stage_counts, "other_stages": sorted(other_stages),
            "event_type_positive_rows": type_positive,
            "unexpected_event_values": {k: sorted(v) for k, v in type_other.items()}}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/dreamt/DREAMT"))
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--variant", choices=["64", "100"], default="64")
    parser.add_argument("--subjects", default="", help="Comma-separated subject IDs")
    args = parser.parse_args()
    pattern = "S*_whole_df.csv" if args.variant == "64" else "S*_PSG_df.csv"
    files = sorted((args.data / f"data_{args.variant}Hz").glob(pattern))
    if args.subjects:
        subjects = {s.strip() for s in args.subjects.split(",")}
        files = [p for p in files if p.name.split("_")[0] in subjects]
    if args.limit:
        files = files[:args.limit]
    if not files:
        raise SystemExit("No DREAMT 64 Hz subject files found.")
    start = time.monotonic()
    results = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        tasks = {pool.submit(audit_file, path): path for path in files}
        for future in as_completed(tasks):
            path = tasks[future]
            try:
                result = future.result()
            except Exception as exc:
                result = {"subject": path.name.split("_")[0], "file": path.name, "error": str(exc)}
            results.append(result)
            print(f"{len(results)}/{len(files)} {path.name} "
                  f"{'ERROR ' + result['error'] if 'error' in result else 'done'} "
                  f"({time.monotonic() - start:.1f}s)", flush=True)
    results.sort(key=lambda row: row["subject"])
    out = args.data / "derived"
    out.mkdir(parents=True, exist_ok=True)
    payload = {"source": f"DREAMT 2.2 data_{args.variant}Hz", "blank_event_semantics": "unverified",
               "event_run_assumption": "1 denotes positive; blanks are not confirmed negative",
               "results": results}
    (out / f"audit_{args.variant}hz.json").write_text(json.dumps(payload, indent=2))
    metadata = pd.read_csv(args.data / "participant_info.csv", usecols=["SID", "AHI"])
    summary = pd.DataFrame([{k: v for k, v in row.items() if not isinstance(v, (dict, list))}
                            for row in results]).merge(metadata, left_on="subject", right_on="SID", how="left")
    summary.to_csv(out / f"audit_{args.variant}hz.csv", index=False)
    print(f"Saved local audit to {out}; {sum('error' in x for x in results)} errors.")


if __name__ == "__main__":
    main()
