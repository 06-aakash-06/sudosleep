"""Create subject-held-out splits and local waveform caches from DREAMT 2.2.

Event fields use sparse 1/blank encoding. Blank-as-no-marked-event is a
provisional inference, supported by the local AHI audit, not an explicit
guarantee in the supplied metadata. Files and derived outputs remain local.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from audit_dreamt import EVENTS
from wearable_pipeline import preprocess_accelerometer

ROOT = Path("data/dreamt/DREAMT")
OUTPUT = ROOT / "derived"


def derive_epoch_targets(times: np.ndarray, stages: np.ndarray, event: np.ndarray,
                         sample_rate: int = 64) -> tuple[np.ndarray, np.ndarray]:
    """Apply the documented provisional sparse-event target definition."""
    epoch = np.floor((times - times[0]) / 60.0).astype(np.int32)
    count = np.bincount(epoch)
    asleep = np.isin(stages, ["N1", "N2", "N3", "R"])
    sleep_count = np.bincount(epoch, weights=asleep, minlength=len(count))
    event_count = np.bincount(epoch, weights=event & asleep, minlength=len(count))
    valid = (count >= sample_rate * 30) & (sleep_count >= 0.8 * count)
    labels = np.full(len(count), -1, dtype=np.int8)
    labels[valid] = (event_count[valid] >= sample_rate).astype(np.int8)
    return labels, (sleep_count / np.maximum(count, 1)).astype(np.float32)


def create_split(seed: int = 42) -> dict:
    audit_path = OUTPUT / "audit_64hz.csv"
    if not audit_path.is_file():
        raise FileNotFoundError("Run audit_dreamt.py before creating the split.")
    audit = pd.read_csv(audit_path)
    audit["AHI"] = pd.to_numeric(audit["AHI"], errors="coerce")
    audit["difference"] = (audit["runs_per_sleep_h"] - audit["AHI"]).abs()
    good = audit["AHI"].notna() & (audit["difference"] <= 2) & (audit["gaps"] == 0) & \
           (audit["nonmonotonic_or_invalid"] == 0) & (audit["invalid_acc_rows"] == 0)
    chosen = audit.loc[good].copy()
    chosen["severity"] = pd.cut(chosen["AHI"], bins=[-1, 5, 15, 30, np.inf],
                                labels=["low", "mild", "moderate", "severe"], right=False)
    trainval, test = train_test_split(chosen, test_size=0.15, random_state=seed,
                                      stratify=chosen["severity"])
    train, val = train_test_split(trainval, test_size=0.17647, random_state=seed,
                                  stratify=trainval["severity"])
    split = {"seed": seed, "label_semantics": "provisional: blank means no marked event within scored sleep",
             "excluded": audit.loc[~good, "subject"].tolist(),
             "train": sorted(train["subject"].tolist()), "validation": sorted(val["subject"].tolist()),
             "test": sorted(test["subject"].tolist())}
    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT / "subject_splits.json").write_text(json.dumps(split, indent=2))
    return split


def prepare_subject(subject: str) -> dict:
    from wearable_pipeline import EMD
    if EMD is None:
        raise RuntimeError("EMD-signal is required to prepare the proposal pipeline.")
    source = ROOT / "data_64Hz" / f"{subject}_whole_df.csv"
    target = OUTPUT / "features" / f"{subject}.npz"
    if target.is_file():
        with np.load(target, allow_pickle=False) as saved:
            return {"subject": subject, "status": "cached", "windows": len(saved["labels"])}
    start = time.monotonic()
    columns = ["TIMESTAMP", "ACC_X", "ACC_Y", "ACC_Z", "Sleep_Stage", *EVENTS]
    frame = pd.read_csv(source, usecols=columns, dtype={"ACC_X": "float32", "ACC_Y": "float32", "ACC_Z": "float32"})
    times = pd.to_numeric(frame["TIMESTAMP"], errors="raise").to_numpy(dtype=np.float64)
    if len(times) < 3840 or np.any(np.diff(times) <= 0) or np.any(np.diff(times) > 1.5 / 64):
        raise ValueError("Recording is short or has invalid timestamps.")
    event = np.zeros(len(frame), dtype=bool)
    for name in EVENTS:
        values = pd.to_numeric(frame[name], errors="coerce").to_numpy(dtype=float)
        event |= values == 1
    labels, sleep_fraction = derive_epoch_targets(times, frame["Sleep_Stage"].to_numpy(), event)
    source_frame = frame[["TIMESTAMP", "ACC_X", "ACC_Y", "ACC_Z"]].rename(columns={"TIMESTAMP": "time_s"})
    source_frame["record"] = subject
    del frame, event, times
    prepared = preprocess_accelerometer(source_frame, record_column="record", original_sensor_rate_hz=32.0)
    n = len(prepared["signals_3ch"])
    # One minute is positive if at least one second overlaps an annotated event.
    # Windows with insufficient scored sleep or insufficient samples are excluded.
    labels, sleep_fraction = labels[:n], sleep_fraction[:n]
    valid = labels >= 0
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(".tmp")
    try:
        with temporary.open("wb") as handle:
            np.savez(handle, signals=prepared["signals_3ch"], scalars=prepared["scalars"],
                     labels=labels, valid_seconds=prepared["valid_seconds"],
                     sleep_fraction=sleep_fraction,
                     preprocessing_version=prepared["preprocessing_version"], subject=subject)
        temporary.replace(target)
    finally:
        temporary.unlink(missing_ok=True)
    return {"subject": subject, "status": "prepared", "windows": n, "eligible": int(valid.sum()),
            "positive": int((labels == 1).sum()), "seconds": round(time.monotonic() - start, 1),
            "cardiac_nonzero": prepared["record_summaries"][0]["cardiac_nonzero"]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", action="store_true")
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--skip-first", type=int, default=0)
    args = parser.parse_args()
    split_path = OUTPUT / "subject_splits.json"
    split = create_split() if args.split or not split_path.exists() else json.loads(split_path.read_text())
    print("Split sizes:", {k: len(split[k]) for k in ("train", "validation", "test", "excluded")}, flush=True)
    if not args.prepare:
        return
    subjects = split["train"] + split["validation"] + split["test"]
    subjects = subjects[args.skip_first:]
    if args.limit:
        subjects = subjects[:args.limit]
    results = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(prepare_subject, subject): subject for subject in subjects}
        for future in as_completed(futures):
            subject = futures[future]
            try:
                result = future.result()
            except Exception as exc:
                result = {"subject": subject, "status": "error", "error": str(exc)}
            results.append(result)
            print(f"{len(results)}/{len(subjects)} {result}", flush=True)
    (OUTPUT / "preparation_report.json").write_text(json.dumps(results, indent=2))
    if any(r["status"] == "error" for r in results):
        raise SystemExit("One or more recordings failed; inspect preparation_report.json.")


if __name__ == "__main__":
    main()
