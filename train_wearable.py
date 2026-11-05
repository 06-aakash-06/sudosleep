"""Train the existing CNN/Transformer on labeled accelerometer recordings."""

from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd

from wearable_pipeline import preprocess_accelerometer, save_accelerometer_checkpoint


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv", help="CSV with x/y/z, label, and record columns")
    parser.add_argument("--sample-rate", type=float, default=50.0,
                        help="Fallback sample rate when timestamps are absent")
    parser.add_argument("--sensor-rate", type=float, default=None,
                        help="Original accelerometer rate before any file resampling, e.g. 32 for DREAMT")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--output", default="models/sudo_sleep_accelerometer.pt")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    frame = pd.read_csv(args.csv)
    label_col = next((str(col) for col in frame.columns
                      if str(col).strip().lower() in {"label", "apnea_label", "event_label"}), None)
    record_col = next((str(col) for col in frame.columns
                       if str(col).strip().lower() in {"record", "record_id", "recording", "recording_id"}), None)
    subject_col = next((str(col) for col in frame.columns
                        if str(col).strip().lower() in {"subject", "subject_id", "participant", "participant_id"}), None)
    if label_col is None:
        raise SystemExit("Training CSV must contain a minute-level binary `label` column.")
    if record_col is None and subject_col is None:
        raise SystemExit("Training CSV must contain a `record` or `subject` column.")
    record_col = record_col or subject_col
    subject_col = subject_col or record_col
    if frame[record_col].isna().any() or frame[subject_col].isna().any():
        raise SystemExit("Every sample needs a recording and subject identifier.")
    numeric_labels = pd.to_numeric(frame[label_col], errors="coerce")
    if numeric_labels.isna().any() or not numeric_labels.isin([0, 1]).all():
        raise SystemExit("Event labels must be binary 0 or 1 for every sample. PSG sleep stages are not apnea labels.")
    associations = frame[[record_col, subject_col]].astype(str).drop_duplicates()
    if record_col != subject_col and associations[record_col].duplicated().any():
        raise SystemExit("Each recording ID must belong to exactly one subject.")
    subject_by_record = dict(zip(frame[record_col].astype(str), frame[subject_col].astype(str)))

    data = preprocess_accelerometer(
        frame,
        sample_rate_hz=args.sample_rate,
        record_column=record_col,
        label_column=label_col,
        original_sensor_rate_hz=args.sensor_rate,
    )
    labels = np.asarray(data["labels"], dtype=np.float32)
    records = np.asarray(data["records"])
    groups = np.asarray([subject_by_record[str(record)] for record in records])
    data["groups"] = groups
    if np.any(labels < 0):
        raise SystemExit("Every retained one-minute window needs a binary label (0 or 1).")
    if len(np.unique(labels)) < 2:
        raise SystemExit("Training data must contain both normal and event-labeled minutes.")
    if args.folds < 2 or args.epochs < 1 or args.batch_size < 1:
        raise SystemExit("Use at least two folds and positive epochs and batch size.")
    if len(np.unique(groups)) < args.folds:
        raise SystemExit(
            f"Need at least {args.folds} independent subjects for {args.folds}-fold GroupKFold; "
            f"found {len(np.unique(groups))}."
        )

    import torch
    import pipeline as model_pipeline

    torch.manual_seed(42)
    np.random.seed(42)
    models, metrics = model_pipeline.train_model(
        df=pd.DataFrame(),
        precomputed=data,
        n_folds=args.folds,
        epochs=args.epochs,
        batch_size=args.batch_size,
        device=model_pipeline.get_device(),
    )
    save_accelerometer_checkpoint(models, metrics, labels, records, args.output,
                                  data["preprocessing_version"])
    print(json.dumps({
        "checkpoint": args.output,
        "records": int(len(np.unique(records))),
        "subjects": int(len(np.unique(groups))),
        "windows": int(len(labels)),
        "metrics": metrics,
    }, indent=2, default=float))


if __name__ == "__main__":
    main()
