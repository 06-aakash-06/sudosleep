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
                       if str(col).strip().lower() in {"record", "record_id", "subject", "subject_id"}), None)
    if label_col is None:
        raise SystemExit("Training CSV must contain a minute-level binary `label` column.")
    if record_col is None:
        raise SystemExit("Training CSV must contain a `record` column for subject-grouped validation.")

    data = preprocess_accelerometer(
        frame,
        sample_rate_hz=args.sample_rate,
        record_column=record_col,
        label_column=label_col,
    )
    labels = np.asarray(data["labels"], dtype=np.float32)
    records = np.asarray(data["records"])
    if np.any(labels < 0):
        raise SystemExit("Every retained one-minute window needs a binary label (0 or 1).")
    if len(np.unique(labels)) < 2:
        raise SystemExit("Training data must contain both normal and event-labeled minutes.")
    if len(np.unique(records)) < args.folds:
        raise SystemExit(
            f"Need at least {args.folds} independent recordings for {args.folds}-fold GroupKFold; "
            f"found {len(np.unique(records))}."
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
    save_accelerometer_checkpoint(models, metrics, labels, records, args.output)
    print(json.dumps({
        "checkpoint": args.output,
        "records": int(len(np.unique(records))),
        "windows": int(len(labels)),
        "metrics": metrics,
    }, indent=2, default=float))


if __name__ == "__main__":
    main()
