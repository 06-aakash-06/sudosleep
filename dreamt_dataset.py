"""Read a bounded segment of an authorized DREAMT CSV without loading all channels."""
from __future__ import annotations

from pathlib import Path
import re

import numpy as np
import pandas as pd

ORIGINAL_ACCEL_HZ = 32.0
DEFAULT_DIRECTORY = Path(__file__).resolve().parent / "data/dreamt"


def _normalise(value: str) -> str:
    return re.sub(r"[^a-z0-9]", "", value.lower())


def load_dreamt_segment(path: Path, start_seconds: float, duration_seconds: float,
                        chunksize: int = 100_000) -> pd.DataFrame:
    """Load acceleration and optional sleep-stage labels from one local DREAMT file.

    The files are resampled to 64 or 100 Hz; the physical accelerometer rate
    remains 32 Hz. PSG sleep stages are reference stages, not apnea labels.
    """
    path = Path(path)
    if start_seconds < 0 or not 30 <= duration_seconds <= 3600:
        raise ValueError("Choose a non-negative start and 30–3600 seconds of data.")
    header = pd.read_csv(path, nrows=0)
    names = {_normalise(str(col)): str(col) for col in header.columns}
    required = {"timestamp": "time_s", "accx": "x", "accy": "y", "accz": "z"}
    if not set(required).issubset(names):
        raise ValueError("DREAMT CSV must contain TIMESTAMP, ACC_X, ACC_Y, and ACC_Z.")
    rename = {names[key]: value for key, value in required.items()}
    if "sleepstage" in names:
        rename[names["sleepstage"]] = "psg_sleep_stage"
    pieces = []
    last_timestamp = -np.inf
    end_seconds = start_seconds + duration_seconds
    with pd.read_csv(path, usecols=list(rename), chunksize=chunksize, low_memory=False) as reader:
        for chunk in reader:
            chunk = chunk.rename(columns=rename)
            timestamps = pd.to_numeric(chunk["time_s"], errors="coerce")
            finite = timestamps.dropna()
            if len(finite) and (not finite.is_monotonic_increasing or float(finite.iloc[0]) < last_timestamp):
                raise ValueError("DREAMT timestamps must be ordered within the file.")
            if len(finite):
                last_timestamp = float(finite.iloc[-1])
            selected = chunk.loc[(timestamps >= start_seconds) & (timestamps < end_seconds)].copy()
            if len(selected):
                selected["time_s"] = timestamps.loc[selected.index].astype(float) - start_seconds
                pieces.append(selected)
            if len(finite) and float(finite.iloc[-1]) >= end_seconds:
                break
    if not pieces:
        raise ValueError("No samples found in the requested time range.")
    frame = pd.concat(pieces, ignore_index=True)
    for axis in ("x", "y", "z"):
        frame[axis] = pd.to_numeric(frame[axis], errors="coerce")
    frame = frame.dropna(subset=["time_s", "x", "y", "z"]).reset_index(drop=True)
    if len(frame) < 2 or float(frame["time_s"].iloc[-1] - frame["time_s"].iloc[0]) < 29:
        raise ValueError("The selected segment has less than 30 seconds of usable samples.")
    intervals = np.diff(frame["time_s"].to_numpy(dtype=float))
    typical = float(np.median(intervals))
    if typical <= 0 or np.any(intervals <= 0) or np.any(intervals > typical * 1.5):
        raise ValueError("The selected segment contains timestamp gaps or duplicates; select a continuous segment.")
    frame["time_s"] -= float(frame["time_s"].iloc[0])
    frame["record"] = path.stem
    return frame
