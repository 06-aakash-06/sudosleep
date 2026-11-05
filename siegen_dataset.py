"""Read the University of Siegen ICHI 2014 wrist sleep dataset safely.

The archive contains run-length encoded 100 Hz accelerometer samples and PSG
sleep stages. It does not provide apnea-event labels. The axis values are
un-calibrated byte counts; no physical units are assumed.
"""
from __future__ import annotations

import argparse
import io
import re
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

DATASET_URL = "https://www.eti.uni-siegen.de/ubicomp/home/datasets/ichi14/ichi14_dataset.zip?lang=de"
DEFAULT_ARCHIVE = Path(__file__).resolve().parent / "data/siegen/ichi14_dataset.zip"
_RECORD = re.compile(r"data/(p[0-9]+[a-z]?)\.npy\Z")
STAGE_NAMES = {0: "Unknown", 1: "N3", 2: "N2", 3: "N1", 5: "REM", 6: "Wake", 7: "Movement"}


def download_dataset(path: Path = DEFAULT_ARCHIVE) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".zip.part")
    try:
        with urllib.request.urlopen(DATASET_URL, timeout=60) as response, temporary.open("wb") as out:
            while chunk := response.read(1024 * 1024):
                out.write(chunk)
        with zipfile.ZipFile(temporary) as archive:
            if archive.testzip() is not None or not archive_records(temporary):
                raise ValueError("Dataset archive is incomplete or contains no recordings.")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)
    return path


def archive_records(path: Path = DEFAULT_ARCHIVE) -> list[str]:
    with zipfile.ZipFile(path) as archive:
        return sorted(m.group(1) for n in archive.namelist() if (m := _RECORD.fullmatch(n)))


def _load_record(path: Path, record: str) -> np.ndarray:
    if not re.fullmatch(r"p[0-9]+[a-z]?", record):
        raise ValueError("Invalid recording ID.")
    with zipfile.ZipFile(path) as archive:
        member = archive.getinfo(f"data/{record}.npy")
        if member.file_size > 100_000_000:
            raise ValueError("Recording is unexpectedly large.")
        data = np.load(io.BytesIO(archive.read(member)), allow_pickle=False)
    if not data.dtype.names or not {"t", "d", "x", "y", "z", "gt"}.issubset(data.dtype.names):
        raise ValueError("Recording is missing expected sensor fields.")
    return data


def recording_info(path: Path, record: str) -> dict:
    data = _load_record(path, record)
    if len(data) < 2 or np.any(data["d"] == 0):
        raise ValueError("Recording has invalid run lengths.")
    elapsed = (data["t"] - data["t"][0]) * 86400.0
    if np.any(np.diff(elapsed) <= 0):
        raise ValueError("Recording timestamps are not strictly increasing.")
    scored = np.flatnonzero(np.isin(data["gt"], [1, 2, 3, 5, 6, 7]))
    asleep = np.flatnonzero(np.isin(data["gt"], [1, 2, 3, 5]))
    return {"duration_seconds": float(elapsed[-1] + int(data["d"][-1]) / 100.0),
            "first_scored_seconds": float(elapsed[scored[0]]) if len(scored) else 0.0,
            "first_sleep_seconds": float(elapsed[asleep[0]]) if len(asleep) else 0.0}


def load_segment(path: Path, record: str, start_seconds: float = 0.0,
                 duration_seconds: float = 300.0) -> pd.DataFrame:
    """Expand only the selected time range, preserving run timings and stages."""
    if start_seconds < 0 or duration_seconds < 30 or duration_seconds > 3600:
        raise ValueError("Choose a non-negative start and a duration from 30 to 3600 seconds.")
    data = _load_record(path, record)
    if len(data) < 2 or np.any(data["d"] == 0):
        raise ValueError("Recording has invalid run lengths.")
    starts = (data["t"] - data["t"][0]) * 86400.0
    differences = np.diff(starts)
    if np.any(differences <= 0):
        raise ValueError("Recording timestamps are not strictly increasing.")
    ends = np.r_[starts[1:], starts[-1] + int(data["d"][-1]) / 100.0]
    selection = (starts < start_seconds + duration_seconds) & (ends > start_seconds)
    runs = data[selection]
    run_starts, run_ends = starts[selection], ends[selection]
    if not len(runs):
        raise ValueError("The selected segment is outside the recording.")
    counts = runs["d"].astype(np.int64)
    offsets = np.arange(int(counts.sum())) - np.repeat(np.r_[0, np.cumsum(counts)[:-1]], counts)
    times = np.repeat(run_starts, counts) + offsets * np.repeat((run_ends - run_starts) / counts, counts)
    inside = (times >= start_seconds) & (times < start_seconds + duration_seconds)
    times = times[inside]
    if len(times) < 2:
        raise ValueError("The selected segment has too few samples.")
    relative_times = times - times[0]
    # The archive records a timestamp for each run; actual sample timing
    # varies slightly around the nominal 100 Hz. Put the expanded signal on
    # a uniform 100 Hz grid before passing it to fixed-rate signal filters.
    grid = np.arange(0.0, min(duration_seconds, relative_times[-1]), 0.01)
    if len(grid) < 3000:
        raise ValueError("The selected segment contains less than 30 seconds of samples.")
    # The published files store uint8 sensor counts. Center the selected
    # recording segment; their physical calibration is not supplied here.
    columns = {}
    for axis in ("x", "y", "z"):
        values = np.repeat(runs[axis], counts)[inside].astype(np.float32)
        columns[axis] = (np.interp(grid, relative_times, values) - np.median(values)).astype(np.float32)
    columns["time_s"] = grid
    source_stages = np.repeat(runs["gt"], counts)[inside].astype(np.int8)
    positions = np.searchsorted(relative_times, grid, side="right") - 1
    columns["sleep_stage_code"] = source_stages[np.maximum(positions, 0)]
    columns["record"] = record
    return pd.DataFrame(columns)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--download", action="store_true", help="Download the original archive")
    parser.add_argument("--archive", type=Path, default=DEFAULT_ARCHIVE)
    args = parser.parse_args()
    if args.download:
        download_dataset(args.archive)
    print(f"Archive: {args.archive} | recordings: {len(archive_records(args.archive))}")
