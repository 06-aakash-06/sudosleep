"""Wearable accelerometer preprocessing and experimental event screening.

The derived high-frequency channel is a cardiac-band *motion proxy*. It is not
an ECG reconstruction. The built-in fallback is a transparent heuristic; a
learned prediction is used only with a checkpoint trained on accelerometer
recordings.
"""

from __future__ import annotations

import re
from fractions import Fraction
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pywt
from scipy.integrate import trapezoid
from scipy.signal import butter, find_peaks, resample_poly, sosfiltfilt, welch

try:
    from PyEMD import EMD
except ImportError:  # The dashboard keeps a marked bandpass fallback.
    EMD = None

MODEL_FS = 100
EPOCH_SECONDS = 60
EPOCH_SAMPLES = MODEL_FS * EPOCH_SECONDS
N_SCALAR_FEATS = 12
FEATURE_NAMES = [
    "IBI variability proxy", "LF/HF proxy", "Respiratory amplitude",
    "Respiratory variability", "Cardiac-band motion RMS", "Wavelet energy",
    "Low-frequency wavelet energy", "Motion variability", "Zero crossings",
    "Spectral entropy", "Motion range", "Respiratory spectral flatness",
]

_AXIS_ALIASES = {
    "x": {"x", "ax", "accx", "accelx", "accelerometerx", "accelerationx"},
    "y": {"y", "ay", "accy", "accely", "accelerometery", "accelerationy"},
    "z": {"z", "az", "accz", "accelz", "accelerometerz", "accelerationz"},
}
_TIME_ALIASES = {"time", "timestamp", "datetime", "ts", "time_s", "timestamp_s"}


def _normalise_name(value: Any) -> str:
    return re.sub(r"[^a-z0-9]", "", str(value).lower())


def _find_axis_columns(frame: pd.DataFrame) -> dict[str, str]:
    normalised = {_normalise_name(col): str(col) for col in frame.columns}
    result: dict[str, str] = {}
    for axis, aliases in _AXIS_ALIASES.items():
        match = next((normalised[key] for key in aliases if key in normalised), None)
        if match is None:
            raise ValueError(
                "CSV needs three accelerometer columns (x/y/z or acc_x/acc_y/acc_z)."
            )
        result[axis] = match
    return result


def _find_time_column(frame: pd.DataFrame, requested: str | None) -> str | None:
    if requested:
        if requested not in frame.columns:
            raise ValueError(f"Timestamp column {requested!r} is not in the CSV.")
        return requested
    for col in frame.columns:
        if _normalise_name(col) in {_normalise_name(x) for x in _TIME_ALIASES}:
            return str(col)
    return None


def _timestamp_seconds(values: pd.Series, column_name: str) -> np.ndarray | None:
    """Convert a common seconds/ms/us/ns or datetime timestamp to seconds."""
    numeric = pd.to_numeric(values, errors="coerce")
    if numeric.notna().mean() >= 0.9:
        raw = numeric.to_numpy(dtype=np.float64)
        delta = np.diff(raw)
        delta = delta[np.isfinite(delta) & (delta > 0)]
        if not len(delta):
            return None
        key = _normalise_name(column_name)
        median_delta = float(np.median(delta))
        if "nano" in key or key.endswith("ns") or median_delta > 1e7:
            divisor = 1e9
        elif "micro" in key or key.endswith("us") or median_delta > 1e4:
            divisor = 1e6
        elif "milli" in key or key.endswith("ms") or median_delta > 1:
            divisor = 1e3
        else:
            divisor = 1.0
        return raw / divisor

    parsed = pd.to_datetime(values, errors="coerce", utc=True)
    if parsed.notna().mean() < 0.9:
        return None
    seconds = parsed.astype("int64").to_numpy(dtype=np.float64) / 1e9
    seconds[parsed.isna().to_numpy()] = np.nan
    return seconds


def infer_sample_rate(
    frame: pd.DataFrame,
    fallback_hz: float = 50.0,
    time_column: str | None = None,
) -> tuple[float, str | None]:
    """Infer Hz from a timestamp column when possible, else use the fallback."""
    column = _find_time_column(frame, time_column)
    if column is not None:
        seconds = _timestamp_seconds(frame[column], column)
        if seconds is not None:
            delta = np.diff(seconds)
            delta = delta[np.isfinite(delta) & (delta > 0)]
            if len(delta):
                hz = 1.0 / float(np.median(delta))
                if 1.0 <= hz <= 2000.0:
                    return hz, column
    if not np.isfinite(fallback_hz) or fallback_hz < 8:
        raise ValueError("Set a sampling rate of at least 8 Hz to extract respiration.")
    return float(fallback_hz), None


def _bandpass(signal: np.ndarray, low: float, high: float, fs: float) -> np.ndarray:
    if len(signal) < 32 or low <= 0 or high <= low or high >= fs / 2:
        return np.zeros_like(signal, dtype=np.float32)
    sos = butter(4, [low, high], btype="bandpass", fs=fs, output="sos")
    try:
        return sosfiltfilt(sos, signal).astype(np.float32)
    except ValueError:
        return np.zeros_like(signal, dtype=np.float32)


def _resample_to_model_rate(signal: np.ndarray, source_hz: float) -> np.ndarray:
    ratio = Fraction(MODEL_FS / source_hz).limit_denominator(10000)
    result = resample_poly(signal, ratio.numerator, ratio.denominator)
    return np.asarray(result, dtype=np.float32)


def _emd_scg_proxy(signal: np.ndarray, source_hz: float) -> tuple[np.ndarray, bool]:
    """Reconstruct a cardiac-band motion proxy from EMD IMFs in the SCG band.

    The input remains accelerometry. This does not reconstruct ECG morphology.
    At lower source rates, the usable upper edge is limited by the sensor's
    original Nyquist frequency even though the signal was resampled to 100 Hz.
    """
    signal = np.asarray(signal, dtype=np.float64)
    upper_hz = min(30.0, source_hz * 0.48, MODEL_FS * 0.48)
    if upper_hz <= 10.0 or len(signal) < MODEL_FS * 8:
        return np.zeros(len(signal), dtype=np.float32), EMD is not None
    if EMD is None:
        return _bandpass(signal, 10.0, upper_hz, MODEL_FS), False

    try:
        imfs = EMD().emd(signal, max_imf=10)
        if imfs.ndim != 2 or not len(imfs):
            return np.zeros(len(signal), dtype=np.float32), True
        selected = []
        for imf in imfs:
            freq, power = welch(imf, fs=MODEL_FS, nperseg=min(2048, len(imf)))
            band = (freq >= 10.0) & (freq <= upper_hz)
            if not band.any():
                continue
            peak = float(freq[band][np.argmax(power[band])])
            band_power = float(trapezoid(power[band], freq[band]))
            total_power = float(trapezoid(power, freq))
            if 10.0 <= peak <= upper_hz and band_power >= 0.20 * max(total_power, 1e-12):
                selected.append(imf)
        if not selected:
            return np.zeros(len(signal), dtype=np.float32), True
        return np.sum(selected, axis=0).astype(np.float32), True
    except (ValueError, FloatingPointError):
        return np.zeros(len(signal), dtype=np.float32), True


def _safe_std(values: np.ndarray) -> float:
    value = float(np.std(values)) if len(values) else 0.0
    return value if np.isfinite(value) else 0.0


def _feature_row(raw: np.ndarray, resp: np.ndarray, cardiac: np.ndarray,
                 valid_samples: int, cardiac_available: bool,
                 pause_reference: float) -> tuple[np.ndarray, float]:
    raw = raw[:valid_samples].astype(np.float64)
    resp = resp[:valid_samples].astype(np.float64)
    cardiac = cardiac[:valid_samples].astype(np.float64)
    if not len(raw):
        return np.zeros(N_SCALAR_FEATS, dtype=np.float32), 0.0

    peaks, _ = find_peaks(cardiac, height=_safe_std(cardiac) * 1.2,
                          distance=int(0.4 * MODEL_FS))
    ibi = np.diff(peaks) / MODEL_FS if len(peaks) >= 3 else np.array([])
    ibi = ibi[(ibi > 0.3) & (ibi < 2.0)]
    hrv = float(np.std(ibi)) if len(ibi) >= 2 else 0.0
    lf_hf = 0.0
    if len(ibi) >= 6:
        try:
            ibi_times = np.r_[0.0, np.cumsum(ibi)]
            ibi_values = np.r_[ibi[0], ibi]
            uniform_times = np.arange(0.0, ibi_times[-1], 0.25)
            if len(uniform_times) >= 16:
                uniform_ibi = np.interp(uniform_times, ibi_times, ibi_values)
                freqs, psd_ibi = welch(uniform_ibi, fs=4.0,
                                       nperseg=min(64, len(uniform_ibi)))
                low = (freqs >= 0.04) & (freqs < 0.15)
                high = (freqs >= 0.15) & (freqs <= 0.40)
                low_power = trapezoid(psd_ibi[low], freqs[low]) if low.any() else 0.0
                high_power = trapezoid(psd_ibi[high], freqs[high]) if high.any() else 0.0
                lf_hf = float(np.log1p(low_power / (high_power + 1e-12)))
        except (ValueError, FloatingPointError):
            lf_hf = 0.0

    scales = np.arange(4, 24)
    try:
        coeffs, _ = pywt.cwt(resp, scales, "morl", sampling_period=1 / MODEL_FS)
        energies = np.sum(np.square(np.abs(coeffs)), axis=1)
        wavelet_total = float(np.sum(energies))
        wavelet_low = float(np.sum(energies[-max(1, len(energies) // 3):]))
    except (ValueError, FloatingPointError):
        wavelet_total = wavelet_low = 0.0

    try:
        _, psd = welch(resp, fs=MODEL_FS, nperseg=min(512, len(resp)))
        psd = np.maximum(psd, 1e-12)
        spectral_entropy = float(-np.sum((psd / psd.sum()) * np.log2(psd / psd.sum())))
        spectral_flatness = float(np.exp(np.mean(np.log(psd))) / np.mean(psd))
    except (ValueError, FloatingPointError):
        spectral_entropy = spectral_flatness = 0.0

    scalars = np.asarray([
        hrv,
        lf_hf,
        float(np.ptp(resp)),
        float(np.std(resp)),
        float(np.sqrt(np.mean(cardiac ** 2))) if cardiac_available else 0.0,
        wavelet_total,
        wavelet_low,
        float(np.std(raw)),
        float(np.sum(np.diff(np.sign(raw)) != 0)),
        spectral_entropy,
        float(np.percentile(raw, 95) - np.percentile(raw, 5)),
        spectral_flatness,
    ], dtype=np.float32)
    # A bounded respiration-envelope pause fraction used by the transparent baseline.
    envelope = np.abs(resp)
    threshold = max(float(pause_reference) * 0.18, 1e-8)
    pause_fraction = float(np.mean(envelope < threshold))
    return scalars, pause_fraction


def _process_record(
    frame: pd.DataFrame,
    record_name: str,
    fallback_hz: float,
    time_column: str | None,
    label_column: str | None,
) -> dict[str, Any]:
    axis_columns = _find_axis_columns(frame)
    resolved_hz, detected_time = infer_sample_rate(frame, fallback_hz, time_column)
    time_name = detected_time
    axis_values = frame[[axis_columns["x"], axis_columns["y"], axis_columns["z"]]].apply(
        pd.to_numeric, errors="coerce"
    )
    valid = axis_values.notna().all(axis=1)
    if time_name and time_name in frame.columns:
        stamp = _timestamp_seconds(frame[time_name], time_name)
        valid &= np.isfinite(stamp)
        axis_values = axis_values.loc[valid]
        stamp = stamp[valid.to_numpy()]
        order = np.argsort(stamp, kind="stable")
        axis_values = axis_values.iloc[order]
    else:
        axis_values = axis_values.loc[valid]

    if len(axis_values) < int(resolved_hz * 30):
        raise ValueError(f"{record_name}: need at least 30 seconds of usable samples.")

    xyz = axis_values.to_numpy(dtype=np.float64)
    centered_axes = [xyz[:, axis] - np.median(xyz[:, axis]) for axis in range(3)]
    resampled_axes = np.stack(
        [_resample_to_model_rate(axis, resolved_hz) for axis in centered_axes], axis=0
    )
    magnitude = np.sqrt(np.sum(np.square(xyz), axis=1))
    magnitude -= np.median(magnitude)
    raw = _resample_to_model_rate(magnitude, resolved_hz)

    # Preserve signed breathing motion by selecting the axis with the clearest
    # respiratory-band energy instead of filtering the always-positive norm.
    resp_axes = np.stack(
        [_bandpass(axis, 0.1, 0.5, MODEL_FS) for axis in resampled_axes], axis=0
    )
    resp_axis = int(np.argmax(np.std(resp_axes, axis=1)))
    resp = resp_axes[resp_axis]
    cardiac_available = resolved_hz >= 40.0
    cardiac = np.zeros_like(raw)
    emd_used = False
    if cardiac_available:
        # Decompose minute-sized chunks so an overnight recording does not
        # require a single prohibitively large EMD operation.
        for start in range(0, len(raw), EPOCH_SAMPLES):
            stop = min(start + EPOCH_SAMPLES, len(raw))
            axis_proxies = []
            for axis in resampled_axes:
                proxy, used_emd = _emd_scg_proxy(axis[start:stop], resolved_hz)
                axis_proxies.append(proxy)
                emd_used = emd_used or used_emd
            best_axis = int(np.argmax([_safe_std(proxy) for proxy in axis_proxies]))
            cardiac[start:stop] = axis_proxies[best_axis]

    pause_reference = float(np.percentile(np.abs(resp), 70)) if len(resp) else 0.0

    n_epochs = int(np.ceil(len(raw) / EPOCH_SAMPLES))
    signals = np.zeros((n_epochs, 3, EPOCH_SAMPLES), dtype=np.float32)
    scalars = np.zeros((n_epochs, N_SCALAR_FEATS), dtype=np.float32)
    valid_seconds = np.zeros(n_epochs, dtype=np.float32)
    pause_fraction = np.zeros(n_epochs, dtype=np.float32)
    labels = np.full(n_epochs, -1, dtype=np.float32)
    source_labels: np.ndarray | None = None
    if label_column and label_column in frame.columns:
        source_labels = pd.to_numeric(frame[label_column], errors="coerce").to_numpy(dtype=np.float64)
        source_labels = source_labels[valid.to_numpy()]
        if time_name and time_name in frame.columns:
            source_labels = source_labels[order]

    for epoch in range(n_epochs):
        start = epoch * EPOCH_SAMPLES
        stop = min(start + EPOCH_SAMPLES, len(raw))
        length = stop - start
        if length < MODEL_FS * 15:
            continue
        raw_ep = raw[start:stop]
        resp_ep = resp[start:stop]
        cardiac_ep = cardiac[start:stop]
        signals[epoch, 0, :length] = raw_ep
        signals[epoch, 1, :length] = resp_ep
        signals[epoch, 2, :length] = cardiac_ep
        scalars[epoch], pause_fraction[epoch] = _feature_row(
            raw_ep, resp_ep, cardiac_ep, length, cardiac_available,
            pause_reference,
        )
        valid_seconds[epoch] = length / MODEL_FS
        if source_labels is not None:
            source_start = int(epoch * EPOCH_SECONDS * resolved_hz)
            source_stop = min(int((epoch + 1) * EPOCH_SECONDS * resolved_hz), len(source_labels))
            labels_ep = source_labels[source_start:source_stop]
            labels_ep = labels_ep[np.isfinite(labels_ep)]
            if len(labels_ep):
                labels[epoch] = float(np.mean(labels_ep >= 0.5) >= 0.5)

    keep = valid_seconds > 0
    signals, scalars = signals[keep], scalars[keep]
    valid_seconds, pause_fraction, labels = (
        valid_seconds[keep], pause_fraction[keep], labels[keep]
    )
    if not len(signals):
        raise ValueError(f"{record_name}: no complete or usable minute windows found.")

    # Per-record channel scaling keeps wearable units comparable while retaining
    # minute-to-minute changes in respiratory amplitude.
    for channel in range(3):
        values = signals[:, channel, :]
        mask = np.arange(EPOCH_SAMPLES)[None, :] < (valid_seconds[:, None] * MODEL_FS)
        samples = values[mask]
        mean = float(np.mean(samples)) if len(samples) else 0.0
        std = float(np.std(samples)) if len(samples) else 1.0
        if std < 1e-8:
            std = 1.0
        values[:] = (values - mean) / std
        values[~mask] = 0.0

    return {
        "signals_3ch": signals,
        "scalars": scalars,
        "labels": labels,
        "records": np.full(len(signals), str(record_name), dtype=object),
        "valid_seconds": valid_seconds,
        "pause_fraction": pause_fraction,
        "cardiac_available": cardiac_available,
        "cardiac_method": "EMD IMF reconstruction" if emd_used else "bandpass fallback",
        "cardiac_nonzero": bool(np.any(np.abs(cardiac) > 1e-8)),
        "sample_rate_hz": resolved_hz,
        "samples_model_fs": raw,
        "resp_signal": resp,
        "cardiac_proxy": cardiac,
    }


def preprocess_accelerometer(
    frame: pd.DataFrame,
    sample_rate_hz: float = 50.0,
    time_column: str | None = None,
    record_column: str | None = None,
    label_column: str | None = None,
) -> dict[str, Any]:
    """Create one-minute, 3-channel motion windows and 12 scalar features."""
    if frame.empty:
        raise ValueError("The uploaded CSV is empty.")
    if record_column is None:
        record_column = next((str(c) for c in frame.columns
                              if _normalise_name(c) in {"record", "recordid", "subject", "subjectid"}), None)
    if record_column is not None and record_column not in frame.columns:
        raise ValueError(f"Record column {record_column!r} is not in the CSV.")
    groups = frame.groupby(record_column, sort=False, dropna=False) if record_column else [("uploaded", frame)]
    records = []
    for name, part in groups:
        if pd.isna(name):
            continue
        records.append(_process_record(
            part, str(name), sample_rate_hz, time_column, label_column
        ))
    if not records:
        raise ValueError("No usable recordings were found in the CSV.")

    combined: dict[str, Any] = {}
    for key in ("signals_3ch", "scalars", "labels", "records", "valid_seconds", "pause_fraction"):
        combined[key] = np.concatenate([r[key] for r in records], axis=0)
    combined["cwt_energies"] = combined["scalars"][:, 5].copy()
    combined["record_summaries"] = [
        {
            "record": str(r["records"][0]),
            "sample_rate_hz": float(r["sample_rate_hz"]),
            "cardiac_available": bool(r["cardiac_available"]),
            "cardiac_method": r["cardiac_method"],
            "cardiac_nonzero": bool(r["cardiac_nonzero"]),
            "samples_model_fs": r["samples_model_fs"],
            "resp_signal": r["resp_signal"],
            "cardiac_proxy": r["cardiac_proxy"],
        }
        for r in records
    ]
    combined["feature_names"] = FEATURE_NAMES.copy()
    combined["model_fs"] = MODEL_FS
    combined["preprocessing_version"] = "emd_scg_v1" if EMD is not None else "bandpass_fallback_v1"
    return combined


def make_synthetic_demo(duration_minutes: int = 8,
                        sample_rate_hz: float = 50.0) -> pd.DataFrame:
    """Create a deterministic illustrative signal with one motion-suppression interval."""
    if duration_minutes < 3 or sample_rate_hz < 40:
        raise ValueError("The demo needs at least 3 minutes and a 40 Hz sample rate.")
    count = int(duration_minutes * 60 * sample_rate_hz)
    time = np.arange(count, dtype=np.float64) / sample_rate_hz
    amplitude = np.full(count, 0.22, dtype=np.float64)
    event = (time >= 3.0 * 60) & (time < 5.0 * 60)
    amplitude[event] = 0.006
    breathing = amplitude * (
        np.sin(2 * np.pi * 0.24 * time)
        + 0.12 * np.sin(2 * np.pi * 0.48 * time + 0.3)
    )
    pulse_gate = np.maximum(np.sin(2 * np.pi * 1.1 * time), 0.0) ** 6
    cardiac_motion = 0.018 * pulse_gate * np.sin(2 * np.pi * 9.0 * time)
    rng = np.random.default_rng(7)
    noise = rng.normal(0.0, 0.004, size=(count, 3))
    return pd.DataFrame({
        "time_s": time,
        "acc_x": breathing + cardiac_motion + noise[:, 0],
        "acc_y": 0.55 * breathing - 0.25 * cardiac_motion + noise[:, 1],
        "acc_z": 1.0 + 0.18 * breathing + cardiac_motion + noise[:, 2],
        "record": "synthetic-demo",
    })


def heuristic_scores(precomputed: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    """Return uncalibrated scores using a separate baseline for each record."""
    scalars = np.nan_to_num(precomputed["scalars"], nan=0.0, posinf=0.0, neginf=0.0)
    amp = np.maximum(scalars[:, 2], 0.0)
    variability = np.maximum(scalars[:, 3], 0.0)
    pause = np.clip(np.nan_to_num(precomputed["pause_fraction"]), 0.0, 1.0)
    records = np.asarray(precomputed.get("records", np.zeros(len(amp), dtype=int)))
    contributions = np.zeros((len(amp), 3), dtype=np.float64)
    for record in np.unique(records):
        idx = np.flatnonzero(records == record)
        record_amp, record_var = amp[idx], variability[idx]
        base_amp = max(float(np.percentile(record_amp[record_amp > 1e-8], 75))
                       if np.any(record_amp > 1e-8) else 0.0, 1e-8)
        base_var = max(float(np.percentile(record_var[record_var > 1e-8], 75))
                       if np.any(record_var > 1e-8) else 0.0, 1e-8)
        contributions[idx] = np.column_stack([
            0.35 * np.clip((0.85 - record_amp / base_amp) / 0.85, 0.0, 1.0),
            0.20 * np.clip((0.85 - record_var / base_var) / 0.85, 0.0, 1.0),
            0.45 * np.clip((pause[idx] - 0.08) / 0.42, 0.0, 1.0),
        ])
    scores = np.clip(contributions.sum(axis=1), 0.01, 0.99)
    return scores.astype(np.float32), contributions.astype(np.float32)


def heuristic_shap_attributions(
    precomputed: dict[str, Any], sample_index: int, max_background: int = 24
) -> tuple[list[str], np.ndarray, float, float]:
    """Explain one fallback score with Kernel SHAP over its three inputs.

    This intentionally explains the transparent heuristic only. It does not
    attribute a CNN/Transformer prediction.
    """
    try:
        import shap
    except ImportError as exc:
        raise RuntimeError("Install the `shap` dependency to show SHAP explanations.") from exc

    scalars = np.nan_to_num(
        np.asarray(precomputed["scalars"], dtype=np.float64),
        nan=0.0, posinf=0.0, neginf=0.0,
    )
    pauses = np.clip(
        np.nan_to_num(np.asarray(precomputed["pause_fraction"], dtype=np.float64)),
        0.0, 1.0,
    )
    if not 0 <= sample_index < len(scalars):
        raise IndexError("Selected minute is outside the processed recording.")

    records = np.asarray(precomputed.get("records", np.zeros(len(scalars), dtype=int)))
    same_record = np.flatnonzero(records == records[sample_index])
    sample_index = int(np.flatnonzero(same_record == sample_index)[0])
    scalars, pauses = scalars[same_record], pauses[same_record]

    inputs = np.column_stack([np.maximum(scalars[:, 2], 0.0),
                              np.maximum(scalars[:, 3], 0.0), pauses])
    amp = inputs[:, 0]
    variability = inputs[:, 1]
    base_amp = max(float(np.percentile(amp[amp > 1e-8], 75))
                   if np.any(amp > 1e-8) else 0.0, 1e-8)
    base_var = max(float(np.percentile(variability[variability > 1e-8], 75))
                   if np.any(variability > 1e-8) else 0.0, 1e-8)

    def score_fn(values: np.ndarray) -> np.ndarray:
        values = np.asarray(values, dtype=np.float64)
        amp_suppression = np.clip((0.85 - values[:, 0] / base_amp) / 0.85, 0.0, 1.0)
        var_suppression = np.clip((0.85 - values[:, 1] / base_var) / 0.85, 0.0, 1.0)
        pause_driver = np.clip((values[:, 2] - 0.08) / 0.42, 0.0, 1.0)
        return np.clip(
            0.35 * amp_suppression + 0.20 * var_suppression + 0.45 * pause_driver,
            0.01, 0.99,
        )

    if len(inputs) > max_background:
        background_indices = np.linspace(0, len(inputs) - 1, max_background, dtype=int)
        background = inputs[background_indices]
    else:
        background = inputs
    explainer = shap.KernelExplainer(score_fn, background, link="identity")
    values = explainer.shap_values(
        inputs[sample_index:sample_index + 1], nsamples=2 ** (inputs.shape[1] + 1)
    )
    if isinstance(values, list):
        values = values[0]
    attributions = np.asarray(values, dtype=np.float64).reshape(-1, inputs.shape[1])[0]
    baseline = float(np.asarray(explainer.expected_value).reshape(-1)[0])
    predicted = float(score_fn(inputs[sample_index:sample_index + 1])[0])
    return ["Respiratory amplitude", "Respiratory variability", "Pause fraction"], \
        attributions, baseline, predicted


def default_transition_matrix(normal_stay: float = 0.90,
                             event_stay: float = 0.85) -> np.ndarray:
    """A visible persistence prior for the heuristic fallback, not learned data."""
    normal_stay = float(np.clip(normal_stay, 0.5, 0.999))
    event_stay = float(np.clip(event_stay, 0.5, 0.999))
    return np.asarray([
        [normal_stay, 1.0 - normal_stay],
        [1.0 - event_stay, event_stay],
    ], dtype=np.float64)


def estimate_transition_matrix(
    labels: np.ndarray, records: np.ndarray, alpha: float = 1.0
) -> np.ndarray:
    """Estimate a smoothed 2-state transition matrix without crossing records."""
    counts = np.full((2, 2), float(alpha), dtype=np.float64)
    for record in np.unique(records):
        idx = np.flatnonzero(records == record)
        seq = np.asarray(labels[idx], dtype=int)
        for left, right in zip(seq[:-1], seq[1:]):
            if left in (0, 1) and right in (0, 1):
                counts[left, right] += 1.0
    return counts / counts.sum(axis=1, keepdims=True)


def viterbi_smooth(
    scores: np.ndarray,
    transition_matrix: np.ndarray | None = None,
    initial: tuple[float, float] = (0.95, 0.05),
) -> np.ndarray:
    """Decode the most likely two-state sequence from candidate scores."""
    emissions = np.clip(np.asarray(scores, dtype=np.float64), 1e-6, 1 - 1e-6)
    if not len(emissions):
        return np.asarray([], dtype=np.int8)
    transition = np.asarray(
        transition_matrix if transition_matrix is not None else default_transition_matrix(),
        dtype=np.float64,
    )
    if transition.shape != (2, 2) or np.any(transition < 0):
        raise ValueError("Transition matrix must be a non-negative 2 x 2 matrix.")
    transition = transition / np.maximum(transition.sum(axis=1, keepdims=True), 1e-12)
    log_transition = np.log(np.maximum(transition, 1e-12))
    log_initial = np.log(np.maximum(np.asarray(initial, dtype=np.float64), 1e-12))
    log_emission = np.column_stack([np.log1p(-emissions), np.log(emissions)])
    scores_dp = np.empty((len(emissions), 2), dtype=np.float64)
    back = np.zeros((len(emissions), 2), dtype=np.int8)
    scores_dp[0] = log_initial + log_emission[0]
    for i in range(1, len(emissions)):
        candidates = scores_dp[i - 1][:, None] + log_transition
        back[i] = np.argmax(candidates, axis=0)
        scores_dp[i] = np.max(candidates, axis=0) + log_emission[i]
    states = np.empty(len(emissions), dtype=np.int8)
    states[-1] = int(np.argmax(scores_dp[-1]))
    for i in range(len(emissions) - 2, -1, -1):
        states[i] = back[i + 1, states[i + 1]]
    return states


def apply_markov_smoothing(
    scores: np.ndarray,
    records: np.ndarray,
    transition_matrix: np.ndarray | None = None,
) -> np.ndarray:
    states = np.zeros(len(scores), dtype=np.int8)
    for record in np.unique(records):
        idx = np.flatnonzero(records == record)
        states[idx] = viterbi_smooth(scores[idx], transition_matrix)
    return states


def candidate_event_rate(states: np.ndarray, records: np.ndarray,
                         valid_seconds: np.ndarray) -> tuple[int, float, float]:
    """Count contiguous candidate runs and return runs/hour (not clinical AHI)."""
    event_runs = 0
    duration_hours = float(np.sum(valid_seconds) / 3600.0)
    for record in np.unique(records):
        seq = states[records == record]
        if len(seq):
            event_runs += int(np.sum((seq == 1) & np.r_[True, seq[:-1] == 0]))
    rate = event_runs / duration_hours if duration_hours > 0 else 0.0
    return event_runs, duration_hours, float(rate)


def predict_with_accelerometer_checkpoint(
    precomputed: dict[str, Any], checkpoint_path: str | Path
) -> tuple[np.ndarray, np.ndarray]:
    """Run only a checkpoint explicitly tagged as accelerometer-trained."""
    import torch

    try:
        import pipeline as model_pipeline
    except Exception as exc:
        raise RuntimeError(f"The CNN/Transformer runtime is unavailable: {exc}") from exc

    device = model_pipeline.get_device()
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    if not isinstance(checkpoint, dict) or checkpoint.get("input_modality") != "accelerometer":
        raise ValueError("Checkpoint is not tagged as trained for accelerometer input.")
    if checkpoint.get("preprocessing_version") != precomputed.get("preprocessing_version"):
        raise ValueError("Checkpoint preprocessing does not match the current signal pipeline. Retrain it with this version.")
    if checkpoint.get("feature_names") != FEATURE_NAMES:
        raise ValueError("Checkpoint feature definitions do not match the current pipeline.")
    states = checkpoint.get("model_states")
    if not states:
        raise ValueError("Checkpoint contains no model weights.")
    transition = np.asarray(checkpoint.get("transition_matrix", default_transition_matrix()))
    labels = np.zeros(len(precomputed["labels"]), dtype=np.float32)
    dataset = model_pipeline.TemporalContextDataset(
        precomputed["signals_3ch"], precomputed["scalars"], labels,
        precomputed["records"],
    )
    loader = torch.utils.data.DataLoader(dataset, batch_size=128, shuffle=False, num_workers=0)
    ensemble_probs = []
    for state in states:
        model = model_pipeline.SudoSleepNet().to(device)
        model.load_state_dict(state)
        model.eval()
        probs = []
        with torch.no_grad():
            for signal, scalar, _ in loader:
                logits = model(signal.to(device), scalar.to(device)).squeeze(-1)
                probs.extend(torch.sigmoid(logits).cpu().numpy().tolist())
        ensemble_probs.append(np.asarray(probs, dtype=np.float32))
    return np.mean(ensemble_probs, axis=0), transition


def save_accelerometer_checkpoint(
    models: list[Any], metrics: dict[str, Any], labels: np.ndarray,
    records: np.ndarray, output_path: str | Path, preprocessing_version: str,
) -> None:
    import torch
    from torch.optim.swa_utils import AveragedModel

    model_states = []
    for model in models:
        model_states.append(
            model.module.state_dict() if isinstance(model, AveragedModel)
            else model.state_dict()
        )
    artifact = {
        "artifact_version": 2,
        "input_modality": "accelerometer",
        "preprocessing_version": preprocessing_version,
        "architecture": "SudoSleepNet",
        "feature_names": FEATURE_NAMES,
        "model_states": model_states,
        "transition_matrix": estimate_transition_matrix(labels, records).tolist(),
        "metrics": {key: value for key, value in metrics.items()
                    if isinstance(value, (float, int, str, list))},
    }
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(artifact, path)
