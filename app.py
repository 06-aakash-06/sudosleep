"""Streamlit demo for accelerometer-derived respiratory signal exploration."""

from __future__ import annotations

from pathlib import Path
import json

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots
from scipy.signal import spectrogram
from dreamt_dataset import DEFAULT_DIRECTORY as DREAMT_DIRECTORY, ORIGINAL_ACCEL_HZ, load_dreamt_segment, recording_start
from siegen_dataset import DEFAULT_ARCHIVE, STAGE_NAMES, archive_records, load_segment, recording_info

from wearable_pipeline import (
    apply_markov_smoothing,
    candidate_event_rate,
    default_transition_matrix,
    explain_accelerometer_model_scalars,
    heuristic_shap_attributions,
    heuristic_scores,
    make_synthetic_demo,
    predict_with_accelerometer_checkpoint,
    preprocess_accelerometer,
)


st.set_page_config(
    page_title="SudoSleep | Wearable Prototype",
    page_icon="🌙",
    layout="wide",
)


@st.cache_data(show_spinner=False)
def cached_siegen_segment(archive_path: str, record: str, start: float, duration: float,
                          archive_mtime_ns: int) -> pd.DataFrame:
    return load_segment(Path(archive_path), record, start, duration)


@st.cache_data(show_spinner=False)
def cached_preprocess(frame: pd.DataFrame, sample_rate: float, sensor_rate: float | None) -> dict:
    return preprocess_accelerometer(frame, sample_rate_hz=sample_rate,
                                    original_sensor_rate_hz=sensor_rate)


@st.cache_data(show_spinner=False)
def cached_dreamt_segment(path: str, start: float, duration: float, file_mtime_ns: int) -> pd.DataFrame:
    return load_dreamt_segment(Path(path), start, duration)


def signal_figure(summary: dict, downsample_limit: int = 12000) -> go.Figure:
    raw = np.asarray(summary["samples_model_fs"])
    resp = np.asarray(summary["resp_signal"])
    cardiac = np.asarray(summary["cardiac_proxy"])
    stride = max(1, int(np.ceil(len(raw) / downsample_limit)))
    seconds = np.arange(0, len(raw), stride) / 100.0

    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.06,
                        subplot_titles=("Motion magnitude", "Respiratory-band proxy",
                                        "EMD-derived SCG motion proxy"))
    fig.add_trace(go.Scattergl(x=seconds, y=raw[::stride], name="Motion",
                               line={"width": 0.8}), row=1, col=1)
    fig.add_trace(go.Scattergl(x=seconds, y=resp[::stride], name="Respiration",
                               line={"width": 0.8, "color": "#159895"}), row=2, col=1)
    fig.add_trace(go.Scattergl(x=seconds, y=cardiac[::stride], name="Cardiac-band proxy",
                               line={"width": 0.7, "color": "#9b5de5"}), row=3, col=1)
    fig.update_xaxes(title_text="Seconds", row=3, col=1)
    fig.update_layout(height=650, margin={"l": 10, "r": 10, "t": 60, "b": 10},
                      showlegend=False)
    return fig


def spectrogram_figure(resp: np.ndarray, fs: int = 100) -> go.Figure:
    sample = np.asarray(resp[: min(len(resp), fs * 300)], dtype=np.float64)
    if len(sample) < 256:
        return go.Figure().update_layout(title="Not enough samples for a spectrogram")
    freq, times, power = spectrogram(sample, fs=fs, nperseg=512, noverlap=384)
    keep = freq <= 2.0
    z = 10.0 * np.log10(power[keep] + 1e-12)
    fig = go.Figure(go.Heatmap(
        x=times, y=freq[keep], z=z,
        colorscale="Viridis", colorbar={"title": "dB"},
    ))
    fig.update_layout(title="Respiratory-proxy spectrogram (first 5 minutes)",
                      xaxis_title="Seconds", yaxis_title="Frequency (Hz)",
                      height=380, margin={"l": 10, "r": 10, "t": 55, "b": 10})
    return fig


def build_epoch_table(data: dict, scores: np.ndarray, contributions: np.ndarray,
                      states: np.ndarray, raw_threshold: float) -> pd.DataFrame:
    records = np.asarray(data["records"]).astype(str)
    lengths = np.asarray(data["valid_seconds"], dtype=float)
    rows = []
    elapsed_by_record: dict[str, float] = {}
    for i, record in enumerate(records):
        start = elapsed_by_record.get(record, 0.0)
        duration = float(lengths[i])
        rows.append({
            "record": record,
            "minute": int(round(start / 60.0)) + 1,
            "start_seconds": round(start, 1),
            "window_seconds": round(duration, 1),
            "candidate_score": float(scores[i]),
            "raw_candidate": bool(scores[i] >= raw_threshold),
            "markov_state": "candidate" if states[i] == 1 else "normal-like",
            "low_amplitude_driver": float(contributions[i, 0]),
            "low_variability_driver": float(contributions[i, 1]),
            "pause_driver": float(contributions[i, 2]),
        })
        elapsed_by_record[record] = start + duration
    return pd.DataFrame(rows)


st.title("SudoSleep")
st.caption("Accelerometer respiratory-pattern prototype")
st.warning(
    "Experimental research demo. Motion-derived proxies are not ECG, and the "
    "candidate-event rate is not a clinical AHI or diagnosis."
)

with st.sidebar:
    st.header("Recording")
    dreamt_available = DREAMT_DIRECTORY.exists() and any(DREAMT_DIRECTORY.rglob("S*_df.csv"))
    source_choice = st.radio(
        "Data source", ["Siegen sleep study", "DREAMT CSV", "Upload CSV", "Synthetic demo"],
        index=1 if dreamt_available else (0 if DEFAULT_ARCHIVE.exists() else 2),
    )
    uploaded = st.file_uploader("Upload accelerometer CSV", type=["csv"]) if source_choice == "Upload CSV" else None
    selected_study_record = None
    selected_dreamt_file = None
    segment_start = 0
    segment_minutes = 2
    if source_choice == "Siegen sleep study":
        if DEFAULT_ARCHIVE.is_file():
            try:
                study_records = archive_records(DEFAULT_ARCHIVE)
                selected_study_record = st.selectbox("Study recording", study_records)
                study_info = recording_info(DEFAULT_ARCHIVE, selected_study_record)
                max_start = max(0, int(study_info["duration_seconds"] // 60) - 1)
                initial_start = min(max_start, int(study_info["first_sleep_seconds"] // 60))
                segment_start = st.number_input("Start minute", min_value=0, max_value=max_start,
                                                value=initial_start, step=1)
                max_minutes = max(1, min(30, int((study_info["duration_seconds"] - 60 * segment_start) // 60)))
                segment_minutes = st.number_input("Minutes to analyze", min_value=1, max_value=max_minutes,
                                                  value=min(2, max_minutes), step=1)
                st.caption("Select a short segment first: EMD can be slow for long recordings.")
            except Exception as exc:
                st.error(f"Could not read the study archive: {exc}")
                st.stop()
        else:
            st.info("Download the study with: python siegen_dataset.py --download")
    if source_choice == "DREAMT CSV":
        files = sorted(DREAMT_DIRECTORY.rglob("S*_df.csv")) if DREAMT_DIRECTORY.exists() else []
        if files:
            selected_dreamt_file = st.selectbox("DREAMT recording", files,
                                                format_func=lambda path: str(path.relative_to(DREAMT_DIRECTORY)))
            try:
                first_minute = min(1440, int(np.ceil(recording_start(selected_dreamt_file) / 60)))
            except Exception as exc:
                st.error(f"Could not read the DREAMT recording: {exc}")
                st.stop()
            segment_start = st.number_input("Start minute", min_value=0, max_value=1440, value=first_minute, step=1)
            segment_minutes = st.number_input("Minutes to analyze", min_value=1, max_value=30, value=2, step=1)
        else:
            st.info("Put your authorized DREAMT CSV files under data/dreamt/.")
    fallback_hz = st.number_input(
        "Sampling rate when timestamps are absent (Hz)",
        min_value=8.0, max_value=1000.0, value=50.0, step=1.0,
        help="A timestamp column takes precedence when it can be parsed.",
    )
    original_rate = ORIGINAL_ACCEL_HZ if source_choice == "DREAMT CSV" else None
    if source_choice == "Upload CSV" and st.checkbox("File was resampled; specify original sensor rate"):
        original_rate = st.number_input("Original accelerometer sampling rate (Hz)",
                                        min_value=8.0, max_value=1000.0, value=32.0, step=1.0,
                                        help="For DREAMT E4 acceleration, use 32 Hz even for the resampled 64/100 Hz files.")
    dreamt_checkpoint = DREAMT_DIRECTORY / "DREAMT/derived/dreamt_accelerometer.pt"
    checkpoint_text = st.text_input(
        "Accelerometer-trained model checkpoint",
        value=str(dreamt_checkpoint) if dreamt_checkpoint.exists() else "models/sudo_sleep_accelerometer.pt",
        help="Only checkpoints created by train_wearable.py are accepted.",
    )
    normal_stay = st.slider("Baseline normal-state persistence", 0.50, 0.99, 0.90, 0.01)
    event_stay = st.slider("Baseline candidate-state persistence", 0.50, 0.99, 0.85, 0.01)
    raw_threshold = st.slider("Raw candidate-score threshold", 0.05, 0.95, 0.35, 0.05)
    st.caption("CSV axes: x/y/z, acc_x/acc_y/acc_z, or equivalent.")

st.markdown(
    "**Processing path:** accelerometer → motion cleanup and 60-second windows → "
    "respiratory and cardiac-band motion proxies → trained CNN/Transformer "
    "when a compatible checkpoint is available (otherwise a labeled heuristic) "
    "→ Markov/Viterbi sequence smoothing → candidate-event summary → feature drivers."
)

if source_choice == "Siegen sleep study" and selected_study_record is None:
    st.info("The Siegen dataset archive is not available locally. See the sidebar for the download command.")
    st.stop()

if source_choice == "DREAMT CSV" and selected_dreamt_file is None:
    st.info("DREAMT requires approved PhysioNet access. After downloading, place the CSV files in data/dreamt/.")
    st.stop()

if source_choice == "Upload CSV" and uploaded is None:
    st.info("Upload a CSV to inspect motion signals and candidate respiratory windows.")
    with st.expander("What the model can and cannot do"):
        st.write(
            "The CNN plus two Transformer stages are implemented in pipeline.py. "
            "The historical notebook trained on PhysioNet ECG, which does not make "
            "that checkpoint suitable for accelerometer data. The app uses the "
            "deep model only when you train an accelerometer checkpoint with "
            "labeled wearable recordings. Otherwise it displays a transparent "
            "signal heuristic so the dashboard remains usable without claiming "
            "validated apnea detection."
        )
    st.stop()

try:
    if source_choice == "Siegen sleep study":
        with st.spinner("Reading and expanding the selected study segment..."):
            source_frame = cached_siegen_segment(
                str(DEFAULT_ARCHIVE), selected_study_record,
                float(segment_start) * 60, float(segment_minutes) * 60,
                DEFAULT_ARCHIVE.stat().st_mtime_ns,
            )
        st.info(
            "The Siegen archive provides PSG sleep stages, not respiratory-event labels. "
            "This segment can demonstrate signal processing but cannot validate apnea detection or AHI. "
            "Sensor values are uncalibrated counts."
        )
    elif source_choice == "DREAMT CSV":
        with st.spinner("Reading the selected DREAMT segment..."):
            source_frame = cached_dreamt_segment(
                str(selected_dreamt_file), float(segment_start) * 60,
                float(segment_minutes) * 60, selected_dreamt_file.stat().st_mtime_ns,
            )
        st.info("DREAMT acceleration was sampled at 32 Hz before resampling. PSG sleep stages are not apnea-event labels.")
    elif source_choice == "Upload CSV":
        uploaded.seek(0)
        source_frame = pd.read_csv(uploaded)
    else:
        source_frame = make_synthetic_demo()
        st.info("Showing a synthetic example only; these are generated signals, not recorded patient data.")
    with st.spinner("Extracting signal features and EMD motion proxy..."):
        data = cached_preprocess(source_frame, float(fallback_hz),
                                 float(original_rate) if original_rate is not None else None)
except Exception as exc:
    st.error(f"Could not process this recording: {exc}")
    st.stop()

checkpoint_path = Path(checkpoint_text).expanduser()
backend = "Signal heuristic"
transition = default_transition_matrix(normal_stay, event_stay)
model_note = "No accelerometer-trained checkpoint found; using the experimental heuristic."
baseline_scores, contributions = heuristic_scores(data)
explanation_data = data
explanation_indices = np.arange(len(data["signals_3ch"]))

if checkpoint_path.is_file():
    try:
        if source_choice == "DREAMT CSV":
            subject = selected_dreamt_file.name.split("_")[0]
            cache_path = DREAMT_DIRECTORY / "DREAMT/derived/features" / f"{subject}.npz"
            if not cache_path.is_file():
                raise ValueError("The full-night feature cache is needed for consistent model normalization.")
            with np.load(cache_path, allow_pickle=False) as cached:
                signals, scalars = cached["signals"], cached["scalars"]
                version = str(cached["preprocessing_version"])
            first = int(segment_start)
            indices = np.arange(first, first + len(data["signals_3ch"]))
            if np.any(indices >= len(signals)):
                raise ValueError("The selected segment extends beyond the full-night model cache.")
            whole_night = {"signals_3ch": signals, "scalars": scalars,
                           "labels": np.zeros(len(signals)), "records": np.full(len(signals), subject),
                           "preprocessing_version": version}
            scores, transition = predict_with_accelerometer_checkpoint(whole_night, checkpoint_path, indices=indices)
            explanation_data, explanation_indices = whole_night, indices
        else:
            scores, transition = predict_with_accelerometer_checkpoint(data, checkpoint_path)
        backend = "Accelerometer CNN + Transformer ensemble"
        model_note = f"Loaded accelerometer checkpoint: {checkpoint_path}"
    except Exception as exc:
        scores = baseline_scores
        model_note = f"Checkpoint could not be used ({exc}); using the heuristic."
else:
    scores = baseline_scores

records = np.asarray(data["records"]).astype(str)
states = apply_markov_smoothing(scores, records, transition)
event_runs, total_hours, events_per_hour = candidate_event_rate(
    states, records, np.asarray(data["valid_seconds"])
)
table = build_epoch_table(data, scores, contributions, states, float(raw_threshold))
if source_choice == "Siegen sleep study":
    minute_stages = []
    for _, row in table.iterrows():
        start = float(row["start_seconds"])
        stop = start + float(row["window_seconds"])
        codes = source_frame.loc[(source_frame["time_s"] >= start) & (source_frame["time_s"] < stop),
                                 "sleep_stage_code"]
        minute_stages.append(STAGE_NAMES.get(int(codes.value_counts().index[0]), "Unknown") if len(codes) else "Unknown")
    table["psg_sleep_stage_majority"] = minute_stages
    staged_sleep_hours = float(source_frame["sleep_stage_code"].isin([1, 2, 3, 5]).sum() / 100.0 / 3600.0)
else:
    estimated_sleep_hours = st.number_input(
        "Estimated sleep hours covered by these recordings (0 if unknown)",
        min_value=0.0, max_value=24.0, value=0.0, step=0.05, format="%.3f",
        help="Enter actual estimated sleep time, not just the length of the recording.",
    )

st.caption(model_note)
unavailable = [str(s["record"]) for s in data["record_summaries"]
               if not s["cardiac_available"]]
if unavailable:
    st.info("Cardiac-band proxy unavailable at the original sensor rate for: " + ", ".join(unavailable))
limited_band = [s for s in data["record_summaries"] if s["cardiac_available"] and s["original_sensor_rate_hz"] < 62.5]
if limited_band:
    st.info("The SCG frequency band is capped below the original sensor Nyquist frequency; the full 10–30 Hz band is unavailable at these rates.")
methods = sorted({str(s["cardiac_method"]) for s in data["record_summaries"]
                 if s["cardiac_available"]})
if methods and methods != ["EMD IMF reconstruction"]:
    st.info("SCG proxy method in use: " + ", ".join(methods) + ".")
empty_proxy = [str(s["record"]) for s in data["record_summaries"]
               if s["cardiac_available"] and not s["cardiac_nonzero"]]
if empty_proxy:
    st.info("No cardiac-band components were selected for: " + ", ".join(empty_proxy))

raw_candidates = int(np.sum(scores >= raw_threshold))
candidate_windows = int(np.sum(states == 1))
left, middle, right = st.columns(3)
left.metric("Raw candidate windows", raw_candidates)
middle.metric("Smoothed candidate runs", event_runs)
if source_choice == "Siegen sleep study":
    right.metric("PSG-staged sleep in segment", f"{staged_sleep_hours * 60:.1f} min")
elif estimated_sleep_hours > 0:
    right.metric("Candidate runs per estimated sleep hour", f"{event_runs / estimated_sleep_hours:.1f}")
else:
    right.metric("Candidate runs per estimated sleep hour", "—")
st.caption(f"Analyzed {total_hours:.2f} recorded hours using **{backend}**.")
st.caption(
    "Candidate runs are exploratory signal patterns, not scored respiratory events. "
    "A candidate rate divided by sleep time is not validated AHI."
)

with st.expander("Model training and held-out evaluation"):
    report_path = DREAMT_DIRECTORY / "DREAMT/derived/dreamt_results.json"
    if report_path.is_file():
        try:
            report = json.loads(report_path.read_text())
            st.warning(report.get("label_caveat", "Check label provenance before interpreting results."))
            st.caption(report.get("note", ""))
            st.write("CNN/Transformer held-out test:", report.get("test", {}))
            st.write("Random-forest baseline held-out test:", report.get("baseline_test", {}))
            st.write("Coarse candidate-run rate MAE versus PSG AHI:", report.get("test_candidate_rate_mae_vs_ahi"))
            history = pd.DataFrame(report.get("training_history", []))
            if not history.empty:
                st.line_chart(history.set_index("epoch")[["validation_auprc"]])
        except Exception as exc:
            st.error(f"Could not read the local evaluation report: {exc}")
    else:
        st.info("No completed DREAMT training report is available yet.")
        status_path = DREAMT_DIRECTORY / "DREAMT/derived/overnight_status.json"
        if status_path.is_file():
            try:
                status = json.loads(status_path.read_text())
                count = len(list((status_path.parent / "features").glob("*.npz")))
                st.caption(f"Local run: {status.get('stage', 'unknown')} · {count} subject caches prepared · "
                           f"last update {status.get('updated_utc', 'unknown')}")
                if status.get("error"):
                    st.error(status["error"])
            except (OSError, ValueError):
                st.caption("Local run status is temporarily unavailable.")

record_names = [str(s["record"]) for s in data["record_summaries"]]
selected_record = st.selectbox("Recording to inspect", record_names)
summary = next(s for s in data["record_summaries"] if str(s["record"]) == selected_record)
if source_choice == "Siegen sleep study" or (source_choice == "DREAMT CSV" and "psg_sleep_stage" in source_frame):
    stage_column = "sleep_stage_code" if source_choice == "Siegen sleep study" else "psg_sleep_stage"
    stages = source_frame[["time_s", stage_column]].copy()
    stages["period"] = (stages["time_s"] // 30).astype(int)
    stage_summary = stages.groupby("period", sort=True)[stage_column].agg(
        lambda values: values.dropna().value_counts().index[0] if values.notna().any() else "Unknown"
    ).reset_index()
    if source_choice == "Siegen sleep study":
        stage_summary["stage"] = stage_summary[stage_column].map(STAGE_NAMES).fillna("Unknown")
    else:
        stage_summary["stage"] = stage_summary[stage_column].astype(str).replace({"R": "REM"})
    stage_fig = go.Figure(go.Scatter(
        x=stage_summary["period"] * 0.5, y=stage_summary["stage"],
        mode="lines+markers", line_shape="hv", name="PSG sleep stage",
    ))
    stage_fig.update_layout(title="PSG sleep stages (30-second summaries)",
                            xaxis_title="Minutes into selected segment", yaxis_title="Sleep stage",
                            height=280, margin={"l": 10, "r": 10, "t": 55, "b": 10})
    st.plotly_chart(stage_fig, width="stretch")
st.plotly_chart(signal_figure(summary), width="stretch")
st.plotly_chart(spectrogram_figure(summary["resp_signal"]), width="stretch")

if source_choice == "DREAMT CSV":
    event_columns = {"psg_obstructive_apnea": "Obstructive apnea", "psg_central_apnea": "Central apnea",
                     "psg_hypopnea": "Hypopnea", "psg_multiple_events": "Multiple events",
                     "psg_mixed_apnea": "Mixed apnea"}
    event_chart = go.Figure()
    for column, label in event_columns.items():
        if column in source_frame:
            marked = pd.to_numeric(source_frame[column], errors="coerce").eq(1)
            seconds = np.unique(np.floor(source_frame.loc[marked, "time_s"]).astype(int))
            event_chart.add_trace(go.Scatter(x=seconds / 60.0, y=[label] * len(seconds),
                                             mode="markers", marker={"size": 5}, name=label))
    event_chart.update_layout(title="PSG-marked respiratory events", xaxis_title="Minutes into segment",
                              height=280, showlegend=False, margin={"l": 10, "r": 10, "t": 55, "b": 10})
    st.plotly_chart(event_chart, width="stretch")
    st.caption("Only timestamps marked 1 are shown. Blank event fields are treated as provisional unmarked intervals in the research audit.")

record_table = table[table["record"] == selected_record].reset_index(drop=True)
record_indices = np.flatnonzero(records == selected_record)
sequence = go.Figure()
sequence.add_trace(go.Scatter(
    x=record_table["minute"], y=record_table["candidate_score"],
    mode="lines+markers", name="Candidate score", line={"color": "#159895"},
))
sequence.add_trace(go.Scatter(
    x=record_table.loc[states[record_indices] == 1, "minute"],
    y=record_table.loc[states[record_indices] == 1, "candidate_score"],
    mode="markers", name="Markov state: candidate",
    marker={"color": "#ef476f", "size": 9, "symbol": "diamond"},
))
sequence.add_hline(y=float(raw_threshold), line_dash="dash", line_color="gray")
sequence.update_layout(title="Per-minute candidate sequence", xaxis_title="Minute",
                       yaxis_title="Score (uncalibrated 0–1)", height=350,
                       margin={"l": 10, "r": 10, "t": 55, "b": 10})
st.plotly_chart(sequence, width="stretch")

st.subheader("SHAP explanation of the baseline")
st.caption(
    "SHAP attributes the transparent heuristic score to respiratory amplitude, "
    "variability, and pause fraction. It does not explain the CNN/Transformer "
    "prediction when a trained model is selected."
)
selected_minute = st.selectbox(
    "Minute to explain", record_table["minute"].tolist(),
    index=int(np.argmax(scores[record_indices])),
)
selected_position = int(record_table.index[record_table["minute"] == selected_minute][0])
selected_row_index = int(record_indices[selected_position])
try:
    shap_names, shap_values, shap_base, shap_score = heuristic_shap_attributions(
        data, selected_row_index
    )
    shap_chart = go.Figure(go.Bar(
        x=shap_values,
        y=shap_names,
        orientation="h",
        marker_color=["#ef476f" if value >= 0 else "#159895" for value in shap_values],
    ))
    shap_chart.add_vline(x=0, line_color="gray", line_width=1)
    shap_chart.update_layout(
        xaxis_title="SHAP value (effect on baseline score)", height=280,
        margin={"l": 10, "r": 10, "t": 20, "b": 10},
    )
    st.plotly_chart(shap_chart, width="stretch")
    st.caption(
        f"Baseline score: {shap_score:.3f} = expected score {shap_base:.3f} "
        f"+ SHAP contributions."
    )
except Exception as exc:
    st.info(f"SHAP is unavailable ({exc}); showing baseline drivers instead.")
    driver_names = ["Low respiratory amplitude", "Low respiratory variability", "Pause fraction"]
    driver_values = contributions[selected_row_index]
    drivers = go.Figure(go.Bar(
        x=driver_values, y=driver_names, orientation="h",
        marker_color=["#ef476f", "#f4a261", "#457b9d"],
    ))
    drivers.update_layout(xaxis_title="Contribution to heuristic score", height=260,
                          margin={"l": 10, "r": 10, "t": 20, "b": 10})
    st.plotly_chart(drivers, width="stretch")

if backend == "Accelerometer CNN + Transformer ensemble":
    st.subheader("CNN/Transformer scalar explanation")
    st.caption("Approximate Kernel SHAP changes the selected minute's scalar features relative to the training median while holding the signal and other context fixed. It does not explain waveform contributions.")
    if st.button("Explain selected model prediction"):
        try:
            with st.spinner("Calculating model scalar attributions..."):
                names, values, expected, predicted = explain_accelerometer_model_scalars(
                    explanation_data, checkpoint_path, int(explanation_indices[selected_row_index])
                )
            order = np.argsort(np.abs(values))
            model_shap = go.Figure(go.Bar(
                x=values[order], y=np.asarray(names)[order], orientation="h",
                marker_color=["#ef476f" if value >= 0 else "#159895" for value in values[order]],
            ))
            model_shap.update_layout(xaxis_title="Conditional SHAP effect on model probability", height=420)
            st.plotly_chart(model_shap, width="stretch")
            st.caption(f"Model probability {predicted:.3f}; background probability {expected:.3f}.")
        except Exception as exc:
            st.error(f"Could not explain the model prediction: {exc}")

with st.expander("Minute-level output"):
    st.dataframe(record_table, width="stretch", hide_index=True)
st.download_button(
    "Download all recording results as CSV",
    data=table.to_csv(index=False).encode("utf-8"),
    file_name="sudosleep_candidate_windows.csv",
    mime="text/csv",
)

with st.expander("How sequence smoothing works"):
    st.write(
        "The app decodes the highest-scoring sequence under a two-state Markov "
        "model using Viterbi. Without a trained checkpoint, the sidebar values "
        "set the heuristic's persistence prior; a "
        "trained accelerometer checkpoint carries transition probabilities "
        "estimated from its labeled training recordings."
    )
    st.code(np.array2string(transition, precision=3), language="text")
