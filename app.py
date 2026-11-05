"""Streamlit demo for accelerometer-derived respiratory signal exploration."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots
from scipy.signal import spectrogram
from siegen_dataset import DEFAULT_ARCHIVE, STAGE_NAMES, archive_records, load_segment, recording_info

from wearable_pipeline import (
    apply_markov_smoothing,
    candidate_event_rate,
    default_transition_matrix,
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
def cached_preprocess(frame: pd.DataFrame, sample_rate: float) -> dict:
    return preprocess_accelerometer(frame, sample_rate_hz=sample_rate)


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
    source_choice = st.radio(
        "Data source", ["Siegen sleep study", "Upload CSV", "Synthetic demo"],
        index=0 if DEFAULT_ARCHIVE.exists() else 1,
    )
    uploaded = st.file_uploader("Upload accelerometer CSV", type=["csv"]) if source_choice == "Upload CSV" else None
    selected_study_record = None
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
    fallback_hz = st.number_input(
        "Sampling rate when timestamps are absent (Hz)",
        min_value=8.0, max_value=1000.0, value=50.0, step=1.0,
        help="A timestamp column takes precedence when it can be parsed.",
    )
    checkpoint_text = st.text_input(
        "Accelerometer-trained model checkpoint",
        value="models/sudo_sleep_accelerometer.pt",
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
    elif source_choice == "Upload CSV":
        uploaded.seek(0)
        source_frame = pd.read_csv(uploaded)
    else:
        source_frame = make_synthetic_demo()
        st.info("Showing a synthetic example only; these are generated signals, not recorded patient data.")
    with st.spinner("Extracting signal features and EMD motion proxy..."):
        data = cached_preprocess(source_frame, float(fallback_hz))
except Exception as exc:
    st.error(f"Could not process this recording: {exc}")
    st.stop()

checkpoint_path = Path(checkpoint_text).expanduser()
backend = "Signal heuristic"
transition = default_transition_matrix(normal_stay, event_stay)
model_note = "No accelerometer-trained checkpoint found; using the experimental heuristic."
baseline_scores, contributions = heuristic_scores(data)

if checkpoint_path.is_file():
    try:
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
    st.info("Cardiac-band proxy omitted below 40 Hz for: " + ", ".join(unavailable))
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

record_names = [str(s["record"]) for s in data["record_summaries"]]
selected_record = st.selectbox("Recording to inspect", record_names)
summary = next(s for s in data["record_summaries"] if str(s["record"]) == selected_record)
if source_choice == "Siegen sleep study":
    stages = source_frame[["time_s", "sleep_stage_code"]].copy()
    stages["period"] = (stages["time_s"] // 30).astype(int)
    stage_summary = stages.groupby("period", sort=True)["sleep_stage_code"].agg(
        lambda values: values.value_counts().index[0]
    ).reset_index()
    stage_summary["stage"] = stage_summary["sleep_stage_code"].map(STAGE_NAMES).fillna("Unknown")
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
sequence.add_hline(y=0.5, line_dash="dash", line_color="gray")
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
    index=int(np.argmax(baseline_scores[record_indices])),
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
