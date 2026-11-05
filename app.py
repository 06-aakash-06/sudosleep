"""Streamlit demo for accelerometer-derived respiratory signal exploration."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots
from scipy.signal import spectrogram

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
    uploaded = st.file_uploader("Upload accelerometer CSV", type=["csv"])
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

if uploaded is None and not st.session_state.get("synthetic_demo", False):
    st.info("Upload a CSV to inspect motion signals and candidate respiratory windows.")
    if st.button("Run the synthetic demonstration"):
        st.session_state["synthetic_demo"] = True
        st.rerun()
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
    if uploaded is not None:
        st.session_state["synthetic_demo"] = False
        uploaded.seek(0)
        source_frame = pd.read_csv(uploaded)
    else:
        source_frame = make_synthetic_demo()
        st.info("Showing a synthetic example only; these are generated signals, not recorded patient data.")
    data = preprocess_accelerometer(source_frame, sample_rate_hz=float(fallback_hz))
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
default_sleep_hours = min(max(total_hours, 0.05), 24.0)
estimated_sleep_hours = st.number_input(
    "Estimated sleep hours covered by these recordings",
    min_value=0.05, max_value=24.0, value=default_sleep_hours, step=0.25,
    help="The default assumes all recorded time was sleep. Adjust it if you know the actual sleep duration.",
)
candidate_runs_per_sleep_hour = event_runs / max(float(estimated_sleep_hours), 1e-8)
table = build_epoch_table(data, scores, contributions, states, float(raw_threshold))

st.caption(model_note)
unavailable = [str(s["record"]) for s in data["record_summaries"]
               if not s["cardiac_available"]]
if unavailable:
    st.info("Cardiac-band proxy omitted below 40 Hz for: " + ", ".join(unavailable))
methods = sorted({str(s["cardiac_method"]) for s in data["record_summaries"]
                 if s["cardiac_available"]})
if methods and methods != ["EMD IMF reconstruction"]:
    st.info("SCG proxy method in use: " + ", ".join(methods) + ".")

raw_candidates = int(np.sum(scores >= raw_threshold))
candidate_windows = int(np.sum(states == 1))
left, middle, right = st.columns(3)
left.metric("Raw candidate windows", raw_candidates)
middle.metric("Smoothed candidate runs", event_runs)
right.metric("Candidate runs per estimated sleep hour", f"{candidate_runs_per_sleep_hour:.1f}")
st.caption(f"Analyzed {total_hours:.2f} recorded hours using **{backend}**.")
st.caption(
    "This is an experimental AHI-like rate: smoothed candidate runs divided by "
    "the sleep duration you entered. It is not clinical AHI because the event "
    "detector and sleep-time estimate are not validated."
)

record_names = [str(s["record"]) for s in data["record_summaries"]]
selected_record = st.selectbox("Recording to inspect", record_names)
summary = next(s for s in data["record_summaries"] if str(s["record"]) == selected_record)
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
