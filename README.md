# SudoSleep

SudoSleep is a hackathon prototype for exploring respiratory patterns in
consumer-wearable accelerometer recordings. It includes an uploadable
Streamlit dashboard, motion-derived respiratory and cardiac-band proxies, a
transparent candidate-event baseline, sequence smoothing, and an optional
CNN/Transformer model path for a model trained on labeled accelerometer data.

The project work period was January–June 2026, as supplied by the project
author. The files currently present include March 2026 ECG benchmark notebooks
and a consolidated pipeline; this repository does not invent separate
month-by-month milestones.

## What runs in the app

The app accepts a CSV with three accelerometer axes. It derives a respiratory
band signal and, when the source sampling rate is high enough, a cardiac-band
mechanical proxy. It divides the recording into one-minute windows, displays
the signals and a respiratory spectrogram, scores candidate low-respiration
windows, smooths the sequence with a two-state Viterbi/Markov model, and shows
SHAP values for the baseline score.

The current baseline is a signal heuristic. The existing 1D CNN and Transformer
architecture is in `pipeline.py`; the saved notebook trained it on PhysioNet
ECG, not accelerometer data. The app only loads a deep model when a checkpoint
was trained for accelerometer input using `train_wearable.py`. No such trained
checkpoint is included. Without one, the app clearly labels results as
experimental candidate scores.

The dashboard's hourly candidate rate is an exploratory index. It is not a
clinical AHI, diagnosis, or validated apnea event count. Accelerometer proxy
signals are not ECG reconstructions. SHAP explains the transparent heuristic
score only; it does not explain the optional neural model's predictions.

## Run locally

Python 3.11–3.13 is recommended for the broadest PyTorch wheel support.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
streamlit run app.py
```

Upload a CSV containing columns named `x`, `y`, and `z` (or common equivalents
such as `acc_x`, `acc_y`, `acc_z`). A timestamp column is optional; when it is
absent, enter the sensor's sampling rate in the sidebar. The app expects
roughly uniform sampling and at least 30 seconds of usable signal. Sampling
rates below 40 Hz do not produce a cardiac-band proxy.

## Train an accelerometer model

The training CSV must contain three axis columns, a `label` column with a
minute-level binary apnea label repeated for samples in that minute, and a
`record` column identifying independent recordings or subjects. Supply at
least five independent records for the default subject-grouped five-fold
training.

```bash
python train_wearable.py labeled_accelerometer.csv \
  --sample-rate 50 \
  --output models/sudo_sleep_accelerometer.pt
```

Training can take a while and requires PyTorch. The resulting checkpoint is
ignored by Git; keep it locally or store it in an appropriate model-artifact
service. Do not use an ECG-trained checkpoint for accelerometer recordings.

## ECG benchmark data

The original notebook uses PhysioNet's Apnea-ECG database. The data are not
committed because the download is about 580 MB and should be fetched by each
user. Download it from [PhysioNet Apnea-ECG](https://physionet.org/content/apnea-ecg/1.0.0/challenge/)
and place the records in `data/`. Cite the database authors and comply with
its Open Data Commons Attribution License 1.0 when redistributing derived
data.

The notebook's saved output is historical: it reports 10 scalar features,
while the current `pipeline.py` uses 12. Its full-dataset classification
report is not a held-out test result. Treat the recorded out-of-fold AUC as a
result from that earlier code snapshot, not as validation of the current
accelerometer path.

## Repository contents

- `app.py`: Streamlit upload and visualization interface.
- `wearable_pipeline.py`: accelerometer preprocessing, proxy extraction,
  heuristic scoring, sequence smoothing, and optional compatible model loading.
- `train_wearable.py`: subject-grouped training for labeled accelerometer CSVs.
- `pipeline.py`: original ECG research pipeline and CNN/Transformer model.
- `pipelinefinal.ipynb`, `pipeline_v2.ipynb`: historical experiment notebooks.
