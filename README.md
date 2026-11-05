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
band signal and, when the source sampling rate is high enough, an EMD-derived
10–30 Hz seismocardiography (SCG) motion proxy. At lower source rates, the
usable band is capped below the original Nyquist frequency. If EMD is missing,
the dashboard labels its bandpass fallback. Neither signal is reconstructed
ECG. It divides the recording into one-minute windows, displays
the signals and a respiratory spectrogram, scores candidate low-respiration
windows, smooths the sequence with a two-state Viterbi/Markov model, and shows
SHAP values for the baseline score.

The current baseline is a signal heuristic. The existing 1D CNN and Transformer
architecture is in `pipeline.py`; the saved notebook trained it on PhysioNet
ECG, not accelerometer data. The app only loads a deep model when a checkpoint
was trained for accelerometer input using `train_wearable.py`. No such trained
checkpoint is included. Without one, the app clearly labels results as
experimental candidate scores.

The dashboard can divide smoothed candidate runs by the sleep duration entered
by the user to show an experimental AHI-like rate. It is not a clinical AHI,
diagnosis, or validated apnea event count. Accelerometer proxy
signals are not ECG reconstructions. SHAP explains the transparent heuristic
score only; it does not explain the optional neural model's predictions.

## Proposal pipeline status

The wearable path now includes per-record z-score normalization, EMD-based
10–30 Hz SCG proxy extraction (limited by the original sensor Nyquist rate), a
respiratory proxy, the CNN with squeeze-excitation channel attention and
within-minute plus five-minute Transformer context, Viterbi sequence smoothing,
and a user-entered sleep-time denominator for an experimental candidate rate.
SHAP currently explains the heuristic score; it is not used to filter neural
predictions. The proposal's Cheyne–Stokes constraint, sleep-stage/position/REM
classification, and paired accelerometer/PSG validation are not implemented.
The proposal's 16-month schedule and AUC target are plans, not results. The
archived five-model checkpoint was trained in the ECG notebook and is not
compatible with accelerometer input.

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
such as `acc_x`, `acc_y`, `acc_z`, or `accel_x`, `accel_y`, `accel_z`). A timestamp column is optional; when it is
absent, enter the sensor's sampling rate in the sidebar. The app expects
roughly uniform sampling and at least 30 seconds of usable signal. Sampling
rates below 40 Hz do not produce a cardiac-band proxy.

To explore the University of Siegen ICHI 2014 study in the dashboard, download
its archive locally and select **Siegen sleep study** in the sidebar:

```bash
python siegen_dataset.py --download
streamlit run app.py
```

The app reads a selected short segment directly from the ZIP. It expands the
run-length encoding and regularizes the original timestamps to a 100 Hz grid;
the stored axes are uncalibrated integer sensor counts. The archive has 47
recording files from 42 patients, some with multiple recordings. Its PSG labels
describe sleep stages, not apnea events. They are shown separately and never
used as apnea training labels. A zero EMD proxy means no component met the
current experimental selection rule, not that cardiac activity was absent.
The data stay in the ignored `data/` directory. The [dataset authors](https://www.eti.uni-siegen.de/ubicomp/home/datasets/ichi14/)
permit scientific non-commercial use with attribution.

## Train an accelerometer model

The training CSV must contain three axis columns, a `label` column with a
minute-level binary apnea label repeated for samples in that minute, and a
`record` column identifying each recording. For multiple nights from the same
person, also include a `subject` column so every night from that person stays
in the same validation fold. If `subject` is absent, the trainer assumes each
record is a different subject. Supply at least five independent subjects for the
default subject-grouped five-fold training. Sleep-stage codes from the Siegen
study are not apnea labels and cannot be used to train this classifier.

```bash
python train_wearable.py labeled_accelerometer.csv \
  --sample-rate 50 \
  --output models/sudo_sleep_accelerometer.pt
```

Training can take a while and requires PyTorch. The resulting checkpoint is
ignored by Git; keep it locally or store it in an appropriate model-artifact
service. Checkpoints record their preprocessing version and feature names;
older or mismatched checkpoints must be retrained. Do not use an ECG-trained
checkpoint for accelerometer recordings.

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
- `siegen_dataset.py`: safe loading and timestamp reconstruction for the
  run-length encoded Siegen sleep-stage archive.
- `pipeline.py`: original ECG research pipeline and CNN/Transformer model.
- `pipelinefinal.ipynb`, `pipeline_v2.ipynb`: historical experiment notebooks.
