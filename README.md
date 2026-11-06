# SudoSleep

SudoSleep is a hackathon prototype for exploring respiratory patterns in
consumer-wearable accelerometer recordings. It includes an uploadable
React research workspace and legacy Streamlit dashboard, motion-derived respiratory and cardiac-band proxies, a
transparent candidate-event baseline, sequence smoothing, and an optional
CNN/Transformer model path for a model trained on labeled accelerometer data.

The project work period was January–June 2026, as supplied by the project
author. The files currently present include March 2026 ECG benchmark notebooks
and a consolidated pipeline; this repository does not invent separate
month-by-month milestones.

## Research workspace (React + Python)

The primary interface is a local React/TypeScript workspace backed by the
existing Python signal and model pipeline. It has a horizontal research index,
a whole-night target strip, linked signal cursors and zoom, PSG stage/event
tracks, selected-segment predictions and exports, conditional neural scalar
SHAP, and a separate held-out model study. Evaluation reads saved artifacts;
it does not retrain the model when a page opens.

```bash
# Install Python requirements into your existing environment first.
python -m pip install -r requirements.txt
npm ci --prefix web
npm run build --prefix web
./run_workspace.sh
```

Open http://127.0.0.1:8000. For frontend development, keep the API on port 8000
and run `npm run dev --prefix web`; Vite forwards `/api` to the local API.
The authorized DREAMT files, prepared caches, checkpoint, splits and evaluation
artifacts must already exist under `data/dreamt/DREAMT/derived/` for the DREAMT
workflow. Missing artifacts produce explicit errors. The UI also accepts short
CSV uploads and a synthetic signal example through the heuristic path. The
Streamlit app remains available for the Siegen workflow and legacy controls.
The API binds to loopback and serves the built UI; no dataset is published.

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
signals are not ECG reconstructions. SHAP explains the heuristic score. A compatible DREAMT model also supports approximate conditional SHAP for scalar features, holding waveform and other context fixed.

## Proposal pipeline status

The wearable path now includes per-record z-score normalization, EMD-based
10–30 Hz SCG proxy extraction (limited by the original sensor Nyquist rate), a
respiratory proxy, the CNN with squeeze-excitation channel attention and
within-minute plus five-minute Transformer context, Viterbi sequence smoothing,
and a user-entered sleep-time denominator for an experimental candidate rate.
SHAP explains the heuristic and, when a compatible checkpoint exists, conditional scalar contributions to the neural model; it does not explain waveform contributions. The proposal's Cheyne–Stokes constraint and sleep-stage/position/REM classification are not implemented. A DREAMT paired-PSG research evaluation is available once its local training run completes.
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
rates must support frequencies above 10 Hz to produce a cardiac-band proxy.
For resampled input, specify the original sensor rate in the sidebar. DREAMT's
accelerometer was recorded at 32 Hz even when the distributed file is at 100 Hz;
its usable proxy band is therefore limited to approximately 10–15.4 Hz.

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

For DREAMT, obtain approved access and download version 2.2 from the
[official PhysioNet page](https://physionet.org/content/dreamt/2.2.0/).
Place authorized CSV files anywhere under `data/dreamt/`, then choose **DREAMT
CSV** in the app. The loader streams only the selected time range and the
timestamp, three accelerometer axes, and optional PSG sleep stage. It rejects
segments with timestamp gaps rather than treating them as continuous. It keeps
the original 32 Hz sensor rate when processing the distributed resampled CSVs.
The local audit found sparse event fields encoded as 1 or blank. Treating blank as no marked event during scored sleep is a provisional interpretation: the supplied metadata does not explicitly define blanks. Observed marked-event rates were within 2 events/hour of the supplied PSG AHI for 98 of 100 subjects; the two discrepancies are excluded from the initial split. PSG sleep-stage labels are not respiratory-event labels. DREAMT is restricted data and must stay local under its data-use agreement.

## DREAMT research workflow

With the authorized files under `data/dreamt/DREAMT/`, run:

```bash
python audit_dreamt.py --workers 3
python prepare_dreamt.py --split --prepare --workers 3
python train_dreamt.py --epochs 8 --batch-size 2 --max-hours 4
streamlit run app.py
```

Preparation is computationally intensive and can take many hours. The fixed split
is by subject (68 train, 15 validation, 15 test, 2 excluded); scalar
normalization and smoothing transitions are learned from training subjects only.
A positive one-minute window requires at least one second of a marked event and
at least 80% PSG-scored sleep. The held-out test is evaluated after model and
threshold selection. Reports include a scalar random-forest baseline, window
classification metrics, coarse event matching and candidate-run rate comparison.
These experimental counts are not validated clinical AHI. The local overnight
supervisor can resume missing caches with `python run_dreamt_overnight.py` and
stores its status and logs under `data/dreamt/DREAMT/derived/`. Dataset files,
subject-level results and trained artifacts are ignored by Git.

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

For a resampled DREAMT file with appropriate event labels, add `--sensor-rate 32`.
The official DREAMT release requires a PhysioNet account and signed data-use
agreement. The local event-label audit and its provisional interpretation are described above. Sleep stages must not be used as apnea labels.

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
- `dreamt_dataset.py`: bounded, rate-aware reading of authorized DREAMT CSVs.
- `audit_dreamt.py`, `prepare_dreamt.py`, `train_dreamt.py`: local label audit, subject split, caching, training and evaluation.
- `run_dreamt_overnight.py`: resumable preparation and training supervisor.
- `pipeline.py`: original ECG research pipeline and CNN/Transformer model.
- `pipelinefinal.ipynb`, `pipeline_v2.ipynb`: historical experiment notebooks.
