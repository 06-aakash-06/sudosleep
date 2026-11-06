# SudoSleep

A local research workspace for exploring respiratory events in wearable
accelerometer recordings. The dark React interface connects to a Python API,
a CNN/Transformer model and the authorized DREAMT dataset on your computer.

**Start here:** [Dataset access and download](docs/DATASET_SETUP.md) ·
[Reproduce the research](docs/REPRODUCING.md) ·
[Pipeline and limitations](docs/PIPELINE.md) ·
[Contributing research changes](CONTRIBUTING.md)

## What is implemented

```text
Wrist accelerometer (originally 32 Hz)
  → preprocessing and 60-second windows
  → respiratory proxy + EMD cardiac-band motion proxy
  → 1D-CNN + squeeze-excitation + Transformer context
  → minute-level scores
  → Markov/Viterbi smoothing
  → candidate runs and exploratory rate comparison
  → React visualization + conditional scalar SHAP
```

The model uses accelerometer-derived waveforms and scalar features. PSG stages
and respiratory annotations supply reference context and provisional targets;
PSG waveforms are not model inputs. The cardiac proxy is mechanical motion,
not reconstructed ECG. DREAMT's 32 Hz accelerometer limits its usable cardiac
band to approximately 10–15.4 Hz.

## Set up on a teammate's computer

Prerequisites: Python 3.11+ with compatible scientific/PyTorch wheels, Node.js
22+ and npm. This checkout was tested on macOS with Python 3.14.7, Node 26.8.1
and PyTorch 2.10.0. Apple Silicon uses MPS when available; other machines use
the device selected by PyTorch. Runtime and results can vary by hardware.

```bash
git clone https://github.com/06-aakash-06/sudosleep.git
cd sudosleep
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
npm ci --prefix web
npm run build --prefix web
```

**The clone does not include DREAMT, feature caches or trained weights.** Each
teammate obtains their own dataset access, downloads the required files and
follows [the complete setup guide](docs/DATASET_SETUP.md). Then:

```bash
python check_workspace.py --stage raw
python audit_dreamt.py --workers 3
python prepare_dreamt.py --split --prepare --workers 3
python train_dreamt.py --epochs 8 --batch-size 2 --max-hours 4
python check_workspace.py --stage trained
./run_workspace.sh
```

Open [the local workspace](http://127.0.0.1:8000). Full-night EMD preparation can
take many hours; training follows preparation. A dataset download by itself
is not sufficient to reproduce the model views. See [resuming and verifying
the run](docs/REPRODUCING.md). On Windows use the equivalent environment
activation and run `python -m uvicorn research_api:app --host 127.0.0.1 --port 8000`
after the preflight check.

## Inspect the research

| View | What you can examine |
| --- | --- |
| Night desk | Whole-night target strip, selected signal tracks, linked cursors/zoom, spectrogram, PSG annotations, model states and CSV export |
| Model study | Saved held-out metrics, baseline comparison, ROC/PR curves, confusion matrix, event matching, subject rate differences and training history |
| Signal drivers | Raw scalar features and approximate conditional neural SHAP for a selected minute |
| Data & methods | Audit, split, implemented stages, assumptions, synthetic example and short CSV uploads |

The API serves the built interface on loopback. It reads local artifacts and
does not start training when the dashboard opens. Short CSV uploads and the
synthetic example use a separately labelled heuristic. The [legacy
Streamlit workflow](docs/LEGACY.md) retains Siegen exploration and generic
accelerometer training controls.

## Current experiment and limits

One completed local run used 68 training, 15 validation and 15 held-out test
subjects, with two audit outliers excluded. Five epochs were recorded before
validation-based early stopping. On 3,566 eligible test windows:

| Model | AUROC | AUPRC | F1 |
| --- | ---: | ---: | ---: |
| CNN + Transformer | 0.733 | 0.530 | 0.570 |
| Random forest | 0.735 | 0.544 | 0.565 |

The neural model has not demonstrated an advantage over the baseline. Coarse
one-to-one event matching gave about 9.0% recall, and candidate-run rate MAE
versus supplied PSG AHI was 23.24/hour. These are results from the completed
local experiment, not results guaranteed by a fresh clone or another machine.
The source artifacts remain local under `data/dreamt/DREAMT/derived/`.

Event fields contain `1` and blanks. Blank-as-no-marked-event during scored
sleep is a **provisional interpretation**, not an explicit guarantee in the
supplied metadata. Candidate counts/rates are not validated clinical AHI.
SHAP explains conditional scalar contributions, not waveform causation.
[Read the full target definitions and evaluation limits](docs/PIPELINE.md).

## Repository map

| Location | Purpose |
| --- | --- |
| `web/` | Primary React/TypeScript interface and npm lockfile |
| `research_api.py`, `run_workspace.sh` | Local API and application launcher |
| `dreamt_dataset.py`, `audit_dreamt.py` | Bounded CSV reading and whole-cohort audit |
| `prepare_dreamt.py`, `train_dreamt.py` | Subject split, feature caching, training and held-out evaluation |
| `wearable_pipeline.py`, `pipeline.py` | Wearable preprocessing/inference and shared neural architecture |
| `check_workspace.py` | Raw-data, prepared-cache and trained-artifact preflight |
| `run_dreamt_overnight.py` | Resumable preparation/training supervisor |
| `app.py`, `siegen_dataset.py`, `train_wearable.py` | Legacy/additional workflows |
| `tests/` | Target, timestamp, inference and explanation checks |
| `archive/ecg/` | Historical ECG experiments; not wearable validation |
| `data/`, `outputs/` | Local-only data and generated outputs; ignored by Git |

```bash
python -m unittest discover -s tests -v
npm run build --prefix web
```

For frontend development, run the Python API on port 8000 and
`npm run dev --prefix web` in another terminal. The Vite development server
forwards `/api` to Python. Rebuild `web/` after frontend edits when using the
single-server launcher.
