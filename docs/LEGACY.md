# Legacy and additional workflows

The primary interface is React + Python at http://127.0.0.1:8000. These older
entry points are retained for useful workflows and research history.

## Streamlit and Siegen

```bash
python siegen_dataset.py --download
streamlit run app.py
```

The Siegen archive contains sleep-stage annotations, not apnea event targets.
The loader expands run-length encoded sensor values and reconstructs the time
grid. See the [dataset authors' page](https://www.eti.uni-siegen.de/ubicomp/home/datasets/ichi14/)
for access, attribution and use terms. The legacy app also accepts DREAMT CSVs,
short uploads and a synthetic example, with explicit model/heuristic labels.

## Generic labelled accelerometer training

`train_wearable.py` accepts three acceleration axes, binary minute-level
`label`, and `record`. Include `subject` to group multiple nights from the same
person. With no subject column, records are assumed independent people. The
default grouped training requires at least five independent subjects.

```bash
python train_wearable.py labelled_accelerometer.csv \
  --sample-rate 50 --output models/sudo_sleep_accelerometer.pt
```

Use `--sensor-rate 32` for resampled DREAMT acceleration. Do not turn Siegen
sleep-stage codes into apnea labels, or load an ECG-trained checkpoint as a
wearable model. This generic trainer is distinct from `train_dreamt.py` and
its fixed held-out DREAMT experiment.

## Historical notebooks

See [archive/ecg/](../archive/ecg/README.md). Notebooks are preserved unchanged;
saved historical outputs and older feature dimensions are not current
wearable-model metrics. No archived notebook is required to launch the app.
