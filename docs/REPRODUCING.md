# Reproduce and inspect the DREAMT experiment

Run from the repository root in the activated Python environment. First
complete [dataset setup](DATASET_SETUP.md). Raw files, caches and generated
results remain under the ignored `data/dreamt/DREAMT/` directory.

## Audit → split → prepare → train → evaluate

```bash
python check_workspace.py --stage raw
python audit_dreamt.py --workers 3
python prepare_dreamt.py --split --prepare --workers 3
python check_workspace.py --stage prepared
python train_dreamt.py --epochs 8 --batch-size 2 --max-hours 4
python check_workspace.py --stage trained
npm ci --prefix web
npm run build --prefix web
./run_workspace.sh
```

Open http://127.0.0.1:8000. This is a loopback address on the computer running
the API; it is not a hosted link to the original author's Mac.

The split uses seed 42, AHI-stratified subject groups and the audit eligibility
rules. The original local release produced 68/15/15 subjects plus two excluded
outliers. Keep those exclusions and target rules visible when interpreting
results. Matching a seed does not guarantee bitwise identical model weights
across different devices or scientific-library versions.

Training allows up to eight epochs, early stopping after three epochs without
improved validation AUPRC, and a four-hour optimization budget. Validation and
final evaluation add overhead beyond that budget. Preparation is separate and
can take many hours. Reduce `--workers` if memory or temperature is a concern.
The held-out test is evaluated after model/threshold selection, not every epoch.

## Outputs and their purpose

All paths below are relative to `data/dreamt/DREAMT/derived/`:

| Artifact | Created by | Used for |
| --- | --- | --- |
| `audit_64hz.json`, `audit_64hz.csv` | audit | Event coverage, timestamps, stages, AHI comparison |
| `subject_splits.json` | preparation | Fixed subject ownership and exclusions |
| `features/Sxxx.npz` | preparation | Full-night waveforms, scalars, labels and eligibility |
| `best_dreamt_state.pt` | training | Best validation-selected network state |
| `dreamt_accelerometer.pt` | training | Dashboard-compatible checkpoint, scaler and transitions |
| `baseline_random_forest.joblib` | training | Scalar baseline model |
| `test_predictions.csv` | evaluation | Saved held-out minute scores/targets; ROC and PR curves |
| `dreamt_results.json` | evaluation | Metrics, history, subject rate comparisons and caveats |
| `overnight_status.json`, `overnight.log` | optional supervisor | Progress and failure diagnostics |

A direct successful `train_dreamt.py` run does not need the overnight supervisor
or its status file to display the completed evaluation.

## Resume after interruption

Completed per-subject feature caches are written atomically and reused. Rerun:

```bash
python prepare_dreamt.py --prepare --workers 3
```

This fills missing caches. Do not recreate the split after training has started.
The preprocessing cache version is checked by the preflight. After changing
preprocessing, preserve an experiment copy locally and regenerate compatible
caches deliberately; existing caches are not automatically recomputed.

To prepare missing caches and then train sequentially:

```bash
python -u run_dreamt_overnight.py --workers 3 --max-hours 4 \
  > data/dreamt/DREAMT/derived/overnight.log 2>&1
```

Use this only when another preparation/training job is not already running.
The supervisor **starts a new training run** after preparation; it does not
resume optimizer state or skip a previously completed model. Training and
reports use fixed output filenames, so back up an existing experiment locally
before starting a replacement. On a Mac, prefix the command with `caffeinate -i`
to prevent idle sleep; keep a laptop powered with its lid open. No command can
continue executing through a shutdown.

Track the supervisor with:

```bash
tail -f data/dreamt/DREAMT/derived/overnight.log
```

Training logs report epoch, batch progress, loss and validation AUPRC. The
current UI reads artifacts when loaded; refresh after a completed run to see
new results. The currently running API may cache two nights, so restart it
after deliberately replacing feature caches.

## Review the result

1. **Data & methods:** confirm the audit, original 32 Hz sampling, split and
   provisional event-label interpretation.
2. **Night desk:** choose a subject and segment, compare PSG annotations with
   candidate states, zoom the proxy signals, inspect the spectral view and
   export selected-minute results.
3. **Model study:** inspect baseline comparison, confusion matrix, saved neural
   ROC/PR curves, event matching, per-subject rate errors and learning history.
4. **Signal drivers:** choose a minute and compute conditional scalar SHAP.
   Uploaded/synthetic data use a labelled heuristic rather than this neural path.

Use [CONTRIBUTING.md](../CONTRIBUTING.md) to record proposed changes and protect
the held-out test from repeated tuning. The original run's low event recall
and baseline parity are unresolved research findings, not completion criteria
for a clinically usable system.
