# Historical ECG experiments

These notebooks were moved here intact to preserve the research history while
keeping the repository root focused on the current wearable workflow:

- `pipelinefinal.ipynb`
- `pipeline_v2.ipynb`

Their saved outputs concern ECG experiments, including PhysioNet Apnea-ECG,
and are not evidence that the current accelerometer model achieves those
metrics. Some saved feature dimensions differ from the current code. The
notebooks have not been rerun as part of this cleanup; relative data paths may
require adjustment when run from this archive.

The shared active `pipeline.py` remains at the repository root because the
wearable training/inference code imports its model architecture. An empty
`Untitled.ipynb` scratch notebook was removed; Git history retains it.

For fresh wearable work start with the [main README](../../README.md), not
these notebooks. For the original ECG data use the official
[Apnea-ECG resource](https://physionet.org/content/apnea-ecg/1.0.0/) and its terms.
