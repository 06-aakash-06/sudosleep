# Pipeline: code, targets and limits

## Stage-to-code map

| Stage | Current implementation | How to inspect it |
| --- | --- | --- |
| Dataset audit | `audit_dreamt.py` | All-subject timestamp, acceleration, sleep-stage and sparse-event checks |
| Grouping and targets | `prepare_dreamt.py` | Seed-42 subject split; exclusions from audit/AHI discrepancy |
| Signal preparation | `wearable_pipeline.py` | Axis handling, rate-aware resampling, proxy extraction, full-night normalization and minute features |
| Neural encoder | `pipeline.py` → `SudoSleepNet` | 1D-CNN, squeeze-excitation attention, within-minute and five-minute context |
| Supervised fitting | `train_dreamt.py` | Training-only scalar scaler, baseline, weighted sampling, validation selection and final test |
| State decoding | `wearable_pipeline.py` | Two-state Viterbi with transitions learned from training labels |
| Event/rate evaluation | `train_dreamt.py` | Coarse one-to-one event overlap and candidate-run rates |
| Explanation | `wearable_pipeline.py` | Conditional Kernel SHAP for selected-minute scalar inputs |
| API and interface | `research_api.py`, `web/src/` | Local artifact serving, bounded segment reads, inference, plots and exports |

`pipeline.py` also contains older ECG functions. It remains an active dependency
for the neural architecture; moving it into the historical archive would break
current imports. The archived notebooks contain ECG experiments, not validation
of the wearable model.

## Physiological representation

Accelerometer magnitude, respiratory-band motion and an EMD-selected
cardiac-band motion proxy form three waveform channels, accompanied by 12
scalars. The original DREAMT accelerometer rate is 32 Hz. Resampling to
64/100 Hz does not restore frequencies beyond the original Nyquist limit;
the requested 10–30 Hz cardiac band is therefore limited to roughly 10–15.4 Hz.
A zero selected proxy means no component met the extraction rule, not absence
of cardiac activity. ECG morphology reconstruction has not been demonstrated.

For DREAMT inference the app uses full-night prepared caches to keep
normalization consistent with training. Segment signal tracks are drawn from
those caches; PSG stages/events are read separately from the raw CSV. The
neural model has ±2 minutes of context and is an offline research model rather
than a strictly causal real-time detector.

## What a positive minute means

Only predominantly PSG-scored sleep windows are eligible: at least 30 seconds
of samples, at least 80% labelled N1/N2/N3/REM, and at least one second of marked
event overlap during sleep to be positive. Other eligible windows become
provisional negatives; wake/preparation/missing or insufficiently scored
windows are excluded from evaluation. The UI may show exploratory model
predictions outside that eligibility and labels them accordingly.

The supplied fields use `1` and blank. Metadata defines positive markings but
not explicit blank-as-negative semantics. That interpretation is provisional,
supported by the local marked-run/AHI comparison, and is not verified clinical
ground truth. The file field `Multiple_Events` also differs from a mixed-apnea
name mentioned in the supplied metadata. Do not silently equate the two.

## Evaluation boundaries

- Subjects are kept disjoint across train, validation and test. No five-minute
  context crosses a subject boundary.
- Scalar median/IQR scaling and transition estimates use training data only.
- Validation AUPRC selects weights; validation F1 selects raw-score thresholds.
- The random forest is a scalar baseline; the heuristic is a separate signal
  scoring path. They are not interchangeable comparisons.
- Raw model thresholding and Markov states are different outputs. The dashboard
  labels both; Markov decoding uses transition/emission probabilities rather
  than the displayed raw-score threshold.
- Dashboard Markov decoding is limited to the selected segment. Evaluation
  smoothing respects contiguous eligible sleep blocks; their boundary states
  can differ.
- Event matching is greedy one-to-one with at least one second of overlap.
  Predictions have minute boundaries; reference events are restricted to
  eligible scored-sleep minutes. This can merge or miss shorter events.
- Candidate-run rates use eligible sleep duration, whereas supplied PSG AHI is
  a study-level value. The rate comparison is exploratory, not clinical AHI.
- SHAP holds waveform inputs and other context fixed. It explains the selected
  scalar-feature contribution against a training-median background; it neither
  explains the whole waveform model nor validates the prediction.

Cheyne–Stokes constraints and predicted sleep-stage/position/REM classifiers
are not implemented. PSG sleep stages in the UI are reference annotations.
The proposal's target performance and schedule are not measured outcomes.
