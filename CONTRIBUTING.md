# Collaborating on SudoSleep

Start with [dataset access](docs/DATASET_SETUP.md), then
[reproduce the local workflow](docs/REPRODUCING.md). Code access is separate
from DREAMT access; never commit the dataset, feature caches, weights,
subject-level exports, credentials or local environment directories.

## Review before proposing changes

- Trace the current stages in [the pipeline map](docs/PIPELINE.md).
- Inspect several recordings, including model misses, in the Night desk.
- Compare the neural model with the scalar baseline and inspect event recall,
  not only window AUROC.
- Record label uncertainty, original sampling limits and boundary effects.

Use training subjects for model development and validation subjects for
selection. Repeatedly tuning against the existing test scores compromises
that test as an independent estimate. For a materially changed study, propose
an evaluation protocol before running a new final comparison; do not silently
reassign test subjects or revise target definitions after seeing test results.

## Make a reviewable change

Create a branch, e.g. `codex/improve-event-decoding`, and describe:

1. The concrete finding and the proposed change.
2. Dataset version, cohort/split, target definition and preprocessing version.
3. Hardware, software versions and commands actually run.
4. Validation results, comparison baseline and relevant failure cases.
5. Whether a result is measured, provisional or still untested.

Keep raw evidence locally. Public PRs should contain code and suitably
aggregated findings, with no subject-level data or patient traces. Existing
research history stays in `archive/ecg/`; add current experiments through
reviewed scripts rather than another unlabeled scratch notebook.

## Checks

```bash
python -m unittest discover -s tests -v
python check_workspace.py --stage trained
npm ci --prefix web
npm run build --prefix web
```

For API/UI changes, run the real flow locally: subject selection → segment
analysis → PSG comparison → explanation/export. Inspect desktop and mobile
states, including missing-artifact errors. Do not claim an API response alone
proves the visible UI works.

If preprocessing changes, update its version tag and deliberately rebuild
caches/checkpoints. Existing `.npz` files are reused by preparation, so simply
rerunning the command is not enough to regenerate an old cache. Preserve an
experiment copy locally before replacing outputs; fixed training filenames
are overwritten by a new run.
