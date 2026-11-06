"""Validate local DREAMT inputs and artifacts without training or changing files."""
from __future__ import annotations
import argparse
import csv
import json
import pickle
from pathlib import Path
import sys
import zipfile

ROOT = Path(__file__).resolve().parent
COLUMNS = {'TIMESTAMP', 'ACC_X', 'ACC_Y', 'ACC_Z', 'Sleep_Stage',
           'Obstructive_Apnea', 'Central_Apnea', 'Hypopnea', 'Multiple_Events'}


def check_layout(data: Path, stage: str) -> list[tuple[bool, str]]:
    checks = []
    def check(ok, message):
        checks.append((bool(ok), message))
        return bool(ok)
    files = sorted((data / 'data_64Hz').glob('S*_whole_df.csv'))
    check(len(files) == 100, f'64 Hz recordings: {len(files)}/100')
    subjects = {p.name.split('_')[0] for p in files}
    bad = []
    for file in files:
        try:
            with file.open(newline='') as handle:
                header = next(csv.reader(handle))
            if not COLUMNS.issubset(header):
                bad.append(file.name)
        except (OSError, UnicodeError, StopIteration):
            bad.append(file.name)
    check(bool(files) and not bad, f'Recording headers: {len(bad)} invalid')
    metadata = data / 'participant_info.csv'
    try:
        with metadata.open(newline='') as handle:
            rows = list(csv.DictReader(handle))
        ids = [r['SID'] for r in rows]
        ahis = [float(r['AHI']) for r in rows]
        import math
        check(len(ids) == len(set(ids)) and subjects == set(ids) and bool(ids),
              'Metadata subjects match the recording files without duplicates')
        check(all(math.isfinite(a) and a >= 0 for a in ahis), 'Metadata AHI values are finite and nonnegative')
    except (OSError, KeyError, ValueError):
        check(False, 'participant_info.csv must contain valid SID and AHI columns')
    check((data / 'Metadata.txt').is_file(), 'Metadata.txt is present')
    if stage == 'raw':
        return checks
    derived = data / 'derived'
    try:
        audit = json.loads((derived / 'audit_64hz.json').read_text())
        results = audit['results']
        check({r['subject'] for r in results} == subjects and len(results) == 100
              and not any('error' in r for r in results), 'Full audit covers all 100 recordings without file errors')
        check((derived / 'audit_64hz.csv').is_file(), 'Audit CSV is present')
    except (OSError, ValueError, KeyError, TypeError):
        check(False, 'Run audit_dreamt.py to generate the complete audit')
    try:
        split = json.loads((derived / 'subject_splits.json').read_text())
        groups = [split[k] for k in ('train', 'validation', 'test', 'excluded')]
        flat = [s for group in groups for s in group]
        if not check(len(flat) == len(set(flat)) and set(flat) == subjects
                     and all(groups[:3]), 'Split subjects are disjoint and cover the raw cohort including exclusions'):
            return checks
        expected = groups[0] + groups[1] + groups[2]
        check(split.get('seed') == 42, 'Split uses documented seed 42')
    except (OSError, ValueError, KeyError, TypeError):
        check(False, 'Run prepare_dreamt.py --split to create the subject split')
        return checks
    try:
        import numpy as np
    except ImportError:
        check(False, 'Install requirements.txt before checking prepared artifacts')
        return checks
    invalid = []
    for subject in expected:
        path = derived / 'features' / f'{subject}.npz'
        try:
            # Read array headers instead of loading hundreds of nights of waveforms.
            with zipfile.ZipFile(path) as archive:
                with archive.open('signals.npy') as handle:
                    version = np.lib.format.read_magic(handle)
                    shape, _, dtype = (np.lib.format.read_array_header_1_0(handle) if version == (1, 0)
                                       else np.lib.format.read_array_header_2_0(handle))
                    if dtype.hasobject or len(shape) != 3 or shape[1:] != (3, 6000):
                        raise ValueError('Invalid waveform shape/dtype')
            with np.load(path, allow_pickle=False) as f:
                labels = f['labels']
                if (str(f['preprocessing_version']) != 'emd_scg_v2_rate_aware'
                    or str(f['subject']) != subject or labels.shape != (shape[0],)
                    or f['scalars'].shape != (shape[0], 12)
                    or not np.isin(labels, [-1, 0, 1]).all()
                    or f['valid_seconds'].shape != labels.shape
                    or f['sleep_fraction'].shape != labels.shape
                    or not np.isfinite(f['scalars']).all()
                    or not np.isfinite(f['valid_seconds']).all()
                    or not ((f['valid_seconds'] >= 0) & (f['valid_seconds'] <= 60)).all()
                    or not np.isfinite(f['sleep_fraction']).all()
                    or not ((f['sleep_fraction'] >= 0) & (f['sleep_fraction'] <= 1)).all()):
                    raise ValueError('Incompatible cache')
        except (OSError, ValueError, KeyError, EOFError, zipfile.BadZipFile):
            invalid.append(subject)
    check(not invalid, f'Feature caches: {len(expected)-len(invalid)}/{len(expected)} compatible')
    if stage == 'prepared':
        return checks
    try:
        import torch
        from wearable_pipeline import FEATURE_NAMES
        model = torch.load(derived / 'dreamt_accelerometer.pt', map_location='cpu', weights_only=True)
        center, scale = np.asarray(model['scalar_center']), np.asarray(model['scalar_scale'])
        check(model.get('input_modality') == 'accelerometer'
              and model.get('preprocessing_version') == 'emd_scg_v2_rate_aware'
              and model.get('feature_names') == FEATURE_NAMES and bool(model.get('model_states'))
              and center.shape == scale.shape == (12,) and np.isfinite(center).all()
              and np.isfinite(scale).all() and (scale > 0).all(), 'Wearable checkpoint metadata and scaler are compatible')
    except (ImportError, OSError, ValueError, KeyError, RuntimeError, EOFError, pickle.UnpicklingError):
        check(False, 'A compatible locally produced dreamt_accelerometer.pt is required')
    try:
        import pandas as pd
        report = json.loads((derived / 'dreamt_results.json').read_text())
        predictions = pd.read_csv(derived / 'test_predictions.csv')
        check(len(predictions) == report['test']['n_windows']
              and set(predictions['subject']) == set(split['test'])
              and predictions['label'].isin([0, 1]).all()
              and predictions['score'].between(0, 1).all(), 'Held-out prediction count, subjects and score ranges match the report')
        check(all(k in report for k in ['baseline_test', 'event_matching', 'per_subject', 'training_history']),
              'Evaluation report includes baseline, events, subject comparisons and history')
    except (ImportError, OSError, ValueError, KeyError, TypeError):
        check(False, 'Complete dreamt_results.json and test_predictions.csv are required')
    return checks


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, default=ROOT / 'data/dreamt/DREAMT')
    parser.add_argument('--stage', choices=['raw', 'prepared', 'trained'], default='raw')
    args = parser.parse_args()
    results = check_layout(args.data, args.stage)
    for ok, message in results:
        print(f'{"PASS" if ok else "FAIL"}  {message}')
    success = all(ok for ok, _ in results)
    print(f'\n{args.stage.upper()}: {"READY" if success else "NOT READY"}')
    if not success:
        print('See docs/DATASET_SETUP.md and docs/REPRODUCING.md for the next steps.')
    return 0 if success else 1


if __name__ == '__main__':
    sys.exit(main())
