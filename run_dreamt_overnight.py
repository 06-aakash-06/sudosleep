"""Resume local DREAMT preparation and train only after all caches exist."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent
DERIVED = ROOT / 'data/dreamt/DREAMT/derived'


def update(stage, **details):
    DERIVED.mkdir(parents=True, exist_ok=True)
    status = {'stage': stage, 'updated_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), **details}
    temp = DERIVED / 'overnight_status.tmp'
    temp.write_text(json.dumps(status, indent=2))
    temp.replace(DERIVED / 'overnight_status.json')
    print(json.dumps(status), flush=True)


def alive(pid):
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--wait-pids', nargs='*', type=int, default=[])
    parser.add_argument('--workers', type=int, default=3)
    parser.add_argument('--max-hours', type=float, default=4)
    args = parser.parse_args()
    os.chdir(ROOT)
    env = os.environ.copy()
    env.update({'OMP_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1', 'PYTORCH_ENABLE_MPS_FALLBACK': '1'})
    try:
        while any(alive(pid) for pid in args.wait_pids):
            update('waiting_for_existing_preparation', prepared=len(list((DERIVED / 'features').glob('*.npz'))))
            time.sleep(60)
        update('preparing_missing_subjects')
        subprocess.run([sys.executable, '-u', 'prepare_dreamt.py', '--prepare', '--workers', str(args.workers)], check=True, env=env)
        split = json.loads((DERIVED / 'subject_splits.json').read_text())
        subjects = split['train'] + split['validation'] + split['test']
        missing = [s for s in subjects if not (DERIVED / 'features' / f'{s}.npz').is_file()]
        if missing:
            raise RuntimeError(f'{len(missing)} subject caches are missing after preparation')
        update('training', prepared=len(subjects))
        subprocess.run([sys.executable, '-u', 'train_dreamt.py', '--epochs', '8', '--batch-size', '2', '--max-hours', str(args.max_hours)], check=True, env=env)
        update('complete', prepared=len(subjects))
    except Exception as exc:
        update('failed', error=str(exc))
        raise


if __name__ == '__main__':
    main()
