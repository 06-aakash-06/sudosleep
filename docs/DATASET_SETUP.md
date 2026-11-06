# DREAMT: access, download and local layout

## 1. Obtain access yourself

Use the official [DREAMT version 2.2.0 page on PhysioNet](https://physionet.org/content/dreamt/2.2.0/).
Create/sign in to your own PhysioNet account and complete the project's data
use agreement and any requirements shown there. The current page lists access
for registered users who sign its specified agreement; follow the live page
if that changes. Do not use another person's login or credentials.

Version 2.2.0 fixes ACC/PSG alignment. Use this version for the documented
experiment rather than silently substituting a different release. The dataset
is described by Wang, Yang, Shetty and Dunn (2026),
[DOI 10.13026/3f7y-2d80](https://doi.org/10.13026/3f7y-2d80).
See the official page for the full dataset, publication and PhysioNet citations.

## 2. Download the files after access is enabled

Use the download controls/instructions in the authenticated **Files** section.
This repository deliberately does not embed a password, access token, download
cookie or a substitute public mirror. Each teammate downloads through their
own account. Do not upload DREAMT to GitHub or attach it to issues/PRs.

For the current React research workflow, download:

- All 100 `data_64Hz/Sxxx_whole_df.csv` recordings.
- `participant_info.csv` (must contain `SID` and `AHI`).
- `Metadata.txt` for signal and annotation interpretation.

The 64 Hz files in the locally audited release contain the required event
fields. `data_100Hz/` is optional for this app: it contains additional PSG
waveforms for deeper multimodal research and is supported by the legacy
loader. The neural wearable input still uses acceleration only. If the 64 Hz
files you receive lack the event fields below, stop and investigate the release;
do not manufacture labels from sleep stages.

Use the page's reported sizes to budget disk space, plus room for extraction,
feature caches and your Python environment. The original local copy with both
64 Hz and 100 Hz directories occupies roughly 114 GB; that is not a minimum
size for the 64 Hz-only workflow.

## 3. Put the extracted files in this exact layout

Run commands from the repository root:

```text
sudosleep/
  data/
    dreamt/
      DREAMT/
        Metadata.txt
        participant_info.csv
        data_64Hz/
          S002_whole_df.csv
          S003_whole_df.csv
          ... (100 recordings)
        data_100Hz/                 # optional
          S002_PSG_df.csv
          ...
        derived/                    # created locally by the scripts
```

The outer archive may add directory levels. Move its extracted contents into
this layout; avoid `DREAMT/DREAMT/data_64Hz/` or placing raw CSVs in `derived/`.
The scripts intentionally use the same layout so your teammate need not edit
hardcoded paths. `data/` is Git-ignored. Keep access credentials out of files
and shell history.

## 4. Verify before expensive processing

```bash
source .venv/bin/activate
python check_workspace.py --stage raw
```

The preflight checks recording count and headers, subject/metadata alignment
and required files. Required recording columns are `TIMESTAMP`, `ACC_X`,
`ACC_Y`, `ACC_Z`, `Sleep_Stage`, `Obstructive_Apnea`, `Central_Apnea`,
`Hypopnea` and `Multiple_Events`. It does not scan all samples; the next step
is the full timestamp/event audit in [REPRODUCING.md](REPRODUCING.md).

If the preflight fails, correct the layout/download before preparing features.
A login HTML page saved with a `.csv` extension is not a valid recording.
Do not rename unrelated columns to force a pass.

## What a teammate does and does not get from GitHub

GitHub contains code, documentation and historical experiments. It does not
contain restricted raw records, subject-level evaluation exports, cache arrays
or the DREAMT-trained weights. The complete independent path is to download
the authorized dataset, audit it, prepare caches and train/evaluate locally.
This produces every artifact consumed by the dashboard. Downloading DREAMT
alone will not create the saved checkpoint or evaluation report.

Do not assume that caches, subject exports or model weights can be redistributed
freely. Any separately proposed transfer should first be checked against the
applicable agreement and both collaborators' authorization. The documented
setup does not depend on such a transfer.
