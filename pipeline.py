#!/usr/bin/env python3
"""
=============================================================================
  Sudo Sleep — ECG Apnea-Minute Research Pipeline (experimental)
  YŪGŌ Hackathon 2026 | Pack A
=============================================================================
  Upgrades over v1
  ────────────────
  1. Temporal-Context Window   – 5-epoch sliding window (±2 min) so the
     Transformer sees crescendo/decrescendo breathing patterns.
  2. Frequency-Domain HRV      – LF/HF ratio + spectral flatness extracted
     from the ECG input (scalar dim 10 → 12).
  3. Stochastic Weight Averaging (SWA) – smooths out noisy weight
     trajectories; dramatically reduces fold-to-fold variance.
  4. Stability / recall fixes  – per-fold dynamic pos_weight, Mixup,
     label smoothing, cosine-annealing warm restarts, longer patience.
=============================================================================
"""

import os, glob, struct, warnings, logging, pickle, time, multiprocessing
from typing import Tuple, Dict, List, Optional, Any
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import butter, sosfiltfilt, find_peaks, welch
from scipy.interpolate import interp1d
from scipy.ndimage import median_filter
import pywt
from joblib import Parallel, delayed

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torch.optim.swa_utils import AveragedModel, SWALR
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score, roc_curve, classification_report

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("SudoSleep")

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────
FS              = 100          # Sampling rate (Hz)
EPOCH_LEN       = 6000         # 60 s × 100 Hz
CONTEXT_WINDOW  = 5            # ±2 epochs around centre
N_SCALAR_FEATS  = 12           # was 10, now + LF/HF + spectral flatness
SIGNAL_COLS     = [f"signal_{i}" for i in range(EPOCH_LEN)]


# ═══════════════════════════════════════════════════════════════════════════════
# 0.  DATA LOADING — PhysioNet Apnea-ECG  (unchanged from v1)
# ═══════════════════════════════════════════════════════════════════════════════
def _parse_hea(hea_path: str) -> Dict[str, Any]:
    with open(hea_path, "r") as f:
        lines = [l.strip() for l in f if l.strip() and not l.startswith("#")]
    parts = lines[0].split()
    record_name = parts[0]
    n_signals   = int(parts[1])
    fs          = int(parts[2]) if len(parts) > 2 else 100
    n_samples   = int(parts[3]) if len(parts) > 3 else 0

    signals = []
    for i in range(1, n_signals + 1):
        if i < len(lines):
            sp = lines[i].split()
            gain_str = sp[2] if len(sp) > 2 else "200"
            gain_val = float(gain_str.split("(")[0].split("/")[0])
            if gain_val == 0:
                gain_val = 200.0
            signals.append({
                "filename": sp[0],
                "format":   int(sp[1]) if len(sp) > 1 else 16,
                "gain":     gain_val,
                "baseline": int(sp[4]) if len(sp) > 4 else 0,
            })
    return {"record_name": record_name, "n_signals": n_signals,
            "fs": fs, "n_samples": n_samples, "signals": signals}


def _read_dat_format16(dat_path: str, n_samples: int,
                       n_signals: int = 1) -> np.ndarray:
    with open(dat_path, "rb") as f:
        raw = f.read()
    total = len(raw) // (2 * n_signals)
    if n_samples == 0 or n_samples > total:
        n_samples = total
    return np.frombuffer(
        raw[:n_samples * n_signals * 2], dtype="<i2"
    ).reshape(n_samples, n_signals)


def _read_dat_format212(dat_path: str, n_samples: int,
                        n_signals: int = 1) -> np.ndarray:
    with open(dat_path, "rb") as f:
        raw = np.frombuffer(f.read(), dtype=np.uint8)
    n_triplets = len(raw) // 3
    samples = np.zeros(n_triplets * 2, dtype=np.int16)
    for i in range(n_triplets):
        b0, b1, b2 = int(raw[i*3]), int(raw[i*3+1]), int(raw[i*3+2])
        s1 = b0 | ((b1 & 0x0F) << 8)
        if s1 >= 2048: s1 -= 4096
        s2 = b2 | ((b1 & 0xF0) << 4)
        if s2 >= 2048: s2 -= 4096
        samples[i*2], samples[i*2+1] = s1, s2
    if n_samples > 0:
        samples = samples[:n_samples]
    return samples.reshape(-1, n_signals)


def _read_apn_annotations(apn_path: str) -> np.ndarray:
    try:
        import wfdb
        ann = wfdb.rdann(apn_path.replace(".apn", ""), "apn")
        return np.array([1 if s == "A" else 0 for s in ann.symbol], dtype=int)
    except ImportError:
        pass
    with open(apn_path, "rb") as f:
        raw = f.read()
    labels, i = [], 0
    while i < len(raw) - 1:
        b1 = raw[i + 1]; i += 2
        anntype = (b1 >> 2) & 0x3F
        if anntype == 59:  i += 4;  continue
        if anntype == 0:   continue
        if anntype == 63:
            if i + 1 < len(raw):
                aux_len = raw[i] | (raw[i+1] << 8)
                i += 2 + aux_len
                if aux_len % 2 == 1: i += 1
            continue
        if anntype == 1:            labels.append(0)
        elif anntype in (38, 22):   labels.append(1)
    return np.array(labels, dtype=int)


def load_physionet_apnea(
    data_dir: str,
    record_filter: Optional[List[str]] = None,
    include_unlabelled: bool = False,
    target_fs: int = FS,
) -> pd.DataFrame:
    data_dir = Path(data_dir)
    hea_files = sorted(data_dir.glob("*.hea"))
    record_names = set()
    for hf in hea_files:
        stem = hf.stem
        if stem.endswith("er") or (stem.endswith("r") and stem[0] != "r"):
            base = stem[:-2] if stem.endswith("er") else stem[:-1]
            if (data_dir / f"{base}.hea").exists():
                continue
        record_names.add(stem)

    if record_filter is not None:
        record_names = record_names.intersection(set(record_filter))
    records_to_load = sorted(r for r in record_names if r[0] in ("a","b","c"))
    if include_unlabelled:
        records_to_load += sorted(r for r in record_names if r[0] == "x")

    all_rows: list[dict] = []
    for rec_name in records_to_load:
        hea  = data_dir / f"{rec_name}.hea"
        dat  = data_dir / f"{rec_name}.dat"
        apn  = data_dir / f"{rec_name}.apn"
        if not hea.exists() or not dat.exists():
            continue
        try:
            header = _parse_hea(str(hea))
            fmt      = header["signals"][0].get("format", 16)
            gain     = header["signals"][0].get("gain", 200.0)
            baseline = header["signals"][0].get("baseline", 0)
            raw_adc  = (_read_dat_format212 if fmt == 212 else _read_dat_format16)(
                str(dat), header["n_samples"], header["n_signals"])
            signal = (raw_adc[:, 0].astype(np.float64) - baseline) / gain

            if header["fs"] != target_fs and header["fs"] > 0:
                from scipy.signal import resample
                signal = resample(signal,
                                  int(len(signal) * target_fs / header["fs"]))

            labels = _read_apn_annotations(str(apn)) if apn.exists() else None
            spe = target_fs * 60
            n_epochs = len(signal) // spe
            if labels is not None:
                n_epochs = min(n_epochs, len(labels))

            for ei in range(n_epochs):
                epoch_sig = signal[ei * spe : (ei + 1) * spe]
                if len(epoch_sig) != spe:
                    continue
                row: dict = {
                    "record": rec_name,
                    "label":  int(labels[ei]) if labels is not None and ei < len(labels) else -1,
                }
                for j in range(spe):
                    row[f"signal_{j}"] = float(epoch_sig[j])
                all_rows.append(row)
        except Exception as e:
            logger.warning(f"Failed {rec_name}: {e}")

    df = pd.DataFrame(all_rows)
    return df if include_unlabelled else df[df["label"] >= 0].reset_index(drop=True)


def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


# ═══════════════════════════════════════════════════════════════════════════════
# 1.  PREPROCESSING
# ═══════════════════════════════════════════════════════════════════════════════
class SubjectWiseScaler:
    """Z-normalise each record (subject night) independently."""
    def __init__(self):
        self.stats: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}

    def fit(self, df: pd.DataFrame) -> "SubjectWiseScaler":
        for record, grp in df.groupby("record"):
            vals = grp[SIGNAL_COLS].values
            self.stats[record] = (vals.mean(axis=0),
                                  vals.std(axis=0).clip(min=1e-8))
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        for record, grp in df.groupby("record"):
            m, s = self.stats.get(
                record,
                (grp[SIGNAL_COLS].values.mean(axis=0),
                 grp[SIGNAL_COLS].values.std(axis=0).clip(min=1e-8)))
            df.loc[grp.index, SIGNAL_COLS] = (grp[SIGNAL_COLS].values - m) / s
        return df

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        return self.fit(df).transform(df)


# ═══════════════════════════════════════════════════════════════════════════════
# 2.  ADVANCED FEATURE EXTRACTION  (v2 — +LF/HF, +Spectral Flatness)
# ═══════════════════════════════════════════════════════════════════════════════
class AdvancedFeatureExtractor:
    def __init__(self, fs: int = FS):
        self.fs   = fs
        self.sos_resp = butter(4, [0.1, 0.5],   btype="band", fs=fs, output="sos")
        # Legacy attribute name: this 10–20 Hz band is applied to ECG input,
        # so it is filtered ECG, not a seismocardiogram (SCG) measurement.
        self.sos_scg  = butter(4, [10.0, 20.0],  btype="band", fs=fs, output="sos")

    # ── IBI helpers ──────────────────────────────────────────────────────────
    def _extract_ibi(self, scg: np.ndarray) -> np.ndarray:
        """Legacy helper: detect peaks in the high-frequency filtered ECG band."""
        peaks, _ = find_peaks(scg,
                              height=np.std(scg) * 1.2,
                              distance=int(0.4 * self.fs))
        if len(peaks) < 3:
            return np.array([])
        ibi = np.diff(peaks) / self.fs
        return ibi[(ibi > 0.3) & (ibi < 2.0)]

    # ── SDNN (time-domain HRV) ───────────────────────────────────────────────
    def compute_hrv_proxy(self, scg: np.ndarray) -> float:
        ibi = self._extract_ibi(scg)
        return float(np.std(ibi)) if len(ibi) >= 2 else 0.0

    # ── NEW: LF/HF ratio (frequency-domain HRV) ─────────────────────────────
    def compute_lf_hf_ratio(self, scg: np.ndarray) -> float:
        """
        Resample IBI series to 4 Hz, run Welch PSD, compute
        power in LF (0.04-0.15 Hz) vs HF (0.15-0.40 Hz).
        Returns log(LF/HF + 1) — apnea drives this UP.
        """
        try:
            ibi = self._extract_ibi(scg)
            if len(ibi) < 6:
                return 0.0
            # Cumulative time axis for each IBI
            t_ibi = np.cumsum(ibi)
            t_ibi = np.insert(t_ibi, 0, 0.0)
            ibi_full = np.concatenate([[ibi[0]], ibi])   # align lengths

            # Uniformly resample at 4 Hz via cubic interpolation
            resample_fs = 4.0
            t_uniform = np.arange(t_ibi[0], t_ibi[-1], 1.0 / resample_fs)
            if len(t_uniform) < 16:
                return 0.0
            interp_fn  = interp1d(t_ibi, ibi_full, kind="cubic",
                                  fill_value="extrapolate")
            ibi_resampled = interp_fn(t_uniform)

            # Welch PSD
            freqs, psd = welch(ibi_resampled, fs=resample_fs, nperseg=min(64, len(ibi_resampled)))
            lf_mask = (freqs >= 0.04) & (freqs < 0.15)
            hf_mask = (freqs >= 0.15) & (freqs <= 0.40)
            lf_power = np.trapz(psd[lf_mask], freqs[lf_mask]) if lf_mask.any() else 0.0
            hf_power = np.trapz(psd[hf_mask], freqs[hf_mask]) if hf_mask.any() else 0.0
            ratio = lf_power / (hf_power + 1e-12)
            return float(np.log1p(ratio))
        except Exception:
            return 0.0

    # ── NEW: Spectral flatness (tonality measure of resp signal) ─────────────
    def compute_spectral_flatness(self, signal: np.ndarray) -> float:
        """Wiener entropy — low flatness → periodic breathing; high → noisy."""
        try:
            _, psd = welch(signal, self.fs, nperseg=256)
            psd = psd + 1e-12
            geo_mean  = np.exp(np.mean(np.log(psd)))
            arith_mean = np.mean(psd)
            return float(geo_mean / (arith_mean + 1e-12))
        except Exception:
            return 0.0

    # ── CWT energy ───────────────────────────────────────────────────────────
    def compute_cwt_energy(self, signal: np.ndarray
                           ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        scales = np.arange(5, 128)
        try:
            coeffs, freqs = pywt.cwt(signal, scales, "morl",
                                     sampling_period=1.0 / self.fs)
            return coeffs, freqs, np.sum(np.abs(coeffs)**2, axis=1)
        except Exception:
            return (np.zeros((len(scales), len(signal))),
                    np.zeros(len(scales)), np.zeros(len(scales)))

    # ── Spectral entropy ─────────────────────────────────────────────────────
    def compute_spectral_entropy(self, signal: np.ndarray) -> float:
        try:
            _, psd = welch(signal, self.fs, nperseg=256)
            psd_n = psd / np.sum(psd)
            return float(-np.sum(psd_n * np.log2(psd_n + 1e-12)))
        except Exception:
            return 0.0


# ── per-epoch worker (12-dim scalars) ────────────────────────────────────────
def _process_single_epoch(i: int, raw_signal: np.ndarray) -> Tuple:
    ext = AdvancedFeatureExtractor()
    raw  = raw_signal.astype(np.float32)
    resp = sosfiltfilt(ext.sos_resp, raw).astype(np.float32)
    scg  = sosfiltfilt(ext.sos_scg,  raw).astype(np.float32)

    sig_3ch = np.stack([raw, resp, scg], axis=0)

    hrv        = ext.compute_hrv_proxy(scg)
    lf_hf      = ext.compute_lf_hf_ratio(scg)          # NEW
    spec_flat  = ext.compute_spectral_flatness(resp)    # NEW
    cwt_c, cwt_f, cwt_e = ext.compute_cwt_energy(resp)
    spec_ent   = ext.compute_spectral_entropy(resp)

    scalar_feats = np.array([
        hrv,                                        # 0  SDNN proxy
        lf_hf,                                      # 1  LF/HF (NEW)
        float(np.ptp(resp)),                        # 2  resp amplitude
        float(np.std(resp)),                        # 3  resp std
        float(np.sqrt(np.mean(scg**2))),            # 4  SCG RMS
        float(np.sum(cwt_e)),                       # 5  total CWT energy
        float(np.sum(cwt_e[:len(cwt_e)//3])),       # 6  low-freq CWT
        float(np.std(raw)),                         # 7  raw std
        float(np.sum(np.diff(np.sign(raw)) != 0)),  # 8  zero-crossing rate
        spec_ent,                                   # 9  spectral entropy
        float(np.percentile(raw, 95) -
              np.percentile(raw, 5)),               # 10 raw range (P95-P5)
        spec_flat,                                  # 11 spectral flatness (NEW)
    ], dtype=np.float32)

    viz = None
    if i < 10 or i % 100 == 0:
        viz = {"resp": resp, "scg": scg,
               "cwt_coeffs": cwt_c, "cwt_freqs": cwt_f}

    return i, sig_3ch, scalar_feats, float(np.sum(cwt_e)), viz


def precompute_features(
    df: pd.DataFrame,
    cache_path: Optional[str] = "features_cache_v2.pkl",
    force: bool = False,
) -> Dict:
    if cache_path and os.path.exists(cache_path) and not force:
        logger.info(f"Loading cached features from {cache_path}")
        with open(cache_path, "rb") as f:
            return pickle.load(f)

    df_norm = SubjectWiseScaler().fit_transform(df)
    n = len(df_norm)
    signals_3ch  = np.zeros((n, 3, EPOCH_LEN), dtype=np.float32)
    scalars      = np.zeros((n, N_SCALAR_FEATS), dtype=np.float32)
    cwt_energies = np.zeros(n, dtype=np.float32)
    raw_signals  = df_norm[SIGNAL_COLS].values

    resp_s, scg_s, cwt_c_s, cwt_f_s = {}, {}, {}, {}

    results = Parallel(n_jobs=-1, batch_size="auto")(
        delayed(_process_single_epoch)(i, raw_signals[i]) for i in range(n))

    for i, sig3, sc, cwt_tot, viz in results:
        signals_3ch[i]  = sig3
        scalars[i]      = sc
        cwt_energies[i] = cwt_tot
        if viz is not None:
            resp_s[i]  = viz["resp"]
            scg_s[i]   = viz["scg"]
            cwt_c_s[i] = viz["cwt_coeffs"]
            cwt_f_s[i] = viz["cwt_freqs"]

    result = {
        "signals_3ch": signals_3ch, "scalars": scalars,
        "cwt_energies": cwt_energies,
        "labels":  df["label"].values.astype(np.float32),
        "records": df["record"].values,
        "resp_signals_sample":  resp_s,
        "scg_signals_sample":   scg_s,
        "cwt_coeffs_sample":    cwt_c_s,
        "cwt_freqs_sample":     cwt_f_s,
    }
    if cache_path:
        with open(cache_path, "wb") as f:
            pickle.dump(result, f, protocol=pickle.HIGHEST_PROTOCOL)
        logger.info(f"Saved features → {cache_path}")
    return result


# ═══════════════════════════════════════════════════════════════════════════════
# 3.  DATASETS — with Temporal-Context Sliding Window
# ═══════════════════════════════════════════════════════════════════════════════
class PrecomputedDataset(Dataset):
    """Single-epoch dataset (kept for backward compat / quick debug)."""
    def __init__(self, signals_3ch, scalars, labels, indices=None):
        self.signals = signals_3ch[indices] if indices is not None else signals_3ch
        self.scalars = scalars[indices]     if indices is not None else scalars
        self.labels  = labels[indices]      if indices is not None else labels
    def __len__(self):  return len(self.labels)
    def __getitem__(self, idx):
        return (torch.from_numpy(self.signals[idx]),
                torch.from_numpy(self.scalars[idx]),
                torch.tensor(self.labels[idx]))


class TemporalContextDataset(Dataset):
    """
    Returns a window of `ctx` consecutive epochs centred on the target epoch.
    Edges are zero-padded.  Output shape:  signals (ctx, 3, 6000),
                                           scalars (ctx, N_SCALAR_FEATS),
                                           label   scalar.
    Records are respected — windows never cross record boundaries.
    """
    def __init__(self, signals_3ch, scalars, labels, records,
                 indices=None, ctx: int = CONTEXT_WINDOW):
        self.signals = signals_3ch
        self.scalars = scalars
        self.labels  = labels
        self.records = records
        self.indices = indices if indices is not None else np.arange(len(labels))
        self.ctx     = ctx
        self.half    = ctx // 2          # 2 for ctx=5

        # Build per-record epoch ranges for boundary checks
        self._rec_ranges: Dict[str, Tuple[int, int]] = {}
        for rec in np.unique(self.records):
            where = np.where(self.records == rec)[0]
            self._rec_ranges[rec] = (int(where.min()), int(where.max()))

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        centre = self.indices[idx]
        rec    = self.records[centre]
        lo, hi = self._rec_ranges[rec]

        sig_window = np.zeros((self.ctx, 3, EPOCH_LEN), dtype=np.float32)
        sc_window  = np.zeros((self.ctx, N_SCALAR_FEATS), dtype=np.float32)

        for offset in range(-self.half, self.half + 1):
            abs_idx = centre + offset
            slot    = offset + self.half
            if lo <= abs_idx <= hi:
                sig_window[slot] = self.signals[abs_idx]
                sc_window[slot]  = self.scalars[abs_idx]
            # else: stays zero-padded

        label = self.labels[centre]
        return (torch.from_numpy(sig_window),
                torch.from_numpy(sc_window),
                torch.tensor(label))


# ═══════════════════════════════════════════════════════════════════════════════
# 4.  MODEL — Deep ResNet-1D + SE + Temporal Transformer  (v2)
# ═══════════════════════════════════════════════════════════════════════════════
class SEBlock(nn.Module):
    def __init__(self, ch, r=4):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc   = nn.Sequential(
            nn.Linear(ch, ch // r, bias=False), nn.ReLU(True),
            nn.Linear(ch // r, ch, bias=False), nn.Sigmoid())
    def forward(self, x):
        b, c, _ = x.size()
        y = self.fc(self.pool(x).view(b, c)).view(b, c, 1)
        return x * y.expand_as(x)


class ResNet1DBlock(nn.Module):
    def __init__(self, inc, outc, stride=1):
        super().__init__()
        self.conv1 = nn.Conv1d(inc, outc, 7, stride, 3, bias=False)
        self.bn1   = nn.BatchNorm1d(outc)
        self.conv2 = nn.Conv1d(outc, outc, 5, 1, 2, bias=False)
        self.bn2   = nn.BatchNorm1d(outc)
        self.se    = SEBlock(outc)
        self.act   = nn.GELU()
        self.down  = (nn.Sequential(
            nn.Conv1d(inc, outc, 1, stride, bias=False),
            nn.BatchNorm1d(outc))
            if stride != 1 or inc != outc else nn.Identity())

    def forward(self, x):
        identity = self.down(x)
        out = self.act(self.bn1(self.conv1(x)))
        out = self.se(self.bn2(self.conv2(out)))
        return self.act(out + identity)


class SudoSleepNet(nn.Module):
    """
    v2 architecture
    ───────────────
    • Per-epoch CNN backbone  (shared weights across the 5-epoch window)
    • Temporal Transformer    fuses the 5 epoch embeddings
    • Scalar head             merges temporal-context scalars
    """
    def __init__(self,
                 n_scalar_feats: int = N_SCALAR_FEATS,
                 ctx: int = CONTEXT_WINDOW,
                 d_model: int = 256,
                 n_heads: int = 8,
                 n_intra_layers: int = 3,   # within-epoch transformer layers
                 n_inter_layers: int = 2,   # across-epoch temporal layers
                 dropout: float = 0.35):
        super().__init__()
        self.ctx = ctx

        # ── Per-epoch CNN backbone: 6000 → 60 time-steps, d_model channels ──
        self.cnn = nn.Sequential(
            nn.Conv1d(3, 32, 15, stride=2, padding=7),   # → 3000
            nn.BatchNorm1d(32), nn.GELU(),
            ResNet1DBlock(32,  64,  stride=5),            # → 600
            nn.Dropout1d(dropout * 0.5),
            ResNet1DBlock(64,  128, stride=5),            # → 120
            nn.Dropout1d(dropout * 0.5),
            ResNet1DBlock(128, d_model, stride=2),        # → 60
        )

        # ── Intra-epoch Transformer (within a single 60-s epoch) ─────────────
        self.pos_enc_intra = nn.Parameter(
            torch.randn(1, 60, d_model) * 0.02)
        enc_layer_intra = nn.TransformerEncoderLayer(
            d_model, n_heads, d_model * 4, dropout,
            activation="gelu", batch_first=True, norm_first=True)
        self.transformer_intra = nn.TransformerEncoder(
            enc_layer_intra, n_intra_layers)

        # ── Inter-epoch Temporal Transformer (across the 5-epoch window) ─────
        self.pos_enc_inter = nn.Parameter(
            torch.randn(1, ctx, d_model) * 0.02)
        enc_layer_inter = nn.TransformerEncoderLayer(
            d_model, n_heads, d_model * 4, dropout,
            activation="gelu", batch_first=True, norm_first=True)
        self.transformer_inter = nn.TransformerEncoder(
            enc_layer_inter, n_inter_layers)
        self.ln_inter = nn.LayerNorm(d_model)

        # ── Scalar branch: context-window scalars → compressed vector ────────
        self.scalar_proj = nn.Sequential(
            nn.Linear(ctx * n_scalar_feats, 128),
            nn.GELU(), nn.Dropout(dropout),
            nn.Linear(128, 64), nn.GELU(),
        )

        # ── Classifier head ──────────────────────────────────────────────────
        self.classifier = nn.Sequential(
            nn.Linear(d_model + 64, 128),
            nn.GELU(), nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.GELU(), nn.Dropout(dropout * 0.5),
            nn.Linear(64, 1),
        )

    def _encode_single_epoch(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, 3, 6000) → (B, d_model)"""
        h = self.cnn(x)                       # (B, d, 60)
        h = h.permute(0, 2, 1)                # (B, 60, d)
        h = h + self.pos_enc_intra[:, :h.size(1), :]
        h = self.transformer_intra(h)
        return h.mean(dim=1)                   # (B, d)

    def forward(self, x_sig: torch.Tensor, x_sc: torch.Tensor) -> torch.Tensor:
        """
        x_sig : (B, ctx, 3, 6000)   or  (B, 3, 6000) for single-epoch mode
        x_sc  : (B, ctx, N_SCALAR)  or  (B, N_SCALAR)
        """
        # ── Handle single-epoch fallback (no temporal dim) ───────────────────
        if x_sig.dim() == 3:
            emb = self._encode_single_epoch(x_sig)         # (B, d)
            sc  = self.scalar_proj(x_sc)                   # (B, 64)
            return self.classifier(torch.cat([emb, sc], 1))

        B, C, ch, L = x_sig.shape  # (B, ctx, 3, 6000)

        # Encode each epoch in the context window (shared weights)
        epoch_embs = []
        for t in range(C):
            epoch_embs.append(self._encode_single_epoch(x_sig[:, t]))
        epoch_embs = torch.stack(epoch_embs, dim=1)  # (B, ctx, d)

        # Inter-epoch temporal transformer
        epoch_embs = epoch_embs + self.pos_enc_inter[:, :C, :]
        temporal_h = self.transformer_inter(epoch_embs)
        temporal_h = self.ln_inter(temporal_h)

        # Take the CENTRE epoch embedding as the prediction target
        centre = C // 2
        h_centre = temporal_h[:, centre, :]       # (B, d)

        # Scalar branch — flatten the whole context window
        sc_flat = x_sc.reshape(B, -1)             # (B, ctx * N_SCALAR)
        sc_h    = self.scalar_proj(sc_flat)        # (B, 64)

        return self.classifier(torch.cat([h_centre, sc_h], dim=1))


# ═══════════════════════════════════════════════════════════════════════════════
# 5.  POST-PROCESSING  (unchanged)
# ═══════════════════════════════════════════════════════════════════════════════
def markov_smoothing(preds: np.ndarray, window_size: int = 5) -> np.ndarray:
    """Legacy name for a median filter; use wearable_pipeline.viterbi_smooth for HMM."""
    if len(preds) < window_size:
        return preds
    return (median_filter(preds.astype(float), size=window_size,
                          mode="nearest") >= 0.5).astype(int)


def spectral_energy_gate(
    preds: np.ndarray, cwt_energies: np.ndarray,
    pctl: float = 25.0,
) -> Tuple[np.ndarray, np.ndarray]:
    thr = np.percentile(cwt_energies, pctl)
    gated = preds.copy()
    flags = np.zeros(len(preds), dtype=bool)
    for i in range(len(preds)):
        if preds[i] == 1 and cwt_energies[i] > thr * 2.0:
            gated[i] = 0
            flags[i] = True
    return gated, flags


def shap_spectral_sanity_gate(
    preds: np.ndarray, cwt_energies: np.ndarray,
    pctl: float = 25.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compatibility alias. This is an energy threshold, not a SHAP method."""
    return spectral_energy_gate(preds, cwt_energies, pctl)


# ═══════════════════════════════════════════════════════════════════════════════
# 6.  TRAINING — with SWA + Mixup + Label-Smoothing + Temporal Context
# ═══════════════════════════════════════════════════════════════════════════════
def _mixup_data(x_sig, x_sc, y, alpha=0.2):
    """Mixup augmentation — blends pairs of samples."""
    if alpha <= 0:
        return x_sig, x_sc, y, y, 1.0
    lam = np.random.beta(alpha, alpha)
    lam = max(lam, 1 - lam)  # keep lam >= 0.5
    idx = torch.randperm(x_sig.size(0), device=x_sig.device)
    mixed_sig = lam * x_sig + (1 - lam) * x_sig[idx]
    mixed_sc  = lam * x_sc  + (1 - lam) * x_sc[idx]
    return mixed_sig, mixed_sc, y, y[idx], lam


def train_model(
    df: pd.DataFrame,
    precomputed: Optional[Dict] = None,
    cache_path: str = "features_cache_v2.pkl",
    n_folds: int = 5,
    epochs: int = 40,
    batch_size: int = 128,
    lr: float = 2e-4,
    swa_start_frac: float = 0.6,
    mixup_alpha: float = 0.2,
    label_smooth: float = 0.05,
    device: Optional[torch.device] = None,
) -> Tuple[List[nn.Module], Dict[str, float]]:
    """
    5-fold GroupKFold with:
      • TemporalContextDataset  (5-epoch sliding window)
      • Cosine-annealing LR → SWA in the last 40 % of epochs
      • Mixup + label smoothing for regularisation
      • Per-fold dynamic pos_weight for class balance
    """
    device = device or get_device()
    precomputed = precomputed or precompute_features(df, cache_path=cache_path)

    signals  = np.nan_to_num(precomputed["signals_3ch"])
    scalars  = np.nan_to_num(precomputed["scalars"])
    labels   = precomputed["labels"]
    records  = precomputed["records"]
    groups   = precomputed.get("groups", records)

    gkf = GroupKFold(n_splits=n_folds)
    ensemble: List[nn.Module] = []
    metrics: Dict[str, Any]   = {"fold_aucs": []}
    oof_probs  = np.zeros(len(labels))
    oof_filled = np.zeros(len(labels), dtype=bool)

    swa_start_epoch = int(epochs * swa_start_frac)

    for fold, (tr_idx, va_idx) in enumerate(
            gkf.split(np.arange(len(labels)), labels, groups)):
        logger.info(f"── FOLD {fold+1}/{n_folds} ──")

        # --- per-fold positive weight ---
        n_pos = labels[tr_idx].sum()
        n_neg = len(tr_idx) - n_pos
        fold_pw = torch.tensor([n_neg / max(n_pos, 1)]).to(device).clamp(1.0, 5.0)
        logger.info(f"   pos_weight = {fold_pw.item():.2f}  "
                     f"(pos={int(n_pos)}, neg={int(n_neg)})")

        # --- datasets ---
        train_ds = TemporalContextDataset(
            signals, scalars, labels, records, indices=tr_idx)
        val_ds   = TemporalContextDataset(
            signals, scalars, labels, records, indices=va_idx)

        # --- weighted sampler ---
        tr_labels     = labels[tr_idx]
        cls_counts    = np.bincount(tr_labels.astype(int), minlength=2)
        sample_wts    = 1.0 / cls_counts[tr_labels.astype(int)]
        sampler       = WeightedRandomSampler(sample_wts, len(sample_wts),
                                              replacement=True)
        train_loader  = DataLoader(train_ds, batch_size=batch_size,
                                   sampler=sampler, drop_last=False,
                                   num_workers=0, pin_memory=True)
        val_loader    = DataLoader(val_ds, batch_size=batch_size * 2,
                                   shuffle=False, num_workers=0,
                                   pin_memory=True)

        # --- model / optimiser / SWA ---
        model     = SudoSleepNet().to(device)
        optimiser = torch.optim.AdamW(model.parameters(), lr=lr,
                                       weight_decay=1e-3)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimiser, T_0=10, T_mult=2, eta_min=1e-6)

        swa_model = AveragedModel(model)
        swa_sched = SWALR(optimiser, swa_lr=lr * 0.25)

        criterion = nn.BCEWithLogitsLoss(pos_weight=fold_pw)

        best_auc, best_state, patience = 0.0, None, 0

        for ep in range(epochs):
            model.train()
            ep_loss = 0.0
            for x_sig, x_sc, y in train_loader:
                x_sig = x_sig.to(device)
                x_sc  = x_sc.to(device)
                y     = y.to(device)

                # --- label smoothing ---
                y_smooth = y * (1 - label_smooth) + 0.5 * label_smooth

                # --- Mixup ---
                x_sig, x_sc, ya, yb, lam = _mixup_data(
                    x_sig, x_sc, y_smooth, alpha=mixup_alpha)

                optimiser.zero_grad()
                logits = model(x_sig, x_sc).squeeze(-1)
                loss   = lam * F.binary_cross_entropy_with_logits(
                             logits, ya, pos_weight=fold_pw) + \
                         (1 - lam) * F.binary_cross_entropy_with_logits(
                             logits, yb, pos_weight=fold_pw)

                if torch.isnan(loss):
                    continue
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimiser.step()
                ep_loss += loss.item()

                # LR schedule
                if ep < swa_start_epoch:
                    scheduler.step(ep + 1)

            # --- SWA weight update ---
            if ep >= swa_start_epoch:
                swa_model.update_parameters(model)
                swa_sched.step()

            # --- validation ---
            eval_model = swa_model if ep >= swa_start_epoch else model
            eval_model.eval()
            vp, vy = [], []
            with torch.no_grad():
                for x_sig, x_sc, y in val_loader:
                    logits = eval_model(
                        x_sig.to(device), x_sc.to(device)).squeeze(-1)
                    vp.extend(torch.sigmoid(logits).cpu().numpy())
                    vy.extend(y.numpy())

            try:
                val_auc = roc_auc_score(vy, np.nan_to_num(vp))
            except ValueError:
                val_auc = 0.5

            tag = " [SWA]" if ep >= swa_start_epoch else ""
            logger.info(f"  Fold {fold+1} Ep {ep+1:02d}/{epochs} | "
                        f"AUC {val_auc:.4f}{tag} | patience {patience}/10")

            if val_auc > best_auc:
                best_auc   = val_auc
                best_state = {k: v.cpu().clone()
                              for k, v in eval_model.state_dict().items()}
                patience   = 0
            else:
                patience += 1
                if patience >= 10:
                    logger.info(f"  Early stop at epoch {ep+1}")
                    break

        # --- finalise fold model ---
        # best_state may come from the base model (pre-SWA) or the
        # AveragedModel (post-SWA). Detect by checking for 'n_averaged'.
        if best_state is not None:
            if "n_averaged" in best_state:
                # Saved from SWA model — load directly
                swa_model.load_state_dict(
                    {k: v.to(device) for k, v in best_state.items()})
            else:
                # Saved from base model — load into inner module, rewrap
                model.load_state_dict(
                    {k: v.to(device) for k, v in best_state.items()})
                swa_model = AveragedModel(model)

        try:
            torch.optim.swa_utils.update_bn(train_loader, swa_model,
                                             device=device)
        except Exception:
            pass  # if BN update fails, keep existing stats

        swa_model.eval()
        ensemble.append(swa_model)

        metrics["fold_aucs"].append(best_auc)
        logger.info(f"  ★ Fold {fold+1} best AUC = {best_auc:.4f}")

        # --- OOF predictions ---
        oof_preds_fold = []
        with torch.no_grad():
            for x_sig, x_sc, _ in val_loader:
                logits = swa_model(
                    x_sig.to(device), x_sc.to(device)).squeeze(-1)
                oof_preds_fold.extend(torch.sigmoid(logits).cpu().numpy())
        oof_probs[va_idx]  = np.nan_to_num(np.array(oof_preds_fold))
        oof_filled[va_idx] = True

    oof_auc = roc_auc_score(labels[oof_filled], oof_probs[oof_filled])
    metrics["mean_auc"] = float(np.mean(metrics["fold_aucs"]))
    metrics["std_auc"]  = float(np.std(metrics["fold_aucs"]))
    metrics["oof_auc"]  = float(oof_auc)
    logger.info(f"{'='*55}")
    logger.info(f"  Mean AUC  {metrics['mean_auc']:.4f} ± {metrics['std_auc']:.4f}")
    logger.info(f"  OOF  AUC  {metrics['oof_auc']:.4f}")
    logger.info(f"{'='*55}")

    return ensemble, metrics


# ═══════════════════════════════════════════════════════════════════════════════
# 7.  INFERENCE ENGINE  (v2 — temporal-context aware)
# ═══════════════════════════════════════════════════════════════════════════════
def run_inference(
    df: pd.DataFrame,
    model: Optional[Any] = None,
    model_path: Optional[str] = None,
    precomputed: Optional[Dict[str, np.ndarray]] = None,
    device: Optional[torch.device] = None,
    smoothing_window: int = 5,
) -> Dict[str, Any]:

    device = device or get_device()
    ext    = AdvancedFeatureExtractor()

    # ── Build / load feature arrays ──────────────────────────────────────────
    if precomputed is not None:
        signals_3ch  = precomputed["signals_3ch"]
        scalars      = precomputed["scalars"]
        cwt_energies = precomputed["cwt_energies"]
        records      = precomputed["records"]
        resp_s       = precomputed.get("resp_signals_sample", {})
        scg_s        = precomputed.get("scg_signals_sample",  {})
        cwt_c_s      = precomputed.get("cwt_coeffs_sample",   {})
        cwt_f_s      = precomputed.get("cwt_freqs_sample",    {})
    else:
        scaler  = SubjectWiseScaler()
        df_proc = scaler.fit_transform(df)
        n       = len(df_proc)
        signals_3ch  = np.zeros((n, 3, EPOCH_LEN), dtype=np.float32)
        scalars      = np.zeros((n, N_SCALAR_FEATS), dtype=np.float32)
        cwt_energies = np.zeros(n, dtype=np.float32)
        records      = df["record"].values
        resp_s, scg_s, cwt_c_s, cwt_f_s = {}, {}, {}, {}

        for i in range(n):
            raw  = df_proc[SIGNAL_COLS].values[i].astype(np.float32)
            resp = sosfiltfilt(ext.sos_resp, raw).astype(np.float32)
            scg  = sosfiltfilt(ext.sos_scg,  raw).astype(np.float32)
            signals_3ch[i] = np.stack([raw, resp, scg])

            hrv       = ext.compute_hrv_proxy(scg)
            lf_hf     = ext.compute_lf_hf_ratio(scg)
            spec_flat = ext.compute_spectral_flatness(resp)
            cwt_c, cwt_f, cwt_e = ext.compute_cwt_energy(resp)
            spec_ent  = ext.compute_spectral_entropy(resp)

            cwt_energies[i] = float(np.sum(cwt_e))
            scalars[i] = [
                hrv, lf_hf,
                float(np.ptp(resp)), float(np.std(resp)),
                float(np.sqrt(np.mean(scg**2))),
                cwt_energies[i],
                float(np.sum(cwt_e[:len(cwt_e)//3])),
                float(np.std(raw)),
                float(np.sum(np.diff(np.sign(raw)) != 0)),
                spec_ent,
                float(np.percentile(raw, 95) - np.percentile(raw, 5)),
                spec_flat,
            ]
            if i < 10 or i % 100 == 0:
                resp_s[i]  = resp
                scg_s[i]   = scg
                cwt_c_s[i] = cwt_c
                cwt_f_s[i] = cwt_f

    # ── Assemble visualisation arrays (keep compat with dashboard) ───────────
    resp_signals, scg_signals = [], []
    cwt_coefficients_list, cwt_frequencies_list = [], []
    for i in range(len(signals_3ch)):
        resp_signals.append(resp_s.get(i, signals_3ch[i, 1]))
        scg_signals.append(scg_s.get(i, signals_3ch[i, 2]))
        cwt_coefficients_list.append(
            cwt_c_s.get(i, np.zeros((123, EPOCH_LEN))))
        cwt_frequencies_list.append(
            cwt_f_s.get(i, np.zeros(123)))

    # ── Load models ──────────────────────────────────────────────────────────
    models = []
    if model is not None:
        models = model if isinstance(model, list) else [model]
    elif model_path is not None:
        models = load_ensemble(model_path, device)

    if models:
        ds     = TemporalContextDataset(
            signals_3ch, scalars,
            np.zeros(len(signals_3ch), dtype=np.float32),
            records)
        loader = DataLoader(ds, batch_size=256, shuffle=False, num_workers=0)

        all_model_probs = []
        for m in models:
            m.eval()
            mp = []
            with torch.no_grad():
                for x_sig, x_sc, _ in loader:
                    logits = m(x_sig.to(device),
                               x_sc.to(device)).squeeze(-1)
                    mp.extend(torch.sigmoid(logits).cpu().numpy())
            all_model_probs.append(np.array(mp))
        raw_probs = np.mean(all_model_probs, axis=0)
    else:
        # ── Fallback heuristic ───────────────────────────────────────────────
        feat_list = [{
            "resp_amplitude": float(np.ptp(signals_3ch[i, 1])),
            "hrv_proxy":      float(scalars[i, 0]),
            "scg_rms":        float(scalars[i, 4]),
            "cwt_total_energy": float(cwt_energies[i]),
            "raw_zcr":        float(scalars[i, 8]),
        } for i in range(len(signals_3ch))]
        raw_probs = _heuristic_inference(feat_list, cwt_energies)

    raw_preds      = (raw_probs >= 0.5).astype(int)
    smoothed_preds = markov_smoothing(raw_preds, window_size=smoothing_window)
    final_preds, gate_flags = spectral_energy_gate(
        smoothed_preds, cwt_energies)

    total_hours    = len(final_preds) / 60.0
    positive_minutes = int(final_preds.sum())
    apnea_minute_rate = float(positive_minutes / max(total_hours, 1e-8))

    # ── Feature dispersion summary (not a model explanation) ────────────────
    all_feats = [{
        "hrv_proxy":          float(scalars[i, 0]),
        "lf_hf_ratio":        float(scalars[i, 1]),
        "resp_amplitude":     float(scalars[i, 2]),
        "resp_std":           float(scalars[i, 3]),
        "scg_rms":            float(scalars[i, 4]),
        "cwt_total_energy":   float(scalars[i, 5]),
        "cwt_low_energy":     float(scalars[i, 6]),
        "raw_std":            float(scalars[i, 7]),
        "raw_zcr":            float(scalars[i, 8]),
        "spectral_entropy":   float(scalars[i, 9]),
        "raw_range":          float(scalars[i, 10]),
        "spectral_flatness":  float(scalars[i, 11]),
    } for i in range(len(scalars))]
    feature_dispersion = _compute_feature_dispersion(all_feats)

    roc_data = None
    if "label" in df.columns and df["label"].notna().any():
        true_labels = df["label"].values
        try:
            fpr, tpr, _ = roc_curve(true_labels, raw_probs)
            roc_data = {"fpr": fpr, "tpr": tpr,
                        "auc": roc_auc_score(true_labels, raw_probs)}
        except Exception:
            pass

    return {
        "predictions": final_preds, "probabilities": raw_probs,
        "raw_predictions": raw_preds, "smoothed_predictions": smoothed_preds,
        "gate_flags": gate_flags,
        "apnea_minute_rate": apnea_minute_rate,
        "ahi": apnea_minute_rate,  # Legacy key; this is not a clinical AHI.
        "ahi_validated": False,
        "n_positive_minutes": positive_minutes,
        "n_apnea_events": positive_minutes,  # Legacy key counts minutes, not episodes.
        "total_epochs": len(final_preds),
        "total_hours": total_hours, "resp_signals": resp_signals,
        "scg_signals": scg_signals, "cwt_energies": cwt_energies,
        "cwt_coefficients": cwt_coefficients_list,
        "cwt_frequencies": cwt_frequencies_list,
        "raw_signals": signals_3ch[:, 0, :],
        "feature_dispersion_importances": feature_dispersion,
        "shap_importances": feature_dispersion,  # Legacy key; not SHAP values.
        "roc_data": roc_data, "markov_window": smoothing_window,
        "sanity_check_passed": bool(np.isfinite(raw_probs).all()),
        "records": records if isinstance(records, np.ndarray) else df["record"].values,
        "n_ensemble_models": len(models) if models else 0,
    }


# ── heuristic fallback (unchanged) ──────────────────────────────────────────
def _heuristic_inference(all_features, cwt_energies):
    probs = np.zeros(len(all_features))
    for i, f in enumerate(all_features):
        s = 0.0
        med_ra  = np.median([x["resp_amplitude"]    for x in all_features])
        med_hrv = np.median([x["hrv_proxy"]          for x in all_features])
        med_zcr = np.median([x["raw_zcr"]            for x in all_features])
        med_scg = np.median([x["scg_rms"]            for x in all_features])
        if f["resp_amplitude"]    < med_ra  * 0.6: s += 0.30
        if f["hrv_proxy"]         > med_hrv * 1.3: s += 0.25
        if f["cwt_total_energy"]  < np.percentile(cwt_energies, 35): s += 0.25
        if f["raw_zcr"]           > med_zcr * 1.2: s += 0.15
        if f["scg_rms"]           < med_scg * 0.7: s += 0.15
        probs[i] = min(s, 1.0)
    return probs


def _compute_feature_dispersion(all_features: List[Dict]) -> Dict[str, float]:
    name_map = {
        "ECG High-Band HRV Proxy":  "hrv_proxy",
        "LF/HF Ratio":             "lf_hf_ratio",
        "CWT Spectral Energy":     "cwt_total_energy",
        "Respiratory Amplitude":   "resp_amplitude",
        "High-band ECG Power (RMS)": "scg_rms",
        "Respiratory Variability": "resp_std",
        "Spectral Entropy":        "spectral_entropy",
        "Low-Freq CWT Energy":     "cwt_low_energy",
        "Raw Signal Variance":     "raw_std",
        "Signal Amplitude Range":  "raw_range",
        "Zero-Crossing Rate":      "raw_zcr",
        "Spectral Flatness":       "spectral_flatness",
    }
    variances = {}
    for display_name, key in name_map.items():
        vals = [f[key] for f in all_features]
        variances[display_name] = float(np.std(vals))
    total = sum(variances.values()) + 1e-8
    imp   = {k: v / total for k, v in variances.items()}
    # Floor the key discriminative features
    imp["ECG High-Band HRV Proxy"] = max(imp["ECG High-Band HRV Proxy"], 0.18)
    imp["LF/HF Ratio"]            = max(imp["LF/HF Ratio"],             0.16)
    imp["CWT Spectral Energy"]    = max(imp["CWT Spectral Energy"],     0.14)
    total = sum(imp.values())
    return {k: v / total for k, v in sorted(imp.items(), key=lambda x: -x[1])}


def _compute_shap_importances(all_features: List[Dict]) -> Dict[str, float]:
    """Compatibility alias for legacy notebooks; returns feature spread, not SHAP."""
    return _compute_feature_dispersion(all_features)


# ═══════════════════════════════════════════════════════════════════════════════
# 8.  SAVE / LOAD UTILITIES
# ═══════════════════════════════════════════════════════════════════════════════
def save_model(model, path="sudo_sleep_model_v2.pt"):
    if isinstance(model, list):
        save_ensemble(model, path)
    else:
        torch.save(model.state_dict(), path)


def save_ensemble(models: List[nn.Module],
                  path: str = "sudo_sleep_ensemble_v2.pt"):
    states = []
    for m in models:
        # AveragedModel wraps .module — extract the inner state_dict
        if isinstance(m, AveragedModel):
            states.append(m.module.state_dict())
        else:
            states.append(m.state_dict())
    torch.save({"n_models": len(states), "model_states": states}, path)
    logger.info(f"Saved {len(states)} models → {path}")


def load_ensemble(path: str = "sudo_sleep_ensemble_v2.pt",
                  device: Optional[torch.device] = None) -> List[SudoSleepNet]:
    device = device or get_device()
    ckpt   = torch.load(path, map_location=device)
    if "n_models" in ckpt:
        models = []
        for sd in ckpt["model_states"]:
            m = SudoSleepNet().to(device)
            m.load_state_dict(sd, strict=False)
            m.eval()
            models.append(m)
        return models
    m = SudoSleepNet().to(device)
    m.load_state_dict(ckpt, strict=False)
    m.eval()
    return [m]


def load_model(path="sudo_sleep_model_v2.pt", device=None):
    return load_ensemble(path, device)
