"""Train a subject-held-out DREAMT accelerometer prototype on local caches.

Targets rely on a provisional sparse-annotation interpretation documented by
prepare_dreamt.py. All predictions, weights and reports remain under data/.
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (average_precision_score, confusion_matrix, f1_score,
                             precision_recall_curve, precision_score, recall_score, roc_auc_score)
import torch
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from pipeline import SudoSleepNet, get_device
from wearable_pipeline import FEATURE_NAMES, estimate_transition_matrix, viterbi_smooth

ROOT = Path("data/dreamt/DREAMT")
DERIVED = ROOT / "derived"


class NightWindows(Dataset):
    def __init__(self, subjects: list[str]):
        self.nights = []
        self.indices = []
        self.targets = []
        self.center = self.scale = None
        for subject in subjects:
            path = DERIVED / "features" / f"{subject}.npz"
            if not path.is_file():
                raise FileNotFoundError(f"Missing local feature cache for {subject}; run prepare_dreamt.py.")
            with np.load(path, allow_pickle=False) as f:
                version = str(f["preprocessing_version"])
                if version != "emd_scg_v2_rate_aware":
                    raise ValueError(f"Incompatible feature cache: {subject} ({version}).")
                signals = f["signals"]
                scalars = np.nan_to_num(f["scalars"], nan=0.0, posinf=0.0, neginf=0.0)
                labels = f["labels"]
                valid_seconds = f["valid_seconds"]
                sleep_fraction = f["sleep_fraction"]
            night_id = len(self.nights)
            self.nights.append({"subject": subject, "signals": signals, "scalars": scalars,
                                "labels": labels, "valid_seconds": valid_seconds,
                                "sleep_fraction": sleep_fraction})
            for i in np.flatnonzero(labels >= 0):
                self.indices.append((night_id, int(i)))
                self.targets.append(int(labels[i]))
        self.targets = np.asarray(self.targets, dtype=np.float32)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        night_id, center = self.indices[index]
        night = self.nights[night_id]
        signal = np.zeros((5, 3, 6000), dtype=np.float32)
        scalar = np.zeros((5, len(FEATURE_NAMES)), dtype=np.float32)
        for slot, i in enumerate(range(center - 2, center + 3)):
            if 0 <= i < len(night["labels"]):
                signal[slot] = night["signals"][i]
                values = night["scalars"][i]
                scalar[slot] = np.clip((values - self.center) / self.scale, -10, 10) if self.center is not None else values
        return torch.from_numpy(signal), torch.from_numpy(scalar), torch.tensor(self.targets[index])

    def scalar_matrix(self):
        values = np.stack([self.nights[n]["scalars"][i] for n, i in self.indices])
        return np.clip((values - self.center) / self.scale, -10, 10) if self.center is not None else values


def probabilities(model, dataset, device, batch_size):
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    output = []
    model.eval()
    with torch.no_grad():
        for signal, scalar, _ in loader:
            output.extend(torch.sigmoid(model(signal.to(device), scalar.to(device)).squeeze(-1)).cpu().numpy())
    return np.asarray(output, dtype=float)


def choose_threshold(labels, scores):
    precision, recall, thresholds = precision_recall_curve(labels, scores)
    f1 = 2 * precision[:-1] * recall[:-1] / np.maximum(precision[:-1] + recall[:-1], 1e-12)
    return float(thresholds[int(np.argmax(f1))]) if len(thresholds) else 0.5


def metrics(labels, scores, threshold):
    predictions = (scores >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(labels, predictions, labels=[0, 1]).ravel()
    return {"n_windows": int(len(labels)), "positive_windows": int(labels.sum()),
            "auroc": float(roc_auc_score(labels, scores)) if len(np.unique(labels)) == 2 else None,
            "auprc": float(average_precision_score(labels, scores)) if len(np.unique(labels)) == 2 else None,
            "threshold": float(threshold), "precision": float(precision_score(labels, predictions, zero_division=0)),
            "recall": float(recall_score(labels, predictions, zero_division=0)),
            "specificity": float(tn / max(tn + fp, 1)), "f1": float(f1_score(labels, predictions, zero_division=0)),
            "confusion": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)}}


def per_subject(dataset, scores, threshold):
    metadata = pd.read_csv(ROOT / "participant_info.csv", usecols=["SID", "AHI"]).set_index("SID")
    output = []
    offset = 0
    for night in dataset.nights:
        valid = np.flatnonzero(night["labels"] >= 0)
        selected = scores[offset:offset + len(valid)]
        offset += len(valid)
        sequence = np.zeros(len(night["labels"]), dtype=bool)
        sequence[valid] = selected >= threshold
        runs = int(np.sum(sequence & np.r_[True, ~sequence[:-1]]))
        sleep_hours = float(np.sum(night["valid_seconds"][valid] * night["sleep_fraction"][valid]) / 3600)
        output.append({"subject": night["subject"], "eligible_windows": len(valid),
                       "positive_windows": int(np.sum(night["labels"][valid] == 1)),
                       "candidate_runs": runs, "eligible_sleep_hours": sleep_hours,
                       "candidate_runs_per_hour": runs / sleep_hours if sleep_hours else None,
                       "reference_ahi": float(metadata.loc[night["subject"], "AHI"])})
    return output


def smooth_by_night(dataset, scores, transition):
    """Decode separate contiguous scored-sleep blocks without crossing gaps."""
    result = np.zeros(len(scores), dtype=np.int8)
    offset = 0
    for night in dataset.nights:
        valid = np.flatnonzero(night["labels"] >= 0)
        for group in np.split(np.arange(len(valid)), np.flatnonzero(np.diff(valid) != 1) + 1):
            if len(group):
                positions = offset + group
                result[positions] = viterbi_smooth(scores[positions], transition)
        offset += len(valid)
    return result


def intervals_from_mask(mask, step=60.0):
    mask = np.asarray(mask, dtype=bool)
    starts = np.flatnonzero(mask & np.r_[True, ~mask[:-1]])
    ends = np.flatnonzero(mask & np.r_[~mask[1:], True]) + 1
    return [(float(start * step), float(end * step)) for start, end in zip(starts, ends)]


def event_match_counts(reference, predicted):
    """Greedy one-to-one matching on at least one second of temporal overlap."""
    edges = []
    for i, (a, b) in enumerate(reference):
        for j, (c, d) in enumerate(predicted):
            overlap = min(b, d) - max(a, c)
            if overlap >= 1.0:
                edges.append((overlap, i, j))
    used_reference, used_predicted = set(), set()
    for _, i, j in sorted(edges, reverse=True):
        if i not in used_reference and j not in used_predicted:
            used_reference.add(i)
            used_predicted.add(j)
    tp = len(used_reference)
    return tp, len(predicted) - tp, len(reference) - tp


def reference_intervals(subject, eligible):
    """Read marked PSG events intersecting eligible scored-sleep windows."""
    from audit_dreamt import EVENTS
    path = ROOT / "data_64Hz" / f"{subject}_whole_df.csv"
    columns = ["TIMESTAMP", "Sleep_Stage", *EVENTS]
    intervals = []
    open_start = None
    last_time = None
    with pd.read_csv(path, usecols=columns, chunksize=200_000, low_memory=False) as reader:
        for chunk in reader:
            t = pd.to_numeric(chunk["TIMESTAMP"], errors="raise").to_numpy(dtype=float)
            events = np.zeros(len(chunk), dtype=bool)
            for name in EVENTS:
                events |= pd.to_numeric(chunk[name], errors="coerce").to_numpy(dtype=float) == 1
            epoch = np.floor(t / 60).astype(int)
            valid_index = (epoch >= 0) & (epoch < len(eligible))
            in_eligible = np.zeros(len(epoch), dtype=bool)
            in_eligible[valid_index] = eligible[epoch[valid_index]]
            events &= in_eligible & chunk["Sleep_Stage"].isin(["N1", "N2", "N3", "R"]).to_numpy()
            prev = np.r_[open_start is not None, events[:-1]]
            for i in np.flatnonzero(events != prev):
                if events[i]:
                    open_start = float(t[i])
                elif open_start is not None:
                    intervals.append((open_start, float(t[i])))
                    open_start = None
            if len(t):
                last_time = float(t[-1])
    if open_start is not None and last_time is not None:
        intervals.append((open_start, last_time + 1 / 64.0))
    return intervals


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--max-hours", type=float, default=4)
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1 or args.max_hours <= 0:
        raise SystemExit("Epochs, batch size and time budget must be positive.")
    torch.manual_seed(42)
    np.random.seed(42)
    split = json.loads((DERIVED / "subject_splits.json").read_text())
    train, val, test = [NightWindows(split[k]) for k in ("train", "validation", "test")]
    if any(len(np.unique(ds.targets)) < 2 for ds in (train, val, test)):
        raise SystemExit("Each split must contain both positive and negative eligible windows.")
    # Fit scalar normalization only on training subjects; zero-padding then
    # corresponds to the training median rather than an extreme raw value.
    train_scalars = train.scalar_matrix()
    scalar_center = np.median(train_scalars, axis=0).astype(np.float32)
    scalar_scale = (np.percentile(train_scalars, 75, axis=0) - np.percentile(train_scalars, 25, axis=0)).astype(np.float32)
    scalar_scale[scalar_scale < 1e-6] = 1.0
    del train_scalars
    for dataset in (train, val, test):
        dataset.center, dataset.scale = scalar_center, scalar_scale
    print("Eligible windows train/val/test:", len(train), len(val), len(test), flush=True)
    device = get_device()
    print("Device:", device, "| provisional blank-label assumption; subject split seed 42", flush=True)

    # Baseline: scalar features; validation determines threshold, never test.
    baseline = RandomForestClassifier(n_estimators=120, min_samples_leaf=5, class_weight="balanced_subsample",
                                      random_state=42, n_jobs=2)
    baseline.fit(train.scalar_matrix(), train.targets)
    val_base = baseline.predict_proba(val.scalar_matrix())[:, 1]
    base_threshold = choose_threshold(val.targets, val_base)
    joblib.dump(baseline, DERIVED / "baseline_random_forest.joblib")

    counts = np.bincount(train.targets.astype(int), minlength=2)
    weights = 1.0 / counts[train.targets.astype(int)]
    sampler = WeightedRandomSampler(weights, len(weights), replacement=True,
                                    generator=torch.Generator().manual_seed(42))
    loader = DataLoader(train, batch_size=args.batch_size, sampler=sampler, num_workers=0)
    model = SudoSleepNet().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-3)
    best_state = None
    best_ap = -1.0
    history = []
    patience = 0
    start = time.monotonic()
    for epoch in range(args.epochs):
        model.train()
        total, batches = 0.0, 0
        for signal, scalar, label in loader:
            optimizer.zero_grad(set_to_none=True)
            loss = torch.nn.functional.binary_cross_entropy_with_logits(
                model(signal.to(device), scalar.to(device)).squeeze(-1), label.to(device))
            if not torch.isfinite(loss):
                raise RuntimeError("Non-finite training loss; aborting rather than saving invalid weights.")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total += float(loss.detach().cpu())
            batches += 1
            if batches % 500 == 0:
                print(f"epoch {epoch+1} batch {batches}/{len(loader)} loss {total / batches:.4f}", flush=True)
            if time.monotonic() - start >= args.max_hours * 3600:
                break
        val_scores = probabilities(model, val, device, args.batch_size * 2)
        ap = float(average_precision_score(val.targets, val_scores))
        history.append({"epoch": epoch + 1, "batches": batches, "mean_loss": total / max(batches, 1),
                        "validation_auprc": ap, "elapsed_seconds": round(time.monotonic() - start, 1)})
        print("Validation:", history[-1], flush=True)
        if ap > best_ap:
            best_ap = ap
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience = 0
            torch.save(best_state, DERIVED / "best_dreamt_state.pt")
        else:
            patience += 1
        if patience >= 3 or time.monotonic() - start >= args.max_hours * 3600:
            break
    if best_state is None:
        raise RuntimeError("No valid trained model was produced.")
    model.load_state_dict(best_state)
    model.eval()
    val_scores = probabilities(model, val, device, args.batch_size * 2)
    threshold = choose_threshold(val.targets, val_scores)
    train_labels = np.concatenate([n["labels"] for n in train.nights])
    train_records = np.concatenate([np.full(len(n["labels"]), n["subject"]) for n in train.nights])
    transition = estimate_transition_matrix(train_labels, train_records).tolist()
    # This is the only held-out test evaluation, after model and thresholds are frozen.
    test_scores = probabilities(model, test, device, args.batch_size * 2)
    test_baseline = baseline.predict_proba(test.scalar_matrix())[:, 1]
    test_states = smooth_by_night(test, test_scores, transition)
    per_night = per_subject(test, test_states, 0.5)
    finite_pairs = [(x["candidate_runs_per_hour"], x["reference_ahi"]) for x in per_night
                    if x["candidate_runs_per_hour"] is not None and np.isfinite(x["reference_ahi"])]
    ahi_mae = float(np.mean([abs(a - b) for a, b in finite_pairs])) if finite_pairs else None
    event_tp = event_fp = event_fn = 0
    prediction_rows = []
    offset = 0
    for night in test.nights:
        valid = np.flatnonzero(night["labels"] >= 0)
        states = np.zeros(len(night["labels"]), dtype=bool)
        states[valid] = test_states[offset:offset + len(valid)]
        ref = reference_intervals(night["subject"], night["labels"] >= 0)
        tp, fp, fn = event_match_counts(ref, intervals_from_mask(states))
        event_tp += tp
        event_fp += fp
        event_fn += fn
        for i, score, state in zip(valid, test_scores[offset:offset + len(valid)], test_states[offset:offset + len(valid)]):
            prediction_rows.append({"subject": night["subject"], "minute": int(i), "label": int(night["labels"][i]),
                                    "score": float(score), "smoothed_state": int(state)})
        offset += len(valid)
    pd.DataFrame(prediction_rows).to_csv(DERIVED / "test_predictions.csv", index=False)
    smooth_metrics = metrics(test.targets, test_states.astype(float), 0.5)
    smooth_metrics["auroc"] = smooth_metrics["auprc"] = None
    artifact = {"artifact_version": 2, "input_modality": "accelerometer",
                "preprocessing_version": "emd_scg_v2_rate_aware", "architecture": "SudoSleepNet",
                "feature_names": FEATURE_NAMES, "model_states": [best_state],
                "transition_matrix": transition, "threshold": threshold,
                "scalar_center": scalar_center.astype(float).tolist(),
                "scalar_scale": scalar_scale.astype(float).tolist(),
                "scalar_background": np.median(train.scalar_matrix(), axis=0).astype(float).tolist(),
                "data_provenance": "DREAMT 2.2; provisional blank-as-no-marked-event assumption"}
    torch.save(artifact, DERIVED / "dreamt_accelerometer.pt")
    report = {"label_caveat": split["label_semantics"], "excluded": split["excluded"],
              "note": "One-minute targets indicate at least one second of marked event during predominantly scored sleep; AHI-like run rate is coarse and not clinical AHI.",
              "training_history": history, "validation": metrics(val.targets, val_scores, threshold),
              "test": metrics(test.targets, test_scores, threshold), "smoothed_test": smooth_metrics,
              "baseline_test": metrics(test.targets, test_baseline, base_threshold),
              "event_matching": {"true_positive": event_tp, "false_positive": event_fp, "false_negative": event_fn,
                                 "precision": event_tp / max(event_tp + event_fp, 1),
                                 "recall": event_tp / max(event_tp + event_fn, 1),
                                 "definition": "Greedy one-to-one >=1 second overlap; predictions use minute boundaries and reference events are restricted to eligible scored-sleep minutes."},
              "test_candidate_rate_mae_vs_ahi": ahi_mae, "per_subject": per_night}
    (DERIVED / "dreamt_results.json").write_text(json.dumps(report, indent=2))
    print("Test aggregate:", json.dumps(report["test"]), flush=True)
    print("Saved local model and report to", DERIVED, flush=True)


if __name__ == "__main__":
    main()
