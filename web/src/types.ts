export type Metric = {
  n_windows: number;
  positive_windows: number;
  auroc: number;
  auprc: number;
  precision: number;
  recall: number;
  specificity: number;
  f1: number;
  threshold: number;
  confusion: { tn: number; fp: number; fn: number; tp: number };
};
export type Subject = {
  subject: string;
  split: string;
  minutes: number;
  sleep_hours: number;
  reference_ahi: number;
};
export type Point = { minute: number; value: number };
export type Signal = { name: string; points: Point[]; available: boolean };
export type Spectral = {
  frequency: number[];
  minutes: number[];
  power: number[][];
};
export type Segment = {
  subject: string;
  start: number;
  stop: number;
  backend: string;
  signals: Signal[];
  spectrogram: Spectral;
  stages: { minute: number; stage: string }[];
  events: { type: string; start: number; end: number }[];
  windows: {
    minute: number;
    score: number;
    state: number;
    target: number;
    features: number[];
  }[];
  feature_names: string[];
  note: string;
};
export type Overview = {
  report: {
    test: Metric;
    baseline_test: Metric;
    smoothed_test: Metric;
    label_caveat: string;
    note: string;
    test_candidate_rate_mae_vs_ahi: number;
    event_matching: {
      precision: number;
      recall: number;
      true_positive: number;
      false_positive: number;
      false_negative: number;
      definition: string;
    };
    training_history: {
      epoch: number;
      mean_loss: number;
      validation_auprc: number;
      elapsed_seconds: number;
    }[];
    per_subject: {
      subject: string;
      candidate_runs_per_hour: number;
      reference_ahi: number;
      eligible_windows: number;
    }[];
  };
  split: {
    train: string[];
    validation: string[];
    test: string[];
    excluded: string[];
    label_semantics: string;
  };
  status: { stage: string; updated_utc: string };
  audit: {
    subjects: number;
    samples: number;
    hours: number;
    gaps: number;
    invalid: number;
  };
  curves: { roc: { x: number; y: number }[]; pr: { x: number; y: number }[] };
};
export async function api<T>(path: string, init?: RequestInit): Promise<T> {
  const r = await fetch(`/api/${path}`, init);
  if (!r.ok) {
    const e = await r.json().catch(() => ({ detail: r.statusText }));
    throw new Error(e.detail || "Request failed");
  }
  return r.json();
}
