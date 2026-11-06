import { useEffect, useRef, useState } from "react";
import {
  Activity,
  ArrowDownToLine,
  ArrowRight,
  ArrowUpRight,
  Check,
  FlaskConical,
  Info,
  LoaderCircle,
  Moon,
  Play,
  RefreshCw,
  SlidersHorizontal,
  Upload,
  X,
} from "lucide-react";
import {
  ResponsiveContainer,
  LineChart,
  Line,
  XAxis,
  YAxis,
  Tooltip,
  CartesianGrid,
  ScatterChart,
  Scatter,
  ReferenceLine,
} from "recharts";
import {
  api,
  type Overview,
  type Subject,
  type Segment,
  type Signal,
  type Spectral,
} from "./types";
import { Curve, SignalChart, Spectrogram } from "./Charts";

type View = "night" | "study" | "drivers" | "methods";
type Explanation = {
  features: { name: string; value: number }[];
  base: number;
  score: number;
};
const fmt = (v: number | null | undefined, d = 3) =>
  v === undefined || v === null ? "—" : v.toFixed(d);
const pct = (v: number) => `${(v * 100).toFixed(1)}%`;
const clock = (minute: number) =>
  `${Math.floor(minute / 60)
    .toString()
    .padStart(2, "0")}:${(minute % 60).toString().padStart(2, "0")}`;
const NAV: { id: View; name: string; detail: string }[] = [
  { id: "night", name: "Night desk", detail: "Explore a recording" },
  { id: "study", name: "Model study", detail: "Evaluate the evidence" },
  { id: "drivers", name: "Signal drivers", detail: "Inspect a prediction" },
  { id: "methods", name: "Data & methods", detail: "Trace the pipeline" },
];
function Stat({
  label,
  value,
  detail,
}: {
  label: string;
  value: string;
  detail?: string;
}) {
  return (
    <div className="stat">
      <span>{label}</span>
      <strong>{value}</strong>
      {detail ? <small>{detail}</small> : null}
    </div>
  );
}
function Notice({ children }: { children: React.ReactNode }) {
  return (
    <div className="notice">
      <Info size={15} />
      <span>{children}</span>
    </div>
  );
}
export default function App() {
  const [zoom, setZoom] = useState<[number, number] | null>(null);
  const explainRequest = useRef(0);
  const [view, setView] = useState<View>("night");
  const [overview, setOverview] = useState<Overview | null>(null);
  const [subjects, setSubjects] = useState<Subject[]>([]);
  const [subject, setSubject] = useState("S002");
  const [timeline, setTimeline] = useState<
    { minute: number; target: number; sleep_fraction: number }[]
  >([]);
  const [start, setStart] = useState(222);
  const [duration, setDuration] = useState(5);
  const [segment, setSegment] = useState<Segment | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const [initError, setInitError] = useState("");
  const [selectedMinute, setSelectedMinute] = useState(222);
  const [explanation, setExplanation] = useState<Explanation | null>(null);
  const [explainBusy, setExplainBusy] = useState(false);
  const [explainError, setExplainError] = useState("");
  const [settings, setSettings] = useState(false);
  const [sourceResult, setSourceResult] = useState<{
    signals: Signal[];
    spectrogram: Spectral;
    backend: string;
    scores: number[];
  } | null>(null);
  const [sourceBusy, setSourceBusy] = useState(false);
  const [sourceError, setSourceError] = useState("");
  useEffect(() => {
    if (!settings) return;
    const handle = (e: KeyboardEvent) => {
      if (e.key === "Escape") setSettings(false);
      if (e.key === "Tab") {
        const nodes = [
          ...document.querySelectorAll<HTMLElement>(
            ".settings-modal button,.settings-modal a",
          ),
        ];
        const first = nodes[0],
          last = nodes.at(-1);
        if (e.shiftKey && document.activeElement === first) {
          e.preventDefault();
          last?.focus();
        } else if (!e.shiftKey && document.activeElement === last) {
          e.preventDefault();
          first?.focus();
        }
      }
    };
    document.addEventListener("keydown", handle);
    return () => document.removeEventListener("keydown", handle);
  }, [settings]);
  useEffect(() => {
    window.scrollTo({ top: 0, behavior: "instant" });
  }, [view]);
  useEffect(() => {
    const controller = new AbortController();
    Promise.allSettled([
      api<Overview>("overview", { signal: controller.signal }),
      api<Subject[]>("recordings", { signal: controller.signal }),
    ])
      .then(([o, s]) => {
        if (o.status === "fulfilled") setOverview(o.value);
        else if (o.reason.name !== "AbortError") setInitError(o.reason.message);
        if (s.status === "fulfilled") setSubjects(s.value);
        else if (s.reason.name !== "AbortError") setError(s.reason.message);
      })
      .catch((e) => {
        if (e.name !== "AbortError") setInitError(e.message);
      });
    return () => controller.abort();
  }, []);
  useEffect(() => {
    const controller = new AbortController();
    setTimeline([]);
    api<{ timeline: typeof timeline }>(`night/${subject}`, {
      signal: controller.signal,
    })
      .then((d) => setTimeline(d.timeline))
      .catch((e) => {
        if (e.name !== "AbortError") setError(e.message);
      });
    return () => controller.abort();
  }, [subject]);
  async function analyze(s = subject, m = start, d = duration) {
    setBusy(true);
    setError("");
    setExplanation(null);
    setZoom(null);
    try {
      const data = await api<Segment>(`segment/${s}?start=${m}&duration=${d}`);
      setSegment(data);
      setSelectedMinute(data.start);
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setBusy(false);
    }
  }
  useEffect(() => {
    void analyze("S002", 222, 5);
  }, []);
  const current = subjects.find((s) => s.subject === subject);
  const aligned =
    segment?.subject === subject &&
    segment.start === start &&
    segment.stop === Math.min(start + duration, current?.minutes || Infinity);
  const selected = segment?.windows.find((w) => w.minute === selectedMinute);
  const report = overview?.report;
  function changeSubject(s: string) {
    explainRequest.current++;
    setZoom(null);
    setSubject(s);
    setStart(222);
    setExplanation(null);
    setExplainError("");
  }
  function chooseMinute(n: number) {
    explainRequest.current++;
    setSelectedMinute(Math.min((segment?.stop || Infinity) - 1, Math.floor(n)));
    setExplanation(null);
    setExplainError("");
  }
  async function explain() {
    if (!segment) return;
    const request = ++explainRequest.current;
    setExplainBusy(true);
    setExplainError("");
    try {
      const result = await api<Explanation>(
        `explain/${segment.subject}/${selectedMinute}`,
      );
      if (request === explainRequest.current) setExplanation(result);
    } catch (e) {
      if (request === explainRequest.current)
        setExplainError((e as Error).message);
    } finally {
      setExplainBusy(false);
    }
  }
  function download() {
    if (!segment) return;
    const rows = [
      "minute,model_score,smoothed_state,provisional_target",
      ...segment.windows.map(
        (w) => `${w.minute},${w.score},${w.state},${w.target}`,
      ),
    ];
    const url = URL.createObjectURL(
      new Blob([rows.join("\n")], { type: "text/csv" }),
    );
    const a = document.createElement("a");
    a.href = url;
    a.download = `${segment.subject}_segment_predictions.csv`;
    a.click();
    URL.revokeObjectURL(url);
  }
  async function alternative(file?: File) {
    setSourceBusy(true);
    setSourceError("");
    try {
      const form = new FormData();
      if (file) form.append("file", file);
      setSourceResult(
        await api(
          file ? "upload?sample_rate=50" : "demo",
          file ? { method: "POST", body: form } : undefined,
        ),
      );
    } catch (e) {
      setSourceError((e as Error).message);
    } finally {
      setSourceBusy(false);
    }
  }
  return (
    <>
      <header className="masthead">
        <a
          className="brand"
          href="#"
          onClick={(e) => {
            e.preventDefault();
            setView("night");
          }}
        >
          <span className="brand-mark">
            <Moon size={22} />
          </span>
          SudoSleep<span className="brand-edition">RESEARCH / 01</span>
        </a>
        <div className="masthead-right">
          <span className="local-dot" /> Local research workspace{" "}
          <button
            className="icon-button"
            aria-label="Open workspace settings"
            onClick={() => setSettings(true)}
          >
            <SlidersHorizontal size={18} />
          </button>
        </div>
      </header>
      <nav className="research-index" aria-label="Research views">
        {NAV.map((n, i) => (
          <button
            key={n.id}
            className={view === n.id ? "active" : ""}
            onClick={() => setView(n.id)}
            aria-current={view === n.id ? "page" : undefined}
          >
            <span className="index-number">0{i + 1}</span>
            <span>
              <strong>{n.name}</strong>
              <small>{n.detail}</small>
            </span>
            <ArrowUpRight size={17} />
          </button>
        ))}
      </nav>
      <main>
        {initError ? (
          <div className="error" role="alert">
            Could not load research artifacts: {initError}{" "}
            <button onClick={() => location.reload()}>Retry</button>
          </div>
        ) : null}
        <section className="page-heading">
          <div>
            <p className="eyebrow">
              <span className="tiny-line" />
              {view === "night"
                ? "RECORDING EXPLORER"
                : view === "study"
                  ? "HELD-OUT EVALUATION"
                  : view === "drivers"
                    ? "CONDITIONAL EXPLAINABILITY"
                    : "RESEARCH PROVENANCE"}
            </p>
            <h1>
              {view === "night"
                ? "Inside the night."
                : view === "study"
                  ? "Let the evidence speak."
                  : view === "drivers"
                    ? "What moved the prediction?"
                    : "From motion to meaning."}
            </h1>
            <p className="intro">
              {view === "night"
                ? "Follow the signal. Compare the reference. Inspect the prediction."
                : view === "study"
                  ? "One frozen test cohort. Two models. A clear view of where the system stands."
                  : view === "drivers"
                    ? "Inspect scalar feature contributions to a selected minute of the recording."
                    : "The data, assumptions and processing stages behind this experiment."}
            </p>
          </div>
          <div className="heading-stamp">
            <FlaskConical size={20} />
            <span>
              DREAMT 2.2
              <small>Wearable accelerometry · research prototype</small>
            </span>
          </div>
        </section>
        {view === "night" || view === "drivers" ? (
          <>
            <div className="recording-bar">
              <label className="subject-control">
                <span className="eyebrow">RECORDING</span>
                <select
                  aria-label="Recording"
                  value={subject}
                  onChange={(e) => changeSubject(e.target.value)}
                >
                  {subjects.length ? (
                    subjects.map((s) => (
                      <option key={s.subject} value={s.subject}>
                        {s.subject} · {s.split}
                      </option>
                    ))
                  ) : (
                    <option>S002</option>
                  )}
                </select>
              </label>
              <div className="record-meta">
                <span className="pill">
                  {current?.split || "Loading"} subject
                </span>
                <span>
                  {current
                    ? `${(current.minutes / 60).toFixed(1)}h recorded`
                    : "Reading artifacts…"}
                </span>
                <span>
                  {current
                    ? `${current.sleep_hours.toFixed(1)}h PSG sleep`
                    : ""}
                </span>
              </div>
              <label>
                Start minute
                <input
                  aria-label="Start minute"
                  type="number"
                  min={0}
                  max={Math.max(0, (current?.minutes || 600) - duration)}
                  value={start}
                  onChange={(e) =>
                    setStart(Math.max(0, Number(e.target.value)))
                  }
                />
              </label>
              <label>
                Window
                <select
                  aria-label="Segment duration"
                  value={duration}
                  onChange={(e) => setDuration(Number(e.target.value))}
                >
                  {[2, 5, 10, 15].map((v) => (
                    <option key={v} value={v}>
                      {v} minutes
                    </option>
                  ))}
                </select>
              </label>
              <button
                className="primary"
                onClick={() => analyze()}
                disabled={busy}
              >
                {busy ? (
                  <LoaderCircle size={16} className="spin" />
                ) : (
                  <Play size={14} />
                )}{" "}
                {busy ? "Analyzing" : "Analyze segment"}
              </button>
            </div>
            <div className="night-ruler">
              <div className="ruler-label">
                <span>THE WHOLE NIGHT</span>
                <span>Click to move the analysis window</span>
              </div>
              <div
                className="night-strip"
                role="group"
                aria-label="Whole night provisional targets"
              >
                {timeline.map((t) => (
                  <button
                    key={t.minute}
                    tabIndex={t.minute === start ? 0 : -1}
                    onKeyDown={(e) => {
                      if (e.key === "ArrowRight" || e.key === "ArrowLeft") {
                        e.preventDefault();
                        setStart(
                          Math.max(
                            0,
                            Math.min(
                              timeline.length - duration,
                              start + (e.key === "ArrowRight" ? 1 : -1),
                            ),
                          ),
                        );
                      }
                    }}
                    aria-label={`Minute ${t.minute}, ${t.target < 0 ? "not eligible" : t.target === 1 ? "marked event target" : "unmarked sleep target"}`}
                    title={`Minute ${t.minute} · ${t.target < 0 ? "Not eligible" : t.target === 1 ? "Marked event target" : "Unmarked sleep target"}`}
                    className={`${t.target < 0 ? "unknown" : t.target === 1 ? "positive" : "negative"} ${t.minute >= start && t.minute < start + duration ? "selected" : ""}`}
                    onClick={() =>
                      setStart(
                        Math.min(
                          t.minute,
                          Math.max(0, timeline.length - duration),
                        ),
                      )
                    }
                  />
                ))}
              </div>
              <div className="ruler-times">
                <span>00:00</span>
                <span>{clock(Math.floor(timeline.length / 2))}</span>
                <span>{clock(timeline.length)}</span>
              </div>
              <div className="legend">
                <span>
                  <i className="legend-positive" />
                  Marked-event minute
                </span>
                <span>
                  <i className="legend-negative" />
                  Unmarked sleep minute
                </span>
                <span>
                  <i className="legend-unknown" />
                  Outside eligible sleep
                </span>
                <span className="legend-note">
                  Provisional targets · ≥80% scored sleep
                </span>
              </div>
            </div>
            {error ? (
              <div className="error" role="alert">
                {error}
                <button onClick={() => analyze()}>Retry</button>
              </div>
            ) : null}
            {segment && !aligned ? (
              <Notice>
                Controls changed. Analyze the new segment to update the
                displayed results. Currently showing {segment.subject}, minutes{" "}
                {segment.start}–{segment.stop}.
              </Notice>
            ) : null}
            {busy ? (
              <div className="loading-line" role="status">
                <LoaderCircle size={15} className="spin" />
                Reading the recording and running the saved model…
              </div>
            ) : null}
          </>
        ) : null}
        {view === "night" && segment ? (
          <>
            <Notice>
              {segment.windows.filter((w) => w.target === -1).length} of{" "}
              {segment.windows.length} selected minutes fall outside the
              scored-sleep eligibility used for model evaluation. Predictions
              for these minutes are exploratory.
            </Notice>
            <div className="segment-summary">
              <Stat
                label="SELECTED SEGMENT"
                value={`${segment.start}–${segment.stop} min`}
                detail={`${segment.subject} · ${(segment.stop - segment.start).toFixed(0)} minutes`}
              />
              <Stat
                label="MEAN MODEL SCORE"
                value={fmt(
                  segment.windows.reduce((a, w) => a + w.score, 0) /
                    segment.windows.length,
                )}
                detail="Uncalibrated probability"
              />
              <Stat
                label="CANDIDATE MINUTES"
                value={`${segment.windows.filter((w) => w.state === 1).length} / ${segment.windows.length}`}
                detail="After Markov smoothing"
              />
              <Stat
                label="PSG MARKED INTERVALS"
                value={`${segment.events.length}`}
                detail="Across the selected segment"
              />
            </div>
            <section className="workbench">
              <div className="section-top">
                <div>
                  <span className="eyebrow">01 / SIGNAL WORKBENCH</span>
                  <h2>A shared view of the signal.</h2>
                </div>
                <span className="model-badge">
                  <Check size={13} /> CNN + Transformer loaded
                </span>
              </div>
              <div className="zoom-controls">
                <span>
                  Linked signal cursor · click a waveform to select a minute
                </span>
                <button
                  className="text-button"
                  onClick={() =>
                    setZoom([
                      Math.max(segment.start, selectedMinute),
                      Math.min(segment.stop, selectedMinute + 1),
                    ])
                  }
                >
                  Zoom to selected minute
                </button>
                <button
                  className="text-button"
                  onClick={() => setZoom(null)}
                  disabled={!zoom}
                >
                  Reset view
                </button>
              </div>
              <div className="signals-layout">
                <div className="signal-stack">
                  {segment.signals.map((signal, i) => (
                    <SignalChart
                      key={signal.name}
                      signal={signal}
                      index={i}
                      domain={zoom || [segment.start, segment.stop]}
                      onMinute={chooseMinute}
                    />
                  ))}
                </div>
                <div className="reading-note">
                  <span className="eyebrow">READING THIS SEGMENT</span>
                  <h3>
                    Movement carries
                    <br />
                    more than motion.
                  </h3>
                  <p>
                    The respiratory and cardiac-band channels are mechanical
                    proxies extracted from wrist acceleration.
                  </p>
                  <div className="note-rule" />
                  <span className="eyebrow">ORIGINAL SENSOR</span>
                  <strong className="sensor-value">
                    32 <small>Hz</small>
                  </strong>
                  <p>
                    The original sampling rate limits the cardiac band to
                    approximately 10–15.4 Hz.
                  </p>
                  <button
                    className="text-button"
                    onClick={() => setView("methods")}
                  >
                    Explore the pipeline <ArrowRight size={15} />
                  </button>
                </div>
              </div>
              <Spectrogram data={segment.spectrogram} />
            </section>
            <section className="workbench">
              <div className="section-top">
                <div>
                  <span className="eyebrow">02 / REFERENCE & PREDICTION</span>
                  <h2>Put the predictions in context.</h2>
                </div>
                <button className="secondary" onClick={download}>
                  <ArrowDownToLine size={15} />
                  Export segment
                </button>
              </div>
              <div className="event-tracks">
                <div className="event-label">PSG stages</div>
                <div className="stage-track">
                  {segment.stages.map((s, i) => (
                    <span
                      key={i}
                      title={s.stage}
                      className={`stage stage-${s.stage}`}
                      style={{ width: `${100 / segment.stages.length}%` }}
                    >
                      {i === 0 || segment.stages[i - 1].stage !== s.stage
                        ? s.stage === "R"
                          ? "REM"
                          : s.stage
                        : ""}
                    </span>
                  ))}
                </div>
                <div className="event-label">PSG events</div>
                <div className="annotation-track">
                  {segment.events.map((e, i) => (
                    <span
                      key={i}
                      title={`${e.type} · minute ${e.start.toFixed(2)}`}
                      style={{
                        left: `${((e.start - segment.start) / (segment.stop - segment.start)) * 100}%`,
                        width: `${Math.max(0.25, ((e.end - e.start) / (segment.stop - segment.start)) * 100)}%`,
                      }}
                    />
                  ))}
                  {segment.events.length === 0 ? (
                    <small>No marked respiratory events in this interval</small>
                  ) : null}
                </div>
                <div className="event-label">Model state</div>
                <div className="state-track">
                  {segment.windows.map((w) => (
                    <button
                      key={w.minute}
                      title={`Minute ${w.minute}: ${w.state ? "candidate" : "normal"}`}
                      className={w.state ? "candidate" : ""}
                      onClick={() => {
                        chooseMinute(w.minute);
                        setView("drivers");
                      }}
                    >
                      {w.state ? "Candidate" : "Normal"}
                    </button>
                  ))}
                </div>
              </div>
              <ResponsiveContainer width="100%" height={220}>
                <LineChart
                  data={segment.windows}
                  margin={{ top: 25, right: 30, bottom: 10, left: -15 }}
                >
                  <CartesianGrid vertical={false} stroke="#e6e5dd" />
                  <XAxis dataKey="minute" tickFormatter={(v) => `${v}m`} />
                  <YAxis domain={[0, 1]} />
                  <Tooltip />
                  <ReferenceLine
                    y={overview?.report.test.threshold}
                    stroke="#bc795c"
                    strokeDasharray="5 4"
                    label="Validation threshold"
                  />
                  <Line
                    dataKey="score"
                    name="Model score"
                    stroke="#247c72"
                    strokeWidth={2}
                    dot={{ r: 4 }}
                  />
                </LineChart>
              </ResponsiveContainer>
              <Notice>
                {segment.note} The displayed rate and candidate states do not
                establish clinical AHI.
              </Notice>
            </section>
            <div className="next-step">
              <div>
                <span className="eyebrow">LOOK BEYOND THIS RECORDING</span>
                <h3>How well does it generalize?</h3>
              </div>
              <button className="text-button" onClick={() => setView("study")}>
                Open the model study <ArrowRight size={18} />
              </button>
            </div>
          </>
        ) : null}
        {view === "study" && report && overview ? (
          <>
            <div className="study-banner">
              <div>
                <span className="eyebrow">COMPLETED EXPERIMENT</span>
                <h2>Held out. Measured. Reported.</h2>
              </div>
              <div className="cohort-split">
                <span>
                  <b>{overview.split.train.length}</b>train
                </span>
                <span>
                  <b>{overview.split.validation.length}</b>validation
                </span>
                <span className="test-split">
                  <b>{overview.split.test.length}</b>test
                </span>
              </div>
            </div>
            <div className="segment-summary">
              <Stat
                label="TEST AUROC"
                value={fmt(report.test.auroc)}
                detail="CNN + Transformer"
              />
              <Stat
                label="TEST AUPRC"
                value={fmt(report.test.auprc)}
                detail={`Positive prevalence ${pct(report.test.positive_windows / report.test.n_windows)}`}
              />
              <Stat
                label="TEST F1"
                value={fmt(report.test.f1)}
                detail={`${report.test.n_windows.toLocaleString()} eligible windows`}
              />
              <Stat
                label="EVENT RECALL"
                value={pct(report.event_matching.recall)}
                detail="Minute-resolution event matching"
              />
            </div>
            <Notice>
              The neural model and random forest perform similarly on window
              classification. Low event recall limits the current event-counting
              approach. Labels rely on a provisional interpretation of blank
              event fields.
            </Notice>
            <section className="workbench">
              <div className="section-top">
                <div>
                  <span className="eyebrow">01 / MODEL COMPARISON</span>
                  <h2>Two approaches, the same test set.</h2>
                </div>
                <span className="pill">Subject-disjoint evaluation</span>
              </div>
              <div className="table-scroll">
                <table>
                  <thead>
                    <tr>
                      <th>Model</th>
                      {[
                        "AUROC",
                        "AUPRC",
                        "Precision",
                        "Recall",
                        "Specificity",
                        "F1",
                      ].map((x) => (
                        <th key={x}>{x}</th>
                      ))}
                    </tr>
                  </thead>
                  <tbody>
                    {[
                      { name: "CNN + Transformer", m: report.test },
                      { name: "Random forest", m: report.baseline_test },
                      {
                        name: "CNN + Transformer + Markov",
                        m: report.smoothed_test,
                      },
                    ].map(({ name, m }) => (
                      <tr key={name}>
                        <td>
                          <span className="table-dot" />
                          {name}
                        </td>
                        {[
                          m.auroc,
                          m.auprc,
                          m.precision,
                          m.recall,
                          m.specificity,
                          m.f1,
                        ].map((x, i) => (
                          <td key={i}>{fmt(x)}</td>
                        ))}
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
              <p className="caption">
                Thresholds chosen on validation subjects. Test data used after
                model selection. The curves below are derived from the saved
                neural-model test predictions.
              </p>
              <div className="two-columns">
                <Curve
                  title="ROC curve"
                  data={overview.curves.roc}
                  xLabel="False positive rate"
                  yLabel="True positive rate"
                  reference
                />
                <Curve
                  title="Precision–recall curve"
                  data={overview.curves.pr}
                  xLabel="Recall"
                  yLabel="Precision"
                  color="#bc795c"
                />
              </div>
            </section>
            <section className="two-columns">
              <div className="workbench">
                <span className="eyebrow">02 / CLASSIFICATION ERRORS</span>
                <h2>Where the model misses.</h2>
                <div className="confusion">
                  <span />
                  <span>Predicted negative</span>
                  <span>Predicted positive</span>
                  <span>Reference negative</span>
                  <div className="correct">
                    <b>{report.test.confusion.tn}</b>
                    <small>True negative</small>
                  </div>
                  <div>
                    <b>{report.test.confusion.fp}</b>
                    <small>False positive</small>
                  </div>
                  <span>Reference positive</span>
                  <div>
                    <b>{report.test.confusion.fn}</b>
                    <small>False negative</small>
                  </div>
                  <div className="correct">
                    <b>{report.test.confusion.tp}</b>
                    <small>True positive</small>
                  </div>
                </div>
                <p className="caption">
                  Reference here means the provisional minute-level target,
                  rather than an independently validated diagnosis.
                </p>
              </div>
              <div className="workbench">
                <span className="eyebrow">03 / EVENT RECOVERY</span>
                <h2>Counting remains the challenge.</h2>
                <div className="event-metrics">
                  <Stat
                    label="PRECISION"
                    value={pct(report.event_matching.precision)}
                  />
                  <Stat
                    label="RECALL"
                    value={pct(report.event_matching.recall)}
                  />
                </div>
                <div className="count-list">
                  <span>
                    Matched events<b>{report.event_matching.true_positive}</b>
                  </span>
                  <span>
                    Unmatched predicted runs
                    <b>{report.event_matching.false_positive}</b>
                  </span>
                  <span>
                    Unmatched reference events
                    <b>{report.event_matching.false_negative}</b>
                  </span>
                </div>
                <p className="caption">{report.event_matching.definition}</p>
              </div>
            </section>
            <section className="workbench">
              <div className="section-top">
                <div>
                  <span className="eyebrow">04 / SUBJECT VARIABILITY</span>
                  <h2>A rate is only part of the story.</h2>
                </div>
                <Stat
                  label="CANDIDATE RATE MAE VS PSG AHI"
                  value={`${fmt(report.test_candidate_rate_mae_vs_ahi, 2)} /hr`}
                />
              </div>
              <div className="two-columns">
                <div>
                  <ResponsiveContainer width="100%" height={280}>
                    <ScatterChart
                      margin={{ top: 15, right: 30, bottom: 20, left: 10 }}
                    >
                      <CartesianGrid stroke="#e6e5dd" />
                      <XAxis
                        type="number"
                        dataKey="reference_ahi"
                        name="PSG AHI"
                      />
                      <YAxis
                        type="number"
                        dataKey="candidate_runs_per_hour"
                        name="Candidate runs/hr"
                      />
                      <Tooltip cursor={{ strokeDasharray: "3 3" }} />
                      <Scatter data={report.per_subject} fill="#247c72" />
                    </ScatterChart>
                  </ResponsiveContainer>
                  <p className="axis-label">
                    PSG AHI (horizontal) · Candidate runs/hour (vertical)
                  </p>
                </div>
                <div className="table-scroll subject-table">
                  <table>
                    <thead>
                      <tr>
                        <th>Subject</th>
                        <th>PSG AHI</th>
                        <th>Candidate /hr</th>
                      </tr>
                    </thead>
                    <tbody>
                      {report.per_subject.map((s) => (
                        <tr key={s.subject}>
                          <td>{s.subject}</td>
                          <td>{fmt(s.reference_ahi, 1)}</td>
                          <td>{fmt(s.candidate_runs_per_hour, 1)}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>
              <Notice>
                Candidate rates use eligible scored-sleep intervals; supplied
                PSG AHI covers the original study. Minute-resolution runs merge
                events and are not a clinical AHI estimate.
              </Notice>
            </section>
            <section className="workbench">
              <span className="eyebrow">05 / TRAINING RECORD</span>
              <h2>The learning trace.</h2>
              <div className="two-columns">
                <div>
                  <h3>Validation AUPRC</h3>
                  <ResponsiveContainer width="100%" height={240}>
                    <LineChart data={report.training_history}>
                      <CartesianGrid stroke="#e6e5dd" />
                      <XAxis dataKey="epoch" />
                      <YAxis domain={[0, 1]} />
                      <Tooltip />
                      <Line
                        dataKey="validation_auprc"
                        stroke="#247c72"
                        strokeWidth={2}
                      />
                    </LineChart>
                  </ResponsiveContainer>
                </div>
                <div>
                  <h3>Mean training loss</h3>
                  <ResponsiveContainer width="100%" height={240}>
                    <LineChart data={report.training_history}>
                      <CartesianGrid stroke="#e6e5dd" />
                      <XAxis dataKey="epoch" />
                      <YAxis domain={["auto", "auto"]} />
                      <Tooltip />
                      <Line
                        dataKey="mean_loss"
                        stroke="#bc795c"
                        strokeWidth={2}
                      />
                    </LineChart>
                  </ResponsiveContainer>
                </div>
              </div>
              <p className="caption">
                {report.training_history.length} recorded epochs · best model
                selected using validation AUPRC · compute time{" "}
                {fmt(
                  (report.training_history.at(-1)?.elapsed_seconds || 0) / 60,
                  1,
                )}{" "}
                minutes.
              </p>
            </section>
          </>
        ) : null}
        {view === "drivers" && segment ? (
          <section className="workbench">
            <div className="section-top">
              <div>
                <span className="eyebrow">CONDITIONAL SCALAR SHAP</span>
                <h2>One minute. Twelve features.</h2>
              </div>
              <label className="minute-picker">
                Minute
                <select
                  value={selectedMinute}
                  onChange={(e) => chooseMinute(Number(e.target.value))}
                >
                  {segment.windows.map((w) => (
                    <option key={w.minute} value={w.minute}>
                      {w.minute}
                    </option>
                  ))}
                </select>
              </label>
            </div>
            <div className="two-columns">
              <div>
                <p className="intro">
                  {segment.subject} · minute {selectedMinute}
                </p>
                <div className="event-metrics">
                  <Stat label="MODEL SCORE" value={fmt(selected?.score)} />
                  <Stat
                    label="MARKOV STATE"
                    value={selected?.state ? "Candidate" : "Normal"}
                  />
                </div>
                <p>
                  Estimate how the scalar features influence this model
                  prediction, holding the waveforms and other minutes fixed. The
                  background is the training median.
                </p>
                <button
                  className="primary"
                  disabled={explainBusy}
                  onClick={explain}
                >
                  {explainBusy ? (
                    <LoaderCircle className="spin" size={16} />
                  ) : (
                    <Activity size={16} />
                  )}{" "}
                  {explainBusy
                    ? "Computing attributions…"
                    : "Explain this minute"}
                </button>
                <p className="caption">
                  Approximate Kernel SHAP explains scalar effects only. It does
                  not explain waveform contributions or establish causation.
                </p>
                {explainError ? (
                  <div className="error" role="alert">
                    {explainError}
                  </div>
                ) : null}
                {explanation ? (
                  <>
                    <p className="equation">
                      {fmt(explanation.base)} <span>background</span> →{" "}
                      <b>{fmt(explanation.score)}</b> <span>prediction</span>
                    </p>
                    {[...explanation.features]
                      .sort((a, b) => Math.abs(b.value) - Math.abs(a.value))
                      .map((f) => (
                        <div className="attribution" key={f.name}>
                          <span>{f.name}</span>
                          <div>
                            <i
                              style={{
                                width: `${Math.max(1, (Math.abs(f.value) / Math.max(...explanation.features.map((x) => Math.abs(x.value)))) * 100)}%`,
                                background: f.value > 0 ? "#bd795b" : "#247c72",
                              }}
                            />
                          </div>
                          <b>
                            {f.value >= 0 ? "+" : ""}
                            {f.value.toFixed(4)}
                          </b>
                        </div>
                      ))}
                  </>
                ) : (
                  <div className="shap-placeholder">
                    <Activity size={32} />
                    <h3>Trace the contributing features.</h3>
                    <p>
                      Run the explanation to see which scalar features raise or
                      lower this minute’s model score.
                    </p>
                  </div>
                )}
              </div>
              <div>
                <h3>Extracted scalar features</h3>
                <table className="feature-table">
                  <thead>
                    <tr>
                      <th>Feature</th>
                      <th>Raw value</th>
                    </tr>
                  </thead>
                  <tbody>
                    {segment.feature_names.map((name, i) => (
                      <tr key={name}>
                        <td>{name}</td>
                        <td>{selected?.features[i]?.toPrecision(5) || "—"}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
                <p className="caption">
                  Raw features shown before the training-only robust scaler.
                  Model inputs are median/IQR scaled and clipped to ±10.
                </p>
                <button
                  className="text-button"
                  onClick={() => setView("night")}
                >
                  Return to the signal <ArrowRight size={16} />
                </button>
              </div>
            </div>
          </section>
        ) : null}
        {view === "methods" && overview ? (
          <>
            <div className="segment-summary">
              <Stat
                label="AUDITED SUBJECTS"
                value={`${overview.audit.subjects}`}
              />
              <Stat
                label="RECORDED HOURS"
                value={fmt(overview.audit.hours, 1)}
              />
              <Stat
                label="SAMPLES CHECKED"
                value={`${(overview.audit.samples / 1e6).toFixed(1)}M`}
              />
              <Stat label="TIMESTAMP GAPS" value={`${overview.audit.gaps}`} />
            </div>
            <section className="workbench">
              <span className="eyebrow">THE IMPLEMENTED PIPELINE</span>
              <h2>A physiological proxy, built from movement.</h2>
              <div className="pipeline">
                {[
                  [
                    "01",
                    "Wrist acceleration",
                    "32 Hz original sensor; distributed at 64/100 Hz.",
                  ],
                  [
                    "02",
                    "Clean & window",
                    "Rate-aware resampling, normalization and 60-second windows.",
                  ],
                  [
                    "03",
                    "Motion proxies",
                    "Respiratory band + EMD-selected cardiac-band components.",
                  ],
                  [
                    "04",
                    "CNN + Transformer",
                    "Local waveform encoding, scalars and five-minute context.",
                  ],
                  [
                    "05",
                    "Markov decoding",
                    "Training-derived transitions smooth minute-level predictions.",
                  ],
                  [
                    "06",
                    "Candidate sequence",
                    "Exploratory event runs, rate comparisons and scalar SHAP.",
                  ],
                ].map(([n, title, body]) => (
                  <div key={n}>
                    <span>{n}</span>
                    <h3>{title}</h3>
                    <p>{body}</p>
                  </div>
                ))}
              </div>
            </section>
            <div className="two-columns">
              <section className="workbench">
                <span className="eyebrow">LABELS & ELIGIBILITY</span>
                <h2>What the targets actually mean.</h2>
                <p>
                  Event fields are encoded as 1 or blank. A value of 1 marks an
                  event; interpreting blanks as no marked event during scored
                  sleep remains provisional.
                </p>
                <p>
                  A minute is eligible with at least 30 seconds of samples and
                  at least 80% scored sleep. A positive target requires at least
                  one second overlapping a marked event during sleep.
                </p>
                <Notice>
                  These motion proxies are not reconstructed ECG. PSG stages
                  provide reference context; they are not used as apnea labels.
                </Notice>
              </section>
              <section className="workbench">
                <span className="eyebrow">AUDIT & SPLIT</span>
                <h2>Keep the subjects apart.</h2>
                <p>
                  The initial cohort excludes{" "}
                  {overview.split.excluded.join(", ")} because marked-event
                  rates differ from supplied AHI by more than 2 events/hour.
                </p>
                <p>
                  Fixed seed 42: {overview.split.train.length} training,{" "}
                  {overview.split.validation.length} validation and{" "}
                  {overview.split.test.length} test subjects. Scaling and
                  transition parameters are fitted on training subjects only.
                </p>
                <div className="count-list">
                  <span>
                    Invalid acceleration rows<b>{overview.audit.invalid}</b>
                  </span>
                  <span>
                    Original accelerometer rate<b>32 Hz</b>
                  </span>
                  <span>
                    Cardiac proxy band<b>~10–15.4 Hz</b>
                  </span>
                </div>
                <a
                  className="text-button"
                  href="https://physionet.org/content/dreamt/2.2.0/"
                  target="_blank"
                  rel="noreferrer"
                >
                  DREAMT dataset documentation <ArrowUpRight size={16} />
                </a>
              </section>
            </div>
            <section className="workbench">
              <span className="eyebrow">OTHER INPUTS</span>
              <h2>Bring a short signal to the desk.</h2>
              <p>
                Upload a CSV with x/y/z axes and optional timestamps (maximum 20
                MB, 15 minutes; fallback 50 Hz), or explore a generated example.
                These paths use the signal heuristic, separately from the
                DREAMT-trained model.
              </p>
              <div className="button-row">
                <label className="secondary upload-button">
                  <Upload size={16} />
                  Upload CSV
                  <input
                    type="file"
                    accept=".csv"
                    disabled={sourceBusy}
                    onChange={(e) => {
                      if (e.target.files?.[0])
                        void alternative(e.target.files[0]);
                    }}
                  />
                </label>
                <button
                  className="secondary"
                  onClick={() => alternative()}
                  disabled={sourceBusy}
                >
                  {sourceBusy ? (
                    <LoaderCircle size={16} className="spin" />
                  ) : (
                    <Play size={15} />
                  )}
                  Synthetic example
                </button>
                <a
                  className="text-button"
                  href="http://127.0.0.1:8501"
                  target="_blank"
                  rel="noreferrer"
                >
                  Legacy Siegen workspace <ArrowUpRight size={15} />
                </a>
              </div>
              {sourceError ? (
                <div className="error" role="alert">
                  {sourceError}
                </div>
              ) : null}
              {sourceResult ? (
                <>
                  <h3>{sourceResult.backend}</h3>
                  <Notice>
                    {sourceResult.backend.includes("synthetic")
                      ? "Generated signals; not recorded patient data."
                      : "Uploaded data are processed locally. Scores are heuristic, not the DREAMT neural model."}
                  </Notice>
                  {sourceResult.signals.map((s, i) => (
                    <SignalChart signal={s} index={i} key={s.name} />
                  ))}
                  <p className="caption">
                    Minute scores:{" "}
                    {sourceResult.scores.map((v) => v.toFixed(3)).join(" · ")}
                  </p>
                </>
              ) : null}
            </section>
          </>
        ) : null}
        {!segment && (view === "night" || view === "drivers") && !busy ? (
          <div className="empty-state">
            <Moon size={28} />
            <h2>Your night desk is ready.</h2>
            <p>Select a recording and analyze a segment to begin.</p>
          </div>
        ) : null}
      </main>
      <footer>
        <span>
          <Moon size={14} /> SudoSleep <b>/</b> Wearable sleep research
        </span>
        <span>Experimental analysis · local data · no clinical diagnosis</span>
        <button onClick={() => setView("methods")}>
          Methods & limitations <ArrowUpRight size={13} />
        </button>
      </footer>
      {settings ? (
        <div className="modal-backdrop" onClick={() => setSettings(false)}>
          <section
            className="settings-modal"
            role="dialog"
            aria-modal="true"
            aria-labelledby="settings-title"
            onClick={(e) => e.stopPropagation()}
          >
            <button
              autoFocus
              className="icon-button modal-close"
              onClick={() => setSettings(false)}
              aria-label="Close workspace settings"
            >
              <X size={20} />
            </button>
            <span className="eyebrow">WORKSPACE</span>
            <h2 id="settings-title">Local, by design.</h2>
            <p>
              The Python API reads your authorized DREAMT files and trained
              artifacts on this Mac. The interface is served locally.
            </p>
            <div className="count-list">
              <span>
                Model backend<b>CNN + Transformer</b>
              </span>
              <span>
                Preparation / training
                <b>{overview?.status.stage || "Unknown"}</b>
              </span>
              <span>
                Neural decision threshold<b>{fmt(report?.test.threshold)}</b>
              </span>
              <span>
                Inference context<b>5 minutes</b>
              </span>
            </div>
            <p className="caption">
              Heuristic persistence controls belong to the legacy workspace. The
              neural model uses its saved transition matrix. Keep restricted
              datasets and subject-level exports local.
            </p>
            <button className="secondary" onClick={() => location.reload()}>
              <RefreshCw size={15} />
              Refresh artifacts
            </button>
          </section>
        </div>
      ) : null}
    </>
  );
}
