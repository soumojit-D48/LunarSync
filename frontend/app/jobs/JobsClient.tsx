"use client";

import { useEffect, useState, type ReactNode } from "react";
import { useRouter } from "next/navigation";
import { ArrowRight, Upload } from "lucide-react";
import { createJob, getBackendMode, listJobs, type Job, type MatcherType, type TransformModel } from "@/lib/api-client";
import { MATCHER_LABELS } from "@/lib/mock-data";
import { JobStatusBadge } from "@/components/jobs/JobStatusBadge";
import { ConsoleNav } from "@/components/ConsoleNav";

const SOURCES = ["OHRC", "TMC-2", "IIRS"] as const;
const REFERENCES = ["LRO NAC", "SELENE TC"] as const;

function Field({ label, children }: { label: string; children: ReactNode }) {
  return (
    <label className="block">
      <span className="mb-1.5 block font-mono text-[10px] tracking-[0.16em] text-ash">{label}</span>
      {children}
    </label>
  );
}

const inputCls =
  "w-full rounded-md bg-void/60 px-3 py-2.5 font-mono text-xs text-bone ring-1 ring-line outline-none transition-colors focus:ring-signal";

function NewJobCard({ onCreated }: { onCreated: (id: string) => void }) {
  const [srcName, setSrcName] = useState("");
  const [srcFile, setSrcFile] = useState<File | null>(null);
  const [source, setSource] = useState<(typeof SOURCES)[number]>("OHRC");
  const [reference, setReference] = useState<(typeof REFERENCES)[number]>("LRO NAC");
  const [matcher, setMatcher] = useState<MatcherType>("superpoint-superglue");
  const [model, setModel] = useState<TransformModel>("homography");
  const [busy, setBusy] = useState(false);
  const [submitError, setSubmitError] = useState<string | null>(null);

  const submit = async () => {
    setBusy(true);
    setSubmitError(null);
    try {
      const { jobId } = await createJob({
        pairLabel: `${source} ↔ ${reference} · ${srcName || "uploaded frame"}`,
        matcherType: matcher,
        transformModel: model,
        sourceFile: srcFile ?? undefined,
        sourceSensor: source,
        referenceSensor: reference,
      });
      onCreated(jobId);
    } catch (e) {
      setSubmitError(e instanceof Error ? e.message : "Upload failed — is the live backend running?");
    } finally {
      setBusy(false);
    }
  };

  return (
    <div className="panel-glass rounded-xl p-5 ring-1 ring-white/10 sm:p-6">
      <div className="mb-4 flex items-center gap-2">
        <Upload className="size-4 text-signal" />
        <h2 className="font-mono text-xs tracking-[0.18em] text-bone">NEW REGISTRATION</h2>
      </div>
      <label className="block cursor-pointer rounded-lg border border-dashed border-line p-6 text-center transition-colors hover:border-signal/60 hover:bg-signal/5">
        <span className="mb-1 block font-mono text-[10px] tracking-[0.16em] text-ash">SOURCE IMAGE · UPLOAD</span>
        <span className="block truncate font-mono text-xs text-bone">{srcName || "Drop your Chandrayaan-2 frame here or click to browse"}</span>
        <span className="mt-1 block font-mono text-[10px] text-ash">GEOTIFF · PDS/IMG · PNG · JPG</span>
        <input type="file" accept=".tif,.tiff,.img,.lbl,.png,.jpg" className="hidden" onChange={(e) => { const f = e.target.files?.[0] ?? null; setSrcFile(f); setSrcName(f?.name ?? ""); }} />
      </label>
      <p className="mt-3 font-mono text-[10px] leading-relaxed tracking-[0.08em] text-ash">
        {srcFile
          ? "LIVE BACKEND — YOUR FILE IS SWEPT AGAINST THE ACTIVE REFERENCE SET (SEE MODE BADGE ABOVE)."
          : "REFERENCE FRAMES COME FROM THE ACTIVE BACKEND SET — UPLOAD ONLY YOUR SOURCE FRAME."}
      </p>
      <div className="mt-4 grid gap-4 sm:grid-cols-2">
        <Field label="SOURCE SENSOR">
          <select value={source} onChange={(e) => setSource(e.target.value as typeof source)} className={inputCls}>
            {SOURCES.map((s) => <option key={s}>{s}</option>)}
          </select>
        </Field>
        <Field label="REFERENCE · FROM ARCHIVE">
          <select value={reference} onChange={(e) => setReference(e.target.value as typeof reference)} className={inputCls}>
            {REFERENCES.map((s) => <option key={s}>{s}</option>)}
          </select>
        </Field>
        <Field label="MATCHER">
          <select value={matcher} onChange={(e) => setMatcher(e.target.value as MatcherType)} className={inputCls}>
            {(Object.keys(MATCHER_LABELS) as MatcherType[]).map((m) => (
              <option key={m} value={m}>{MATCHER_LABELS[m]}</option>
            ))}
          </select>
        </Field>
        <Field label="TRANSFORM MODEL">
          <select value={model} onChange={(e) => setModel(e.target.value as TransformModel)} className={inputCls}>
            {(["similarity", "affine", "homography"] as const).map((m) => <option key={m}>{m}</option>)}
          </select>
        </Field>
      </div>
      <button
        type="button"
        onClick={submit}
        disabled={busy}
        className="group mt-5 inline-flex items-center gap-2 rounded-md bg-signal px-5 py-3 font-mono text-sm font-semibold text-void ring-1 ring-signal/40 transition-colors hover:bg-bone disabled:opacity-50"
      >
        {busy ? "QUEUING…" : "Start matching"}
        <ArrowRight className="size-4 transition-transform group-hover:translate-x-0.5" />
      </button>
      {submitError ? (
        <p className="mt-3 font-mono text-[11px] leading-relaxed text-destructive">{submitError}</p>
      ) : null}
    </div>
  );
}

export default function JobsPage() {
  const router = useRouter();
  const [jobs, setJobs] = useState<Job[] | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [mode, setMode] = useState<string>("…");
  const [refCount, setRefCount] = useState<number | null>(null);

  useEffect(() => {
    let cancelled = false;
    listJobs()
      .then(({ jobs }) => {
        if (!cancelled) setJobs(jobs);
      })
      .catch((e) => {
        if (!cancelled) setError(e instanceof Error ? e.message : "failed");
      });
    getBackendMode()
      .then((m) => {
        if (cancelled) return;
        setMode(m.mode ?? "unknown");
        const n = m.cloud?.seeded_count;
        if (typeof n === "number") setRefCount(n);
      })
      .catch(() => {});
    return () => {
      cancelled = true;
    };
  }, []);

  return (
    <div className="relative min-h-screen bg-void text-bone">
      <ConsoleNav />
      <div className="mx-auto max-w-7xl px-5 py-10 sm:px-8">
        <p className="font-mono text-[11px] tracking-[0.2em] text-signal">CONSOLE — JOBS</p>
        <h1 className="mt-2 text-3xl font-semibold tracking-tight sm:text-4xl">Registration jobs</h1>
        <p className="mt-2 max-w-[60ch] text-sm text-mist">
          Every source↔reference pair queued through the pipeline, with live status,
          match evidence and evaluation on the detail page.
        </p>

        <div className="mt-8">
          <NewJobCard onCreated={(id) => router.push(`/jobs/${id}`)} />
        </div>

        <div className="mt-8 overflow-hidden rounded-xl ring-1 ring-white/10">
          <div className="grid grid-cols-[1fr_auto] gap-3 border-b border-line/60 bg-surface/60 px-4 py-3 font-mono text-[10px] tracking-[0.16em] text-ash sm:grid-cols-[1fr_180px_150px_90px_90px]">
            <span>PAIR</span>
            <span className="hidden sm:block">MATCHER</span>
            <span className="hidden sm:block">STATUS</span>
            <span className="hidden text-right sm:block">RMSE</span>
            <span className="hidden text-right sm:block">INLIERS</span>
          </div>
          {error ? (
            <p className="px-4 py-8 font-mono text-xs text-destructive">Failed to load jobs: {error}</p>
          ) : !jobs ? (
            <p className="px-4 py-8 font-mono text-xs text-mist">Loading jobs…</p>
          ) : (
            jobs.map((j) => (
              <button
                key={j.id}
                type="button"
                onClick={() => router.push(`/jobs/${j.id}`)}
                className="grid w-full grid-cols-[1fr_auto] items-center gap-3 border-b border-line/40 px-4 py-3.5 text-left transition-colors last:border-0 hover:bg-signal/5 sm:grid-cols-[1fr_180px_150px_90px_90px]"
              >
                <span>
                  <span className="block text-sm font-medium text-bone">{j.pairLabel}</span>
                  <span className="mt-0.5 block font-mono text-[10px] text-ash">
                    {j.id} · {j.meta.sourceSensor} {j.meta.sourceGsdM} m/px → {j.meta.referenceSensor} {j.meta.referenceGsdM} m/px
                  </span>
                </span>
                <span className="hidden font-mono text-[11px] text-mist sm:block">{MATCHER_LABELS[j.matcherType]}</span>
                <span className="hidden sm:block"><JobStatusBadge status={j.status} /></span>
                <span className="hidden text-right font-mono text-xs tabular text-bone sm:block">—</span>
                <span className="hidden text-right font-mono text-xs tabular text-bone sm:block">—</span>
                <span className="sm:hidden"><JobStatusBadge status={j.status} /></span>
              </button>
            ))
          )}
        </div>
      </div>
    </div>
  );
}
