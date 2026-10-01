"use client";

import Link from "next/link";
import { ArrowLeft, Download } from "lucide-react";
import { STAGES, MATCHER_LABELS, type JobResult } from "@/lib/mock-data";
import { useJob } from "@/hooks/use-job";
import { JobStatusBadge } from "@/components/jobs/JobStatusBadge";
import { ConsoleNav } from "@/components/ConsoleNav";
import { MatchViewer } from "@/components/viewer/MatchViewer";
import { CoverageGrid, MetricCards } from "@/components/metrics/MetricCards";
import sourceImg from "@/assets/lunar-source.jpg";
import referenceImg from "@/assets/lunar-reference.jpg";

function download(name: string, text: string, type: string) {
  const url = URL.createObjectURL(new Blob([text], { type }));
  const a = document.createElement("a");
  a.href = url;
  a.download = name;
  a.click();
  URL.revokeObjectURL(url);
}

function Downloads({ result }: { result: JobResult }) {
  const csv = [
    "id,src_x,src_y,ref_x,ref_y,confidence,is_inlier",
    ...result.matches.map((m) =>
      [m.id, m.srcX.toFixed(4), m.srcY.toFixed(4), m.refX.toFixed(4), m.refY.toFixed(4), m.confidence.toFixed(3), m.isInlier].join(","),
    ),
  ].join("\n");
  return (
    <div className="flex flex-wrap gap-2">
      <button
        type="button"
        onClick={() => download(`${result.job.id}-matches.csv`, csv, "text/csv")}
        className="inline-flex items-center gap-2 rounded-md px-4 py-2 font-mono text-xs text-bone ring-1 ring-line transition-colors hover:ring-mist"
      >
        <Download className="size-3.5" /> Match points CSV
      </button>
      <button
        type="button"
        onClick={() =>
          download(`${result.job.id}-report.json`, JSON.stringify({ job: result.job, transform: result.transform, report: result.report }, null, 2), "application/json")
        }
        className="inline-flex items-center gap-2 rounded-md px-4 py-2 font-mono text-xs text-bone ring-1 ring-line transition-colors hover:ring-mist"
      >
        <Download className="size-3.5" /> Report JSON
      </button>
    </div>
  );
}

function Progress({ stage }: { stage: string }) {
  const idx = STAGES.indexOf(stage as (typeof STAGES)[number]);
  const pct = idx < 0 ? 4 : Math.round(((idx + 1) / STAGES.length) * 100);
  return (
    <div className="panel-glass rounded-xl p-5 ring-1 ring-white/10">
      <div className="flex items-center justify-between font-mono text-[11px] tracking-[0.14em]">
        <span className="text-ash">PIPELINE STAGE</span>
        <span className="text-signal">{stage.replaceAll("_", " ").toUpperCase()}</span>
      </div>
      <div className="mt-3 h-1.5 overflow-hidden rounded-full bg-void/70">
        <div className="h-full rounded-full bg-signal transition-all duration-700" style={{ width: `${pct}%` }} />
      </div>
      <ol className="mt-4">
        {STAGES.map((s, i) => {
          const done = i < idx;
          const current = i === idx;
          return (
            <li key={s} className="relative flex gap-3 pb-4 last:pb-0">
              {i < STAGES.length - 1 && (
                <span
                  aria-hidden
                  className={`absolute left-[5px] top-4 h-[calc(100%-1rem)] w-px ${i < idx ? "bg-signal/60" : "bg-line/60"}`}
                />
              )}
              <span
                aria-hidden
                className={`mt-1 size-[11px] shrink-0 rounded-full ring-1 ${
                  done
                    ? "bg-signal ring-signal/40"
                    : current
                      ? "reticle bg-signal ring-signal/40"
                      : "bg-void ring-line"
                }`}
              />
              <span className="flex w-full items-center justify-between gap-3">
                <span
                  className={`font-mono text-[11px] tracking-[0.1em] ${current ? "text-signal" : done ? "text-bone" : "text-ash"}`}
                >
                  {s.replaceAll("_", " ").toUpperCase()}
                </span>
                <span className="font-mono text-[10px] text-ash">
                  {done ? "DONE" : current ? "RUNNING…" : `STEP ${i + 1}/${STAGES.length}`}
                </span>
              </span>
            </li>
          );
        })}
      </ol>
      <p className="mt-3 font-mono text-[10px] text-mist">Polling job status every 2s…</p>
    </div>
  );
}

export default function JobDetail({ id }: { id: string }) {
  const { job, result, error, reconnecting } = useJob(id);

  return (
    <div className="relative min-h-screen bg-void text-bone">
      <ConsoleNav />
      <div className="mx-auto max-w-7xl px-5 py-10 sm:px-8">
        <Link href="/jobs" className="inline-flex items-center gap-2 font-mono text-xs text-mist transition-colors hover:text-bone">
          <ArrowLeft className="size-3.5" /> ALL JOBS
        </Link>

        {error ? (
          <p className="mt-8 font-mono text-sm text-destructive">Failed to load job: {error}</p>
        ) : !job ? (
          <p className="mt-8 font-mono text-sm text-mist">
            {reconnecting ? "Backend waking up (free tier sleeps) — still polling…" : "Loading job…"}
          </p>
        ) : (
          <div className="mt-4">
            <div className="flex flex-wrap items-center gap-3">
              <JobStatusBadge status={job.status} />
              <span className="font-mono text-[11px] text-ash">{job.id}</span>
            </div>
            <h1 className="mt-2 max-w-[30ch] text-3xl font-semibold tracking-tight sm:text-4xl">{job.pairLabel}</h1>
            <p className="mt-2 font-mono text-xs text-mist">
              {MATCHER_LABELS[job.matcherType]} · {job.transformModel} · GSD {job.meta.sourceGsdM} → {job.meta.referenceGsdM} m/px · Δsun {job.meta.sunDeltaDeg}°
            </p>
            <div className="mt-4 grid gap-3 sm:grid-cols-2">
              <div className="rounded-xl bg-void/60 p-4 ring-1 ring-line">
                <div className="font-mono text-[10px] tracking-[0.16em] text-ash">SOURCE · YOUR UPLOAD</div>
                <div className="mt-1 font-mono text-xs text-bone">{job.meta.sourceSensor} frame</div>
              </div>
              <div className="rounded-xl bg-signal/5 p-4 ring-1 ring-signal/25">
                <div className="font-mono text-[10px] tracking-[0.16em] text-signal">REFERENCE · AUTO-MATCHED FROM ARCHIVE</div>
                <div className="mt-1 font-mono text-xs text-bone">{job.meta.referenceSensor} · {job.meta.referenceFrameId ?? "matching archive…"}</div>
              </div>
            </div>

            {job.status === "FAILED" ? (
              <div className="mt-6 rounded-xl bg-destructive/10 p-5 font-mono text-xs leading-relaxed text-destructive ring-1 ring-destructive/30">
                FAILED AT {job.currentStage.toUpperCase()} — {job.errorMessage}
                {job.validation ? (
                  <div className="mt-2 text-destructive/80">
                    ARCHIVE EVIDENCE: {job.validation.inliers ?? 0} inliers · {((job.validation.inlier_ratio ?? 0) * 100).toFixed(1)}% ratio · {((job.validation.coverage ?? 0) * 100).toFixed(1)}% coverage
                  </div>
                ) : null}
              </div>
            ) : null}

            {(job.status === "PENDING" || job.status === "RUNNING") && (
              <div className="mt-6"><Progress stage={job.currentStage} /></div>
            )}

            {result && (
              <div className="mt-8 space-y-8">
                <section>
                  <h2 className="mb-3 font-mono text-[11px] tracking-[0.2em] text-signal">MATCH EVIDENCE</h2>
                  <MatchViewer
                    matches={result.matches}
                    sourceImg={result.images?.source ?? sourceImg}
                    referenceImg={result.images?.reference ?? referenceImg}
                    sourceLabel={result.images ? "SOURCE · your upload" : undefined}
                    referenceLabel={result.images ? "REFERENCE · archive match" : undefined}
                  />
                </section>
                {result.images ? (
                  <section>
                    <h2 className="mb-3 font-mono text-[11px] tracking-[0.2em] text-signal">REGISTERED PRODUCT · WARPED</h2>
                    <div className="overflow-hidden rounded-xl ring-1 ring-line">
                      {/* eslint-disable-next-line @next/next/no-img-element */}
                      <img src={result.images.warped} alt="Warped source registered to reference" className="aspect-[16/9] w-full object-cover" />
                    </div>
                    <p className="mt-2 font-mono text-[10px] tracking-[0.08em] text-ash">
                      SOURCE WARPED INTO THE REFERENCE FRAME · HOMOGRAPHY FROM {result.report.inlierCount} INLIERS
                    </p>
                  </section>
                ) : null}
                <section>
                  <h2 className="mb-3 font-mono text-[11px] tracking-[0.2em] text-signal">EVALUATION</h2>
                  <div className="grid gap-3 lg:grid-cols-[1fr_280px]">
                    <MetricCards report={result.report} />
                    <CoverageGrid matches={result.matches} />
                  </div>
                </section>
                <section>
                  <h2 className="mb-3 font-mono text-[11px] tracking-[0.2em] text-signal">TRANSFORM · {result.transform.modelType.toUpperCase()}</h2>
                  <pre className="overflow-x-auto rounded-xl bg-void/60 p-4 font-mono text-[11px] leading-relaxed text-mist ring-1 ring-line">
                    {JSON.stringify(result.transform.parameters, null, 2)}
                  </pre>
                </section>
                {result.report.sweep && result.report.sweep.length > 0 ? (
                  <section>
                    <h2 className="mb-3 font-mono text-[11px] tracking-[0.2em] text-signal">ARCHIVE SWEEP · {result.report.sweep.length} FRAME{result.report.sweep.length === 1 ? "" : "S"}</h2>
                    <div className="overflow-hidden rounded-xl ring-1 ring-line">
                      <div className="grid grid-cols-[1fr_auto_auto_auto] gap-3 border-b border-line/60 bg-surface/60 px-4 py-2.5 font-mono text-[10px] tracking-[0.14em] text-ash sm:grid-cols-[1fr_140px_80px_80px_80px]">
                        <span>FRAME</span>
                        <span className="hidden sm:block">LAT / LON</span>
                        <span className="text-right">SCORE %</span>
                        <span className="text-right">INLIERS</span>
                        <span className="hidden text-right sm:block">RMSE</span>
                      </div>
                      {result.report.sweep.map((s, i) => (
                        <div key={s.file} className={`grid grid-cols-[1fr_auto_auto_auto] items-center gap-3 border-b border-line/40 px-4 py-2.5 font-mono text-[11px] last:border-0 sm:grid-cols-[1fr_140px_80px_80px_80px] ${i === 0 ? "bg-signal/5" : ""}`}>
                          <span className="truncate text-bone">{i === 0 ? "★ " : ""}{s.file}</span>
                          <span className="hidden text-mist sm:block">{s.lat ?? "?"} / {s.lon ?? "?"}</span>
                          <span className="text-right tabular text-bone">{s.score.toFixed(1)}</span>
                          <span className="text-right tabular text-mist">{s.inliers}</span>
                          <span className="hidden text-right tabular text-mist sm:block">{s.rmse.toFixed(2)}</span>
                        </div>
                      ))}
                    </div>
                  </section>
                ) : null}
                <section>
                  <h2 className="mb-3 font-mono text-[11px] tracking-[0.2em] text-signal">DOWNLOADS</h2>
                  <Downloads result={result} />
                </section>
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
}
