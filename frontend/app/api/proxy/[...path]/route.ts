import { NextRequest, NextResponse } from "next/server";
import {
  JOB_FIXTURES,
  STAGES,
  type EvaluationReport,
  type Job,
  type JobFixture,
  type MatchPoint,
  type Transform,
} from "@/lib/mock-data";

// In-memory dummy store. Module state survives across requests in dev/prod
// single-instance; it resets on redeploy — fine for fixtures.
let store: Map<string, JobFixture> | null = null;
const polls = new Map<string, number>();

// Live backend (FastAPI). When reachable, real notebook-pipeline results flow
// through; otherwise everything falls back to the mock store above.
const BACKEND = process.env.BACKEND_URL ?? "http://localhost:8000";

async function fromBackend(path: string, init?: RequestInit) {
  const ctrl = new AbortController();
  const t = setTimeout(() => ctrl.abort(), 25000);
  try {
    const r = await fetch(`${BACKEND}/api/${path}`, { ...init, cache: "no-store", signal: ctrl.signal });
    if (r.status === 404) {
      const e = new Error("backend 404") as Error & { backend404?: boolean };
      e.backend404 = true;
      throw e;
    }
    if (!r.ok) throw new Error(`backend ${r.status}`);
    return r.json();
  } finally {
    clearTimeout(t);
  }
}

function withProxyImages(result: any) {
  if (result?.images) {
    const id = result.job?.id ?? "";
    const images: Record<string, string> = {};
    for (const [k, v] of Object.entries(result.images)) {
      const url = String(v);
      // Absolute URLs (e.g. Cloudinary results persisted in Neon) pass through.
      if (/^https?:\/\//.test(url)) {
        images[k] = url;
        continue;
      }
      const name = url.split("/").pop();
      images[k] = `/api/proxy/files/${id}/${name}`;
    }
    return { ...result, images };
  }
  return result;
}

function getStore(): Map<string, JobFixture> {
  if (!store) {
    store = new Map(
      JOB_FIXTURES.map((f) => [f.job.id, JSON.parse(JSON.stringify(f)) as JobFixture]),
    );
  }
  return store;
}

function advance(job: Job): Job {
  if (job.status === "SUCCEEDED" || job.status === "FAILED") return job;
  const n = (polls.get(job.id) ?? 0) + 1;
  polls.set(job.id, n);
  const needed = job.pollsNeeded ?? 6;
  if (job.status === "PENDING" && n >= 1) {
    job.status = "RUNNING";
    job.startedAt = new Date().toISOString();
  }
  if (job.status === "RUNNING") {
    const idx = Math.min(STAGES.length - 1, Math.floor(((n - 1) / needed) * STAGES.length));
    job.currentStage = STAGES[idx];
    if (n >= needed + 1) {
      job.status = "SUCCEEDED";
      job.currentStage = "evaluation";
      job.completedAt = new Date().toISOString();
    }
  }
  return job;
}

export async function GET(_req: NextRequest, ctx: { params: Promise<{ path: string[] }> }) {
  const segments = (await ctx.params).path ?? [];

  // Same-origin image bytes for live-backend job artifacts.
  if (segments.length === 3 && segments[0] === "files") {
    const r = await fetch(`${BACKEND}/outputs/${segments[1]}/${segments[2]}`, { cache: "no-store" });
    if (!r.ok) return NextResponse.json({ error: "file not found" }, { status: 404 });
    const buf = await r.arrayBuffer();
    return new Response(buf, { headers: { "Content-Type": "image/jpeg", "Cache-Control": "public, max-age=3600" } });
  }

  // Live-backend jobs (live-*) go straight to FastAPI; mock jobs (job-*)
  // are served locally. A real backend 404 falls through to the mock store;
  // anything else (waking backend, slow sweep, timeout) returns 502 so the
  // client keeps polling instead of showing a fake "job not found".
  if (segments.length >= 2 && segments[0] === "jobs" && segments[1].startsWith("live-")) {
    try {
      const data = await fromBackend(segments.join("/"));
      return NextResponse.json(data?.job && data?.matches ? withProxyImages(data) : data);
    } catch (e) {
      if (!(e as { backend404?: boolean })?.backend404) {
        return NextResponse.json(
          { error: `backend not answering (${e instanceof Error ? e.message : "error"}) — retrying` },
          { status: 502 },
        );
      }
      /* backend 404: fall through to mock (404 there if truly unknown) */
    }
  }

  const s = getStore();

  // Reference-registry + backend-mode passthrough (cloud seeding UI).
  if (segments.length === 1 && segments[0] === "references") {
    try {
      return NextResponse.json(await fromBackend("references"));
    } catch {
      return NextResponse.json({ references: [], mode: "unknown", warning: "backend unreachable" });
    }
  }
  if (segments.length === 1 && segments[0] === "reference-backend") {
    try {
      return NextResponse.json(await fromBackend("reference-backend"));
    } catch {
      return NextResponse.json({ mode: "unknown", warning: "backend unreachable" });
    }
  }

  if (segments.length === 1 && segments[0] === "jobs") {
    const mockJobs = [...s.values()].map((f) => f.job);
    try {
      const data = await fromBackend("jobs");
      const liveJobs = Array.isArray(data?.jobs) ? data.jobs : [];
      return NextResponse.json({ jobs: [...liveJobs, ...mockJobs] });
    } catch {
      return NextResponse.json({ jobs: mockJobs });
    }
  }
  if (segments.length === 2 && segments[0] === "jobs") {
    const f = s.get(segments[1]);
    if (!f) return NextResponse.json({ error: "job not found" }, { status: 404 });
    return NextResponse.json({ job: advance(f.job) });
  }
  if (segments.length === 3 && segments[0] === "jobs") {
    const f = s.get(segments[1]);
    if (!f) return NextResponse.json({ error: "job not found" }, { status: 404 });
    const kind = segments[2];
    if (kind === "result")
      return NextResponse.json({
        job: f.job,
        matches: f.matches,
        transform: f.transform,
        report: f.report,
      });
    if (kind === "matches") return NextResponse.json({ matches: f.matches });
    if (kind === "report") return NextResponse.json({ report: f.report });
  }
  return NextResponse.json({ error: "unknown endpoint" }, { status: 404 });
}

export async function POST(req: NextRequest, ctx: { params: Promise<{ path: string[] }> }) {
  const segments = (await ctx.params).path ?? [];
  const contentType = req.headers.get("content-type") ?? "";
  // Reference seeding (Cloudinary + Neon) — forward multipart to backend.
  if (segments.length === 1 && segments[0] === "references") {
    if (!contentType.includes("multipart/form-data")) {
      return NextResponse.json({ error: "expected multipart upload" }, { status: 400 });
    }
    try {
      const form = await req.formData();
      const r = await fetch(`${BACKEND}/api/references`, { method: "POST", body: form });
      const body = await r.json().catch(() => ({}));
      if (!r.ok) return NextResponse.json(body, { status: r.status });
      return NextResponse.json(body, { status: 201 });
    } catch (e) {
      return NextResponse.json(
        { error: `live backend unreachable (${e instanceof Error ? e.message : "error"})` },
        { status: 502 },
      );
    }
  }
  if (segments.length !== 1 || segments[0] !== "jobs") {
    return NextResponse.json({ error: "unknown endpoint" }, { status: 404 });
  }
  // Real upload with file bytes → forward to the live backend (notebook pipeline).
  if (contentType.includes("multipart/form-data")) {
    try {
      const form = await req.formData();
      const r = await fetch(`${BACKEND}/api/jobs`, { method: "POST", body: form });
      if (!r.ok) throw new Error(`backend ${r.status}`);
      return NextResponse.json(await r.json(), { status: 201 });
    } catch (e) {
      return NextResponse.json(
        { error: `live backend unreachable (${e instanceof Error ? e.message : "error"}). Start it with: uvicorn main:app --port 8000` },
        { status: 502 },
      );
    }
  }
  const body = (await req.json().catch(() => ({}))) as Partial<Job> & {
    matches?: MatchPoint[];
    transform?: Transform;
    report?: EvaluationReport;
  };
  const id = `job-${Date.now().toString(36)}`;
  const fixture: JobFixture = {
    job: {
      id,
      pairLabel: body.pairLabel ?? "OHRC ↔ LRO NAC · new upload",
      meta: body.meta ?? {
        sourceSensor: "OHRC",
        referenceSensor: "LRO NAC",
        referenceFrameId: "M1414653521LE",
        sourceGsdM: 0.25,
        referenceGsdM: 0.6,
        sourceSunElevationDeg: 30,
        referenceSunElevationDeg: 36,
        sunDeltaDeg: 6,
      },
      status: "PENDING",
      currentStage: "queued",
      matcherType: body.matcherType ?? "superpoint-superglue",
      transformModel: body.transformModel ?? "homography",
      createdAt: new Date().toISOString(),
      pollsNeeded: 7,
    },
    matches: body.matches ?? [],
    transform: body.transform ?? { modelType: "homography", parameters: [1, 0, 0, 0, 1, 0, 0, 0, 1] },
    report: body.report ?? {
      rmseX: 0.62, rmseY: 0.57, inlierCount: 27, inlierRatio: 0.84,
      coverageScore: 0.78, processingTimeS: 4.6, reliability: "high",
    },
  };
  // Seed plausible matches when the client doesn't supply any.
  if (fixture.matches.length === 0) {
    const seed = [...id].reduce((a, c) => a + c.charCodeAt(0), 0);
    const { JOB_FIXTURES: base } = await import("@/lib/mock-data");
    fixture.matches = JSON.parse(JSON.stringify(base[0].matches)) as MatchPoint[];
    fixture.matches.forEach((m, i) => {
      m.id = `m-${seed}-${i}`;
    });
  }
  const s = getStore();
  // Newest first.
  const next = new Map<string, JobFixture>([[id, fixture], ...s]);
  store = next;
  polls.set(id, 0);
  return NextResponse.json({ jobId: id, job: fixture.job }, { status: 201 });
}

export async function DELETE(_req: NextRequest, ctx: { params: Promise<{ path: string[] }> }) {
  const segments = (await ctx.params).path ?? [];
  if (segments.length === 2 && segments[0] === "references") {
    try {
      const r = await fetch(`${BACKEND}/api/references/${segments[1]}`, { method: "DELETE" });
      const body = await r.json().catch(() => ({}));
      if (!r.ok) return NextResponse.json(body, { status: r.status });
      return NextResponse.json(body);
    } catch (e) {
      return NextResponse.json(
        { error: `live backend unreachable (${e instanceof Error ? e.message : "error"})` },
        { status: 502 },
      );
    }
  }
  return NextResponse.json({ error: "unknown endpoint" }, { status: 404 });
}

export type { Job, JobFixture };
