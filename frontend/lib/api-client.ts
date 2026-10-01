// Typed fetch wrapper around /api/proxy (dummy backend today, FastAPI tomorrow).
import type {
  EvaluationReport,
  Job,
  JobResult,
  MatcherType,
  MatchPoint,
  TransformModel,
} from "@/lib/mock-data";

export type { EvaluationReport, Job, JobResult, MatcherType, MatchPoint, TransformModel };

const BASE = "/api/proxy";

async function json<T>(res: Response): Promise<T> {
  if (!res.ok) {
    let detail = "";
    try {
      const body = await res.json();
      detail = (body as any)?.detail ?? (body as any)?.error ?? "";
    } catch {
      /* ignore */
    }
    throw new Error(detail ? `API ${res.status}: ${detail}` : `API ${res.status}`);
  }
  return res.json() as Promise<T>;
}

export function listJobs() {
  return fetch(`${BASE}/jobs`).then((r) => json<{ jobs: Job[] }>(r));
}

export function getJob(id: string) {
  return fetch(`${BASE}/jobs/${id}`).then((r) => json<{ job: Job }>(r));
}

export function getResult(id: string) {
  return fetch(`${BASE}/jobs/${id}/result`).then((r) => json<JobResult>(r));
}

export function getMatches(id: string) {
  return fetch(`${BASE}/jobs/${id}/matches`).then((r) => json<{ matches: MatchPoint[] }>(r));
}

export function getReport(id: string) {
  return fetch(`${BASE}/jobs/${id}/report`).then((r) => json<{ report: EvaluationReport }>(r));
}

export type CloudReference = {
  id: string;
  publicId?: string;
  secureUrl: string;
  filename: string;
  lat?: string | null;
  lon?: string | null;
  width?: number | null;
  height?: number | null;
  createdAt?: string;
};

export function getBackendMode() {
  return fetch(`${BASE}/reference-backend`).then((r) =>
    json<{ mode: string; cloud?: any; hf?: any }>(r),
  );
}

export function listReferences() {
  return fetch(`${BASE}/references`).then((r) =>
    json<{ references: CloudReference[]; mode?: string }>(r),
  );
}

export function uploadReference(input: { file: File; lat?: string; lon?: string }) {
  const form = new FormData();
  form.append("file", input.file);
  if (input.lat) form.append("lat", input.lat);
  if (input.lon) form.append("lon", input.lon);
  return fetch(`${BASE}/references`, { method: "POST", body: form }).then((r) =>
    json<{ reference: CloudReference }>(r),
  );
}

export function deleteReference(id: string) {
  return fetch(`${BASE}/references/${id}`, { method: "DELETE" }).then((r) =>
    json<{ deleted: string }>(r),
  );
}

export function createJob(input: {
  pairLabel: string;
  matcherType: MatcherType;
  transformModel: TransformModel;
  sourceFile?: File;
  referenceFile?: File;
  sourceSensor?: string;
  referenceSensor?: string;
}) {
  if (input.sourceFile) {
    const form = new FormData();
    form.append("source", input.sourceFile);
    if (input.referenceFile) form.append("reference", input.referenceFile);
    form.append("pairLabel", input.pairLabel);
    form.append("matcherType", input.matcherType);
    form.append("transformModel", input.transformModel);
    if (input.sourceSensor) form.append("sourceSensor", input.sourceSensor);
    if (input.referenceSensor) form.append("referenceSensor", input.referenceSensor);
    return fetch(`${BASE}/jobs`, { method: "POST", body: form }).then((r) =>
      json<{ jobId: string; job: Job }>(r),
    );
  }
  return fetch(`${BASE}/jobs`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(input),
  }).then((r) => json<{ jobId: string; job: Job }>(r));
}
