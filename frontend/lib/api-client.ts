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
    if (res.status === 413) {
      throw new Error(
        "Upload too large (platform limit ~4.5MB). Export a smaller crop or lower-resolution copy and retry.",
      );
    }
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

/** Downscale oversized uploads in-browser so they clear the ~4.5MB
 *  serverless body limit. Budget is PIXEL-based (not long-side), so extreme
 *  strips keep a usable short side: a fixed 2000px long side would crush a
 *  18290x400 strip into a 44px sliver SIFT can't read. Matching downsamples
 *  to ≤1200px server-side anyway, so nothing geometric is lost. */
async function shrinkImage(file: File, pixelBudget = 3_000_000): Promise<File> {
  if (file.size <= 3.5 * 1024 * 1024) return file;
  try {
    const bmp = await createImageBitmap(file);
    let scale = Math.min(1, Math.sqrt(pixelBudget / (bmp.width * bmp.height)));
    if (scale >= 1) return file;
    const canvas = document.createElement("canvas");
    const draw = (s: number) => {
      canvas.width = Math.max(1, Math.round(bmp.width * s));
      canvas.height = Math.max(1, Math.round(bmp.height * s));
      const ctx = canvas.getContext("2d");
      if (!ctx) return false;
      ctx.drawImage(bmp, 0, 0, canvas.width, canvas.height);
      return true;
    };
    if (!draw(scale)) return file;
    let quality = 0.85;
    let blob = await new Promise<Blob | null>((res) =>
      canvas.toBlob(res, "image/jpeg", quality),
    );
    while (blob && blob.size > 3.5 * 1024 * 1024 && quality > 0.45) {
      quality -= 0.15;
      blob = await new Promise<Blob | null>((res) =>
        canvas.toBlob(res, "image/jpeg", quality),
      );
    }
    if (blob && blob.size > 3.5 * 1024 * 1024 && scale > 0.25) {
      // Still too big: halve dimensions once more (keeps strips readable).
      if (!draw(scale / 2)) return file;
      blob = await new Promise<Blob | null>((res) =>
        canvas.toBlob(res, "image/jpeg", 0.8),
      );
    }
    if (typeof bmp.close === "function") bmp.close();
    if (!blob) return file;
    return new File([blob], file.name.replace(/\.[^.]+$/, "") + ".jpg", {
      type: "image/jpeg",
    });
  } catch {
    return file; // e.g. browser can't decode TIF — let the server try
  }
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
    json<{ references: CloudReference[]; mode?: string; warning?: string }>(r),
  );
}

export function uploadReference(input: { file: File; lat?: string; lon?: string }) {
  return shrinkImage(input.file).then((file) => {
    const form = new FormData();
    form.append("file", file);
    if (input.lat) form.append("lat", input.lat);
    if (input.lon) form.append("lon", input.lon);
    return fetch(`${BASE}/references`, { method: "POST", body: form }).then((r) =>
      json<{ reference: CloudReference }>(r),
    );
  });
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
    const files = [input.sourceFile, input.referenceFile].filter(Boolean) as File[];
    return Promise.all(files.map((f) => shrinkImage(f))).then(([source, reference]) => {
      const form = new FormData();
      form.append("source", source);
      if (reference) form.append("reference", reference);
      form.append("pairLabel", input.pairLabel);
      form.append("matcherType", input.matcherType);
      form.append("transformModel", input.transformModel);
      if (input.sourceSensor) form.append("sourceSensor", input.sourceSensor);
      if (input.referenceSensor) form.append("referenceSensor", input.referenceSensor);
      return fetch(`${BASE}/jobs`, { method: "POST", body: form }).then((r) =>
        json<{ jobId: string; job: Job }>(r),
      );
    });
  }
  return fetch(`${BASE}/jobs`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(input),
  }).then((r) => json<{ jobId: string; job: Job }>(r));
}
