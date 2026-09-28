// Dummy backend — types mirror backend/app/schemas/ (Pydantic) so swapping
// to the real FastAPI later only means repointing lib/api-client.ts.

export type JobStatus = "PENDING" | "RUNNING" | "SUCCEEDED" | "FAILED";
export type Reliability = "high" | "low" | "failed";
export type MatcherType = "superpoint-superglue" | "sift" | "akaze" | "rift2";
export type TransformModel = "homography" | "affine" | "similarity";

export type MatchPoint = {
  id: string;
  srcX: number; // 0..1 fractions of the source tile
  srcY: number;
  refX: number; // 0..1 fractions of the reference tile
  refY: number;
  confidence: number; // 0..1
  isInlier: boolean;
};

export type Transform = {
  modelType: TransformModel;
  parameters: number[]; // 3x3 row-major
};

export type SweepEntry = {
  file: string;
  lat?: string;
  lon?: string;
  score: number;
  inliers: number;
  rmse: number;
  runtimeS?: number;
  error?: string;
};

export type EvaluationReport = {
  rmseX: number;
  rmseY: number;
  inlierCount: number;
  inlierRatio: number;
  coverageScore: number; // 0..1 grid occupancy
  processingTimeS: number;
  reliability: Reliability;
  reliabilityReason?: string;
  sweep?: SweepEntry[];
};

export type ImagePairMeta = {
  sourceSensor: "OHRC" | "TMC-2" | "IIRS";
  referenceSensor: "LRO NAC" | "SELENE TC";
  referenceFrameId?: string; // archive frame auto-matched — user never uploads this
  sourceGsdM: number;
  referenceGsdM: number;
  sourceSunElevationDeg: number;
  referenceSunElevationDeg: number;
  sunDeltaDeg: number;
};

export type Job = {
  id: string;
  pairLabel: string;
  meta: ImagePairMeta;
  status: JobStatus;
  currentStage: string;
  matcherType: MatcherType;
  transformModel: TransformModel;
  errorMessage?: string;
  validation?: {
    inliers?: number;
    inlier_ratio?: number;
    coverage?: number;
  };
  createdAt: string;
  startedAt?: string;
  completedAt?: string;
  pollsNeeded?: number; // simulated polls before SUCCEEDED (RUNNING jobs)
};

export type JobResult = {
  job: Job;
  matches: MatchPoint[];
  transform: Transform;
  report: EvaluationReport;
  images?: {
    source: string;
    reference: string;
    warped: string;
  };
};

export const STAGES = [
  "ingestion",
  "preprocessing",
  "overlap_estimation",
  "feature_extraction",
  "matching",
  "outlier_rejection",
  "uniform_selection",
  "subpixel_refinement",
  "transform_fit",
  "warping",
  "evaluation",
] as const;

export const MATCHER_LABELS: Record<MatcherType, string> = {
  "superpoint-superglue": "SuperPoint + SuperGlue",
  sift: "SIFT + RANSAC",
  akaze: "AKAZE + RANSAC",
  rift2: "RIFT2 + RANSAC",
};

// Deterministic RNG so fixtures are stable across reloads.
function mulberry32(seed: number) {
  let a = seed >>> 0;
  return () => {
    a |= 0;
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

const HOMOGRAPHY = [1.012, -0.004, 18.4, 0.006, 1.008, -11.2, 0.00001, -0.00002, 1];

function makeMatches(seed: number, count: number, outlierRatio: number): MatchPoint[] {
  const rng = mulberry32(seed);
  const pts: MatchPoint[] = [];
  // Well-spread grid base + jitter, so coverage looks uniform.
  const cols = Math.ceil(Math.sqrt(count));
  for (let i = 0; i < count; i++) {
    const gx = (i % cols) / cols;
    const gy = Math.floor(i / cols) / Math.ceil(count / cols);
    const outlier = rng() < outlierRatio;
    const jx = (rng() - 0.5) * 0.09;
    const jy = (rng() - 0.5) * 0.09;
    const sx = Math.min(0.97, Math.max(0.03, gx + 0.04 + (rng() - 0.5) * 0.05));
    const sy = Math.min(0.97, Math.max(0.03, gy + 0.04 + (rng() - 0.5) * 0.05));
    pts.push({
      id: `m-${seed}-${i}`,
      srcX: sx,
      srcY: sy,
      refX: outlier
        ? Math.min(0.97, Math.max(0.03, sx + jx * 4))
        : Math.min(0.97, Math.max(0.03, sx * 0.985 + 0.012 + jx * 0.15)),
      refY: outlier
        ? Math.min(0.97, Math.max(0.03, sy + jy * 4))
        : Math.min(0.97, Math.max(0.03, sy * 0.985 + 0.008 + jy * 0.15)),
      confidence: outlier ? 0.3 + rng() * 0.25 : 0.72 + rng() * 0.27,
      isInlier: !outlier,
    });
  }
  return pts;
}

function reportFor(matches: MatchPoint[], timeS: number, reliability: Reliability, reason?: string): EvaluationReport {
  const inliers = matches.filter((m) => m.isInlier);
  const err = (m: MatchPoint) => Math.hypot(m.refX - m.srcX, m.refY - m.srcY) * 400;
  const sq = inliers.map((m) => err(m) ** 2);
  const rmse = Math.sqrt(sq.reduce((a, b) => a + b, 0) / Math.max(1, sq.length));
  const cells = new Set(inliers.map((m) => `${Math.floor(m.refX * 6)}:${Math.floor(m.refY * 6)}`));
  return {
    rmseX: +(rmse * 0.72).toFixed(2),
    rmseY: +(rmse * 0.66).toFixed(2),
    inlierCount: inliers.length,
    inlierRatio: +(inliers.length / matches.length).toFixed(2),
    coverageScore: +(cells.size / 36).toFixed(2),
    processingTimeS: timeS,
    reliability,
    reliabilityReason: reason,
  };
}

export type JobFixture = {
  job: Job;
  matches: MatchPoint[];
  transform: Transform;
  report: EvaluationReport;
};

const T = (h: number) => `2026-09-${String(10 - h).padStart(2, "0")}T1${h % 10}:24:00Z`;

export const JOB_FIXTURES: JobFixture[] = [
  {
    job: {
      id: "job-7f3a-ohrc-nac",
      pairLabel: "OHRC ↔ LRO NAC · equatorial highlands",
      meta: { sourceSensor: "OHRC", referenceSensor: "LRO NAC", sourceGsdM: 0.25, referenceGsdM: 0.55, sourceSunElevationDeg: 34, referenceSunElevationDeg: 41, sunDeltaDeg: 9 },
      status: "SUCCEEDED",
      currentStage: "evaluation",
      matcherType: "superpoint-superglue",
      transformModel: "homography",
      createdAt: T(1), startedAt: T(1), completedAt: T(1),
    },
    matches: makeMatches(11, 32, 0.16),
    transform: { modelType: "homography", parameters: HOMOGRAPHY },
    report: { rmseX: 0, rmseY: 0, inlierCount: 0, inlierRatio: 0, coverageScore: 0, processingTimeS: 4.2, reliability: "high" },
  },
  {
    job: {
      id: "job-9c1e-tmc2-nac",
      pairLabel: "TMC-2 ↔ LRO NAC · mare edge",
      meta: { sourceSensor: "TMC-2", referenceSensor: "LRO NAC", sourceGsdM: 5, referenceGsdM: 1.1, sourceSunElevationDeg: 52, referenceSunElevationDeg: 38, sunDeltaDeg: 21 },
      status: "SUCCEEDED",
      currentStage: "evaluation",
      matcherType: "sift",
      transformModel: "homography",
      createdAt: T(2), startedAt: T(2), completedAt: T(2),
    },
    matches: makeMatches(29, 26, 0.3),
    transform: { modelType: "homography", parameters: HOMOGRAPHY },
    report: { rmseX: 0, rmseY: 0, inlierCount: 0, inlierRatio: 0, coverageScore: 0, processingTimeS: 612, reliability: "high" },
  },
  {
    job: {
      id: "job-44b2-ohrc-polar",
      pairLabel: "OHRC ↔ LRO NAC · south polar, low sun",
      meta: { sourceSensor: "OHRC", referenceSensor: "LRO NAC", sourceGsdM: 0.28, referenceGsdM: 0.9, sourceSunElevationDeg: 6, referenceSunElevationDeg: 31, sunDeltaDeg: 46 },
      status: "RUNNING",
      currentStage: "matching",
      matcherType: "superpoint-superglue",
      transformModel: "homography",
      createdAt: T(0), startedAt: T(0),
      pollsNeeded: 6,
    },
    matches: makeMatches(47, 24, 0.34),
    transform: { modelType: "homography", parameters: HOMOGRAPHY },
    report: { rmseX: 0, rmseY: 0, inlierCount: 0, inlierRatio: 0, coverageScore: 0, processingTimeS: 0, reliability: "high" },
  },
  {
    job: {
      id: "job-51d8-iirs-wac",
      pairLabel: "IIRS ↔ SELENE TC · farside highlands",
      meta: { sourceSensor: "IIRS", referenceSensor: "SELENE TC", sourceGsdM: 80, referenceGsdM: 10, sourceSunElevationDeg: 44, referenceSunElevationDeg: 47, sunDeltaDeg: 5 },
      status: "SUCCEEDED",
      currentStage: "evaluation",
      matcherType: "rift2",
      transformModel: "affine",
      createdAt: T(3), startedAt: T(3), completedAt: T(3),
    },
    matches: makeMatches(63, 18, 0.22),
    transform: { modelType: "affine", parameters: [0.998, 0.003, 4.1, -0.002, 1.001, -2.4, 0, 0, 1] },
    report: { rmseX: 0, rmseY: 0, inlierCount: 0, inlierRatio: 0, coverageScore: 0, processingTimeS: 48.6, reliability: "low", reliabilityReason: "Coarse 8:1 scale gap — affine fit held, but coverage is thin in smooth-mare cells." },
  },
  {
    job: {
      id: "job-88f0-ohrc-night",
      pairLabel: "OHRC ↔ LRO NAC · terminator pass",
      meta: { sourceSensor: "OHRC", referenceSensor: "LRO NAC", sourceGsdM: 0.25, referenceGsdM: 0.6, sourceSunElevationDeg: 3, referenceSunElevationDeg: 55, sunDeltaDeg: 71 },
      status: "FAILED",
      currentStage: "outlier_rejection",
      matcherType: "akaze",
      transformModel: "homography",
      errorMessage: "Too few inliers survived RANSAC (3/41) — no reliable registration. Try the learned matcher or a closer sun-angle pair.",
      createdAt: T(4), startedAt: T(4),
    },
    matches: makeMatches(91, 20, 0.8),
    transform: { modelType: "homography", parameters: HOMOGRAPHY },
    report: { rmseX: 0, rmseY: 0, inlierCount: 0, inlierRatio: 0, coverageScore: 0, processingTimeS: 402, reliability: "failed", reliabilityReason: "Registration unreliable — reported as failure, not a clean result." },
  },
  {
    job: {
      id: "job-b2c7-ohrc-selene",
      pairLabel: "OHRC ↔ SELENE TC · crater field",
      meta: { sourceSensor: "OHRC", referenceSensor: "SELENE TC", sourceGsdM: 0.3, referenceGsdM: 10, sourceSunElevationDeg: 28, referenceSunElevationDeg: 33, sunDeltaDeg: 7 },
      status: "PENDING",
      currentStage: "queued",
      matcherType: "superpoint-superglue",
      transformModel: "homography",
      createdAt: T(0),
      pollsNeeded: 9,
    },
    matches: makeMatches(107, 30, 0.2),
    transform: { modelType: "homography", parameters: HOMOGRAPHY },
    report: { rmseX: 0, rmseY: 0, inlierCount: 0, inlierRatio: 0, coverageScore: 0, processingTimeS: 0, reliability: "high" },
  },
];

// Archive frames auto-matched per job (reference is never user-uploaded).
const ARCHIVE_FRAMES: Record<string, string> = {
  "job-7f3a-ohrc-nac": "M1414653521LE",
  "job-9c1e-tmc2-nac": "M1183457892RE",
  "job-44b2-ohrc-polar": "M1357924680LE",
  "job-51d8-iirs-wac": "TCO_MAP_02_N42E015",
  "job-88f0-ohrc-night": "M1092837465LE",
  "job-b2c7-ohrc-selene": "TCO_MAP_02_S18E122",
};

for (const f of JOB_FIXTURES) {
  f.job.meta.referenceFrameId =
    ARCHIVE_FRAMES[f.job.id] ?? "M0000000000LE";
}

// Fill derived report numbers from the generated matches.
for (const f of JOB_FIXTURES) {
  if (f.job.status === "FAILED") {
    f.report = { ...reportFor(f.matches, f.report.processingTimeS, "failed", f.report.reliabilityReason), inlierCount: 3 };
  } else {
    const base = reportFor(f.matches, f.report.processingTimeS || 5, f.report.reliability, f.report.reliabilityReason);
    f.report = { ...base, processingTimeS: f.report.processingTimeS };
  }
}
