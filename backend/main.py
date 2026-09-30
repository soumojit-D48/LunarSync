"""LunaMatch dynamic backend — HF-backed archive sweep + notebook pipeline.

No database: jobs live in an in-memory dict, image artifacts on local disk.
Reference frames are dynamic: HF dataset (lunar-team-2026/data-image via
HF_TOKEN in .env) first, local data/reference/ as offline fallback.
"""

import json
import os
import threading
import time
import uuid
from pathlib import Path

import cv2
import numpy as np
from fastapi import BackgroundTasks, FastAPI, File, Form, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

try:
    from dotenv import load_dotenv  # type: ignore

    load_dotenv()
except Exception:
    pass

from pipeline.hf_matcher import parse_metadata as hf_parse_metadata
from pipeline.hf_matcher import sweep as hf_sweep
from pipeline.hf_matcher import to_frontend as hf_to_frontend
from pipeline.hf_store import ensure_dataset, list_reference_images, status as hf_status
from pipeline.lunar_validator import LunarValidationError
from pipeline.run_pipeline import VALID_EXTENSIONS, run_registration, sweep_archive

BASE = Path(__file__).resolve().parent
ROOT = BASE.parent
OUT_DIR = BASE / "static" / "outputs"
JOB_STORE = BASE / "static" / "jobs.json"
REF_DIRS = [ROOT / "data" / "reference", BASE / "data" / "reference"]
OUT_DIR.mkdir(parents=True, exist_ok=True)

HF_LIMIT = int(os.getenv("HF_SWEEP_LIMIT", "0") or 0) or None
HF_EARLY_STOP = float(os.getenv("HF_EARLY_STOP", "60.0") or 60.0)
_HF_SNAPSHOT = None  # lazy-cached snapshot dir

STAGES = ["ingestion", "preprocessing", "overlap_estimation", "feature_extraction",
          "matching", "outlier_rejection", "uniform_selection", "subpixel_refinement",
          "transform_fit", "warping", "evaluation"]

app = FastAPI(title="LunaMatch demo backend")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000", "*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

jobs: dict = {}
_lock = threading.Lock()


def _load_jobs():
    if JOB_STORE.exists():
        try:
            data = json.loads(JOB_STORE.read_text(encoding="utf-8"))
            for jid, job in data.items():
                job.pop("result", None)
                jobs[jid] = job
        except Exception:
            pass


def _save_jobs():
    try:
        serializable = {}
        for jid, job in jobs.items():
            j = {k: v for k, v in job.items() if k not in ("result", "src_bytes", "ref_bytes")}
            serializable[jid] = j
        JOB_STORE.write_text(json.dumps(serializable, indent=2, default=str), encoding="utf-8")
    except Exception:
        pass


_load_jobs()


def _archive_files() -> list:
    """Dynamic listing: HF snapshot first, local dirs as fallback."""
    global _HF_SNAPSHOT
    if _HF_SNAPSHOT is None:
        try:
            _HF_SNAPSHOT = ensure_dataset()
        except Exception:
            _HF_SNAPSHOT = None
    files = list_reference_images(_HF_SNAPSHOT, REF_DIRS)
    if files:
        return files
    for d in REF_DIRS:
        if d.exists():
            files = sorted(p for p in d.iterdir()
                           if p.suffix.lower() in VALID_EXTENSIONS and p.is_file())
            if files:
                return files
    return []


def _default_reference() -> bytes:
    files = _archive_files()
    if not files:
        raise RuntimeError("No reference image: upload one or add files to data/reference/.")
    return files[0].read_bytes()


def _clean_sweep(rows: list) -> list:
    """Strip any non-JSON internals; coerce numpy scalars to plain types."""
    clean = []
    for r in rows or []:
        try:
            clean.append({
                "file": str(r.get("file", "?")),
                "lat": str(r.get("lat", "?")),
                "lon": str(r.get("lon", "?")),
                "score": float(r.get("score", 0.0)),
                "inliers": int(r.get("inliers", 0)),
                "rmse": float(r.get("rmse", 0.0)),
                "overlap": float(r.get("overlap", 0.0)),
                "iou": float(r.get("iou", 0.0)),
                "model": str(r.get("model", "?")),
                "runtimeS": float(r.get("runtimeS", 0.0)),
            })
        except Exception:
            continue
    return clean


def _decode_gray(data: bytes):
    arr = np.frombuffer(data, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError("Could not decode image (use PNG/JPG/TIF).")
    if img.shape[0] > img.shape[1]:
        img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    return img


def _run_job(job_id: str, src_bytes: bytes, ref_bytes: bytes | None):
    job = jobs[job_id]
    try:
        job["status"] = "RUNNING"
        job["startedAt"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

        def on_stage(stage: str):
            job["currentStage"] = stage
            with _lock:
                _save_jobs()

        if ref_bytes is not None:
            # Explicit pair: user uploaded both frames.
            out = run_registration(src_bytes, ref_bytes, on_stage=on_stage)
        else:
            # Dynamic archive mode: HF dataset first, local fallback.
            from pipeline.lunar_validator import validate_lunar_image

            t0 = time.time()
            on_stage("ingestion")
            query_img = _decode_gray(src_bytes)
            on_stage("preprocessing")
            validate_lunar_image(query_img)
            files = _archive_files()
            if not files:
                raise RuntimeError(
                    "Archive empty: check HF_TOKEN/HF_DATASET or add frames to data/reference/."
                )
            sweep = hf_sweep(query_img, files, on_stage=on_stage,
                             limit=HF_LIMIT, early_stop=HF_EARLY_STOP)
            wres = sweep["winner_result"]
            out = hf_to_frontend(query_img, sweep["winner_image"], wres, sweep["elapsed"])
            out["report"]["sweep"] = _clean_sweep(sweep["sweep"])
            out["report"]["processingTimeS"] = round(time.time() - t0, 2)
            out["winner"] = {"file": sweep["winner_row"]["file"],
                             "score": float(sweep["winner_row"]["score"])}
            meta = hf_parse_metadata(sweep["winner_row"]["file"])
            job["meta"] = {
                **job["meta"],
                "referenceFrameId": sweep["winner_row"]["file"],
                "referenceSunElevationDeg": 0,
                "sunDeltaDeg": 0,
            }
            job["pairLabel"] = (
                f"{job['meta']['sourceSensor']} → {sweep['winner_row']['file']} · HF sweep"
            )

        job_dir = OUT_DIR / job_id
        job_dir.mkdir(parents=True, exist_ok=True)
        for name, blob in out["images"].items():
            (job_dir / f"{name}.jpg").write_bytes(blob)
        job["result"] = {
            "job": _public_job(job),
            "matches": out["matches"],
            "transform": out["transform"],
            "report": out["report"],
            "images": {
                "source": f"/outputs/{job_id}/source.jpg",
                "reference": f"/outputs/{job_id}/reference.jpg",
                "warped": f"/outputs/{job_id}/warped.jpg",
            },
        }
        job["status"] = "SUCCEEDED"
        job["currentStage"] = "evaluation"
        if out.get("report", {}).get("reliability") == "failed":
            job["status"] = "FAILED"
            job["errorMessage"] = out["report"].get("reliabilityReason", "Registration failed")
    except LunarValidationError as exc:
        job["status"] = "FAILED"
        job["currentStage"] = "outlier_rejection"
        job["errorMessage"] = exc.reason
        if exc.details:
            job["validation"] = exc.details
    except Exception as exc:  # noqa: BLE001 — surfaced to the UI, never silent
        job["status"] = "FAILED"
        job["errorMessage"] = str(exc)[:500]
    finally:
        job["completedAt"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        with _lock:
            jobs[job_id] = job
            _save_jobs()


def _public_job(job: dict) -> dict:
    return {k: v for k, v in job.items() if k not in ("result", "src_bytes", "ref_bytes")}


@app.get("/health")
def health():
    try:
        hfs = hf_status()
    except Exception as exc:
        hfs = {"error": str(exc)[:200]}
    return {"ok": True, "jobs": len(jobs), "hf": hfs}


@app.get("/api/jobs")
def list_jobs():
    with _lock:
        items = [_public_job(j) for j in jobs.values()]
    return {"jobs": items}


@app.post("/api/jobs", status_code=201)
def create_job(
    background: BackgroundTasks,
    source: UploadFile = File(...),
    reference: UploadFile | None = File(default=None),
    pairLabel: str = Form(default="OHRC ↔ LRO NAC · live run"),
    matcherType: str = Form(default="superpoint-superglue"),
    transformModel: str = Form(default="homography"),
    sourceSensor: str = Form(default="OHRC"),
    referenceSensor: str = Form(default="LRO NAC"),
):
    src_bytes = source.file.read()
    # No reference uploaded → sweep the root data/reference archive.
    ref_bytes = reference.file.read() if reference is not None else None
    job_id = f"live-{uuid.uuid4().hex[:8]}"
    now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    job = {
        "id": job_id,
        "pairLabel": pairLabel,
        "meta": {
            "sourceSensor": sourceSensor,
            "referenceSensor": referenceSensor,
            "referenceFrameId": "live-archive",
            "sourceGsdM": 0.25,
            "referenceGsdM": 0.6,
            "sourceSunElevationDeg": 30,
            "referenceSunElevationDeg": 36,
            "sunDeltaDeg": 6,
        },
        "status": "PENDING",
        "currentStage": "queued",
        "matcherType": matcherType,
        "transformModel": transformModel,
        "createdAt": now,
    }
    with _lock:
        jobs[job_id] = job
        _save_jobs()
    background.add_task(_run_job, job_id, src_bytes, ref_bytes)
    return {"jobId": job_id, "job": _public_job(job)}


@app.get("/api/jobs/{job_id}")
def get_job(job_id: str):
    job = jobs.get(job_id)
    if job is None:
        from fastapi import HTTPException
        raise HTTPException(404, "job not found")
    return {"job": _public_job(job)}


@app.get("/api/jobs/{job_id}/result")
def get_result(job_id: str):
    job = jobs.get(job_id)
    if job is None or "result" not in job:
        from fastapi import HTTPException
        raise HTTPException(404, "result not ready")
    return job["result"]


@app.get("/api/jobs/{job_id}/matches")
def get_matches(job_id: str):
    return {"matches": get_result(job_id)["matches"]}


@app.get("/api/jobs/{job_id}/report")
def get_report(job_id: str):
    return {"report": get_result(job_id)["report"]}


app.mount("/outputs", StaticFiles(directory=OUT_DIR), name="outputs")
