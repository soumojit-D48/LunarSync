"""LunaMatch demo backend — FastAPI over the notebook pipeline. No database:
jobs live in an in-memory dict, image artifacts on local disk.
"""

import io
import threading
import time
import uuid
from pathlib import Path

import cv2
import numpy as np
from fastapi import BackgroundTasks, FastAPI, File, Form, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from pipeline.lunar_validator import LunarValidationError
from pipeline.run_pipeline import VALID_EXTENSIONS, run_registration, sweep_archive

BASE = Path(__file__).resolve().parent
ROOT = BASE.parent
OUT_DIR = BASE / "static" / "outputs"
# Primary archive: root data/reference (lat/lon filenames). Fallback: backend copy.
REF_DIRS = [ROOT / "data" / "reference", BASE / "data" / "reference"]
OUT_DIR.mkdir(parents=True, exist_ok=True)

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


def _archive_files() -> list:
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


def _run_job(job_id: str, src_bytes: bytes, ref_bytes: bytes | None):
    job = jobs[job_id]
    try:
        job["status"] = "RUNNING"
        job["startedAt"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

        def on_stage(stage: str):
            job["currentStage"] = stage

        if ref_bytes is not None:
            # Explicit pair: user uploaded both frames.
            out = run_registration(src_bytes, ref_bytes, on_stage=on_stage)
        else:
            # Archive mode: sweep root data/reference, keep the winner.
            files = _archive_files()
            if not files:
                raise RuntimeError("Archive empty: add frames to data/reference/.")
            out = sweep_archive(src_bytes, files, on_stage=on_stage)
            winner = out.get("winner") or {}
            job["meta"] = {
                **job["meta"],
                "referenceFrameId": winner.get("file", "archive"),
                "referenceSunElevationDeg": 0,
                "sunDeltaDeg": 0,
            }
            job["pairLabel"] = f"{job['meta']['sourceSensor']} → {winner.get('file', 'archive')} · live sweep"

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


def _public_job(job: dict) -> dict:
    return {k: v for k, v in job.items() if k not in ("result", "src_bytes", "ref_bytes")}


@app.get("/health")
def health():
    return {"ok": True, "jobs": len(jobs)}


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
