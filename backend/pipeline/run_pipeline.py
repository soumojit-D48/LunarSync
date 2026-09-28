"""Single-pair registration + archive sweep — notebook logic, API-ready outputs.

No database: caller holds the returned dict in memory / on disk.
Match coordinates are 0..1 fractions of the display canvases (800x450),
matching the frontend MatchViewer contract.
"""

import io
import os
import re
import time

import cv2
import numpy as np

from .lunar_validator import validate_lunar_image, LunarValidationError
from .matching import match_pair
from .preprocessing import run_loop_27_preprocessing
from .sift_matching import match_pair_sift

DISPLAY_W, DISPLAY_H = 800, 450
MAX_MATCHES = 150
GRID_COLS, GRID_ROWS = 8, 6
VALID_EXTENSIONS = (".png", ".jpg", ".jpeg", ".tif", ".tiff")


def _decode_gray(data: bytes) -> np.ndarray:
    arr = np.frombuffer(data, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError("Could not decode image (use PNG/JPG/TIF).")
    if img.shape[0] > img.shape[1]:  # transposition guard from notebook
        img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    return img


def extract_coordinates_from_filename(filename: str) -> tuple:
    """Notebook's Option-C parser: 'minus' text becomes real negatives."""
    base = os.path.splitext(os.path.basename(filename))[0]
    normalized = re.sub(r"minus", "-", base, flags=re.IGNORECASE)
    lat_m = re.search(r"lat(?:itude)?(?:[-_]\s*|(?=\d|-))(-?\d+)", normalized, re.IGNORECASE)
    lon_m = re.search(r"lon(?:g|gitude)?(?:[-_]\s*|(?=\d|-))(-?\d+)", normalized, re.IGNORECASE)
    return (lat_m.group(1) if lat_m else "Unknown",
            lon_m.group(1) if lon_m else "Unknown")


def _to_display(gray: np.ndarray) -> np.ndarray:
    return cv2.resize(gray, (DISPLAY_W, DISPLAY_H), interpolation=cv2.INTER_AREA)


def _encode_jpg(gray: np.ndarray) -> bytes:
    ok, buf = cv2.imencode(".jpg", gray, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
    if not ok:
        raise RuntimeError("JPEG encode failed.")
    return bytes(buf)


def _coverage(matches: list) -> float:
    cells = set()
    for m in matches:
        if m["isInlier"]:
            cells.add((min(GRID_COLS - 1, int(m["srcX"] * GRID_COLS)),
                       min(GRID_ROWS - 1, int(m["srcY"] * GRID_ROWS))))
    return len(cells) / (GRID_COLS * GRID_ROWS)


def _require_usable_match(out: dict) -> None:
    """Reject a plausible-looking but weak geometric fit.

    Four points can always produce a homography with an artificially tiny
    residual. The notebook avoids returning those fits; apply the same gate
    here so unrelated uploads do not produce a confident-looking warp.
    """
    report = out["report"]
    if (
        report["inlierCount"] < 10
        or report["inlierRatio"] < 0.15
        or not report.get("footprintValid", False)
    ):
        raise LunarValidationError(
            "Uploaded image does not appear to be a lunar surface frame — no reliable correspondence found in the archive.",
            {
                "inliers": report["inlierCount"],
                "inlier_ratio": report["inlierRatio"],
                "coverage": report["coverageScore"],
            },
        )


def _validate_footprint(H: np.ndarray, shape: tuple) -> tuple:
    """Reject degenerate homographies that collapse the image footprint.

    Adapted from the Outlier Rejection notebook's validate_transformed_footprint.
    """
    h, w = shape
    corners = np.float32([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]])
    try:
        transformed = cv2.perspectiveTransform(corners.reshape(-1, 1, 2), H)
    except cv2.error:
        return False, 0.0
    if not np.isfinite(transformed).all():
        return False, 0.0
    area = abs(float(cv2.contourArea(transformed)))
    original_area = max(float((w - 1) * (h - 1)), 1.0)
    area_ratio = area / original_area
    if area <= 1.0 or area_ratio < 1e-3:
        return False, area_ratio
    try:
        if not cv2.isContourConvex(transformed):
            return False, area_ratio
    except cv2.error:
        return False, area_ratio
    return True, area_ratio


def _register_pair(user_raw: np.ndarray, ref_raw: np.ndarray,
                   t0: float, on_stage) -> dict:
    """Core single-pair run on decoded grayscale arrays."""
    report_stage = on_stage or (lambda _s: None)
    user_h, user_w = user_raw.shape
    ref_h, ref_w = ref_raw.shape
    scale = user_h / max(ref_h, 1)
    ref_aligned = cv2.resize(ref_raw, (max(int(ref_w * scale), 1), user_h),
                             interpolation=cv2.INTER_LINEAR)
    user_proc = run_loop_27_preprocessing(user_raw)
    ref_proc = run_loop_27_preprocessing(ref_aligned)

    on_stage("feature_extraction")
    on_stage("matching")
    m = match_pair(user_proc, ref_proc)

    # Fallback: if LoFTR is weak, try SIFT+RootSIFT
    if m["n_candidates"] < 10 or m["inlier_count"] < 10:
        m_sift = match_pair_sift(user_proc, ref_proc)
        if m_sift["inlier_count"] > m["inlier_count"]:
            m = m_sift

    on_stage("outlier_rejection")
    mk0, mk1, mask = m["mkpts0"], m["mkpts1"], m["inlier_mask"]
    order = sorted(range(len(mk0)), key=lambda i: (not mask[i], i))
    order = order[:MAX_MATCHES]
    matches = [{
        "id": f"m-live-{i}",
        "srcX": float(np.clip(mk0[j][0] / user_proc.shape[1], 0, 1)),
        "srcY": float(np.clip(mk0[j][1] / user_proc.shape[0], 0, 1)),
        "refX": float(np.clip(mk1[j][0] / ref_proc.shape[1], 0, 1)),
        "refY": float(np.clip(mk1[j][1] / ref_proc.shape[0], 0, 1)),
        "confidence": 0.85 if mask[j] else 0.25,
        "isInlier": bool(mask[j]),
    } for i, j in enumerate(order)]

    report_stage("transform_fit")
    H = m["H"]
    rmse = m["rmse"] if m["rmse"] != 999.0 else 0.0

    footprint_valid, area_ratio = _validate_footprint(H, user_proc.shape)
    if not footprint_valid:
        raise LunarValidationError(
            "Geometric outlier rejection discarded the proposed alignment.",
            {"footprint_area_ratio": round(area_ratio, 6)},
        )

    report_stage("warping")
    sx, sy = DISPLAY_W / user_proc.shape[1], DISPLAY_H / user_proc.shape[0]
    rx, ry = DISPLAY_W / ref_proc.shape[1], DISPLAY_H / ref_proc.shape[0]
    H_disp = np.array([[rx, 0, 0], [0, ry, 0], [0, 0, 1]]) @ H @ np.array(
        [[1 / sx, 0, 0], [0, 1 / sy, 0], [0, 0, 1]])
    src_disp = _to_display(user_raw)
    ref_disp = _to_display(ref_raw)
    try:
        warped = cv2.warpPerspective(src_disp, H_disp, (DISPLAY_W, DISPLAY_H))
    except cv2.error:
        warped = src_disp

    report_stage("evaluation")
    coverage = _coverage(matches)
    elapsed = time.time() - t0
    n_in, ratio = m["inlier_count"], m["inlier_ratio"]
    if n_in >= 15 and ratio >= 0.6 and rmse < 3.0:
        reliability, reason = "high", None
    elif n_in >= 4 and ratio >= 0.3:
        reliability, reason = "low", "Few inliers or high RMSE — treat alignment as uncertain."
    else:
        reliability, reason = "failed", "No reliable correspondence found for this pair."

    return {
        "matches": matches,
        "transform": {"modelType": "homography",
                      "parameters": [float(v) for v in H.reshape(-1)]},
        "report": {
            "rmseX": round(rmse, 3), "rmseY": round(rmse, 3),
            "inlierCount": n_in, "inlierRatio": round(ratio, 3),
            "coverageScore": round(coverage, 3),
            "processingTimeS": round(elapsed, 2),
            "reliability": reliability, "reliabilityReason": reason,
            "matchPercentage": round(ratio * 100, 2),
            "candidates": m["n_candidates"],
            "footprintValid": footprint_valid,
            "footprintAreaRatio": round(area_ratio, 6),
        },
        "images": {
            "source": _encode_jpg(src_disp),
            "reference": _encode_jpg(ref_disp),
            "warped": _encode_jpg(warped),
        },
    }


def run_registration(src_bytes: bytes, ref_bytes: bytes, on_stage=None) -> dict:
    """Run the full notebook pipeline on one uploaded pair."""
    t0 = time.time()
    report_stage = on_stage or (lambda _s: None)
    report_stage("ingestion")
    user_raw = _decode_gray(src_bytes)
    ref_raw = _decode_gray(ref_bytes)
    report_stage("preprocessing")
    validate_lunar_image(user_raw)
    out = _register_pair(user_raw, ref_raw, t0, report_stage)
    _require_usable_match(out)
    return out


def sweep_archive(src_bytes: bytes, ref_files: list,
                  termination_threshold: float = 60.0, on_stage=None) -> dict:
    """Compare one upload against every archive frame; return the winner.

    Mirrors the notebook's folder sweep: best matchPercentage wins, early stop
    once a candidate crosses the threshold. Returns the winner's full result
    plus a ranked `sweep` table for the UI.
    """
    t0 = time.time()
    report_stage = on_stage or (lambda _s: None)
    report_stage("ingestion")
    user_raw = _decode_gray(src_bytes)
    validate_lunar_image(user_raw)

    sweep, best, best_result = [], None, None
    for path in ref_files:
        loop_t0 = time.time()
        report_stage("preprocessing")
        try:
            ref_raw = _decode_gray(path.read_bytes())
            out = _register_pair(user_raw, ref_raw, loop_t0, report_stage)
        except Exception as exc:  # noqa: BLE001 — one bad frame must not kill the sweep
            sweep.append({"file": path.name, "score": 0.0, "inliers": 0,
                          "rmse": 0.0, "error": str(exc)[:120]})
            continue
        rep = out["report"]
        lat, lon = extract_coordinates_from_filename(path.name)
        entry = {"file": path.name, "lat": lat, "lon": lon,
                 "score": rep["matchPercentage"], "inliers": rep["inlierCount"],
                 "rmse": rep["rmseX"],
                 "runtimeS": round(time.time() - loop_t0, 2)}
        sweep.append(entry)
        if best is None or entry["score"] > best["score"]:
            best, best_result = entry, out
        if entry["score"] >= termination_threshold:
            break

    if best_result is None:
        raise LunarValidationError(
            "No readable reference frame was available in the local archive."
        )
    _require_usable_match(best_result)
    sweep.sort(key=lambda e: e["score"], reverse=True)
    best_result["report"]["sweep"] = sweep
    best_result["report"]["processingTimeS"] = round(time.time() - t0, 2)
    best_result["winner"] = best
    return best_result
