"""HF dynamic matcher — fuses Matching_service.py Tier-2 with
Outlier Rejection.html geometry.

- SIFT + RootSIFT (no CLAHE/photometric touch, per notebook)
- mutual Lowe-ratio across thresholds + cross-check fallback
- partial-affine / full-affine / homography via RANSAC/MAGSAC
- footprint validation + 8x8 coverage + polygon overlap/IoU
- phase-correlation subpixel refinement on the winner only

Works on raw gray arrays (query upload + HF dataset frames), so the
backend stays fully dynamic: no static data/reference/ required.
"""

import logging
import re
import time
from pathlib import Path

import cv2
import numpy as np

logger = logging.getLogger("LunaMatch.hf_matcher")

IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp")

MAX_FEATURES = 4000
SIFT_CONTRAST = 0.01
SIFT_EDGE = 10
FEATURE_MAX_DIM = 1500
RATIO_THRESHOLDS = (0.75, 0.80, 0.85)
RANSAC_THRESH = 5.0
MAX_ITERS = 3000
CONFIDENCE = 0.995
MIN_CANDIDATES = 6
MIN_INLIERS = 6
MIN_RATIO = 0.08
MIN_CELLS = 2
GRID_ROWS, GRID_COLS = 8, 8
SUBPIXEL_RADIUS = 12
MIN_PHASE_RESPONSE = 0.05

DISPLAY_W, DISPLAY_H = 800, 450
MAX_MATCHES = 150


# ---------- filename metadata (Outlier html + minus/plus handling) ----------
def parse_metadata(name: str) -> dict:
    stem = Path(name).stem.strip()
    stem = re.sub(r"_processed$", "", stem, flags=re.IGNORECASE)
    s = stem
    s = re.sub(r"(?i)minus", "-", s)
    s = re.sub(r"(?i)plus", "+", s)
    s = re.sub(r"\((\d+)\)", r"_\1", s)
    s = s.replace(" ", "_")
    s = re.sub(r"_+", "_", s).strip("_")
    m = re.match(
        r"^lat_([+-]?\d+(?:\.\d+)?)_([+-]?\d+(?:\.\d+)?)_lon_"
        r"([+-]?\d+(?:\.\d+)?)_([+-]?\d+(?:\.\d+)?)_(\d+)$",
        s,
        flags=re.IGNORECASE,
    )
    if not m:
        # Fallback: first lat/lon numbers found (Matching_service.py style).
        low = stem.lower().replace("minus", "-")
        lat_m = re.search(r"lat(?:itude)?(?:[-_\s]*|(?=\d|-))(-?\d+(?:\.\d+)?)", low)
        lon_m = re.search(r"lon(?:g|gitude)?(?:[-_\s]*|(?=\d|-))(-?\d+(?:\.\d+)?)", low)
        return {
            "lat": lat_m.group(1) if lat_m else "Unknown",
            "lon": lon_m.group(1) if lon_m else "Unknown",
        }
    lat1, lat2, lon1, lon2, num = m.groups()
    return {
        "lat_min": min(float(lat1), float(lat2)),
        "lat_max": max(float(lat1), float(lat2)),
        "lon_min": min(float(lon1), float(lon2)),
        "lon_max": max(float(lon1), float(lon2)),
        "image_number": int(num),
        "lat": str(min(float(lat1), float(lat2))),
        "lon": str(min(float(lon1), float(lon2))),
    }


# ---------- SIFT ----------
def _sift():
    try:
        return cv2.SIFT_create(
            nfeatures=MAX_FEATURES,
            contrastThreshold=SIFT_CONTRAST,
            edgeThreshold=SIFT_EDGE,
            enable_precise_upscale=True,
        )
    except TypeError:
        return cv2.SIFT_create(
            nfeatures=MAX_FEATURES,
            contrastThreshold=SIFT_CONTRAST,
            edgeThreshold=SIFT_EDGE,
        )


def _resize_capped(img: np.ndarray):
    h, w = img.shape
    scale = min(1.0, FEATURE_MAX_DIM / max(h, w))
    if scale >= 1.0:
        return img, 1.0
    return cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA), scale


def extract(img: np.ndarray):
    view, scale = _resize_capped(img)
    kp, desc = _sift().detectAndCompute(view, None)
    if scale != 1.0 and kp:
        inv = 1.0 / scale
        for k in kp:
            k.pt = (k.pt[0] * inv, k.pt[1] * inv)
            k.size *= inv
    if desc is not None:
        desc = desc.astype(np.float32)
        desc /= desc.sum(axis=1, keepdims=True) + 1e-12
        desc = np.sqrt(desc)
    return kp or [], desc


def _ratio(d1, d2, ratio):
    if d1 is None or d2 is None or len(d1) < 2 or len(d2) < 2:
        return []
    bf = cv2.BFMatcher(cv2.NORM_L2)
    good = []
    for pair in bf.knnMatch(d1, d2, k=2):
        if len(pair) == 2 and pair[0].distance < ratio * pair[1].distance:
            good.append(pair[0])
    return good


def _mutual(d1, d2, ratio):
    fwd = _ratio(d1, d2, ratio)
    rev = _ratio(d2, d1, ratio)
    rev_set = {(m.trainIdx, m.queryIdx) for m in rev}
    return [m for m in fwd if (m.queryIdx, m.trainIdx) in rev_set]


def _crosscheck(d1, d2):
    if d1 is None or d2 is None:
        return []
    m = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True).match(d1, d2)
    return sorted(m, key=lambda x: x.distance)


# ---------- geometry (Outlier Rejection.html) ----------
def _tpts(pts, M, model):
    pts = np.asarray(pts, dtype=np.float32)
    if model == "homography":
        return cv2.perspectiveTransform(pts.reshape(-1, 1, 2), M).reshape(-1, 2)
    return cv2.transform(pts.reshape(-1, 1, 2), M).reshape(-1, 2)


def _sym_err(src, dst, M, model):
    try:
        fwd = _tpts(src, M, model)
        inv = np.linalg.inv(M) if model == "homography" else cv2.invertAffineTransform(M)
        bwd = _tpts(dst, inv, model)
    except (cv2.error, np.linalg.LinAlgError):
        return None
    if not (np.isfinite(fwd).all() and np.isfinite(bwd).all()):
        return None
    fe = np.linalg.norm(fwd - dst, axis=1)
    be = np.linalg.norm(bwd - src, axis=1)
    return np.sqrt((fe**2 + be**2) / 2.0), float(np.sqrt(np.mean(fe**2)))


def _record(model, M, mask, src, dst):
    if M is None or mask is None:
        return None
    mask = mask.ravel().astype(bool)
    if mask.sum() < 1:
        return None
    r = _sym_err(src, dst, M, model)
    if r is None:
        return None
    sym, rmse = r
    return {
        "model": model, "matrix": M, "mask": mask,
        "inliers": int(mask.sum()),
        "inlier_ratio": float(mask.sum()) / max(len(src), 1),
        "rmse": rmse, "sym_rmse": float(np.sqrt(np.mean(sym[mask] ** 2))),
    }


def _est_partial(src, dst):
    if len(src) < 3:
        return None
    try:
        M, m = cv2.estimateAffinePartial2D(
            src, dst, method=cv2.RANSAC, ransacReprojThreshold=RANSAC_THRESH,
            maxIters=MAX_ITERS, confidence=CONFIDENCE, refineIters=10)
    except cv2.error:
        return None
    return _record("partial_affine", M, m, src, dst)


def _est_full(src, dst):
    if len(src) < 3:
        return None
    try:
        M, m = cv2.estimateAffine2D(
            src, dst, method=cv2.RANSAC, ransacReprojThreshold=RANSAC_THRESH,
            maxIters=MAX_ITERS, confidence=CONFIDENCE, refineIters=10)
    except cv2.error:
        return None
    return _record("full_affine", M, m, src, dst)


def _est_h(src, dst):
    if len(src) < 4:
        return None
    method = getattr(cv2, "USAC_MAGSAC", cv2.RANSAC)
    try:
        M, m = cv2.findHomography(src.reshape(-1, 1, 2), dst.reshape(-1, 1, 2),
                                  method, RANSAC_THRESH, None, MAX_ITERS, CONFIDENCE)
    except cv2.error:
        return None
    if M is None:
        try:
            M, m = cv2.findHomography(src.reshape(-1, 1, 2), dst.reshape(-1, 1, 2),
                                      cv2.RANSAC, RANSAC_THRESH)
        except cv2.error:
            return None
    return _record("homography", M, m, src, dst)


def _coverage(pts, shape):
    h, w = shape
    cells = set()
    for x, y in pts:
        c = min(GRID_COLS - 1, max(0, int(x / max(w, 1) * GRID_COLS)))
        r = min(GRID_ROWS - 1, max(0, int(y / max(h, 1) * GRID_ROWS)))
        cells.add((r, c))
    return len(cells), 100.0 * len(cells) / (GRID_ROWS * GRID_COLS)


def _footprint(M, model, shape):
    h, w = shape
    corners = np.float32([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]])
    try:
        t = _tpts(corners, M, model)
    except (cv2.error, np.linalg.LinAlgError):
        return False, 0.0
    if not np.isfinite(t).all():
        return False, 0.0
    area = abs(float(cv2.contourArea(t.astype(np.float32))))
    ratio = area / max(float((w - 1) * (h - 1)), 1.0)
    if area <= 1.0 or ratio < 1e-3:
        return False, ratio
    if model == "homography":
        try:
            if not cv2.isContourConvex(t.astype(np.float32)):
                return False, ratio
        except cv2.error:
            return False, ratio
    return True, ratio


def _overlap(q_img, db_img, M, model):
    h1, w1 = q_img.shape
    h2, w2 = db_img.shape
    src = np.float32([[0, 0], [w1 - 1, 0], [w1 - 1, h1 - 1], [0, h1 - 1]])
    ref = np.float32([[0, 0], [w2 - 1, 0], [w2 - 1, h2 - 1], [0, h2 - 1]])
    try:
        t = _tpts(src, M, model).astype(np.float32)
    except (cv2.error, np.linalg.LinAlgError):
        return 0.0, 0.0
    if not np.isfinite(t).all():
        return 0.0, 0.0
    sa, ra = abs(cv2.contourArea(t)), abs(cv2.contourArea(ref))
    if sa <= 0 or ra <= 0:
        return 0.0, 0.0
    try:
        inter, _ = cv2.intersectConvexConvex(t, ref)
    except cv2.error:
        return 0.0, 0.0
    if inter <= 0:
        return 0.0, 0.0
    union = sa + ra - inter
    return float(min(100.0, 100.0 * inter / min(sa, ra))), float(
        min(100.0, 100.0 * inter / union) if union > 0 else 0.0)


def _key(g):
    comp = {"partial_affine": 0, "full_affine": 1, "homography": 2}
    return (g["inliers"], g["inlier_ratio"], g.get("cells", 0),
            -g["sym_rmse"], -comp[g["model"]])


def compare_pair(q_img, q_kp, q_desc, db_img):
    """Full Outlier-Rejection comparison of one query vs one DB frame."""
    t0 = time.time()
    fail = {"file": "", "score": 0.0, "inliers": 0, "rmse": 0.0}
    try:
        db_kp, db_desc = extract(db_img)
    except Exception as exc:
        fail["error"] = str(exc)[:120]
        return None, fail
    if db_desc is None or q_desc is None or len(q_kp) < 4 or len(db_kp) < 4:
        fail["error"] = "descriptor_starvation"
        return None, fail

    attempts = []
    for r in RATIO_THRESHOLDS:
        attempts.append((f"mutual_{r:.2f}", _mutual(q_desc, db_desc, r)))
    attempts.append(("crosscheck", _crosscheck(q_desc, db_desc)))

    best, best_fail = None, None
    for method, matches in attempts:
        if len(matches) < MIN_CANDIDATES:
            cand = {"method": method, "candidates": len(matches), "inliers": 0}
            if best_fail is None or len(matches) > best_fail["candidates"]:
                best_fail = cand
            continue
        src = np.float32([q_kp[m.queryIdx].pt for m in matches])
        dst = np.float32([db_kp[m.trainIdx].pt for m in matches])
        cands = [g for g in (_est_partial(src, dst), _est_full(src, dst), _est_h(src, dst)) if g]
        if not cands:
            continue
        for g in cands:
            cells, cov = _coverage(src[g["mask"]], q_img.shape)
            g["cells"], g["coverage"] = cells, cov
            ok, ar = _footprint(g["matrix"], g["model"], q_img.shape)
            g["valid"], g["area_ratio"] = ok, ar
        usable = [g for g in cands if g["valid"] and g["inliers"] >= MIN_INLIERS
                  and g["inlier_ratio"] >= MIN_RATIO and g["cells"] >= MIN_CELLS]
        cands.sort(key=_key, reverse=True)
        if not usable:
            fb = cands[0]
            diag = {"method": method, "candidates": len(matches),
                    "inliers": fb["inliers"], "inlier_ratio": fb["inlier_ratio"]}
            if best_fail is None or (diag["inliers"], diag["inlier_ratio"]) > (
                    best_fail.get("inliers", 0), best_fail.get("inlier_ratio", 0.0)):
                best_fail = diag
            continue
        usable.sort(key=_key, reverse=True)
        g = usable[0]
        ov, iou = _overlap(q_img, db_img, g["matrix"], g["model"])
        cand = {"method": method, "matches": matches, "src": src, "dst": dst,
                "geo": g, "q_kp": q_kp, "db_kp": db_kp, "overlap": ov, "iou": iou}
        if best is None or _key(g) > _key(best["geo"]):
            best = cand
    if best is None:
        fail.update({"candidates": (best_fail or {}).get("candidates", 0),
                     "inliers": (best_fail or {}).get("inliers", 0)})
        return None, fail
    g = best["geo"]
    ov, iou = best["overlap"], best["iou"]
    score = g["inlier_ratio"] * 100.0 * (0.5 + 0.5 * min(g["cells"] / 16.0, 1.0))
    result = {
        "method": best["method"], "matches": best["matches"],
        "src": best["src"], "dst": best["dst"], "geo": g,
        "q_kp": best["q_kp"], "db_kp": best["db_kp"],
        "overlap": ov, "iou": iou, "score": score,
        "inliers": g["inliers"], "inlier_ratio": g["inlier_ratio"],
        "cells": g["cells"], "coverage": g["coverage"],
        "rmse": g["sym_rmse"], "model": g["model"],
        "matrix": g["matrix"], "mask": g["mask"],
        "runtime": round(time.time() - t0, 2),
    }
    return result, None


def _refine_winner(q_img, db_img, res):
    """Phase-correlation subpixel refit on uniform control points (notebook §13)."""
    try:
        mask = res["mask"]
        src, dst = res["src"], res["dst"]
        idx = np.flatnonzero(mask)
        if len(idx) == 0:
            return res
        # Uniform 8x8 selection: best (lowest residual proxy = match distance order) per cell.
        order = np.argsort([res["matches"][i].distance for i in idx])
        seen, sel = set(), []
        h, w = q_img.shape
        for i in order:
            gi = int(idx[i])
            x, y = src[gi]
            cell = (min(7, max(0, int(y / max(h, 1) * 8))), min(7, max(0, int(x / max(w, 1) * 8))))
            if cell not in seen:
                seen.add(cell)
                sel.append(gi)
            if len(sel) >= 40:
                break
        if len(sel) < 4:
            return res
        c_src = src[sel]
        # Predict with current model then phase-correlate patches.
        pred = _tpts(c_src, res["matrix"], res["model"])
        refined = []
        responses = []
        for s, p in zip(c_src, pred):
            for img, pt in ((q_img, s), (db_img, p)):
                pass
            r = SUBPIXEL_RADIUS
            sx, sy, dx, dy = int(round(s[0])), int(round(s[1])), int(round(p[0])), int(round(p[1]))
            if not (sx - r >= 0 and sy - r >= 0 and sx + r + 1 <= q_img.shape[1]
                    and sy - r >= 0 and dx - r >= 0 and dy - r >= 0
                    and dx + r + 1 <= db_img.shape[1] and dy + r + 1 <= db_img.shape[0]):
                refined.append(p)
                continue
            p1 = q_img[sy - r:sy + r + 1, sx - r:sx + r + 1].astype(np.float32)
            p2 = db_img[dy - r:dy + r + 1, dx - r:dx + r + 1].astype(np.float32)
            p1 -= p1.mean()
            p2 -= p2.mean()
            if p1.std() < 1e-6 or p2.std() < 1e-6:
                refined.append(p)
                continue
            win = cv2.createHanningWindow(p1.shape, cv2.CV_32F)
            try:
                shift, resp = cv2.phaseCorrelate(p1, p2, win)
            except cv2.error:
                refined.append(p)
                continue
            if float(resp) < MIN_PHASE_RESPONSE:
                refined.append(p)
                continue
            q = np.array([p[0] + shift[0], p[1] + shift[1]], dtype=np.float32)
            q[0] = np.clip(q[0], 0, db_img.shape[1] - 1)
            q[1] = np.clip(q[1], 0, db_img.shape[0] - 1)
            refined.append(q)
            responses.append(float(resp))
        refined = np.asarray(refined, dtype=np.float32)
        model = res["model"]
        if model == "partial_affine":
            M2, m2 = cv2.estimateAffinePartial2D(c_src, refined, method=cv2.RANSAC,
                                                ransacReprojThreshold=RANSAC_THRESH)
        elif model == "full_affine":
            M2, m2 = cv2.estimateAffine2D(c_src, refined, method=cv2.RANSAC,
                                         ransacReprojThreshold=RANSAC_THRESH)
        else:
            M2, m2 = cv2.findHomography(c_src.reshape(-1, 1, 2), refined.reshape(-1, 1, 2),
                                        getattr(cv2, "USAC_MAGSAC", cv2.RANSAC), RANSAC_THRESH)
        if M2 is not None and m2 is not None and int((m2.ravel() == 1).sum()) >= MIN_INLIERS:
            rec = _record(model, M2, m2, c_src, refined)
            if rec is not None and rec["sym_rmse"] < res["rmse"]:
                res = {**res, "matrix": M2, "mask_refined": m2.ravel().astype(bool),
                       "rmse": rec["sym_rmse"], "refined": True,
                       "subpix_points": len(responses),
                       "mean_phase": float(np.mean(responses)) if responses else None}
                # Rebuild src/dst/matches view for frontend from control set.
                res["src"], res["dst"] = c_src, refined
        return res
    except Exception:
        return res


def sweep(query_img: np.ndarray, ref_paths: list[Path], on_stage=None,
          limit: int | None = None, early_stop: float = 60.0) -> dict:
    """Dynamic sweep over HF (or local) frames. Returns winner + ranked sweep."""
    t0 = time.time()
    report = on_stage or (lambda _s: None)
    report("feature_extraction")
    q_kp, q_desc = extract(query_img)
    if q_desc is None or len(q_kp) < 4:
        raise ValueError("Query texture starvation: insufficient interest landmarks.")
    paths = ref_paths[:limit] if limit else ref_paths
    sweep_rows, best, best_path, best_db, best_res = [], None, None, None, None
    for p in paths:
        report("matching")
        try:
            arr = np.frombuffer(Path(p).read_bytes(), dtype=np.uint8)
            db_img = cv2.imdecode(arr, cv2.IMREAD_GRAYSCALE)
            if db_img is None:
                raise ValueError("undecodable")
        except Exception as exc:
            sweep_rows.append({"file": Path(p).name, "score": 0.0,
                               "inliers": 0, "rmse": 0.0, "error": str(exc)[:120]})
            continue
        res, fail = compare_pair(query_img, q_kp, q_desc, db_img)
        meta = parse_metadata(Path(p).name)
        if res is None:
            sweep_rows.append({"file": Path(p).name, "lat": meta.get("lat", "?"),
                               "lon": meta.get("lon", "?"), "score": 0.0,
                               "inliers": (fail or {}).get("inliers", 0), "rmse": 0.0,
                               "runtimeS": 0.0})
            continue
        row = {"file": Path(p).name, "lat": meta.get("lat", "?"), "lon": meta.get("lon", "?"),
               "score": round(res["score"], 2), "inliers": res["inliers"],
               "rmse": round(res["rmse"], 3), "overlap": round(res["overlap"], 1),
               "iou": round(res["iou"], 1), "model": res["model"],
               "runtimeS": res["runtime"]}
        sweep_rows.append(row)
        if best is None or row["score"] > best["score"]:
            best, best_path, best_db, best_res = dict(row), p, db_img, res
        if row["score"] >= early_stop:
            break
    if best is None or best_res is None:
        raise ValueError("No correspondence found in the reference set.")
    report("outlier_rejection")
    winner_res = _refine_winner(query_img, best_db, best_res)
    report("subpixel_refinement")
    sweep_rows.sort(key=lambda e: e["score"], reverse=True)
    return {"winner_row": best, "winner_path": best_path, "winner_image": best_db,
            "winner_result": winner_res, "sweep": sweep_rows,
            "query_kp_count": len(q_kp), "elapsed": round(time.time() - t0, 2)}


def _to_display(gray: np.ndarray) -> np.ndarray:
    return cv2.resize(gray, (DISPLAY_W, DISPLAY_H), interpolation=cv2.INTER_AREA)


def _encode(gray: np.ndarray) -> bytes:
    ok, buf = cv2.imencode(".jpg", gray, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
    if not ok:
        raise RuntimeError("JPEG encode failed.")
    return bytes(buf)


def to_frontend(query_img: np.ndarray, db_img: np.ndarray, res: dict, elapsed: float) -> dict:
    """Convert winner geometry to the /api/jobs/result contract (0..1 coords)."""
    qh, qw = query_img.shape
    dh, dw = db_img.shape
    src, dst, mask = res["src"], res["dst"], res["mask"]
    order = sorted(range(len(src)), key=lambda i: (not mask[i], i))[:MAX_MATCHES]
    matches = [{
        "id": f"m-hf-{i}",
        "srcX": float(np.clip(src[j][0] / max(qw, 1), 0, 1)),
        "srcY": float(np.clip(src[j][1] / max(qh, 1), 0, 1)),
        "refX": float(np.clip(dst[j][0] / max(dw, 1), 0, 1)),
        "refY": float(np.clip(dst[j][1] / max(dh, 1), 0, 1)),
        "confidence": 0.85 if mask[j] else 0.25,
        "isInlier": bool(mask[j]),
    } for i, j in enumerate(order)]
    M = res["matrix"]
    H = M if res["model"] == "homography" else np.vstack([M, [0, 0, 1]])
    sx, sy = DISPLAY_W / max(qw, 1), DISPLAY_H / max(qh, 1)
    rx, ry = DISPLAY_W / max(dw, 1), DISPLAY_H / max(dh, 1)
    H_disp = np.array([[rx, 0, 0], [0, ry, 0], [0, 0, 1]]) @ H @ np.array(
        [[1 / sx, 0, 0], [0, 1 / sy, 0], [0, 0, 1]])
    try:
        warped = cv2.warpPerspective(_to_display(query_img), H_disp, (DISPLAY_W, DISPLAY_H))
    except cv2.error:
        warped = _to_display(query_img)
    cells = {(min(7, int(m["srcX"] * 8)), min(5, int(m["srcY"] * 6))) for m in matches if m["isInlier"]}
    coverage = len(cells) / 48.0
    n_in, ratio, rmse = res["inliers"], res["inlier_ratio"], res["rmse"]
    if n_in >= 10 and ratio >= 0.10 and res.get("overlap", 0) >= 10.0 and rmse <= 6.0:
        reliability, reason = "high", None
    elif n_in >= 6 and ratio >= 0.08:
        reliability, reason = "low", "Weak geometry — treat alignment as uncertain."
    else:
        reliability, reason = "failed", "No reliable correspondence for this pair."
    return {
        "matches": matches,
        "transform": {"modelType": res["model"],
                      "parameters": [float(v) for v in np.asarray(H).reshape(-1)]},
        "report": {
            "rmseX": round(float(rmse), 3), "rmseY": round(float(rmse), 3),
            "inlierCount": n_in, "inlierRatio": round(float(ratio), 3),
            "coverageScore": round(float(coverage), 3),
            "processingTimeS": round(float(elapsed), 2),
            "reliability": reliability, "reliabilityReason": reason,
            "matchPercentage": round(float(res.get("score", ratio * 100)), 2),
            "candidates": len(src), "method": res.get("method"),
            "overlap": round(float(res.get("overlap", 0)), 1),
            "iou": round(float(res.get("iou", 0)), 1),
            "footprintValid": True,
        },
        "images": {"source": _encode(_to_display(query_img)),
                   "reference": _encode(_to_display(db_img)),
                   "warped": _encode(warped)},
    }
