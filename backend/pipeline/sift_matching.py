"""SIFT + RootSIFT fallback matcher — adapted from Matching_service.py.

Used when LoFTR produces too few candidates. Uses SIFT features with
RootSIFT normalization, BFMatcher ratio test, and MAGSAC homography.
"""

import cv2
import numpy as np


def match_pair_sift(user_proc: np.ndarray, ref_proc: np.ndarray) -> dict:
    """Match two preprocessed grayscale images using SIFT + RootSIFT.

    Returns keypoints in FULL working-resolution coordinates plus fit stats.
    """
    # Resize for speed — SIFT on full-res lunar frames is too slow on CPU
    max_dim = 800
    uh, uw = user_proc.shape
    rh, rw = ref_proc.shape
    u_scale = min(1.0, max_dim / max(uh, uw))
    r_scale = min(1.0, max_dim / max(rh, rw))
    u_img = cv2.resize(user_proc, (int(uw * u_scale), int(uh * u_scale)), interpolation=cv2.INTER_AREA) if u_scale < 1.0 else user_proc
    r_img = cv2.resize(ref_proc, (int(rw * r_scale), int(rh * r_scale)), interpolation=cv2.INTER_AREA) if r_scale < 1.0 else ref_proc

    sift = cv2.SIFT_create(nfeatures=2000, contrastThreshold=0.002, edgeThreshold=20)

    kp0, desc0 = sift.detectAndCompute(user_proc, None)
    kp1, desc1 = sift.detectAndCompute(ref_proc, None)

    n_candidates = 0
    inlier_count = 0
    rmse = 999.0
    H = np.eye(3, dtype=float)
    inlier_mask = np.zeros(0, dtype=bool)
    mkpts0 = np.zeros((0, 2))
    mkpts1 = np.zeros((0, 2))

    if desc0 is None or desc1 is None or len(kp0) < 4 or len(kp1) < 4:
        return {
            "mkpts0": mkpts0,
            "mkpts1": mkpts1,
            "inlier_mask": inlier_mask,
            "H": H,
            "n_candidates": 0,
            "inlier_count": 0,
            "inlier_ratio": 0.0,
            "rmse": 999.0,
            "confidence": 0.0,
        }

    # RootSIFT normalization
    desc0 = desc0.astype(np.float32)
    desc0 /= desc0.sum(axis=1, keepdims=True) + 1e-12
    desc0 = np.sqrt(desc0)

    desc1 = desc1.astype(np.float32)
    desc1 /= desc1.sum(axis=1, keepdims=True) + 1e-12
    desc1 = np.sqrt(desc1)

    bf = cv2.BFMatcher(cv2.NORM_L2)
    raw_matches = bf.knnMatch(desc0, desc1, k=2)

    good_matches = []
    for m_ratio in raw_matches:
        if len(m_ratio) == 2 and m_ratio[0].distance < 0.75 * m_ratio[1].distance:
            good_matches.append(m_ratio[0])

    n_candidates = len(good_matches)

    if n_candidates >= 6:
        mkpts0 = np.float32([kp0[m.queryIdx].pt for m in good_matches]).reshape(-1, 2)
        mkpts1 = np.float32([kp1[m.trainIdx].pt for m in good_matches]).reshape(-1, 2)

        # Scale back to full resolution
        mkpts0[:, 0] /= u_scale
        mkpts0[:, 1] /= u_scale
        mkpts1[:, 0] /= r_scale
        mkpts1[:, 1] /= r_scale

        H_fit, inliers = cv2.findHomography(mkpts0, mkpts1, cv2.USAC_MAGSAC, 3.0, 0.99, 3000)
        if H_fit is not None:
            H = H_fit
        if inliers is not None:
            inlier_mask = inliers.ravel() == 1
            inlier_count = int(np.sum(inlier_mask))
            if inlier_count > 0:
                pts0_in = mkpts0[inlier_mask].reshape(-1, 1, 2)
                pts1_in = mkpts1[inlier_mask].reshape(-1, 1, 2)
                projected = cv2.perspectiveTransform(pts0_in, H)
                rmse = float(np.sqrt(np.mean((pts1_in - projected) ** 2)))

    inlier_ratio = (inlier_count / n_candidates) if n_candidates else 0.0
    if inlier_count >= 5:
        confidence = (inlier_count / (inlier_count + (rmse / 5.0) + 0.1)) * 100
        confidence = float(min(max(confidence, inlier_ratio * 100), 100.0))
    else:
        confidence = 0.0

    return {
        "mkpts0": mkpts0,
        "mkpts1": mkpts1,
        "inlier_mask": inlier_mask,
        "H": H,
        "n_candidates": n_candidates,
        "inlier_count": inlier_count,
        "inlier_ratio": inlier_ratio,
        "rmse": rmse,
        "confidence": confidence,
    }
