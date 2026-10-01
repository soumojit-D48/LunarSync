"""Learned matching — LoFTR (outdoor) + MAGSAC, adapted from Untitled3.ipynb.

Single-pair version of the notebook's folder-sweep loop body. Runs on CPU or
CUDA depending on availability.

torch/kornia are imported LAZILY inside get_matcher() so plain SIFT sweeps
(cloud + HF archive modes) boot lean on small instances. Only explicit-pair
uploads pay the torch cost.
"""

import cv2
import numpy as np

# Notebook's anti-freeze working resolution.
DOWN_W, DOWN_H = 800, 200

_matcher = None
_device = None


def get_matcher():
    global _matcher, _device
    if _matcher is None:
        import torch  # noqa: PLC0415 — lazy so cloud/HF sweeps boot without torch
        import kornia as K  # noqa: PLC0415
        import kornia.feature as KF  # noqa: PLC0415

        _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        _matcher = KF.LoFTR(pretrained="outdoor").to(_device)
        _matcher.eval()
        get_matcher.K = K
        get_matcher.torch = torch
    return _matcher


def match_pair(user_proc: np.ndarray, ref_proc: np.ndarray) -> dict:
    """Match two preprocessed grayscale images.

    Returns keypoints in FULL working-resolution coordinates plus fit stats.
    """
    user_h, user_w = user_proc.shape
    ref_h, ref_w = ref_proc.shape

    user_down = cv2.resize(user_proc, (DOWN_W, DOWN_H), interpolation=cv2.INTER_AREA)
    ref_down = cv2.resize(ref_proc, (DOWN_W, DOWN_H), interpolation=cv2.INTER_AREA)

    matcher = get_matcher()
    K, torch, DEVICE = get_matcher.K, get_matcher.torch, _device
    t_user = K.image.image_to_tensor(user_down, keepdim=False).float().to(DEVICE) / 255.0
    t_ref = K.image.image_to_tensor(ref_down, keepdim=False).float().to(DEVICE) / 255.0

    with torch.inference_mode():
        try:
            correspondences = matcher({"image0": t_user, "image1": t_ref})
            mkpts0 = correspondences["keypoints0"].cpu().numpy()
            mkpts1 = correspondences["keypoints1"].cpu().numpy()
        except Exception:
            mkpts0, mkpts1 = np.zeros((0, 2)), np.zeros((0, 2))

    n_candidates = len(mkpts0)
    inlier_count = 0
    rmse = 999.0
    H = np.eye(3, dtype=float)
    inlier_mask = np.zeros(n_candidates, dtype=bool)

    if n_candidates >= 6:
        mkpts0[:, 0] *= user_w / DOWN_W
        mkpts0[:, 1] *= user_h / DOWN_H
        mkpts1[:, 0] *= ref_w / DOWN_W
        mkpts1[:, 1] *= ref_h / DOWN_H

        H_fit, inliers = cv2.findHomography(mkpts0, mkpts1, cv2.USAC_MAGSAC, 15.0, 0.99, 2000)
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
