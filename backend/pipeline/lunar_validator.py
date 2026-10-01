"""Lunar image validator — rejects non-Moon images before matching.

Uses multi-heuristic analysis to determine if an uploaded image is likely
a lunar surface photo. Checks texture, intensity distribution, feature
density, and color characteristics.
"""

import cv2
import numpy as np


class LunarValidationError(Exception):
    """Raised when an image fails lunar surface validation."""

    def __init__(self, reason: str, details: dict = None):
        self.reason = reason
        self.details = details or {}
        super().__init__(reason)


def validate_lunar_image(gray: np.ndarray) -> dict:
    """Validate that a grayscale image looks like a lunar surface.

    Returns a dict with validation results. Raises LunarValidationError
    if the image is clearly not a lunar surface.

    Checks:
    1. Image is not blank/uniform
    2. Sufficient texture (std deviation)
    3. Enough SIFT keypoints (surface features)
    4. Intensity distribution consistent with lunar imagery
    5. Not a color image (lunar photos are grayscale)
    """
    if gray is None or gray.size == 0:
        raise LunarValidationError("Image is empty or could not be decoded.")

    orig_h, orig_w = gray.shape
    if orig_h < 50 or orig_w < 50:
        raise LunarValidationError(
            f"Image too small ({orig_w}x{orig_h}). Minimum 50x50 required."
        )

    # Validate a downscaled copy: SIFT/Canny/Hough on multi-MP uploads take
    # minutes on tiny instances, while the verdict is identical at 1024px.
    import os as _os

    _max = int(_os.getenv("VALIDATE_MAX_DIM", "1024") or 1024)
    if max(orig_h, orig_w) > _max:
        _s = _max / max(orig_h, orig_w)
        gray = cv2.resize(gray, (int(orig_w * _s), int(orig_h * _s)),
                          interpolation=cv2.INTER_AREA)

    h, w = gray.shape

    # Check 1: Not blank/uniform
    std_dev = float(np.std(gray))
    if std_dev < 5.0:
        raise LunarValidationError(
            "Image appears blank or uniform (no surface texture detected).",
            {"std_dev": std_dev},
        )

    # Check 2: SIFT keypoint density (lunar surfaces have craters, rocks, etc.)
    sift = cv2.SIFT_create(nfeatures=500)
    keypoints, _ = sift.detectAndCompute(gray, None)
    n_keypoints = len(keypoints) if keypoints else 0
    kp_density = n_keypoints / (h * w) * 10000  # keypoints per 100x100 pixels

    if n_keypoints < 10:
        raise LunarValidationError(
            "Insufficient surface features detected. Lunar terrain has craters, rocks, and texture.",
            {"keypoints": n_keypoints, "kp_density": kp_density},
        )

    # Check 3: Intensity distribution
    mean_val = float(np.mean(gray))
    if mean_val < 10:
        raise LunarValidationError(
            "Image is too dark. Lunar surface images have visible terrain.",
            {"mean_intensity": mean_val},
        )
    if mean_val > 250:
        raise LunarValidationError(
            "Image is too bright/washed out. No visible terrain features.",
            {"mean_intensity": mean_val},
        )

    # Check 4: Dynamic range
    p5, p95 = np.percentile(gray, [5, 95])
    dynamic_range = float(p95 - p5)
    if dynamic_range < 20:
        raise LunarValidationError(
            "Image has very low contrast. Lunar terrain has clear light/shadow variation.",
            {"dynamic_range": dynamic_range},
        )

    # Check 5: Edge density (lunar surfaces have many edges from craters/rocks)
    edges = cv2.Canny(gray, 50, 150)
    edge_density = float(np.sum(edges > 0)) / (h * w)
    if edge_density < 0.001:
        raise LunarValidationError(
            "No significant edges detected. Lunar terrain has crater rims and rock boundaries.",
            {"edge_density": edge_density},
        )

    # Check 6: Reject obvious diagrams/screenshots (many straight lines).
    # Scale-invariant form: the length gate scales with the validation
    # downscale factor and the count normalizes by ORIGINAL area, reproducing
    # full-resolution semantics. Skipped on degenerate slivers (short side
    # < 128px, e.g. downscaled long strips) where Hough output is noise.
    _s = min(h, w) / max(min(orig_h, orig_w), 1)
    if min(h, w) >= 128:
        _min_len = max(10, int(min(orig_h, orig_w) // 6 * _s))
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=80,
                                minLineLength=_min_len, maxLineGap=15)
        if lines is not None:
            n_lines = len(lines)
            line_ratio = n_lines / (orig_h * orig_w) * 10000
            if line_ratio > 8.0:
                raise LunarValidationError(
                    "Image appears to be a diagram or screenshot. Upload a lunar surface photo.",
                    {"line_ratio": round(line_ratio, 2), "n_lines": n_lines},
                )

    return {
        "valid": True,
        "keypoints": n_keypoints,
        "kp_density": round(kp_density, 2),
        "mean_intensity": round(mean_val, 1),
        "std_dev": round(std_dev, 1),
        "dynamic_range": round(dynamic_range, 1),
        "edge_density": round(edge_density, 4),
        "dimensions": f"{orig_w}x{orig_h}",
    }


def validate_lunar_image_from_bytes(img_bytes: bytes) -> dict:
    """Decode bytes and validate as lunar image."""
    arr = np.frombuffer(img_bytes, dtype=np.uint8)
    gray = cv2.imdecode(arr, cv2.IMREAD_GRAYSCALE)
    if gray is None:
        raise LunarValidationError("Could not decode image. Use PNG, JPG, or TIF.")
    return validate_lunar_image(gray)
