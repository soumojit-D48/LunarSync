"""HF reference store — dynamic replacement for static data/reference/.

Uses the raw-image dataset from Outlier Rejection.html
(`lunar-team-2026/data-image`, 626 files via snapshot_download)
authenticated with HF_TOKEN from .env.

Falls back to local data/reference/ when HF is unreachable so the
backend never goes static-dead on deploy without network/token.
"""

import logging
import os
from pathlib import Path

logger = logging.getLogger("LunaMatch.hf_store")

IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp")

# Dataset from the Outlier Rejection notebook (raw images, not descriptors).
HF_DATASET_DEFAULT = "lunar-team-2026/data-image"
# Legacy precomputed-descriptor repo from Matching_service.py (kept for reference).
HF_FEATURES_REPO_DEFAULT = "PokiMew/extracted-features"


def hf_token() -> str:
    # Lazy import so backend boots even without python-dotenv installed.
    try:
        from dotenv import load_dotenv  # type: ignore

        load_dotenv()
    except Exception:
        pass
    return (os.getenv("HF_TOKEN") or "").strip().strip("\"'")


def hf_dataset() -> str:
    try:
        from dotenv import load_dotenv  # type: ignore

        load_dotenv()
    except Exception:
        pass
    return (os.getenv("HF_DATASET") or HF_DATASET_DEFAULT).strip()


def local_snapshot() -> Path | None:
    """Return cached HF snapshot dir if already downloaded (no network)."""
    try:
        from huggingface_hub import snapshot_download  # type: ignore

        # local_files_only avoids network; raises if not cached.
        p = snapshot_download(
            repo_id=hf_dataset(),
            repo_type="dataset",
            allow_patterns=["*.png", "*.jpg", "*.jpeg", "*.tif", "*.tiff", "*.bmp"],
            local_files_only=True,
        )
        return Path(p)
    except Exception:
        return None


def ensure_dataset() -> Path | None:
    """Download (and cache) the HF image dataset. Returns snapshot dir or None."""
    token = hf_token()
    try:
        from huggingface_hub import snapshot_download  # type: ignore

        local_path = snapshot_download(
            repo_id=hf_dataset(),
            repo_type="dataset",
            allow_patterns=["*.png", "*.jpg", "*.jpeg", "*.tif", "*.tiff", "*.bmp"],
            max_workers=8,
            token=token or None,
        )
        logger.info("HF dataset ready at %s", local_path)
        return Path(local_path)
    except Exception as exc:
        logger.warning("HF dataset unavailable (%s); using local archive.", exc)
        return local_snapshot()


def list_reference_images(snapshot: Path | None, local_dirs: list[Path]) -> list[Path]:
    """Dynamic listing: HF snapshot first, then local dirs as fallback."""
    out: list[Path] = []
    if snapshot is not None and snapshot.exists():
        for root, _, files in os.walk(snapshot):
            for f in files:
                if f.lower().endswith(IMAGE_EXTENSIONS):
                    out.append(Path(root) / f)
        if out:
            out.sort()
            logger.info("Using %d HF reference frames from %s", len(out), snapshot)
            return out
    for d in local_dirs:
        if d.exists():
            files = sorted(
                p for p in d.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS and p.is_file()
            )
            if files:
                logger.info("Using %d local reference frames from %s", len(files), d)
                return files
    return out


def status() -> dict:
    snap = local_snapshot()
    return {
        "dataset": hf_dataset(),
        "token_configured": bool(hf_token()),
        "cached_snapshot": str(snap) if snap else None,
    }
