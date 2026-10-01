"""Cloud reference backend — Cloudinary (pixels) + Neon Postgres (rows).

Same matching logic as the HF path (SIFT + outlier rejection in
hf_matcher); only the image set differs: the user-seeded Cloudinary
photos listed in Neon, not the 626-frame HF archive.

Toggle with REFERENCE_BACKEND=hf|cloud (.env). HF code is untouched.
"""

import logging
import os
import re
import uuid
from pathlib import Path

logger = logging.getLogger("LunaMatch.cloud_store")

BASE = Path(__file__).resolve().parent.parent
CLOUD_CACHE = BASE / "static" / "cloud_cache"
CLOUD_CACHE.mkdir(parents=True, exist_ok=True)


def _env(name: str, default: str = "") -> str:
    try:
        from dotenv import load_dotenv  # type: ignore

        load_dotenv()
    except Exception:
        pass
    return (os.getenv(name) or default).strip()


def mode() -> str:
    return (_env("REFERENCE_BACKEND", "hf") or "hf").lower()


def db_url() -> str:
    return _env("DATABASE_URL", "")


def cloudinary_configured() -> bool:
    return bool(
        _env("CLOUDINARY_CLOUD_NAME")
        and _env("CLOUDINARY_API_KEY")
        and _env("CLOUDINARY_API_SECRET")
    )


# ---------- Neon ----------
def _conn():
    import psycopg  # type: ignore

    url = db_url()
    if not url:
        raise RuntimeError("DATABASE_URL is not set (Neon connection string).")
    return psycopg.connect(url, connect_timeout=10)


def ensure_schema() -> None:
    """Create reference_frames table if missing. No-op without DATABASE_URL."""
    if not db_url():
        return
    with _conn() as c, c.cursor() as cur:
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS reference_frames (
              id TEXT PRIMARY KEY,
              public_id TEXT UNIQUE NOT NULL,
              secure_url TEXT NOT NULL,
              filename TEXT NOT NULL,
              lat TEXT,
              lon TEXT,
              width INT,
              height INT,
              created_at TIMESTAMPTZ DEFAULT now()
            )
            """
        )
        c.commit()


def list_references() -> list[dict]:
    ensure_schema()
    if not db_url():
        return []
    with _conn() as c, c.cursor() as cur:
        cur.execute(
            "SELECT id, public_id, secure_url, filename, lat, lon, width, height,"
            " to_char(created_at,'YYYY-MM-DD\"T\"HH24:MI:SS\"Z\"') FROM reference_frames"
            " ORDER BY created_at DESC"
        )
        rows = cur.fetchall()
    return [
        {"id": r[0], "publicId": r[1], "secureUrl": r[2], "filename": r[3],
         "lat": r[4], "lon": r[5], "width": r[6], "height": r[7], "createdAt": r[8]}
        for r in rows
    ]


def insert_reference(public_id: str, secure_url: str, filename: str,
                     lat: str | None, lon: str | None,
                     width: int | None, height: int | None) -> dict:
    ensure_schema()
    ref_id = f"ref-{uuid.uuid4().hex[:8]}"
    with _conn() as c, c.cursor() as cur:
        cur.execute(
            "INSERT INTO reference_frames"
            " (id, public_id, secure_url, filename, lat, lon, width, height)"
            " VALUES (%s,%s,%s,%s,%s,%s,%s,%s)",
            (ref_id, public_id, secure_url, filename, lat, lon, width, height),
        )
        c.commit()
    return {"id": ref_id, "publicId": public_id, "secureUrl": secure_url,
            "filename": filename, "lat": lat, "lon": lon}


def delete_reference(ref_id: str) -> dict | None:
    ensure_schema()
    with _conn() as c, c.cursor() as cur:
        cur.execute("SELECT public_id FROM reference_frames WHERE id=%s", (ref_id,))
        row = cur.fetchone()
        if not row:
            return None
        cur.execute("DELETE FROM reference_frames WHERE id=%s", (ref_id,))
        c.commit()
    return {"public_id": row[0]}


# ---------- Cloudinary ----------
def _cloudinary():
    import cloudinary  # type: ignore

    cloudinary.config(
        cloud_name=_env("CLOUDINARY_CLOUD_NAME"),
        api_key=_env("CLOUDINARY_API_KEY"),
        api_secret=_env("CLOUDINARY_API_SECRET"),
        secure=True,
    )
    return cloudinary


def upload_image(data: bytes, filename: str, folder: str | None = None) -> dict:
    """Upload bytes to Cloudinary, return {public_id, secure_url, width, height}."""
    if not cloudinary_configured():
        raise RuntimeError(
            "Cloudinary is not configured "
            "(CLOUDINARY_CLOUD_NAME/API_KEY/API_SECRET)."
        )
    import cloudinary.uploader  # type: ignore

    _cloudinary()
    target = folder or _env("CLOUDINARY_FOLDER", "lunarsync-refs") or "lunarsync-refs"
    # Temp-file upload (robust for all formats/sizes; data-URI posts are flaky).
    import tempfile

    suffix = Path(filename).suffix or ".png"
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    try:
        tmp.write(data)
        tmp.close()
        res = cloudinary.uploader.upload(
            tmp.name,
            folder=target,
            public_id=f"{Path(filename).stem}-{uuid.uuid4().hex[:6]}",
            resource_type="image",
            overwrite=False,
        )
    finally:
        try:
            Path(tmp.name).unlink(missing_ok=True)
        except Exception:
            pass
    return {"public_id": res.get("public_id", ""),
            "secure_url": res.get("secure_url", ""),
            "width": res.get("width"), "height": res.get("height")}


def destroy_image(public_id: str) -> None:
    try:
        import cloudinary.uploader  # type: ignore

        _cloudinary()
        cloudinary.uploader.destroy(public_id)
    except Exception as exc:
        logger.warning("Cloudinary destroy failed for %s: %s", public_id, exc)


# ---------- materialize cloud set to local files for the sweep ----------
def _safe_name(public_id: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", public_id) + ".jpg"


def materialize(refs: list[dict]) -> list[Path]:
    """Download seeded Cloudinary URLs to a local cache dir; return Paths.

    Cache hit avoids re-download. Same gray-decode path as HF frames, so
    hf_matcher.sweep() runs identical logic over this smaller set.
    """
    import urllib.request

    paths: list[Path] = []
    for r in refs:
        url = r.get("secureUrl") or ""
        if not url:
            continue
        dest = CLOUD_CACHE / _safe_name(r.get("publicId") or r.get("id", "ref"))
        if not dest.exists() or dest.stat().st_size == 0:
            req = urllib.request.Request(url, headers={"User-Agent": "LunaMatch/1.0"})
            with urllib.request.urlopen(req, timeout=30) as resp, open(dest, "wb") as f:
                f.write(resp.read())
        # Name the Path after the original filename so metadata parsing works.
        alias = CLOUD_CACHE / (r.get("filename") or dest.name)
        if alias != dest and not alias.exists():
            try:
                alias.write_bytes(dest.read_bytes())
            except Exception:
                alias = dest
        paths.append(alias if alias.exists() else dest)
    return sorted(paths)


def status() -> dict:
    s: dict = {"mode": mode(), "database_configured": bool(db_url()),
               "cloudinary_configured": cloudinary_configured()}
    if db_url():
        try:
            s["seeded_count"] = len(list_references())
        except Exception as exc:
            s["error"] = str(exc)[:200]
    else:
        s["seeded_count"] = 0
    return s
