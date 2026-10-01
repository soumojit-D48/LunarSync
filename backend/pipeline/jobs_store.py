"""Neon-backed job persistence — jobs survive backend restarts.

Free-tier instances lose memory + local disk on every sleep/restart, which
orphaned live jobs (permanent 404s). Jobs + results now live in Neon as JSON;
result images are pushed to Cloudinary so they survive too. Local disk stays
as a fast-path cache only.
"""

import json
import logging

logger = logging.getLogger("LunaMatch.jobs_store")


def _db():
    from pipeline import cloud_store as cs

    return cs


def available() -> bool:
    try:
        return bool(_db().db_url())
    except Exception:
        return False


def ensure_schema() -> None:
    cs = _db()
    if not cs.db_url():
        return
    with cs._conn() as c, c.cursor() as cur:
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS jobs (
              id TEXT PRIMARY KEY,
              data JSONB NOT NULL,
              result JSONB,
              updated_at TIMESTAMPTZ DEFAULT now()
            )
            """
        )
        c.commit()


def _to_jsonable(obj):
    if isinstance(obj, dict):
        return {k: _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(v) for v in obj]
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    if isinstance(obj, (bytes, bytearray)):
        return None  # raw blobs never belong in the jobs table
    return str(obj)


def save_job(job: dict) -> None:
    cs = _db()
    if not cs.db_url():
        return
    ensure_schema()
    with cs._conn() as c, c.cursor() as cur:
        cur.execute(
            "INSERT INTO jobs (id, data, updated_at) VALUES (%s,%s,now())"
            " ON CONFLICT (id) DO UPDATE SET data=EXCLUDED.data, updated_at=now()",
            (job["id"], json.dumps(_to_jsonable(job))),
        )
        c.commit()


def save_result(job_id: str, result: dict) -> None:
    cs = _db()
    if not cs.db_url():
        return
    ensure_schema()
    with cs._conn() as c, c.cursor() as cur:
        cur.execute(
            "UPDATE jobs SET result=%s, updated_at=now() WHERE id=%s",
            (json.dumps(_to_jsonable(result)), job_id),
        )
        c.commit()


def load_job(job_id: str) -> dict | None:
    cs = _db()
    if not cs.db_url():
        return None
    ensure_schema()
    with cs._conn() as c, c.cursor() as cur:
        cur.execute("SELECT data FROM jobs WHERE id=%s", (job_id,))
        row = cur.fetchone()
    return dict(row[0]) if row else None


def load_result(job_id: str) -> dict | None:
    cs = _db()
    if not cs.db_url():
        return None
    ensure_schema()
    with cs._conn() as c, c.cursor() as cur:
        cur.execute("SELECT result FROM jobs WHERE id=%s", (job_id,))
        row = cur.fetchone()
    if not row or row[0] is None:
        return None
    return dict(row[0]) if isinstance(row[0], dict) else json.loads(row[0])


def list_jobs(limit: int = 100) -> list[dict]:
    cs = _db()
    if not cs.db_url():
        return []
    ensure_schema()
    with cs._conn() as c, c.cursor() as cur:
        cur.execute("SELECT data FROM jobs ORDER BY updated_at DESC LIMIT %s", (limit,))
        return [dict(r[0]) for r in cur.fetchall()]


def upload_result_images(job_id: str, images: dict[str, bytes]) -> dict[str, str]:
    """Push result JPGs to Cloudinary; return {name: secure_url}."""
    cs = _db()
    out: dict[str, str] = {}
    for name, blob in images.items():
        try:
            up = cs.upload_image(blob, f"{job_id}-{name}.jpg",
                                 folder="lunarsync-results")
            out[name] = up["secure_url"]
        except Exception as exc:
            logger.warning("Result image upload failed (%s): %s", name, exc)
    return out
