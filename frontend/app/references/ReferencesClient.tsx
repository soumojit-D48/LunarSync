"use client";

import Link from "next/link";
import { useCallback, useEffect, useState } from "react";
import { ArrowRight, ImagePlus, Trash2 } from "lucide-react";
import { ConsoleNav } from "@/components/ConsoleNav";
import {
  deleteReference,
  getBackendMode,
  listReferences,
  uploadReference,
  type CloudReference,
} from "@/lib/api-client";

const inputCls =
  "w-full rounded-md bg-void/60 px-3 py-2.5 font-mono text-xs text-bone ring-1 ring-line outline-none transition-colors focus:ring-signal";

export default function ReferencesPage() {
  const [refs, setRefs] = useState<CloudReference[] | null>(null);
  const [mode, setMode] = useState<string>("…");
  const [backendDown, setBackendDown] = useState(false);
  const [file, setFile] = useState<File | null>(null);
  const [lat, setLat] = useState("");
  const [lon, setLon] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const refresh = useCallback(async () => {
    try {
      const [r, m] = await Promise.all([listReferences(), getBackendMode()]);
      setRefs(r.references ?? []);
      setMode(r.warning ? "waking…" : (m.mode ?? "unknown"));
      setBackendDown(Boolean(r.warning));
    } catch (e) {
      setError(e instanceof Error ? e.message : "failed to load");
    }
  }, []);

  useEffect(() => {
    refresh();
  }, [refresh]);

  const submit = async () => {
    if (!file) return;
    setBusy(true);
    setError(null);
    try {
      await uploadReference({ file, lat: lat || undefined, lon: lon || undefined });
      setFile(null);
      setLat("");
      setLon("");
      await refresh();
    } catch (e) {
      setError(e instanceof Error ? e.message : "upload failed");
    } finally {
      setBusy(false);
    }
  };

  const remove = async (id: string) => {
    setError(null);
    try {
      await deleteReference(id);
      await refresh();
    } catch (e) {
      setError(e instanceof Error ? e.message : "delete failed");
    }
  };

  return (
    <div className="relative min-h-screen bg-void text-bone">
      <ConsoleNav />
      <div className="mx-auto max-w-7xl px-5 py-10 sm:px-8">
        <p className="font-mono text-[11px] tracking-[0.2em] text-signal">CONSOLE — REFERENCES</p>
        <h1 className="mt-2 text-3xl font-semibold tracking-tight sm:text-4xl">Cloud reference set</h1>
        <p className="mt-2 max-w-[70ch] text-sm text-mist">
          Seed photos here — each upload goes to Cloudinary and its link is stored in Neon Postgres
          automatically. When <span className="font-mono text-bone">REFERENCE_BACKEND=cloud</span>,
          every query on <Link href="/jobs" className="text-signal hover:underline">/jobs</Link> is
          compared <em>only</em> against this set with the same SIFT + outlier-rejection logic as the
          HF archive. Current backend mode:{" "}
          <span className="font-mono text-signal">{mode}</span>
          {mode !== "cloud" && !backendDown ? " (uploads still work; switch the backend env to use them for matching)" : null}
        </p>
        {backendDown ? (
          <p className="mt-3 rounded-md px-3 py-2 font-mono text-[11px] tracking-[0.08em] text-amber-300 ring-1 ring-amber-300/30">
            BACKEND UNREACHABLE — Render free tier sleeps after 15 min idle. Wait ~1 min and refresh;
            your seeded photos are safe in Neon/Cloudinary.
          </p>
        ) : null}

        <div className="mt-8 grid gap-6 lg:grid-cols-[380px_1fr]">
          <div className="panel-glass h-fit rounded-xl p-5 ring-1 ring-white/10">
            <div className="mb-4 flex items-center gap-2">
              <ImagePlus className="size-4 text-signal" />
              <h2 className="font-mono text-xs tracking-[0.18em] text-bone">SEED REFERENCE</h2>
            </div>
            <label className="block cursor-pointer rounded-lg border border-dashed border-line p-6 text-center transition-colors hover:border-signal/60 hover:bg-signal/5">
              <span className="mb-1 block font-mono text-[10px] tracking-[0.16em] text-ash">REFERENCE PHOTO</span>
              <span className="block truncate font-mono text-xs text-bone">
                {file?.name ?? "Click to browse · PNG · JPG · TIF"}
              </span>
              <input
                type="file"
                accept=".png,.jpg,.jpeg,.tif,.tiff,.bmp"
                className="hidden"
                onChange={(e) => setFile(e.target.files?.[0] ?? null)}
              />
            </label>
            <div className="mt-4 grid grid-cols-2 gap-3">
              <label className="block">
                <span className="mb-1.5 block font-mono text-[10px] tracking-[0.16em] text-ash">LAT (OPT)</span>
                <input value={lat} onChange={(e) => setLat(e.target.value)} placeholder="e.g. -12.5" className={inputCls} />
              </label>
              <label className="block">
                <span className="mb-1.5 block font-mono text-[10px] tracking-[0.16em] text-ash">LON (OPT)</span>
                <input value={lon} onChange={(e) => setLon(e.target.value)} placeholder="e.g. 34.1" className={inputCls} />
              </label>
            </div>
            <button
              type="button"
              onClick={submit}
              disabled={busy || !file}
              className="group mt-5 inline-flex items-center gap-2 rounded-md bg-signal px-5 py-3 font-mono text-sm font-semibold text-void ring-1 ring-signal/40 transition-colors hover:bg-bone disabled:opacity-50"
            >
              {busy ? "UPLOADING…" : "Seed reference"}
              <ArrowRight className="size-4 transition-transform group-hover:translate-x-0.5" />
            </button>
            {error ? <p className="mt-3 font-mono text-[11px] leading-relaxed text-destructive">{error}</p> : null}
          </div>

          <div className="overflow-hidden rounded-xl ring-1 ring-white/10">
            <div className="border-b border-line/60 bg-surface/60 px-4 py-3 font-mono text-[10px] tracking-[0.16em] text-ash">
              SEEDED SET{refs ? ` · ${refs.length}` : ""}
            </div>
            {!refs ? (
              <p className="px-4 py-8 font-mono text-xs text-mist">Loading references…</p>
            ) : refs.length === 0 ? (
              <p className="px-4 py-8 font-mono text-xs text-mist">
                Empty — seed your first reference photo. Queries in cloud mode will fail until at least one exists.
              </p>
            ) : (
              <ul className="grid gap-3 p-4 sm:grid-cols-2 xl:grid-cols-3">
                {refs.map((r) => (
                  <li key={r.id} className="overflow-hidden rounded-lg ring-1 ring-line/60">
                    {/* eslint-disable-next-line @next/next/no-img-element */}
                    <img src={r.secureUrl} alt={r.filename} className="aspect-video w-full object-cover" />
                    <div className="flex items-start justify-between gap-2 p-3">
                      <div className="min-w-0">
                        <p className="truncate font-mono text-[11px] text-bone">{r.filename}</p>
                        <p className="mt-0.5 font-mono text-[10px] text-ash">
                          {r.lat ?? "?"}° · {r.lon ?? "?"}° · {r.id}
                        </p>
                      </div>
                      <button
                        type="button"
                        onClick={() => remove(r.id)}
                        className="rounded-md p-1.5 text-ash transition-colors hover:bg-destructive/10 hover:text-destructive"
                        title="Delete reference"
                      >
                        <Trash2 className="size-4" />
                      </button>
                    </div>
                  </li>
                ))}
              </ul>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
