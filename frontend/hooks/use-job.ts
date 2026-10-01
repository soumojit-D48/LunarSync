"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { getJob, getResult, type Job, type JobResult } from "@/lib/api-client";

/** Polls GET /jobs/{id} every 2s until the job reaches a terminal state. */
export function useJob(id: string) {
  const [job, setJob] = useState<Job | null>(null);
  const [result, setResult] = useState<JobResult | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [reconnecting, setReconnecting] = useState(false);
  const timer = useRef<ReturnType<typeof setInterval> | null>(null);
  const fails = useRef(0);

  const stop = useCallback(() => {
    if (timer.current) clearInterval(timer.current);
    timer.current = null;
  }, []);

  useEffect(() => {
    let cancelled = false;
    const tick = async () => {
      try {
        const { job: next } = await getJob(id);
        if (cancelled) return;
        fails.current = 0;
        setReconnecting(false);
        setJob(next);
        if (next.status === "SUCCEEDED") {
          stop();
          const full = await getResult(id);
          if (!cancelled) setResult(full);
        } else if (next.status === "FAILED") {
          stop();
          // Best-effort evidence (ranked sweep + top-match viewer) is stored
          // even on FAILED jobs — show the comparison, not just the verdict.
          try {
            const full = await getResult(id);
            if (!cancelled) setResult(full);
          } catch {
            /* no evidence stored for this failure */
          }
        }
      } catch (e) {
        if (cancelled) return;
        const msg = e instanceof Error ? e.message : "request failed";
        // 404 on a live-* job = backend restarted (free-tier sleep wipes the
        // in-memory + on-disk job store; Neon holds only references). Gone for good.
        if (/404/.test(msg)) {
          setError(
            id.startsWith("live-")
              ? "Job expired — the free-tier backend restarted and forgot it (seeds are safe). Please re-upload."
              : msg,
          );
          stop();
          return;
        }
        // Transient (backend waking up) — keep polling, don't alarm yet.
        fails.current += 1;
        if (fails.current >= 3) setReconnecting(true);
        if (fails.current >= 45) {
          setError("Backend unreachable for ~90s — it may be asleep. Refresh to retry.");
          stop();
        }
      }
    };
    tick();
    timer.current = setInterval(tick, 2000);
    return () => {
      cancelled = true;
      stop();
    };
  }, [id, stop]);

  return { job, result, error, reconnecting };
}
