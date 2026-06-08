// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { authFetch, getAuthToken } from "@/features/auth";
import { useCallback, useEffect, useRef, useState } from "react";

// First check shortly after load, then re-surface as an hourly reminder.
const FIRST_CHECK_DELAY_MS = 8000;
const REMINDER_INTERVAL_MS = 60 * 60 * 1000; // ~1 hour
// The banner fades on its own after this long (non-invasive). It re-appears on
// the next hourly reminder while an update is still available.
const AUTO_HIDE_MS = 10000;
// While an update is applying, poll the job state at this cadence.
const JOB_POLL_INTERVAL_MS = 3000;

export interface LlamaUpdateJob {
  state: "idle" | "running" | "success" | "error";
  message: string;
  from_tag: string | null;
  to_tag: string | null;
  error: string | null;
}

export interface LlamaUpdateStatus {
  supported: boolean;
  update_available: boolean;
  installed_tag: string | null;
  latest_tag: string | null;
  job: LlamaUpdateJob;
}

function parseStatus(value: unknown): LlamaUpdateStatus | null {
  if (!value || typeof value !== "object") return null;
  const s = value as Record<string, unknown>;
  const job = (s.job ?? {}) as Record<string, unknown>;
  return {
    supported: s.supported === true,
    update_available: s.update_available === true,
    installed_tag: typeof s.installed_tag === "string" ? s.installed_tag : null,
    latest_tag: typeof s.latest_tag === "string" ? s.latest_tag : null,
    job: {
      state: (job.state as LlamaUpdateJob["state"]) ?? "idle",
      message: typeof job.message === "string" ? job.message : "",
      from_tag: typeof job.from_tag === "string" ? job.from_tag : null,
      to_tag: typeof job.to_tag === "string" ? job.to_tag : null,
      error: typeof job.error === "string" ? job.error : null,
    },
  };
}

async function fetchStatus(forceRefresh = false): Promise<LlamaUpdateStatus | null> {
  if (!getAuthToken()) return null;
  try {
    const res = await authFetch(
      `/api/llama/update-status${forceRefresh ? "?force_refresh=true" : ""}`,
    );
    if (!res.ok) return null;
    return parseStatus(await res.json());
  } catch {
    return null;
  }
}

interface UseLlamaUpdateCheckOptions {
  enabled?: boolean;
}

export interface LlamaApplyResult {
  ok: boolean;
  tag?: string | null;
  error?: string | null;
}

/**
 * Polls the backend for a newer llama.cpp prebuilt. When one exists, `visible`
 * is true for a 10s window (fades on its own), and re-surfaces every ~hour as a
 * gentle reminder. `apply()` triggers the in-place swap and tracks the job.
 */
export function useLlamaUpdateCheck({ enabled = true }: UseLlamaUpdateCheckOptions = {}) {
  const [status, setStatus] = useState<LlamaUpdateStatus | null>(null);
  const [visible, setVisible] = useState(false);
  const [applying, setApplying] = useState(false);
  const hideTimer = useRef<ReturnType<typeof setTimeout> | null>(null);
  const pollTimer = useRef<ReturnType<typeof setInterval> | null>(null);

  const clearHideTimer = useCallback(() => {
    if (hideTimer.current) {
      clearTimeout(hideTimer.current);
      hideTimer.current = null;
    }
  }, []);

  const armAutoHide = useCallback(() => {
    clearHideTimer();
    hideTimer.current = setTimeout(() => setVisible(false), AUTO_HIDE_MS);
  }, [clearHideTimer]);

  // Surface the banner for the auto-hide window when an update is available.
  const surfaceIfAvailable = useCallback(
    (next: LlamaUpdateStatus | null) => {
      if (!next) return;
      setStatus(next);
      if (next.job.state === "running") {
        // A swap is in progress (e.g. started in another tab) -- keep it up.
        setApplying(true);
        setVisible(true);
        clearHideTimer();
        return;
      }
      if (next.update_available) {
        setVisible(true);
        armAutoHide();
      }
    },
    [armAutoHide, clearHideTimer],
  );

  useEffect(() => {
    if (!enabled) return;
    let canceled = false;

    const firstTimer = setTimeout(() => {
      fetchStatus().then((s) => {
        if (!canceled) surfaceIfAvailable(s);
      });
    }, FIRST_CHECK_DELAY_MS);

    const reminder = setInterval(() => {
      fetchStatus().then((s) => {
        if (!canceled) surfaceIfAvailable(s);
      });
    }, REMINDER_INTERVAL_MS);

    return () => {
      canceled = true;
      clearTimeout(firstTimer);
      clearInterval(reminder);
      clearHideTimer();
      if (pollTimer.current) clearInterval(pollTimer.current);
    };
  }, [enabled, surfaceIfAvailable, clearHideTimer]);

  const dismiss = useCallback(() => {
    clearHideTimer();
    setVisible(false);
  }, [clearHideTimer]);

  const apply = useCallback(async (): Promise<LlamaApplyResult> => {
    if (applying) return { ok: false, error: "already running" };
    setApplying(true);
    setVisible(true);
    clearHideTimer();
    try {
      const res = await authFetch("/api/llama/update", { method: "POST" });
      if (!res.ok) {
        setApplying(false);
        armAutoHide();
        return { ok: false, error: `HTTP ${res.status}` };
      }
    } catch (e) {
      setApplying(false);
      armAutoHide();
      return { ok: false, error: String(e) };
    }

    // Poll the job to completion.
    return await new Promise<LlamaApplyResult>(
      (resolve) => {
        if (pollTimer.current) clearInterval(pollTimer.current);
        pollTimer.current = setInterval(async () => {
          const s = await fetchStatus();
          if (!s) return;
          setStatus(s);
          if (s.job.state === "success") {
            if (pollTimer.current) clearInterval(pollTimer.current);
            setApplying(false);
            setVisible(false);
            resolve({ ok: true, tag: s.job.to_tag });
          } else if (s.job.state === "error") {
            if (pollTimer.current) clearInterval(pollTimer.current);
            setApplying(false);
            armAutoHide();
            resolve({ ok: false, error: s.job.error });
          }
        }, JOB_POLL_INTERVAL_MS);
      },
    );
  }, [applying, armAutoHide, clearHideTimer]);

  return {
    status: enabled ? status : null,
    visible: enabled && visible,
    applying,
    apply,
    dismiss,
  };
}
