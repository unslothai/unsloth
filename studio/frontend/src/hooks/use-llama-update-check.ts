// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { authFetch, getAuthToken } from "@/features/auth";
import { useCallback, useEffect, useRef, useState } from "react";

// First check shortly after load, then re-surface as an hourly reminder. The
// banner stays up until the user dismisses it (click outside / X) or updates.
const FIRST_CHECK_DELAY_MS = 1000;
const REMINDER_INTERVAL_MS = 60 * 60 * 1000; // ~1 hour
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

async function fetchStatus(
  forceRefresh = false,
): Promise<LlamaUpdateStatus | null> {
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
 * becomes true ~1s after load and stays up until the user dismisses it (click
 * outside / X) or updates; it re-surfaces every ~hour as a reminder. `apply()`
 * triggers the in-place swap and tracks the job.
 */
export function useLlamaUpdateCheck({
  enabled = true,
}: UseLlamaUpdateCheckOptions = {}) {
  const [status, setStatus] = useState<LlamaUpdateStatus | null>(null);
  const [visible, setVisible] = useState(false);
  const [applying, setApplying] = useState(false);
  const pollTimer = useRef<ReturnType<typeof setInterval> | null>(null);

  const clearPollTimer = useCallback(() => {
    if (pollTimer.current) {
      clearInterval(pollTimer.current);
      pollTimer.current = null;
    }
  }, []);

  // Poll the job to completion. Shared by apply() and surfaceIfAvailable() so a
  // job is tracked once whoever noticed it; onDone resolves with the result.
  const startJobPoll = useCallback(
    (onDone?: (result: LlamaApplyResult) => void) => {
      clearPollTimer();
      pollTimer.current = setInterval(async () => {
        const s = await fetchStatus();
        if (!s) return;
        setStatus(s);
        if (s.job.state === "running") return;
        clearPollTimer();
        setApplying(false);
        if (s.job.state === "success") {
          setVisible(false);
          onDone?.({ ok: true, tag: s.job.to_tag });
        } else if (s.job.state === "error") {
          // Leave the banner up so the user can retry; clearing applying drops
          // the "Updating..." state.
          onDone?.({ ok: false, error: s.job.error });
        } else {
          // idle without a terminal result (job reset): stop tracking.
          onDone?.({ ok: false, error: "update did not complete" });
        }
      }, JOB_POLL_INTERVAL_MS);
    },
    [clearPollTimer],
  );

  // Surface the banner when an update is available; it stays up until dismissed.
  const surfaceIfAvailable = useCallback(
    (next: LlamaUpdateStatus | null) => {
      if (!next) return;
      setStatus(next);
      if (next.job.state === "running") {
        // Swap in progress (e.g. another tab): keep the banner up and track the
        // job so "Updating..." clears when it finishes instead of sticking.
        setApplying(true);
        setVisible(true);
        if (!pollTimer.current) startJobPoll();
        return;
      }
      if (next.update_available) {
        setVisible(true);
      }
    },
    [startJobPoll],
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
      clearPollTimer();
    };
  }, [enabled, surfaceIfAvailable, clearPollTimer]);

  const dismiss = useCallback(() => {
    setVisible(false);
  }, []);

  const apply = useCallback(async (): Promise<LlamaApplyResult> => {
    if (applying) return { ok: false, error: "already running" };
    setApplying(true);
    setVisible(true);
    let action: {
      started?: boolean;
      reason?: string | null;
      message?: string | null;
    } | null = null;
    try {
      const res = await authFetch("/api/llama/update", { method: "POST" });
      if (!res.ok) {
        setApplying(false);
        return { ok: false, error: `HTTP ${res.status}` };
      }
      try {
        action = await res.json();
      } catch {
        action = null;
      }
    } catch (e) {
      setApplying(false);
      return { ok: false, error: String(e) };
    }

    // 200 without a started job (no marker / installer missing) leaves it idle,
    // so surface the reason instead of polling forever. already_running is the
    // exception: a job is in flight, so track it to completion below.
    if (
      action &&
      action.started === false &&
      action.reason !== "already_running"
    ) {
      setApplying(false);
      return {
        ok: false,
        error: action.message ?? action.reason ?? "update was not started",
      };
    }

    return await new Promise<LlamaApplyResult>((resolve) =>
      startJobPoll(resolve),
    );
  }, [applying, startJobPoll]);

  return {
    status: enabled ? status : null,
    visible: enabled && visible,
    applying,
    apply,
    dismiss,
  };
}
