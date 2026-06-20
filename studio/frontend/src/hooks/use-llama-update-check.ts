// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { authFetch, getAuthToken } from "@/features/auth";
import { refreshHardwareInfo } from "@/hooks/use-hardware-info";
import { useCallback, useEffect, useRef, useState } from "react";

// First check shortly after load, then re-surface as an hourly reminder. The
// banner stays up until the user explicitly acts on it (X, Update, or
// Remind me later).
const FIRST_CHECK_DELAY_MS = 1000;
const REMINDER_INTERVAL_MS = 60 * 60 * 1000; // ~1 hour
// "Remind me later" re-surfaces sooner than the hourly reminder.
const SNOOZE_DELAY_MS = 15 * 60 * 1000; // ~15 minutes
// Poll cadence while applying. Short so the installer's ~5% milestones are
// observed instead of a fast download finishing between two slow polls.
const JOB_POLL_INTERVAL_MS = 500;

export interface LlamaUpdateJob {
  state: "idle" | "running" | "success" | "error";
  message: string;
  from_tag: string | null;
  to_tag: string | null;
  error: string | null;
  // Download fraction (0..1) while running, 1 on success, null when unknown.
  progress: number | null;
}

export interface LlamaUpdateStatus {
  supported: boolean;
  update_available: boolean;
  installed_tag: string | null;
  latest_tag: string | null;
  // Download size of the prebuilt Update would fetch, in bytes (null if unknown).
  update_size_bytes: number | null;
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
    update_size_bytes:
      typeof s.update_size_bytes === "number" ? s.update_size_bytes : null,
    job: {
      state: (job.state as LlamaUpdateJob["state"]) ?? "idle",
      message: typeof job.message === "string" ? job.message : "",
      from_tag: typeof job.from_tag === "string" ? job.from_tag : null,
      to_tag: typeof job.to_tag === "string" ? job.to_tag : null,
      error: typeof job.error === "string" ? job.error : null,
      progress: typeof job.progress === "number" ? job.progress : null,
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

// Update probes force a refresh so a newly published build is not masked by the
// backend's 24h release cache (the banner would otherwise lag up to a day). The
// job-progress poll below stays cached; it only reads local job state.
const recheckStatus = () => fetchStatus(true);

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
 * becomes true ~1s after load and stays up until the user dismisses it (X),
 * snoozes it ("Remind me later", ~15 min), or updates; it re-surfaces every
 * ~hour as a reminder. `apply()` triggers the in-place swap and tracks the
 * job.
 */
export function useLlamaUpdateCheck({
  enabled = true,
}: UseLlamaUpdateCheckOptions = {}) {
  const [status, setStatus] = useState<LlamaUpdateStatus | null>(null);
  const [visible, setVisible] = useState(false);
  const [applying, setApplying] = useState(false);
  const pollTimer = useRef<ReturnType<typeof setInterval> | null>(null);
  const snoozeTimer = useRef<ReturnType<typeof setTimeout> | null>(null);

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
          void refreshHardwareInfo();
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
    if (!enabled) {
      // Disabled mid-update: stop showing and tracking, and clear `applying`
      // so the banner's animation loop stops too. Re-enabling re-detects a
      // still-running job below and resumes tracking via surfaceIfAvailable.
      setVisible(false);
      setApplying(false);
      return;
    }
    let canceled = false;

    const firstTimer = setTimeout(() => {
      recheckStatus().then((s) => {
        if (!canceled) surfaceIfAvailable(s);
      });
    }, FIRST_CHECK_DELAY_MS);

    const reminder = setInterval(() => {
      recheckStatus().then((s) => {
        if (!canceled) surfaceIfAvailable(s);
      });
    }, REMINDER_INTERVAL_MS);

    return () => {
      canceled = true;
      clearTimeout(firstTimer);
      clearInterval(reminder);
      clearPollTimer();
      if (snoozeTimer.current) {
        clearTimeout(snoozeTimer.current);
        snoozeTimer.current = null;
      }
    };
  }, [enabled, surfaceIfAvailable, clearPollTimer]);

  const dismiss = useCallback(() => {
    setVisible(false);
  }, []);

  // Hide now, re-check and re-surface after SNOOZE_DELAY_MS.
  const snooze = useCallback(() => {
    setVisible(false);
    if (snoozeTimer.current) clearTimeout(snoozeTimer.current);
    snoozeTimer.current = setTimeout(() => {
      snoozeTimer.current = null;
      recheckStatus().then(surfaceIfAvailable);
    }, SNOOZE_DELAY_MS);
  }, [surfaceIfAvailable]);

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
    snooze,
  };
}
