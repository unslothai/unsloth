// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { authFetch, getAuthToken } from "@/features/auth";
import { refreshHardwareInfo } from "@/hooks/use-hardware-info";
import { useCallback, useEffect, useRef, useState } from "react";

// Initial check plus hourly reminders until dismissed or applied.
const FIRST_CHECK_DELAY_MS = 1000;
const REMINDER_INTERVAL_MS = 60 * 60 * 1000; // ~1 hour
// Snooze checks sooner than the hourly reminder.
const SNOOZE_DELAY_MS = 15 * 60 * 1000; // ~15 minutes
// Poll fast enough to catch installer progress milestones.
const JOB_POLL_INTERVAL_MS = 500;

export interface LlamaUpdateJob {
  state: "idle" | "running" | "success" | "error";
  message: string;
  from_tag: string | null;
  to_tag: string | null;
  reload_required: boolean | null;
  error: string | null;
  // Download fraction while running, 1 on success.
  progress: number | null;
  // Set once the job leaves "running"; identifies a completed job so a
  // repeated fetch of the same success can be told apart from the next one.
  finished_at: string | null;
}

export interface LlamaUpdateStatus {
  supported: boolean;
  update_available: boolean;
  installed_tag: string | null;
  latest_tag: string | null;
  // Prebuilt download size in bytes, if known.
  update_size_bytes: number | null;
  job: LlamaUpdateJob;
}

function parseJob(value: unknown): LlamaUpdateJob {
  const job = (value ?? {}) as Record<string, unknown>;
  return {
    state: (job.state as LlamaUpdateJob["state"]) ?? "idle",
    message: typeof job.message === "string" ? job.message : "",
    from_tag: typeof job.from_tag === "string" ? job.from_tag : null,
    to_tag: typeof job.to_tag === "string" ? job.to_tag : null,
    reload_required:
      typeof job.reload_required === "boolean" ? job.reload_required : null,
    error: typeof job.error === "string" ? job.error : null,
    progress: typeof job.progress === "number" ? job.progress : null,
    finished_at: typeof job.finished_at === "string" ? job.finished_at : null,
  };
}

function parseStatus(value: unknown): LlamaUpdateStatus | null {
  if (!value || typeof value !== "object") return null;
  const s = value as Record<string, unknown>;
  return {
    supported: s.supported === true,
    update_available: s.update_available === true,
    installed_tag: typeof s.installed_tag === "string" ? s.installed_tag : null,
    latest_tag: typeof s.latest_tag === "string" ? s.latest_tag : null,
    update_size_bytes:
      typeof s.update_size_bytes === "number" ? s.update_size_bytes : null,
    job: parseJob(s.job),
  };
}

// The backend job persists as "success" until the next update starts (it's a
// single in-memory record, not per-tab), so a fresh mount -- a new tab, or a
// page reload of a tab that already resynced -- would otherwise replay the
// same completed job forever. Persist the handled marker outside React state
// so it survives both, and is shared across tabs in this browser.
const HANDLED_RELOAD_STORAGE_KEY = "unsloth_llama_update_reload_handled_at";

function getHandledReloadAt(): string | null {
  try {
    return localStorage.getItem(HANDLED_RELOAD_STORAGE_KEY);
  } catch {
    return null;
  }
}

function setHandledReloadAt(finishedAt: string | null): void {
  if (!finishedAt) return;
  try {
    localStorage.setItem(HANDLED_RELOAD_STORAGE_KEY, finishedAt);
  } catch {
    // storage unavailable
  }
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

// Manual checks bypass the 24h release cache; job polls read local state.
const recheckStatus = () => fetchStatus(true);

interface UseLlamaUpdateCheckOptions {
  enabled?: boolean;
  /**
   * Called when a completed update reports `reload_required` (i.e. it unloaded
   * the active model server-side). Consumers use it to resync the chat runtime
   * so the model selector drops to "select model" instead of pointing at a
   * model that now 400s on send. Fires for both this tab's own apply() and a
   * cross-tab update mirrored through the background poll.
   */
  onReloadRequired?: () => void;
}

export interface LlamaMachine {
  hostname: string;
  platform: string;
}

/** The pending swap a confirmation prompt describes: the exact build (from ->
 *  to), the host it targets, and the single-use token that applies it. */
export interface LlamaApplyTarget {
  token: string;
  machine: LlamaMachine | null;
  fromTag: string | null;
  toTag: string | null;
}

export interface LlamaApplyResult {
  ok: boolean;
  // Post-swap build tag (the "to" version).
  tag?: string | null;
  // Build the swap replaced (the "from" version).
  fromTag?: string | null;
  // Host the swap ran on, so the result can name which machine changed.
  machine?: LlamaMachine | null;
  reloadRequired?: boolean | null;
  error?: string | null;
}

function parseMachine(value: unknown): LlamaMachine | null {
  if (!value || typeof value !== "object") return null;
  const m = value as Record<string, unknown>;
  return {
    hostname: typeof m.hostname === "string" ? m.hostname : "",
    platform: typeof m.platform === "string" ? m.platform : "",
  };
}

/** Tracks llama.cpp update visibility and apply progress. */
export function useLlamaUpdateCheck({
  enabled = true,
  onReloadRequired,
}: UseLlamaUpdateCheckOptions = {}) {
  const [status, setStatus] = useState<LlamaUpdateStatus | null>(null);
  const [visible, setVisible] = useState(false);
  const [applying, setApplying] = useState(false);
  const pollTimer = useRef<ReturnType<typeof setInterval> | null>(null);
  const snoozeTimer = useRef<ReturnType<typeof setTimeout> | null>(null);
  // Read through a ref so startJobPoll stays stable (apply/surfaceIfAvailable
  // depend on it) while still calling the latest callback.
  const onReloadRequiredRef = useRef(onReloadRequired);
  useEffect(() => {
    onReloadRequiredRef.current = onReloadRequired;
  }, [onReloadRequired]);
  // Fires the callback once per completed job, whether this tab watched it run
  // or only saw the persisted "success" after the fact (e.g. another tab
  // applied it). Keyed by finished_at and seeded from localStorage so a fresh
  // mount (new tab, or a page reload of a tab that already resynced) doesn't
  // replay a job some tab already handled.
  const reloadNotifiedForRef = useRef<string | null>(getHandledReloadAt());

  const clearPollTimer = useCallback(() => {
    if (pollTimer.current) {
      clearInterval(pollTimer.current);
      pollTimer.current = null;
    }
  }, []);

  // Shared by the poll path (this tab watched the job run), the surface path
  // (this tab only saw the persisted success), and apply()'s stale-click path
  // (the job came back embedded in a "not started" response) so none of them
  // can drop or double-fire the notification.
  const notifyReloadIfNeeded = useCallback(
    (job: Pick<LlamaUpdateJob, "state" | "reload_required" | "finished_at">) => {
      if (
        job.state === "success" &&
        job.reload_required &&
        job.finished_at !== reloadNotifiedForRef.current
      ) {
        reloadNotifiedForRef.current = job.finished_at;
        setHandledReloadAt(job.finished_at);
        onReloadRequiredRef.current?.();
      }
    },
    [],
  );

  // Used by apply() and another-tab job tracking.
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
          // The update unloads the running model server-side, so the chat
          // runtime still points at a model that now 400s on send. Let the
          // consumer drop the selector to "select model" instead of waiting for
          // a page reload. Fires here (not just from apply's onDone) so a
          // cross-tab update mirrored through this poll is covered too.
          notifyReloadIfNeeded(s.job);
          onDone?.({
            ok: true,
            tag: s.job.to_tag,
            reloadRequired: s.job.reload_required,
          });
        } else if (s.job.state === "error") {
          // Keep the banner visible so retry is available.
          onDone?.({ ok: false, error: s.job.error });
        } else {
          onDone?.({ ok: false, error: "update did not complete" });
        }
      }, JOB_POLL_INTERVAL_MS);
    },
    [clearPollTimer, notifyReloadIfNeeded],
  );

  const surfaceIfAvailable = useCallback(
    (next: LlamaUpdateStatus | null) => {
      if (!next) return;
      setStatus(next);
      if (next.job.state === "running") {
        // Another tab is applying; show progress here too.
        setApplying(true);
        setVisible(true);
        if (!pollTimer.current) startJobPoll();
        return;
      }
      // A completed job persists as "success" until the next update starts, so
      // a tab that missed the running window entirely (mounted, or only checks
      // hourly and misses both the running and just-finished moments) still
      // needs to resync here, not just from the poll path above.
      notifyReloadIfNeeded(next.job);
      if (next.update_available) {
        setVisible(true);
      }
    },
    [startJobPoll, notifyReloadIfNeeded],
  );

  useEffect(() => {
    if (!enabled) {
      // Re-enabling will rediscover any still-running job.
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

  // Cross-tab nudge: a tab that only checks hourly would otherwise stay
  // pointed at a server-unloaded model for up to an hour after a DIFFERENT
  // open tab applies an update. The storage event only fires in other tabs
  // (never the one that wrote it), so this recheck fires promptly there
  // without this tab redundantly re-triggering itself.
  useEffect(() => {
    if (!enabled) return;
    const onStorage = (event: StorageEvent) => {
      if (
        event.key === HANDLED_RELOAD_STORAGE_KEY &&
        event.newValue &&
        event.newValue !== reloadNotifiedForRef.current
      ) {
        recheckStatus().then(surfaceIfAvailable);
      }
    };
    window.addEventListener("storage", onStorage);
    return () => window.removeEventListener("storage", onStorage);
  }, [enabled, surfaceIfAvailable]);

  const dismiss = useCallback(() => {
    setVisible(false);
  }, []);

  const snooze = useCallback(() => {
    setVisible(false);
    if (snoozeTimer.current) clearTimeout(snoozeTimer.current);
    snoozeTimer.current = setTimeout(() => {
      snoozeTimer.current = null;
      recheckStatus().then(surfaceIfAvailable);
    }, SNOOZE_DELAY_MS);
  }, [surfaceIfAvailable]);

  const apply = useCallback(
    async (
      confirm: (target: LlamaApplyTarget) => boolean | Promise<boolean>,
    ): Promise<LlamaApplyResult> => {
      if (applying) return { ok: false, error: "already running" };

      // Step 1: describe the pending swap. /confirm force-refreshes and mints a
      // single-use token bound to the offered build, and reports the host + the
      // from/to tags. No install starts here, so `applying` stays false while the
      // confirmation prompt is open.
      let target: LlamaApplyTarget;
      try {
        const cres = await authFetch("/api/llama/update/confirm", {
          method: "POST",
        });
        if (!cres.ok) return { ok: false, error: `HTTP ${cres.status}` };
        const confirmResp = (await cres.json().catch(() => null)) as {
          appliable?: boolean;
          reason?: string | null;
          confirm_token?: string | null;
          machine?: unknown;
          installed_tag?: string | null;
          latest_tag?: string | null;
        } | null;
        if (!confirmResp?.appliable || !confirmResp.confirm_token) {
          return {
            ok: false,
            error: confirmResp?.reason ?? "update is not applicable",
          };
        }
        target = {
          token: confirmResp.confirm_token,
          machine: parseMachine(confirmResp.machine),
          fromTag:
            typeof confirmResp.installed_tag === "string"
              ? confirmResp.installed_tag
              : null,
          toTag:
            typeof confirmResp.latest_tag === "string"
              ? confirmResp.latest_tag
              : null,
        };
      } catch (e) {
        return { ok: false, error: String(e) };
      }

      // Step 2: require an explicit user confirmation of the exact build + host
      // before the destructive binary swap. Declining aborts, untouched.
      const accepted = await confirm(target);
      if (!accepted) return { ok: false, error: "canceled" };

      // Step 3: apply with the confirmed single-use token.
      setApplying(true);
      setVisible(true);
      let action: {
        started?: boolean;
        reason?: string | null;
        message?: string | null;
        job?: unknown;
      } | null = null;
      try {
        const res = await authFetch("/api/llama/update", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ confirm_token: target.token }),
        });
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

      // Non-started jobs stay idle; already_running is tracked below.
      if (
        action &&
        action.started === false &&
        action.reason !== "already_running"
      ) {
        // A stale banner's click can land after another tab already applied the
        // update (e.g. "up_to_date"): the response still carries that tab's
        // completed job, so process reload_required here too, not just from the
        // poll path -- otherwise this rejection silently drops it.
        notifyReloadIfNeeded(parseJob(action.job));
        setApplying(false);
        return {
          ok: false,
          error: action.message ?? action.reason ?? "update was not started",
          machine: target.machine,
          fromTag: target.fromTag,
          tag: target.toTag,
        };
      }

      // Surface the confirmed host + from/to build alongside the job's post-swap
      // tag, so the result can report what changed and where.
      const polled = await new Promise<LlamaApplyResult>((resolve) =>
        startJobPoll(resolve),
      );
      return {
        ...polled,
        machine: target.machine,
        fromTag: target.fromTag,
        tag: polled.tag ?? target.toTag,
      };
    },
    [applying, startJobPoll, notifyReloadIfNeeded],
  );

  return {
    status: enabled ? status : null,
    visible: enabled && visible,
    applying,
    apply,
    dismiss,
    snooze,
  };
}
