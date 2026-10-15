// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/** What the results poll needs, kept apart from the Tasks page so the startup bundle carries no
 *  more than this. Only type imports, so the tests load it directly. */

import type { InterpolationValues, TranslationKey } from "@/i18n";

/** A task was added, rescheduled or started: the results poll looks again now. */
export const SCHEDULED_TASKS_CHANGED_EVENT = "unsloth-scheduled-tasks-changed";
/** Runs finished: a loaded Tasks page re-reads its list to show them. */
export const SCHEDULED_TASK_RUNS_FINISHED_EVENT = "unsloth-scheduled-task-runs-finished";

export const ACTIVE_POLL_MS = 2_000;
export const IDLE_POLL_MS = 5 * 60_000;
// Past the due time, so the run has been claimed by the time the poll asks.
const DUE_SLACK_MS = 3_000;
/** Runs that finished longer ago than this when Studio is reopened are not announced again. */
export const CATCH_UP_MS = 24 * 60 * 60_000;
// A run committed just after the server read its clock finished before the time it reported, so
// each poll asks from a minute earlier and skips the runs it has already announced.
const CURSOR_SLACK_MS = 60_000;
// Far more than finish inside one slack window, since runs go one at a time.
const SEEN_IDS_LIMIT = 50;

/** How long until the results poll should look again: quickly while a run is going, just after
 *  the next one is due, and otherwise every few minutes for tasks made in another tab. */
export function nextPollDelay(activity: {
  now: number;
  activeRuns: number;
  nextRunAt: number | null;
}): number {
  if (activity.activeRuns > 0) return ACTIVE_POLL_MS;
  if (activity.nextRunAt === null) return IDLE_POLL_MS;
  const untilDue = activity.nextRunAt - activity.now + DUE_SLACK_MS;
  return Math.min(IDLE_POLL_MS, Math.max(ACTIVE_POLL_MS, untilDue));
}

/** What the results poll has announced, kept across launches so a reopened Studio reports the
 *  runs that finished while it was closed. */
export interface SeenRuns {
  /** Server time from which finished runs are still to be announced. */
  since: number;
  /** Runs already announced at or after `since`. */
  ids: string[];
}

/** The stored record, if it is still within the catch-up window. Without one, announcing starts
 *  from the server's present and covers only what finishes from then on. */
export function readSeenRuns(stored: string | null, now: number): SeenRuns | null {
  let value: unknown;
  try {
    value = JSON.parse(stored ?? "null");
  } catch {
    return null;
  }
  const { since, ids } = (value ?? {}) as Partial<SeenRuns>;
  if (typeof since !== "number" || !Number.isFinite(since)) return null;
  if (since > now || now - since > CATCH_UP_MS) return null;
  return {
    since,
    ids: (Array.isArray(ids) ? ids : [])
      .filter((id): id is string => typeof id === "string")
      .slice(-SEEN_IDS_LIMIT),
  };
}

/** The record after a poll the server answered at `serverNow`, having announced `announced`. It
 *  moves on even when nothing finished, so it is never too old to resume from. */
export function advanceSeenRuns(
  seen: SeenRuns | null,
  serverNow: number,
  announced: readonly string[],
): SeenRuns {
  return {
    since: Math.max(seen?.since ?? 0, serverNow - CURSOR_SLACK_MS),
    ids: [...(seen?.ids ?? []), ...announced].slice(-SEEN_IDS_LIMIT),
  };
}

type Translate = (key: TranslationKey, values?: InterpolationValues) => string;

// The backend's error codes; anything else carries its own text.
const ERROR_KEYS: Record<string, TranslationKey> = {
  no_model: "scheduledTasks.errors.noModel",
  timeout: "scheduledTasks.errors.timeout",
  cancelled: "scheduledTasks.errors.cancelled",
  interrupted: "scheduledTasks.errors.interrupted",
  empty: "scheduledTasks.errors.empty",
};

export function runErrorText(
  run: { errorCode: string | null; error: string | null },
  t: Translate,
): string {
  const key = run.errorCode ? ERROR_KEYS[run.errorCode] : undefined;
  if (key) return t(key);
  return run.error?.trim() || t("scheduledTasks.errors.unknown");
}
