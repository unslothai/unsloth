// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/** What the results poll needs, kept apart from the Tasks page so the startup bundle carries no
 *  more than this. Only type imports, so the tests load it directly. */

import type { InterpolationValues, TranslationKey } from "@/i18n";
import type { FinishedScheduledTaskRun, ScheduledTaskActivity } from "./api";

/** A task was added, rescheduled or started: the results poll looks again now. */
export const SCHEDULED_TASKS_CHANGED_EVENT = "unsloth-scheduled-tasks-changed";
/** Runs finished: a loaded Tasks page re-reads its list to show them. */
export const SCHEDULED_TASK_RUNS_FINISHED_EVENT = "unsloth-scheduled-task-runs-finished";

/** Where the results poll keeps each account's record, the account's name appended. Listed in
 *  `ACCOUNT_KEYED_PREFIXES`, so switching accounts leaves it for when this one signs in again. */
export const SEEN_KEY_PREFIX = "unsloth_scheduled_tasks_seen:";

export const ACTIVE_POLL_MS = 2_000;
export const IDLE_POLL_MS = 5 * 60_000;
// Past the due time, so the run has been claimed by the time the poll asks.
const DUE_SLACK_MS = 3_000;
/** Runs that finished longer ago than this when Studio is reopened are not announced again. */
export const CATCH_UP_MS = 24 * 60 * 60_000;
// A run committed just after the server read its clock finished before the time it reported, so
// each poll asks from a minute earlier and skips the runs it has already announced.
const CURSOR_SLACK_MS = 60_000;
// A whole answer's worth (the server's page size), so every run that can come back, from the slack
// window or a page boundary, is still on the list.
const SEEN_IDS_LIMIT = 200;
// Answers one poll reads before announcing; a longer backlog carries on at the next poll.
const PAGES_PER_POLL = 10;

/** How long until the results poll should look again: quickly while a run is going or results are
 *  still to be read, just after the next one is due, and otherwise every few minutes for tasks made
 *  in another tab. */
export function nextPollDelay(activity: {
  now: number;
  activeRuns: number;
  nextRunAt: number | null;
  more?: boolean;
}): number {
  if (activity.more || activity.activeRuns > 0) return ACTIVE_POLL_MS;
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

/** The record as stored, with when this browser saved it. */
export function serializeSeenRuns(seen: SeenRuns, now: number): string {
  return JSON.stringify({ since: seen.since, ids: seen.ids, savedAt: now });
}

/** The stored record, if it was saved within the catch-up window. Its age is read on this
 *  browser's clock, from `savedAt`: `since` is the server's time, which this clock need not agree
 *  with. Without one, announcing starts from the server's present and covers only what finishes
 *  from then on. */
export function readSeenRuns(stored: string | null, now: number): SeenRuns | null {
  let value: unknown;
  try {
    value = JSON.parse(stored ?? "null");
  } catch {
    return null;
  }
  const { since, ids, savedAt } = (value ?? {}) as Partial<SeenRuns & { savedAt: number }>;
  if (typeof since !== "number" || !Number.isFinite(since)) return null;
  // Either way: a clock set back since the save puts it in the future.
  if (typeof savedAt !== "number" || !(Math.abs(now - savedAt) <= CATCH_UP_MS)) return null;
  return {
    since,
    ids: (Array.isArray(ids) ? ids : [])
      .filter((id): id is string => typeof id === "string")
      .slice(-SEEN_IDS_LIMIT),
  };
}

/** The record after one answer from the server, having announced `announced`. A full answer stops
 *  at its last run, where the next one picks up. Otherwise the record moves to the server's present
 *  even when nothing finished, so it is never too old to resume from. */
export function advanceSeenRuns(
  seen: SeenRuns | null,
  answer: Pick<ScheduledTaskActivity, "now" | "runs" | "more">,
  announced: readonly string[],
): SeenRuns {
  const lastFinished = answer.runs.at(-1)?.finishedAt;
  const reached =
    answer.more && typeof lastFinished === "number"
      ? lastFinished
      : answer.now - CURSOR_SLACK_MS;
  return {
    since: Math.max(seen?.since ?? 0, reached),
    ids: [...(seen?.ids ?? []), ...announced].slice(-SEEN_IDS_LIMIT),
  };
}

/** Which of `fresh` this tab announces, and the record the account's tabs share once it has. Another
 *  tab may have announced some of them already, which `shared` lists, and they stay quiet here. */
export function claimAnnouncements<T extends { id: string }>(
  shared: SeenRuns | null,
  cursor: SeenRuns,
  fresh: readonly T[],
): { mine: T[]; record: SeenRuns } {
  const announced = new Set(shared?.ids);
  const mine = fresh.filter((run) => !announced.has(run.id));
  return {
    mine,
    record: {
      since: Math.max(shared?.since ?? 0, cursor.since),
      ids: [...(shared?.ids ?? []), ...mine.map((run) => run.id)].slice(-SEEN_IDS_LIMIT),
    },
  };
}

/** The runs finished since `seen` that it has not announced, read an answer at a time, with the
 *  record moved past them. `last` is the final answer, which paces the next poll. */
export async function collectFinishedRuns(
  seen: SeenRuns | null,
  fetchAnswer: (since: number | null) => Promise<ScheduledTaskActivity>,
): Promise<{ fresh: FinishedScheduledTaskRun[]; seen: SeenRuns; last: ScheduledTaskActivity }> {
  const fresh: FinishedScheduledTaskRun[] = [];
  let record = seen;
  for (let page = 1; ; page += 1) {
    const last = await fetchAnswer(record?.since ?? null);
    const announced = new Set(record?.ids);
    const unseen = last.runs.filter((run) => !announced.has(run.id));
    fresh.push(...unseen);
    record = advanceSeenRuns(
      record,
      last,
      unseen.map((run) => run.id),
    );
    if (!last.more || page >= PAGES_PER_POLL) return { fresh, seen: record, last };
  }
}

/** The finished runs worth a toast: one that succeeded but whose chat is already gone (deleted,
 *  say, from another tab) has nothing to open, and it did not fail either. */
export function runsToAnnounce<T extends { status: string; threadId: string | null }>(
  runs: readonly T[],
): T[] {
  return runs.filter((run) => run.status !== "succeeded" || run.threadId !== null);
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
