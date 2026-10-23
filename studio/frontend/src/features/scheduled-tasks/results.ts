// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0


import type { InterpolationValues, TranslationKey } from "@/i18n";
import type { FinishedScheduledTaskRun, ScheduledTaskActivity } from "./api";

/** A task was added, rescheduled or started: the results poll looks again now. */
export const SCHEDULED_TASKS_CHANGED_EVENT = "unsloth-scheduled-tasks-changed";
/** Runs finished: a loaded Tasks page re-reads its list to show them. */
export const SCHEDULED_TASK_RUNS_FINISHED_EVENT = "unsloth-scheduled-task-runs-finished";

/** Per-account results-poll record key; in `ACCOUNT_KEYED_PREFIXES`, so it survives account switches. */
export const SEEN_KEY_PREFIX = "unsloth_scheduled_tasks_seen:";
export const UNSEEN_KEY_PREFIX = "unsloth_scheduled_tasks_unseen:";

export const ACTIVE_POLL_MS = 2_000;
export const IDLE_POLL_MS = 5 * 60_000;
// Every hidden tab polling every 2 s through a long CPU run adds up; their notifications can wait.
export const HIDDEN_POLL_MS = 30_000;
const SEEN_SAVE_STEP_MS = 60_000;
const DUE_SLACK_MS = 3_000;
/** Runs that finished further behind the server's present than this are not announced. */
export const CATCH_UP_MS = 24 * 60 * 60_000;
// A run committed just after the server read its clock predates that time: poll from a minute earlier, skip seen ids.
const CURSOR_SLACK_MS = 60_000;
// The server's page size, so every run the slack window or a page boundary returns stays listed.
const SEEN_IDS_LIMIT = 200;
const PAGES_PER_POLL = 10;

/** Delay before the next results poll: short while runs are active or unread, else just after the next due time. */
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

export function pollDelayFor(delay: number, hidden: boolean): number {
  return hidden ? Math.max(delay, HIDDEN_POLL_MS) : delay;
}

export function shouldSaveSeen(
  stored: SeenRuns | null,
  record: SeenRuns,
  claimed: number,
): boolean {
  return claimed > 0 || stored === null || record.since - stored.since >= SEEN_SAVE_STEP_MS;
}

/** What the results poll has announced, persisted so a reopened Studio reports runs finished while closed. */
export interface SeenRuns {
  /** Server time from which finished runs are still to be announced. */
  since: number;
  /** Runs already announced at or after `since`. */
  ids: string[];
}

/** The stored record to resume from; the server's first answer, not this clock, decides how far behind it is. */
export function readSeenRuns(stored: string | null): SeenRuns | null {
  let value: unknown;
  try {
    value = JSON.parse(stored ?? "null");
  } catch {
    return null;
  }
  const { since, ids } = (value ?? {}) as Partial<SeenRuns>;
  if (typeof since !== "number" || !Number.isFinite(since)) return null;
  return {
    since,
    ids: (Array.isArray(ids) ? ids : [])
      .filter((id): id is string => typeof id === "string")
      .slice(-SEEN_IDS_LIMIT),
  };
}

/** The record after one server answer: a full answer stops at its last run, else it moves to the server's present. */
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

/** Which of `fresh` this tab announces (skipping ones another tab claimed), and the shared record after. */
export function claimAnnouncements<T extends { id: string; finishedAt: number | null }>(
  shared: SeenRuns | null,
  cursor: SeenRuns,
  fresh: readonly T[],
): { mine: T[]; record: SeenRuns } {
  const announced = new Set(shared?.ids);
  const claimedBefore = shared?.since ?? Number.NEGATIVE_INFINITY;
  const mine = fresh.filter(
    (run) => !announced.has(run.id) && (run.finishedAt ?? Number.POSITIVE_INFINITY) >= claimedBefore,
  );
  return {
    mine,
    record: {
      since: Math.max(shared?.since ?? 0, cursor.since),
      ids: [...(shared?.ids ?? []), ...mine.map((run) => run.id)].slice(-SEEN_IDS_LIMIT),
    },
  };
}

/** Unannounced runs since `seen`, read page by page; a record older than the catch-up window resumes from it (server clock only). */
export async function collectFinishedRuns(
  seen: SeenRuns | null,
  fetchAnswer: (since: number | null) => Promise<ScheduledTaskActivity>,
): Promise<{ fresh: FinishedScheduledTaskRun[]; seen: SeenRuns; last: ScheduledTaskActivity }> {
  const fresh: FinishedScheduledTaskRun[] = [];
  let record = seen;
  for (let page = 1; ; page += 1) {
    let last = await fetchAnswer(record?.since ?? null);
    const oldest = last.now - CATCH_UP_MS;
    if (page === 1 && record !== null && record.since < oldest) {
      record = { since: oldest, ids: record.ids };
      last = await fetchAnswer(oldest);
    }
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

/** Finished runs worth announcing: a succeeded run whose chat is already gone is skipped. */
export function runsToAnnounce<T extends { status: string; threadId: string | null }>(
  runs: readonly T[],
): T[] {
  return runs.filter((run) => run.status !== "succeeded" || run.threadId !== null);
}

export function runsToNotify<T extends { status: string; taskNotify: string }>(
  runs: readonly T[],
): T[] {
  return runs.filter(
    (run) => run.taskNotify === "all" || (run.taskNotify === "failures" && run.status === "failed"),
  );
}

type Translate = (key: TranslationKey, values?: InterpolationValues) => string;

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
