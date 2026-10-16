// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";
import { loadWithStubs } from "./helpers/module-stubs.ts";
import * as formatFastApiError from "../src/lib/format-fastapi-error.ts";
import type * as ScheduledTasksApi from "../src/features/scheduled-tasks/api.ts";
import {
  ACTIVE_POLL_MS,
  CATCH_UP_MS,
  IDLE_POLL_MS,
  advanceSeenRuns,
  claimAnnouncements,
  collectFinishedRuns,
  nextPollDelay,
  readSeenRuns,
  runErrorText,
  runsToAnnounce,
  serializeSeenRuns,
} from "../src/features/scheduled-tasks/results.ts";
import {
  type ScheduleSummary,
  type TaskSchedule,
  defaultScheduleForm,
  formFromSchedule,
  relativeTimeParts,
  sameSchedule,
  scheduleFormComplete,
  scheduleFromForm,
  summarizeSchedule,
  titleFromPrompt,
} from "../src/features/scheduled-tasks/schedule.ts";

// ICU puts a narrow no-break space before AM/PM; compare the words, not the spacing.
function plain(summary: ScheduleSummary): ScheduleSummary {
  const values = Object.fromEntries(
    Object.entries(summary.values).map(([key, value]) => [
      key,
      typeof value === "string" ? value.replace(/\s/g, " ") : value,
    ]),
  );
  return { ...summary, values } as ScheduleSummary;
}

test("a new task runs every day at 09:00, and a one-time one starts tomorrow", () => {
  // Wednesday 14 October 2026, in the test machine's own zone.
  assert.deepEqual(defaultScheduleForm(new Date(2026, 9, 14, 16, 30)), {
    repeat: "daily",
    time: "09:00",
    date: "2026-10-15",
    days: [3],
    dayOfMonth: 14,
  });
  // The last day of a year rolls over into the next one.
  assert.equal(defaultScheduleForm(new Date(2026, 11, 31, 12)).date, "2027-01-01");
});

test("the form round-trips every kind of schedule", () => {
  const now = new Date(2026, 9, 14);
  const schedules: TaskSchedule[] = [
    { kind: "once", at: "2026-10-15T09:30" },
    { kind: "daily", time: "07:00" },
    { kind: "weekly", time: "18:00", days: [1, 2, 3, 4, 5] },
    { kind: "weekly", time: "18:00", days: [2, 6] },
    { kind: "monthly", time: "08:15", day: 31 },
  ];
  for (const schedule of schedules) {
    assert.deepEqual(scheduleFromForm(formFromSchedule(schedule, now)), schedule);
  }
});

test("Monday to Friday reads back as Weekdays, anything else as Weekly", () => {
  const now = new Date(2026, 9, 14);
  const weekdays = formFromSchedule({ kind: "weekly", time: "09:00", days: [1, 2, 3, 4, 5] }, now);
  const weekly = formFromSchedule({ kind: "weekly", time: "09:00", days: [1, 2, 3, 4] }, now);
  assert.equal(weekdays.repeat, "weekdays");
  assert.deepEqual([weekly.repeat, weekly.days], ["weekly", [1, 2, 3, 4]]);
});

test("weekly days are sent once each, in order", () => {
  const form = { ...defaultScheduleForm(new Date(2026, 9, 14)), repeat: "weekly" as const, days: [5, 1, 5] };
  assert.deepEqual(scheduleFromForm(form), { kind: "weekly", time: "09:00", days: [1, 5] });
});

test("an incomplete form cannot be saved", () => {
  const form = defaultScheduleForm(new Date(2026, 9, 14));
  assert.equal(scheduleFormComplete(form), true);
  assert.equal(scheduleFormComplete({ ...form, time: "" }), false);
  assert.equal(scheduleFormComplete({ ...form, time: "24:00" }), false);
  assert.equal(scheduleFormComplete({ ...form, repeat: "once", date: "" }), false);
  assert.equal(scheduleFormComplete({ ...form, repeat: "weekly", days: [] }), false);
  assert.equal(scheduleFormComplete({ ...form, repeat: "monthly", dayOfMonth: 32 }), false);
  // Daily ignores the fields it does not use.
  assert.equal(scheduleFormComplete({ ...form, days: [], date: "" }), true);
});

test("schedules compare by what they mean", () => {
  assert.ok(
    sameSchedule(
      { kind: "weekly", time: "09:00", days: [1, 3] },
      { kind: "weekly", time: "09:00", days: [3, 1] },
    ),
  );
  assert.ok(!sameSchedule({ kind: "daily", time: "09:00" }, { kind: "daily", time: "09:01" }));
  assert.ok(
    !sameSchedule(
      { kind: "daily", time: "09:00" },
      { kind: "weekly", time: "09:00", days: [1, 2, 3, 4, 5, 6, 7] },
    ),
  );
  assert.ok(
    !sameSchedule(
      { kind: "monthly", time: "09:00", day: 1 },
      { kind: "monthly", time: "09:00", day: 2 },
    ),
  );
});

test("schedules are described in the user's locale", () => {
  assert.deepEqual(plain(summarizeSchedule({ kind: "daily", time: "21:05" }, "en-US")), {
    key: "daily",
    values: { time: "9:05 PM" },
  });
  assert.deepEqual(summarizeSchedule({ kind: "daily", time: "21:05" }, "de"), {
    key: "daily",
    values: { time: "21:05" },
  });
  assert.deepEqual(
    plain(summarizeSchedule({ kind: "weekly", time: "09:00", days: [1, 3, 5] }, "en-US")),
    { key: "weekly", values: { days: "Mon, Wed, and Fri", time: "9:00 AM" } },
  );
  assert.equal(
    summarizeSchedule({ kind: "weekly", time: "09:00", days: [1, 2, 3, 4, 5] }, "en-US").key,
    "weekdays",
  );
  // Every day of the week is every day.
  assert.equal(
    summarizeSchedule({ kind: "weekly", time: "09:00", days: [1, 2, 3, 4, 5, 6, 7] }, "en-US").key,
    "daily",
  );
  assert.deepEqual(plain(summarizeSchedule({ kind: "once", at: "2026-10-15T09:00" }, "en-US")), {
    key: "once",
    values: { date: "Oct 15, 2026", time: "9:00 AM" },
  });
  assert.deepEqual(plain(summarizeSchedule({ kind: "monthly", time: "09:00", day: 31 }, "en-US")), {
    key: "monthly",
    values: { day: 31, time: "9:00 AM" },
  });
});

test("the results poll looks quickly while a run is going and again just after one is due", () => {
  assert.equal(nextPollDelay({ now: 0, activeRuns: 1, nextRunAt: null }), ACTIVE_POLL_MS);
  assert.equal(nextPollDelay({ now: 0, activeRuns: 0, nextRunAt: null }), IDLE_POLL_MS);
  assert.equal(nextPollDelay({ now: 0, activeRuns: 0, nextRunAt: 60_000 }), 63_000);
  // Due in a second: asks again just after, once the run has been claimed.
  assert.equal(nextPollDelay({ now: 0, activeRuns: 0, nextRunAt: 1_000 }), 4_000);
  assert.equal(nextPollDelay({ now: 10_000, activeRuns: 0, nextRunAt: 0 }), ACTIVE_POLL_MS);
  assert.equal(nextPollDelay({ now: 0, activeRuns: 0, nextRunAt: 3_600_000 }), IDLE_POLL_MS);
  // Results still to be read: the next page comes soon.
  assert.equal(nextPollDelay({ now: 0, activeRuns: 0, nextRunAt: null, more: true }), ACTIVE_POLL_MS);
});

test("announcing resumes from a record saved within the catch-up window, by this browser's clock", () => {
  const now = 10 * CATCH_UP_MS;
  const stored = (savedAt: number, since = 5_000, ids: unknown = ["run-1"]) =>
    JSON.stringify({ since, ids, savedAt });
  assert.deepEqual(readSeenRuns(stored(now - 60_000), now), { since: 5_000, ids: ["run-1"] });
  // `since` is the server's time: a browser clock behind it or far ahead of it is no reason to drop it.
  assert.deepEqual(readSeenRuns(stored(now, now + 3_600_000), now), { since: now + 3_600_000, ids: ["run-1"] });
  assert.deepEqual(readSeenRuns(stored(now, now - 3 * CATCH_UP_MS), now)?.since, now - 3 * CATCH_UP_MS);
  // A clock set back since the save puts the save ahead of it, which counts as recent.
  assert.deepEqual(readSeenRuns(stored(now + 60_000), now), { since: 5_000, ids: ["run-1"] });
  assert.deepEqual(readSeenRuns(stored(now + 2 * CATCH_UP_MS), now), { since: 5_000, ids: ["run-1"] });
  assert.equal(readSeenRuns(null, now), null);
  assert.equal(readSeenRuns("not json", now), null);
  assert.equal(readSeenRuns(JSON.stringify({ ids: [], savedAt: now }), now), null);
  assert.equal(readSeenRuns(JSON.stringify({ since: 5_000, ids: [] }), now), null);
  // Junk ids are dropped rather than failing the whole record.
  assert.deepEqual(readSeenRuns(stored(now, 5_000, ["a", 7, null, "b"]), now), { since: 5_000, ids: ["a", "b"] });
  // What is saved reads back.
  const seen = { since: 5_000, ids: ["a"] };
  assert.deepEqual(readSeenRuns(serializeSeenRuns(seen, now), now + 60_000), seen);
});

test("a record older than the catch-up window resumes from that window before now, on the server's clock", () => {
  // Saved two days ago, a minute behind the server then: the server's now is about since + slack + age.
  const savedAt = 1_000_000_000;
  const now = savedAt + 2 * CATCH_UP_MS;
  const since = 7_000_000;
  const resumed = readSeenRuns(JSON.stringify({ since, ids: ["old"], savedAt }), now);
  assert.deepEqual(resumed, { since: since + 60_000 + CATCH_UP_MS, ids: [] });
});

test("the record moves on every poll, a minute behind the server, and remembers what it announced", () => {
  const answer = (now: number) => ({ now, runs: [], more: false });
  const first = advanceSeenRuns(null, answer(600_000), []);
  assert.deepEqual(first, { since: 540_000, ids: [] });
  const second = advanceSeenRuns(first, answer(900_000), ["run-1"]);
  assert.deepEqual(second, { since: 840_000, ids: ["run-1"] });
  // A poll answered by a server clock behind the record never moves it back.
  assert.equal(advanceSeenRuns(second, answer(700_000), []).since, 840_000);
  const many = Array.from({ length: 260 }, (_, index) => `run-${index}`);
  const capped = advanceSeenRuns(second, answer(900_000), many).ids;
  assert.equal(capped.length, 200);
  assert.equal(capped.at(-1), "run-259");
});

function finished(id: string, finishedAt: number): ScheduledTasksApi.FinishedScheduledTaskRun {
  return {
    id,
    taskId: "task-1",
    source: "manual",
    status: "succeeded",
    scheduledFor: null,
    createdAt: finishedAt,
    startedAt: finishedAt,
    finishedAt,
    threadId: `chat-${id}`,
    model: null,
    errorCode: null,
    error: null,
    taskTitle: "Morning brief",
  };
}

test("results past one answer are read a page at a time and announced once", async () => {
  const runs = [finished("a", 1_000), finished("b", 2_000), finished("c", 3_000)];
  const asked: (number | null)[] = [];
  // Two runs a page, and `since` is inclusive, so the next page repeats the boundary run.
  const fetchAnswer = async (since: number | null) => {
    asked.push(since);
    const after = runs.filter((run) => since !== null && (run.finishedAt ?? 0) >= since);
    return { now: 100_000, runs: after.slice(0, 2), more: after.length > 2, activeRuns: 0, nextRunAt: null };
  };
  const { fresh, seen, last } = await collectFinishedRuns({ since: 0, ids: [] }, fetchAnswer);
  assert.deepEqual(asked, [0, 2_000]);
  assert.deepEqual(fresh.map((run) => run.id), ["a", "b", "c"]);
  assert.deepEqual(seen, { since: 40_000, ids: ["a", "b", "c"] });
  assert.equal(last.more, false);
});

test("a run that succeeded after its chat was deleted is not announced, least of all as failed", () => {
  const ready = finished("ready", 1_000);
  const gone = { ...finished("gone", 2_000), threadId: null };
  const failed = { ...finished("failed", 3_000), status: "failed" as const, threadId: null };
  assert.deepEqual(runsToAnnounce([ready, gone, failed]).map((run) => run.id), ["ready", "failed"]);
});

test("of an account's tabs, the first to claim a run announces it, and the record keeps the furthest cursor", () => {
  const fresh = [finished("a", 3_000), finished("b", 3_500)];
  // Another tab claimed "a", and its cursor is just past it.
  const { mine, record } = claimAnnouncements({ since: 3_200, ids: ["a"] }, { since: 4_000, ids: ["a", "b"] }, fresh);
  assert.deepEqual(mine.map((run) => run.id), ["b"]);
  assert.deepEqual(record, { since: 4_000, ids: ["a", "b"] });
  // One that caught up on a long backlog moved the cursor past runs its list has since let go of.
  const late = claimAnnouncements({ since: 3_200, ids: [] }, { since: 4_000, ids: [] }, fresh);
  assert.deepEqual(late.mine.map((run) => run.id), ["b"]);
  // The record never falls back to a slower tab's cursor.
  assert.equal(claimAnnouncements({ since: 5_000, ids: [] }, { since: 4_000, ids: [] }, fresh).record.since, 5_000);
  // With no record yet, everything new is this tab's.
  const first = claimAnnouncements(null, { since: 4_000, ids: [] }, fresh);
  assert.deepEqual(first.mine.map((run) => run.id), ["a", "b"]);
  assert.deepEqual(first.record, { since: 4_000, ids: ["a", "b"] });
});

test("a backlog longer than one poll reads is picked up again shortly", async () => {
  let pages = 0;
  const endless = async () => {
    pages += 1;
    return { now: 100_000, runs: [finished(`run-${pages}`, pages)], more: true, activeRuns: 0, nextRunAt: null };
  };
  const { fresh, seen, last } = await collectFinishedRuns(null, endless);
  assert.equal(pages, 10);
  assert.equal(fresh.length, 10);
  // Up to the last run read, not the server's present, so nothing after it is skipped.
  assert.equal(seen.since, 10);
  assert.equal(nextPollDelay(last), ACTIVE_POLL_MS);
});

test("a failed run is worded from its error code, else from the server's text", () => {
  const t = (key: string) => `<${key}>`;
  assert.equal(
    runErrorText({ errorCode: "no_model", error: null }, t),
    "<scheduledTasks.errors.noModel>",
  );
  assert.equal(runErrorText({ errorCode: null, error: "  Context full.  " }, t), "Context full.");
  assert.equal(
    runErrorText({ errorCode: "something_new", error: null }, t),
    "<scheduledTasks.errors.unknown>",
  );
});

test("relative times use the largest whole unit", () => {
  assert.deepEqual(relativeTimeParts(0, 30_000), [-30, "second"]);
  assert.deepEqual(relativeTimeParts(0, 5 * 60_000), [-5, "minute"]);
  assert.deepEqual(relativeTimeParts(0, 2 * 3_600_000), [-2, "hour"]);
  assert.deepEqual(relativeTimeParts(3 * 86_400_000, 0), [3, "day"]);
});

function client(respond: (path: string, init?: RequestInit) => Response) {
  const requests: { method: string; path: string; body: unknown }[] = [];
  const api = loadWithStubs<typeof ScheduledTasksApi>(
    new URL("../src/features/scheduled-tasks/api.ts", import.meta.url),
    {
      "@/features/auth": {
        authFetch: async (path: string, init?: RequestInit) => {
          requests.push({
            method: init?.method ?? "GET",
            path,
            body: typeof init?.body === "string" ? JSON.parse(init.body) : undefined,
          });
          return respond(path, init);
        },
        getAuthSessionEpoch: () => 1,
      },
      "@/lib/format-fastapi-error": formatFastApiError,
    },
  );
  return { api, requests };
}

test("the client reaches every scheduled task route", async () => {
  const c = client((path, init) =>
    init?.method === "DELETE"
      ? new Response(null, { status: 204 })
      : Response.json(path === "/api/scheduled-tasks" && !init?.method ? { tasks: [] } : {}),
  );
  const schedule: TaskSchedule = { kind: "daily", time: "09:00" };
  await c.api.listScheduledTasks();
  await c.api.createScheduledTask({ title: "Brief", prompt: "Hi", schedule, timezone: "UTC" });
  await c.api.updateScheduledTask("a/b", { enabled: false });
  assert.equal(await c.api.deleteScheduledTask("t1"), undefined);
  await c.api.runScheduledTaskNow("t1");
  await c.api.getScheduledTaskActivity(null);
  await c.api.getScheduledTaskActivity(42);
  assert.deepEqual(
    c.requests.map(({ method, path }) => [method, path]),
    [
      ["GET", "/api/scheduled-tasks"],
      ["POST", "/api/scheduled-tasks"],
      ["PATCH", "/api/scheduled-tasks/a%2Fb"],
      ["DELETE", "/api/scheduled-tasks/t1"],
      ["POST", "/api/scheduled-tasks/t1/run"],
      ["GET", "/api/scheduled-tasks/runs"],
      ["GET", "/api/scheduled-tasks/runs?since=42"],
    ],
  );
  assert.deepEqual(c.requests[1].body, {
    title: "Brief",
    prompt: "Hi",
    schedule,
    timezone: "UTC",
  });
  assert.deepEqual(c.requests[2].body, { enabled: false });
});

test("a refused request carries the server's reason", async () => {
  const c = client(() => Response.json({ detail: "Pick a time in the future." }, { status: 400 }));
  await assert.rejects(c.api.updateScheduledTask("t1", { enabled: true }), {
    message: "Pick a time in the future.",
  });
});

test("a task left unnamed is named after its instructions' first line", () => {
  assert.equal(titleFromPrompt("  What's the weather today?\nIn Paris."), "What's the weather today?");
  assert.equal(titleFromPrompt("Summarize   three\tpapers"), "Summarize three papers");
  const long = "Write a short upbeat note to start the day with three ideas for staying focused";
  const title = titleFromPrompt(long);
  assert.ok(title.endsWith("…") && title.length <= 61, title);
  assert.ok(long.startsWith(title.slice(0, -1)), "cut at a word, from the start");
});
