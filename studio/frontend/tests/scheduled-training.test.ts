// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import {
  activeTrainingSchedule,
  isTrainingScheduleActive,
  latestTrainingScheduleProblem,
  trainingScheduleDraftValid,
} from "../src/features/training/lib/scheduled-training.ts";
import type { TrainingSchedule } from "../src/features/training/types/api.ts";
import type * as ScheduledTrainingStore from "../src/features/training/stores/scheduled-training-store.ts";
import {
  formatLocalMinute,
  isClockTime,
  isDateKey,
  localDateKey,
  nextQuarterHour,
  timeChoices,
  zoneLabel,
} from "../src/lib/schedule-time.ts";
import { loadWithStubs } from "./helpers/module-stubs.ts";
import * as zustand from "zustand";

function schedule(
  id: string,
  status: TrainingSchedule["status"],
): TrainingSchedule {
  return {
    id,
    status,
    startAt: 0,
    startAtLocal: "2026-10-30T09:00",
    timezone: "UTC",
    modelName: "unsloth/Llama-3.2-1B-Instruct",
    projectName: null,
    createdAt: 0,
    updatedAt: 0,
    jobId: null,
    error: null,
  };
}

test("a fresh start time is the next quarter hour, never in the past", () => {
  assert.deepEqual(nextQuarterHour(new Date(2026, 9, 29, 10, 7, 42)), {
    date: "2026-10-29",
    time: "10:15",
  });
  assert.deepEqual(nextQuarterHour(new Date(2026, 9, 29, 10, 46)), {
    date: "2026-10-29",
    time: "11:00",
  });
  assert.deepEqual(nextQuarterHour(new Date(2026, 9, 29, 23, 50)), {
    date: "2026-10-30",
    time: "00:00",
  });
});

test("wall-clock shapes are checked, not just parsed", () => {
  assert.equal(isClockTime("00:00"), true);
  assert.equal(isClockTime("23:59"), true);
  assert.equal(isClockTime("24:00"), false);
  assert.equal(isClockTime("9:00"), false);
  assert.equal(isDateKey("2026-10-29"), true);
  // The shape, not the calendar: the server validates the day itself.
  assert.equal(isDateKey("2026-13-01"), true);
  assert.equal(isDateKey("2026-1-01"), false);
  assert.equal(isDateKey("29.10.2026"), false);
  assert.equal(localDateKey(new Date(2026, 0, 2, 3, 4)), "2026-01-02");
});

test("a scheduled draft must still be ahead", () => {
  const now = new Date(2026, 9, 29, 10, 7);
  assert.equal(trainingScheduleDraftValid("2026-10-29", "10:15", now), true);
  assert.equal(trainingScheduleDraftValid("2026-10-30", "09:00", now), true);
  assert.equal(trainingScheduleDraftValid("2026-10-29", "10:00", now), false);
  assert.equal(trainingScheduleDraftValid("2026-10-29", "10:07", now), false);
  assert.equal(trainingScheduleDraftValid("2026-10-29", "9:00", now), false);
  assert.equal(trainingScheduleDraftValid("", "10:15", now), false);
});

test("the waiting start and the newest problem are picked from the list", () => {
  const pending = schedule("pending", "pending");
  const started = schedule("started", "started");
  const failed = schedule("failed", "failed");
  const missed = schedule("missed", "missed");
  const cancelled = schedule("cancelled", "cancelled");

  assert.equal(isTrainingScheduleActive(pending), true);
  assert.equal(isTrainingScheduleActive(schedule("starting", "starting")), true);
  assert.equal(isTrainingScheduleActive(failed), false);

  assert.equal(activeTrainingSchedule([started, pending]), pending);
  assert.equal(activeTrainingSchedule([started, cancelled]), null);

  assert.equal(latestTrainingScheduleProblem([failed, missed], null), failed);
  assert.equal(latestTrainingScheduleProblem([failed, missed], "failed"), missed);
  assert.equal(latestTrainingScheduleProblem([started, cancelled], null), null);
  assert.equal(latestTrainingScheduleProblem([failed], "failed"), null);
});

test("the list offers quarter hours and keeps a time picked off the grid", () => {
  const quarterHours = timeChoices("09:15");
  assert.equal(quarterHours.length, 96);
  assert.equal(quarterHours[0], "00:00");
  assert.equal(quarterHours[95], "23:45");

  const offGrid = timeChoices("09:07");
  assert.equal(offGrid.length, 97);
  assert.deepEqual(offGrid.slice(36, 39), ["09:00", "09:07", "09:15"]);
});

test("times are written in the locale's clock and a minute is a wall clock", () => {
  const written = formatLocalMinute("2026-10-29T09:05", "en-US").replace(
    /\s/g,
    " ",
  );
  assert.match(written, /^Oct 29, 2026,? 9:05/);
  assert.equal(zoneLabel("UTC", "en-US", Date.UTC(2026, 9, 29)), "UTC");
  // An unknown zone has no name, so the zone id itself is the fallback.
  assert.equal(zoneLabel("Mars/Olympus", "en-US", 0), "Mars/Olympus");
});

function loadScheduledTrainingStore(api: {
  listTrainingSchedules: () => Promise<unknown>;
  createTrainingSchedule?: (input: unknown) => Promise<TrainingSchedule>;
  cancelTrainingSchedule?: (id: string) => Promise<void>;
}): typeof ScheduledTrainingStore {
  return loadWithStubs<typeof ScheduledTrainingStore>(
    new URL(
      "../src/features/training/stores/scheduled-training-store.ts",
      import.meta.url,
    ),
    {
      zustand,
      "@/features/auth": {
        AUTH_SESSION_CLEARED_EVENT: "signed-out",
        getAuthSessionEpoch: () => 1,
      },
      "@/lib/schedule-time.ts": {
        localDateKey: () => "2026-10-29",
        nextQuarterHour: () => ({ date: "2026-10-29", time: "10:15" }),
        pad: (value: number) => String(value).padStart(2, "0"),
      },
      "../api/train-api": {
        listTrainingSchedules: api.listTrainingSchedules,
        createTrainingSchedule:
          api.createTrainingSchedule ??
          (async () => schedule("created", "pending")),
        cancelTrainingSchedule: api.cancelTrainingSchedule ?? (async () => undefined),
      },
      "../lib/scheduled-training": {
        activeTrainingSchedule,
        isTrainingScheduleActive,
        latestTrainingScheduleProblem,
        trainingScheduleDraftValid,
      },
    },
  );
}

test("a schedule write drops a list read that was already in flight", async () => {
  const globals = globalThis as Record<string, unknown>;
  const savedWindow = globals.window;
  globals.window = new EventTarget();
  try {
    let answer: (list: unknown) => void = () => undefined;
    const store = loadScheduledTrainingStore({
      listTrainingSchedules: () =>
        new Promise((resolve) => {
          answer = resolve;
        }),
      createTrainingSchedule: async () => schedule("created", "pending"),
    });
    const state = store.useScheduledTrainingStore;
    state.setState({ status: "ready" });
    const refresh = state.getState().refresh();
    await state.getState().create({
      start_at: "2026-10-30T09:00",
      timezone: "UTC",
      request: {} as never,
    });
    answer({ schedules: [], now: 0 });
    await refresh;
    assert.deepEqual(
      state.getState().schedules.map((row) => row.id),
      ["created"],
    );
  } finally {
    globals.window = savedWindow;
  }
});

test("cancelling drops a list read that was already in flight", async () => {
  const globals = globalThis as Record<string, unknown>;
  const savedWindow = globals.window;
  globals.window = new EventTarget();
  try {
    let answer: (list: unknown) => void = () => undefined;
    const store = loadScheduledTrainingStore({
      listTrainingSchedules: () =>
        new Promise((resolve) => {
          answer = resolve;
        }),
    });
    const state = store.useScheduledTrainingStore;
    state.setState({ schedules: [schedule("waiting", "pending")], status: "ready" });
    const refresh = state.getState().refresh();
    await state.getState().cancel("waiting");
    answer({ schedules: [schedule("waiting", "pending")], now: 0 });
    await refresh;
    assert.equal(state.getState().schedules[0]?.status, "cancelled");
  } finally {
    globals.window = savedWindow;
  }
});

test("turning scheduling on replaces a draft that is already in the past", () => {
  const globals = globalThis as Record<string, unknown>;
  const savedWindow = globals.window;
  globals.window = new EventTarget();
  try {
    const state = loadScheduledTrainingStore({
      listTrainingSchedules: async () => ({ schedules: [], now: 0 }),
    }).useScheduledTrainingStore;
    state.setState({
      draftEnabled: false,
      draftDate: "2020-01-01",
      draftTime: "00:00",
    });
    state.getState().setDraftEnabled(true);
    assert.equal(state.getState().draftEnabled, true);
    assert.equal(state.getState().draftDate, "2026-10-29");
    assert.equal(state.getState().draftTime, "10:15");
  } finally {
    globals.window = savedWindow;
  }
});

test("signing out forgets the account's scheduled starts", () => {
  const globals = globalThis as Record<string, unknown>;
  const savedWindow = globals.window;
  const window = new EventTarget();
  globals.window = window;
  try {
    const state = loadScheduledTrainingStore({
      listTrainingSchedules: async () => ({ schedules: [], now: 0 }),
    }).useScheduledTrainingStore;
    state.setState({
      schedules: [schedule("waiting", "pending")],
      status: "ready",
      dismissedId: "waiting",
    });
    window.dispatchEvent(new Event("signed-out"));
    assert.deepEqual(state.getState().schedules, []);
    assert.equal(state.getState().status, "idle");
    assert.equal(state.getState().dismissedId, null);
  } finally {
    globals.window = savedWindow;
  }
});
