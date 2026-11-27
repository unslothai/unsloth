// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import { loadWithStubs } from "./helpers/module-stubs.ts";

const settle = () => new Promise<void>((resolve) => setImmediate(resolve));
const src = (path: string) => new URL(`../src/${path}`, import.meta.url);

function setup() {
  const target = new EventTarget();
  (globalThis as { document?: unknown }).document = {
    hidden: false,
    addEventListener: target.addEventListener.bind(target),
    removeEventListener: target.removeEventListener.bind(target),
  };
  (globalThis as { window?: unknown }).window = globalThis;
  let server = { phase: "training", job_id: "job-1" };
  let requests = 0;
  const notifications: string[] = [];
  const state: Record<string, unknown> = {
    jobId: "job-1",
    resetGeneration: 0,
    startRequestId: null,
    phase: "training",
    isTrainingRunning: true,
    isStarting: false,
    stopRequested: false,
    hasHydrated: false,
    isHydrating: false,
    currentStep: 10,
    lastEventId: null,
    startError: null,
    startModelName: "model",
    applyStatus: (p: { phase: string }) => {
      state.phase = p.phase;
      state.isTrainingRunning = p.phase === "training";
    },
    applyMetrics: () => {},
    setHydrating: (v: boolean) => {
      state.isHydrating = v;
    },
    setHasHydrated: (v: boolean) => {
      state.hasHydrated = v;
    },
    setSseConnected: () => {},
    setStartError: () => {},
  };
  const store = Object.assign(
    (select: (s: typeof state) => unknown) => select(state),
    // A snapshot per read, as zustand gives, so before/after transitions are visible.
    { getState: () => ({ ...state }), subscribe: () => () => {} },
  );
  const cleanups: (() => void)[][] = [[], []];
  let owner = 0;
  const react = {
    useEffect: (effect: () => (() => void) | undefined) => {
      const cleanup = effect();
      if (cleanup) cleanups[owner].push(cleanup);
    },
  };
  const stubs: Record<string, unknown> = {
    react,
    "../api/train-api": {
      getTrainingStatus: async () => {
        requests += 1;
        return { ...server };
      },
      getTrainingMetrics: async () => ({}),
      isAbortError: () => false,
      streamTrainingProgress: () => new Promise(() => {}),
    },
    "@/features/settings/low-disk-check": { checkDiskSpace: async () => {} },
    "@/lib/single-flight-request": loadWithStubs(
      src("lib/single-flight-request.ts"),
      {},
    ),
    "../lib/training-status-request": loadWithStubs(
      src("features/training/lib/training-status-request.ts"),
      {},
    ),
    "../lib/training-status-poll-owner": loadWithStubs(
      src("features/training/lib/training-status-poll-owner.ts"),
      {},
    ),
    "../stores/training-runtime-store": {
      useTrainingRuntimeStore: store,
      isTrainingStartPending: (s: typeof state) =>
        Boolean(s.isStarting || s.startRequestId || s.isTrainingRunning),
    },
    "@/features/auth": { hasAuthToken: () => true },
    "@/features/hub": { useHfTokenStore: store },
    "@/lib/native-notifications": {
      notifyNative: async (n: { key: string }) => {
        notifications.push(n.key);
      },
      safeNotificationLabel: (v: unknown) => v,
      sanitizeNotificationBody: (v: unknown) => v,
    },
    "../lib/training-stream-scope": {
      createTrainingStreamScope: () => ({ jobId: "job-1" }),
      isTrainingProgressForScope: () => true,
      isTrainingStreamScopeCurrent: () => true,
    },
    "../stores/training-config-store": { useTrainingConfigStore: store },
  };
  const watch = loadWithStubs<{ useTrainingCompletionWatch: () => void }>(
    src("features/training/hooks/use-training-completion-watch.ts"),
    stubs,
  );
  const lifecycle = loadWithStubs<{ useTrainingRuntimeLifecycle: () => void }>(
    src("features/training/hooks/use-training-runtime-lifecycle.ts"),
    stubs,
  );
  return {
    state,
    notifications,
    requests: () => requests,
    setServer: (next: typeof server) => {
      server = next;
    },
    mountWatch: () => {
      owner = 0;
      watch.useTrainingCompletionWatch();
    },
    mountTrainPage: () => {
      owner = 1;
      lifecycle.useTrainingRuntimeLifecycle();
    },
    leaveTrainPage: () => {
      for (const cleanup of cleanups[1].splice(0)) cleanup();
    },
    unmountAll: () => {
      for (const list of cleanups)
        for (const cleanup of list.splice(0)) cleanup();
    },
  };
}

test("on the Train page the completion watch leaves status polling to the lifecycle", async (t) => {
  t.mock.timers.enable({ apis: ["setTimeout", "setInterval", "Date"] });
  const s = setup();
  try {
    s.mountTrainPage();
    await settle();
    s.mountWatch();
    const start = s.requests();
    for (let i = 0; i < 60; i++) {
      t.mock.timers.tick(1000);
      await settle();
    }
    assert.equal(
      s.requests() - start,
      20,
      "one status request per 3 s, not 30 a minute",
    );

    s.leaveTrainPage();
    const away = s.requests();
    for (let i = 0; i < 60; i++) {
      t.mock.timers.tick(1000);
      await settle();
    }
    assert.equal(
      s.requests() - away,
      10,
      "on another route the watch polls every 6 s",
    );

    s.mountTrainPage();
    await settle();
    s.setServer({ phase: "completed", job_id: "job-1" });
    for (let i = 0; i < 12; i++) {
      t.mock.timers.tick(1000);
      await settle();
    }
    assert.deepEqual(
      s.notifications,
      ["training-completed:job-1"],
      "completion notifies once",
    );
  } finally {
    s.unmountAll();
  }
});

test("the watch keeps polling a pending start the lifecycle does not poll", async (t) => {
  t.mock.timers.enable({ apis: ["setTimeout", "setInterval", "Date"] });
  const s = setup();
  s.setServer({ phase: "idle", job_id: "" });
  Object.assign(s.state, {
    phase: "idle",
    isTrainingRunning: false,
    isStarting: true,
    startRequestId: "req-1",
    jobId: null,
  });
  try {
    s.mountTrainPage();
    await settle();
    Object.assign(s.state, { phase: "idle", isTrainingRunning: false });
    s.mountWatch();
    const start = s.requests();
    for (let i = 0; i < 12; i++) {
      t.mock.timers.tick(1000);
      await settle();
    }
    assert.equal(s.requests() - start, 2, "the watch still polls every 6 s");
  } finally {
    s.unmountAll();
  }
});
