// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";

import { loadWithStubs, stubJsxRuntime } from "./helpers/module-stubs.ts";

function stubDocument() {
  const target = new EventTarget();
  const doc = {
    hidden: false,
    addEventListener: target.addEventListener.bind(target),
    removeEventListener: target.removeEventListener.bind(target),
    dispatch: () => target.dispatchEvent(new Event("visibilitychange")),
  };
  (globalThis as { document?: unknown }).document = doc;
  return doc;
}

const settle = () => new Promise<void>((resolve) => setImmediate(resolve));

function stubReact() {
  const cleanups: (() => void)[] = [];
  return {
    cleanups,
    react: {
      useEffect: (effect: () => (() => void) | undefined) => {
        const cleanup = effect();
        if (cleanup) cleanups.push(cleanup);
      },
      useState: <T>(initial: T) => [initial, () => {}],
      useRef: <T>(initial: T) => ({ current: initial }),
    },
  };
}

test("training metrics skip hidden ticks, refetch on return, status keeps polling", async (t) => {
  t.mock.timers.enable({ apis: ["setTimeout", "setInterval", "Date"] });
  const doc = stubDocument();
  const { cleanups, react } = stubReact();
  let statusCalls = 0;
  let metricsCalls = 0;
  const state: Record<string, unknown> = {
    jobId: "job-1",
    phase: "training",
    isTrainingRunning: true,
    isStarting: false,
    hasHydrated: true,
    isHydrating: false,
    currentStep: 10,
    lastEventId: null,
    startError: null,
    stopRequested: false,
    applyStatus: () => {},
    applyMetrics: () => {},
    setHydrating: () => {},
    setHasHydrated: () => {},
    setSseConnected: () => {},
    setStartError: () => {},
  };
  const store = { getState: () => state, subscribe: () => () => {} };
  const lifecycle = loadWithStubs<{ useTrainingRuntimeLifecycle: () => void }>(
    new URL(
      "../src/features/training/hooks/use-training-runtime-lifecycle.ts",
      import.meta.url,
    ),
    {
      react,
      "@/features/auth": { hasAuthToken: () => true },
      "@/features/hub": { useHfTokenStore: store },
      "@/lib/native-notifications": {
        notifyNative: async () => {},
        safeNotificationLabel: (v: unknown) => v,
        sanitizeNotificationBody: (v: unknown) => v,
      },
      "../api/train-api": {
        getTrainingStatus: async () => {
          statusCalls += 1;
          return { start_request_state: "accepted" };
        },
        getTrainingMetrics: async () => {
          metricsCalls += 1;
          return {};
        },
        isAbortError: () => false,
        streamTrainingProgress: () => new Promise(() => {}),
      },
      "../lib/training-status-request": {
        isTrainingStatusRequestCurrent: () => true,
        trainingStatusRequestKey: () => "key",
      },
      "../lib/training-stream-scope": {
        createTrainingStreamScope: () => ({ jobId: "job-1" }),
        isTrainingProgressForScope: () => true,
        isTrainingStreamScopeCurrent: () => true,
      },
      "../stores/training-config-store": { useTrainingConfigStore: store },
      "../stores/training-runtime-store": { useTrainingRuntimeStore: store },
    },
  );

  try {
    lifecycle.useTrainingRuntimeLifecycle();
    await settle();
    assert.equal(statusCalls, 1, "hydrate polls status once");
    assert.equal(metricsCalls, 1, "hydrate polls metrics once");

    t.mock.timers.tick(15_000);
    await settle();
    assert.equal(metricsCalls, 4, "a visible tab polls metrics every 5 s");
    assert.equal(statusCalls, 6, "a visible tab polls status every 3 s");

    doc.hidden = true;
    doc.dispatch();
    t.mock.timers.tick(60_000);
    await settle();
    assert.equal(metricsCalls, 4, "a hidden tab skips the metrics history");
    assert.equal(statusCalls, 26, "status keeps polling while hidden");

    doc.hidden = false;
    doc.dispatch();
    await settle();
    assert.equal(
      metricsCalls,
      5,
      "returning to the tab fetches metrics at once",
    );

    doc.hidden = true;
    doc.dispatch();
    t.mock.timers.tick(1_000);
    doc.hidden = false;
    doc.dispatch();
    await settle();
    assert.equal(
      metricsCalls,
      5,
      "a quick tab switch with no skipped tick sends nothing extra",
    );
  } finally {
    for (const cleanup of cleanups) cleanup();
  }
});

function loadPollingHook<T>(path: string, react: unknown, urls: string[]) {
  return loadWithStubs<T>(new URL(path, import.meta.url), {
    react,
    "react/jsx-runtime": stubJsxRuntime(),
    "@/features/auth": {
      authFetch: async (url: string) => {
        urls.push(url);
        return { ok: true, json: async () => ({}) };
      },
    },
    "@/components/section-card": { SectionCard: () => null },
    "@/i18n": { useT: () => (key: string) => key },
    "@/lib/utils": { cn: () => "" },
    "@hugeicons/core-free-icons": {},
    "@hugeicons/react": { HugeiconsIcon: () => null },
    "./offload-panel-layout": {},
  });
}

for (const [name, path, exportName, url, intervalMs] of [
  [
    "offload",
    "../src/features/studio/sections/offload-panel.tsx",
    "useOffloadState",
    "/api/train/offload",
    2_000,
  ],
  [
    "hardware",
    "../src/hooks/use-gpu-utilization.ts",
    "useGpuUtilization",
    "/api/train/hardware",
    10_000,
  ],
] as const) {
  test(`${name} polling skips hidden ticks and refetches on return`, async (t) => {
    t.mock.timers.enable({ apis: ["setTimeout", "setInterval", "Date"] });
    const doc = stubDocument();
    const { cleanups, react } = stubReact();
    const urls: string[] = [];
    const mod = loadPollingHook<Record<string, (enabled: boolean) => unknown>>(
      path,
      react,
      urls,
    );
    try {
      mod[exportName](true);
      await settle();
      assert.deepEqual(urls, [url], "mount fetches once");

      t.mock.timers.tick(intervalMs * 2);
      await settle();
      assert.equal(urls.length, 3, "a visible tab keeps its cadence");

      doc.hidden = true;
      doc.dispatch();
      t.mock.timers.tick(60_000);
      await settle();
      assert.equal(urls.length, 3, "a hidden tab does not fetch");

      doc.hidden = false;
      doc.dispatch();
      await settle();
      assert.equal(urls.length, 4, "returning to the tab fetches at once");

      doc.hidden = true;
      doc.dispatch();
      t.mock.timers.tick(intervalMs / 4);
      doc.hidden = false;
      doc.dispatch();
      await settle();
      assert.equal(
        urls.length,
        4,
        "a quick tab switch with no skipped tick sends nothing extra",
      );

      for (const cleanup of cleanups) cleanup();
      doc.dispatch();
      t.mock.timers.tick(intervalMs * 3);
      await settle();
      assert.equal(
        urls.length,
        4,
        "unmount removes the timer and the listener",
      );
    } finally {
      for (const cleanup of cleanups) cleanup();
    }
  });
}

test("the runs grid gates its running poll on visibility like the sidebar", () => {
  const source = readFileSync(
    new URL("../src/features/studio/history-card-grid.tsx", import.meta.url),
    "utf8",
  );
  const start = source.indexOf("if (!hasRunningRun) return;");
  assert.ok(start > 0, "running poll effect not found");
  const effect = source.slice(
    start,
    source.indexOf("}, [hasRunningRun", start),
  );
  assert.match(
    effect,
    /setInterval\(\(\) => \{\s*if \(!document\.hidden\) void poll\(\);\s*\}, RUNNING_POLL_INTERVAL_MS\)/,
    "interval ticks must skip while hidden",
  );
  assert.match(
    effect,
    /addEventListener\("visibilitychange", onVisibilityChange\)/,
  );
  assert.match(
    effect,
    /removeEventListener\("visibilitychange", onVisibilityChange\)/,
  );
  assert.match(
    effect,
    /const onVisibilityChange = \(\) => \{\s*if \(\s*!document\.hidden &&\s*Date\.now\(\) - lastPollAt >= RUNNING_POLL_INTERVAL_MS\s*\) \{\s*void poll\(\);/,
    "returning to the tab must refetch once a tick was skipped",
  );
});
