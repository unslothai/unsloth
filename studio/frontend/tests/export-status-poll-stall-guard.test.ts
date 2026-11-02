// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import { loadWithStubs } from "./helpers/module-stubs.ts";

type ExportLifecycle = {
  useExportRuntimeLifecycle: () => void;
};

const STATUS_POLL_INTERVAL_MS = 5_000;
const IDLE_STATUS_POLL_INTERVAL_MS = 30_000;

/** A document stub whose `hidden` the test flips; dispatch() fires visibilitychange. */
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

function loadLifecycle(
  state: Record<string, unknown>,
  getExportStatus: () => Promise<unknown>,
): ExportLifecycle & { cleanup: () => void } {
  let cleanup: (() => void) | undefined;
  const lifecycle = loadWithStubs<ExportLifecycle>(
    new URL(
      "../src/features/export/hooks/use-export-runtime-lifecycle.ts",
      import.meta.url,
    ),
    {
      react: {
        useEffect: (effect: () => (() => void) | undefined) => {
          cleanup = effect();
        },
      },
      "@/features/auth": { hasAuthToken: () => true },
      "../api/export-api": {
        fetchExportLogs: async () => ({ entries: [] }),
        getExportStatus,
        streamExportLogs: async () => {},
      },
      "../stores/export-runtime-store": {
        useExportRuntimeStore: {
          getState: () => state,
          subscribe: () => () => {},
        },
      },
    },
  );
  return { ...lifecycle, cleanup: () => cleanup?.() };
}

const settle = () => new Promise<void>((resolve) => setImmediate(resolve));

test("status polling does not stack while the previous request is stalled", async (t) => {
  t.mock.timers.enable({ apis: ["setTimeout", "setInterval"] });
  const doc = stubDocument();
  let finishStatusRequest: (() => void) | undefined;
  let statusRequests = 0;
  const stalledStatusRequest = new Promise<void>((resolve) => {
    finishStatusRequest = resolve;
  });
  const state = {
    isExporting: false,
    backendActive: true,
    lastSeq: 0,
    applyBackendStatus: () => {},
    setConnected: () => {},
    appendLogs: () => {},
    appendLog: () => {},
  };
  const lifecycle = loadLifecycle(state, async () => {
    statusRequests += 1;
    await stalledStatusRequest;
    return {};
  });

  try {
    lifecycle.useExportRuntimeLifecycle();
    assert.equal(statusRequests, 1, "mount must start one status request");

    t.mock.timers.tick(STATUS_POLL_INTERVAL_MS * 3);
    doc.dispatch();
    assert.equal(
      statusRequests,
      1,
      "timer ticks must not start requests behind a stalled poll",
    );

    finishStatusRequest?.();
    await settle();
    t.mock.timers.tick(STATUS_POLL_INTERVAL_MS);
    assert.equal(
      statusRequests,
      2,
      "polling must resume after the request settles",
    );
  } finally {
    lifecycle.cleanup();
    finishStatusRequest?.();
  }
});

test("an idle app polls export status slowly and not at all while hidden", async (t) => {
  t.mock.timers.enable({ apis: ["setTimeout", "setInterval"] });
  const doc = stubDocument();
  let statusRequests = 0;
  const state = {
    isExporting: false,
    backendActive: false,
    lastSeq: 0,
    applyBackendStatus: () => {},
    setConnected: () => {},
    appendLogs: () => {},
    appendLog: () => {},
  };
  const lifecycle = loadLifecycle(state, async () => {
    statusRequests += 1;
    return {};
  });

  try {
    lifecycle.useExportRuntimeLifecycle();
    await settle();
    assert.equal(statusRequests, 1);

    t.mock.timers.tick(STATUS_POLL_INTERVAL_MS);
    await settle();
    assert.equal(statusRequests, 1, "idle must not poll at the active cadence");
    t.mock.timers.tick(IDLE_STATUS_POLL_INTERVAL_MS - STATUS_POLL_INTERVAL_MS);
    await settle();
    assert.equal(statusRequests, 2);

    doc.hidden = true;
    t.mock.timers.tick(IDLE_STATUS_POLL_INTERVAL_MS * 3);
    await settle();
    assert.equal(statusRequests, 2, "a hidden idle tab skips its polls");

    doc.hidden = false;
    doc.dispatch();
    await settle();
    assert.equal(statusRequests, 3, "returning to the tab polls at once");
  } finally {
    lifecycle.cleanup();
  }
});
