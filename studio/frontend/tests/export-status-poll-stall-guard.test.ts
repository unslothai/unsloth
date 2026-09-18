// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import { loadWithStubs } from "./helpers/module-stubs.ts";

type ExportLifecycle = {
  useExportRuntimeLifecycle: () => void;
};

const STATUS_POLL_INTERVAL_MS = 5_000;

test("status polling does not stack while the previous request is stalled", async (t) => {
  t.mock.timers.enable({ apis: ["setInterval"] });

  let cleanup: (() => void) | undefined;
  let finishStatusRequest: (() => void) | undefined;
  let statusRequests = 0;
  const stalledStatusRequest = new Promise<void>((resolve) => {
    finishStatusRequest = resolve;
  });
  const state = {
    isExporting: false,
    lastSeq: 0,
    applyBackendStatus: () => {},
    setConnected: () => {},
    appendLogs: () => {},
    appendLog: () => {},
  };

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
        getExportStatus: async () => {
          statusRequests += 1;
          await stalledStatusRequest;
          return {};
        },
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

  try {
    lifecycle.useExportRuntimeLifecycle();
    assert.equal(statusRequests, 1, "mount must start one status request");

    t.mock.timers.tick(STATUS_POLL_INTERVAL_MS * 3);
    assert.equal(
      statusRequests,
      1,
      "timer ticks must not start requests behind a stalled poll",
    );

    finishStatusRequest?.();
    await new Promise<void>((resolve) => setImmediate(resolve));
    t.mock.timers.tick(STATUS_POLL_INTERVAL_MS);
    assert.equal(
      statusRequests,
      2,
      "polling must resume after the request settles",
    );
  } finally {
    cleanup?.();
    finishStatusRequest?.();
  }
});
