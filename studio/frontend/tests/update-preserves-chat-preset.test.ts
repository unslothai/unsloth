// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Selecting a preset is persisted through a short debounce. A llama.cpp update
// restarts the backend before page-exit events flush it, leaving the other presets
// intact while the active selection falls back to Default.
// The desktop path is covered in tauri-update-schedule.test.ts, which already
// drives that hook.

import assert from "node:assert/strict";
import test from "node:test";

import { loadWithStubs } from "./helpers/module-stubs.ts";

type ApplyResult = { ok: boolean; error?: string };

function hookHarness(trace: string[]) {
  const hook = loadWithStubs<{
    useLlamaUpdateCheck: (options?: unknown) => {
      apply: () => Promise<ApplyResult>;
    };
  }>(new URL("../src/hooks/use-llama-update-check.ts", import.meta.url), {
    react: {
      useState: <T>(initial: T) => [initial, () => undefined],
      useRef: <T>(initial: T) => ({ current: initial }),
      useEffect: () => undefined,
      useCallback: <T>(fn: T) => fn,
    },
    "@/features/auth": {
      authFetch: async (path: string) => {
        trace.push(`fetch:${path}`);
        // Short-circuits apply() right after the flush, so the test never enters
        // the job poll.
        return { ok: false, status: 503 };
      },
      getAuthToken: () => "token",
    },
    "@/features/chat": {
      flushPendingChatSettings: async () => {
        trace.push("flush");
      },
    },
    "@/hooks/use-hardware-info": { refreshHardwareInfo: async () => undefined },
    "@/lib/llama-job-events": {
      signalRunningLlamaJob: () => undefined,
      subscribeToLlamaJobStarted: () => () => undefined,
    },
    "@/lib/llama-job-lifecycle": {
      llamaUpdateAdoptsRunningJob: () => false,
      llamaUpdatePresentation: () => ({}),
    },
  });
  return hook.useLlamaUpdateCheck();
}

test("a llama.cpp update flushes chat settings before it starts the job", async () => {
  const trace: string[] = [];
  const controller = hookHarness(trace);

  const result = await controller.apply();

  assert.equal(result.ok, false);
  const flush = trace.indexOf("flush");
  const start = trace.indexOf("fetch:/api/llama/update");
  assert.ok(flush >= 0, "the llama update path did not flush chat settings");
  assert.ok(start >= 0, "the llama update was never requested");
  assert.ok(
    flush < start,
    `the flush must precede the update request: ${trace.join(", ")}`,
  );
});
