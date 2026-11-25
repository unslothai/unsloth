// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Settings > Engine versions and About read the llama.cpp build from the hardware info cache, so
// every action that swaps llama-server has to drop it, or they keep the previous install's version.

import assert from "node:assert/strict";
import { mock, test } from "node:test";
import * as lifecycle from "../src/lib/llama-job-lifecycle.ts";
import type * as HookModule from "../src/hooks/use-llama-update-check.ts";
import { readSrc } from "./helpers/kit.ts";
import { loadWithStubs } from "./helpers/module-stubs.ts";

function loadHook(jobState: string) {
  const calls: string[] = [];
  const react = {
    useState: (initial: unknown) => [initial, () => {}],
    useRef: (current: unknown) => ({ current }),
    useCallback: (fn: unknown) => fn,
    useEffect() {},
  };
  const job = (state: string) => ({
    state,
    operation: "update",
    started_at: "2026-11-25T10:00:00Z",
    finished_at: state === "running" ? null : "2026-11-25T10:01:00Z",
  });
  const authFetch = async (url: string) => ({
    ok: true,
    json: async () =>
      url === "/api/llama/update"
        ? { started: true, job: job("running") }
        : { update_available: false, job: job(jobState) },
  });
  const hook = loadWithStubs<typeof HookModule>(
    new URL("../src/hooks/use-llama-update-check.ts", import.meta.url),
    {
      react,
      "@/features/auth": { authFetch, getAuthToken: () => "token" },
      "@/features/model-picker/api/llama-flags": {
        invalidateLlamaFlagCatalog: () => calls.push("flags"),
      },
      "@/hooks/use-hardware-info": {
        refreshHardwareInfo: async () => calls.push("hardware"),
      },
      "@/lib/model-lifecycle-events": { notifyModelLifecycle() {} },
      "@/lib/llama-job-events": {
        signalRunningLlamaJob() {},
        subscribeToLlamaJobStarted: () => () => {},
      },
      "@/lib/llama-job-lifecycle": lifecycle,
    },
  );
  return { hook, calls };
}

const flush = () => new Promise((resolve) => setImmediate(resolve));

test("the shared refresh drops the flag catalog and re-reads the hardware info", () => {
  const { hook, calls } = loadHook("idle");
  hook.afterLlamaInstallChanged();
  assert.deepEqual(calls, ["flags", "hardware"]);
});

for (const state of ["success", "error"]) {
  test(`an update or switch job that ends in ${state} runs the shared refresh once`, async () => {
    mock.timers.enable({ apis: ["setInterval"] });
    try {
      const { hook, calls } = loadHook(state);
      const done = hook.useLlamaUpdateCheck().apply();
      await flush();
      mock.timers.tick(500);
      await done;
      assert.deepEqual(calls, ["flags", "hardware"]);
    } finally {
      mock.timers.reset();
    }
  });
}

test("saving or resetting the llama.cpp folder runs the shared refresh", () => {
  const section = readSrc(
    "features/settings/components/llama-backend-section.tsx",
  );
  const row = section.slice(section.indexOf("<LlamaCppPathRow"));
  assert.match(
    row,
    /^<LlamaCppPathRow\s+onChanged=\{\(\) => \{\s*afterLlamaInstallChanged\(\);\s*void refresh\(true\);\s*\}\}/,
  );
});
