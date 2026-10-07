// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { mock, test } from "node:test";
import * as lifecycle from "../src/lib/llama-job-lifecycle.ts";
import type * as HookModule from "../src/hooks/use-llama-update-check.ts";
import { loadWithStubs } from "./helpers/module-stubs.ts";

type Deferred = { resolve: (body: unknown) => void };

function harness() {
  const applying: boolean[] = [];
  const pending: Deferred[] = [];
  let stateIndex = 0;
  const react = {
    useState(initial: unknown) {
      const index = stateIndex++;
      return [
        initial,
        (value: unknown) => {
          if (index === 2) applying.push(value as boolean);
        },
      ];
    },
    useRef: (current: unknown) => ({ current }),
    useCallback: (fn: unknown) => fn,
    useEffect() {},
  };
  const authFetch = async (url: string) => {
    if (url === "/api/llama/update") {
      return {
        ok: true,
        json: async () => ({ started: true, job: job("running") }),
      };
    }
    const body = await new Promise((resolve) => pending.push({ resolve }));
    return { ok: true, json: async () => body };
  };
  const hookModule = loadWithStubs<typeof HookModule>(
    new URL("../src/hooks/use-llama-update-check.ts", import.meta.url),
    {
      react,
      "@/features/auth": { authFetch, getAuthToken: () => "token" },
      "@/hooks/use-hardware-info": { refreshHardwareInfo: async () => {} },
      "@/lib/llama-job-events": {
        signalRunningLlamaJob() {},
        subscribeToLlamaJobStarted: () => () => {},
      },
      "@/lib/llama-job-lifecycle": lifecycle,
    },
  );
  return { hook: hookModule.useLlamaUpdateCheck(), applying, pending };
}

function job(state: string) {
  return { state, operation: "update", started_at: "2026-08-18T13:02:21Z" };
}

const flush = () => new Promise((resolve) => setImmediate(resolve));

test("a poll that resolves after the job finished cannot re-pin the update toast", async () => {
  mock.timers.enable({ apis: ["setInterval"] });
  try {
    const { hook, applying, pending } = harness();
    const done = hook.apply();
    await flush();
    mock.timers.tick(500);
    mock.timers.tick(500);
    await flush();
    assert.equal(pending.length, 2);
    pending[1].resolve({ update_available: false, job: job("success") });
    assert.equal((await done).ok, true);
    pending[0].resolve({ update_available: true, job: job("running") });
    await flush();
    assert.equal(applying.at(-1), false);
  } finally {
    mock.timers.reset();
  }
});
