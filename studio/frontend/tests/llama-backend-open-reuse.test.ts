// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { mock, test } from "node:test";
import type * as ApiModule from "../src/features/settings/api/llama-backend.ts";
import type * as SwitchModule from "../src/features/settings/hooks/use-llama-backend-switch.ts";
import * as lifecycle from "../src/lib/llama-job-lifecycle.ts";
import { readSrc } from "./helpers/kit.ts";
import { loadWithStubs } from "./helpers/module-stubs.ts";

const flush = () => new Promise((resolve) => setImmediate(resolve));

function loadApi(urls: string[]) {
  return loadWithStubs<typeof ApiModule>(
    new URL("../src/features/settings/api/llama-backend.ts", import.meta.url),
    {
      "@/features/auth": {
        authFetch: async (url: string) => {
          urls.push(url);
          return { ok: true, json: async () => ({}) };
        },
      },
      "@/lib/format-fastapi-error": { readFastApiError: async () => "" },
      "./llama-backend-payload": {
        parseLlamaBackendStatus: (raw: unknown) => raw,
        parseLlamaBackendSwitchStarted: (raw: unknown) => raw,
      },
    },
  );
}

test("the status URL carries max_age_seconds only on a forced read that asks for it", async () => {
  const urls: string[] = [];
  const api = loadApi(urls);
  await api.loadLlamaBackendStatus();
  await api.loadLlamaBackendStatus(true);
  await api.loadLlamaBackendStatus(true, 120);
  await api.loadLlamaBackendStatus(false, 120);
  assert.deepEqual(urls, [
    "/api/llama/backend",
    "/api/llama/backend?force_refresh=true",
    "/api/llama/backend?force_refresh=true&max_age_seconds=120",
    "/api/llama/backend",
  ]);
});

function loadSwitchHook(loads: unknown[][]) {
  const switchJob = (state: string) => ({
    state,
    operation: "switch",
    startedAt: "2026-11-25T10:00:00Z",
    message: "",
    error: null,
  });
  let current = switchJob("idle");
  // The first useState is the loaded status, so apply() has a request to send.
  const states: unknown[] = [
    { backendRequest: "cuda", backend: "cpu", job: current },
  ];
  const react = {
    useState: (initial: unknown) => [
      states.length ? states.shift() : initial,
      () => {},
    ],
    useRef: (value: unknown) => ({ current: value }),
    useCallback: (fn: unknown) => fn,
    useEffect(effect: () => (() => void) | undefined) {
      effect();
    },
  };
  const mod = loadWithStubs<typeof SwitchModule>(
    new URL(
      "../src/features/settings/hooks/use-llama-backend-switch.ts",
      import.meta.url,
    ),
    {
      react,
      "@/hooks/use-llama-update-check": { afterLlamaInstallChanged() {} },
      "@/i18n": { useT: () => (key: string) => key },
      "@/lib/llama-job-events": {
        signalLlamaJobStarted() {},
        subscribeToLlamaJobStarted: () => () => {},
      },
      "@/lib/llama-job-lifecycle": lifecycle,
      "@/lib/toast": { toast: { success() {}, error() {} } },
      "../api/llama-backend": {
        loadLlamaBackendStatus: async (...args: unknown[]) => {
          loads.push(args);
          return { backendRequest: "cuda", backend: "cuda", job: current };
        },
        switchLlamaBackend: async () => {
          current = switchJob("running");
          return { started: true, job: current };
        },
      },
      "../lib/llama-backend-labels": { backendDisplayName: (b: string) => b },
    },
  );
  return {
    mod,
    finishJob() {
      current = switchJob("success");
    },
  };
}

test("only opening the picker accepts a recent resolve; the install poll stays forced", async () => {
  const previous = (globalThis as { window?: unknown }).window;
  (globalThis as { window?: unknown }).window = {
    addEventListener() {},
    removeEventListener() {},
    dispatchEvent() {},
  };
  mock.timers.enable({ apis: ["setTimeout", "setInterval"] });
  try {
    const loads: unknown[][] = [];
    const { mod, finishJob } = loadSwitchHook(loads);
    assert.ok(mod.OPEN_REUSE_SECONDS > 0 && mod.OPEN_REUSE_SECONDS <= 300);
    const hook = mod.useLlamaBackendSwitch();
    await flush();
    assert.deepEqual(loads, [[true, mod.OPEN_REUSE_SECONDS]]);

    loads.length = 0;
    hook.apply();
    await flush();
    finishJob();
    mock.timers.tick(700);
    await flush();
    assert.ok(loads.length > 0);
    for (const args of loads) {
      assert.equal(args[0], true);
      assert.equal(args[1], undefined);
    }

    loads.length = 0;
    await hook.refresh(true);
    assert.deepEqual(loads, [[true, undefined]]);
  } finally {
    mock.timers.reset();
    (globalThis as { window?: unknown }).window = previous;
  }
});

test("the folder-change refresh passes no reuse window", () => {
  const section = readSrc(
    "features/settings/components/llama-backend-section.tsx",
  );
  assert.equal(section.match(/refresh\(/g)?.length, 1);
  assert.match(section, /void refresh\(true\);/);
});
