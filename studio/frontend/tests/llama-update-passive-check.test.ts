// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { mock, test } from "node:test";
import type * as HookModule from "../src/hooks/use-llama-update-check.ts";
import * as lifecycle from "../src/lib/llama-job-lifecycle.ts";
import { loadWithStubs } from "./helpers/module-stubs.ts";

const STATUS_URL = "/api/llama/update-status";
const FORCED_URL = `${STATUS_URL}?force_refresh=true`;

function mount(options: { passive?: boolean }) {
  const requests: string[] = [];
  const cleanups: (() => void)[] = [];
  const react = {
    useState: (initial: unknown) => [initial, () => {}],
    useRef: (current: unknown) => ({ current }),
    useCallback: (fn: unknown) => fn,
    useEffect(effect: () => (() => void) | undefined) {
      const cleanup = effect();
      if (cleanup) cleanups.push(cleanup);
    },
  };
  const authFetch = async (url: string) => {
    requests.push(url);
    return {
      ok: true,
      json: async () => ({ supported: true, job: { state: "idle" } }),
    };
  };
  const hookModule = loadWithStubs<typeof HookModule>(
    new URL("../src/hooks/use-llama-update-check.ts", import.meta.url),
    {
      react,
      "@/features/auth": { authFetch, getAuthToken: () => "token" },
      "@/hooks/use-hardware-info": { refreshHardwareInfo: async () => {} },
      "@/lib/model-lifecycle-events": { notifyModelLifecycle() {} },
      "@/lib/llama-job-events": {
        signalRunningLlamaJob() {},
        subscribeToLlamaJobStarted: () => () => {},
      },
      "@/lib/llama-job-lifecycle": lifecycle,
    },
  );
  const hook = hookModule.useLlamaUpdateCheck(options);
  return {
    hook,
    requests,
    unmount: () => {
      for (const cleanup of cleanups) cleanup();
    },
  };
}

const flush = () => new Promise((resolve) => setImmediate(resolve));

function withWindow(run: () => Promise<void>) {
  return async () => {
    const previous = (globalThis as { window?: unknown }).window;
    (globalThis as { window?: unknown }).window = {
      addEventListener() {},
      removeEventListener() {},
      dispatchEvent() {},
    };
    mock.timers.enable({ apis: ["setTimeout", "setInterval"] });
    try {
      await run();
    } finally {
      mock.timers.reset();
      (globalThis as { window?: unknown }).window = previous;
    }
  };
}

test(
  "a passive mount reads the cached status once and never forces a refresh",
  withWindow(async () => {
    const { hook, requests, unmount } = mount({ passive: true });
    await flush();
    assert.deepEqual(requests, [STATUS_URL]);

    mock.timers.tick(3 * 60 * 60 * 1000);
    await flush();
    assert.deepEqual(requests, [STATUS_URL]);

    await hook.refresh();
    assert.deepEqual(requests, [STATUS_URL, FORCED_URL]);
    unmount();
  }),
);

test(
  "the default mount still forces its first check and the hourly reminder",
  withWindow(async () => {
    const { requests, unmount } = mount({});
    await flush();
    assert.deepEqual(requests, []);

    mock.timers.tick(1000);
    await flush();
    assert.deepEqual(requests, [FORCED_URL]);

    mock.timers.tick(60 * 60 * 1000);
    await flush();
    assert.deepEqual(requests, [FORCED_URL, FORCED_URL]);
    unmount();
  }),
);

test("the Settings Engines section mounts the hook passively", async () => {
  const { readFileSync } = await import("node:fs");
  const section = readFileSync(
    new URL(
      "../src/features/settings/components/engine-versions-section.tsx",
      import.meta.url,
    ),
    "utf8",
  );
  assert.match(
    section,
    /useLlamaUpdateCheck\(\{\s*onReloadRequired: resyncInferenceStatusAfterServerModelChange,\s*passive: true,\s*\}\)/,
  );
});
