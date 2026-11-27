// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { mock, test } from "node:test";
import type * as HookModule from "../src/hooks/use-llama-update-check.ts";
import * as lifecycle from "../src/lib/llama-job-lifecycle.ts";
import { loadWithStubs } from "./helpers/module-stubs.ts";

const STATUS_URL = "/api/llama/update-status";
const FORCED_URL = `${STATUS_URL}?force_refresh=true`;
const REUSE_MS = 5 * 60 * 1000;

type World = {
  requests: string[];
  failNext: number;
  surfaced: unknown[];
  load: () => typeof HookModule;
};

function world(): World {
  const w: World = {
    requests: [],
    failNext: 0,
    surfaced: [],
    load: () =>
      loadWithStubs<typeof HookModule>(
        new URL("../src/hooks/use-llama-update-check.ts", import.meta.url),
        {
          react: {
            useState: (initial: unknown) => [
              initial,
              (value: unknown) => {
                if (value && typeof value === "object" && "job" in value) {
                  w.surfaced.push(value);
                }
              },
            ],
            useRef: (current: unknown) => ({ current }),
            useCallback: (fn: unknown) => fn,
            useEffect(effect: () => (() => void) | undefined) {
              const cleanup = effect();
              if (cleanup) cleanups.push(cleanup);
            },
          },
          "@/features/auth": {
            authFetch: async (url: string) => {
              w.requests.push(url);
              await new Promise((resolve) => setImmediate(resolve));
              if (w.failNext > 0) {
                w.failNext -= 1;
                return { ok: false, json: async () => ({}) };
              }
              return {
                ok: true,
                json: async () => ({
                  supported: true,
                  update_available: true,
                  installed_tag: "b9493",
                  latest_tag: "b9518",
                  job: { state: "idle" },
                }),
              };
            },
            getAuthToken: () => "token",
          },
          "@/hooks/use-hardware-info": { refreshHardwareInfo: async () => {} },
          "@/lib/model-lifecycle-events": { notifyModelLifecycle() {} },
          "@/lib/llama-job-events": {
            signalRunningLlamaJob() {},
            subscribeToLlamaJobStarted: () => () => {},
          },
          "@/lib/llama-job-lifecycle": lifecycle,
        },
      ),
  };
  return w;
}

let cleanups: (() => void)[] = [];
function mount(module: typeof HookModule, options = {}) {
  cleanups = [];
  const hook = module.useLlamaUpdateCheck(options);
  const own = cleanups;
  return {
    hook,
    unmount: () => {
      for (const cleanup of own) cleanup();
    },
  };
}

const flush = async () => {
  for (let i = 0; i < 4; i++) await new Promise((r) => setImmediate(r));
};

async function advance(ms: number) {
  mock.timers.tick(ms);
  await flush();
}

function withBrowser(run: () => Promise<void>) {
  return async () => {
    const g = globalThis as Record<string, unknown>;
    const previous = { window: g.window, localStorage: g.localStorage };
    const store = new Map<string, string>();
    g.window = {
      addEventListener() {},
      removeEventListener() {},
      dispatchEvent() {},
    };
    g.localStorage = {
      getItem: (key: string) => store.get(key) ?? null,
      setItem: (key: string, value: string) => void store.set(key, value),
      removeItem: (key: string) => void store.delete(key),
    };
    mock.timers.enable({
      apis: ["setTimeout", "setInterval", "Date"],
      now: 1_800_000_000_000,
    });
    try {
      await run();
    } finally {
      mock.timers.reset();
      g.window = previous.window;
      g.localStorage = previous.localStorage;
    }
  };
}

test(
  "the corner card and the Audio page landing together share one forced read",
  withBrowser(async () => {
    const w = world();
    const module = w.load();
    mount(module);
    mount(module, { enabled: true });
    await advance(1000);
    assert.deepEqual(w.requests, [FORCED_URL]);
    assert.equal(w.surfaced.length, 2);
    assert.equal(w.surfaced[0], w.surfaced[1]);
  }),
);

test(
  "re-entering the Audio tab inside the reuse window reads the cached status",
  withBrowser(async () => {
    const w = world();
    const module = w.load();
    mount(module);
    await advance(1000);
    for (let i = 0; i < 3; i++) {
      const audio = mount(module, { enabled: true });
      await advance(30_000);
      audio.unmount();
    }
    assert.deepEqual(w.requests, [
      FORCED_URL,
      STATUS_URL,
      STATUS_URL,
      STATUS_URL,
    ]);
    await advance(REUSE_MS);
    mount(module, { enabled: true });
    await advance(1000);
    assert.equal(w.requests.at(-1), FORCED_URL);
  }),
);

test(
  "a reload inside the window reuses the forced read another page load made",
  withBrowser(async () => {
    const w = world();
    mount(w.load());
    await advance(1000);
    await advance(60_000);
    mount(w.load());
    await advance(1000);
    assert.deepEqual(w.requests, [FORCED_URL, STATUS_URL]);
  }),
);

test(
  "the hourly reminder still forces",
  withBrowser(async () => {
    const w = world();
    const module = w.load();
    mount(module);
    mount(module, { enabled: true });
    await advance(1000);
    await advance(60 * 60 * 1000);
    assert.deepEqual(w.requests, [FORCED_URL, FORCED_URL]);
  }),
);

test(
  "an explicit Check forces even right after an automatic check",
  withBrowser(async () => {
    const w = world();
    const module = w.load();
    mount(module);
    await advance(1000);
    const settings = mount(module, { passive: true });
    await flush();
    const next = await settings.hook.refresh();
    assert.deepEqual(w.requests, [FORCED_URL, STATUS_URL, FORCED_URL]);
    assert.equal(next?.latest_tag, "b9518");
  }),
);

test(
  "a failed forced read is not reused: the next automatic check forces",
  withBrowser(async () => {
    const w = world();
    const module = w.load();
    w.failNext = 1;
    mount(module);
    await advance(1000);
    mount(module, { enabled: true });
    await advance(1000);
    assert.deepEqual(w.requests, [FORCED_URL, FORCED_URL]);
    assert.equal(w.surfaced.length, 1);
  }),
);

test(
  "a recorded check from the future (clock moved back) does not suppress forcing",
  withBrowser(async () => {
    const w = world();
    (globalThis as unknown as { localStorage: Storage }).localStorage.setItem(
      "unsloth_llama_update_forced_at",
      String(Date.now() + 60_000),
    );
    mount(w.load());
    await advance(1000);
    assert.deepEqual(w.requests, [FORCED_URL]);
  }),
);
