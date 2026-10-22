// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";
import { vulkanRetryDelayMs } from "../src/hooks/system-discovery.ts";
import type * as SystemHooks from "../src/hooks/use-system.ts";
import { loadWithStubs } from "./helpers/module-stubs.ts";

function stubDocument() {
  const target = new EventTarget();
  const doc = {
    hidden: false,
    addEventListener: target.addEventListener.bind(target),
    removeEventListener: target.removeEventListener.bind(target),
    dispatch: () => target.dispatchEvent(new Event("visibilitychange")),
  };
  const globals = globalThis as { document?: unknown; window?: unknown };
  globals.document = doc;
  globals.window = globalThis;
  return doc;
}

function loadSystem(body: unknown) {
  const requests: { url: string; init?: RequestInit }[] = [];
  let cleanup: (() => void) | undefined;
  const system = loadWithStubs<typeof SystemHooks>(
    new URL("../src/hooks/use-system.ts", import.meta.url),
    {
      react: {
        useState: (initial: unknown) => [initial, () => {}],
        useMemo: (factory: () => unknown) => factory(),
        useEffect: (effect: () => (() => void) | undefined) => {
          cleanup = effect();
        },
      },
      "@/features/auth": {
        authFetch: async (url: string, init?: RequestInit) => {
          requests.push({ url, init });
          return Response.json(body);
        },
      },
    },
    { relativePassthrough: true },
  );
  return { system, requests, cleanup: () => cleanup?.() };
}

const settle = async () => {
  for (let i = 0; i < 4; i++) {
    await new Promise<void>((resolve) => setImmediate(resolve));
  }
};

test("the system poll pauses in a hidden tab and resumes at once on return", async (t) => {
  t.mock.timers.enable({ apis: ["setTimeout"] });
  const doc = stubDocument();
  const { system, requests, cleanup } = loadSystem({
    memory: { available_gb: 8 },
  });
  try {
    system.useSystemInfo({ pollMs: 5000 });
    await settle();
    assert.equal(requests.length, 1);
    assert.ok(
      requests[0].init?.signal instanceof AbortSignal,
      "the fetch carries a timeout",
    );

    t.mock.timers.tick(5000);
    await settle();
    assert.equal(requests.length, 2);

    doc.hidden = true;
    t.mock.timers.tick(5000 * 4);
    await settle();
    assert.equal(requests.length, 2, "no polls while hidden");

    doc.hidden = false;
    doc.dispatch();
    await settle();
    assert.equal(requests.length, 3, "one poll on becoming visible");
    t.mock.timers.tick(5000);
    await settle();
    assert.equal(requests.length, 4, "and the loop runs again");
  } finally {
    cleanup();
  }
});

test("an unavailable Vulkan inventory is re-probed with backoff, not every 3 s", async (t) => {
  t.mock.timers.enable({ apis: ["setTimeout"] });
  const doc = stubDocument();
  const { system, requests } = loadSystem({
    memory: { available_gb: 8 },
    inference_gpu: { backend: "vulkan", available: false, devices: [] },
  });
  const unsubscribe = system.subscribeSystemInfo(() => {}, {
    retryUnavailableVulkan: true,
  });
  try {
    await system.fetchSystemInfo();
    await settle();
    assert.equal(requests.length, 1);
    for (const [index, delay] of [3000, 6000, 12000].entries()) {
      t.mock.timers.tick(delay - 1);
      await settle();
      assert.equal(requests.length, index + 1, `not before ${delay} ms`);
      t.mock.timers.tick(1);
      await settle();
      assert.equal(requests.length, index + 2);
    }

    doc.hidden = true;
    t.mock.timers.tick(60_000);
    await settle();
    assert.equal(requests.length, 4, "a hidden tab defers its retry");
    doc.hidden = false;
    doc.dispatch();
    await settle();
    assert.equal(requests.length, 5);
  } finally {
    unsubscribe();
  }
});

test("Vulkan retry delay doubles from 3 s to a 60 s cap", () => {
  assert.deepEqual(
    [0, 1, 2, 3, 4, 5, 10].map(vulkanRetryDelayMs),
    [3000, 6000, 12000, 24000, 48000, 60000, 60000],
  );
});
