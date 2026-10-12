// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";
import type * as EstimateHook from "../src/features/model-picker/hooks/use-memory-estimate.ts";
import { loadWithStubs } from "./helpers/module-stubs.ts";

test("resident estimates wait for fresh probes and discard obsolete refreshes", async (t) => {
  t.mock.timers.enable({ apis: ["setTimeout"] });
  let state: unknown;
  const refs: { current: unknown }[] = [];
  let cursor = 0;
  let previousDeps: unknown[] = [];
  let cleanup: (() => void) | undefined;
  let commit: (() => void) | undefined;
  const probes: ((result: unknown) => void)[] = [];
  const estimate = { available: true, totalBytes: 10 * 1024 ** 3 };
  const hook = loadWithStubs<typeof EstimateHook>(
    new URL(
      "../src/features/model-picker/hooks/use-memory-estimate.ts",
      import.meta.url,
    ),
    {
      react: {
        useState: (initial: unknown) => {
          state ??= initial;
          return [
            state,
            (next: unknown) => {
              state = typeof next === "function" ? next(state) : next;
            },
          ];
        },
        useRef: (initial: unknown) => (refs[cursor++] ??= { current: initial }),
        useEffect: (
          effect: () => (() => void) | undefined,
          deps: unknown[],
        ) => {
          if (deps.some((dep, i) => dep !== previousDeps[i])) {
            commit = () => {
              cleanup?.();
              previousDeps = deps;
              cleanup = effect();
            };
          }
        },
      },
      "../api/memory-estimate": { fetchMemoryEstimate: async () => estimate },
      "@/hooks/use-system": {
        fetchSystemInfo: (options: unknown) => {
          assert.deepEqual(options, { refreshMemory: true });
          return new Promise((resolve) => probes.push(resolve));
        },
      },
    },
    { relativePassthrough: true },
  );
  const render = (nCtx: number | null, refreshMemory = true) => {
    cursor = 0;
    const shown = hook.useMemoryEstimate(
      nCtx === null ? null : { modelPath: "model", nCtx },
      { refreshMemory },
    );
    commit?.();
    commit = undefined;
    return shown;
  };
  const settle = async () => {
    for (let i = 0; i < 8; i++) await Promise.resolve();
  };
  render(1024);
  t.mock.timers.tick(250);
  await settle();
  assert.equal(render(1024).estimate, null);
  probes[0]({ memory_refreshed: true });
  await settle();
  assert.equal(render(1024).estimate, estimate);
  assert.equal(render(2048).stale, true);
  t.mock.timers.tick(250);
  await settle();
  assert.equal(render(null).estimate, null);
  probes[1]({ memory_refreshed: true });
  await settle();
  assert.equal(render(2048).estimate, null);
  t.mock.timers.tick(250);
  await settle();
  probes[2]({});
  await settle();
  assert.equal(render(2048).estimate, null);
  render(2048, false);
  t.mock.timers.tick(250);
  await settle();
  assert.equal(render(2048, false).estimate, estimate);
  assert.equal(probes.length, 3);
  assert.equal(render(2048).stale, true);
  cleanup?.();
});
