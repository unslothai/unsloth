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

test("implicit layer placement tracks free memory and shares one probe across slider changes", async (t) => {
  t.mock.timers.enable({ apis: ["setTimeout"] });
  const refreshes: ((value: unknown) => void)[] = [];
  let previous: unknown[] = [];
  let cleanup: (() => void) | undefined;
  let calls = 0;
  const hook = loadWithStubs<typeof EstimateHook>(
    new URL("../src/features/model-picker/hooks/use-memory-estimate.ts", import.meta.url),
    {
      react: {
        useState: (state: unknown) => [state, () => {}],
        useRef: (current: unknown) => ({ current }),
        useEffect: (effect: () => () => void, deps: unknown[]) => {
          if (deps.some((dep, i) => dep !== previous[i])) {
            cleanup?.();
            previous = deps;
            cleanup = effect();
          }
        },
      },
      "../api/memory-estimate": { fetchMemoryEstimate: async () => { calls++; return {}; } },
      "@/hooks/use-system": { fetchSystemInfo: (opts: { force: boolean; gpuMaxAgeMs: number }) => {
        assert.ok(opts.force && opts.gpuMaxAgeMs < 10_000);
        return new Promise((resolve) => refreshes.push(resolve));
      } },
    },
    { relativePassthrough: true },
  );
  type Request = NonNullable<Parameters<typeof hook.useMemoryEstimate>[0]>;
  const render = (free: number | null, request: Partial<Request> = {}) => {
    const gpuDevices = [0, 1, 2].map((index) => ({
      index, name: `Card ${index}`, indexKind: "physical" as const, memoryTotalGb: 24,
      memoryFreeKnown: index !== 0 || free !== null,
      memoryFreeGb: index === 0 ? free ?? 0 : 10,
      pinnable: true, diffusionPinnable: true, sharedMemory: false,
    }));
    hook.useMemoryEstimate({
      modelPath: "model", gpuMemoryMode: "manual", gpuLayers: 10, selectedGpuIds: [1, 0], ...request,
    }, { gpuDevices });
    t.mock.timers.tick(250);
  };
  const settle = async () => {
    for (let i = 0; i < 8; i++) await Promise.resolve();
  };
  render(null);
  render(0);
  assert.equal(refreshes.length, 1);
  assert.equal(calls, 0);
  // A failed system probe still leaves the ordinary aggregate estimate available.
  refreshes[0](null);
  await settle();
  assert.equal(calls, 1);
  render(12);
  await settle();
  assert.equal(calls, 2);
  render(12);
  await settle();
  assert.equal(calls, 2);
  for (const fixed of [
    { tensorSplit: [2, 1] },
    { llamaExtraArgs: ["-ts", "2,1"] },
    { selectedGpuIds: [1, 2] },
    { gpuMemoryMode: "auto" },
    { gpuLayers: null },
  ]) {
    render(12, fixed);
    await settle();
    const prior: number = calls;
    render(8, fixed);
    await settle();
    assert.equal(calls, prior);
  }
  assert.equal(refreshes.length, 1);
  cleanup?.();
});
