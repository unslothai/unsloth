// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test, { mock } from "node:test";
import * as benchMath from "../src/features/benchmarks/lib/bench-math.ts";
import { loadWithStubs } from "./helpers/module-stubs.ts";

type Call = { url: string; body: Record<string, unknown> | null };

function harness() {
  const fetches: Call[] = [];
  const loads: Record<string, unknown>[] = [];
  const authFetch = async (url: string, init?: { body?: string }) => {
    fetches.push({ url, body: init?.body ? JSON.parse(init.body) : null });
    return new Response("{}", { status: 200 });
  };
  const status = {
    active_model: "org/model-GGUF",
    is_gguf: true,
    gguf_variant: "Q4_K_M",
  };
  const runner = loadWithStubs<{
    runBenchmark: (
      config: unknown,
      events: unknown,
      signal: AbortSignal,
    ) => Promise<{ outcomes: { state: string; reason?: string }[] }>;
  }>(
    new URL("../src/features/benchmarks/api/bench-runner.ts", import.meta.url),
    {
      "@/features/auth": { authFetch },
      "@/features/chat": {
        getInferenceStatus: async () => status,
        // Never settles on its own: only the runner's abort ends it, like a hung backend load.
        loadModel: (
          payload: Record<string, unknown>,
          opts: { signal: AbortSignal },
        ) => {
          loads.push(payload);
          return new Promise((_, reject) =>
            opts.signal.addEventListener("abort", () =>
              reject(opts.signal.reason),
            ),
          );
        },
        useChatRuntimeStore: {
          getState: () => ({ hydratePersistedSettings: async () => undefined }),
        },
      },
      "@/hooks/gpu-memory-display": {
        gpuMemoryDisplay: () => ({ usageGpu: null }),
      },
      "@/hooks/gpu-vram": { resolveGpuVramUsedGb: () => null },
      "./chat-base": {
        chatBaseLoad: () => ({
          model_path: "org/model-GGUF",
          llama_extra_args: [],
        }),
        chatSampling: () => ({ temperature: 0 }),
        chatSettings: () => [],
      },
      "../lib/bench-math": benchMath,
    },
  );
  const config = {
    sweep: "context",
    variants: [{ label: "ctx 4096", load: { max_seq_length: 4096 } }],
    baseline: null,
    promptSet: "",
    customPrompt: "",
    rotatePrompts: false,
    maxTokens: 16,
    warmup: 0,
    repetitions: 1,
    temperature: 0,
    seed: 1,
    restoreAfter: false,
  };
  const events = {
    onStart: () => undefined,
    onOutcome: () => undefined,
    onResult: () => undefined,
    onProgress: () => undefined,
  };
  return { runner, config, events, fetches, loads };
}

async function untilLoadStarts(loads: unknown[]) {
  for (let i = 0; i < 200 && loads.length === 0; i++) await Promise.resolve();
  assert.equal(loads.length, 1, "the variant load never started");
}

function cancelsOf(fetches: Call[]) {
  return fetches.filter((f) => f.url === "/api/inference/unload");
}

test("a variant load that hits the time limit is cancelled on the backend", async (t) => {
  (globalThis as { window?: unknown }).window = globalThis;
  mock.timers.enable({ apis: ["setTimeout"] });
  t.after(() => mock.timers.reset());
  const { runner, config, events, fetches, loads } = harness();
  const running = runner.runBenchmark(
    config,
    events,
    new AbortController().signal,
  );
  await untilLoadStarts(loads);
  mock.timers.tick(8 * 60_000 + 1);
  const run = await running;

  assert.equal(run.outcomes[0].state, "error");
  assert.match(run.outcomes[0].reason ?? "", /Loading took over 8 min/);
  const cancels = cancelsOf(fetches);
  assert.equal(cancels.length, 1);
  assert.ok(loads[0].load_request_id, "the load carries no request id");
  assert.deepEqual(cancels[0].body, {
    model_path: "org/model-GGUF",
    cancel_load_request_id: loads[0].load_request_id,
  });
});

test("stopping the sweep mid-load cancels that load on the backend", async (t) => {
  (globalThis as { window?: unknown }).window = globalThis;
  mock.timers.enable({ apis: ["setTimeout"] });
  t.after(() => mock.timers.reset());
  const { runner, config, events, fetches, loads } = harness();
  const ctl = new AbortController();
  const running = runner.runBenchmark(config, events, ctl.signal);
  await untilLoadStarts(loads);
  ctl.abort();
  const run = await running;

  assert.equal(run.outcomes[0].state, "cancelled");
  assert.deepEqual(
    cancelsOf(fetches).map((c) => c.body?.cancel_load_request_id),
    [loads[0].load_request_id],
  );
});
