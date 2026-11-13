// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test, { after } from "node:test";
import { createServer } from "vite";
import { installLocalStorageFake } from "./helpers/kit.ts";

installLocalStorageFake();
Object.assign(window.location, { href: "http://localhost/" });
const vite = await createServer({ server: { middlewareMode: true } });
after(() => vite.close());
const overrides = await vite.ssrLoadModule(
  "/src/features/model-picker/api/model-overrides.ts",
);

async function bodyOf(
  config: unknown,
  options?: Record<string, unknown>,
): Promise<Record<string, unknown>> {
  const bodies: Record<string, unknown>[] = [];
  const original = globalThis.fetch;
  globalThis.fetch = (async (_url, init) => {
    bodies.push(JSON.parse(String(init?.body)));
    return Response.json({ overrides: {}, removed_keys: [] });
  }) as typeof fetch;
  try {
    await overrides.putModelOverride("org/model", "Q4_K_M", config, options);
  } finally {
    globalThis.fetch = original;
  }
  assert.equal(bodies.length, 1);
  return bodies[0];
}

test("an eviction clears the engine group and MLX offloads but keeps the launch flags", async () => {
  const body = await bodyOf(null, { keepLaunchFlags: true });
  assert.equal(body.remove, false);
  assert.equal("llama_extra_args" in body, false);
  assert.equal(body.engine, "auto");
  assert.equal(body.engine_precision, "auto");
  assert.equal(body.engine_parallelism, "tensor");
  assert.equal(body.engine_kv_cache_dtype, "auto");
  assert.equal(body.engine_seed, null);
  assert.equal(body.engine_gpu_memory_utilization, null);
  assert.equal(body.mlx_ple_offload, null);
  assert.equal(body.mlx_expert_offload, null);
});

test("a forget still sends an explicit empty flag list and no fields", async () => {
  const body = await bodyOf(null);
  assert.equal(body.remove, true);
  assert.deepEqual(body.llama_extra_args, []);
  assert.equal("engine" in body, false);
});
