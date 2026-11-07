// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import {
  installLocalStorageFake,
  registerStoreStubResolver,
} from "./helpers/kit.ts";

registerStoreStubResolver();
installLocalStorageFake();

const { putModelOverride } = await import(
  "../src/features/model-picker/api/model-overrides.ts"
);
const { DEFAULT_PER_MODEL_CONFIG } = await import(
  "../src/features/model-picker/model-config/per-model-config.ts"
);
const { setAuthFetchHandler } = await import("./helpers/store-stubs/auth.ts");

const REPO = "unsloth/Qwen3-4B-GGUF";
const SNAPSHOT = "C:/hf/models--unsloth--Qwen3-4B-GGUF/snapshots/abc";

test("a forget and a later save under another alias reach the server in order", async () => {
  const arrived: string[] = [];
  const releases: (() => void)[] = [];
  setAuthFetchHandler(async (_input, init) => {
    const body = JSON.parse(String(init?.body ?? "{}"));
    // Hold each request until released, so arrival order is what the client chose.
    await new Promise<void>((resolve) => releases.push(resolve));
    arrived.push(
      body.remove ? `forget ${body.model_id}` : `save ${body.model_id}`,
    );
    return new Response(JSON.stringify({ overrides: {}, removed_keys: [] }), {
      status: 200,
      headers: { "content-type": "application/json" },
    });
  });
  try {
    const save = { ...DEFAULT_PER_MODEL_CONFIG, nParallel: 2 };
    const forget = putModelOverride(REPO, "Q4_K_M", null);
    const later = putModelOverride(SNAPSHOT, "Q4_K_M", save);
    await new Promise((resolve) => setTimeout(resolve, 0));
    // Only the forget may be on the wire; the alias save waits behind it.
    assert.equal(releases.length, 1);
    releases.shift()?.();
    await forget;
    await new Promise((resolve) => setTimeout(resolve, 0));
    releases.shift()?.();
    await later;
    assert.deepEqual(arrived, [
      `forget ${REPO}:Q4_K_M`,
      `save ${SNAPSHOT}:Q4_K_M`,
    ]);
  } finally {
    setAuthFetchHandler(null);
  }
});
