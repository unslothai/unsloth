// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// A tab opened after a load has no baseline; status publishes requested_llama_extra_args
// so a failed switch can restore the previous model's arguments.

import assert from "node:assert/strict";
import { test } from "node:test";

import { readSrc, registerBundlerResolver } from "./helpers/kit.ts";

registerBundlerResolver();
const { resolveLlamaExtraArgsSeed } = await import(
  "../src/features/chat/lib/resolve-llama-extra-args-seed.ts"
);

function seed(
  incoming: string[] | null | undefined,
  { model = false, gguf = true, seedLoadParams = true } = {},
) {
  return resolveLlamaExtraArgsSeed({
    incoming,
    isGguf: gguf,
    hydratingExistingModel: model,
    seedLoadParams,
  });
}

const APPLIER = readSrc("features/chat/lib/apply-inference-status-to-store.ts");
const RUNTIME = readSrc("features/chat/hooks/use-chat-model-runtime.ts");
const API_TYPES = readSrc("features/chat/types/api.ts");

test("the status type carries the running arguments", () => {
  assert.match(API_TYPES, /requested_llama_extra_args\?: string\[\] \| null;/);
});

test("the applier seeds the loaded baseline from the status echo", () => {
  assert.match(
    APPLIER,
    /resolveLlamaExtraArgsSeed\(\{\s*incoming: status\.requested_llama_extra_args,\s*isGguf: status\.is_gguf \?\? true,\s*hydratingExistingModel,\s*seedLoadParams,/,
  );
  assert.deepEqual(seed(["--numa", "distribute"]), {
    loadedLlamaExtraArgs: ["--numa", "distribute"],
  });
  assert.deepEqual(seed(["--numa"], { gguf: false }), {});
});

test("the CLI adoption path hydrates settings while it owns the load lease", () => {
  assert.match(
    APPLIER,
    /const seedLoadParams = options\.seedLoadParams \?\? !prevState\.modelLoading/,
  );
  assert.match(
    APPLIER,
    /seedLoadParams: options\?\.allowWhileModelLoading/,
  );
  assert.match(
    APPLIER,
    /!status\.active_model \|\|\s*\(status\.loading\?\.length \?\? 0\) > 0 \|\|\s*isSpeechOnlyStatus\(status\)/,
  );
});

test("an older backend that omits the field changes nothing for the same model", () => {
  // undefined means the server does not publish it; keep the first-hand baseline.
  assert.deepEqual(seed(undefined), {});
});

test("an older backend's model change drops the previous model's arguments", () => {
  assert.deepEqual(seed(undefined, { model: true }), { loadedLlamaExtraArgs: null });
});

test("the baseline follows a same-model reload from elsewhere", () => {
  // Other clients can reload with different args, so a pinned baseline would resurrect stale ones.
  assert.deepEqual(seed(null), { loadedLlamaExtraArgs: null });
  assert.deepEqual(seed(["--a"], { seedLoadParams: false }), {});
  assert.deepEqual(seed(undefined, { model: true, seedLoadParams: false }), {});
});

test("the rollback still resends that baseline explicitly", () => {
  assert.match(
    RUNTIME,
    /rollbackState\.loadedLlamaExtraArgs != null\s*\n?\s*\? \{ llama_extra_args: rollbackState\.loadedLlamaExtraArgs \}/,
  );
});

test("an explicit empty list is kept apart from an unknown one", () => {
  // Omitting the field makes /load inherit, so a known-empty list must be sent as [].
  assert.match(RUNTIME, /loadLlamaExtraArgs !== undefined\s*\n?\s*\? \(loadLlamaExtraArgs \?\? \[\]\)/);
  assert.deepEqual(seed([]), { loadedLlamaExtraArgs: [] });
});
