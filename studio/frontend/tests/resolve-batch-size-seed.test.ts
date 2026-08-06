// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import { registerBundlerResolver } from "./helpers/kit.ts";

registerBundlerResolver();

const { resolveBatchSizeSeed } = await import(
  "../src/features/chat/lib/resolve-batch-size-seed.ts"
);

function seed(
  incoming: number | null | undefined,
  value: number | null,
  loaded: number | null,
  overrides: { isGguf?: boolean; seedLoadParams?: boolean } = {},
) {
  return resolveBatchSizeSeed({
    incoming,
    isGguf: overrides.isGguf ?? true,
    previous: { value, loaded },
    seedLoadParams: overrides.seedLoadParams ?? true,
  });
}

test("a steady echo never touches the pair", () => {
  assert.deepEqual(seed(4096, 4096, 4096), {});
  assert.deepEqual(seed(null, null, null), {});
});

test("an in-flight load owns the params", () => {
  assert.deepEqual(seed(4096, null, null, { seedLoadParams: false }), {});
});

test("an older backend omitting the field says nothing", () => {
  assert.deepEqual(seed(undefined, 2048, 2048), {});
});

test("a blank control follows an external move off the defaults", () => {
  // the echo is the requested size, so leaving it blank would revert it on Apply
  assert.deepEqual(seed(4096, null, null), { loaded: 4096, value: 4096 });
});

test("a clean control follows a same-model move", () => {
  assert.deepEqual(seed(1024, 4096, 4096), { loaded: 1024, value: 1024 });
});

test("a pending edit keeps the control while the baseline advances", () => {
  assert.deepEqual(seed(1024, 8192, 4096), { loaded: 1024 });
  // typed against a server still at the defaults
  assert.deepEqual(seed(1024, 8192, null), { loaded: 1024 });
});

test("a null echo clears a clean control along with the baseline", () => {
  assert.deepEqual(seed(null, 4096, 4096), { loaded: null, value: null });
  // dirty control keeps its intent
  assert.deepEqual(seed(null, 8192, 4096), { loaded: null });
});

test("a non-gguf status clears like a null echo", () => {
  assert.deepEqual(seed(undefined, 4096, 4096, { isGguf: false }), {
    loaded: null,
    value: null,
  });
});
