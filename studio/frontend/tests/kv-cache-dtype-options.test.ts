// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import { registerBundlerResolver } from "./helpers/kit.ts";

registerBundlerResolver();
const { kvCacheDtypeOptions } = await import(
  "../src/features/model-picker/model-config/per-model-config.ts"
);

test("iq4_nl is offered only where llama.cpp runs its FlashAttention", () => {
  for (const backend of ["vulkan", "cpu"]) {
    assert.ok(kvCacheDtypeOptions(backend, null).includes("iq4_nl"));
  }
  for (const backend of ["cuda", "rocm", "metal", null]) {
    const options = kvCacheDtypeOptions(backend, null);
    assert.ok(!options.includes("iq4_nl"));
    assert.deepEqual(options, ["bf16", "q8_0", "q4_0", "q4_1", "q5_0", "q5_1", "f32"]);
  }
});

test("a selected iq4_nl stays listed so the trigger can show it", () => {
  assert.ok(kvCacheDtypeOptions("cuda", "iq4_nl").includes("iq4_nl"));
});
