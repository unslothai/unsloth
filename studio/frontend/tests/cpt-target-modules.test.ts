// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import { registerBundlerResolver } from "./helpers/kit.ts";

registerBundlerResolver();

const {
  CPT_TARGET_MODULES,
  DEFAULT_HYPERPARAMS,
  getCptUiTargetModules,
  resolveCptTargetModules,
} = await import("../src/config/training.ts");

test("resolveCptTargetModules keeps all-linear for architecture-specific models", () => {
  assert.deepEqual(resolveCptTargetModules(["all-linear"]), [
    "all-linear",
    "embed_tokens",
    "lm_head",
  ]);
});

test("resolveCptTargetModules keeps Llama defaults for standard adapters", () => {
  assert.deepEqual(
    resolveCptTargetModules(DEFAULT_HYPERPARAMS.targetModules),
    CPT_TARGET_MODULES,
  );
});

test("getCptUiTargetModules exposes all-linear in the CPT settings UI", () => {
  assert.deepEqual(
    getCptUiTargetModules(["all-linear", "embed_tokens", "lm_head"]),
    ["all-linear", "embed_tokens", "lm_head"],
  );
  assert.deepEqual(getCptUiTargetModules(CPT_TARGET_MODULES), CPT_TARGET_MODULES);
});
