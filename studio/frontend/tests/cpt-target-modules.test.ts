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
  isCptTargetModuleActive,
  resolveCptTargetModules,
  toggleCptTargetModule,
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

test("resolveCptTargetModules treats all-linear as an exclusive sentinel", () => {
  assert.deepEqual(
    resolveCptTargetModules(["all-linear", "embed_tokens", "lm_head"]),
    ["all-linear", "embed_tokens", "lm_head"],
  );
  assert.deepEqual(
    resolveCptTargetModules(["all-linear", "q_proj"]),
    CPT_TARGET_MODULES,
  );
});

test("CPT target controls switch all-linear without mixed state", () => {
  assert.deepEqual(getCptUiTargetModules(), [
    "all-linear",
    ...CPT_TARGET_MODULES,
  ]);
  assert.equal(
    isCptTargetModuleActive(
      ["all-linear", "embed_tokens", "lm_head"],
      "all-linear",
    ),
    true,
  );
  assert.equal(
    isCptTargetModuleActive(["all-linear", "q_proj"], "all-linear"),
    false,
  );
  assert.equal(
    isCptTargetModuleActive(["all-linear", "q_proj"], "q_proj"),
    true,
  );
  assert.deepEqual(
    toggleCptTargetModule(
      ["all-linear", "embed_tokens", "lm_head"],
      "all-linear",
    ),
    CPT_TARGET_MODULES,
  );
  assert.deepEqual(toggleCptTargetModule(CPT_TARGET_MODULES, "all-linear"), [
    "all-linear",
    "embed_tokens",
    "lm_head",
  ]);
  assert.deepEqual(
    toggleCptTargetModule(["all-linear", "embed_tokens", "lm_head"], "q_proj"),
    ["embed_tokens", "lm_head", "q_proj"],
  );
});

test("CPT embedding controls do not change the LoRA target mode", () => {
  assert.deepEqual(
    toggleCptTargetModule(
      ["all-linear", "embed_tokens", "lm_head"],
      "embed_tokens",
    ),
    ["all-linear", "lm_head"],
  );
  assert.deepEqual(
    toggleCptTargetModule(["all-linear", "lm_head"], "embed_tokens"),
    ["all-linear", "lm_head", "embed_tokens"],
  );
  assert.deepEqual(toggleCptTargetModule(["q_proj"], "all-linear"), [
    "all-linear",
  ]);
  assert.deepEqual(
    toggleCptTargetModule(["all-linear", "lm_head"], "all-linear"),
    [...DEFAULT_HYPERPARAMS.targetModules, "lm_head"],
  );
});
