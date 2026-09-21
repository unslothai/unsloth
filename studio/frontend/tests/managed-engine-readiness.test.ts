// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

import assert from "node:assert/strict";
import test from "node:test";
import type { EngineStatus } from "../src/features/model-picker/api/engines.ts";
import { registerStoreStubResolver } from "./helpers/kit.ts";

registerStoreStubResolver();
const { isEngineReady } = await import("../src/features/model-picker/api/engines.ts");

const installed: EngineStatus = {
  engine: "vllm", version: "new", installed_version: "old",
  installed: true, current: false, in_use: false, can_rollback: true,
  unsupported_reason: null,
  job: { state: "success", phase: "ready", message: "" },
};

test("an outdated engine requires an update unless explicitly restored", () => {
  assert.equal(isEngineReady(installed), false);
  assert.equal(isEngineReady({ ...installed, current: true }), true);
  assert.equal(isEngineReady({ ...installed, restored: true }), true);
  assert.equal(isEngineReady({ ...installed, restored: false }), false);
});

test("restoration does not bypass installation or platform requirements", () => {
  const restored = { ...installed, restored: true };
  assert.equal(isEngineReady(undefined), false);
  assert.equal(isEngineReady({ ...restored, installed: false }), false);
  assert.equal(isEngineReady({ ...restored, unsupported_reason: "Unsupported GPU" }), false);
  assert.equal(isEngineReady({ ...restored, job: { ...restored.job, state: "running" } }), false);
});

test("failed or cancelled repairs leave the restored installation loadable", () => {
  for (const state of ["error", "cancelled"]) {
    assert.equal(isEngineReady({
      ...installed, restored: true, job: { ...installed.job, state },
    }), true);
  }
});
