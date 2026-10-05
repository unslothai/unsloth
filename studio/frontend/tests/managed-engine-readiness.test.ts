// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

import assert from "node:assert/strict";
import test from "node:test";
import type { EngineStatus } from "../src/features/model-picker/api/engines.ts";
import { registerStoreStubResolver } from "./helpers/kit.ts";

registerStoreStubResolver();
const {
  isEngineReady,
  wslNoticeKey,
  convertsToInteger,
  precisionAfterEngineSwitch,
  supportsPrecision,
} = await import("../src/features/model-picker/api/engines.ts");

const installed: EngineStatus = {
  engine: "vllm",
  version: "new",
  installed_version: "old",
  installed: true,
  current: false,
  in_use: false,
  can_rollback: true,
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
  assert.equal(
    isEngineReady({ ...restored, unsupported_reason: "Unsupported GPU" }),
    false,
  );
  assert.equal(
    isEngineReady({ ...restored, job: { ...restored.job, state: "running" } }),
    false,
  );
});

test("failed or cancelled repairs leave the restored installation loadable", () => {
  for (const state of ["error", "cancelled"]) {
    assert.equal(
      isEngineReady({
        ...installed,
        restored: true,
        job: { ...installed.job, state },
      }),
      true,
    );
  }
});

test("Windows users are told about WSL2, the UAC prompt and a restart before installing", () => {
  assert.equal(wslNoticeKey(installed), null);
  assert.equal(wslNoticeKey({ ...installed, host: "local" }), null);
  assert.equal(
    wslNoticeKey({ ...installed, host: "wsl" }),
    "managedEngines.wslSetup",
  );
  assert.equal(
    wslNoticeKey({
      ...installed,
      host: "wsl",
      wsl: { state: "ready", distro: null },
    }),
    "managedEngines.wslSetup",
  );
  assert.equal(
    wslNoticeKey({
      ...installed,
      host: "wsl",
      wsl: { state: "restart_required", distro: null },
    }),
    "managedEngines.wslRestart",
  );
  assert.equal(
    wslNoticeKey({
      ...installed,
      host: "wsl",
      wsl: { state: "ready", distro: "UnslothStudio" },
    }),
    "managedEngines.wslReady",
  );
});

test("SGLang 0.5.18 and newer offer no load-time INT8 or 4-bit conversion", () => {
  assert.equal(convertsToInteger(installed), true);
  const sglang = { ...installed, engine: "sglang" as const };
  assert.equal(
    convertsToInteger({ ...sglang, installed_version: "0.5.17" }),
    true,
  );
  assert.equal(
    convertsToInteger({ ...sglang, installed_version: "0.5.20" }),
    false,
  );
  assert.equal(
    convertsToInteger({
      ...sglang,
      installed_version: null,
      version: "0.5.20",
    }),
    false,
  );
});

test("switching to an engine that cannot convert resets INT4 / INT8 before loading", () => {
  const sglang = {
    ...installed,
    engine: "sglang" as const,
    installed_version: "0.5.20",
  };
  assert.equal(precisionAfterEngineSwitch("int4", sglang), "auto");
  assert.equal(precisionAfterEngineSwitch("int8", sglang), "auto");
  assert.equal(precisionAfterEngineSwitch("fp8", sglang), "fp8");
  assert.equal(precisionAfterEngineSwitch("int4", installed), "int4");
  assert.equal(precisionAfterEngineSwitch("int8", undefined), "int8");
});

test("an AMD build offers only the precisions it loads, and a retained 4-bit resets", () => {
  const rocm = {
    ...installed,
    platform: "rocm" as const,
    precisions: ["auto", "bf16", "fp16", "int8", "fp8"],
  };
  assert.equal(supportsPrecision(rocm, "int4"), false);
  for (const precision of ["auto", "bf16", "fp16", "int8", "fp8"]) {
    assert.equal(supportsPrecision(rocm, precision), true);
  }
  assert.equal(precisionAfterEngineSwitch("int4", rocm), "auto");
  assert.equal(precisionAfterEngineSwitch("fp8", rocm), "fp8");
  // An older backend reports no list, so nothing beyond the SGLang rule is withheld.
  assert.equal(supportsPrecision(installed, "int4"), true);
});
