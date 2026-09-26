// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

import assert from "node:assert/strict";
import test from "node:test";
import { readSrc } from "./helpers/kit.ts";

const composer = readSrc("features/chat/shared-composer.tsx");
const start = composer.indexOf("const paneEngine = ");
const end = composer.indexOf("let loadTrustRemoteCode", start);
assert.ok(start >= 0 && end > start);
const paneFields = new Function(
  "targetIsGguf",
  "ownConfig",
  "reconcilePersistedGpuIds",
  `${composer.slice(start, end)} return paneEngineFields;`,
) as (
  targetIsGguf: boolean,
  ownConfig: Record<string, unknown>,
  reconcile: (ids: number[] | null) => number[] | null,
) => Record<string, unknown>;
const reconcile = (ids: number[] | null) => ids;

test("a compare pane loads its remembered vLLM or SGLang settings", () => {
  const fields = paneFields(
    false,
    {
      engine: "sglang",
      enginePrecision: "int4",
      engineParallelism: "pipeline",
      selectedGpuIds: [1, 2],
    },
    reconcile,
  );
  assert.deepEqual(fields, {
    engine: "sglang",
    engine_precision: "int4",
    engine_parallelism: "pipeline",
    load_in_4bit: false,
    gpu_ids: [1, 2],
  });
  assert.equal(composer.match(/\.\.\.paneEngineFields,/g)?.length, 2);
});

test("a default or GGUF compare pane keeps the default backend and no GPU pick", () => {
  for (const [isGguf, config] of [
    [false, { selectedGpuIds: [1] }],
    [true, { engine: "vllm", selectedGpuIds: [1] }],
  ] as const) {
    const fields = paneFields(isGguf, config, reconcile);
    assert.equal(fields.engine, "auto");
    assert.equal(fields.load_in_4bit, true);
    assert.equal("gpu_ids" in fields, false);
  }
});
