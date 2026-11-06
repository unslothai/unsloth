// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

/** Run settings "Load in 4-bit" for default-engine Hugging Face models (#5639). */

import assert from "node:assert/strict";
import test from "node:test";

import { readSrc, registerStoreStubResolver } from "./helpers/kit.ts";

registerStoreStubResolver();
const { normalizePerModelConfig, isDefaultConfig, DEFAULT_PER_MODEL_CONFIG } =
  await import("../src/features/model-picker/model-config/per-model-config.ts");
const { loadedConfigSignature } = await import(
  "../src/features/model-picker/model-config/config-signature.ts"
);
const { toApiOverride, fromApiOverride } = await import(
  "../src/features/model-picker/api/model-overrides.ts"
);
const { residentRuntimeMatchesConfig } = await import(
  "../src/features/chat/lib/resident-config-match.ts"
);

const standing = {
  speculativeType: "auto",
  gpuMemoryMode: "auto" as const,
  gpuLayers: -1,
  nCpuMoe: 0,
  reconcileGpuIds: (ids: number[] | null) => ids,
  resolveContextLength: (value: number | null) => value ?? 0,
  parallelSlots: 1,
  splitRatio: null,
  normalizeSpeculative: (value: string | null | undefined) => value ?? null,
};

test("4-bit stays the default, and only an explicit off is a setting", () => {
  assert.equal(normalizePerModelConfig({}).loadIn4bit, true);
  assert.equal(normalizePerModelConfig({ loadIn4bit: "no" }).loadIn4bit, true);
  assert.equal(isDefaultConfig(normalizePerModelConfig({})), true);
  const off = normalizePerModelConfig({ loadIn4bit: false });
  assert.equal(off.loadIn4bit, false);
  assert.equal(isDefaultConfig(off), false);
  assert.notEqual(
    loadedConfigSignature(off),
    loadedConfigSignature(DEFAULT_PER_MODEL_CONFIG),
  );
});

test("the server row carries only a 16-bit choice and reads it back", () => {
  assert.equal(
    "load_in_4bit" in toApiOverride(DEFAULT_PER_MODEL_CONFIG),
    false,
  );
  const row = toApiOverride({ ...DEFAULT_PER_MODEL_CONFIG, loadIn4bit: false });
  assert.equal(row.load_in_4bit, false);
  assert.equal(fromApiOverride(row).loadIn4bit, false);
  assert.equal(fromApiOverride({}).loadIn4bit, true);
});

test("a resident loaded at the other precision is reloaded, GGUF never is", () => {
  const sixteen = { ...DEFAULT_PER_MODEL_CONFIG, loadIn4bit: false };
  const four = DEFAULT_PER_MODEL_CONFIG;
  const residentFour = {
    engine: "auto" as const,
    is_gguf: false,
    load_in_4bit: true,
  };
  const residentSixteen = { ...residentFour, load_in_4bit: false };
  assert.equal(
    residentRuntimeMatchesConfig(residentFour, four, standing),
    true,
  );
  assert.equal(
    residentRuntimeMatchesConfig(residentSixteen, sixteen, standing),
    true,
  );
  assert.equal(
    residentRuntimeMatchesConfig(residentFour, sixteen, standing),
    false,
  );
  assert.equal(
    residentRuntimeMatchesConfig(residentSixteen, four, standing),
    false,
  );
  const gguf = { engine: "auto" as const, is_gguf: true, load_in_4bit: null };
  assert.equal(
    residentRuntimeMatchesConfig(gguf, sixteen, standing),
    residentRuntimeMatchesConfig(gguf, four, standing),
  );
});

test("every default-engine load path sends the choice, vLLM and SGLang never 4-bit", () => {
  const runtime = readSrc("features/chat/hooks/use-chat-model-runtime.ts");
  assert.match(
    runtime,
    /load_in_4bit:\s*loadEngineFields\.engine === "auto" && loadIn4bitChoice,/,
  );
  assert.match(runtime, /await validateModel\(\{\s*\.\.\.loadRequest\(\),/);
  const composer = readSrc("features/chat/shared-composer.tsx");
  assert.match(
    composer,
    /load_in_4bit: paneEngine === "auto" && ownConfig\.loadIn4bit !== false,/,
  );
  const adapter = readSrc("features/chat/api/chat-adapter.ts");
  assert.equal(
    (adapter.match(/load_in_4bit: config\.loadIn4bit !== false,/g) ?? [])
      .length,
    1,
  );
});

test("only a default-engine resident hydrates the switch, and an unload resets 16-bit", () => {
  const applier = readSrc(
    "features/chat/lib/apply-inference-status-to-store.ts",
  );
  const start = applier.indexOf("function residentEngineParams(");
  const body = applier.slice(
    applier.indexOf("return {", start),
    applier.indexOf("\n}\n", start),
  );
  const hydrate = new Function("status", body) as (
    status: Record<string, unknown>,
  ) => Record<string, unknown>;
  assert.equal(hydrate({ load_in_4bit: false }).loadIn4bit, false);
  assert.equal(hydrate({ load_in_4bit: true }).loadIn4bit, true);
  // A managed engine leaves the default-engine choice as the user left it.
  assert.equal(
    "loadIn4bit" in hydrate({ engine: "vllm", load_in_4bit: false }),
    false,
  );
  assert.equal(hydrate({ is_gguf: true, load_in_4bit: null }).loadIn4bit, true);

  const runtime = readSrc("features/chat/hooks/use-chat-model-runtime.ts");
  assert.match(
    runtime,
    /liveBeforeLoad\.params\.loadIn4bit === false &&\s*\(switchesModel \|\| !liveBeforeLoad\.params\.checkpoint\)/,
  );
});
