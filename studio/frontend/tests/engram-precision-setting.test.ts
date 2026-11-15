// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// engramPrecision: null loads the quant's own Engram table, a precision swaps in the repo's
// standalone table. GGUF llama-server loads only.

import assert from "node:assert/strict";
import test from "node:test";

import {
  installLocalStorageFake,
  readSrc,
  registerStoreStubResolver,
} from "./helpers/kit.ts";

registerStoreStubResolver();
const { store } = installLocalStorageFake();

const { fromApiOverride, toApiOverride } = await import(
  "../src/features/model-picker/api/model-overrides.ts"
);
const {
  DEFAULT_PER_MODEL_CONFIG,
  isDefaultConfig,
  normalizeEngramPrecision,
  normalizePerModelConfig,
  resolveInitialConfig,
  sameEngramPrecision,
  savePerModelConfig,
} = await import(
  "../src/features/model-picker/model-config/per-model-config.ts"
);
const { perModelConfigsEqual } = await import(
  "../src/features/model-picker/model-config/apply-per-model-config.ts"
);
const { loadedConfigSignature } = await import(
  "../src/features/model-picker/model-config/config-signature.ts"
);
const { llamaServerLoadFields } = await import(
  "../src/features/chat/lib/load-request.ts"
);
const { clearedServerTuningState, committedServerTuningState } = await import(
  "../src/features/chat/lib/server-tuning-fields.ts"
);
const { createRunConfigLink, parseRunConfigLink } = await import(
  "./helpers/sharing-links.ts"
);

const MODEL = "unsloth/Model-GGUF";
const VARIANT = "UD-Q4_K_XL";

function storedVersion(): number {
  const map = JSON.parse(store.get("unsloth_model_configs") ?? "{}") as Record<
    string,
    { version: number }
  >;
  const entries = Object.values(map);
  assert.equal(entries.length, 1);
  return entries[0].version;
}

test("normalize keeps a precision label and drops blank or malformed ones", () => {
  assert.equal(normalizeEngramPrecision(" Q8_0 "), "Q8_0");
  assert.equal(normalizeEngramPrecision("mxfp8"), "mxfp8");
  for (const value of [
    "",
    "  ",
    "Q8-0",
    "../x",
    "a".repeat(33),
    8,
    true,
    null,
    undefined,
  ]) {
    assert.equal(normalizeEngramPrecision(value), null, String(value));
  }
  assert.equal(normalizePerModelConfig({}).engramPrecision, null);
  assert.equal(
    normalizePerModelConfig({ engramPrecision: "BF16" }).engramPrecision,
    "BF16",
  );
});

test("precisions compare case-insensitively, null only with null", () => {
  assert.equal(sameEngramPrecision("q8_0", "Q8_0"), true);
  assert.equal(sameEngramPrecision(null, undefined), true);
  assert.equal(sameEngramPrecision(null, ""), true);
  assert.equal(sameEngramPrecision("Q8_0", null), false);
  assert.equal(sameEngramPrecision("Q8_0", "BF16"), false);
});

test("a set precision is not the default, stamps v15, and round trips through storage", () => {
  const swapped = { ...DEFAULT_PER_MODEL_CONFIG, engramPrecision: "Q8_0" };
  assert.equal(isDefaultConfig(swapped), false);
  assert.equal(
    isDefaultConfig({ ...DEFAULT_PER_MODEL_CONFIG, engramPrecision: "  " }),
    true,
  );

  store.clear();
  store.set("unsloth_model_configs_migrated", "1");
  assert.ok(savePerModelConfig(MODEL, VARIANT, swapped));
  assert.equal(storedVersion(), 15);
  assert.equal(
    resolveInitialConfig(MODEL, VARIANT).config.engramPrecision,
    "Q8_0",
  );

  // Only a set precision raises the version: older clients keep rewriting everything else.
  store.clear();
  store.set("unsloth_model_configs_migrated", "1");
  assert.ok(
    savePerModelConfig(MODEL, VARIANT, {
      ...DEFAULT_PER_MODEL_CONFIG,
      mlxFastVerify: true,
    }),
  );
  assert.equal(storedVersion(), 14);
  assert.ok(
    savePerModelConfig(MODEL, VARIANT, {
      ...DEFAULT_PER_MODEL_CONFIG,
      loadMode: "mmap",
    }),
  );
  assert.equal(storedVersion(), 5);
});

test("the server row always carries the precision, null clearing it", () => {
  const swapped = { ...DEFAULT_PER_MODEL_CONFIG, engramPrecision: "MXFP8" };
  assert.equal(toApiOverride(swapped).engram_precision, "MXFP8");
  const cleared = toApiOverride(DEFAULT_PER_MODEL_CONFIG);
  assert.equal("engram_precision" in cleared, true);
  assert.equal(cleared.engram_precision, null);
  assert.equal(
    fromApiOverride({ engram_precision: "BF16" }).engramPrecision,
    "BF16",
  );
  assert.equal(fromApiOverride({}, swapped).engramPrecision, "MXFP8");
  assert.equal(fromApiOverride({}).engramPrecision, null);
});

test("every override save says it mirrors the precision", () => {
  const source = readSrc("features/model-picker/api/model-overrides.ts");
  const send = source.slice(
    source.indexOf("async function sendModelOverride("),
  );
  assert.match(send, /mirrors_engram_precision: true/);
});

test("only a GGUF llama-server load sends a set precision", () => {
  const settings = { engramPrecision: "Q8_0" };
  assert.equal(
    llamaServerLoadFields(settings, { isGguf: true, isDiffusion: false })
      .engram_precision,
    "Q8_0",
  );
  for (const target of [
    { isGguf: true, isDiffusion: true },
    { isGguf: false, isDiffusion: false },
  ]) {
    assert.equal(
      "engram_precision" in llamaServerLoadFields(settings, target),
      false,
    );
  }
  const blank = llamaServerLoadFields(
    { engramPrecision: null },
    { isGguf: true, isDiffusion: false },
  );
  assert.equal("engram_precision" in blank, false);
});

test("a load commits the precision as its baseline; diffusion and a clear drop it", () => {
  const committed = committedServerTuningState({ engramPrecision: "BF16" });
  assert.deepEqual(
    [committed.engramPrecision, committed.loadedEngramPrecision],
    ["BF16", "BF16"],
  );
  const diffusion = committedServerTuningState(
    { engramPrecision: "BF16" },
    true,
  );
  assert.deepEqual(
    [diffusion.engramPrecision, diffusion.loadedEngramPrecision],
    [null, null],
  );
  const cleared = clearedServerTuningState();
  assert.deepEqual(
    [cleared.engramPrecision, cleared.loadedEngramPrecision],
    [null, null],
  );
});

test("a changed precision changes the editor identity and the draft comparison", () => {
  const base = { ...DEFAULT_PER_MODEL_CONFIG };
  const swapped = { ...base, engramPrecision: "Q8_0" };
  assert.equal(perModelConfigsEqual(base, swapped), false);
  assert.equal(
    perModelConfigsEqual(swapped, { ...base, engramPrecision: "q8_0" }),
    true,
  );
  assert.notEqual(loadedConfigSignature(base), loadedConfigSignature(swapped));
});

test("a shared link carries the precision, and a shared null is not a precision named null", () => {
  const link = createRunConfigLink({ config: { engramPrecision: "Q8_0" } });
  const parsed = parseRunConfigLink(link);
  assert.equal(parsed.kind, "valid");
  assert.equal(
    parsed.kind === "valid" && parsed.value.config.engramPrecision,
    "Q8_0",
  );

  const nullLink = createRunConfigLink({ config: { engramPrecision: null } });
  const parsedNull = parseRunConfigLink(nullLink);
  assert.equal(parsedNull.kind, "valid");
  assert.equal(
    parsedNull.kind === "valid" && parsedNull.value.config.engramPrecision,
    null,
  );

  const bad = parseRunConfigLink(
    nullLink.replace("engramPrecision=null", "engramPrecision=Q8-0"),
  );
  assert.equal(bad.kind, "invalid");
});
