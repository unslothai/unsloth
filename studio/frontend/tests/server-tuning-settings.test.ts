// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// The four llama-server tuning controls in Run settings: Mmap/Mlock, the draft
// KV cache dtype, Checkpoints and Cache RAM. Normalization (what a stored blob
// can and cannot say), the load payload's omit-when-blank rule, and the
// extra-arguments diagnostics that name the control a typed flag duplicates.

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";
import { fileURLToPath } from "node:url";

import { registerBundlerResolver } from "./helpers/kit.ts";

registerBundlerResolver();

const {
  CACHE_RAM_MAX,
  CACHE_RAM_MIN,
  CTX_CHECKPOINTS_MAX,
  DEFAULT_PER_MODEL_CONFIG,
  LOAD_MODES,
  canonicalizeLoadMode,
  isDefaultConfig,
  normalizeCacheRam,
  normalizeCtxCheckpoints,
  normalizePerModelConfig,
} = await import(
  "../src/features/model-picker/model-config/per-model-config.ts"
);
const { loadedConfigSignature } = await import(
  "../src/features/model-picker/model-config/config-signature.ts"
);
const { diagnoseExtraArgs } = await import(
  "../src/features/model-picker/model-config/llama-extra-args.ts"
);
const {
  clearedServerTuningState,
  committedServerTuningState,
  serverTuningLoadPayload,
} = await import("../src/features/chat/lib/server-tuning-fields.ts");

test("every documented load mode is offered, and auto is the unset sentinel", () => {
  assert.deepEqual(
    [...LOAD_MODES],
    ["auto", "none", "mmap", "mlock", "mmap+mlock", "dio"],
  );
  // "auto" IS llama.cpp's default, so it is stored as null and never emitted.
  assert.equal(canonicalizeLoadMode("auto"), null);
  assert.equal(canonicalizeLoadMode(" MMAP+MLOCK "), "mmap+mlock");
  // Repaired spellings are refused, not guessed at: llama-server exits on one.
  assert.equal(canonicalizeLoadMode("mmap + mlock"), null);
  assert.equal(canonicalizeLoadMode("swap"), null);
  assert.equal(canonicalizeLoadMode(42), null);
});

test("checkpoints and cache RAM clamp instead of refusing", () => {
  assert.equal(normalizeCtxCheckpoints(0), 0);
  assert.equal(normalizeCtxCheckpoints(1e6), CTX_CHECKPOINTS_MAX);
  assert.equal(normalizeCtxCheckpoints(-5), 0);
  assert.equal(normalizeCtxCheckpoints(null), null);
  // -1 (no limit) and 0 (disabled) are values here, not "unset"
  assert.equal(normalizeCacheRam(-1), CACHE_RAM_MIN);
  assert.equal(normalizeCacheRam(0), 0);
  assert.equal(normalizeCacheRam(-99), CACHE_RAM_MIN);
  assert.equal(normalizeCacheRam(1e12), CACHE_RAM_MAX);
  assert.equal(normalizeCacheRam("2048"), null);
});

test("a stored draft cache dtype needs a mode that loads a separate drafter", () => {
  const kept = normalizePerModelConfig({
    speculativeType: "dspark",
    specDraftCacheDtype: "q8_0",
  });
  assert.equal(kept.specDraftCacheDtype, "q8_0");
  // ngram loads no draft model, so there is no draft context for it to apply to.
  const dropped = normalizePerModelConfig({
    speculativeType: "ngram",
    specDraftCacheDtype: "q8_0",
  });
  assert.equal(dropped.specDraftCacheDtype, null);
  // and a dtype llama.cpp has no cache for is dropped whatever the mode
  assert.equal(
    normalizePerModelConfig({
      speculativeType: "dflash",
      specDraftCacheDtype: "q3_k",
    }).specDraftCacheDtype,
    null,
  );
});

test("the four take part in the editor's identity", () => {
  // loadedConfigSignature keys the Run settings instance, so a field missing from
  // it leaves the panel showing saved values over a model running different ones,
  // and Apply then writes those back. (The reload comparison itself is swept by
  // resident-config-match-accelerator-matrix.test.ts.)
  const base = loadedConfigSignature(normalizePerModelConfig({}));
  for (const patch of [
    { loadMode: "dio" },
    { ctxCheckpoints: 8 },
    { cacheRam: 0 },
    { speculativeType: "dspark", specDraftCacheDtype: "q8_0" },
  ]) {
    assert.notEqual(
      loadedConfigSignature(normalizePerModelConfig(patch)),
      base,
      `${JSON.stringify(patch)} must read as a change`,
    );
  }
  assert.equal(loadedConfigSignature(normalizePerModelConfig({})), base);
});

test("a record only claims the new schema version when it carries one", () => {
  // toStoredConfig stamps the OLDEST version that understands every field
  // present, so an older client can still rewrite a record it fully knows.
  const source = readFileSync(
    fileURLToPath(
      new URL(
        "../src/features/model-picker/model-config/per-model-config.ts",
        import.meta.url,
      ),
    ),
    "utf8",
  );
  assert.match(source, /const STORAGE_SCHEMA_VERSION = 6;/);
  assert.match(source, /const PRE_MLX_SPECULATIVE_SCHEMA_VERSION = 5;/);
  assert.match(source, /const PRE_SERVER_TUNING_SCHEMA_VERSION = 4;/);
  assert.match(
    source,
    /hasServerTuning\s*\n?\s*\?\s*PRE_MLX_SPECULATIVE_SCHEMA_VERSION/,
  );
});

test("blank knobs are omitted from the load payload", () => {
  // A null counts as SET on the backend, which strips the matching flag out of
  // any inherited extra arguments. Blank means "no opinion", so it must not be
  // present at all.
  assert.deepEqual(
    serverTuningLoadPayload({
      loadMode: null,
      specDraftCacheDtype: null,
      ctxCheckpoints: null,
      cacheRam: null,
    }),
    {},
  );
  assert.deepEqual(
    serverTuningLoadPayload({
      loadMode: "dio",
      specDraftCacheDtype: "q8_0",
      ctxCheckpoints: 0,
      cacheRam: -1,
    }),
    {
      // biome-ignore lint/style/useNamingConvention: API schema
      load_mode: "dio",
      // biome-ignore lint/style/useNamingConvention: API schema
      spec_draft_cache_type: "q8_0",
      // biome-ignore lint/style/useNamingConvention: API schema
      ctx_checkpoints: 0,
      // biome-ignore lint/style/useNamingConvention: API schema
      cache_ram: -1,
    },
  );
});

test("a launch commits the click-time values, and diffusion commits none", () => {
  const values = { loadMode: "mmap", ctxCheckpoints: 8, cacheRam: 2048 };
  const committed = committedServerTuningState(values);
  assert.equal(committed.loadMode, "mmap");
  // control and baseline move together: the baseline is what the rollback resends
  assert.equal(committed.loadedLoadMode, "mmap");
  assert.equal(committed.loadedCtxCheckpoints, 8);
  // The diffusion runner launches no llama-server, so a value recorded against
  // it would be carried onto the next GGUF by a saved preset.
  assert.deepEqual(
    committedServerTuningState(values, true),
    clearedServerTuningState(),
  );
});

test("a typed flag is told which control it duplicates", () => {
  const named = (text: string) =>
    diagnoseExtraArgs(text, null, {})
      .map((entry) => entry.message)
      .join(" ");
  assert.match(named("--ctx-checkpoints 8"), /Checkpoints/);
  assert.match(named("-cram 2048"), /Cache RAM/);
  assert.match(named("--spec-draft-type-k q8_0"), /Spec Decoding KV Cache Dtype/);
  assert.match(named("--swa-checkpoints 4"), /Checkpoints/);
  // Not a denial: the extras are appended last, so the typed flag is what runs.
  assert.match(named("--ctx-checkpoints 8"), /wins/);
});

test("the load mode is reported as removed, not as winning, under Model Memory", () => {
  // apply_model_memory_policy runs before the extras reach the command line, so
  // saying a typed --load-mode wins would be false.
  const messages = diagnoseExtraArgs("--load-mode dio", null, {
    keepResident: true,
  }).map((entry) => entry.message);
  assert.ok(
    messages.some((message) => /removed/.test(message)),
    messages.join(" "),
  );
});

test("a config whose only change is one of the four is not read as default", () => {
  // savePerModelConfig DELETES an entry it judges default, so a tuning-only save
  // never reached storage: Run settings reported that defaults were kept and
  // unticked Remember, while the server row it had just mirrored held the value.
  for (const patch of [
    { loadMode: "mmap" },
    { specDraftCacheDtype: "q8_0", speculativeType: "dspark" },
    { ctxCheckpoints: 0 },
    { ctxCheckpoints: 64 },
    { cacheRam: 0 },
    { cacheRam: -1 },
  ]) {
    const config = normalizePerModelConfig({
      ...DEFAULT_PER_MODEL_CONFIG,
      ...patch,
    });
    assert.equal(isDefaultConfig(config), false, JSON.stringify(patch));
  }
  assert.equal(
    isDefaultConfig(normalizePerModelConfig({ ...DEFAULT_PER_MODEL_CONFIG })),
    true,
  );
});
