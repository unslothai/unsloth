// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// #18002: when the last used model fails to auto-load, the stand-in must say so and must not
// replace the remembered model, or the next Send loads the stand-in again.

import assert from "node:assert/strict";
import test from "node:test";

import { readSrc, registerBundlerResolver } from "./helpers/kit.ts";

registerBundlerResolver();

const { rememberedMissNote, sameVariant, withRememberedMiss } = await import(
  "../src/features/chat/utils/auto-load-remembered.ts"
);
const { loadFallbackNotice } = await import(
  "../src/features/chat/utils/mmproj-fallback.ts"
);

const source = readSrc("features/chat/api/chat-adapter.ts");

const CANDIDATE_RECORD =
  /if \(!\(loadResp\.is_lora \?\? false\) && missedNote === null\) \{\s*recordLastLocalModelLoad\(\{\s*id: candidate\.id,/;
const DEFAULT_RECORD =
  /if \(rememberedMiss\.current === null\) \{\s*recordLastLocalModelLoad\(\{\s*id: DEFAULT_CHAT_MODEL_REPO,/;
const PASSES_MISS =
  /loadAutoLoadCandidate\(\s*candidate,\s*isRememberedCandidate \? null : rememberedMiss\.current,\s*\)/;
const NOTES_ON_FAILURE =
  /return \{ loaded: true, blockedByTrustRemoteCode: false \};\s*\}\s*noteRememberedMiss\(\);\s*\} catch \{\s*options\?\.abortSignal\?\.throwIfAborted\(\);\s*noteRememberedMiss\(\);/;
const RECORD_CALLS = /recordLastLocalModelLoad\(\{/g;

test("a fallback toast leads with why the last used model was skipped and warns", () => {
  const plain = loadFallbackNotice(
    "Loaded unsloth/SmolLM2-135M-Instruct-GGUF (Q4_K_M)",
    null,
    null,
  );
  assert.equal(plain.degraded, false);
  const note = rememberedMissNote(
    "unsloth/gemma-4-26B-A4B-it-GGUF (UD-Q4_K_XL)",
    "llama-server failed to start",
  );
  const notice = withRememberedMiss(plain, note);
  assert.equal(notice.degraded, true);
  assert.equal(notice.title, plain.title);
  assert.equal(
    notice.description,
    "Your last used model unsloth/gemma-4-26B-A4B-it-GGUF (UD-Q4_K_XL) did not load: llama-server failed to start.",
  );
});

test("the miss note keeps the other notices and is inert without a miss", () => {
  const cpu = loadFallbackNotice("Loaded x", "vulkan_startup_crash", null);
  const combined = withRememberedMiss(cpu, rememberedMissNote("big", null));
  assert.ok(
    combined.description?.startsWith("Your last used model big did not load. "),
  );
  assert.ok(combined.description?.endsWith(cpu.description ?? "<none>"));
  assert.deepEqual(withRememberedMiss(cpu, null), cpu);
  assert.equal(
    rememberedMissNote("big", "Out of memory."),
    "Your last used model big did not load: Out of memory.",
  );
});

test("the miss note drops the llama-server log tail and stays short", () => {
  const detail =
    "Failed to load model: llama-server failed to start. Check that the GGUF file is valid and you have enough memory.\n\nllama-server output:\n0.00.428.167 E gguf_init_from_reader: failed to read key-value pairs\nFull log: /home/u/.unsloth/studio/logs/llama.log";
  assert.equal(
    rememberedMissNote("unsloth/Qwen3-0.6B-GGUF (Q4_K_M)", detail),
    "Your last used model unsloth/Qwen3-0.6B-GGUF (Q4_K_M) did not load: Failed to load model: llama-server failed to start. Check that the GGUF file is valid and you have enough memory.",
  );
  const long = rememberedMissNote("m", "x".repeat(500));
  assert.ok(long.endsWith("…"));
  assert.ok(long.length < 260);
});

test("quant names match the way the remembered quant is picked", () => {
  assert.equal(sameVariant("UD-Q4_K_XL", "ud-q4_k_xl"), true);
  assert.equal(sameVariant(null, null), true);
  assert.equal(sameVariant("Q4_K_M", "Q8_0"), false);
  assert.equal(sameVariant(null, "Q4_K_M"), false);
});

test("a stand-in load never overwrites the remembered model", () => {
  assert.equal(
    source.match(RECORD_CALLS)?.length,
    2,
    "every auto-load record site must be gated",
  );
  assert.match(source, CANDIDATE_RECORD);
  assert.match(source, DEFAULT_RECORD);
  assert.match(source, PASSES_MISS);
  assert.match(source, NOTES_ON_FAILURE);
});
