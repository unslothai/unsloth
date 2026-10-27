// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";
import { readSrc, readText, registerBundlerResolver } from "./helpers/kit.ts";

registerBundlerResolver();

const { llamaServerLoadFields, validateRequestBody } = await import(
  "../src/features/chat/lib/load-request.ts"
);

function validateModelRequestFields(): string[] {
  const source = readText("../../backend/models/inference.py");
  const start = source.indexOf("class ValidateModelRequest(BaseModel):");
  assert.ok(start >= 0, "ValidateModelRequest moved");
  const body = source.slice(start, source.indexOf("\nclass ", start + 1));
  return [...body.matchAll(/^ {4}(\w+):/gm)].map((match) => match[1]);
}

function loadModelRequestFields(): Set<string> {
  const source = readSrc("features/chat/types/api.ts");
  const start = source.indexOf("export interface LoadModelRequest {");
  const body = source.slice(start, source.indexOf("\n}\n", start));
  return new Set(
    [...body.matchAll(/^ {2}(\w+)\??:/gm)].map((match) => match[1]),
  );
}

const VALIDATE_ONLY = new Set([
  "include_context_length",
  "include_chat_template",
]);
const RENAMED = new Map([["native_path_lease", "nativePathLease"]]);

test("every field /validate prices can ride the load request", () => {
  const loadFields = loadModelRequestFields();
  const missing = validateModelRequestFields().filter(
    (field) =>
      !VALIDATE_ONLY.has(field) && !loadFields.has(RENAMED.get(field) ?? field),
  );
  assert.deepEqual(missing, []);
});

test("/validate receives every load setting it prices, not a hand-kept subset", () => {
  const fields = validateModelRequestFields().filter(
    (field) =>
      !VALIDATE_ONLY.has(field) && !RENAMED.has(field) && field !== "hf_token",
  );
  const request = Object.fromEntries(
    fields.map((field) => [field, `set:${field}`]),
  );
  const body = validateRequestBody(
    { ...request, nativePathLease: "lease" } as never,
    "token",
  ) as Record<string, unknown>;
  for (const field of fields) assert.equal(body[field], `set:${field}`, field);
  for (const field of [
    "ctx_checkpoints",
    "spec_draft_cache_type",
    "tensor_split",
  ]) {
    assert.ok(fields.includes(field), field);
  }
  assert.equal(body.native_path_lease, "lease");
  assert.equal(body.hf_token, "token");
  assert.equal("nativePathLease" in JSON.parse(JSON.stringify(body)), false);
});

test("an unset non-nullable /validate field gets its default, not a null the route refuses", () => {
  const body = validateRequestBody(
    {
      model_path: "org/m",
      hf_token: null,
      max_seq_length: 0,
      load_in_4bit: true,
      is_lora: false,
      tensor_parallel: null,
      disable_vision: null,
    },
    null,
  ) as Record<string, unknown>;
  assert.deepEqual(
    {
      engine: body.engine,
      engine_precision: body.engine_precision,
      engine_parallelism: body.engine_parallelism,
      engine_kv_cache_dtype: body.engine_kv_cache_dtype,
      tensor_parallel: body.tensor_parallel,
      disable_vision: body.disable_vision,
      reasoning_budget: body.reasoning_budget,
      reasoning_budget_message: body.reasoning_budget_message,
    },
    {
      engine: "auto",
      engine_precision: "auto",
      engine_parallelism: "tensor",
      engine_kv_cache_dtype: "auto",
      tensor_parallel: false,
      disable_vision: false,
      reasoning_budget: -1,
      reasoning_budget_message: "",
    },
  );
});

const TUNED = {
  reasoningBudget: 512,
  reasoningBudgetMessage: "stop",
  nBatch: 1024,
  nUbatch: 256,
  ctxCheckpoints: 0,
  specDraftCacheDtype: "q8_0",
  loadMode: "mmap",
  cacheRam: 4096,
  llamaExtraArgs: ["--ctx-size", "8192"],
};

test("llama-server's knobs reach a GGUF llama-server and nothing else", () => {
  assert.deepEqual(
    llamaServerLoadFields(TUNED, { isGguf: true, isDiffusion: false }),
    {
      reasoning_budget: 512,
      reasoning_budget_message: "stop",
      n_batch: 1024,
      n_ubatch: 256,
      load_mode: "mmap",
      spec_draft_cache_type: "q8_0",
      ctx_checkpoints: 0,
      cache_ram: 4096,
      llama_extra_args: ["--ctx-size", "8192"],
    },
  );
  assert.deepEqual(
    llamaServerLoadFields(TUNED, { isGguf: true, isDiffusion: true }),
    {
      reasoning_budget: -1,
      reasoning_budget_message: "",
      n_batch: 1024,
      n_ubatch: 256,
    },
  );
  assert.deepEqual(
    llamaServerLoadFields(TUNED, { isGguf: false, isDiffusion: false }),
    {
      reasoning_budget: -1,
      reasoning_budget_message: "",
    },
  );
});

test("blank knobs are omitted, and pass-through args only once the config has read them", () => {
  const llama = { isGguf: true, isDiffusion: false };
  assert.deepEqual(
    llamaServerLoadFields({ nBatch: null, ctxCheckpoints: null }, llama),
    {
      reasoning_budget: -1,
      reasoning_budget_message: "",
    },
  );
  assert.equal("llama_extra_args" in llamaServerLoadFields({}, llama), false);
  assert.deepEqual(
    llamaServerLoadFields({ llamaExtraArgs: null }, llama).llama_extra_args,
    [],
  );
});

test("every chat load path sends one request to /validate and /load", () => {
  const runtime = readSrc("features/chat/hooks/use-chat-model-runtime.ts");
  assert.match(runtime, /await validateModel\(\{\s*\.\.\.loadRequest\(\),/);
  assert.match(
    runtime,
    /await validateModel\(loadRequest\(\), \{ signal: abortCtrl\.signal \}\)/,
  );
  assert.match(runtime, /await loadModel\(\{\s*\.\.\.loadRequest\(\),/);

  const composer = readSrc("features/chat/shared-composer.tsx");
  assert.equal(
    composer.match(/await validateModel\(panePayload\(\),/g)?.length,
    2,
  );
  assert.match(composer, /await loadModel\(\{\s*\.\.\.panePayload\(\),/);

  const adapter = readSrc("features/chat/api/chat-adapter.ts");
  assert.match(adapter, /validateModel\(request\)/);
  assert.match(
    adapter,
    /canAutoLoadRecordingFailures\(failureLabel, request\)/,
  );
  assert.match(adapter, /canAutoLoad\(request\)/);
  assert.equal(
    adapter.match(/await loadModel\(\{ \.\.\.request, alongside: true \}\)/g)
      ?.length,
    2,
  );
});

test("the llama-server knobs are spelled in one place", () => {
  for (const file of [
    "features/chat/shared-composer.tsx",
    "features/chat/api/chat-adapter.ts",
    "features/chat/api/chat-api.ts",
  ]) {
    const source = readSrc(file);
    assert.doesNotMatch(
      source,
      /llama_extra_args:|serverTuningLoadPayload\(|n_batch:/,
      file,
    );
  }
  // The runtime's one remaining spelling replays the previous server's own echo on a rollback.
  const runtime = readSrc("features/chat/hooks/use-chat-model-runtime.ts");
  assert.equal(runtime.match(/serverTuningLoadPayload\(/g)?.length, 1);
  assert.match(
    runtime,
    /\.\.\.serverTuningLoadPayload\(\{\s*loadMode: rollbackState\.loadedLoadMode,/,
  );
});
