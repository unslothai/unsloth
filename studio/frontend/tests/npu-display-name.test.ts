// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";
import type * as NpuApi from "../src/features/npu/api.ts";
import * as displayName from "../src/features/npu/display-name.ts";
import * as formatFastApiError from "../src/lib/format-fastapi-error.ts";
import * as sseJsonEvents from "../src/lib/sse-json-events.ts";
import { loadWithStubs } from "./helpers/module-stubs.ts";

const { npuDisplayName, npuDisplayNameOf, npuRepoLabel } = displayName;

const qwen = {
  id: "qwen3.6-moe-35b-a3b-FLM",
  model_path: "lemonade:qwen3.6-moe-35b-a3b-FLM",
  checkpoint: "qwen3.6-moe:35b-a3b",
  hf_repo: "FastFlowLM/Qwen3.6-35B-A3B-NPU2",
};

test("an NPU model is named by its Hugging Face repo, -NPU2 included", () => {
  assert.equal(npuDisplayName(qwen), "Qwen3.6-35B-A3B-NPU2");
  assert.equal(npuRepoLabel(qwen), "FastFlowLM/Qwen3.6-35B-A3B-NPU2");
});

test("an NPU model with no repo keeps its Lemonade id", () => {
  for (const hf_repo of [undefined, null, "", "  "]) {
    const model = { id: "gemma3-4b-FLM", hf_repo };
    assert.equal(npuDisplayName(model), "gemma3-4b-FLM");
    assert.equal(npuRepoLabel(model), "gemma3-4b-FLM");
  }
});

test("a RyzenAI repo shows -NPU in place of its -onnx-ryzenai-npu suffix", () => {
  const llama = {
    id: "ryzenai-Llama-3.2-1B-Instruct-onnx-ryzenai-npu",
    hf_repo: "amd/Llama-3.2-1B-Instruct-onnx-ryzenai-npu",
  };
  assert.equal(npuDisplayName(llama), "Llama-3.2-1B-Instruct-NPU");
  assert.equal(
    npuRepoLabel(llama),
    "amd/Llama-3.2-1B-Instruct-onnx-ryzenai-npu",
  );
});

test("a core build's repo drops only its owner", () => {
  const core = {
    id: "qwen3.5-9b-core",
    hf_repo: "amd/Qwen3.5-9B-Q4_G128-GGUF",
  };
  assert.equal(npuDisplayName(core), "Qwen3.5-9B-Q4_G128-GGUF");
});

test("a converted Q4NX model is named by its folder wherever it shows", () => {
  const converted = {
    id: "local-qwen3-0.6b-q4-1-q4nx-aaea03b7-q4nx-FLM",
    model_path: "lemonade:local-qwen3-0.6b-q4-1-q4nx-aaea03b7-q4nx-FLM",
    local: { folder: "q4nx/Qwen3-0.6B-Q4_1-q4nx" },
  };
  assert.equal(npuDisplayName(converted), "Qwen3-0.6B-Q4_1-q4nx");
  assert.equal(
    npuDisplayNameOf([converted], converted.model_path),
    "Qwen3-0.6B-Q4_1-q4nx",
  );
  assert.equal(
    npuDisplayName({ ...converted, local: { folder: "/mnt/run/npu-q4nx" } }),
    "run",
  );
});

test("a loaded model is named by its id or its lemonade: path", () => {
  assert.equal(
    npuDisplayNameOf([qwen], qwen.model_path),
    "Qwen3.6-35B-A3B-NPU2",
  );
  assert.equal(npuDisplayNameOf([qwen], qwen.id), "Qwen3.6-35B-A3B-NPU2");
  assert.equal(npuDisplayNameOf([qwen], "unsloth/Qwen3-8B-GGUF"), null);
  assert.equal(npuDisplayNameOf(null, qwen.model_path), null);
});

test("NPU search finds a model by its old id, its tag and its repo", () => {
  const api = loadWithStubs<typeof NpuApi>(
    new URL("../src/features/npu/api.ts", import.meta.url),
    {
      "@/features/auth": { authFetch: async () => new Response(null) },
      "@/features/auth/api": { authFetch: async () => new Response(null) },
      "@/lib/format-fastapi-error": formatFastApiError,
      "@/lib/sse-json-events": sseJsonEvents,
      "./display-name": displayName,
    },
    { relativePassthrough: true },
  );
  const row = {
    ...qwen,
    size_gb: 20,
    downloaded: true,
    labels: [],
    supports_vision: false,
    supports_reasoning: true,
    supports_tools: true,
    max_context_length: null,
    resume_percent: null,
  };
  const found = (query: string) =>
    api.npuRowsFor([row], { onDevice: false, query }).map((m) => m.id);
  for (const query of [
    "qwen3.6-moe-35b-a3b-FLM",
    "moe-35b",
    "qwen3.6-moe:35b",
    "Qwen3.6-35B-A3B-NPU2",
    "fastflowlm/",
  ]) {
    assert.deepEqual(found(query), [qwen.id], query);
  }
  assert.deepEqual(found("gemma"), []);
});
