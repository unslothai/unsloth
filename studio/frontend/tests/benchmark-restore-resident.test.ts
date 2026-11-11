// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";
import * as benchMath from "../src/features/benchmarks/lib/bench-math.ts";
import { readSrc } from "./helpers/kit.ts";
import { loadWithStubs } from "./helpers/module-stubs.ts";

// Chat has 8K / f16 resident while 32K / q4_0 is selected but not applied yet.
const state = {
  activeGgufVariant: "Q4_K_M",
  customContextLength: 32768,
  loadedCustomContextLength: 8192,
  kvCacheDtype: "q4_0",
  loadedKvCacheDtype: "f16",
  speculativeType: "ngram",
  loadedSpeculativeType: "off",
  specDraftNMax: 8,
  loadedSpecDraftNMax: null,
  nParallel: 4,
  loadedNParallel: 1,
  reasoningBudget: 512,
  loadedReasoningBudgetRequested: -1,
  reasoningBudgetMessage: "edited",
  loadedReasoningBudgetMessageRequested: "",
  loadedLlamaExtraArgs: ["--top-k", "40"],
  tensorParallel: true,
  loadedTensorParallel: false,
  disableVision: true,
  loadedDisableVision: false,
  gpuMemoryMode: "manual",
  loadedGpuMemoryMode: "auto",
  gpuLayers: 20,
  loadedGpuLayers: null,
  nCpuMoe: 8,
  loadedNCpuMoe: null,
  nBatch: 4096,
  loadedNBatch: null,
  cacheRam: 2048,
  loadedCacheRam: null,
  selectedGpuIds: [1],
  loadedGpuIds: null,
};

const chatBase = loadWithStubs<{
  chatBaseLoad: (st: unknown) => Record<string, unknown>;
  residentLoad: (st: unknown) => Record<string, unknown>;
}>(new URL("../src/features/benchmarks/api/chat-base.ts", import.meta.url), {
  "@/features/chat": {
    GPU_LAYERS_AUTO: -1,
    useChatRuntimeStore: { getState: () => state },
  },
  "zustand/react/shallow": { useShallow: (f: unknown) => f },
  "../lib/bench-math": benchMath,
});

const status = { active_model: "org/model-GGUF", gguf_variant: "Q4_K_M" };

test("the restore payload is the resident load, not pending chat edits", () => {
  const load = chatBase.residentLoad(status);
  assert.equal(load.model_path, "org/model-GGUF");
  assert.equal(load.max_seq_length, 8192);
  assert.equal(load.cache_type_kv, "f16");
  assert.equal(load.speculative_type, "off");
  assert.equal(load.spec_draft_n_max, null);
  assert.equal(load.n_parallel, 1);
  assert.equal(load.reasoning_budget, -1);
  assert.equal(load.tensor_parallel, false);
  assert.equal(load.disable_vision, false);
  assert.equal(load.gpu_memory_mode, "auto");
  assert.equal(load.gpu_layers, -1);
  assert.equal(load.n_cpu_moe, 0);
  for (const unset of ["n_batch", "cache_ram", "gpu_ids"])
    assert.equal(unset in load, false, `${unset} carried a pending edit`);
});

test("the sweep itself still starts from the edited settings", () => {
  const load = chatBase.chatBaseLoad(status);
  assert.equal(load.max_seq_length, 32768);
  assert.equal(load.cache_type_kv, "q4_0");
});

test("the runner restores from the resident load", () => {
  const src = readSrc("features/benchmarks/api/bench-runner.ts");
  assert.match(src, /const originalLoad = residentLoad\(original\);/);
  assert.match(src, /await restore\(original, originalLoad,/);
});
