// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";
import type { ActiveModelConfigState } from "../src/features/model-picker/hooks/use-active-model-config.ts";
import type { PerModelConfig } from "../src/features/model-picker/model-config/per-model-config.ts";
import * as gpuTensorSplit from "../src/hooks/gpu-tensor-split.ts";
import { loadWithStubs } from "./helpers/module-stubs.ts";

function useActiveConfigFor(patch: Record<string, unknown>, gguf = true) {
  const state = {
    params: { checkpoint: "unsloth/Qwen3-0.6B-GGUF", maxSeqLength: 2048 },
    loadedLlamaExtraArgs: null,
    ...patch,
  };
  const { useActiveModelConfig } = loadWithStubs<{
    useActiveModelConfig: () => ActiveModelConfigState;
  }>(
    new URL(
      "../src/features/model-picker/hooks/use-active-model-config.ts",
      import.meta.url,
    ),
    {
      "@/features/chat": {
        isExternalModelId: () => false,
        useChatRuntimeStore: (select: (value: typeof state) => unknown) =>
          select(state),
      },
      "@/config/env": { usePlatformStore: () => ({ deviceType: "cuda" }) },
      "@/features/npu": {
        isNpuModelId: (value: string | null | undefined) =>
          Boolean(value?.startsWith("lemonade:")),
      },
      react: { useMemo: (factory: () => unknown) => factory() },
      "../model-config/per-model-config": {
        isServedByLlamaCpp: () => gguf,
        residentIsServedByMlx: () => false,
      },
    },
  );
  return useActiveModelConfig().config!;
}

function configsEqual(persistedMode: string) {
  return loadWithStubs<{
    perModelConfigsEqual: (
      a: PerModelConfig,
      b: PerModelConfig,
      options?: { followGlobal?: boolean },
    ) => boolean;
  }>(
    new URL(
      "../src/features/model-picker/model-config/apply-per-model-config.ts",
      import.meta.url,
    ),
    {
      "@/features/chat/stores/chat-runtime-store": {
        normalizeSpeculativeType: (value: string | null | undefined) =>
          value == null || value === "" ? null : value.toLowerCase(),
        readPersistedSpeculativeType: () => persistedMode,
      },
      "@/features/chat/presets/preset-policy": {},
      "@/hooks/gpu-tensor-split": gpuTensorSplit,
      "./config-signature": { gpuFieldsSignature: () => "" },
      "./per-model-config": {
        normalizeMaxSeqLength: (value: number | null | undefined) =>
          value ?? null,
      },
    },
  ).perModelConfigsEqual;
}

const BASE: PerModelConfig = {
  customContextLength: null,
  maxSeqLength: null,
  kvCacheDtype: null,
  speculativeType: null,
  specDraftNMax: null,
  nParallel: null,
  reasoningBudget: -1,
  reasoningBudgetMessage: "",
  nBatch: null,
  nUbatch: null,
  tensorParallel: false,
  disableVision: false,
  chatTemplateOverride: null,
};

test("the active GGUF baseline carries the arguments the server runs with", () => {
  const config = useActiveConfigFor({ loadedLlamaExtraArgs: ["--no-warmup"] });
  assert.deepEqual(config.llamaExtraArgs, ["--no-warmup"]);
  assert.deepEqual(
    useActiveConfigFor({ loadedLlamaExtraArgs: [] }).llamaExtraArgs,
    [],
  );
});

test("a loaded NPU model's runtime length is not read back as a context pin", () => {
  const params = { checkpoint: "lemonade:qwen3-0.6b-FLM", maxSeqLength: 4096 };
  assert.equal(useActiveConfigFor({ params }, false).maxSeqLength, null);
  assert.equal(
    useActiveConfigFor({ params: { ...params, checkpoint: "unsloth/Qwen3-0.6B" } }, false)
      .maxSeqLength,
    4096,
  );
});

test("unreported or non-llama.cpp arguments stay absent so the stored row can hydrate", () => {
  assert.equal("llamaExtraArgs" in useActiveConfigFor({}), false);
  assert.equal(
    "llamaExtraArgs" in
      useActiveConfigFor({ loadedLlamaExtraArgs: ["--no-warmup"] }, false),
    false,
  );
});

test("following the global mode, a stored Auto (null) equals the mode it resolves to", () => {
  const runtime = { followGlobal: true };
  const autoEqual = configsEqual("auto");
  assert.ok(autoEqual(BASE, { ...BASE, speculativeType: "auto" }, runtime));
  assert.ok(!autoEqual(BASE, { ...BASE, speculativeType: "off" }, runtime));
  const offEqual = configsEqual("off");
  assert.ok(offEqual(BASE, { ...BASE, speculativeType: "off" }, runtime));
  assert.ok(!offEqual(BASE, { ...BASE, speculativeType: "auto" }, runtime));
});

test("between stored configs a null mode stays distinct from an explicit one", () => {
  const offEqual = configsEqual("off");
  assert.ok(!offEqual(BASE, { ...BASE, speculativeType: "off" }));
  assert.ok(offEqual({ ...BASE, speculativeType: "mtp" }, { ...BASE, speculativeType: "MTP" }));
});
