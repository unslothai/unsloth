// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { register } from "node:module";
import test from "node:test";

import { installLocalStorageFake } from "./helpers/kit.ts";

const { store: localStorageFake } = installLocalStorageFake();
localStorageFake.set("unsloth_chat_settings_imported_to_studio_db", "true");
register("./store-settings-resolver.mjs", import.meta.url);

const { useChatRuntimeStore, saveSpeculativeType } = await import(
  "../src/features/chat/stores/chat-runtime-store.ts"
);
const { applyPresetLoadConfig, EMPTY_PRESET_LOAD_CONFIG } = await import(
  "../src/features/chat/presets/preset-load-config.ts"
);
const { applyModelLoadConfigToRuntime, currentRuntimePerModelConfig } =
  await import(
    "../src/features/model-picker/model-config/apply-per-model-config.ts"
  );

const TEMPLATE = "{% for m in messages %}<|{{ m.role }}|>{{ m.content }}{% endfor %}";
const PRESET_KEYS = new Set([
  ...Object.keys(EMPTY_PRESET_LOAD_CONFIG),
  "gpuMemoryMode",
  "gpuLayers",
  "nCpuMoe",
]);

function setNonDefaultRuntime(): void {
  const { params } = useChatRuntimeStore.getState();
  useChatRuntimeStore.setState({
    params: {
      ...params,
      loadIn4bit: false,
      engine: "vllm",
      enginePrecision: "bf16",
      engineParallelism: "pipeline",
      engineKvCacheDtype: "fp8",
      engineSeed: 7,
      engineGpuMemoryUtilization: 0.8,
    },
    chatTemplateOverride: TEMPLATE,
    specDraftModel: "org/drafter",
    engramPrecision: "q8_0",
    mlxFastVerify: true,
    mlxInt8Prefill: true,
    mlxAnePrefill: true,
    mlxPleOffload: "off",
    mlxExpertOffload: "off",
    splitRatio: [3, 1],
    selectedGpuIds: [1, 0],
    selectedGpuIndexKind: "physical",
    gpuLayers: 12,
    nCpuMoe: 4,
    nParallel: 2,
  } as never);
}

test("applying a preset changes only the fields the preset saves", () => {
  setNonDefaultRuntime();
  const before = currentRuntimePerModelConfig({ includeMaxSeqLength: true });
  applyPresetLoadConfig({
    ...EMPTY_PRESET_LOAD_CONFIG,
    nParallel: 3,
    speculativeType: "dflash",
    gpuMemoryMode: "manual",
    gpuLayers: 20,
  });
  const after = currentRuntimePerModelConfig({ includeMaxSeqLength: true });

  assert.equal(after.nParallel, 3);
  for (const key of Object.keys(before) as (keyof typeof before)[]) {
    if (PRESET_KEYS.has(key)) continue;
    assert.deepEqual(after[key], before[key], key);
  }
});

test("a preset without GPU keys resets the GPU layer knobs and the split", () => {
  setNonDefaultRuntime();
  applyPresetLoadConfig({ ...EMPTY_PRESET_LOAD_CONFIG, nParallel: 3 });
  const state = useChatRuntimeStore.getState();
  assert.equal(state.gpuLayers, -1);
  assert.equal(state.nCpuMoe, 0);
  assert.equal(state.splitRatio, null);
});

test("a CPU-only preset drops the split", () => {
  setNonDefaultRuntime();
  applyPresetLoadConfig({
    ...EMPTY_PRESET_LOAD_CONFIG,
    gpuMemoryMode: "manual",
    gpuLayers: 0,
  });
  assert.equal(useChatRuntimeStore.getState().splitRatio, null);
});

test("a preset turning speculative decoding off drops the drafter", () => {
  setNonDefaultRuntime();
  applyPresetLoadConfig({ ...EMPTY_PRESET_LOAD_CONFIG, speculativeType: "off" });
  assert.equal(useChatRuntimeStore.getState().specDraftModel, null);
});

test("a preset without a speculative mode follows the standing one for the drafter", () => {
  saveSpeculativeType("off");
  try {
    setNonDefaultRuntime();
    applyPresetLoadConfig({ ...EMPTY_PRESET_LOAD_CONFIG, nParallel: 3 });
    assert.equal(useChatRuntimeStore.getState().specDraftModel, null);
  } finally {
    saveSpeculativeType("auto");
  }
});

test("switching to a model without a saved config drops the kept template", () => {
  setNonDefaultRuntime();
  applyPresetLoadConfig({ ...EMPTY_PRESET_LOAD_CONFIG, nParallel: 3 });
  assert.equal(useChatRuntimeStore.getState().chatTemplateOverride, TEMPLATE);
  applyModelLoadConfigToRuntime(null);
  assert.equal(useChatRuntimeStore.getState().chatTemplateOverride, null);
});
