// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// FastFlowLM run settings for AMD NPU models: kept like any run setting, sent with the NPU load,
// and read back from the status so the panel describes the resident model.

import assert from "node:assert/strict";
import { register } from "node:module";
import test from "node:test";

import { installLocalStorageFake, readSrc } from "./helpers/kit.ts";

installLocalStorageFake();
register("./store-settings-resolver.mjs", import.meta.url);

const {
  DEFAULT_PER_MODEL_CONFIG,
  contextPinPatch,
  isDefaultConfig,
  normalizePerModelConfig,
  npuLoadFields,
} = await import("../src/features/model-picker/model-config/per-model-config.ts");
const { perModelConfigsEqual } = await import(
  "../src/features/model-picker/model-config/apply-per-model-config.ts"
);
const { loadedConfigSignature } = await import(
  "../src/features/model-picker/model-config/config-signature.ts"
);
const { fromApiOverride, toApiOverride } = await import(
  "../src/features/model-picker/api/model-overrides.ts"
);
const { useChatRuntimeStore } = await import(
  "../src/features/chat/stores/chat-runtime-store.ts"
);
const { applyActiveModelStatusToStore } = await import(
  "../src/features/chat/lib/apply-inference-status-to-store.ts"
);

const NPU_SETTINGS = {
  npuPowerMode: "turbo",
  npuPrefillChunkLen: 1024,
  npuImgPreResize: 1,
} as const;

test("NPU settings keep a known power mode and in-range whole numbers only", () => {
  const kept = normalizePerModelConfig(NPU_SETTINGS);
  assert.deepEqual(
    [kept.npuPowerMode, kept.npuPrefillChunkLen, kept.npuImgPreResize],
    ["turbo", 1024, 1],
  );
  const bad = normalizePerModelConfig({
    npuPowerMode: "default",
    npuPrefillChunkLen: 0,
    npuImgPreResize: 5,
  });
  assert.deepEqual(
    [bad.npuPowerMode, bad.npuPrefillChunkLen, bad.npuImgPreResize],
    [null, null, null],
  );
  assert.equal(normalizePerModelConfig({ npuPrefillChunkLen: 12.5 }).npuPrefillChunkLen, null);
  assert.equal(normalizePerModelConfig({ npuImgPreResize: 0 }).npuImgPreResize, 0);
});

test("each NPU setting is a change from the default and from the loaded config", () => {
  const base = loadedConfigSignature(DEFAULT_PER_MODEL_CONFIG);
  for (const [key, value] of Object.entries(NPU_SETTINGS)) {
    const changed = { ...DEFAULT_PER_MODEL_CONFIG, [key]: value };
    assert.equal(isDefaultConfig(changed), false, key);
    assert.equal(perModelConfigsEqual(changed, DEFAULT_PER_MODEL_CONFIG), false, key);
    assert.notEqual(loadedConfigSignature(changed), base, key);
  }
});

test("the server copy carries the NPU settings and hydrates them back", () => {
  const set = { ...DEFAULT_PER_MODEL_CONFIG, ...NPU_SETTINGS };
  const row = toApiOverride(set);
  assert.deepEqual(
    [row.npu_power_mode, row.npu_prefill_chunk_len, row.npu_img_pre_resize],
    ["turbo", 1024, 1],
  );
  // Sent as null when unset: the server keeps a stored NPU setting the payload omits.
  const empty = toApiOverride(DEFAULT_PER_MODEL_CONFIG);
  assert.deepEqual(
    [empty.npu_power_mode, empty.npu_prefill_chunk_len, empty.npu_img_pre_resize],
    [null, null, null],
  );
  const hydrated = fromApiOverride(row);
  assert.deepEqual(
    [hydrated.npuPowerMode, hydrated.npuPrefillChunkLen, hydrated.npuImgPreResize],
    ["turbo", 1024, 1],
  );
  assert.equal(fromApiOverride({}, set).npuPowerMode, "turbo");
});

test("an NPU load sends only the settings that are set", () => {
  assert.deepEqual(npuLoadFields(DEFAULT_PER_MODEL_CONFIG), {});
  assert.deepEqual(npuLoadFields({ ...DEFAULT_PER_MODEL_CONFIG, ...NPU_SETTINGS }), {
    npu_power_mode: "turbo",
    npu_prefill_chunk_len: 1024,
    npu_img_pre_resize: 1,
  });
  const runtime = readSrc("features/chat/hooks/use-chat-model-runtime.ts");
  const npuLoad = runtime.slice(
    runtime.indexOf("const loadNpuModel = useCallback("),
    runtime.indexOf("const ejectModel = useCallback("),
  );
  assert.match(npuLoad, /\.\.\.npuLoadFields\(config\)/);
});

test("an NPU load's status seeds the resident settings, and a GPU model's clears them", () => {
  const npu = "lemonade:qwen3.5-9b-FLM";
  applyActiveModelStatusToStore(
    {
      active_model: "qwen3.5-9b-FLM",
      model_identifier: npu,
      is_gguf: false,
      is_npu: true,
      npu_power_mode: "balanced",
      npu_prefill_chunk_len: 2048,
      npu_img_pre_resize: 0,
    } as never,
    { previousCheckpoint: "", seedLoadParams: true },
  );
  let s = useChatRuntimeStore.getState();
  assert.deepEqual(
    [s.npuPowerMode, s.npuPrefillChunkLen, s.npuImgPreResize],
    ["balanced", 2048, 0],
  );
  applyActiveModelStatusToStore(
    {
      active_model: "unsloth/Qwen3-0.6B-GGUF",
      model_identifier: "unsloth/Qwen3-0.6B-GGUF",
      is_gguf: true,
    } as never,
    { previousCheckpoint: npu, seedLoadParams: true },
  );
  s = useChatRuntimeStore.getState();
  assert.deepEqual([s.npuPowerMode, s.npuPrefillChunkLen, s.npuImgPreResize], [null, null, null]);
});

test("lowering the context clamps a saved prefill chunk, so the reload matches the load", () => {
  const saved = { ...DEFAULT_PER_MODEL_CONFIG, maxSeqLength: 16384, npuPrefillChunkLen: 8192 };
  const lowered = { ...saved, ...contextPinPatch(4096, false, saved.npuPrefillChunkLen) };
  assert.equal(lowered.npuPrefillChunkLen, 4096);
  const loaded = { ...lowered, npuPrefillChunkLen: Math.min(8192, 4096) };
  assert.equal(perModelConfigsEqual(lowered, loaded), true);
  assert.equal("npuPrefillChunkLen" in contextPinPatch(4096, false, 1024), false);
  assert.equal("npuPrefillChunkLen" in contextPinPatch(4096, false, null), false);
  assert.equal(contextPinPatch(4096, true, 8192).npuPrefillChunkLen, 4096);
  const page = readSrc("features/model-picker/components/model-config-page.tsx");
  assert.equal(
    (page.match(/contextPinPatch\([^)]*config\.npuPrefillChunkLen\)/g) ?? []).length,
    2,
  );
  assert.match(page, /const NPU_DEFAULT_LABEL = "Default";/);
  assert.match(page, /value=\{config\.npuPrefillChunkLen \?\? ""\}\s*placeholder=\{NPU_DEFAULT_LABEL\}/);
  assert.equal(
    (page.match(/<SelectItem value=\{NPU_DEFAULT_CHOICE\}>\{NPU_DEFAULT_LABEL\}<\/SelectItem>/g) ?? []).length,
    2,
  );
});

test("the run settings panel offers Advanced settings for NPU models", () => {
  const page = readSrc("features/model-picker/components/model-config-page.tsx");
  assert.doesNotMatch(page, /\{!targetIsNpu && \(\s*<>\s*<AdvancedSettingsToggle/);
  assert.match(page, /<NpuAdvancedSettings/);
  for (const label of ["Power Mode", "Prefill Chunk Length", "Image Pre-resize"]) {
    assert.ok(page.includes(label), label);
  }
});

test("Run settings lists the NPU catalog itself, so Image Pre-resize shows without the picker", () => {
  const page = readSrc("features/model-picker/components/model-config-page.tsx");
  assert.match(page, /useNpuStatus\(targetIsNpu && !npuCatalogListed\)/);
  assert.match(
    page,
    /targetIsNpu && !npuCatalogListed && npuStatus\?\.ready === true\)\s*\{\s*void refreshNpuModels\(\)/,
  );
  assert.match(page, /showImagePreResize=\{target\.meta\.isVision === true \|\| npuCatalogVision\}/);
});

test("a failed switch restores the previous NPU model with the settings it was serving", () => {
  const source = readSrc("features/chat/hooks/use-chat-model-runtime.ts");
  const rollback = source.slice(source.indexOf("const rollbackResponse = await loadModel({"));
  assert.match(rollback.slice(0, 4000), /\.\.\.npuLoadFields\(rollbackState\)/);
});
