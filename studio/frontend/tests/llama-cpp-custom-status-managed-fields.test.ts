// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { register } from "node:module";
import test from "node:test";

import { installLocalStorageFake } from "./helpers/kit.ts";

installLocalStorageFake();
register("./store-settings-resolver.mjs", import.meta.url);

const { useChatRuntimeStore } = await import(
  "../src/features/chat/stores/chat-runtime-store.ts"
);
const { applyActiveModelStatusToStore } = await import(
  "../src/features/chat/lib/apply-inference-status-to-store.ts"
);

const MODEL = "unsloth/Qwen3-1.7B-GGUF";
const custom = {
  version: 1,
  mode: "custom",
  ini: "[*]\nctx-size = 3072\ndevice = none\ngpu-layers = 0\n",
  section: null,
} as const;

// What /status reports after a custom load: the backend records the INI's placement as
// manual/0 layers and its ctx-size as the requested context.
function statusFor(config: unknown) {
  return {
    active_model: MODEL,
    model_identifier: MODEL,
    is_gguf: true,
    gpu_memory_mode: "manual",
    gpu_layers: 0,
    requested_context_length: 3072,
    context_length: 3072,
    requested_llama_cpp_config: config,
  } as never;
}

function managedAuto() {
  useChatRuntimeStore.setState({
    modelLoading: false,
    gpuMemoryMode: "auto",
    loadedGpuMemoryMode: "auto",
    gpuLayers: -1,
    loadedGpuLayers: null,
    customContextLength: null,
    loadedCustomContextLength: null,
    params: { ...useChatRuntimeStore.getState().params, checkpoint: MODEL },
  });
}

test("a custom load's INI placement does not overwrite the managed controls", () => {
  managedAuto();
  applyActiveModelStatusToStore(statusFor(custom), { previousCheckpoint: MODEL });
  const s = useChatRuntimeStore.getState();
  assert.equal(s.llamaCppConfig?.mode, "custom");
  assert.equal(s.gpuMemoryMode, "auto", "Use Studio settings would reload with Manual");
  assert.equal(s.gpuLayers, -1, "Use Studio settings would reload with --gpu-layers 0");
  assert.equal(s.customContextLength, null, "Use Studio settings would reload with ctx 3072");
});

test("a managed load reporting the same placement still hydrates the controls", () => {
  managedAuto();
  applyActiveModelStatusToStore(statusFor({ version: 1, mode: "managed" }), {
    previousCheckpoint: MODEL,
  });
  const s = useChatRuntimeStore.getState();
  assert.equal(s.gpuMemoryMode, "manual");
  assert.equal(s.gpuLayers, 0);
});

test("a custom load's response leaves the managed placement alone", async () => {
  const { managedGpuMemoryFields } = await import(
    "../src/features/chat/stores/chat-runtime-store.ts"
  );
  const resp = { is_gguf: true, gpu_memory_mode: "manual", gpu_layers: 0, n_layers: 28 } as const;
  assert.deepEqual(
    managedGpuMemoryFields({ ...resp, requested_llama_cpp_config: custom }),
    { ggufLayerCount: 28, moeLayerCount: null },
  );
  const managed = managedGpuMemoryFields({
    ...resp,
    requested_llama_cpp_config: { mode: "managed" },
  }) as { gpuMemoryMode?: string; gpuLayers?: number };
  assert.equal(managed.gpuMemoryMode, "manual");
  assert.equal(managed.gpuLayers, 0);
});

test("a custom load's KV cache and speculative echo leave the managed controls alone", async () => {
  const { managedKvCacheFields, managedSpeculativeSettings } = await import(
    "../src/features/chat/stores/chat-runtime-store.ts"
  );
  const resp = { cache_type_kv: "f16", speculative_type: "none", spec_draft_n_max: null };
  assert.deepEqual(managedKvCacheFields({ ...resp, requested_llama_cpp_config: custom }), {});
  assert.deepEqual(managedSpeculativeSettings({ ...resp, requested_llama_cpp_config: custom }), {});
  assert.equal(managedKvCacheFields(resp).kvCacheDtype, "f16");

  useChatRuntimeStore.setState({
    modelLoading: false,
    kvCacheDtype: null,
    loadedKvCacheDtype: null,
    speculativeType: "auto",
    loadedSpeculativeType: "auto",
    params: { ...useChatRuntimeStore.getState().params, checkpoint: MODEL },
  });
  applyActiveModelStatusToStore(
    { ...(statusFor(custom) as object), cache_type_kv: "f16", speculative_type: "none" } as never,
    { previousCheckpoint: MODEL },
  );
  const s = useChatRuntimeStore.getState();
  assert.equal(s.kvCacheDtype, null);
  assert.equal(s.speculativeType, "auto");
});
