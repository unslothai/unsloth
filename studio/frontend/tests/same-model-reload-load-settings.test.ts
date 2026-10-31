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

const GGUF = "unsloth/Qwen3-0.6B-GGUF";
const MLX = "mlx-community/Qwen3.5-2B-4bit";

function hydrate(status: Record<string, unknown>, model: string): void {
  useChatRuntimeStore.setState({
    modelLoading: false,
    params: { ...useChatRuntimeStore.getState().params, checkpoint: "" },
  });
  applyActiveModelStatusToStore(status as never, { previousCheckpoint: "" });
  useChatRuntimeStore.setState({
    params: { ...useChatRuntimeStore.getState().params, checkpoint: model },
  });
}

function reload(status: Record<string, unknown>, model: string): void {
  applyActiveModelStatusToStore(status as never, { previousCheckpoint: model });
}

function ggufStatus(over: Record<string, unknown>) {
  return {
    active_model: GGUF,
    model_identifier: GGUF,
    is_gguf: true,
    gguf_variant: "Q8_0",
    cache_type_kv: "f16",
    tensor_parallel: false,
    spec_draft_n_max: 16,
    speculative_type: "ngram-mod",
    chat_template_override: null,
    ...over,
  };
}

function mlxStatus(over: Record<string, unknown>) {
  return {
    active_model: MLX,
    model_identifier: MLX,
    is_gguf: false,
    is_mlx: true,
    gguf_variant: null,
    mlx_kv_quant_requested: null,
    mlx_kv_quant_reason: null,
    mlx_kv_quant_note: null,
    chat_template_override_reason: null,
    mlx_int8_prefill_requested: false,
    mlx_ane_prefill_requested: false,
    ...over,
  };
}

test("a same-model GGUF reload elsewhere moves every clean control and baseline", () => {
  useChatRuntimeStore.setState({ activeGgufVariant: "Q8_0" });
  hydrate(ggufStatus({}), GGUF);
  reload(
    ggufStatus({
      cache_type_kv: "q8_0",
      tensor_parallel: true,
      spec_draft_n_max: 32,
      speculative_type: "off",
      chat_template_override: "{{ messages }}",
    }),
    GGUF,
  );
  const s = useChatRuntimeStore.getState();
  assert.deepEqual(
    [s.kvCacheDtype, s.tensorParallel, s.specDraftNMax, s.speculativeType, s.chatTemplateOverride],
    ["q8_0", true, 32, "off", "{{ messages }}"],
  );
  assert.deepEqual(
    [
      s.loadedKvCacheDtype,
      s.loadedTensorParallel,
      s.loadedSpecDraftNMax,
      s.loadedSpeculativeType,
      s.loadedChatTemplateOverride,
    ],
    ["q8_0", true, 32, "off", "{{ messages }}"],
  );
});

test("a same-model MLX reload elsewhere moves the KV width, int8 prefill and verdict", () => {
  useChatRuntimeStore.setState({ activeGgufVariant: null });
  hydrate(mlxStatus({ mlx_ane_prefill_requested: true }), MLX);
  reload(
    mlxStatus({
      mlx_kv_quant_requested: "8",
      mlx_kv_quant_reason: "Quantized KV cache to 8 bits",
      mlx_kv_quant_note: "from token 256",
      mlx_int8_prefill_requested: true,
      mlx_ane_prefill_requested: false,
      mlx_offload: {
        ple: { applied: 0.5, requested: "auto", reason: null },
        experts: { applied: null, requested: 0.25, reason: "no_quantized_experts" },
        disk_bytes: 1024,
      },
    }),
    MLX,
  );
  const s = useChatRuntimeStore.getState();
  assert.equal(s.mlxKvQuant, "8", "the clean control follows the resident");
  assert.equal(s.loadedMlxKvQuantRequested, "8", "the baseline describes the resident");
  assert.equal(s.mlxKvQuantReason, "Quantized KV cache to 8 bits");
  assert.equal(s.mlxKvQuantNote, "from token 256");
  assert.equal(s.mlxInt8Prefill, true);
  assert.equal(s.loadedMlxInt8PrefillRequested, true);
  assert.equal(s.mlxAnePrefill, false);
  assert.equal(s.loadedMlxAnePrefillRequested, false);
  assert.deepEqual(
    [s.mlxPleOffload, s.loadedMlxPleOffload, s.mlxExpertOffload, s.loadedMlxExpertOffload, s.mlxExpertOffloadReason],
    [null, 0.5, 0.25, null, "no_quantized_experts"],
  );
});

test("a same-model MLX reload elsewhere moves a width that was already seeded", () => {
  useChatRuntimeStore.setState({ activeGgufVariant: null });
  hydrate(
    mlxStatus({
      mlx_kv_quant_requested: "8",
      mlx_kv_quant_reason: "Quantized KV cache to 8 bits",
    }),
    MLX,
  );
  reload(
    mlxStatus({
      mlx_kv_quant_requested: "4",
      mlx_kv_quant_reason: "Quantized KV cache to 4 bits",
    }),
    MLX,
  );
  const s = useChatRuntimeStore.getState();
  assert.equal(s.mlxKvQuant, "4");
  assert.equal(s.loadedMlxKvQuantRequested, "4");
  assert.equal(s.mlxKvQuantReason, "Quantized KV cache to 4 bits");
});

test("a same-model MLX reload keeps a value this tab staged but advances the baseline", () => {
  useChatRuntimeStore.setState({ activeGgufVariant: null });
  hydrate(mlxStatus({ mlx_kv_quant_requested: "8" }), MLX);
  useChatRuntimeStore.setState({ mlxKvQuant: "4" });
  reload(mlxStatus({ mlx_kv_quant_requested: null }), MLX);
  const s = useChatRuntimeStore.getState();
  assert.equal(s.mlxKvQuant, "4");
  assert.equal(s.loadedMlxKvQuantRequested, null);
});

for (const staged of ["int8", "ane"] as const) {
  test(`a steady MLX poll leaves a staged ${staged} value alone while the other follows the server`, () => {
    useChatRuntimeStore.setState({ activeGgufVariant: null });
    hydrate(mlxStatus({ mlx_kv_quant_requested: "8" }), MLX);
    useChatRuntimeStore.setState({ mlxKvQuant: "4", [staged === "int8" ? "mlxInt8Prefill" : "mlxAnePrefill"]: true });
    const other = staged === "int8" ? "mlx_ane_prefill_requested" : "mlx_int8_prefill_requested";
    reload(mlxStatus({ mlx_kv_quant_requested: "8", [other]: true }), MLX);
    const s = useChatRuntimeStore.getState();
    assert.equal(s.mlxKvQuant, "4");
    assert.deepEqual([s.mlxInt8Prefill, s.mlxAnePrefill], [true, true]);
    assert.deepEqual(
      [s.loadedMlxInt8PrefillRequested, s.loadedMlxAnePrefillRequested],
      staged === "int8" ? [false, true] : [true, false],
    );
    assert.equal(s.loadedMlxKvQuantRequested, "8");
  });
}
