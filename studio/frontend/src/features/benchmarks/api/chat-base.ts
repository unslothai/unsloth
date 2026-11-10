// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Chat's Run Settings are the source of truth: every sweep row loads what the sheet
// shows, changed only in the field that row varies.

import {
  type InferenceStatusResponse,
  type loadModel,
  useChatRuntimeStore,
} from "@/features/chat";
import { useShallow } from "zustand/react/shallow";
import { type BaseSetting, fmtTokens, userExtraArgs } from "../lib/bench-math";

type LoadModelRequest = Parameters<typeof loadModel>[0];
type ChatState = ReturnType<typeof useChatRuntimeStore.getState>;

const SPEC_LABEL: Record<string, string> = {
  auto: "Auto",
  off: "Off",
  mtp: "MTP",
  ngram: "Ngram",
  "mtp+ngram": "MTP+Ngram",
  dspark: "DSpark",
  dflash: "DFlash",
};

function orAuto(v: number | null | undefined): string {
  return v === null || v === undefined ? "auto" : String(v);
}

/** The sheet's values in its own words and order. */
export function chatSettings(s: ChatState): BaseSetting[] {
  const spec = s.speculativeType ?? "auto";
  const rows: BaseSetting[] = [
    {
      field: "max_seq_length",
      label: "Context Length",
      value: s.customContextLength ? fmtTokens(s.customContextLength) : "Auto",
    },
    {
      field: "cache_type_kv",
      label: "KV Cache Dtype",
      value: s.kvCacheDtype ?? "f16",
    },
    {
      field: "speculative_type",
      label: "Speculative Decoding",
      value: SPEC_LABEL[spec] ?? spec,
    },
    {
      field: "spec_draft_n_max",
      label: "Draft Tokens",
      value: orAuto(s.specDraftNMax),
    },
    {
      field: "n_parallel",
      label: "Parallel Slots",
      value: orAuto(s.nParallel),
    },
    { field: "n_batch", label: "Batch Size", value: orAuto(s.nBatch) },
    { field: "n_ubatch", label: "Micro-batch Size", value: orAuto(s.nUbatch) },
    {
      field: "tensor_parallel",
      label: "Tensor Parallelism",
      value: s.tensorParallel ? "on" : "off",
    },
    {
      field: "disable_vision",
      label: "Vision",
      value: s.disableVision ? "off" : "on",
    },
    {
      field: "reasoning_budget",
      label: "Reasoning Budget",
      value: String(s.reasoningBudget),
    },
    {
      field: "gpu_memory_mode",
      label: "GPU Memory",
      value:
        s.gpuMemoryMode === "manual"
          ? `Manual · ${s.gpuLayers < 0 ? "auto" : s.gpuLayers} layers${s.nCpuMoe ? ` · ${s.nCpuMoe} MoE on CPU` : ""}`
          : "Default",
    },
  ];
  if (s.reasoningBudgetMessage)
    rows.push({
      field: "reasoning_budget_message",
      label: "Reasoning Budget Message",
      value: s.reasoningBudgetMessage,
    });
  // Every row prints, defaults included, so a chart never leaves a reader guessing what ran.
  rows.push(
    {
      field: "spec_draft_cache_type",
      label: "Spec Decoding KV Cache Dtype",
      value: s.specDraftCacheDtype ?? "same as KV",
    },
    {
      field: "gpu_ids",
      label: "GPU selection",
      value: s.selectedGpuIds?.length ? s.selectedGpuIds.join(", ") : "auto",
    },
    {
      field: "tensor_split",
      label: "Split Ratio",
      value: s.splitRatio ? String(s.splitRatio) : "auto",
    },
    { field: "load_mode", label: "Mmap/Mlock", value: s.loadMode ?? "default" },
    {
      field: "ctx_checkpoints",
      label: "Checkpoints",
      value: orAuto(s.ctxCheckpoints),
    },
    {
      field: "cache_ram",
      label: "Cache RAM",
      value: s.cacheRam === null ? "auto" : `${s.cacheRam} MiB`,
    },
  );
  const extra = userExtraArgs(s.loadedLlamaExtraArgs);
  rows.push({
    field: "llama_extra_args",
    label: "Extra llama-server args",
    value: extra.length ? extra.join(" ") : "none",
  });
  const p = s.params;
  rows.push(
    {
      field: "temperature",
      label: "Temperature",
      value: String(p.temperature),
    },
    { field: "top_p", label: "Top P", value: String(p.topP) },
    { field: "top_k", label: "Top K", value: String(p.topK) },
    {
      field: "min_p",
      label: "Min P",
      value:
        p.minPMode === "server-default" ? "server default" : String(p.minP),
    },
    {
      field: "repetition_penalty",
      label: "Repetition Penalty",
      value: String(p.repetitionPenalty),
    },
    {
      field: "presence_penalty",
      label: "Presence Penalty",
      value: String(p.presencePenalty),
    },
  );
  return rows;
}

export function useChatSettings(): BaseSetting[] {
  return useChatRuntimeStore(
    useShallow((s) =>
      chatSettings(s).map((b) => `${b.field}\u0000${b.label}\u0000${b.value}`),
    ),
  ).map((row) => {
    const [field, label, value] = row.split("\u0000");
    return { field, label, value };
  });
}

/** The load chat would send for the loaded model with its sheet as it stands now. */
export interface ChatSampling {
  temperature: number;
  top_p: number;
  top_k: number;
  min_p?: number;
  repetition_penalty: number;
  presence_penalty: number;
}

/** Chat's sampling, sent with every benchmark generation: MTP acceptance moves with it. */
export function chatSampling(s: ChatState): ChatSampling {
  const p = s.params;
  return {
    temperature: p.temperature,
    top_p: p.topP,
    top_k: p.topK,
    ...(p.minPMode === "server-default" ? {} : { min_p: p.minP }),
    repetition_penalty: p.repetitionPenalty,
    presence_penalty: p.presencePenalty,
  };
}

export function chatBaseLoad(
  status: InferenceStatusResponse,
): LoadModelRequest {
  const s = useChatRuntimeStore.getState();
  const base: LoadModelRequest = {
    model_path: status.active_model ?? "",
    gguf_variant: status.gguf_variant ?? s.activeGgufVariant ?? null,
    hf_token: null,
    load_in_4bit: true,
    is_lora: false,
    max_seq_length: s.customContextLength ?? 0,
    cache_type_kv: s.kvCacheDtype,
    speculative_type: s.speculativeType,
    spec_draft_n_max: s.specDraftNMax,
    n_parallel: s.nParallel,
    reasoning_budget: s.reasoningBudget,
    reasoning_budget_message: s.reasoningBudgetMessage,
    llama_extra_args: userExtraArgs(s.loadedLlamaExtraArgs),
    tensor_parallel: s.tensorParallel,
    disable_vision: s.disableVision,
    gpu_memory_mode: s.gpuMemoryMode,
    gpu_layers: s.gpuLayers,
    n_cpu_moe: s.nCpuMoe,
  };
  // Omitted when blank, as chat does: a null counts as set and strips inherited values.
  if (s.nBatch != null) base.n_batch = s.nBatch;
  if (s.nUbatch != null) base.n_ubatch = s.nUbatch;
  if (s.loadMode != null) base.load_mode = s.loadMode;
  if (s.specDraftCacheDtype != null)
    base.spec_draft_cache_type = s.specDraftCacheDtype;
  if (s.ctxCheckpoints != null) base.ctx_checkpoints = s.ctxCheckpoints;
  if (s.cacheRam != null) base.cache_ram = s.cacheRam;
  if (s.splitRatio) base.tensor_split = s.splitRatio;
  if (s.selectedGpuIds?.length) base.gpu_ids = s.selectedGpuIds;
  return base;
}
