// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

export type MinPMode = "server-default" | "custom";

export interface InferenceParams {
  engineParallelism?: "tensor" | "pipeline" | "data";
  enginePrecision?: "auto" | "bf16" | "fp16" | "int4" | "int8" | "fp8";
  engine?: "auto" | "vllm" | "sglang";
  temperature: number;
  topP: number;
  topK: number;
  minP: number;
  minPMode?: MinPMode;
  repetitionPenalty: number;
  presencePenalty: number;
  maxSeqLength: number;
  maxTokens: number;
  systemPrompt: string;
  systemVariables: string;
  checkpoint: string;
  /** Only enable for repos you trust. */
  trustRemoteCode?: boolean;
  /** Opus 4.6 / 4.7 only; 6x Opus pricing. */
  fastMode?: boolean;
  /** `null` leaves the backend to draw its own seed per request. */
  seed?: number | null;
}

/** llama.cpp reserves 0xFFFFFFFF as LLAMA_DEFAULT_SEED (draw one). */
export const MAX_SAMPLING_SEED = 4_294_967_294;

/** An absent flag reads as false, so a row omitting one makes the gate answer wrong. */
export type SeedGateFlags = Required<
  Pick<ChatModelSummary, "isGguf" | "isMlx" | "isAudio" | "hasAudioInput">
>;

export type ChatModelRow = ChatModelSummary & SeedGateFlags;

export function modelReadsSamplingSeed(
  activeModel: SeedGateFlags | null | undefined,
): boolean {
  // Audio-output models use generateAudio, whose request carries no seed.
  if (activeModel?.isAudio && !activeModel.hasAudioInput) {
    return false;
  }
  // The transformers backend declares no `seed` kwarg, so worker.py drops it.
  return activeModel?.isGguf === true || activeModel?.isMlx === true;
}

export type PersistedInferenceParams = Partial<
  Omit<InferenceParams, "checkpoint" | "engine" | "enginePrecision" | "engineParallelism">
>;

export const DEFAULT_INFERENCE_PARAMS: InferenceParams = {
  temperature: 0.6,
  topP: 0.95,
  topK: 20,
  minP: 0.01,
  minPMode: "server-default",
  repetitionPenalty: 1.0,
  presencePenalty: 0.0,
  maxSeqLength: 4096,
  maxTokens: 8192,
  systemPrompt: "",
  systemVariables: "",
  checkpoint: "",
  trustRemoteCode: false,
  fastMode: false,
  seed: null,
};

export interface ChatModelSummary {
  id: string;
  name: string;
  description?: string;
  isVision: boolean;
  isLora: boolean;
  isGguf?: boolean;
  isMlx?: boolean;
  isAudio?: boolean;
  audioType?: string | null;
  hasAudioInput?: boolean;
  hasVideoInput?: boolean;
}

export interface ChatLoraSummary {
  id: string;
  name: string;
  baseModel: string;
  updatedAt?: number;
  source?: "training" | "exported";
  exportType?: "lora" | "merged" | "gguf";
  sizeBytes?: number | null;
  audioType?: string | null;
}
