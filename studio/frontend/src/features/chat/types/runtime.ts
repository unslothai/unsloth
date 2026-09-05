// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

export interface InferenceParams {
  temperature: number;
  topP: number;
  topK: number;
  minP: number;
  repetitionPenalty: number;
  presencePenalty: number;
  maxSeqLength: number;
  maxTokens: number;
  systemPrompt: string;
  systemVariables: string;
  checkpoint: string;
  /** Allow loading models with custom code (e.g. NVIDIA Nemotron). Only enable for repos you trust. */
  trustRemoteCode?: boolean;
  /** Anthropic fast-mode toggle. Opus 4.6 / 4.7 only; higher OTPS at 6x Opus pricing. Default
   *  false. https://platform.claude.com/docs/en/build-with-claude/fast-mode */
  fastMode?: boolean;
  /** Sampling seed forwarded to the backends that read one so a generation can be reproduced.
   *  `null` is unset, which leaves the backend to draw its own per request. */
  seed?: number | null;
}

/** llama.cpp reads the seed as a uint32 and spends 0xFFFFFFFF on LLAMA_DEFAULT_SEED, its "draw
 *  one" sentinel, so a pin can name every value below that and no other. */
export const MAX_SAMPLING_SEED = 4_294_967_294;

/** An absent flag reads as false, so a row that omits one makes the gate answer wrong. */
export type SeedGateFlags = Required<
  Pick<ChatModelSummary, "isGguf" | "isMlx" | "isAudio" | "hasAudioInput">
>;

/** What the store holds, so every place that mints a row has to state those flags. */
export type ChatModelRow = ChatModelSummary & SeedGateFlags;

/** Whether the loaded model's backend reads a sampling seed. Shared so the panel cannot offer
 *  the field where the request would drop it, and takes the same `models[]` summary the
 *  request body already reads `isGguf` from. */
export function modelReadsSamplingSeed(
  activeModel: SeedGateFlags | null | undefined,
): boolean {
  // An audio-output model answers through generateAudio, whose request carries no seed.
  if (activeModel?.isAudio && !activeModel.hasAudioInput) {
    return false;
  }
  // llama-server takes a seed, and so does MLX. The transformers backend declares no `seed`
  // kwarg, so worker.py's `_backend_declares` gate drops it before generation.
  return activeModel?.isGguf === true || activeModel?.isMlx === true;
}

/** The params that survive a reload. `checkpoint` names the model rather than being one of its
 *  settings, so it is not one of them. */
export type PersistedInferenceParams = Partial<
  Omit<InferenceParams, "checkpoint">
>;

export const DEFAULT_INFERENCE_PARAMS: InferenceParams = {
  temperature: 0.6,
  topP: 0.95,
  topK: 20,
  minP: 0.01,
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
  /** llama-server takes video for this model: mmproj video support, a build with video enabled,
   *  and ffmpeg installed. */
  hasVideoInput?: boolean;
}

export interface ChatLoraSummary {
  id: string;
  name: string;
  baseModel: string;
  updatedAt?: number;
  source?: "training" | "exported";
  exportType?: "lora" | "merged" | "gguf";
  /** Codec when the checkpoint fine-tunes an audio model, else null. */
  audioType?: string | null;
}
