// SPDX-License-Identifier: AGPL-3.0-only - See /studio/LICENSE.AGPL-3.0
// Copyright © 2025 Unsloth AI

export interface InferenceParams {
  temperature: number;
  topP: number;
  topK: number;
  minP: number;
  repetitionPenalty: number;
  maxSeqLength: number;
  maxTokens: number;
  systemPrompt: string;
  checkpoint: string;
  /** Allow loading models with custom code (e.g. NVIDIA Nemotron). Only enable for repos you trust. */
  trustRemoteCode?: boolean;
}

export const DEFAULT_INFERENCE_PARAMS: InferenceParams = {
  temperature: 0.7,
  topP: 0.9,
  topK: 50,
  minP: 0.01,
  repetitionPenalty: 1.1,
  maxSeqLength: 4096,
  maxTokens: 2048,
  systemPrompt: "",
  checkpoint: "",
  trustRemoteCode: false,
};

export interface ChatModelSummary {
  id: string;
  name: string;
  description?: string;
  isVision: boolean;
  isLora: boolean;
  isGguf?: boolean;
  isAudio?: boolean;
  audioType?: string | null;
  hasAudioInput?: boolean;
}

export interface ChatLoraSummary {
  id: string;
  name: string;
  baseModel: string;
  updatedAt?: number;
  source?: "training" | "exported";
  exportType?: "lora" | "merged" | "gguf";
}
