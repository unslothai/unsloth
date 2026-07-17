// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { authFetch } from "@/features/auth";
import { hubTokenHeader } from "@/features/hub";

interface VisionCheckResponse {
  model_name: string;
  is_vision: boolean;
}

interface EmbeddingCheckResponse {
  model_name: string;
  is_embedding: boolean;
}

interface BackendTrainingDefaults {
  max_seq_length?: number;
  num_epochs?: number;
  learning_rate?: number | string;
  optim?: string;
  lr_scheduler_type?: string;
  batch_size?: number;
  gradient_accumulation_steps?: number;
  warmup_steps?: number;
  max_steps?: number;
  save_steps?: number;
  eval_steps?: number;
  weight_decay?: number;
  random_seed?: number;
  vision_image_size?: number | string | null;
  packing?: boolean;
  train_on_completions?: boolean;
  gradient_checkpointing?: "none" | "true" | "unsloth";
  trust_remote_code?: boolean;
}

interface BackendLoraDefaults {
  lora_r?: number;
  lora_alpha?: number;
  lora_dropout?: number;
  target_modules?: string[];
  use_rslora?: boolean;
  use_loftq?: boolean;
  finetune_vision_layers?: boolean;
  finetune_language_layers?: boolean;
  finetune_attention_modules?: boolean;
  finetune_mlp_modules?: boolean;
}

interface BackendLoggingDefaults {
  enable_wandb?: boolean;
  wandb_project?: string;
  enable_tensorboard?: boolean;
  tensorboard_dir?: string;
  log_frequency?: number;
}

export interface BackendModelConfig {
  audio_type?: string | null;
  training?: BackendTrainingDefaults;
  lora?: BackendLoraDefaults;
  logging?: BackendLoggingDefaults;
}

export interface ModelConfigResponse {
  id: string;
  model_name?: string | null;
  config?: BackendModelConfig | null;
  is_vision: boolean;
  is_embedding?: boolean;
  is_audio: boolean;
  is_lora: boolean;
  base_model?: string | null;
  model_type?: "text" | "vision" | "audio" | "embeddings" | null;
  max_position_embeddings?: number | null;
  model_size_bytes?: number | null;
}

export interface LocalModelInfo {
  id: string;
  display_name: string;
  path: string;
  source: "models_dir" | "hf_cache" | "lmstudio" | "custom";
  model_id?: string | null;
  updated_at?: number | null;
}

interface LocalModelListResponse {
  models_dir: string;
  hf_cache_dir?: string | null;
  lmstudio_dirs: string[];
  models: LocalModelInfo[];
}

/** GET /api/models/check-vision; pass the token so a gated/private VLM is not misread as non-vision. */
export async function checkVisionModel(
  modelName: string,
  hfToken?: string | null,
): Promise<boolean> {
  const encoded = encodeURIComponent(modelName);
  const response = await authFetch(`/api/models/check-vision/${encoded}`, {
    headers: hubTokenHeader(hfToken?.trim() || null),
  });
  if (!response.ok) {
    return false;
  }
  const data = (await response.json()) as VisionCheckResponse;
  return data.is_vision;
}

/** GET /api/models/check-embedding; pass the token for gated/private repos. */
export async function checkEmbeddingModel(
  modelName: string,
  hfToken?: string | null,
): Promise<boolean> {
  const encoded = encodeURIComponent(modelName);
  const response = await authFetch(`/api/models/check-embedding/${encoded}`, {
    headers: hubTokenHeader(hfToken?.trim() || null),
  });
  if (!response.ok) {
    return false;
  }
  const data = (await response.json()) as EmbeddingCheckResponse;
  return data.is_embedding;
}

export async function getModelConfig(
  modelName: string,
  signal?: AbortSignal,
  hfToken?: string,
): Promise<ModelConfigResponse> {
  const encoded = encodeURIComponent(modelName);
  const response = await authFetch(`/api/models/config/${encoded}`, {
    headers: hubTokenHeader(hfToken),
    signal,
  });
  if (!response.ok) {
    throw new Error(`Failed to fetch model config (${response.status})`);
  }
  return (await response.json()) as ModelConfigResponse;
}

export async function listLocalModels(
  signal?: AbortSignal,
): Promise<LocalModelInfo[]> {
  const response = await authFetch("/api/models/local", { signal });
  if (!response.ok) {
    throw new Error(`Failed to fetch local models (${response.status})`);
  }
  const data = (await response.json()) as LocalModelListResponse;
  return data.models;
}
