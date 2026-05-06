// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

export interface TrainingStartRequest {
  model_name: string;
  training_type: string;
  hf_token: string | null;
  load_in_4bit: boolean;
  max_seq_length: number;
  /** Allow loading models with custom code. Only enable for repos you trust. */
  trust_remote_code?: boolean;
  hf_dataset: string | null;
  subset: string | null;
  train_split: string | null;
  eval_split: string | null;
  dataset_slice_start: number | null;
  dataset_slice_end: number | null;
  local_datasets: string[];
  local_eval_datasets: string[];
  format_type: string;
  custom_format_mapping?: Record<string, unknown> | null;
  num_epochs: number;
  learning_rate: string;
  /** Optional CPT embedding LR. If omitted, backend uses lr/10; typical range is 2x-10x smaller than main LR. */
  embedding_learning_rate?: number | null;
  batch_size: number;
  gradient_accumulation_steps: number;
  warmup_steps: number | null;
  warmup_ratio: number | null;
  max_steps: number | null;
  save_steps: number;
  eval_steps: number;
  weight_decay: number;
  random_seed: number;
  packing: boolean;
  optim: string;
  lr_scheduler_type: string;
  use_lora: boolean;
  lora_r: number;
  lora_alpha: number;
  lora_dropout: number;
  target_modules: string[];
  gradient_checkpointing: string;
  use_rslora: boolean;
  use_loftq: boolean;
  train_on_completions: boolean;
  finetune_vision_layers: boolean;
  finetune_language_layers: boolean;
  finetune_attention_modules: boolean;
  finetune_mlp_modules: boolean;
  is_dataset_image: boolean;
  is_dataset_audio: boolean;
  is_embedding: boolean;
  enable_wandb: boolean;
  wandb_token: string | null;
  wandb_project: string | null;
  enable_tensorboard: boolean;
  tensorboard_dir: string | null;
  resume_from_checkpoint?: string | null;
}

export interface TrainingStartResponse {
  job_id: string;
  status: "queued" | "error";
  message: string;
  error: string | null;
}

export interface TrainingStopResponse {
  status: "stopped" | "idle";
  message: string;
}
