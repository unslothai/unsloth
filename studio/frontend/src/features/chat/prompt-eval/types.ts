// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

export interface PromptItem {
  id: string;
  text: string;
  category?: string;
}

export interface PromptEvalModelEntry {
  id: string;
  isLora: boolean;
  ggufVariant?: string;
  displayName?: string;
}

export interface PromptEvalConfig {
  id: string;
  name: string;
  prompts: PromptItem[];
  models: PromptEvalModelEntry[];
  /** Snapshot of inference params at run-start */
  maxSeqLength: number;
}

export interface PromptEvalProgress {
  promptIdx: number;
  modelIdx: number;
  totalPrompts: number;
  totalModels: number;
  currentModelName: string;
  phase: "loading" | "generating" | "done";
}

export interface PromptEvalResultRecord {
  run_id: string;
  run_name: string;
  model_id: string;
  model_name: string;
  prompt_id: string;
  prompt_text: string;
  response_text: string;
  latency_ms?: number;
  input_tokens?: number;
  output_tokens?: number;
  tokens_per_sec?: number;
  inference_params: Record<string, unknown>;
  timestamp: string;
}
