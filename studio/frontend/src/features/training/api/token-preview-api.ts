// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { CONTEXT_LENGTHS } from "@/config/training";
import { authFetch } from "@/features/auth/api";
import { hubTokenHeader } from "@/features/hub/lib/hub-token-header";
import { readFastApiError } from "@/lib/format-fastapi-error";
import type { TrainingConfigState } from "../types/config";
import { buildTrainingStartPayload } from "./mappers";

/** [token id, text, trained] */
export type PreviewToken = [number, string, boolean];

export type TokenPreviewSample = {
  tokens: PreviewToken[];
  /** Every ID of the row, including the ones past the displayed tokens. */
  input_ids: number[];
  num_tokens: number;
  num_trained: number;
  num_hidden: number;
  truncated_from: number | null;
  text: string;
};

export type TokenPreviewResponse = {
  samples: TokenPreviewSample[];
  masking: {
    applied: boolean;
    instruction_part?: string | null;
    response_part?: string | null;
    notes: string[];
  };
  final_format: string | null;
  vocab_size: number;
  example_text: string | null;
  max_row_tokens?: number | null;
  recommended_max_seq_length?: number | null;
};

/** Smallest standard CONTEXT_LENGTHS bucket covering `maxTokens` (#18236). */
export function recommendContextLengthForTokenCount(
  maxTokens: number | null | undefined,
  modelMaxLength?: number | null,
): number | null {
  if (typeof maxTokens !== "number" || !Number.isFinite(maxTokens) || maxTokens <= 0) {
    return null;
  }
  const minBucket = CONTEXT_LENGTHS[0] ?? 512;
  const maxBucket = CONTEXT_LENGTHS[CONTEXT_LENGTHS.length - 1] ?? 262144;
  const cap =
    typeof modelMaxLength === "number" &&
    Number.isFinite(modelMaxLength) &&
    modelMaxLength >= minBucket &&
    modelMaxLength <= maxBucket
      ? Math.floor(modelMaxLength)
      : maxBucket;
  for (const bucket of CONTEXT_LENGTHS) {
    if (bucket >= maxTokens) {
      return Math.min(bucket, cap);
    }
  }
  return Math.min(maxBucket, cap);
}

/** Resolve recommended context length from a TokenPreviewResponse (#18236). */
export function resolveRecommendedContextLength(
  response: Pick<TokenPreviewResponse, "recommended_max_seq_length" | "max_row_tokens" | "samples">,
): { recommended: number; maxTokens: number } | null {
  const sampleMax = response.samples.reduce((acc, sample) => {
    const rowLen = sample.truncated_from ?? sample.num_tokens;
    return rowLen > acc ? rowLen : acc;
  }, 0);
  const maxTokens =
    typeof response.max_row_tokens === "number" && response.max_row_tokens > 0
      ? Math.max(response.max_row_tokens, sampleMax)
      : sampleMax;
  const recommended =
    typeof response.recommended_max_seq_length === "number" &&
    response.recommended_max_seq_length > 0
      ? response.recommended_max_seq_length
      : recommendContextLengthForTokenCount(maxTokens);
  if (!recommended || maxTokens <= 0) {
    return null;
  }
  return { recommended, maxTokens };
}

type TokenPreviewArgs = {
  config: TrainingConfigState;
  hfToken: string | null;
  text?: string;
  /** The dataset's format from an earlier preview, so typed text is masked the way the run is. */
  finalFormat?: string | null;
  seed?: number;
  numSamples?: number;
  signal?: AbortSignal;
};

/** The dataset and masking fields come from the training start payload, so the preview reads
 * the rows and markers the run would. */
export async function fetchTokenPreview({
  config,
  hfToken,
  text,
  finalFormat,
  seed = 0,
  numSamples = 3,
  signal,
}: TokenPreviewArgs): Promise<TokenPreviewResponse> {
  const start = buildTrainingStartPayload(config, null);
  const datasetName =
    config.datasetSource === "huggingface"
      ? config.dataset
      : config.uploadedFile;
  const res = await authFetch("/api/train/token-preview", {
    method: "POST",
    signal,
    headers: {
      "Content-Type": "application/json",
      ...hubTokenHeader(hfToken),
    },
    body: JSON.stringify({
      model_name: start.model_name,
      model_type: config.modelType,
      load_in_4bit: start.load_in_4bit,
      dataset_name: datasetName || null,
      subset: start.subset,
      train_split: start.train_split,
      dataset_local_path: start.dataset_local_path,
      dataset_known_cached: start.dataset_known_cached,
      format_type: start.format_type,
      custom_format_mapping: start.custom_format_mapping ?? null,
      train_on_completions: start.train_on_completions,
      is_cpt: config.trainingMethod === "cpt",
      max_seq_length: start.max_seq_length,
      num_samples: numSamples,
      seed,
      text: text ?? null,
      final_format: finalFormat ?? null,
    }),
  });
  if (!res.ok) {
    throw new Error(await readFastApiError(res));
  }
  return res.json();
}
