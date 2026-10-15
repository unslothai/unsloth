// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { authFetch } from "@/features/auth";
import { hubTokenHeader } from "@/features/hub";
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
};

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
