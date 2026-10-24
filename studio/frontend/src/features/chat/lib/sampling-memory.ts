// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import type { InferenceParams } from "../types/runtime";

/** The sampling fields a model can have saved, and their names in `/api/settings/model-sampling-defaults`. */
const API_FIELDS = {
  temperature: "temperature",
  topP: "top_p",
  topK: "top_k",
  minP: "min_p",
  repetitionPenalty: "repetition_penalty",
  presencePenalty: "presence_penalty",
} as const;

type SamplingField = keyof typeof API_FIELDS;

/** One model's saved sampling; a field left out follows the model's recommendation. */
export type SavedSampling = Partial<Record<SamplingField, number>>;

type SamplingParams = Pick<InferenceParams, SamplingField | "minPMode">;

const FIELDS = Object.keys(API_FIELDS) as SamplingField[];

/** What Save stores. Min P on the server's choice stays unsaved. */
export function samplingPayload(params: SamplingParams): SavedSampling {
  const sampling: SavedSampling = {};
  for (const field of FIELDS) {
    if (field !== "minP" || params.minPMode === "custom") {
      sampling[field] = params[field];
    }
  }
  return sampling;
}

/** The request body's `sampling`; null for every field forgets the model. */
export function toApiSampling(
  sampling: SavedSampling | null,
): Record<string, number | null> {
  const body: Record<string, number | null> = {};
  for (const field of FIELDS) {
    body[API_FIELDS[field]] = sampling?.[field] ?? null;
  }
  return body;
}

export function fromApiSampling(
  body: Record<string, unknown> | null | undefined,
): SavedSampling | null {
  const sampling: SavedSampling = {};
  for (const field of FIELDS) {
    const value = body?.[API_FIELDS[field]];
    if (typeof value === "number" && Number.isFinite(value)) {
      sampling[field] = value;
    }
  }
  return Object.keys(sampling).length > 0 ? sampling : null;
}

export function sameSampling(a: SavedSampling, b: SavedSampling): boolean {
  return FIELDS.every((field) => {
    const x = a[field];
    const y = b[field];
    return x === undefined || y === undefined
      ? x === y
      : Math.abs(x - y) <= 1e-9;
  });
}

/** The read-only line beside Remember, worded like Run settings'. */
export function samplingSaveStatus(
  remember: boolean,
  saved: SavedSampling | null,
  current: SavedSampling,
): string | null {
  if (remember) {
    return saved && sameSampling(saved, current)
      ? "Saved for this model"
      : "Unsaved changes";
  }
  return saved ? "Saved settings will be forgotten" : null;
}

/** The model's recommended sampling from an inference status, the app defaults for what it leaves out. */
export function recommendedSampling(
  inference: Record<string, unknown> | null | undefined,
  d: Pick<InferenceParams, SamplingField>,
): Pick<InferenceParams, SamplingField> {
  const recommended = fromApiSampling(inference) ?? {};
  return {
    temperature: recommended.temperature ?? d.temperature,
    topP: recommended.topP ?? d.topP,
    topK: recommended.topK ?? d.topK,
    minP: recommended.minP ?? d.minP,
    // Not part of a model's recommendation.
    repetitionPenalty: d.repetitionPenalty,
    presencePenalty: recommended.presencePenalty ?? d.presencePenalty,
  };
}
