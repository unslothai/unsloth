// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Sampling defaults saved for one model. The backend applies them to OpenAI and Anthropic
// requests that leave a value unset, including the one that switches the model in.

import { authFetch } from "@/features/auth";
import { readFastApiError } from "@/lib/format-fastapi-error";

const URL = "/api/settings/model-sampling-defaults";

export const SAMPLING_DEFAULT_FIELDS = [
  "temperature",
  "top_p",
  "top_k",
  "min_p",
  "repetition_penalty",
  "presence_penalty",
] as const;

export type SamplingDefaultField = (typeof SAMPLING_DEFAULT_FIELDS)[number];
export type SamplingDefaults = Partial<Record<SamplingDefaultField, number>>;

export async function fetchModelSamplingDefaults(): Promise<
  Record<string, SamplingDefaults>
> {
  const res = await authFetch(URL);
  if (!res.ok) {
    throw new Error(
      await readFastApiError(res, "Failed to read sampling defaults"),
    );
  }
  const body = (await res.json()) as {
    defaults?: Record<string, SamplingDefaults>;
  };
  return body.defaults ?? {};
}

/** Replace one model's saved defaults; an empty object forgets the model. */
export async function putModelSamplingDefaults(
  modelId: string,
  sampling: SamplingDefaults,
): Promise<Record<string, SamplingDefaults>> {
  const res = await authFetch(URL, {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      // biome-ignore lint/style/useNamingConvention: API schema
      model_id: modelId,
      sampling,
    }),
  });
  if (!res.ok) {
    throw new Error(
      await readFastApiError(res, "Failed to save sampling defaults"),
    );
  }
  const body = (await res.json()) as {
    defaults?: Record<string, SamplingDefaults>;
  };
  return body.defaults ?? {};
}

/** What a request naming this model gets for every field it leaves out. */
export async function fetchEffectiveSampling(
  modelId: string,
): Promise<SamplingDefaults> {
  const res = await authFetch(
    `/api/inference/sampling-defaults?model=${encodeURIComponent(modelId)}`,
  );
  if (!res.ok) {
    return {};
  }
  return (await res.json()) as SamplingDefaults;
}
