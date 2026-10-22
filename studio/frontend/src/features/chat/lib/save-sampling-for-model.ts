// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { authFetch } from "@/features/auth";
import { readFastApiError } from "@/lib/format-fastapi-error";
import type { InferenceParams } from "../types/runtime";

/** Store the sampling on screen as this model's defaults, the same row the run settings editor writes,
 *  so API requests that leave a value out get it too. Min P on the server's choice stays unsaved. */
export async function saveSamplingForModel(
  modelId: string,
  params: InferenceParams,
): Promise<void> {
  const res = await authFetch("/api/settings/model-sampling-defaults", {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      // biome-ignore lint/style/useNamingConvention: API schema
      model_id: modelId,
      sampling: {
        temperature: params.temperature,
        // biome-ignore lint/style/useNamingConvention: API schema
        top_p: params.topP,
        // biome-ignore lint/style/useNamingConvention: API schema
        top_k: params.topK,
        // biome-ignore lint/style/useNamingConvention: API schema
        repetition_penalty: params.repetitionPenalty,
        // biome-ignore lint/style/useNamingConvention: API schema
        presence_penalty: params.presencePenalty,
        ...(params.minPMode === "custom"
          ? // biome-ignore lint/style/useNamingConvention: API schema
            { min_p: params.minP }
          : {}),
      },
    }),
  });
  if (!res.ok) {
    throw new Error(
      await readFastApiError(res, "Failed to save sampling for this model"),
    );
  }
}
