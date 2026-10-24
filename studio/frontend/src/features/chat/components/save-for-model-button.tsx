// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { Button } from "@/components/ui/button";
import { authFetch } from "@/features/auth";
import { readFastApiError } from "@/lib/format-fastapi-error";
import type { toast } from "@/lib/toast";
import type { useChatRuntimeStore } from "../stores/chat-runtime-store";

// Passed in, not imported: importing them here splits them out of the startup bundle into extra chunks.
type Deps = { store: typeof useChatRuntimeStore; notify: typeof toast };

/** Store the sampling on screen as the loaded model's defaults: new chats with it start from them, and API
 *  requests that leave a value out get them. Min P on the server's choice stays unsaved. */
async function saveSamplingForCurrentModel({
  store,
  notify,
}: Deps): Promise<void> {
  const current = store.getState().params;
  const modelId = current.checkpoint;
  const res = await authFetch("/api/settings/model-sampling-defaults", {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      // biome-ignore lint/style/useNamingConvention: API schema
      model_id: modelId,
      sampling: {
        temperature: current.temperature,
        // biome-ignore lint/style/useNamingConvention: API schema
        top_p: current.topP,
        // biome-ignore lint/style/useNamingConvention: API schema
        top_k: current.topK,
        // biome-ignore lint/style/useNamingConvention: API schema
        repetition_penalty: current.repetitionPenalty,
        // biome-ignore lint/style/useNamingConvention: API schema
        presence_penalty: current.presencePenalty,
        ...(current.minPMode === "custom"
          ? // biome-ignore lint/style/useNamingConvention: API schema
            { min_p: current.minP }
          : {}),
      },
    }),
  });
  if (!res.ok) {
    throw new Error(
      await readFastApiError(res, "Failed to save sampling for this model"),
    );
  }
  const sampling = {
    temperature: current.temperature,
    topP: current.topP,
    topK: current.topK,
    minP: current.minP,
    minPMode: current.minPMode,
    repetitionPenalty: current.repetitionPenalty,
    presencePenalty: current.presencePenalty,
  };
  // Writing the memory also sets the live sliders: skip it if one moved during the save.
  const live = store.getState().params;
  if (
    live.checkpoint === modelId &&
    Object.entries(sampling).every(
      ([key, value]) => live[key as keyof typeof sampling] === value,
    )
  ) {
    store.getState().setRememberedParamsForModel(modelId, sampling);
  }
  notify.success("Sampling saved for this model.");
}

export default function SaveForModelButton(deps: Deps) {
  return (
    <div className="border-t border-[rgb(0_0_0_/_calc(0.13*var(--contrast-edge-gain,1)))] py-5 dark:border-[rgb(255_255_255_/_calc(0.09*var(--contrast-edge-gain,1)))]">
      <Button
        type="button"
        variant="outline"
        size="sm"
        className="w-full"
        onClick={() => {
          saveSamplingForCurrentModel(deps).catch((error: unknown) => {
            deps.notify.error(
              error instanceof Error ? error.message : String(error),
            );
          });
        }}
      >
        Save for this model
      </Button>
    </div>
  );
}
