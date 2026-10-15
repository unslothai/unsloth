// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { useEffect } from "react";
import { useShallow } from "zustand/react/shallow";
import { getHfToken } from "@/features/hub";
import { getDrafterCapability } from "../api/train-api";
import { useTrainingConfigStore } from "../stores/training-config-store";

/** Keep `drafterCapability` current for the selected model while drafter training is on.
 * Mounted with the params section in both modes: the start payload only carries `drafter`
 * once the capability for this exact model says it can train one. */
export function useDrafterCapability(active: boolean): void {
  const { selectedModel, capabilityFor, setDrafterCapability } =
    useTrainingConfigStore(
      useShallow((state) => ({
        selectedModel: state.selectedModel,
        capabilityFor: state.drafterCapability?.model_name ?? null,
        setDrafterCapability: state.setDrafterCapability,
      })),
    );

  useEffect(() => {
    if (!active || !selectedModel || capabilityFor === selectedModel) return;
    const controller = new AbortController();
    getDrafterCapability(selectedModel, controller.signal, getHfToken())
      .then((capability) => {
        if (controller.signal.aborted) return;
        // The backend may echo a resolved path; key the answer on what was asked.
        setDrafterCapability({ ...capability, model_name: selectedModel });
      })
      .catch((error: unknown) => {
        if (controller.signal.aborted) return;
        setDrafterCapability({
          model_name: selectedModel,
          has_native_mtp: false,
          strategy: null,
          supported: false,
          reason: error instanceof Error ? error.message : String(error),
          vllm: false,
          llama_cpp: false,
        });
      });
    return () => controller.abort();
  }, [active, selectedModel, capabilityFor, setDrafterCapability]);
}
