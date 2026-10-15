// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { useEffect } from "react";
import { useShallow } from "zustand/react/shallow";
import { useHfTokenStore } from "@/features/hub";
import { getDrafterCapability } from "../api/train-api";
import { useTrainingConfigStore } from "../stores/training-config-store";

/** Keep `drafterCapability` current for the selected model while drafter training is on.
 * Mounted with the params section in both modes: the start payload only carries `drafter`
 * once the capability for this exact model says it can train one. */
// The last probe, shared by every mounted copy of the hook: one request per (model, token), and a
// failure is retried when the section is reopened or the token changes, never cached as final.
let lastProbe: { key: string; failed: boolean; done: boolean } | null = null;

export function useDrafterCapability(active: boolean): void {
  const { selectedModel, capabilityFor, setDrafterCapability } =
    useTrainingConfigStore(
      useShallow((state) => ({
        selectedModel: state.selectedModel,
        capabilityFor: state.drafterCapability?.model_name ?? null,
        setDrafterCapability: state.setDrafterCapability,
      })),
    );
  const token = useHfTokenStore((state) => state.token);

  useEffect(() => {
    const key = `${selectedModel}\u0000${token}`;
    if (!active) {
      if (lastProbe?.failed) lastProbe = null;
      return;
    }
    if (!selectedModel || (capabilityFor === selectedModel && lastProbe?.key === key)) return;
    lastProbe = { key, failed: false, done: false };
    const controller = new AbortController();
    getDrafterCapability(selectedModel, controller.signal, token)
      .then((capability) => {
        if (controller.signal.aborted) return;
        lastProbe = { key, failed: false, done: true };
        // The backend may echo a resolved path; key the answer on what was asked.
        setDrafterCapability({ ...capability, model_name: selectedModel });
      })
      .catch((error: unknown) => {
        if (controller.signal.aborted) return;
        lastProbe = { key, failed: true, done: true };
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
    return () => {
      controller.abort();
      // An answer still in flight was dropped: let the next run ask again.
      if (lastProbe?.key === key && !lastProbe.done) lastProbe = null;
    };
  }, [active, selectedModel, capabilityFor, token, setDrafterCapability]);
}
