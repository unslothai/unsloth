// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// The Evals model pick. BenchmarksPage owns it because the picker sits in its header, next to the
// tabs, in the slot the sweeps picker uses.

import { useChatModelRuntime, useChatRuntimeStore } from "@/features/chat";
import type {
  ModelOption,
  ModelSelectorChangeMeta,
} from "@/features/model-picker/components/model-selector/types";
import { useCallback, useEffect, useMemo, useState } from "react";

export interface EvalsModel {
  /** On-device GGUFs: evals score through llama-server. */
  models: ModelOption[];

  selectedModel: string | null;
  selectedGgufVariant: string | null;

  /** Chat's runtime load in flight (truthy while a pick is loading), or null. */
  loadingModel: { id: string; displayName: string } | null;
  /** The checkpoint chat currently has resident, i.e. what "Loaded" means here. */
  residentModel: string | null;
  /** Loaded GGUF quant of the resident model. */
  activeGgufVariant: string | null;

  pick: (value: string, meta: ModelSelectorChangeMeta) => void;
}

/** ``active`` is false off the Evals tab, so the model list is only refreshed when it's shown. */
export function useEvalsModel(active = true): EvalsModel {
  const [selectedModel, setSelectedModel] = useState<string | null>(null);
  const [selectedGgufVariant, setSelectedGgufVariant] = useState<string | null>(
    null,
  );

  const { selectModel, loadingModel, refresh } = useChatModelRuntime();
  const inferenceParams = useChatRuntimeStore((s) => s.params);
  const activeGgufVariant = useChatRuntimeStore((s) => s.activeGgufVariant);
  const chatModels = useChatRuntimeStore((s) => s.models);

  // Follow chat's runtime: a model loaded elsewhere (chat, Hub) becomes the pick, and
  // an unloaded runtime clears it.
  useEffect(() => {
    if (
      inferenceParams.checkpoint &&
      inferenceParams.checkpoint !== selectedModel
    ) {
      setSelectedModel(inferenceParams.checkpoint);
      setSelectedGgufVariant(activeGgufVariant);
    } else if (!inferenceParams.checkpoint && selectedModel) {
      setSelectedModel(null);
      setSelectedGgufVariant(null);
    }
  }, [inferenceParams.checkpoint, activeGgufVariant, selectedModel]);

  useEffect(() => {
    if (active) void refresh();
  }, [active, refresh]);

  const models = useMemo<ModelOption[]>(
    () =>
      chatModels
        .filter((m) => m.isGguf)
        .map((m) => ({
          id: m.id,
          name: m.name,
          description: m.description,
          isGguf: m.isGguf,
        })),
    [chatModels],
  );

  const pick = useCallback(
    (value: string, meta: ModelSelectorChangeMeta) => {
      setSelectedModel(value);
      setSelectedGgufVariant(meta.ggufVariant ?? null);
      if (value && value !== (inferenceParams.checkpoint ?? undefined)) {
        void selectModel({
          id: value,
          source: meta.source,
          isLora: meta.isLora,
          ggufVariant: meta.ggufVariant,
          isDownloaded: meta.isDownloaded,
          expectedBytes: meta.expectedBytes,
          isGguf: meta.isGguf,
          forceReload: true,
          throwOnError: true,
        });
      }
    },
    [inferenceParams.checkpoint, selectModel],
  );

  return {
    models,
    selectedModel,
    selectedGgufVariant,
    loadingModel,
    residentModel: inferenceParams.checkpoint ?? null,
    activeGgufVariant,
    pick,
  };
}
