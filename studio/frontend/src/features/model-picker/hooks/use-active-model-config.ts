// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { isExternalModelId, useChatRuntimeStore } from "@/features/chat";
import { useMemo } from "react";
import type { PerModelConfig } from "../model-config/per-model-config";
import { cachedPinnableGpuIndexKind } from "@/hooks/use-gpu-info";

export interface ActiveModelConfigState {
  checkpoint: string | null;
  isGguf: boolean;
  config: PerModelConfig | null;
}

export function useActiveModelConfig(): ActiveModelConfigState {
  const checkpoint = useChatRuntimeStore((s) => s.params.checkpoint) || null;
  const maxSeqLength = useChatRuntimeStore((s) => s.params.maxSeqLength);
  const activeGgufVariant = useChatRuntimeStore((s) => s.activeGgufVariant);
  const ggufContextLength = useChatRuntimeStore((s) => s.ggufContextLength);
  const customContextLength = useChatRuntimeStore((s) => s.customContextLength);
  const kvCacheDtype = useChatRuntimeStore((s) => s.kvCacheDtype);
  const speculativeType = useChatRuntimeStore((s) => s.speculativeType);
  const specDraftNMax = useChatRuntimeStore((s) => s.specDraftNMax);
  const tensorParallel = useChatRuntimeStore((s) => s.tensorParallel);
  const chatTemplateOverride = useChatRuntimeStore(
    (s) => s.chatTemplateOverride,
  );
  const gpuMemoryMode = useChatRuntimeStore((s) => s.gpuMemoryMode);
  const gpuLayers = useChatRuntimeStore((s) => s.gpuLayers);
  const nCpuMoe = useChatRuntimeStore((s) => s.nCpuMoe);
  const selectedGpuIds = useChatRuntimeStore((s) => s.selectedGpuIds);
  const ggufMemoryMode = useChatRuntimeStore((s) => s.ggufMemoryMode);

  const isGguf =
    activeGgufVariant != null ||
    ggufContextLength != null ||
    (checkpoint?.toLowerCase().endsWith(".gguf") ?? false);

  const config = useMemo<PerModelConfig | null>(() => {
    if (!checkpoint || isExternalModelId(checkpoint)) {
      return null;
    }
    const base: PerModelConfig = {
      customContextLength: customContextLength ?? null,
      maxSeqLength: isGguf ? null : maxSeqLength,
      kvCacheDtype: kvCacheDtype ?? null,
      speculativeType: speculativeType ?? "auto",
      specDraftNMax: specDraftNMax ?? null,
      tensorParallel: tensorParallel ?? false,
      chatTemplateOverride: chatTemplateOverride ?? null,
    };
    if (!isGguf) {
      return base;
    }
    return {
      ...base,
      gpuMemoryMode,
      gpuLayers,
      nCpuMoe,
      selectedGpuIds,
      selectedGpuIndexKind:
        selectedGpuIds == null ? null : (cachedPinnableGpuIndexKind() ?? null),
      ggufMemoryMode: ggufMemoryMode ?? undefined,
    };
  }, [
    checkpoint,
    isGguf,
    maxSeqLength,
    customContextLength,
    kvCacheDtype,
    speculativeType,
    specDraftNMax,
    tensorParallel,
    chatTemplateOverride,
    gpuMemoryMode,
    gpuLayers,
    nCpuMoe,
    selectedGpuIds,
    ggufMemoryMode,
  ]);

  return { checkpoint, isGguf, config };
}
