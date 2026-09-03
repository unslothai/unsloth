// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { isExternalModelId, useChatRuntimeStore } from "@/features/chat";
import { useMemo } from "react";
import { usePlatformStore } from "@/config/env";
import type { PerModelConfig } from "../model-config/per-model-config";
import { isServedByMlx } from "../model-config/per-model-config";

export interface ActiveModelConfigState {
  checkpoint: string | null;
  isGguf: boolean;
  config: PerModelConfig | null;
}

export function useActiveModelConfig(): ActiveModelConfigState {
  const checkpoint = useChatRuntimeStore((s) => s.params.checkpoint) || null;
  const deviceType = usePlatformStore((s) => s.deviceType);
  const chatOnlyReason = usePlatformStore((s) => s.chatOnlyReason);
  const maxSeqLength = useChatRuntimeStore((s) => s.params.maxSeqLength);
  const activeGgufVariant = useChatRuntimeStore((s) => s.activeGgufVariant);
  const ggufContextLength = useChatRuntimeStore((s) => s.ggufContextLength);
  const customContextLength = useChatRuntimeStore((s) => s.customContextLength);
  const kvCacheDtype = useChatRuntimeStore((s) => s.kvCacheDtype);
  const mlxKvBits = useChatRuntimeStore((s) => s.mlxKvBits);
  const speculativeType = useChatRuntimeStore((s) => s.speculativeType);
  const specDraftNMax = useChatRuntimeStore((s) => s.specDraftNMax);
  const nParallel = useChatRuntimeStore((s) => s.nParallel);
  const nBatch = useChatRuntimeStore((s) => s.nBatch);
  const nUbatch = useChatRuntimeStore((s) => s.nUbatch);
  const tensorParallel = useChatRuntimeStore((s) => s.tensorParallel);
  const chatTemplateOverride = useChatRuntimeStore(
    (s) => s.chatTemplateOverride,
  );
  const gpuMemoryMode = useChatRuntimeStore((s) => s.gpuMemoryMode);
  const gpuLayers = useChatRuntimeStore((s) => s.gpuLayers);
  const nCpuMoe = useChatRuntimeStore((s) => s.nCpuMoe);
  const selectedGpuIds = useChatRuntimeStore((s) => s.selectedGpuIds);
  const selectedGpuIndexKind = useChatRuntimeStore(
    (s) => s.selectedGpuIndexKind,
  );

  const isGguf =
    activeGgufVariant != null ||
    ggufContextLength != null ||
    (checkpoint?.toLowerCase().endsWith(".gguf") ?? false);

  // Off-backend this stays null, or the model compares unequal to its own defaults
  // over a field it cannot show.
  const effectiveMlxKvBits = isServedByMlx(isGguf, deviceType, chatOnlyReason)
    ? (mlxKvBits ?? null)
    : null;

  const config = useMemo<PerModelConfig | null>(() => {
    if (!checkpoint || isExternalModelId(checkpoint)) {
      return null;
    }
    const base: PerModelConfig = {
      customContextLength: customContextLength ?? null,
      maxSeqLength: isGguf ? null : maxSeqLength,
      kvCacheDtype: kvCacheDtype ?? null,
      mlxKvBits: effectiveMlxKvBits,
      speculativeType: speculativeType ?? "auto",
      specDraftNMax: specDraftNMax ?? null,
      nParallel: nParallel ?? null,
      nBatch: nBatch ?? null,
      nUbatch: nUbatch ?? null,
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
      selectedGpuIndexKind,
    };
  }, [
    checkpoint,
    isGguf,
    maxSeqLength,
    customContextLength,
    kvCacheDtype,
    effectiveMlxKvBits,
    speculativeType,
    specDraftNMax,
    nParallel,
    nBatch,
    nUbatch,
    tensorParallel,
    chatTemplateOverride,
    gpuMemoryMode,
    gpuLayers,
    nCpuMoe,
    selectedGpuIds,
    selectedGpuIndexKind,
  ]);

  return { checkpoint, isGguf, config };
}
