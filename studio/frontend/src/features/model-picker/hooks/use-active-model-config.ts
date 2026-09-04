// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { isExternalModelId, useChatRuntimeStore } from "@/features/chat";
import { usePlatformStore } from "@/config/env";
import { useMemo } from "react";
import {
  type PerModelConfig,
  isServedByLlamaCpp,
  residentIsServedByMlx,
} from "../model-config/per-model-config";

export interface ActiveModelConfigState {
  checkpoint: string | null;
  isGguf: boolean;
  config: PerModelConfig | null;
}

export function useActiveModelConfig(): ActiveModelConfigState {
  const checkpoint = useChatRuntimeStore((s) => s.params.checkpoint) || null;
  const maxSeqLength = useChatRuntimeStore((s) => s.params.maxSeqLength);
  const activeGgufVariant = useChatRuntimeStore((s) => s.activeGgufVariant);
  const loadedIsGguf = useChatRuntimeStore((s) => s.loadedIsGguf);
  const loadedIsMlx = useChatRuntimeStore((s) => s.loadedIsMlx);
  const activeNativePathToken = useChatRuntimeStore(
    (s) => s.activeNativePathToken,
  );
  const customContextLength = useChatRuntimeStore((s) => s.customContextLength);
  const kvCacheDtype = useChatRuntimeStore((s) => s.kvCacheDtype);
  const mlxKvBits = useChatRuntimeStore((s) => s.mlxKvBits);
  const speculativeType = useChatRuntimeStore((s) => s.speculativeType);
  const specDraftNMax = useChatRuntimeStore((s) => s.specDraftNMax);
  const nParallel = useChatRuntimeStore((s) => s.nParallel);
  const nBatch = useChatRuntimeStore((s) => s.nBatch);
  const nUbatch = useChatRuntimeStore((s) => s.nUbatch);
  const specDraftCacheDtype = useChatRuntimeStore(
    (s) => s.specDraftCacheDtype,
  );
  const loadMode = useChatRuntimeStore((s) => s.loadMode);
  const ctxCheckpoints = useChatRuntimeStore((s) => s.ctxCheckpoints);
  const cacheRam = useChatRuntimeStore((s) => s.cacheRam);
  const tensorParallel = useChatRuntimeStore((s) => s.tensorParallel);
  const disableVision = useChatRuntimeStore((s) => s.disableVision);
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

  const isGguf = isServedByLlamaCpp({
    loadedIsGguf,
    activeGgufVariant,
    activeNativePathToken,
    checkpoint,
  });
  const platform = usePlatformStore();
  const isMlx = residentIsServedByMlx(
    isGguf,
    platform.deviceType,
    platform.chatOnlyReason,
    loadedIsMlx,
  );

  // Off-backend this stays null, or the model compares unequal to its own defaults
  // over a field it cannot show.
  const effectiveMlxKvBits = isMlx ? (mlxKvBits ?? null) : null;

  const config = useMemo<PerModelConfig | null>(() => {
    if (!checkpoint || isExternalModelId(checkpoint)) {
      return null;
    }
    const base: PerModelConfig = {
      customContextLength: customContextLength ?? null,
      // A self-sizing backend carries no pin here, exactly as the GGUF path does: this
      // is the runtime's resolved length, and reading it back as the user's choice would
      // pin every reload to whatever the first load happened to get.
      maxSeqLength: isGguf || isMlx ? null : maxSeqLength,
      kvCacheDtype: kvCacheDtype ?? null,
      mlxKvBits: effectiveMlxKvBits,
      speculativeType: speculativeType ?? "auto",
      specDraftNMax: specDraftNMax ?? null,
      nParallel: nParallel ?? null,
      nBatch: nBatch ?? null,
      nUbatch: nUbatch ?? null,
      specDraftCacheDtype: specDraftCacheDtype ?? null,
      loadMode: loadMode ?? null,
      ctxCheckpoints: ctxCheckpoints ?? null,
      cacheRam: cacheRam ?? null,
      tensorParallel: tensorParallel ?? false,
      disableVision: disableVision ?? false,
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
    isMlx,
    maxSeqLength,
    customContextLength,
    kvCacheDtype,
    effectiveMlxKvBits,
    speculativeType,
    specDraftNMax,
    nParallel,
    nBatch,
    nUbatch,
    specDraftCacheDtype,
    loadMode,
    ctxCheckpoints,
    cacheRam,
    tensorParallel,
    disableVision,
    chatTemplateOverride,
    gpuMemoryMode,
    gpuLayers,
    nCpuMoe,
    selectedGpuIds,
    selectedGpuIndexKind,
  ]);

  return { checkpoint, isGguf, config };
}
