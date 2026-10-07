// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { isExternalModelId, useChatRuntimeStore } from "@/features/chat";
import { usePlatformStore } from "@/config/env";
import { isNpuModelId } from "@/features/npu";
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
  const engine = useChatRuntimeStore((s) => s.params.engine ?? "auto");
  const engineParallelism = useChatRuntimeStore((s) => s.params.engineParallelism ?? "tensor");
  const enginePrecision = useChatRuntimeStore((s) => s.params.enginePrecision ?? "auto");
  const activeGgufVariant = useChatRuntimeStore((s) => s.activeGgufVariant);
  const loadedIsGguf = useChatRuntimeStore((s) => s.loadedIsGguf);
  const loadedIsMlx = useChatRuntimeStore((s) => s.loadedIsMlx);
  const activeNativePathToken = useChatRuntimeStore(
    (s) => s.activeNativePathToken,
  );
  const customContextLength = useChatRuntimeStore((s) => s.customContextLength);
  const kvCacheDtype = useChatRuntimeStore((s) => s.kvCacheDtype);
  const mlxKvQuant = useChatRuntimeStore((s) => s.mlxKvQuant);
  const mlxInt8Prefill = useChatRuntimeStore((s) => s.mlxInt8Prefill);
  const speculativeType = useChatRuntimeStore((s) => s.speculativeType);
  const specDraftNMax = useChatRuntimeStore((s) => s.specDraftNMax);
  const nParallel = useChatRuntimeStore((s) => s.nParallel);
  // preserve inherited launch intent until the corresponding control changes.
  const reasoningBudget = useChatRuntimeStore((s) =>
    s.reasoningBudget === s.loadedReasoningBudget
      ? (s.loadedReasoningBudgetRequested ?? s.reasoningBudget)
      : s.reasoningBudget,
  );
  const reasoningBudgetMessage = useChatRuntimeStore(
    (s) =>
      s.reasoningBudgetMessage === s.loadedReasoningBudgetMessage
        ? (s.loadedReasoningBudgetMessageRequested ?? s.reasoningBudgetMessage)
        : s.reasoningBudgetMessage,
  );
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
  const loadedLlamaExtraArgs = useChatRuntimeStore(
    (s) => s.loadedLlamaExtraArgs,
  );
  const gpuMemoryMode = useChatRuntimeStore((s) => s.gpuMemoryMode);
  const gpuLayers = useChatRuntimeStore((s) => s.gpuLayers);
  const nCpuMoe = useChatRuntimeStore((s) => s.nCpuMoe);
  const selectedGpuIds = useChatRuntimeStore((s) => s.selectedGpuIds);
  const selectedGpuIndexKind = useChatRuntimeStore(
    (s) => s.selectedGpuIndexKind,
  );
  const splitRatio = useChatRuntimeStore((s) => s.splitRatio);

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

  // Null off MLX, or the model compares unequal to its own defaults.
  const effectiveMlxKvQuant = isMlx ? (mlxKvQuant ?? null) : null;
  const effectiveMlxInt8Prefill = isMlx && mlxInt8Prefill;

  const config = useMemo<PerModelConfig | null>(() => {
    if (!checkpoint || isExternalModelId(checkpoint)) {
      return null;
    }
    const base: PerModelConfig = {
      engine,
      enginePrecision,
      engineParallelism,
      customContextLength: customContextLength ?? null,
      // Self-sizing backends carry no pin: this is the resolved length, not the user's choice.
      maxSeqLength: isGguf || isMlx || isNpuModelId(checkpoint) ? null : maxSeqLength,
      kvCacheDtype: kvCacheDtype ?? null,
      mlxKvQuant: effectiveMlxKvQuant,
      mlxInt8Prefill: effectiveMlxInt8Prefill,
      speculativeType: speculativeType ?? "auto",
      specDraftNMax: specDraftNMax ?? null,
      nParallel: nParallel ?? null,
      reasoningBudget: isGguf ? reasoningBudget : -1,
      reasoningBudgetMessage: isGguf ? reasoningBudgetMessage : "",
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
      return engine === "vllm" || engine === "sglang"
        ? { ...base, selectedGpuIds, selectedGpuIndexKind }
        : base;
    }
    return {
      ...base,
      gpuMemoryMode,
      gpuLayers,
      nCpuMoe,
      selectedGpuIds,
      selectedGpuIndexKind,
      tensorSplit: splitRatio,
      // Undefined until the runtime reports a list, so the editor can hydrate the stored row.
      ...(loadedLlamaExtraArgs != null
        ? { llamaExtraArgs: [...loadedLlamaExtraArgs] }
        : {}),
    };
  }, [
    checkpoint,
    isGguf,
    isMlx,
    maxSeqLength,
    engine,
    enginePrecision,
    engineParallelism,
    customContextLength,
    kvCacheDtype,
    effectiveMlxKvQuant,
    effectiveMlxInt8Prefill,
    speculativeType,
    specDraftNMax,
    nParallel,
    reasoningBudget,
    reasoningBudgetMessage,
    nBatch,
    nUbatch,
    specDraftCacheDtype,
    loadMode,
    ctxCheckpoints,
    cacheRam,
    tensorParallel,
    disableVision,
    chatTemplateOverride,
    loadedLlamaExtraArgs,
    gpuMemoryMode,
    gpuLayers,
    nCpuMoe,
    selectedGpuIds,
    selectedGpuIndexKind,
    splitRatio,
  ]);

  return { checkpoint, isGguf, config };
}
