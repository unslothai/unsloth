// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Barrel import (lint rule); the model-picker cycle is fine because the call
// happens at runtime, not module eval.
import {
  loadedContextFields,
  resolveResidentInitialConfig,
  savedContextPin,
} from "@/features/model-picker";
// eslint-disable-next-line no-restricted-imports -- Avoid the hub barrel's React and download-manager exports.
import { modelDisplayName } from "@/features/hub/lib/model-identity";
import { getInferenceStatus } from "../api/chat-api";
import { isSpeechOnlyStatus } from "./speech-only-status";
import {
  mergeBackendRecommendedInference,
  replayMaxTokensCap,
} from "../presets/preset-policy";
import { clampReasoningEffortToLevels } from "../provider-capabilities";
import {
  CHAT_REASONING_ENABLED_KEY,
  type ReasoningEffort,
  type ReasoningStyle,
  loadOptionalBool,
  loadedGpuMemoryFields,
  normalizeSpeculativeType,
  noteLoadedModelReasoningMode,
  resolvePreserveThinkingOnLoad,
  resolveToolsEnabledOnLoad,
  useChatRuntimeStore,
} from "../stores/chat-runtime-store";
import {
  type InferenceStatusResponse,
  isMultimodalResponse,
} from "../types/api";
import type { ChatModelRow } from "../types/runtime";
import { resolveQwenThinkingParams } from "../utils/qwen-sampling-table";
import { sameGpuSelection } from "@/hooks/gpu-selection";
import { resolveBatchSizeSeed } from "./resolve-batch-size-seed";
import { resolveChatTemplateSeed } from "./resolve-chat-template-seed";
import { resolveCtxPinSeed } from "./resolve-ctx-pin-seed";
import { shouldSeedVisionSwitch } from "./resolve-vision-switch-seed";

type LocalReasoningEffort = Extract<ReasoningEffort, "low" | "medium" | "high">;

function sameArray<T>(a: T[] | null, b: T[] | null): boolean {
  return JSON.stringify(a) === JSON.stringify(b);
}

// Canonicalises backend / persisted speculative mode values onto the UI modes. Re-exported
// from the store, which owns the vocabulary: a second copy would drift.
export { normalizeSpeculativeType } from "../stores/chat-runtime-store";

export function clampLocalReasoningEffort(
  value: ReasoningEffort,
): LocalReasoningEffort {
  if (value === "low" || value === "medium" || value === "high") {
    return value;
  }
  return "low";
}

/** Reasoning capability fields derived from a load/status response. Centralised so every
 *  load path agrees: a hybrid `enable_thinking_effort` model keeps its high|max|Off
 *  controls instead of falling back to low|medium|high and losing Max/Off. */
export function reasoningCapsFromLoad(resp: {
  reasoning_style?: ReasoningStyle | null;
  reasoning_effort_levels?: string[] | null;
}): {
  reasoningStyle: ReasoningStyle;
  reasoningEffortLevels: readonly ReasoningEffort[];
  supportsReasoningOff: boolean;
} {
  const reasoningStyle: ReasoningStyle =
    resp.reasoning_style ?? "enable_thinking";
  const reasoningEffortLevels: readonly ReasoningEffort[] =
    resp.reasoning_effort_levels && resp.reasoning_effort_levels.length > 0
      ? (resp.reasoning_effort_levels as ReasoningEffort[])
      : (["low", "medium", "high"] as const);
  // enable_thinking and enable_thinking_effort can both be turned off; only the pure
  // gpt-oss-style reasoning_effort is always-on.
  return {
    reasoningStyle,
    reasoningEffortLevels,
    supportsReasoningOff: reasoningStyle !== "reasoning_effort",
  };
}

export function resolveInferenceCheckpointId(
  status: InferenceStatusResponse,
): string | null {
  if (!status.active_model) return null;
  return status.model_identifier ?? status.active_model;
}

function ensureActiveModelInStoreList(
  status: InferenceStatusResponse,
  checkpointId: string,
): void {
  const store = useChatRuntimeStore.getState();
  const caps = {
    // Adopting a model the backend already had mints a row with no catalog entry behind it.
    isMlx: status.is_mlx ?? false,
    isAudio: status.is_audio ?? false,
    audioType: status.audio_type ?? null,
    hasAudioInput: status.has_audio_input ?? false,
    hasVideoInput: status.has_video_input ?? false,
  };
  const existing = store.models.find((model) => model.id === checkpointId);
  if (existing) {
    // Backend capability outranks catalog metadata, and adoption has no later
    // syncModelCapabilities call. Write only on an actual change.
    if (Object.entries(caps).some(([k, v]) => existing[k as keyof typeof caps] !== v)) {
      store.setModels(
        store.models.map((m) => (m.id === checkpointId ? { ...m, ...caps } : m)),
      );
    }
    return;
  }
  const summary: ChatModelRow = {
    id: checkpointId,
    // active_model is already the clean public id; its leaf matches the catalog rows, and the
    // fallback keeps a snapshot path out of the trigger.
    name: modelDisplayName(status.active_model ?? checkpointId),
    isVision: status.is_vision ?? false,
    isLora: false,
    isGguf: status.is_gguf ?? false,
    ...caps,
  };
  store.setModels([...store.models, summary]);
}

export type ApplyInferenceStatusOptions = {
  previousCheckpoint?: string;
  /** activeGgufVariant BEFORE the caller's setCheckpoint synced it: without it a variant-only
   *  switch reads as steady state and the hydration reseed keeps the old quant's baselines. */
  previousGgufVariant?: string | null;
  /** Seed settings while the caller holds the model-loading lease. */
  seedLoadParams?: boolean;
  /** This status belongs to the model already resident when Studio started,
   * so the persisted global sampling snapshot belongs to this checkpoint. */
  adoptingExistingServerModel?: boolean;
};

/** Mirror refresh() hydration so adopted CLI models get reasoning/tools flags. */
export function applyActiveModelStatusToStore(
  status: InferenceStatusResponse,
  options: ApplyInferenceStatusOptions = {},
): void {
  const checkpointId = resolveInferenceCheckpointId(status);
  if (!checkpointId) return;

  // Only reached with a model active, so this is the one place both the status poll and the
  // readopt path can publish residency from. Without it a load looks unloaded for up to 10s.
  useChatRuntimeStore.setState({ residentCheckpoint: checkpointId });

  const store = useChatRuntimeStore.getState();
  const previousCheckpoint =
    options.previousCheckpoint ?? store.params.checkpoint;

  if (status.inference) {
    store.setParams(
      mergeBackendRecommendedInference({
        current: store.params,
        response: status,
        modelId: checkpointId,
        presetSource: store.activePresetSource,
        loadedContextLength: loadedContextFields(status).loadedContextLength,
      }),
      // The model's remembered settings outrank the recommendation, or every poll would undo them,
      // but not past the context it loaded with. context_length is reported for safetensors too,
      // so this is not narrowed to GGUF; absent, there is nothing to cap against.
      {
        fromModelDefaults: true,
        maxTokensCap: replayMaxTokensCap(status.context_length),
        migrateOwnedGlobalQwenDefaults:
          options.adoptingExistingServerModel === true,
      },
    );
  }

  const previousGgufVariant =
    options.previousGgufVariant !== undefined
      ? options.previousGgufVariant
      : store.activeGgufVariant;
  const hydratingExistingModel =
    previousCheckpoint !== checkpointId ||
    previousGgufVariant !== (status.gguf_variant ?? null);
  const supportsReasoning = status.supports_reasoning ?? false;
  const reasoningAlwaysOn = status.reasoning_always_on ?? false;
  const reasoningStyle = status.reasoning_style ?? "enable_thinking";
  // GLM-5.2-style models report their own effort levels; everything else keeps the default
  // low/medium/high.
  // They report high|max.
  const reasoningEffortLevels =
    status.reasoning_effort_levels && status.reasoning_effort_levels.length > 0
      ? (status.reasoning_effort_levels as ReasoningEffort[])
      : (["low", "medium", "high"] as const);
  const supportsPreserveThinking = status.supports_preserve_thinking ?? false;
  const preserveThinkingOnLoad = resolvePreserveThinkingOnLoad(status);
  const supportsTools = status.supports_tools ?? false;
  const storedReasoningEnabled = loadOptionalBool(CHAT_REASONING_ENABLED_KEY);
  const currentSpecType = normalizeSpeculativeType(status.speculative_type);
  const prevState = useChatRuntimeStore.getState();
  const clampedReasoningEffort =
    reasoningStyle === "enable_thinking_effort" ||
    reasoningStyle === "reasoning_effort"
      ? clampReasoningEffortToLevels(
          prevState.reasoningEffort,
          reasoningEffortLevels,
        )
      : clampLocalReasoningEffort(prevState.reasoningEffort);
  const nextDefaultChatTemplate =
    status.chat_template === undefined
      ? prevState.defaultChatTemplate
      : status.chat_template;
  // While a load is in flight, performLoad owns the load params. Seeding them
  // from a stale poll here would clobber the values the load dialog just set.
  const seedLoadParams = options.seedLoadParams ?? !prevState.modelLoading;
  // A model/variant change underneath this tab. The controls in the store belong
  // to the model that just left, so they are reseeded here the way every other
  // load param at this site already is: the echo cannot stand in, since a new
  // model can report the old count.
  const slotsModelChanged = hydratingExistingModel;
  // This model's remembered override, read only on a fresh store or a model
  // change, so a steady poll cannot re-pin a control the user just blanked. A
  // self-sizing backend has no slot fields, so an unseeded store says nothing
  // about it and it goes by the checkpoint alone.
  // Through the resident resolver, not the raw id: an API-driven load reports the
  // snapshot path a cached repo loaded from, while its settings are keyed by the
  // repo id, and the plain lookup misses that record.
  const slotsUnseeded =
    prevState.loadedNParallel === null && prevState.nParallel === null;
  // same rule for the batch-size pair
  const batchesUnseeded =
    prevState.loadedNBatch === null &&
    prevState.nBatch === null &&
    prevState.loadedNUbatch === null &&
    prevState.nUbatch === null;
  const remembered =
    (status.is_gguf
      ? slotsUnseeded || batchesUnseeded || slotsModelChanged
      : hydratingExistingModel)
      ? resolveResidentInitialConfig(checkpointId, status.gguf_variant ?? null)
      : null;
  const rememberedNParallel =
    status.is_gguf && remembered?.remembered
      ? (remembered.config.nParallel ?? null)
      : null;
  const rememberedNBatch =
    status.is_gguf && remembered?.remembered
      ? (remembered.config.nBatch ?? null)
      : null;
  const rememberedNUbatch =
    status.is_gguf && remembered?.remembered
      ? (remembered.config.nUbatch ?? null)
      : null;
  const nBatchSeed = resolveBatchSizeSeed({
    incoming: status.requested_n_batch,
    isGguf: status.is_gguf ?? true,
    previous: { value: prevState.nBatch, loaded: prevState.loadedNBatch },
    seedLoadParams,
    modelChanged: slotsModelChanged,
  });
  const nUbatchSeed = resolveBatchSizeSeed({
    incoming: status.requested_n_ubatch,
    isGguf: status.is_gguf ?? true,
    previous: { value: prevState.nUbatch, loaded: prevState.loadedNUbatch },
    seedLoadParams,
    modelChanged: slotsModelChanged,
  });
  // The llama-server tuning group follows the same rule, and resolveBatchSizeSeed is generic
  // in the value type for exactly this: each is an echo of what the load REQUESTED.
  const loadModeSeed = resolveBatchSizeSeed<string>({
    incoming: status.requested_load_mode,
    isGguf: status.is_gguf ?? true,
    previous: { value: prevState.loadMode, loaded: prevState.loadedLoadMode },
    seedLoadParams,
    modelChanged: slotsModelChanged,
  });
  const specDraftCacheSeed = resolveBatchSizeSeed<string>({
    incoming: status.requested_spec_draft_cache_type,
    isGguf: status.is_gguf ?? true,
    previous: {
      value: prevState.specDraftCacheDtype,
      loaded: prevState.loadedSpecDraftCacheDtype,
    },
    seedLoadParams,
    modelChanged: slotsModelChanged,
  });
  const ctxCheckpointsSeed = resolveBatchSizeSeed({
    incoming: status.requested_ctx_checkpoints,
    isGguf: status.is_gguf ?? true,
    previous: {
      value: prevState.ctxCheckpoints,
      loaded: prevState.loadedCtxCheckpoints,
    },
    seedLoadParams,
    modelChanged: slotsModelChanged,
  });
  const cacheRamSeed = resolveBatchSizeSeed({
    incoming: status.requested_cache_ram,
    isGguf: status.is_gguf ?? true,
    previous: { value: prevState.cacheRam, loaded: prevState.loadedCacheRam },
    seedLoadParams,
    modelChanged: slotsModelChanged,
  });
  // A load sends its context pin as max_seq_length while status exposes the resolved context
  // plus the requested n_ctx, and a positive requested n_ctx does NOT mean a human asked
  // for it. So the pin is re-seeded from what this tab or the saved config recorded, never
  // inferred from the echo alone. See resolveCtxPinSeed for the full rule.
  const ctxPinFields = resolveCtxPinSeed({
    incoming: status.requested_context_length,
    // MLX reports a requested context as well, so the rule below is about any
    // backend that sizes its own window, not llama.cpp alone.
    isGguf: (status.is_gguf ?? true) || (status.is_mlx ?? false),
    isMlx: status.is_mlx ?? false,
    seedLoadParams,
    modelChanged: slotsModelChanged,
    // Both fields: a record written before the MLX pin moved still carries it
    // in maxSeqLength.
    remembered: remembered?.remembered ? savedContextPin(remembered.config) : null,
    // Raw, not the normalised incomingGpuMode/incomingGpuLayers below: the rule
    // needs "Manual with AUTO layers", and those normalise layers to null off
    // manual, which would read as "no layers reported" rather than as Auto.
    gpuMemoryMode: status.gpu_memory_mode ?? null,
    gpuLayers: status.gpu_layers ?? null,
    loadedPin: prevState.loadedCustomContextLength ?? null,
  });
  const incomingGpuMode = status.is_gguf
    ? (status.gpu_memory_mode ?? "auto")
    : null;
  const incomingGpuLayers =
    incomingGpuMode === "manual" ? (status.gpu_layers ?? null) : null;
  const incomingNCpuMoe =
    incomingGpuMode === "manual" ? (status.n_cpu_moe ?? null) : null;
  const incomingSplit =
    incomingGpuMode === "manual" ? (status.tensor_split ?? null) : null;
  const incomingGpuFields = loadedGpuMemoryFields(status);
  const incomingGpuIds = incomingGpuFields.loadedGpuIds;
  const incomingGpuIndexKind = incomingGpuFields.loadedGpuIndexKind;
  const placementOrContextChanged =
    prevState.loadedGpuMemoryMode !== incomingGpuMode ||
    prevState.loadedGpuLayers !== incomingGpuLayers ||
    prevState.loadedNCpuMoe !== incomingNCpuMoe ||
    !sameArray(prevState.loadedSplitRatio, incomingSplit) ||
    !sameGpuSelection(
      {
        ids: prevState.loadedGpuIds,
        indexKind: prevState.loadedGpuIndexKind,
      },
      { ids: incomingGpuIds, indexKind: incomingGpuIndexKind },
    ) ||
    // Only a pin this status will actually move counts: a difference it declines to apply
    // (mid-load, or an unreadable echo) is not one.
    (ctxPinFields.loadedCustomContextLength !== undefined &&
      prevState.loadedCustomContextLength !==
        ctxPinFields.loadedCustomContextLength);
  const gpuMemoryEditsPending =
    (prevState.loadedGpuMemoryMode !== null &&
      prevState.gpuMemoryMode !== prevState.loadedGpuMemoryMode) ||
    (prevState.loadedGpuMemoryMode === "manual" &&
      (prevState.gpuLayers !== prevState.loadedGpuLayers ||
        prevState.nCpuMoe !== prevState.loadedNCpuMoe ||
        !sameArray(prevState.splitRatio, prevState.loadedSplitRatio))) ||
    prevState.customContextLength !== prevState.loadedCustomContextLength;
  const gpuIdsEditPending = !sameGpuSelection(
    {
      ids: prevState.selectedGpuIds,
      indexKind: prevState.selectedGpuIndexKind,
    },
    {
      ids: prevState.loadedGpuIds,
      indexKind: prevState.loadedGpuIndexKind,
    },
  );
  // A same-model reload from another client advances every loaded baseline.
  // Preserve each editable group only when this tab has an unapplied change.
  const preserveSameModelEdits = placementOrContextChanged && !hydratingExistingModel;
  const placementAndContextFields = {
    ...incomingGpuFields,
    ...ctxPinFields,
    ...(preserveSameModelEdits &&
      gpuMemoryEditsPending && {
        gpuMemoryMode: prevState.gpuMemoryMode,
        gpuLayers: prevState.gpuLayers,
        nCpuMoe: prevState.nCpuMoe,
        splitRatio: prevState.splitRatio,
        customContextLength: prevState.customContextLength,
      }),
    ...(preserveSameModelEdits &&
      gpuIdsEditPending && {
        selectedGpuIds: prevState.selectedGpuIds,
        selectedGpuIndexKind: prevState.selectedGpuIndexKind,
      }),
  };

  useChatRuntimeStore.setState({
    supportsReasoning,
    reasoningAlwaysOn,
    reasoningStyle,
    supportsReasoningOff: reasoningStyle !== "reasoning_effort",
    reasoningEffortLevels,
    reasoningEffort: clampedReasoningEffort,
    supportsPreserveThinking,
    ...(hydratingExistingModel && {
      preserveThinking: preserveThinkingOnLoad,
    }),
    supportsTools,
    ...resolveToolsEnabledOnLoad(supportsTools),
    reasoningEnabled: supportsReasoning
      ? reasoningStyle === "reasoning_effort"
        ? true
        : useChatRuntimeStore.getState().reasoningEnabled
      : true,
    ...loadedContextFields(status),
    ...(status.is_gguf
      ? {}
      : { activeNativePathToken: null, activeNativePathExpiresAtMs: null }),
    modelRequiresTrustRemoteCode: status.requires_trust_remote_code ?? false,
    defaultChatTemplate: nextDefaultChatTemplate,
    loadedIsMultimodal: isMultimodalResponse(status),
    loadedIsDiffusion: status.is_diffusion ?? false,
    activeModelIsLocal: status.is_local_model ?? false,
    specFallbackReason: status.spec_fallback_reason ?? null,
    mmprojFallbackReason: status.mmproj_fallback_reason ?? null,
    specDrafterKind: status.spec_drafter_kind ?? null,
    // The spec / KV seeds share the GPU-fields reseed below: a non-GGUF status leaves their
    // baselines null so the "unseeded" guard re-fires every refresh -- hold them too while a
    // staged pick is being edited. hydratingExistingModel reopens every seed, since after an
    // auto-switch the old model's baselines are stale.
    ...(seedLoadParams &&
      (prevState.loadedSpeculativeType === null || hydratingExistingModel) && {
        speculativeType: currentSpecType,
        loadedSpeculativeType: currentSpecType,
      }),
    ...(seedLoadParams &&
      status.spec_draft_n_max !== undefined &&
      (hydratingExistingModel ||
        (prevState.loadedSpecDraftNMax === null &&
          prevState.specDraftNMax === null)) && {
        specDraftNMax: status.spec_draft_n_max ?? null,
        loadedSpecDraftNMax: status.spec_draft_n_max ?? null,
      }),
    ...(seedLoadParams &&
      status.cache_type_kv !== undefined &&
      (prevState.loadedKvCacheDtype === null || hydratingExistingModel) && {
        kvCacheDtype: status.cache_type_kv,
        loadedKvCacheDtype: status.cache_type_kv,
      }),
    ...(seedLoadParams &&
      status.tensor_parallel !== undefined &&
      (prevState.loadedTensorParallel === null || hydratingExistingModel) && {
        tensorParallel: status.tensor_parallel,
        loadedTensorParallel: status.tensor_parallel,
      }),
    // A load knob like tensorParallel above. Without a reseed a tab that never performed the
    // load shows Vision ON over a projector-off server and the next Reload puts it back.
    // Seeded from disable_vision, the request the load ran with, not vision_disabled_by_user,
    // which cannot round-trip a text-only GGUF. See shouldSeedVisionSwitch.
    ...(seedLoadParams &&
      status.disable_vision !== undefined &&
      shouldSeedVisionSwitch({
        incoming: status.disable_vision,
        previous: prevState,
        hydratingExistingModel,
      }) && {
        disableVision: status.disable_vision,
      }),
    // The rollback baseline, unguarded like the mirror below rather than seeded once: it must
    // track the RUNNING server, or a switch failing after a poll restores a stale seed.
    ...(seedLoadParams &&
      status.disable_vision !== undefined && {
        loadedDisableVision: status.disable_vision,
      }),
    // Unguarded, unlike the seed above: this mirrors the live load for the composer's image
    // gate, not a user setting, so every poll must land.
    ...(seedLoadParams &&
      status.vision_disabled_by_user !== undefined && {
        loadedVisionDisabledByUser: status.vision_disabled_by_user,
      }),
    // Hydration only, so a steady poll never rewrites settings the store owns. Width, verdict
    // and request move together; a late reply can overwrite a newer one.
    ...(seedLoadParams &&
      hydratingExistingModel &&
      status.mlx_kv_bits !== undefined &&
      (status.is_mlx === true
        ? {
            mlxKvBits: status.mlx_kv_bits_requested ?? null,
            loadedMlxKvBitsRequested: status.mlx_kv_bits_requested ?? null,
            mlxKvQuantReason: status.mlx_kv_quant_reason ?? null,
            chatTemplateOverrideReason:
              status.chat_template_override_reason ?? null,
            mlxKvQuantNote: status.mlx_kv_quant_note ?? null,
          }
        : {
            // The verdict retires; the editable width is dormant, not wrong.
            loadedMlxKvBitsRequested: null,
            mlxKvQuantReason: null,
            chatTemplateOverrideReason: null,
            mlxKvQuantNote: null,
          })),
    // Recovery for a hydration this tab never saw, and only when nothing is staged: re-seeding
    // over an earlier edit would discard it.
    ...(seedLoadParams &&
      !hydratingExistingModel &&
      status.is_mlx === true &&
      status.mlx_kv_bits !== undefined &&
      prevState.mlxKvBits === null &&
      prevState.loadedMlxKvBitsRequested === null &&
      prevState.mlxKvQuantReason === null &&
      prevState.chatTemplateOverrideReason === null && {
        mlxKvBits: status.mlx_kv_bits_requested ?? null,
        loadedMlxKvBitsRequested: status.mlx_kv_bits_requested ?? null,
        mlxKvQuantReason: status.mlx_kv_quant_reason ?? null,
        chatTemplateOverrideReason: status.chat_template_override_reason ?? null,
        mlxKvQuantNote: status.mlx_kv_quant_note ?? null,
      }),
    // Baseline only, never the control: the echo is the RESOLVED count and would pin a blank
    // "server default" control. The rollback re-sends the baseline, so without this a rollback
    // after a tab reload loses the override.
    ...(seedLoadParams &&
      status.requested_parallel_slots != null &&
      (prevState.loadedNParallel === null || hydratingExistingModel) && {
        loadedNParallel: status.requested_parallel_slots,
      }),
    // A slotless model must not keep the previous GGUF's baseline, since the rollback re-sends
    // it. /status omits the echo for non-GGUF and nulls it for diffusion, so an absent field
    // on a GGUF means an older backend.
    ...(seedLoadParams &&
      (status.is_gguf === false || status.requested_parallel_slots === null) && {
        loadedNParallel: null,
      }),
    // Per-model: a change underneath this tab blanks the control like performLoad's cross-model
    // reset, or the old count follows onto the new model. The baseline still has the rollback.
    ...(seedLoadParams && slotsModelChanged && { nParallel: null }),
    // AFTER that clear, which both a first hydration and a model change trip: either would leave
    // the control blank while the model runs on a remembered override, so the next Apply would
    // save the blank over it. Adopted only when the running count matches.
    ...(seedLoadParams &&
      (slotsUnseeded || slotsModelChanged) &&
      rememberedNParallel != null &&
      rememberedNParallel === status.requested_parallel_slots && {
        nParallel: rememberedNParallel,
      }),
    // What the running server was actually invoked with. Without this a tab opened onto an
    // already-loaded model knows nothing about its pass-through arguments, and a rollback
    // after a failed switch restores that model without them. Adopted from every settled
    // status, not just the first: another client can reload the SAME model with different
    // arguments, and a pinned baseline would resurrect arguments that are not running.
    // seedLoadParams still guards it, so a mid-switch poll cannot overwrite performLoad.
    ...(status.requested_llama_extra_args !== undefined &&
      (status.is_gguf ?? true) &&
      seedLoadParams && {
        loadedLlamaExtraArgs: status.requested_llama_extra_args ?? null,
      }),
    // one rule per batch pair, see resolveBatchSizeSeed
    ...("loaded" in nBatchSeed && { loadedNBatch: nBatchSeed.loaded ?? null }),
    ...("value" in nBatchSeed && { nBatch: nBatchSeed.value ?? null }),
    ...("loaded" in nUbatchSeed && {
      loadedNUbatch: nUbatchSeed.loaded ?? null,
    }),
    ...("value" in nUbatchSeed && { nUbatch: nUbatchSeed.value ?? null }),
    ...("loaded" in loadModeSeed && {
      loadedLoadMode: loadModeSeed.loaded ?? null,
    }),
    ...("value" in loadModeSeed && { loadMode: loadModeSeed.value ?? null }),
    ...("loaded" in specDraftCacheSeed && {
      loadedSpecDraftCacheDtype: specDraftCacheSeed.loaded ?? null,
    }),
    ...("value" in specDraftCacheSeed && {
      specDraftCacheDtype: specDraftCacheSeed.value ?? null,
    }),
    ...("loaded" in ctxCheckpointsSeed && {
      loadedCtxCheckpoints: ctxCheckpointsSeed.loaded ?? null,
    }),
    ...("value" in ctxCheckpointsSeed && {
      ctxCheckpoints: ctxCheckpointsSeed.value ?? null,
    }),
    ...("loaded" in cacheRamSeed && {
      loadedCacheRam: cacheRamSeed.loaded ?? null,
    }),
    ...("value" in cacheRamSeed && { cacheRam: cacheRamSeed.value ?? null }),
    // A swap under this tab resets the controls too, but that clear belongs INSIDE
    // resolveBatchSizeSeed (modelChanged), not after it: the batch echo is the REQUESTED size,
    // so a blanket null here would discard what the seed just adopted from the new model,
    // leaving "default" over an explicit -b / -ub that the next Apply would revert.
    ...(seedLoadParams &&
      (batchesUnseeded || slotsModelChanged) &&
      rememberedNBatch != null &&
      rememberedNBatch === status.requested_n_batch && {
        nBatch: rememberedNBatch,
      }),
    ...(seedLoadParams &&
      (batchesUnseeded || slotsModelChanged) &&
      rememberedNUbatch != null &&
      rememberedNUbatch === status.requested_n_ubatch && {
        nUbatch: rememberedNUbatch,
      }),
    // Re-seed on first hydration, model/variant changes, or a same-model backend
    // placement change. placementAndContextFields preserves dirty local edits in the last
    // case while advancing their loaded baselines.
    ...(seedLoadParams &&
      (prevState.loadedGpuMemoryMode === null ||
        hydratingExistingModel ||
        placementOrContextChanged) &&
      placementAndContextFields),
    // The one load param that only ever seeded from null, so a switch left the previous model's
    // template in the store and Apply saved A's template under B. A same-model reload moves it
    // too, so the seed follows a changed status like the GPU group: baseline always, control
    // only while it still sits on that baseline. See resolveChatTemplateSeed.
    ...resolveChatTemplateSeed({
      incoming: status.chat_template_override,
      previous: {
        chatTemplateOverride: prevState.chatTemplateOverride,
        loadedChatTemplateOverride: prevState.loadedChatTemplateOverride,
      },
      hydratingExistingModel,
      seedLoadParams,
    }),
  });

  ensureActiveModelInStoreList(status, checkpointId);

  if (
    supportsReasoning &&
    hydratingExistingModel &&
    storedReasoningEnabled === null
  ) {
    // Anchored regex: first "Xb"/"X.Xb" after start or [-_/.] so the version literal in
    // "qwen3.5" does not match first, and "Qwen3.5-35B-A3B" yields 35 (total), not 3 (active).
    // Mirrors use-chat-model-runtime.ts and the inline regex in llama_cpp.py.
    let reasoningDefault = true;
    const mid = checkpointId.toLowerCase();
    if (mid.includes("qwen3.5") || mid.includes("qwen3.6")) {
      // Scan path segments right to left so the size nearest the leaf wins over a size-like parent
      // dir; the trailing boundary prevents matching "8bit".
      const sizeRe = /(?:^|[-_.])(\d+\.?\d*)\s*([bm])(?:$|[-_.])/;
      const sizeMatch = mid
        .replace(/\\/g, "/")
        .split("/")
        .reduceRight<RegExpMatchArray | null>(
          (found, seg) => found ?? seg.match(sizeRe),
          null,
        );
      if (sizeMatch) {
        const size = Number.parseFloat(sizeMatch[1]);
        const sizeB = sizeMatch[2] === "m" ? size / 1000 : size;
        if (sizeB <= 9) reasoningDefault = false;
      }
    }
    useChatRuntimeStore.setState({ reasoningEnabled: reasoningDefault });
  }

  noteLoadedModelReasoningMode(
    checkpointId,
    reasoningAlwaysOn || useChatRuntimeStore.getState().reasoningEnabled,
  );

  // Every status merge carries the base family recommendation, including the
  // refresh immediately after performLoad. Layer the active Qwen mode over it
  // so that refresh cannot undo performLoad's thinking table. This also covers
  // startup/CLI/external adoption, while model memory still wins because this
  // remains a defaults update.
  if (status.inference && supportsReasoning) {
    const current = useChatRuntimeStore.getState();
    const qwenParams = resolveQwenThinkingParams(
      checkpointId,
      reasoningAlwaysOn || current.reasoningEnabled,
    );
    if (qwenParams !== null && current.activePresetSource === "builtin-default") {
      current.setParams(
        { ...current.params, ...qwenParams },
        {
          fromModelDefaults: true,
          maxTokensCap: replayMaxTokensCap(status.context_length),
          migrateOwnedGlobalQwenDefaults:
            options.adoptingExistingServerModel === true,
        },
      );
    }
  }
}

/** Adopt a server-loaded model without issuing another inference load. */
export async function tryAdoptServerActiveModel(options?: {
  /** Ignore modelLoading because the caller owns that lease. */
  allowWhileModelLoading?: boolean;
  /** A status the caller already read, so the send path does not fetch twice. */
  status?: InferenceStatusResponse;
}): Promise<boolean> {
  const store = useChatRuntimeStore.getState();
  if (store.params.checkpoint) {
    return true;
  }
  if (store.modelLoading && !options?.allowWhileModelLoading) {
    return false;
  }
  let status: InferenceStatusResponse;
  if (options?.status) {
    status = options.status;
  } else {
    try {
      status = await getInferenceStatus();
    } catch {
      // Status endpoint unavailable: fall back to the normal auto-load path.
      return false;
    }
  }
  // Not something chat can adopt; the sweep below picks a real chat model, which evicts
  // it exactly as an image load would.
  if (
    !status.active_model ||
    (status.loading?.length ?? 0) > 0 ||
    isSpeechOnlyStatus(status)
  ) {
    return false;
  }

  const checkpointId = resolveInferenceCheckpointId(status);
  if (!checkpointId) {
    return false;
  }

  // Preserve concurrent changes unless this caller owns the load lease.
  const latest = useChatRuntimeStore.getState();
  const previousCheckpoint = latest.params.checkpoint;
  if (
    previousCheckpoint ||
    (latest.modelLoading && !options?.allowWhileModelLoading)
  ) {
    return !!previousCheckpoint;
  }
  const previousGgufVariant = latest.activeGgufVariant;
  store.setCheckpoint(checkpointId, status.gguf_variant);
  applyActiveModelStatusToStore(status, {
    previousCheckpoint,
    previousGgufVariant,
    seedLoadParams: options?.allowWhileModelLoading,
    adoptingExistingServerModel: true,
  });
  return true;
}
