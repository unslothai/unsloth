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
import { mergeBackendRecommendedInference } from "../presets/preset-policy";
import { clampReasoningEffortToLevels } from "../provider-capabilities";
import {
  CHAT_REASONING_ENABLED_KEY,
  type ReasoningEffort,
  type ReasoningStyle,
  loadOptionalBool,
  loadedGpuMemoryFields,
  normalizeSpeculativeType,
  resolvePreserveThinkingOnLoad,
  resolveToolsEnabledOnLoad,
  useChatRuntimeStore,
} from "../stores/chat-runtime-store";
import {
  type InferenceStatusResponse,
  isMultimodalResponse,
} from "../types/api";
import type { ChatModelRow } from "../types/runtime";
import { resolveQwenThinkingParams } from "../utils/qwen-params";
import { sameGpuSelection } from "@/hooks/gpu-selection";
import { resolveBatchSizeSeed } from "./resolve-batch-size-seed";
import { resolveChatTemplateSeed } from "./resolve-chat-template-seed";
import { resolveCtxPinSeed } from "./resolve-ctx-pin-seed";
import { shouldSeedVisionSwitch } from "./resolve-vision-switch-seed";

type LocalReasoningEffort = Extract<ReasoningEffort, "low" | "medium" | "high">;

function sameArray<T>(a: T[] | null, b: T[] | null): boolean {
  return JSON.stringify(a) === JSON.stringify(b);
}

// Canonicalises backend / persisted speculative mode values onto the UI
// modes. Re-exported from the store, which owns the vocabulary: a second
// copy meant every new mode had to be added twice or the two would disagree.
export { normalizeSpeculativeType } from "../stores/chat-runtime-store";

export function clampLocalReasoningEffort(
  value: ReasoningEffort,
): LocalReasoningEffort {
  if (value === "low" || value === "medium" || value === "high") {
    return value;
  }
  return "low";
}

/**
 * Reasoning capability fields derived from a model load/status response.
 *
 * Centralises the effort-levels + can-disable derivation so every load path
 * (main load, status sync, shared/Compare composer, first-chat auto-load) agrees:
 * a hybrid GLM-style `enable_thinking_effort` model keeps its high|max|Off
 * controls no matter which path loaded it, instead of falling back to the
 * default low|medium|high and losing Max/Off.
 */
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
  // enable_thinking and enable_thinking_effort can both be turned off; only the
  // pure gpt-oss-style reasoning_effort is always-on.
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
    // active_model is already the clean public id; its leaf matches the catalog rows,
    // and the fallback keeps a snapshot path out of the trigger.
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
  /** activeGgufVariant BEFORE the caller's setCheckpoint synced it to the
   * status -- without it a variant-only switch underneath the tab reads as
   * steady state and the hydration reseed keeps the old quant's baselines. */
  previousGgufVariant?: string | null;
};

/** Mirror refresh() hydration so adopted CLI models get reasoning/tools flags. */
export function applyActiveModelStatusToStore(
  status: InferenceStatusResponse,
  options: ApplyInferenceStatusOptions = {},
): void {
  const checkpointId = resolveInferenceCheckpointId(status);
  if (!checkpointId) return;

  // Only reached with a model actually active, so this is the one place both
  // the status poll and the readopt path can publish residency from. Without
  // it a load would look unloaded until the next poll, up to 10s later.
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
      // The model's own remembered settings outrank the recommendation, or
      // every poll would undo them, but not past the context it loaded with.
      // context_length is reported for a safetensors load too, so this is not
      // narrowed to GGUF; absent, there is nothing to cap against.
      {
        fromModelDefaults: true,
        maxTokensCap: status.context_length ?? undefined,
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
  // GLM-5.2-style models report their own effort levels (e.g. high|max);
  // everything else keeps the default low/medium/high.
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
  const seedLoadParams = !prevState.modelLoading;
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
  // The llama-server tuning group follows the same rule, and resolveBatchSizeSeed
  // is generic in the value type for exactly this: each one is an echo of what the
  // load REQUESTED, which is what makes the steady-poll and dirty-control cases
  // identical to the batch pair's.
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
  // A load sends its context pin as max_seq_length and status only exposes the
  // resolved context plus the requested n_ctx, and a positive requested n_ctx
  // does NOT mean a human asked for it: an Auto same-model reload under a custom
  // preset reports one too. So the pin is re-seeded here from what this tab (or
  // the model's saved config) actually recorded, never inferred from the echo
  // alone, and the echo is trusted only where it is unambiguous. See
  // resolveCtxPinSeed for the full rule, including the mid-load window where
  // status still answers for the OUTGOING model.
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
    // Only a pin this status will actually move counts: a difference it declines
    // to apply (mid-load, or an echo it cannot read intent out of) is not one.
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
    // The spec / KV seeds share the GPU-fields reseed mechanism below: a
    // non-GGUF status leaves their loaded baselines null, so the "unseeded"
    // guard re-fires every refresh -- hold them too while a staged pick's
    // settings are being edited, or the refresh resets the staged edit.
    // hydratingExistingModel reopens every load-param seed: when the active
    // model changed underneath this tab (auto-switch, another client), the
    // old model's baselines are stale and must adopt the new status.
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
    // A load knob like tensorParallel above. Without a reseed a tab that never
    // performed the load shows Vision ON over a model running with its projector
    // off, and the next Reload silently puts the projector back. Seeded from
    // disable_vision -- the request the load ran with -- not
    // vision_disabled_by_user, which is also gated on the model HAVING a projector
    // and so cannot round-trip a text-only GGUF.
    //
    // Unlike tensorParallel the guard is not just "unseeded": the baseline below is
    // unguarded, so an external reload of the same model would move it and the image
    // gate while leaving this control behind. See shouldSeedVisionSwitch.
    ...(seedLoadParams &&
      status.disable_vision !== undefined &&
      shouldSeedVisionSwitch({
        incoming: status.disable_vision,
        previous: prevState,
        hydratingExistingModel,
      }) && {
        disableVision: status.disable_vision,
      }),
    // The rollback baseline, and unguarded like the mirror below rather than
    // seeded once: it has to track the RUNNING server, or a switch that fails
    // after a poll restores whatever the last seed happened to see.
    ...(seedLoadParams &&
      status.disable_vision !== undefined && {
        loadedDisableVision: status.disable_vision,
      }),
    // Unguarded, unlike the seed above: this mirrors the live load for the
    // composer's image gate, not a user setting, so every poll must land.
    ...(seedLoadParams &&
      status.vision_disabled_by_user !== undefined && {
        loadedVisionDisabledByUser: status.vision_disabled_by_user,
      }),
    // Hydration only, so a steady poll never rewrites settings the store owns.
    // Width, verdict and request move together; a late reply can overwrite a newer one.
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
    // Recovery for a hydration this tab never saw, and only when nothing is
    // staged: re-seeding over an earlier edit would discard it.
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
    // Baseline only, never the control: the echo is the RESOLVED count and would
    // pin a blank "server default" control. The rollback re-sends the baseline,
    // so without this a rollback after a tab reload loses the override.
    ...(seedLoadParams &&
      status.requested_parallel_slots != null &&
      (prevState.loadedNParallel === null || hydratingExistingModel) && {
        loadedNParallel: status.requested_parallel_slots,
      }),
    // A slotless model must not keep the previous GGUF's baseline: the rollback
    // re-sends it. /status omits the echo for non-GGUF and sends an explicit
    // null for diffusion, so an absent field on a GGUF is an older backend.
    ...(seedLoadParams &&
      (status.is_gguf === false || status.requested_parallel_slots === null) && {
        loadedNParallel: null,
      }),
    // Per-model: a change underneath this tab blanks the control like
    // performLoad's cross-model reset, or the old count follows onto the new
    // model. The baseline above still carries the rollback.
    ...(seedLoadParams && slotsModelChanged && { nParallel: null }),
    // AFTER that clear, which both a first hydration and a model change trip:
    // either would leave the control blank while the model runs on a remembered
    // override, so the next Apply would save the blank over it. Adopted only
    // when the running count matches, proving it is this model's own.
    ...(seedLoadParams &&
      (slotsUnseeded || slotsModelChanged) &&
      rememberedNParallel != null &&
      rememberedNParallel === status.requested_parallel_slots && {
        nParallel: rememberedNParallel,
      }),
    // What the running server was actually invoked with. Without this a tab opened
    // while a model was already loaded knows nothing about its pass-through
    // arguments (the switch path is where they were recorded), and a rollback after
    // a failed switch restores that model without them: the failed target is left
    // resident, so an omitted field cannot inherit across models either.
    //
    // Adopted from every settled status, not only a first read or a model change:
    // another tab or an API client can reload the SAME model and variant with
    // different arguments, or with none, and a baseline pinned at the first read
    // would resend the old list from the rollback path and resurrect arguments that
    // are not running. seedLoadParams is the guard that matters here, and it is the
    // same one the rest of this block uses: while a load is in flight performLoad
    // owns these values, so a poll that answers mid-switch cannot overwrite them.
    // A backend that does not publish the field changes nothing.
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
    // resolveBatchSizeSeed (modelChanged), not after it: unlike the slot count above,
    // the batch echo is the REQUESTED size, so a blanket null here would also discard
    // the value the seed just adopted from the new model's own echo. The control would
    // then read "default" while the server runs an explicit -b / -ub, and the next
    // Reload or Apply, which omits a blank field, would silently revert it.
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
    // template in the store, which the Hub settings page reads as the new model's loaded config:
    // Apply then saves A's template under B. A same-model reload from another client moves it
    // too, so the seed also follows a changed status the way the GPU group above does: baseline
    // always, control only while it still sits on that baseline. See resolveChatTemplateSeed.
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
    // Anchored regex: first "Xb" / "X.Xb" after start-of-string or
    // [-_/.] so the version literal in "qwen3.5" / "qwen3.6" doesn't match
    // first, and for "Qwen3.5-35B-A3B" the result is 35 (total params),
    // not 3 (MoE active params). Mirrors the regex in
    // use-chat-model-runtime.ts and the inline one in llama_cpp.py.
    let reasoningDefault = true;
    const mid = checkpointId.toLowerCase();
    if (mid.includes("qwen3.5") || mid.includes("qwen3.6")) {
      // Scan path segments right to left so the size nearest the leaf wins over
      // a size-like parent dir; trailing boundary prevents matching "8bit".
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
          maxTokensCap: status.context_length ?? undefined,
        },
      );
    }
  }
}

/**
 * Adopt the model already loaded on the inference server (e.g. via
 * ``unsloth studio run -m``) into the chat UI checkpoint without
 * triggering a new /api/inference/load.
 */
export async function tryAdoptServerActiveModel(): Promise<boolean> {
  const store = useChatRuntimeStore.getState();
  if (store.params.checkpoint) {
    return true;
  }

  let status: InferenceStatusResponse;
  try {
    status = await getInferenceStatus();
  } catch {
    // Status endpoint unavailable: fall back to the normal auto-load path.
    return false;
  }
  // Not something chat can adopt; the sweep below picks a real chat model, which evicts
  // it exactly as an image load would.
  if (!status.active_model || isSpeechOnlyStatus(status)) {
    return false;
  }

  const checkpointId = resolveInferenceCheckpointId(status);
  if (!checkpointId) {
    return false;
  }

  // Re-check after the await: keep a checkpoint the user picked meanwhile.
  const previousCheckpoint = useChatRuntimeStore.getState().params.checkpoint;
  if (previousCheckpoint) {
    return true;
  }
  const previousGgufVariant = useChatRuntimeStore.getState().activeGgufVariant;
  store.setCheckpoint(checkpointId, status.gguf_variant);
  applyActiveModelStatusToStore(status, {
    previousCheckpoint,
    previousGgufVariant,
  });
  return true;
}
