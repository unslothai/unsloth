


// Barrel import (lint rule); the model-picker cycle is fine because the call
// happens at runtime, not module eval.
import { resolveResidentInitialConfig } from "@/features/model-picker";
// eslint-disable-next-line no-restricted-imports -- Avoid the hub barrel's React and download-manager exports.
import { modelDisplayName } from "@/features/hub/lib/model-identity";
import { getInferenceStatus } from "../api/chat-api";
import {
  mergeBackendRecommendedInference,
  resolveManualAutoCtxPin,
} from "../presets/preset-policy";
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
import type { ChatModelSummary } from "../types/runtime";
import { sameGpuSelection } from "@/hooks/gpu-selection";
import { resolveBatchSizeSeed } from "./resolve-batch-size-seed";
import { resolveChatTemplateSeed } from "./resolve-chat-template-seed";

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
  const summary: ChatModelSummary = {
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
  const currentGgufContextLength = status.is_gguf
    ? (status.context_length ?? null)
    : null;
  const ggufMaxContextLength = status.is_gguf
    ? (status.max_context_length ?? null)
    : null;
  const ggufNativeContextLength = status.is_gguf
    ? (status.native_context_length ?? null)
    : null;
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
  // change, so a steady poll cannot re-pin a control the user just blanked.
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
    status.is_gguf && (slotsUnseeded || batchesUnseeded || slotsModelChanged)
      ? resolveResidentInitialConfig(checkpointId, status.gguf_variant ?? null)
      : null;
  const rememberedNParallel = remembered?.remembered
    ? (remembered.config.nParallel ?? null)
    : null;
  const rememberedNBatch = remembered?.remembered
    ? (remembered.config.nBatch ?? null)
    : null;
  const rememberedNUbatch = remembered?.remembered
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
  // A Manual + Auto-layers load sent its positive context pin as max_seq_length,
  // and status only exposes the RESOLVED context; re-seed the pin from the
  // requested value (parity with the load paths' keepCustomCtx). Baselines
  // unconditionally: anything but an applicable pin is null, so a previous
  // model's pin can't survive a model change underneath and reload at the old length.
  const gpuPin = status.is_gguf
    ? resolveManualAutoCtxPin(
        status.gpu_memory_mode ?? "auto",
        status.gpu_layers ?? -1,
        status.requested_context_length ?? null,
      )
    : null;
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
  const gpuStatusChanged =
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
    prevState.loadedCustomContextLength !== gpuPin;
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
  const preserveSameModelEdits = gpuStatusChanged && !hydratingExistingModel;
  const gpuStatusFields = {
    ...incomingGpuFields,
    customContextLength: gpuPin,
    loadedCustomContextLength: gpuPin,
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
    ggufContextLength: currentGgufContextLength,
    ggufMaxContextLength,
    ggufNativeContextLength,
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
    // placement change. gpuStatusFields preserves dirty local edits in the last
    // case while advancing their loaded baselines.
    ...(seedLoadParams &&
      (prevState.loadedGpuMemoryMode === null ||
        hydratingExistingModel ||
        gpuStatusChanged) &&
      gpuStatusFields),
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
  if (!status.active_model) {
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
