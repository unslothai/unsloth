// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import {
  ggufVariantsMatch,
  modelIdsMatch,
} from "../../../lib/model-identity.ts";
import {
  DEFAULT_PER_MODEL_CONFIG,
  type PerModelConfig,
} from "../model-config/per-model-config.ts";
import type { useChatRuntimeStore } from "../stores/chat-runtime-store";
import { normalizeSpeculativeType } from "./speculative-type.ts";

export type LoadedRuntimeConfigState = Pick<
  ReturnType<typeof useChatRuntimeStore.getState>,
  | "params"
  | "activeGgufVariant"
  | "loadedKvCacheDtype"
  | "loadedSpeculativeType"
  | "loadedSpecDraftNMax"
  | "loadedCustomContextLength"
  | "loadedChatTemplateOverride"
>;

export type RuntimePerModelLoadConfig = {
  kvCacheDtype: string | null;
  speculativeType: string | null;
  specDraftNMax: number | null;
  customContextLength: number | null;
  chatTemplateOverride: string | null;
  trustRemoteCode: boolean;
};

function cleanTemplate(value: string | null | undefined): string | null {
  return value?.trim() ? value : null;
}

function normalizeSpecForComparison(value: string | null | undefined): string {
  return normalizeSpeculativeType(value) ?? "auto";
}

function specDraftMatches({
  speculativeType,
  loaded,
  requested,
}: {
  speculativeType: string;
  loaded: number | null | undefined;
  requested: number | null | undefined;
}): boolean {
  if (speculativeType !== "mtp" && speculativeType !== "mtp+ngram") {
    return true;
  }
  return requested == null || (loaded ?? null) === requested;
}

export function normalizeRuntimePerModelLoadConfig(
  config: PerModelConfig = DEFAULT_PER_MODEL_CONFIG,
): RuntimePerModelLoadConfig {
  return {
    kvCacheDtype: config.kvCacheDtype ?? null,
    speculativeType: normalizeSpeculativeType(config.speculativeType),
    specDraftNMax: config.specDraftNMax ?? null,
    customContextLength: config.customContextLength ?? null,
    chatTemplateOverride: cleanTemplate(config.chatTemplateOverride),
    trustRemoteCode: config.trustRemoteCode ?? false,
  };
}

export function loadedRuntimeConfigMatches({
  state,
  modelId,
  ggufVariant,
  config,
}: {
  state: LoadedRuntimeConfigState;
  modelId: string;
  ggufVariant?: string | null;
  config?: PerModelConfig;
}): boolean {
  const normalized = normalizeRuntimePerModelLoadConfig(config);
  const loadedTemplate = cleanTemplate(state.loadedChatTemplateOverride);
  const loadedSpeculativeType = normalizeSpecForComparison(
    state.loadedSpeculativeType,
  );
  const requestedSpeculativeType = normalizeSpecForComparison(
    normalized.speculativeType,
  );
  return (
    modelIdsMatch(state.params.checkpoint, modelId) &&
    ggufVariantsMatch(state.activeGgufVariant, ggufVariant) &&
    (state.params.trustRemoteCode ?? false) === normalized.trustRemoteCode &&
    (state.loadedKvCacheDtype ?? null) === normalized.kvCacheDtype &&
    loadedSpeculativeType === requestedSpeculativeType &&
    specDraftMatches({
      speculativeType: requestedSpeculativeType,
      loaded: state.loadedSpecDraftNMax,
      requested: normalized.specDraftNMax,
    }) &&
    (state.loadedCustomContextLength ?? null) ===
      normalized.customContextLength &&
    loadedTemplate === normalized.chatTemplateOverride
  );
}
