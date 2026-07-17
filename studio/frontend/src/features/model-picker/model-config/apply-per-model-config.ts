// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import {
  defaultInferenceParams,
  normalizeSpeculativeType,
  readPersistedSpeculativeType,
  useChatRuntimeStore,
} from "@/features/chat";
import {
  DEFAULT_PER_MODEL_CONFIG,
  type PerModelConfig,
  normalizeMaxSeqLength,
} from "./per-model-config";

function cleanTemplate(value: string | null | undefined): string | null {
  return value?.trim() ? value : null;
}

export function applyPerModelConfigToRuntime(config: PerModelConfig): void {
  // Fall back to the standing default when the model has no saved
  // maxSeqLength. maxSeqLength is the only per-model field carried on
  // params (the rest are reset below), so without this a model with no
  // remembered config would inherit the previously loaded model's value.
  const maxSeqLength =
    normalizeMaxSeqLength(config.maxSeqLength) ??
    defaultInferenceParams.maxSeqLength;
  const store = useChatRuntimeStore.getState();
  if (maxSeqLength !== store.params.maxSeqLength) {
    store.setParams({ ...store.params, maxSeqLength });
  }
  useChatRuntimeStore.setState({
    customContextLength: config.customContextLength ?? null,
    kvCacheDtype: config.kvCacheDtype ?? null,
    speculativeType:
      normalizeSpeculativeType(config.speculativeType) ??
      readPersistedSpeculativeType(),
    specDraftNMax: config.specDraftNMax ?? null,
    tensorParallel: config.tensorParallel ?? false,
    chatTemplateOverride: cleanTemplate(config.chatTemplateOverride),
  });
}

export function applyModelLoadConfigToRuntime(
  config: PerModelConfig | null | undefined,
): boolean {
  const hasConfig = config != null;
  applyPerModelConfigToRuntime(config ?? DEFAULT_PER_MODEL_CONFIG);
  return hasConfig;
}

export function currentRuntimePerModelConfig(
  options: { includeMaxSeqLength?: boolean } = {},
): PerModelConfig {
  const s = useChatRuntimeStore.getState();
  return {
    customContextLength: s.customContextLength ?? null,
    maxSeqLength: options.includeMaxSeqLength
      ? normalizeMaxSeqLength(s.params.maxSeqLength)
      : null,
    kvCacheDtype: s.kvCacheDtype ?? null,
    speculativeType: normalizeSpeculativeType(s.speculativeType),
    specDraftNMax: s.specDraftNMax ?? null,
    tensorParallel: s.tensorParallel ?? false,
    chatTemplateOverride: cleanTemplate(s.chatTemplateOverride),
  };
}

export function perModelConfigsEqual(
  a: PerModelConfig,
  b: PerModelConfig,
): boolean {
  return (
    (a.customContextLength ?? null) === (b.customContextLength ?? null) &&
    normalizeMaxSeqLength(a.maxSeqLength) ===
      normalizeMaxSeqLength(b.maxSeqLength) &&
    (a.kvCacheDtype ?? null) === (b.kvCacheDtype ?? null) &&
    normalizeSpeculativeType(a.speculativeType) ===
      normalizeSpeculativeType(b.speculativeType) &&
    (a.specDraftNMax ?? null) === (b.specDraftNMax ?? null) &&
    Boolean(a.tensorParallel) === Boolean(b.tensorParallel) &&
    cleanTemplate(a.chatTemplateOverride) ===
      cleanTemplate(b.chatTemplateOverride)
  );
}
